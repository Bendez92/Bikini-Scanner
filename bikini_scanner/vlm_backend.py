from __future__ import annotations

import base64
import json
import logging
import re
import urllib.request
from collections.abc import Callable, Mapping
from concurrent.futures import ThreadPoolExecutor, as_completed
from io import BytesIO
from threading import Event

import numpy as np
from PIL import Image

LOGGER = logging.getLogger(__name__)
VLM_AXES = (
    "person",
    "female",
    "child",
    "adult",
    "bikini",
    "bikini_top",
    "bikini_bottom",
    "midriff",
    "cleavage",
    "nsfw",
)
VLM_PROMPT_VERSION = "vlm-json-v1"


class VLMCancelled(Exception):
    """Raised when a scan cancellation interrupts VLM adjudication."""


def parse_axis_json(text: str, axes: tuple[str, ...] = VLM_AXES) -> dict[str, float]:
    """Extract and clamp an axis confidence object from permissive model output."""
    cleaned = re.sub(r"```(?:json)?", "", str(text), flags=re.IGNORECASE).replace("```", "")
    match = re.search(r"\{.*\}", cleaned, flags=re.DOTALL)
    if match is None:
        raise ValueError("VLM response did not contain a JSON object")
    payload = json.loads(match.group(0))
    if not isinstance(payload, Mapping):
        raise TypeError("VLM response JSON was not an object")
    result: dict[str, float] = {}
    for axis in axes:
        value = payload.get(axis)
        if isinstance(value, bool):
            value = float(value)
        try:
            parsed = float(value)
        except (TypeError, ValueError):
            continue
        if np.isfinite(parsed):
            result[axis] = float(np.clip(parsed, 0.0, 1.0))
    return result


class VLMClient:
    def __init__(
        self,
        base_url: str,
        model: str,
        api_key: str = "",
        timeout: float = 60.0,
        concurrency: int = 4,
    ) -> None:
        self.base_url = base_url.rstrip("/")
        self.model = model
        self.api_key = api_key.strip()
        self.timeout = max(1.0, float(timeout))
        self.concurrency = max(1, int(concurrency))
        self._cancel_event: Event | None = None

    def probe(self) -> bool:
        try:
            request = urllib.request.Request(f"{self.base_url}/models", method="GET")
            with urllib.request.urlopen(request, timeout=min(self.timeout, 10.0)):
                return True
        except Exception as exc:  # noqa: BLE001
            LOGGER.warning("VLM server unavailable at %s: %s", self.base_url, exc)
            return False

    @staticmethod
    def _image_data_url(image: Image.Image) -> str:
        buffer = BytesIO()
        image.convert("RGB").save(buffer, format="JPEG", quality=85)
        encoded = base64.b64encode(buffer.getvalue()).decode("ascii")
        return f"data:image/jpeg;base64,{encoded}"

    def _request(self, images: list[Image.Image]) -> dict[str, float] | None:
        if self._cancel_event is not None and self._cancel_event.is_set():
            raise VLMCancelled
        system = (
            "You are a careful image safety classifier. Return ONLY one JSON object. "
            "Each requested value must be a number from 0 to 1, representing confidence "
            "that the named property is present. Do not include markdown or explanations."
        )
        user = (
            "Judge this image for all of these axes: "
            + ", ".join(VLM_AXES)
            + ". Use 0.5 when uncertain. Return exactly those keys when possible."
        )
        content = [{"type": "text", "text": user}]
        content.extend(
            {"type": "image_url", "image_url": {"url": self._image_data_url(image)}}
            for image in images
        )
        payload = {
            "model": self.model,
            # Deterministic JSON is the calibration contract; compatible local
            # servers do not expose a reliable per-axis probability stream.
            "temperature": 0,
            "max_tokens": 300,
            "messages": [
                {"role": "system", "content": system},
                {
                    "role": "user",
                    "content": content,
                },
            ],
        }
        headers = {"Content-Type": "application/json"}
        if self.api_key:
            headers["Authorization"] = f"Bearer {self.api_key}"
        request = urllib.request.Request(
            f"{self.base_url}/chat/completions",
            data=json.dumps(payload).encode("utf-8"),
            headers=headers,
            method="POST",
        )
        try:
            with urllib.request.urlopen(request, timeout=self.timeout) as response:
                response_payload = json.loads(response.read().decode("utf-8"))
            choices = response_payload.get("choices", [])
            text = ""
            if choices and isinstance(choices[0], Mapping):
                message = choices[0].get("message", {})
                if isinstance(message, Mapping):
                    content = message.get("content", "")
                    text = content if isinstance(content, str) else json.dumps(content)
            return parse_axis_json(text)
        except Exception as exc:  # noqa: BLE001
            LOGGER.warning("VLM judgment failed: %s", exc)
            return None

    def score_images(
        self,
        images: list[list[Image.Image]],
        cancel_event: Event | None = None,
        on_progress: Callable[[int, int], None] | None = None,
    ) -> list[dict[str, float] | None]:
        results: list[dict[str, float] | None] = [None] * len(images)
        if not images:
            return results
        if cancel_event is not None and cancel_event.is_set():
            raise VLMCancelled
        self._cancel_event = cancel_event
        completed = 0
        try:
            with ThreadPoolExecutor(max_workers=self.concurrency) as executor:
                futures = {
                    executor.submit(self._request, image): index for index, image in enumerate(images)
                }
                for future in as_completed(futures):
                    if cancel_event is not None and cancel_event.is_set():
                        for pending in futures:
                            pending.cancel()
                        raise VLMCancelled
                    results[futures[future]] = future.result()
                    completed += 1
                    if on_progress is not None:
                        on_progress(completed, len(images))
        finally:
            self._cancel_event = None
        return results
