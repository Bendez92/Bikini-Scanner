from __future__ import annotations

import base64
import ipaddress
import json
import logging
import re
import urllib.error
import urllib.request
from collections.abc import Callable, Mapping
from concurrent.futures import ThreadPoolExecutor, as_completed
from io import BytesIO
from threading import Event
from typing import Any
from urllib.parse import urlparse

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
# Local VLMs do not need full-resolution input; cap the longest side before sending to
# keep memory and token usage bounded.
VLM_MAX_IMAGE_SIDE = 512
# Hard ceiling on VLM workers. Config/GUI already validates user input, but this
# protects against runaway concurrency if the setting is set programmatically.
VLM_MAX_CONCURRENCY = 16


class VLMCancelled(Exception):
    """Raised when a scan cancellation interrupts VLM adjudication."""


def parse_axis_json(text: str, axes: tuple[str, ...] = VLM_AXES) -> dict[str, float]:
    """Extract and clamp an axis confidence object from permissive model output.

    The model is asked for JSON only, so the fast path is a direct ``json.loads`` of
    the cleaned text. When the model wraps the JSON in prose or markdown, we fall back
    to a brace search. The original greedy ``\\{.*\\}`` regex captured from the first
    ``{`` to the *last* ``}`` in the whole response, so any extra braces in trailing
    prose swallowed non-JSON text and made ``json.loads`` fail. The fallback now tries
    the smallest balanced-looking object first and expands only if that fails to parse.
    """
    cleaned = re.sub(r"```(?:json)?", "", str(text), flags=re.IGNORECASE).replace("```", "").strip()
    payload: object | None = None
    try:
        payload = json.loads(cleaned)
    except (json.JSONDecodeError, ValueError):
        payload = None
    if payload is None:
        # Find every '{' and try to parse from there to each subsequent '}'. The
        # shortest parseable object wins, which avoids grabbing prose that happens to
        # contain braces after the real JSON.
        for start in (i for i, char in enumerate(cleaned) if char == "{"):
            tail = cleaned[start:]
            for end in (j for j, char in enumerate(tail) if char == "}"):
                try:
                    payload = json.loads(tail[: end + 1])
                    break
                except (json.JSONDecodeError, ValueError):
                    continue
            if payload is not None:
                break
    if payload is None:
        raise ValueError("VLM response did not contain a JSON object")
    if not isinstance(payload, Mapping):
        raise TypeError("VLM response JSON was not an object")
    result: dict[str, float] = {}
    for axis in axes:
        value = payload.get(axis)
        if value is None:
            continue
        if isinstance(value, bool):
            value = float(value)
        try:
            parsed = float(value)
        except (TypeError, ValueError):
            continue
        if np.isfinite(parsed):
            result[axis] = float(np.clip(parsed, 0.0, 1.0))
    return result


def is_local_endpoint(base_url: str) -> bool:
    """True when `base_url` provably points at this machine.

    Callers use this to decide whether uploading images to the endpoint needs to be
    confirmed, so an unparseable or merely probable address counts as remote.
    """
    try:
        return VLMClient._is_loopback(urlparse(base_url).netloc)
    except ValueError:
        return False


class VLMClient:
    def __init__(
        self,
        base_url: str,
        model: str,
        api_key: str = "",
        timeout: float = 60.0,
        concurrency: int = 4,
    ) -> None:
        parsed = urlparse(base_url)
        # Enforce what the message has always claimed. urlopen also speaks ftp: and
        # file:, and "has a scheme and a netloc" was not enough to keep those out.
        if parsed.scheme not in {"http", "https"} or not parsed.netloc:
            raise ValueError(f"VLM base URL must be a valid http/https URL: {base_url}")
        if parsed.scheme == "http" and api_key.strip():
            raise ValueError("Refusing to send the VLM API key over plain HTTP")
        self.base_url = base_url.rstrip("/")
        self.model = model
        self.api_key = api_key.strip()
        self.timeout = max(1.0, float(timeout))
        self.concurrency = max(1, min(int(concurrency), VLM_MAX_CONCURRENCY))
        self._cancel_event: Event | None = None
        if not self._is_loopback(parsed.netloc):
            LOGGER.warning(
                "VLM endpoint %s is not on localhost; images will leave this machine", base_url
            )

    @staticmethod
    def _is_loopback(netloc: str) -> bool:
        """Loopback detection for the "images will leave this machine" warning.

        Deliberately errs towards *warning*: anything not provably loopback is treated
        as remote. `ipaddress` handles the IPv6 spellings (::1, ::ffff:127.0.0.1, and
        the zero-compressed forms) that a hand-rolled string check kept missing.
        """
        host = netloc.rsplit(":", 1)[0] if netloc.count(":") == 1 else netloc
        if host.startswith("["):
            host = host.partition("]")[0].lstrip("[")
        host = host.lower().strip()
        if host in {"localhost", "localhost.localdomain"}:
            return True
        try:
            address = ipaddress.ip_address(host)
        except ValueError:
            return False
        if address.is_loopback:
            return True
        mapped = getattr(address, "ipv4_mapped", None)
        return bool(mapped is not None and mapped.is_loopback)

    def probe(self) -> bool:
        try:
            request = urllib.request.Request(f"{self.base_url}/models", method="GET")
            with urllib.request.urlopen(request, timeout=min(self.timeout, 10.0)):
                return True
        except Exception as exc:  # noqa: BLE001
            LOGGER.warning("VLM server unavailable at %s: %s", self.base_url, exc)
            return False

    @staticmethod
    def _image_data_url(image: Image.Image, max_side: int = VLM_MAX_IMAGE_SIDE) -> str:
        image = image.convert("RGB")
        longest = max(image.size)
        if longest > max_side:
            scale = max_side / longest
            image = image.resize(
                (max(1, round(image.width * scale)), max(1, round(image.height * scale))),
                resample=Image.Resampling.LANCZOS,
            )
        buffer = BytesIO()
        image.save(buffer, format="JPEG", quality=85)
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
        content: list[dict[str, Any]] = [{"type": "text", "text": user}]
        content.extend({"type": "image_url", "image_url": {"url": self._image_data_url(image)}} for image in images)
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
        except urllib.error.HTTPError as exc:
            # A 401/403 (bad API key) or 429 (rate limited) is a configuration or
            # capacity problem, not a transient blip. Log it at ERROR with the status so
            # the user can tell their key is wrong from the log viewer instead of seeing
            # every image silently "fail to judge". We still return None so the scan
            # continues with the CLIP result rather than aborting the whole folder.
            level = LOGGER.error if exc.code in (401, 403, 429) else LOGGER.warning
            level("VLM server returned HTTP %d for %s: %s", exc.code, self.base_url, exc.reason)
            return None
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
                futures = {executor.submit(self._request, image): index for index, image in enumerate(images)}
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
