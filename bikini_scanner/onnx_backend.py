from __future__ import annotations

import logging
import sys
from dataclasses import dataclass
from functools import lru_cache
from pathlib import Path
from typing import TYPE_CHECKING, Iterable, Iterator, Sequence

import numpy as np
from PIL import Image

from .backend_utils import DecodedImage, iter_decoded_image_batches
from .config import DEFAULT_MODEL_NAME, REPO_ROOT, ScannerConfig
from .image_formats import register_heif_support

register_heif_support()

LOGGER = logging.getLogger(__name__)
VISION_ONNX_NAME = "clip_vision.onnx"
TEXT_ONNX_NAME = "clip_text.onnx"

if TYPE_CHECKING:
    from transformers import CLIPProcessor


def onnx_model_dir() -> Path:
    # Frozen builds unpack the graphs beside the bundled modules; REPO_ROOT is
    # only meaningful when running from a source checkout.
    bundle_dir = getattr(sys, "_MEIPASS", None)
    if bundle_dir is not None:
        return Path(bundle_dir) / "models"
    return REPO_ROOT / "models"


@dataclass(slots=True)
class ClipOnnxBackend:
    processor: CLIPProcessor
    vision_session: object
    text_session: object
    image_embedding_dim_value: int
    active_device_value: str = "cpu"
    active_precision_value: str = "fp32"

    @property
    def active_device(self) -> str:
        return self.active_device_value

    @property
    def active_precision(self) -> str:
        return self.active_precision_value

    @property
    def image_embedding_dim(self) -> int:
        return self.image_embedding_dim_value

    @classmethod
    def from_config(cls, config: ScannerConfig) -> "ClipOnnxBackend":
        if config.model_name != DEFAULT_MODEL_NAME:
            raise ValueError(
                "The clip-onnx backend currently supports the default CLIP model only: " f"{DEFAULT_MODEL_NAME}"
            )
        try:
            import onnxruntime as ort
        except Exception as exc:  # noqa: BLE001
            raise RuntimeError("clip-onnx requires onnxruntime. Install requirements-onnx.txt first.") from exc
        model_dir = onnx_model_dir()
        vision_path = model_dir / VISION_ONNX_NAME
        text_path = model_dir / TEXT_ONNX_NAME
        if not vision_path.exists() or not text_path.exists():
            raise FileNotFoundError(
                f"Missing ONNX graphs in {model_dir}. Run python -m scripts.export_onnx first."
            )
        try:
            from transformers import CLIPProcessor
        except Exception as exc:
            raise RuntimeError("The ONNX backend requires tokenizer support from transformers.") from exc
        processor = CLIPProcessor.from_pretrained(config.model_name)
        providers = ["CPUExecutionProvider"]
        vision_session = ort.InferenceSession(str(vision_path), providers=providers)
        text_session = ort.InferenceSession(str(text_path), providers=providers)
        output_shape = text_session.get_outputs()[0].shape
        embedding_dim = next((int(dim) for dim in reversed(output_shape) if isinstance(dim, int)), 512)
        return cls(
            processor=processor,
            vision_session=vision_session,
            text_session=text_session,
            image_embedding_dim_value=embedding_dim,
        )

    def _embed_image_batch(self, images: Sequence[Image.Image]) -> list[np.ndarray]:
        # return_tensors="np" keeps torch out of the ONNX build; "pt" would pull
        # the whole framework back in just to hand us an array.
        inputs = self.processor(images=list(images), return_tensors="np")
        pixel_values = np.asarray(inputs["pixel_values"], dtype=np.float32)
        output = self.vision_session.run(None, {"pixel_values": pixel_values})[0]
        output = self._normalize(output)
        return [row.astype(np.float32) for row in output]

    def embed_images(self, paths: Sequence[str | Path], batch_size: int = 16) -> np.ndarray:
        embeddings: list[np.ndarray] = []
        for batch in self.iter_image_batches(paths, batch_size=batch_size):
            valid_images = [record.image for record in batch if record.image is not None]
            if valid_images:
                embeddings.append(self.embed_pil_images(valid_images))
        if not embeddings:
            return np.empty((0, self.image_embedding_dim), dtype=np.float32)
        return np.vstack(embeddings).astype(np.float32)

    def embed_pil_images(self, images: Sequence[Image.Image]) -> np.ndarray:
        if not images:
            return np.empty((0, self.image_embedding_dim), dtype=np.float32)
        return np.vstack(self._embed_image_batch(images)).astype(np.float32)

    def iter_image_batches(
        self, paths: Iterable[str | Path], batch_size: int = 16
    ) -> Iterator[list[DecodedImage]]:
        yield from iter_decoded_image_batches(paths, batch_size=batch_size)

    def embed_texts(self, prompts: Sequence[str]) -> np.ndarray:
        # The exported text graph fixes the sequence axis at model_max_length,
        # so max_length padding is required rather than merely convenient.
        inputs = self.processor(
            text=list(prompts),
            return_tensors="np",
            padding="max_length",
            truncation=True,
            max_length=self.processor.tokenizer.model_max_length,
        )
        feed = {
            "input_ids": np.asarray(inputs["input_ids"], dtype=np.int64),
            "attention_mask": np.asarray(inputs["attention_mask"], dtype=np.int64),
        }
        output = self.text_session.run(None, feed)[0]
        return self._normalize(output).astype(np.float32)

    @staticmethod
    def _normalize(array: np.ndarray) -> np.ndarray:
        if array.size == 0:
            return array.astype(np.float32)
        norms = np.linalg.norm(array, axis=1, keepdims=True)
        norms = np.maximum(norms, 1e-12)
        return (array / norms).astype(np.float32)


@lru_cache(maxsize=None)
def load_onnx_backend(model_name: str) -> ClipOnnxBackend:
    config = ScannerConfig(backend="clip-onnx", model_name=model_name)
    return ClipOnnxBackend.from_config(config)


def get_backend(config: ScannerConfig | None = None) -> ClipOnnxBackend:
    # Deliberately mirrors clip_backend.get_backend so callers can pick a module
    # and stay on it. Routing the ONNX backend through clip_backend would import
    # torch and undo the reason this module exists.
    resolved = config or ScannerConfig(backend="clip-onnx")
    return load_onnx_backend(resolved.model_name)
