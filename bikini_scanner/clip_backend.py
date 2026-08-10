"""The PyTorch CLIP backend, plus the registry that maps a config to a backend.

`ImageEmbeddingBackend` and `ClipBackendBase` live in `backend_utils` (which does not
import torch) and are re-exported here so existing imports keep working.
"""

from __future__ import annotations

import logging
import os
import threading
from collections.abc import Sequence
from dataclasses import dataclass

import numpy as np
import torch
import torch.nn.functional as F
from PIL import Image
from transformers import CLIPModel, CLIPProcessor
from transformers.utils import logging as hf_logging

from .backend_utils import ClipBackendBase, DecodedImage, ImageEmbeddingBackend
from .config import ScannerConfig
from .image_formats import register_heif_support

__all__ = [
    "BACKENDS",
    "ClipBackendBase",
    "ClipTorchBackend",
    "DecodedImage",
    "ImageEmbeddingBackend",
    "get_backend",
]

register_heif_support()

LOGGER = logging.getLogger(__name__)
hf_logging.set_verbosity_error()
_BACKEND_LOAD_LOCK = threading.Lock()
_TORCH_BACKEND_CACHE: dict[tuple[str, str, str, bool], ClipTorchBackend] = {}
_ONNX_BACKEND_CACHE: dict[str, ClipBackendBase] = {}


@dataclass(slots=True)
class ClipTorchBackend(ClipBackendBase):
    model: CLIPModel
    processor: CLIPProcessor
    device: torch.device
    precision: str
    quantized: bool = False
    active_device_value: str = "cpu"
    active_precision_value: str = "fp32"

    @property
    def image_embedding_dim(self) -> int:
        return int(self.model.config.projection_dim)

    def _embed_image_batch(self, images: Sequence[Image.Image]) -> list[np.ndarray]:
        inputs = self.processor(images=list(images), return_tensors="pt")
        pixel_values = inputs["pixel_values"].to(self.device)
        if self.device.type == "cuda" and self.precision == "fp16":
            pixel_values = pixel_values.half()
        with torch.inference_mode():
            vision_outputs = self.model.vision_model(pixel_values=pixel_values)
            features = self.model.visual_projection(vision_outputs.pooler_output)
            features = F.normalize(features, p=2, dim=-1)
        return [row.detach().to("cpu").numpy().astype(np.float32) for row in features]

    def embed_texts(self, prompts: Sequence[str]) -> np.ndarray:
        inputs = self.processor(text=list(prompts), return_tensors="pt", padding=True)
        inputs = {key: value.to(self.device) for key, value in inputs.items()}
        with torch.inference_mode():
            text_outputs = self.model.text_model(**inputs)
            features = self.model.text_projection(text_outputs.pooler_output)
            features = F.normalize(features, p=2, dim=-1)
        return features.detach().to("cpu").numpy().astype(np.float32)


def _resolve_torch_device(config: ScannerConfig) -> torch.device:
    if config.device == "cpu":
        return torch.device("cpu")
    if config.device == "cuda":
        return torch.device("cuda" if torch.cuda.is_available() else "cpu")
    return torch.device("cuda" if torch.cuda.is_available() else "cpu")


def _resolve_torch_precision(config: ScannerConfig, device: torch.device) -> str:
    if config.precision == "fp32":
        return "fp32"
    if config.precision == "fp16":
        return "fp16" if device.type == "cuda" else "fp32"
    return "fp16" if device.type == "cuda" else "fp32"


def _load_clip_torch_backend(model_name: str, device_mode: str, precision_mode: str, quantize_cpu: bool) -> ClipTorchBackend:
    cache_key = (model_name, device_mode, precision_mode, bool(quantize_cpu))
    with _BACKEND_LOAD_LOCK:
        cached = _TORCH_BACKEND_CACHE.get(cache_key)
        if cached is not None:
            return cached
        device_config = ScannerConfig(model_name=model_name, device=device_mode, precision=precision_mode, quantize_cpu=quantize_cpu)
        device = _resolve_torch_device(device_config)
        precision = _resolve_torch_precision(device_config, device)
        LOGGER.info("Loading CLIP model %s on %s (%s)", model_name, device.type, precision)
        if device.type == "cpu":
            cpu_count = max(1, os.cpu_count() or 1)
            torch.set_num_threads(cpu_count)
            try:
                torch.set_num_interop_threads(max(1, cpu_count // 2))
            except RuntimeError:
                pass
        processor = CLIPProcessor.from_pretrained(model_name)
        model = CLIPModel.from_pretrained(model_name)
        model.to(device)
        if device.type == "cuda" and precision == "fp16":
            model = model.half()
        if device.type == "cpu" and quantize_cpu:
            try:
                model = torch.quantization.quantize_dynamic(model, {torch.nn.Linear}, dtype=torch.qint8)
            except Exception:  # noqa: BLE001
                LOGGER.warning("CPU dynamic quantization failed; using the standard model.")
        model.eval()
        backend = ClipTorchBackend(
            model=model,
            processor=processor,
            device=device,
            precision=precision,
            quantized=bool(quantize_cpu and device.type == "cpu"),
            active_device_value=device.type,
            active_precision_value=precision,
        )
        _TORCH_BACKEND_CACHE[cache_key] = backend
        return backend


def _load_clip_onnx_backend(model_name: str) -> ClipBackendBase:
    with _BACKEND_LOAD_LOCK:
        cached = _ONNX_BACKEND_CACHE.get(model_name)
        if cached is not None:
            return cached
        from .onnx_backend import load_onnx_backend

        backend = load_onnx_backend(model_name)
        _ONNX_BACKEND_CACHE[model_name] = backend
        return backend


BACKENDS = {
    "clip-torch": _load_clip_torch_backend,
    "clip-onnx": _load_clip_onnx_backend,
}


def get_backend(config: ScannerConfig | None = None) -> ImageEmbeddingBackend:
    resolved = config or ScannerConfig()
    backend_id = resolved.backend
    if backend_id not in BACKENDS:
        raise ValueError(f"Unsupported backend: {backend_id}")
    if backend_id == "clip-torch":
        return BACKENDS[backend_id](resolved.model_name, resolved.device, resolved.precision, resolved.quantize_cpu)
    return BACKENDS[backend_id](resolved.model_name)
