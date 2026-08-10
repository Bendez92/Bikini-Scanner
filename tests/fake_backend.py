"""A deterministic stand-in for the CLIP backend, for fast hermetic tests.

The real backend downloads ~600 MB from Hugging Face and takes minutes to run a suite.
Almost nothing in this codebase depends on CLIP being *semantically* right - the scan
pipeline, the cascade gates, the caches, the learning loop and the review sampler all
care only that embeddings are stable, normalised, and vary with content.

So this backend derives an embedding from the image's own pixels:

    8x8 RGB thumbnail -> 192 floats -> fixed random projection -> 512-d -> L2 normalise

That makes it deterministic across runs and across machines, but *content-sensitive*,
which matters: a test that rotates an image (or fixes its EXIF orientation) sees the
embedding move, exactly as it would with the real model. Text embeddings are derived
the same way from a hash of the prompt string.

It subclasses ClipBackendBase rather than reimplementing the protocol, so tests still
exercise the real decode path (`iter_decoded_image_batches`, EXIF handling, the
content-hash de-duplication) instead of stubbing it out.
"""

from __future__ import annotations

import hashlib
from collections.abc import Sequence

import numpy as np
from PIL import Image

from bikini_scanner.backend_utils import ClipBackendBase

EMBEDDING_DIM = 512
_THUMB = 8
_PIXEL_FEATURES = _THUMB * _THUMB * 3


def _l2_normalize(matrix: np.ndarray) -> np.ndarray:
    matrix = np.asarray(matrix, dtype=np.float32)
    if matrix.ndim == 1:
        matrix = matrix[None, :]
    norms = np.linalg.norm(matrix, axis=1, keepdims=True)
    norms[norms <= 0] = 1.0
    return (matrix / norms).astype(np.float32)


def _projection() -> np.ndarray:
    """Fixed random projection from the pixel signature into embedding space."""
    generator = np.random.default_rng(20240617)
    return generator.standard_normal((_PIXEL_FEATURES, EMBEDDING_DIM)).astype(np.float32)


_PROJECTION = _projection()


def _pixel_signature(image: Image.Image) -> np.ndarray:
    thumbnail = image.convert("RGB").resize((_THUMB, _THUMB), Image.BILINEAR)
    values = np.asarray(thumbnail, dtype=np.float32).reshape(-1) / 255.0
    if values.size != _PIXEL_FEATURES:
        padded = np.zeros((_PIXEL_FEATURES,), dtype=np.float32)
        padded[: min(values.size, _PIXEL_FEATURES)] = values[:_PIXEL_FEATURES]
        values = padded
    return values - values.mean()


def embedding_for_text(prompt: str) -> np.ndarray:
    """Stable pseudo-embedding for one prompt string."""
    digest = hashlib.sha1(prompt.encode("utf-8")).digest()
    seed = int.from_bytes(digest[:8], "big", signed=False)
    generator = np.random.default_rng(seed)
    return _l2_normalize(generator.standard_normal((EMBEDDING_DIM,), dtype=np.float64).astype(np.float32))[0]


class FakeBackend(ClipBackendBase):
    """Drop-in ImageEmbeddingBackend that needs no model weights and no network."""

    active_device_value = "cpu"
    active_precision_value = "fp32"

    def __init__(self) -> None:
        self.text_calls = 0
        self.image_calls = 0
        self.images_embedded = 0

    @property
    def image_embedding_dim(self) -> int:
        return EMBEDDING_DIM

    def _embed_image_batch(self, images: Sequence[Image.Image]) -> list[np.ndarray]:
        self.image_calls += 1
        self.images_embedded += len(images)
        if not images:
            return []
        signatures = np.vstack([_pixel_signature(image) for image in images])
        projected = signatures @ _PROJECTION
        return list(_l2_normalize(projected))

    def embed_texts(self, prompts: Sequence[str]) -> np.ndarray:
        self.text_calls += 1
        if not prompts:
            return np.empty((0, EMBEDDING_DIM), dtype=np.float32)
        return np.vstack([embedding_for_text(str(prompt)) for prompt in prompts]).astype(np.float32)
