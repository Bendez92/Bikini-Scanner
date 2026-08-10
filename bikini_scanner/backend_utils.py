"""Image decoding helpers and the backend interface every model backend implements.

The base classes live here rather than in `clip_backend` on purpose: `clip_backend`
imports torch and transformers at module scope, so anything that merely wants to *be* a
backend (the ONNX backend, the test double) would otherwise drag ~600 MB of framework
into the process just to inherit twenty lines. `clip_backend` re-exports both names, so
existing imports keep working.
"""

from __future__ import annotations

import hashlib
import logging
import os
from abc import ABC, abstractmethod
from collections.abc import Callable, Iterable, Iterator, Sequence
from concurrent.futures import ThreadPoolExecutor
from dataclasses import dataclass
from io import BytesIO
from itertools import islice
from pathlib import Path
from typing import Protocol

import numpy as np
from PIL import Image

LOGGER = logging.getLogger(__name__)


@dataclass(slots=True)
class DecodedImage:
    path: Path
    image: Image.Image | None
    content_hash: str | None
    error: str | None = None


def decode_image(path: Path) -> DecodedImage:
    try:
        data = path.read_bytes()
        content_hash = hashlib.sha1(data).hexdigest()
        with Image.open(BytesIO(data)) as image:
            return DecodedImage(
                path=path,
                image=image.convert("RGB").copy(),
                content_hash=content_hash,
            )
    except Exception as exc:  # noqa: BLE001
        LOGGER.warning("Skipping unreadable image %s", path, exc_info=False)
        return DecodedImage(path=path, image=None, content_hash=None, error=str(exc))


def decode_worker_count() -> int:
    """How many threads decode images alongside inference.

    JPEG decode is the bottleneck on a cold scan and PIL releases the GIL while it runs,
    so this scales with the machine instead of the previous hard cap of 4. It is still
    bounded: past ~8 threads the decoders start competing with the inference threads for
    the same cores and throughput stops improving. BIKINI_SCANNER_DECODE_WORKERS
    overrides it for benchmarking.
    """
    override = os.environ.get("BIKINI_SCANNER_DECODE_WORKERS", "").strip()
    if override:
        try:
            return max(1, min(32, int(override)))
        except ValueError:
            LOGGER.warning("Ignoring invalid BIKINI_SCANNER_DECODE_WORKERS=%r", override)
    return max(1, min(8, os.cpu_count() or 1))


def iter_decoded_image_batches(
    paths: Iterable[str | Path],
    batch_size: int = 16,
    decoder: Callable[[Path], DecodedImage] = decode_image,
) -> Iterator[list[DecodedImage]]:
    path_iterator = iter(paths)
    first_batch = [Path(path) for path in islice(path_iterator, batch_size)]
    if not first_batch:
        return
    max_workers = decode_worker_count()
    with ThreadPoolExecutor(max_workers=max_workers) as executor:
        decode_futures = [executor.submit(decoder, path) for path in first_batch]
        while decode_futures:
            decoded = [future.result() for future in decode_futures]
            next_batch = [Path(path) for path in islice(path_iterator, batch_size)]
            decode_futures = [executor.submit(decoder, path) for path in next_batch]
            yield decoded


class ImageEmbeddingBackend(Protocol):
    @property
    def image_embedding_dim(self) -> int: ...

    @property
    def active_device(self) -> str: ...

    @property
    def active_precision(self) -> str: ...

    def embed_images(self, paths: Sequence[str | Path], batch_size: int = 16) -> np.ndarray: ...

    def embed_pil_images(self, images: Sequence[Image.Image]) -> np.ndarray: ...

    def iter_image_batches(
        self, paths: Iterable[str | Path], batch_size: int = 16
    ) -> Iterator[list[DecodedImage]]: ...

    def embed_texts(self, prompts: Sequence[str]) -> np.ndarray: ...


class ClipBackendBase(ABC):
    active_device_value: str = "cpu"
    active_precision_value: str = "fp32"

    @property
    @abstractmethod
    def image_embedding_dim(self) -> int:
        raise NotImplementedError

    @property
    def active_device(self) -> str:
        return self.active_device_value

    @property
    def active_precision(self) -> str:
        return self.active_precision_value

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

    @abstractmethod
    def _embed_image_batch(self, images: Sequence[Image.Image]) -> list[np.ndarray]:
        raise NotImplementedError

    @abstractmethod
    def embed_texts(self, prompts: Sequence[str]) -> np.ndarray:
        raise NotImplementedError
