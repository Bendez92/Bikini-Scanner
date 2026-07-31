from __future__ import annotations

import hashlib
import logging
import os
from concurrent.futures import ThreadPoolExecutor
from dataclasses import dataclass
from io import BytesIO
from itertools import islice
from pathlib import Path
from typing import Callable, Iterable, Iterator

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


def iter_decoded_image_batches(
    paths: Iterable[str | Path],
    batch_size: int = 16,
    decoder: Callable[[Path], DecodedImage] = decode_image,
) -> Iterator[list[DecodedImage]]:
    path_iterator = iter(paths)
    first_batch = [Path(path) for path in islice(path_iterator, batch_size)]
    if not first_batch:
        return
    max_workers = max(1, min(4, os.cpu_count() or 1))
    with ThreadPoolExecutor(max_workers=max_workers) as executor:
        decode_futures = [executor.submit(decoder, path) for path in first_batch]
        while decode_futures:
            decoded = [future.result() for future in decode_futures]
            next_batch = [Path(path) for path in islice(path_iterator, batch_size)]
            decode_futures = [executor.submit(decoder, path) for path in next_batch]
            yield decoded
