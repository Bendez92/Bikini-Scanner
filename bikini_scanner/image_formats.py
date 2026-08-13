"""How this app opens an image, so every code path opens it the same way.

The important rule here is EXIF orientation. A camera that is held sideways does not
rotate the pixels; it writes an orientation tag and leaves the raster as the sensor saw
it. Pillow never applies that tag on its own, so `Image.open()` on a portrait phone
photo hands back a landscape image lying on its side.

That matters far more for this app than for most. `regions.py` plans crops *by
position* - the upper band may evidence cleavage but not a bare midriff, the lower band
the reverse - so on a sideways frame those bands run across the subject instead of down
it, and the whole positional-voting scheme silently inverts. The model also sees a
rotated subject, which CLIP scores differently. Anything that decodes an image for
scoring, cropping, display or measurement must go through here.
"""

from __future__ import annotations

import logging
from pathlib import Path
from typing import IO

from PIL import Image, ImageOps

LOGGER = logging.getLogger(__name__)

# Bumped when a change here alters the pixels the model sees, so caches computed by an
# older build are discarded instead of silently reused. 1 = original (orientation tag
# ignored), 2 = EXIF orientation applied.
DECODE_VERSION = 2


def register_heif_support() -> None:
    try:
        import pillow_heif
    except ImportError:
        # pillow-heif is an optional extra; HEIC support is simply unavailable without it.
        return
    try:
        pillow_heif.register_heif_opener()
    except Exception as exc:  # noqa: BLE001
        LOGGER.warning("Unable to register HEIF opener: %s", exc)


def apply_orientation(image: Image.Image) -> Image.Image:
    """Rotate an image to its intended display orientation, per its EXIF tag.

    Returns the image unchanged when there is no orientation tag, which is the common
    case. Never raises: a malformed EXIF block must not cost us the image.
    """
    try:
        oriented = ImageOps.exif_transpose(image)
    except Exception as exc:  # noqa: BLE001
        LOGGER.debug("Could not read EXIF orientation, using the frame as stored: %s", exc)
        # Copy even on the failure path: callers open inside a `with`, and returning the
        # handle-backed image would hand them something that dies when the file closes.
        return image.copy()
    return oriented if oriented is not None else image.copy()


def open_oriented(source: str | Path | IO[bytes]) -> Image.Image:
    """Open an image and apply its EXIF orientation, converted to RGB.

    The result is detached from the file handle, so callers may use it after the source
    is closed. Use this instead of `Image.open(...).convert("RGB")` everywhere.
    """
    with Image.open(source) as handle:
        return apply_orientation(handle).convert("RGB")


def oriented_size(source: str | Path | IO[bytes]) -> tuple[int, int]:
    """Displayed (width, height) without decoding the full raster.

    Reads the header only, then swaps the axes when the orientation tag says the image
    is stored rotated - so a portrait phone photo reports portrait dimensions.
    """
    with Image.open(source) as handle:
        width, height = handle.size
        try:
            orientation = handle.getexif().get(0x0112, 1)
        except Exception:  # noqa: BLE001
            orientation = 1
    # 5, 6, 7 and 8 are the transposed orientations; the stored axes are swapped.
    if int(orientation or 1) in (5, 6, 7, 8):
        return height, width
    return width, height
