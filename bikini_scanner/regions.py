"""Region planning for the detail pass.

CLIP sees a 224x224 square. On a full photo a bikini top is a handful of pixels, which
is why whole-image scoring is weak for cleavage and midriff. This module plans a small
set of crops per image — the face (for the age and sex gates) and the chest/waist bands
below it (for the detail axes) — so each one is scored at a useful resolution.

Geometry is anchored to detected faces because a face box is a reliable scale reference
for the body below it. With no faces, it falls back to fixed bands of the frame.
"""

from __future__ import annotations

from collections.abc import Sequence
from dataclasses import dataclass
from typing import TYPE_CHECKING

from .vision_analysis import FaceBox

if TYPE_CHECKING:
    from PIL import Image

# Bumping this invalidates cached region embeddings, which are keyed by geometry version.
REGION_GEOMETRY_VERSION = 1

FULL_REGION = "full"
# Region kinds, and what each one is scored for.
KIND_FULL = "full"
KIND_FACE = "face"
KIND_CHEST = "chest"
KIND_WAIST = "waist"
KIND_TORSO = "torso"
KIND_BAND = "band"
# The fallback bands are unanchored guesses at where a body is, so they are kept
# distinct by position: the bottom of a frame is not evidence of cleavage, and the top
# is not evidence of a bare midriff.
KIND_BAND_UPPER = "band_upper"
KIND_BAND_MID = "band_mid"
KIND_BAND_LOWER = "band_lower"

# Crops anchored to a detected face sit where the geometry says they do; the fallback
# bands only guess. Callers weight the two differently.
ANCHORED_KINDS = frozenset({KIND_FACE, KIND_CHEST, KIND_WAIST, KIND_TORSO})
UNANCHORED_KINDS = frozenset({KIND_BAND, KIND_BAND_UPPER, KIND_BAND_MID, KIND_BAND_LOWER})

# Crops smaller than this are upscaled noise, not signal.
_MIN_CROP_PX = 48
# Bounds the per-image cost on group photos.
MAX_FACES = 3


@dataclass(frozen=True, slots=True)
class ImageRegion:
    key: str
    kind: str
    box: tuple[int, int, int, int] | None  # (left, top, right, bottom); None means the whole frame


def _clamp_box(
    left: float,
    top: float,
    right: float,
    bottom: float,
    width: int,
    height: int,
) -> tuple[int, int, int, int] | None:
    left_i = max(0, int(round(left)))
    top_i = max(0, int(round(top)))
    right_i = min(int(width), int(round(right)))
    bottom_i = min(int(height), int(round(bottom)))
    if right_i - left_i < _MIN_CROP_PX or bottom_i - top_i < _MIN_CROP_PX:
        return None
    return (left_i, top_i, right_i, bottom_i)


def _face_regions(index: int, face: FaceBox, width: int, height: int) -> list[ImageRegion]:
    """Face box plus the body bands beneath it, sized in multiples of the face."""
    fw = float(face.width)
    fh = float(face.height)
    fx = float(face.x)
    fy = float(face.y)
    regions: list[ImageRegion] = []

    # A little context around the face helps the age and sex axes.
    padded = _clamp_box(fx - 0.35 * fw, fy - 0.45 * fh, fx + fw + 0.35 * fw, fy + fh + 0.35 * fh, width, height)
    if padded is not None:
        regions.append(ImageRegion(key=f"face{index}", kind=KIND_FACE, box=padded))

    # Chest / cleavage band: roughly shoulders to sternum.
    chest = _clamp_box(fx - 1.15 * fw, fy + 1.05 * fh, fx + fw + 1.15 * fw, fy + 2.5 * fh, width, height)
    if chest is not None:
        regions.append(ImageRegion(key=f"chest{index}", kind=KIND_CHEST, box=chest))

    # Waist / midriff band.
    waist = _clamp_box(fx - 1.25 * fw, fy + 2.4 * fh, fx + fw + 1.25 * fw, fy + 4.6 * fh, width, height)
    if waist is not None:
        regions.append(ImageRegion(key=f"waist{index}", kind=KIND_WAIST, box=waist))

    # Whole torso: catches full-body swimwear shots that the bands cut in half.
    torso = _clamp_box(fx - 1.6 * fw, fy + 0.6 * fh, fx + fw + 1.6 * fw, fy + 5.5 * fh, width, height)
    if torso is not None:
        regions.append(ImageRegion(key=f"torso{index}", kind=KIND_TORSO, box=torso))
    return regions


def _fallback_regions(width: int, height: int) -> list[ImageRegion]:
    """No face found: score broad bands so a turned-away or cropped subject still registers."""
    regions: list[ImageRegion] = []
    bands = (
        ("bandtop", KIND_BAND_UPPER, 0.00, 0.55),
        ("bandmid", KIND_BAND_MID, 0.25, 0.80),
        ("bandlow", KIND_BAND_LOWER, 0.45, 1.00),
    )
    for key, kind, top_fraction, bottom_fraction in bands:
        box = _clamp_box(0, height * top_fraction, width, height * bottom_fraction, width, height)
        if box is not None:
            regions.append(ImageRegion(key=key, kind=kind, box=box))
    # Centre square: approximates what a subject-filling crop would look like.
    side = min(width, height)
    left = (width - side) / 2
    top = (height - side) / 2
    centre = _clamp_box(left, top, left + side, top + side, width, height)
    if centre is not None and (width != side or height != side):
        regions.append(ImageRegion(key="centre", kind=KIND_BAND, box=centre))
    return regions


def plan_regions(size: tuple[int, int], faces: Sequence[FaceBox], max_faces: int = MAX_FACES) -> list[ImageRegion]:
    """Regions to embed for one image, always starting with the full frame."""
    width, height = int(size[0]), int(size[1])
    regions: list[ImageRegion] = [ImageRegion(key=FULL_REGION, kind=KIND_FULL, box=None)]
    if width <= 0 or height <= 0:
        return regions
    ranked_faces = sorted(faces, key=lambda face: face.area, reverse=True)[:max_faces]
    if ranked_faces:
        for index, face in enumerate(ranked_faces):
            regions.extend(_face_regions(index, face, width, height))
    else:
        regions.extend(_fallback_regions(width, height))
    return regions


def crop_regions(image: Image.Image, regions: Sequence[ImageRegion]) -> list[tuple[str, Image.Image]]:
    """Materialise the planned crops. Unreadable crops are dropped, never faked."""
    crops: list[tuple[str, Image.Image]] = []
    for region in regions:
        if region.box is None:
            crops.append((region.key, image))
            continue
        try:
            crops.append((region.key, image.crop(region.box)))
        except Exception:  # noqa: BLE001
            continue
    return crops


def region_kind(key: str) -> str:
    """Recover a region's kind from its key, for scoring rules that depend on it.

    Region keys are also cache keys, so this has to keep working for crops embedded by
    an older build; the band keys have never changed, only how they are classified.
    """
    if key == FULL_REGION:
        return KIND_FULL
    for prefix, kind in (
        ("face", KIND_FACE),
        ("chest", KIND_CHEST),
        ("waist", KIND_WAIST),
        ("torso", KIND_TORSO),
        ("bandtop", KIND_BAND_UPPER),
        ("bandmid", KIND_BAND_MID),
        ("bandlow", KIND_BAND_LOWER),
    ):
        if key.startswith(prefix):
            return kind
    return KIND_BAND
