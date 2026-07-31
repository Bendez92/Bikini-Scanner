from __future__ import annotations

import numpy as np
from PIL import Image


def skin_fraction(image: Image.Image) -> float:
    """Return a cheap skin-colour fraction for ranking, never for exclusion.

    The heuristic combines YCbCr and HSV on a small image. It can mistake faces,
    wood, and warm lighting for skin, and under-counts dark skin or unusual lighting,
    so callers must use it only to order otherwise eligible images.
    """
    rgb = image.convert("RGB")
    longest = max(rgb.size)
    if longest > 128:
        scale = 128.0 / longest
        rgb = rgb.resize((max(1, round(rgb.width * scale)), max(1, round(rgb.height * scale))))
    array = np.asarray(rgb, dtype=np.float32) / 255.0
    red, green, blue = array[..., 0], array[..., 1], array[..., 2]
    maximum = array.max(axis=-1)
    minimum = array.min(axis=-1)
    delta = maximum - minimum
    saturation = np.divide(delta, maximum, out=np.zeros_like(delta), where=maximum > 1e-6)
    hue = np.zeros_like(maximum)
    red_mask = (maximum == red) & (delta > 1e-6)
    green_mask = (maximum == green) & (delta > 1e-6)
    blue_mask = (maximum == blue) & (delta > 1e-6)
    hue[red_mask] = ((green[red_mask] - blue[red_mask]) / delta[red_mask]) % 6.0
    hue[green_mask] = (blue[green_mask] - red[green_mask]) / delta[green_mask] + 2.0
    hue[blue_mask] = (red[blue_mask] - green[blue_mask]) / delta[blue_mask] + 4.0
    hue /= 6.0
    y = 0.299 * red + 0.587 * green + 0.114 * blue
    cb = 0.5 - 0.168736 * red - 0.331264 * green + 0.5 * blue
    cr = 0.5 + 0.5 * red - 0.418688 * green - 0.081312 * blue
    ycbcr = (cb > 0.20) & (cb < 0.55) & (cr > 0.28) & (cr < 0.62) & (y > 0.08)
    hsv = (hue < 0.16) | (hue > 0.92)
    hsv &= (saturation > 0.08) & (saturation < 0.9) & (maximum > 0.12)
    return float(np.clip(np.mean(ycbcr & hsv), 0.0, 1.0))
