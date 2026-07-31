from __future__ import annotations

import logging

LOGGER = logging.getLogger(__name__)


def register_heif_support() -> None:
    try:
        import pillow_heif
    except Exception:  # noqa: BLE001
        return
    try:
        pillow_heif.register_heif_opener()
    except Exception as exc:  # noqa: BLE001
        LOGGER.warning("Unable to register HEIF opener: %s", exc)
