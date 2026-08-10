from __future__ import annotations

import json
import logging
import os
from collections.abc import Mapping
from pathlib import Path
from typing import Any

from .safe_io import atomic_write_json, quarantine_broken_file

APP_NAME = "bikini-scanner"
LOGGER = logging.getLogger(__name__)


def prefs_path() -> Path:
    if os.name == "nt":
        base = os.environ.get("APPDATA") or Path.home() / "AppData" / "Roaming"
        return Path(base) / APP_NAME / "prefs.json"
    if sys_platform_startswith("darwin"):
        return Path.home() / "Library" / "Application Support" / APP_NAME / "prefs.json"
    xdg_config = os.environ.get("XDG_CONFIG_HOME")
    base = Path(xdg_config) if xdg_config else Path.home() / ".config"
    return base / APP_NAME / "prefs.json"


def sys_platform_startswith(prefix: str) -> bool:
    import sys

    return sys.platform.startswith(prefix)


def load_user_prefs() -> dict[str, Any]:
    path = prefs_path()
    if not path.exists():
        return {}
    try:
        with path.open("r", encoding="utf-8") as handle:
            payload = json.load(handle)
    except Exception as exc:  # noqa: BLE001
        LOGGER.warning("Ignoring unreadable user prefs %s: %s", path, exc)
        quarantine_broken_file(path, LOGGER, "invalid JSON")
        return {}
    if isinstance(payload, dict):
        return payload
    LOGGER.warning("Ignoring unreadable user prefs %s: invalid payload type", path)
    quarantine_broken_file(path, LOGGER, "invalid payload type")
    return {}


def save_user_prefs(payload: Mapping[str, Any]) -> None:
    atomic_write_json(prefs_path(), dict(payload))
