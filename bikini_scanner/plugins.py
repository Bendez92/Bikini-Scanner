from __future__ import annotations

import importlib.util
import logging
from pathlib import Path
from typing import Any

from .user_prefs import prefs_path

LOGGER = logging.getLogger(__name__)


def plugins_dir() -> Path:
    return prefs_path().parent / "plugins"


def apply_plugins(state: Any, samples: list[dict[str, object]], enabled: bool = False) -> list[dict[str, object]]:
    result = samples
    if not enabled:
        return result
    directory = plugins_dir()
    if not directory.exists():
        return result
    for path in sorted(directory.glob("*.py")):
        try:
            spec = importlib.util.spec_from_file_location(f"bikini_scanner_plugin_{path.stem}", path)
            if spec is None or spec.loader is None:
                continue
            module = importlib.util.module_from_spec(spec)
            spec.loader.exec_module(module)
            hook = getattr(module, "process_results", None)
            if not callable(hook):
                LOGGER.warning("Plugin %s has no process_results(state, samples) hook", path)
                continue
            updated = hook(state, result)
            if updated is not None:
                result = list(updated)
            LOGGER.info("Applied plugin %s", path.name)
        except Exception:
            LOGGER.exception("Skipping failed plugin %s", path)
    return result
