from __future__ import annotations

import importlib.util
import logging
from pathlib import Path
from typing import Any

from .user_prefs import prefs_path

LOGGER = logging.getLogger(__name__)


def plugins_dir() -> Path:
    return prefs_path().parent / "plugins"


def _usable_samples(updated: Any, name: str) -> list[dict[str, object]] | None:
    """Whatever the plugin returned, if the rest of the app can actually render it.

    Everything downstream indexes `sample["path"]`, so a plugin that returns strings, or
    dicts without that key, used to take the results grid down with it — and the
    exception surfaced far from the plugin that caused it. The plugin is skipped instead
    and the previous list is kept.
    """
    if updated is None:
        return None
    try:
        candidates = list(updated)
    except TypeError:
        LOGGER.warning("Plugin %s returned %s, which is not a list of samples; ignoring it", name, type(updated).__name__)
        return None
    usable: list[dict[str, object]] = []
    for entry in candidates:
        if isinstance(entry, dict) and str(entry.get("path", "")):
            usable.append(entry)
    if len(usable) != len(candidates):
        LOGGER.warning(
            "Plugin %s returned %d entries, %d of which had no usable 'path'; ignoring it",
            name,
            len(candidates),
            len(candidates) - len(usable),
        )
        return None
    return usable


def apply_plugins(
    state: Any, samples: list[dict[str, object]], enabled: bool = False
) -> list[dict[str, object]]:
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
            usable = _usable_samples(hook(state, result), path.name)
            if usable is not None:
                result = usable
            LOGGER.info("Applied plugin %s", path.name)
        except Exception:
            LOGGER.exception("Skipping failed plugin %s", path)
    return result
