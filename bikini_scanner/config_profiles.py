from __future__ import annotations

import logging
import json
from pathlib import Path
from typing import Any, Mapping

from .config import ScannerConfig
from .safe_io import atomic_write_json, quarantine_broken_file
from .user_prefs import prefs_path

PROFILES_FILENAME = "profiles.json"
LOGGER = logging.getLogger(__name__)

# Settings that only do anything when `pipeline` is "legacy". Under the default cascade
# pipeline the learned model's share of the score is decided by its own measured AUC
# (see learning._blend_weight), so these two are ignored entirely. A profile that sets
# one without also setting pipeline="legacy" looks like it is tuning the scoring and is
# in fact doing nothing — which is what both built-in profiles used to do.
LEGACY_ONLY_KEYS = frozenset({"classifier_weight", "zero_shot_weight"})

# Deliberately narrow: every key here changes what a scan surfaces under the pipeline
# that actually runs. Neither profile touches the age gate — a profile that quietly
# loosened `exclude_minors` or `minor_threshold` would be an unpleasant surprise, so
# both inherit the defaults.
BUILTIN_PROFILES = {
    "Strict": {
        "threshold": 0.7,
        "nsfw_filter": "exclude",
        "nsfw_threshold": 0.4,
    },
    "Loose": {
        "threshold": 0.2,
        "nsfw_filter": "include",
    },
}


def profiles_path() -> Path:
    return prefs_path().parent / PROFILES_FILENAME


def load_profiles() -> dict[str, dict[str, Any]]:
    path = profiles_path()
    if not path.exists():
        return {}
    try:
        payload = json.loads(path.read_text(encoding="utf-8"))
    except Exception as exc:  # noqa: BLE001
        LOGGER.warning("Ignoring unreadable profiles %s: %s", path, exc)
        quarantine_broken_file(path, LOGGER, "invalid JSON")
        return {}
    if isinstance(payload, dict):
        if all(isinstance(value, dict) for value in payload.values()):
            return {str(name): value for name, value in payload.items()}
        LOGGER.warning("Ignoring unreadable profiles %s: invalid payload type", path)
        quarantine_broken_file(path, LOGGER, "invalid payload type")
        return {}
    LOGGER.warning("Ignoring unreadable profiles %s: invalid payload type", path)
    quarantine_broken_file(path, LOGGER, "invalid payload type")
    return {}


def save_profiles(profiles: dict[str, dict[str, Any]]) -> None:
    atomic_write_json(profiles_path(), profiles)


def profile_names() -> list[str]:
    return [*BUILTIN_PROFILES, *sorted(name for name in load_profiles() if name not in BUILTIN_PROFILES)]


def profile_config(name: str) -> ScannerConfig | None:
    if name in BUILTIN_PROFILES:
        mapping = BUILTIN_PROFILES[name]
    else:
        mapping = load_profiles().get(name)
    return ScannerConfig.from_mapping(mapping) if mapping is not None else None


def inert_keys(mapping: Mapping[str, Any]) -> set[str]:
    """Keys in a profile that will not affect a scan as that profile is configured."""
    if str(mapping.get("pipeline", "")) == "legacy":
        return set()
    return {key for key in LEGACY_ONLY_KEYS if key in mapping}


def save_profile(name: str, config: ScannerConfig) -> None:
    name = name.strip()
    if not name or name in BUILTIN_PROFILES:
        raise ValueError("Choose a non-empty custom profile name.")
    profiles = load_profiles()
    profiles[name] = config.to_dict()
    save_profiles(profiles)


def delete_profile(name: str) -> bool:
    if name in BUILTIN_PROFILES:
        return False
    profiles = load_profiles()
    if name not in profiles:
        return False
    del profiles[name]
    save_profiles(profiles)
    return True
