from __future__ import annotations

import json
import logging
from collections.abc import Mapping
from pathlib import Path
from typing import Any

from .config import HIGH_ACCURACY_MODEL, ScannerConfig
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
# Whole working setups, not one-line threshold tweaks. Each of these is a coherent
# answer to "what am I doing right now", which is what someone reaches for a profile
# to get; a profile that only moved the threshold was no faster than the slider.
BUILTIN_PROFILES = {
    "Strict": {
        "threshold": 0.7,
        "nsfw_filter": "exclude",
        "nsfw_threshold": 0.4,
        "deep_scan": "candidates",
        "exclude_minors": True,
        "minor_threshold": 0.3,
        "require_person": True,
        "person_threshold": 0.5,
    },
    "Loose": {
        "threshold": 0.2,
        "nsfw_filter": "include",
        "deep_scan": "candidates",
        "require_person": False,
        "require_female": False,
        "female_threshold": 0.0,
    },
    "Fast triage": {
        # Whole-frame only and a small batch: for getting a first answer out of a very
        # large folder quickly, accepting that distant subjects will be missed.
        "threshold": 0.35,
        "deep_scan": "off",
        "batch_size": 32,
        "enable_face_detection": False,
        "refine_model": "",
        "vlm_enabled": False,
    },
    "Thorough": {
        # Everything on: crop every image, re-check the borderline ones with the large
        # model. Slow by design, for a folder worth the time.
        "threshold": 0.3,
        "deep_scan": "always",
        "enable_face_detection": True,
        "refine_model": HIGH_ACCURACY_MODEL,
        "exclude_minors": True,
    },
    "Teaching": {
        # Wide net, nothing discarded, pooled learning on: for a session whose point is
        # to label examples rather than to produce a final set of matches.
        "threshold": 0.25,
        "nsfw_filter": "include",
        "deep_scan": "candidates",
        "require_person": False,
        "require_female": False,
        "female_threshold": 0.0,
        "global_learning": True,
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
    mapping = BUILTIN_PROFILES[name] if name in BUILTIN_PROFILES else load_profiles().get(name)
    return ScannerConfig.from_mapping(mapping) if mapping is not None else None


def inert_keys(mapping: Mapping[str, Any]) -> set[str]:
    """Keys in a profile that will not affect a scan as that profile is configured."""
    if str(mapping.get("pipeline", "")) == "legacy":
        return set()
    return {key for key in LEGACY_ONLY_KEYS if key in mapping}


# Settings that are credentials rather than configuration. A profile is meant to be
# saved, copied between machines and shared; a bearer token is none of those things.
SECRET_KEYS = frozenset({"vlm_api_key"})


def without_secrets(mapping: Mapping[str, Any]) -> dict[str, Any]:
    """A config dict safe to write to a file the user may share or sync."""
    return {key: ("" if key in SECRET_KEYS else value) for key, value in mapping.items()}


def save_profile(name: str, config: ScannerConfig) -> None:
    name = name.strip()
    if not name or name in BUILTIN_PROFILES:
        raise ValueError("Choose a non-empty custom profile name.")
    profiles = load_profiles()
    profiles[name] = without_secrets(config.to_dict())
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
