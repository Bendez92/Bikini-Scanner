"""Shared metadata and validation for scalar scanner settings.

Every scalar `ScannerConfig` field appears exactly once in `FIELD_SPECS`, and that
entry is the only place its type, bounds, choices and folder-override status are
written down. `ScannerConfig.from_mapping`, the settings dialog and the folder
override allow-list all read from here, so the three cannot drift apart.
"""

from __future__ import annotations

import logging
import math
from dataclasses import dataclass
from typing import Literal

LOGGER = logging.getLogger(__name__)

Kind = Literal["str", "choice", "bool", "int", "float"]


@dataclass(frozen=True, slots=True)
class FieldSpec:
    key: str
    kind: Kind
    minimum: float | None = None
    maximum: float | None = None
    choices: frozenset[str] | None = None
    folder_overridable: bool = False
    entry_bounds: tuple[float, float] | None = None
    entry_message: str = ""


# The settings dialog rejects a value at or below the bound rather than below it.
EXCLUSIVE_ENTRY_MINIMUMS: frozenset[str] = frozenset({"zero_shot_scale", "minor_threshold"})

_AGE_GATE_ENTRY_MESSAGE = (
    "Minor sensitivity must be above 0 and at most 1. Lower is stricter; to turn the age gate "
    "off entirely, clear the 'Exclude images that may show a minor' checkbox."
)

FIELD_SPECS: tuple[FieldSpec, ...] = (
    FieldSpec("backend", "str"),
    FieldSpec("model_name", "str"),
    FieldSpec(
        "device",
        "choice",
        choices=frozenset({"auto", "cpu", "cuda"}),
        folder_overridable=True,
        entry_message="Device must be auto, cpu, or cuda.",
    ),
    FieldSpec(
        "precision",
        "choice",
        choices=frozenset({"auto", "fp32", "fp16"}),
        folder_overridable=True,
        entry_message="Precision must be auto, fp32, or fp16.",
    ),
    FieldSpec("quantize_cpu", "bool", folder_overridable=True),
    FieldSpec("preload_backend", "bool", folder_overridable=True),
    FieldSpec(
        "batch_size",
        "int",
        minimum=1,
        maximum=1024,
        folder_overridable=True,
        entry_bounds=(1, math.inf),
        entry_message="Batch size must be a positive integer.",
    ),
    FieldSpec(
        "threshold",
        "float",
        minimum=0.0,
        maximum=1.0,
        folder_overridable=True,
        entry_message="Threshold must be between 0 and 1.",
    ),
    FieldSpec(
        "zero_shot_scale",
        "float",
        minimum=0.01,
        maximum=1000.0,
        folder_overridable=True,
        entry_bounds=(0.0, math.inf),
        entry_message="Zero-shot scale must be positive.",
    ),
    FieldSpec(
        "classifier_weight",
        "float",
        minimum=0.0,
        maximum=100.0,
        folder_overridable=True,
        entry_bounds=(0.0, math.inf),
        entry_message="Blend weights must be non-negative.",
    ),
    FieldSpec(
        "zero_shot_weight",
        "float",
        minimum=0.0,
        maximum=100.0,
        folder_overridable=True,
        entry_bounds=(0.0, math.inf),
        entry_message="Blend weights must be non-negative.",
    ),
    FieldSpec(
        "nsfw_filter",
        "choice",
        choices=frozenset({"include", "exclude", "only"}),
        folder_overridable=True,
        entry_message="NSFW mode must be include, exclude, or only.",
    ),
    FieldSpec(
        "nsfw_threshold",
        "float",
        minimum=0.0,
        maximum=1.0,
        folder_overridable=True,
        entry_message="NSFW threshold must be between 0 and 1.",
    ),
    FieldSpec("require_person", "bool", folder_overridable=True),
    FieldSpec(
        "person_threshold",
        "float",
        minimum=0.0,
        maximum=1.0,
        folder_overridable=True,
        entry_message="Person threshold must be between 0 and 1.",
    ),
    FieldSpec("enable_face_detection", "bool", folder_overridable=True),
    # "legacy" skips the cascade, and with it the age gate, so it is not overridable.
    FieldSpec("pipeline", "choice", choices=frozenset({"cascade", "legacy"})),
    FieldSpec(
        "deep_scan",
        "choice",
        choices=frozenset({"candidates", "always", "off"}),
        folder_overridable=True,
        entry_message="Deep scan must be candidates, always, or off.",
    ),
    FieldSpec("person_gate_threshold", "float", minimum=0.0, maximum=1.0, folder_overridable=True),
    FieldSpec("require_female", "bool", folder_overridable=True),
    FieldSpec(
        "female_threshold",
        "float",
        minimum=0.0,
        maximum=1.0,
        folder_overridable=True,
        entry_message="Female cut-off must be between 0 and 1.",
    ),
    FieldSpec("exclude_minors", "bool"),
    # Floored above zero on purpose: at 0 the age gate matches essentially every image,
    # which looks like the scanner finding nothing rather than a bad setting. The dialog
    # is stricter still and refuses 0 outright instead of quietly raising it to 0.01.
    FieldSpec(
        "minor_threshold",
        "float",
        minimum=0.01,
        maximum=1.0,
        entry_bounds=(0.0, 1.0),
        entry_message=_AGE_GATE_ENTRY_MESSAGE,
    ),
    FieldSpec("min_adult_confidence", "float", minimum=0.0, maximum=1.0),
    FieldSpec("max_faces", "int", minimum=1, maximum=32, folder_overridable=True),
    # Age-gate constants: the minimums stop a profile or override from weakening the
    # safety gate to the point of uselessness. The upper bound is the evidence space.
    FieldSpec("child_adult_margin", "float", minimum=0.05, maximum=1.0),
    FieldSpec("strongly_minor_threshold", "float", minimum=0.50, maximum=1.0),
    FieldSpec("face_anchored_margin", "float", minimum=0.02, maximum=1.0),
    FieldSpec("weak_adult_detail", "float", minimum=0.20, maximum=1.0),
    FieldSpec("refine_model", "str"),
    FieldSpec("refine_band", "float", minimum=0.0, maximum=1.0, folder_overridable=True),
    FieldSpec("refine_max_images", "int", minimum=0, maximum=1_000_000, folder_overridable=True),
    FieldSpec("refine_weight", "float", minimum=0.0, maximum=1.0, folder_overridable=True),
    FieldSpec("vlm_enabled", "bool"),
    FieldSpec("vlm_base_url", "str"),
    FieldSpec("vlm_model", "str"),
    FieldSpec("vlm_api_key", "str"),
    FieldSpec("vlm_band", "float", minimum=0.0, maximum=1.0, entry_message="VLM band must be between 0 and 1."),
    FieldSpec(
        "vlm_max_images",
        "int",
        minimum=0,
        maximum=1_000_000,
        entry_message="VLM maximum images must be from 0 to 1,000,000.",
    ),
    FieldSpec(
        "vlm_concurrency",
        "int",
        minimum=1,
        maximum=64,
        entry_message="VLM concurrency must be from 1 to 64.",
    ),
    FieldSpec("vlm_timeout", "float", minimum=1.0, maximum=600.0),
    FieldSpec("vlm_weight", "float", minimum=0.0, maximum=1.0),
    FieldSpec("enable_plugins", "bool"),
    FieldSpec("global_learning", "bool"),
    # Below 1.0 so the zero-shot prompts always retain a voice in the final score.
    FieldSpec("max_learning_weight", "float", minimum=0.0, maximum=0.95),
    FieldSpec("detail_strongest_weight", "float", minimum=0.0, maximum=1.0, folder_overridable=True),
    FieldSpec("detail_average_weight", "float", minimum=0.0, maximum=1.0, folder_overridable=True),
)

SPECS_BY_KEY: dict[str, FieldSpec] = {spec.key: spec for spec in FIELD_SPECS}


class FieldError(ValueError):
    """Carries the message shown to the user verbatim."""


def _spec(key: str, kind: Kind) -> FieldSpec:
    spec = SPECS_BY_KEY[key]
    if spec.kind != kind:
        raise ValueError(f"setting {key!r} is a {spec.kind} field, not {kind}")
    return spec


def _label(key: str) -> str:
    return key.replace("_", " ").capitalize()


def coerce_float(key: str, value: object, default: float) -> float:
    """Parse a float setting, rejecting values the rest of the app cannot cope with.

    The Settings dialog range-checks every one of these, but from_mapping is the funnel
    for input the dialog never sees - imported settings files, saved profiles and
    per-folder overrides. Without the same bounds here a hand-edited or corrupted file
    could set a NaN threshold (nothing ever matches, silently), a zero prompt scale
    (every axis flattens to "no opinion"), or a zero minor_threshold (the age gate
    excludes nearly everything) with no indication anything was wrong.
    """
    spec = _spec(key, "float")
    if value is None:
        return default
    if not isinstance(value, (int, float, str)):
        LOGGER.warning("Ignoring invalid setting %r; using %r", value, default)
        return default
    try:
        parsed = float(value)
    except ValueError:
        LOGGER.warning("Ignoring invalid setting %r; using %r", value, default)
        return default
    if not math.isfinite(parsed):
        LOGGER.warning("Ignoring non-finite setting %r; using %r", value, default)
        return default
    clamped = parsed
    if spec.minimum is not None:
        clamped = max(clamped, spec.minimum)
    if spec.maximum is not None:
        clamped = min(clamped, spec.maximum)
    if clamped != parsed:
        LOGGER.warning("Clamped out-of-range setting %r to %r", parsed, clamped)
    return clamped


def coerce_int(key: str, value: object, default: int) -> int:
    """Parse a persisted integer, falling back to the default when out of range."""
    spec = _spec(key, "int")
    if value is None:
        return default
    if not isinstance(value, (int, float, str)):
        LOGGER.warning("Ignoring invalid setting %r; using %r", value, default)
        return default
    try:
        parsed = int(value)
    except (ValueError, OverflowError):
        LOGGER.warning("Ignoring invalid setting %r; using %r", value, default)
        return default
    below = spec.minimum is not None and parsed < spec.minimum
    above = spec.maximum is not None and parsed > spec.maximum
    if below or above:
        LOGGER.warning("Ignoring out-of-range setting %r; using %r", parsed, default)
        return default
    return parsed


def coerce_choice(key: str, value: object, default: str) -> str:
    """Parse a persisted choice, falling back to the default when unrecognised."""
    spec = _spec(key, "choice")
    if value is None:
        return default
    if not isinstance(value, str):
        LOGGER.warning("Ignoring invalid setting %r; using %r", value, default)
        return default
    parsed = value.strip()
    if spec.choices is None or parsed not in spec.choices:
        LOGGER.warning("Ignoring invalid setting %r; using %r", value, default)
        return default
    return parsed


def _entry_bounds(spec: FieldSpec) -> tuple[float, float]:
    if spec.entry_bounds is not None:
        return spec.entry_bounds
    lower = spec.minimum if spec.minimum is not None else -math.inf
    upper = spec.maximum if spec.maximum is not None else math.inf
    return lower, upper


def _in_entry_bounds(spec: FieldSpec, parsed: float) -> bool:
    lower, upper = _entry_bounds(spec)
    if spec.key in EXCLUSIVE_ENTRY_MINIMUMS:
        return lower < parsed <= upper
    return lower <= parsed <= upper


def parse_float_entry(key: str, raw: str) -> float:
    """Parse a dialog entry, raising `FieldError` with the message the user sees."""
    spec = _spec(key, "float")
    message = spec.entry_message or f"{_label(key)} must be numeric."
    try:
        parsed = float(raw.strip())
    except ValueError:
        raise FieldError(message) from None
    if not math.isfinite(parsed) or not _in_entry_bounds(spec, parsed):
        raise FieldError(message)
    return parsed


def parse_int_entry(key: str, raw: str) -> int:
    """Parse a dialog entry, raising `FieldError` with the message the user sees."""
    spec = _spec(key, "int")
    message = spec.entry_message or f"{_label(key)} must be a whole number."
    try:
        parsed = int(raw.strip())
    except ValueError:
        raise FieldError(message) from None
    if not _in_entry_bounds(spec, parsed):
        raise FieldError(message)
    return parsed


def parse_choice_entry(key: str, raw: str) -> str:
    """Parse a dialog choice, raising `FieldError` with the message the user sees."""
    spec = _spec(key, "choice")
    parsed = raw.strip()
    if spec.choices is None or parsed not in spec.choices:
        listed = ", ".join(sorted(spec.choices or ()))
        raise FieldError(spec.entry_message or f"{_label(key)} must be one of {listed}.")
    return parsed
