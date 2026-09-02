from __future__ import annotations

import logging
import math
from collections.abc import Mapping
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Any

from .prompts import (
    DEFAULT_PROMPT_SET,
    AxisConfig,
    PromptSpec,
    available_prompt_sets,
    load_prompt_set,
)

LOGGER = logging.getLogger(__name__)
REPO_ROOT = Path(__file__).resolve().parents[1]

DEFAULT_BACKEND = "clip-torch"
DEFAULT_MODEL_NAME = "openai/clip-vit-base-patch32"
DEFAULT_DEVICE = "auto"
DEFAULT_PRECISION = "auto"
_DEFAULT_PROMPT_SET = load_prompt_set()
DEFAULT_BIKINI_POSITIVE_PROMPTS = list(_DEFAULT_PROMPT_SET.positive)
DEFAULT_BIKINI_NEGATIVE_PROMPTS = list(_DEFAULT_PROMPT_SET.negative)
# Compatibility aliases for callers that imported the original names.
DEFAULT_POSITIVE_PROMPTS = DEFAULT_BIKINI_POSITIVE_PROMPTS
DEFAULT_NEGATIVE_PROMPTS = DEFAULT_BIKINI_NEGATIVE_PROMPTS
DEFAULT_BATCH_SIZE = 16
# Matched to the cascade's score distribution: on a real folder, accepted images sat at
# 0.40-0.68 and rejected ones below 0.16.
DEFAULT_THRESHOLD = 0.35
# CLIP prompt margins are small (cosine differences of ~0.01-0.05), so a low scale
# squashes every axis to "no opinion" around 0.5. Calibrated against real labels: 40
# spreads the evidence out without saturating it.
DEFAULT_ZERO_SHOT_SCALE = 40.0
DEFAULT_CLASSIFIER_WEIGHT = 0.7
DEFAULT_ZERO_SHOT_WEIGHT = 0.3
DEFAULT_QUANTIZE_CPU = False
DEFAULT_PRELOAD_BACKEND = True
DEFAULT_NSFW_FILTER = "include"
DEFAULT_NSFW_THRESHOLD = 0.5
DEFAULT_REQUIRE_PERSON = False
DEFAULT_PERSON_THRESHOLD = 0.5
DEFAULT_ENABLE_FACE_DETECTION = False

# --- Cascade pipeline -------------------------------------------------------
# The scan runs as gates: people -> female subject -> not a minor -> detail axes.
# Each gate is scored softly (it scales the final score) and enforced hard (below the
# threshold the image is dropped from matches entirely).
DEFAULT_PIPELINE = "cascade"
DEFAULT_DEEP_SCAN = "candidates"
# Only used when require_person is switched on. Off by default because CLIP's "person"
# reading collapses on close-up body shots - the very images this tool exists to find.
DEFAULT_PERSON_GATE_THRESHOLD = 0.35
DEFAULT_REQUIRE_FEMALE = True
# 0.0 means the sex stage only ranks (a non-female subject is pushed down, never
# dropped). Raising it above 0 turns it into a hard filter. Left soft by default
# because on real photos a hard female gate silently binned a genuine match whose
# close-up crop gave CLIP nothing to judge sex from.
DEFAULT_FEMALE_THRESHOLD = 0.0
# Age gating is deliberately biased toward exclusion: CLIP age estimates are coarse, so
# a low bar for "this might be a child" and a floor on positive adult evidence are both
# applied. Raising DEFAULT_MINOR_THRESHOLD makes the scanner *less* careful.
DEFAULT_EXCLUDE_MINORS = True
DEFAULT_MINOR_THRESHOLD = 0.30
DEFAULT_MIN_ADULT_CONFIDENCE = 0.25
DEFAULT_MAX_FACES = 3
# Age-gate internal constants. These were hardcoded in cascade.py; exposing them as
# config fields (with safe minimums) lets a user tighten the gate without code edits.
# The minimums below are enforced in from_mapping and prevent weakening the gate
# to the point of uselessness. Raising any of these makes the scanner LESS careful.
DEFAULT_CHILD_ADULT_MARGIN = 0.10  # child must out-argue adult by this much
DEFAULT_STRONGLY_MINOR_THRESHOLD = 0.75  # overwhelming child evidence, standalone
DEFAULT_FACE_ANCHORED_MARGIN = 0.05  # smaller margin when a real face crop exists
DEFAULT_WEAK_ADULT_DETAIL = 0.35  # detail score above which adult evidence is required
# Opt-in second opinion from a larger model on borderline images only.
DEFAULT_REFINE_MODEL = ""
DEFAULT_REFINE_BAND = 0.18
DEFAULT_REFINE_MAX_IMAGES = 400
DEFAULT_REFINE_WEIGHT = 0.65
DEFAULT_VLM_ENABLED = False
DEFAULT_VLM_BASE_URL = "http://localhost:11434/v1"
DEFAULT_VLM_MODEL = "qwen2.5vl:7b"
DEFAULT_VLM_API_KEY = ""
DEFAULT_VLM_BAND = 0.18
DEFAULT_VLM_MAX_IMAGES = 400
DEFAULT_VLM_CONCURRENCY = 4
DEFAULT_VLM_TIMEOUT = 60.0
DEFAULT_VLM_WEIGHT = 0.65
DEFAULT_ENABLE_PLUGINS = False
HIGH_ACCURACY_MODEL = "openai/clip-vit-large-patch14"
# Accept/REJECT decisions are pooled across folders unless this is turned off.
DEFAULT_GLOBAL_LEARNING = True
# Cap on the learned model's share of the final score. Below 1.0 so zero-shot
# prompts always retain a voice even when the classifier is highly trusted.
DEFAULT_MAX_LEARNING_WEIGHT = 0.85
# How the detail axes combine into the headline score (soft OR, so two weak signals
# reinforce instead of one having to carry the image alone). The strongest single
# axis gets the larger share; the average of all axes adds corroboration.
DEFAULT_DETAIL_STRONGEST_WEIGHT = 0.65
DEFAULT_DETAIL_AVERAGE_WEIGHT = 0.35
DEFAULT_DETAIL_WEIGHTS = {
    "bikini": 1.0,
    "cleavage": 0.85,
    "midriff": 0.85,
    "bikini_top": 0.7,
    "bikini_bottom": 0.7,
}


@dataclass(slots=True)
class ScannerConfig:
    backend: str = DEFAULT_BACKEND
    model_name: str = DEFAULT_MODEL_NAME
    device: str = DEFAULT_DEVICE
    precision: str = DEFAULT_PRECISION
    quantize_cpu: bool = DEFAULT_QUANTIZE_CPU
    preload_backend: bool = DEFAULT_PRELOAD_BACKEND
    # No cache invalidation is needed: _embedding_namespace/_region_namespace identify
    # image embeddings by model, text embeddings are computed per run, and VLM prompts
    # have their own VLM_PROMPT_VERSION.
    prompt_set: str = DEFAULT_PROMPT_SET
    # These top-level fields remain for settings/import compatibility. Primary
    # scoring uses the canonical "bikini" axis below.
    positive_prompts: list[str] = field(default_factory=lambda: list(DEFAULT_BIKINI_POSITIVE_PROMPTS))
    negative_prompts: list[str] = field(default_factory=lambda: list(DEFAULT_BIKINI_NEGATIVE_PROMPTS))
    axis_prompts: dict[str, AxisConfig] = field(default_factory=lambda: load_prompt_set().axes)
    batch_size: int = DEFAULT_BATCH_SIZE
    threshold: float = DEFAULT_THRESHOLD
    zero_shot_scale: float = DEFAULT_ZERO_SHOT_SCALE
    classifier_weight: float = DEFAULT_CLASSIFIER_WEIGHT
    zero_shot_weight: float = DEFAULT_ZERO_SHOT_WEIGHT
    nsfw_filter: str = DEFAULT_NSFW_FILTER
    nsfw_threshold: float = DEFAULT_NSFW_THRESHOLD
    require_person: bool = DEFAULT_REQUIRE_PERSON
    person_threshold: float = DEFAULT_PERSON_THRESHOLD
    enable_face_detection: bool = DEFAULT_ENABLE_FACE_DETECTION
    pipeline: str = DEFAULT_PIPELINE
    deep_scan: str = DEFAULT_DEEP_SCAN
    person_gate_threshold: float = DEFAULT_PERSON_GATE_THRESHOLD
    require_female: bool = DEFAULT_REQUIRE_FEMALE
    female_threshold: float = DEFAULT_FEMALE_THRESHOLD
    exclude_minors: bool = DEFAULT_EXCLUDE_MINORS
    minor_threshold: float = DEFAULT_MINOR_THRESHOLD
    min_adult_confidence: float = DEFAULT_MIN_ADULT_CONFIDENCE
    max_faces: int = DEFAULT_MAX_FACES
    # Age-gate internal constants (see DEFAULT_* docs above). Safe minimums are
    # enforced in from_mapping so a profile or override cannot weaken the gate.
    child_adult_margin: float = DEFAULT_CHILD_ADULT_MARGIN
    strongly_minor_threshold: float = DEFAULT_STRONGLY_MINOR_THRESHOLD
    face_anchored_margin: float = DEFAULT_FACE_ANCHORED_MARGIN
    weak_adult_detail: float = DEFAULT_WEAK_ADULT_DETAIL
    refine_model: str = DEFAULT_REFINE_MODEL
    refine_band: float = DEFAULT_REFINE_BAND
    refine_max_images: int = DEFAULT_REFINE_MAX_IMAGES
    refine_weight: float = DEFAULT_REFINE_WEIGHT
    vlm_enabled: bool = DEFAULT_VLM_ENABLED
    vlm_base_url: str = DEFAULT_VLM_BASE_URL
    vlm_model: str = DEFAULT_VLM_MODEL
    vlm_api_key: str = DEFAULT_VLM_API_KEY
    vlm_band: float = DEFAULT_VLM_BAND
    vlm_max_images: int = DEFAULT_VLM_MAX_IMAGES
    vlm_concurrency: int = DEFAULT_VLM_CONCURRENCY
    vlm_timeout: float = DEFAULT_VLM_TIMEOUT
    vlm_weight: float = DEFAULT_VLM_WEIGHT
    enable_plugins: bool = DEFAULT_ENABLE_PLUGINS
    global_learning: bool = DEFAULT_GLOBAL_LEARNING
    max_learning_weight: float = DEFAULT_MAX_LEARNING_WEIGHT
    detail_strongest_weight: float = DEFAULT_DETAIL_STRONGEST_WEIGHT
    detail_average_weight: float = DEFAULT_DETAIL_AVERAGE_WEIGHT
    detail_weights: dict[str, float] = field(default_factory=lambda: dict(DEFAULT_DETAIL_WEIGHTS))

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)

    @classmethod
    def from_mapping(cls, mapping: Mapping[str, Any] | None) -> ScannerConfig:
        config = cls()
        if not mapping:
            return config
        config.backend = _coerce_str(mapping.get("backend"), config.backend)
        config.model_name = _coerce_str(mapping.get("model_name"), config.model_name)
        config.device = _coerce_choice(mapping.get("device"), config.device, {"auto", "cpu", "cuda"})
        config.precision = _coerce_choice(mapping.get("precision"), config.precision, {"auto", "fp32", "fp16"})
        config.quantize_cpu = _coerce_bool(mapping.get("quantize_cpu"), config.quantize_cpu)
        config.preload_backend = _coerce_bool(mapping.get("preload_backend"), config.preload_backend)
        raw_prompt_set = mapping.get("prompt_set")
        current_prompt_set = config.prompt_set
        prompt_set_names = set(available_prompt_sets())
        config.prompt_set = _coerce_choice(raw_prompt_set, config.prompt_set, prompt_set_names)
        if raw_prompt_set is not None and not (
            isinstance(raw_prompt_set, str) and raw_prompt_set.strip() in prompt_set_names
        ):
            LOGGER.warning("Ignoring invalid prompt set %r; using %r", raw_prompt_set, config.prompt_set)
        if config.prompt_set != current_prompt_set:
            prompt_set = load_prompt_set(config.prompt_set)
            config.positive_prompts = list(prompt_set.positive)
            config.negative_prompts = list(prompt_set.negative)
            config.axis_prompts = prompt_set.axes
        config.positive_prompts = _coerce_list(mapping.get("positive_prompts"), config.positive_prompts)
        config.negative_prompts = _coerce_list(mapping.get("negative_prompts"), config.negative_prompts)
        axis_mapping = mapping.get("axis_prompts")
        if isinstance(axis_mapping, Mapping):
            parsed_axes: dict[str, AxisConfig] = {}
            for axis_name, raw_axis in axis_mapping.items():
                if not isinstance(raw_axis, Mapping):
                    continue
                positive = _coerce_prompt_specs(raw_axis.get("positive"), [])
                negative = _coerce_prompt_specs(raw_axis.get("negative"), [])
                if positive and negative:
                    aggregation = raw_axis.get("aggregation")
                    parsed_axes[str(axis_name)] = AxisConfig(
                        positive=positive,
                        negative=negative,
                        aggregation=aggregation if aggregation in {"weighted_mean", "max"} else "weighted_mean",
                    )
            if parsed_axes:
                config.axis_prompts.update(parsed_axes)
        # Bounds mirror the Settings dialog's validation, which this path bypasses.
        config.batch_size = _coerce_int(mapping.get("batch_size"), config.batch_size, minimum=1, maximum=1024)
        config.threshold = _coerce_float(mapping.get("threshold"), config.threshold, 0.0, 1.0)
        config.zero_shot_scale = _coerce_float(mapping.get("zero_shot_scale"), config.zero_shot_scale, 0.01, 1000.0)
        config.classifier_weight = _coerce_float(mapping.get("classifier_weight"), config.classifier_weight, 0.0, 100.0)
        config.zero_shot_weight = _coerce_float(mapping.get("zero_shot_weight"), config.zero_shot_weight, 0.0, 100.0)
        config.nsfw_filter = _coerce_choice(
            mapping.get("nsfw_filter"), config.nsfw_filter, {"include", "exclude", "only"}
        )
        config.nsfw_threshold = _coerce_float(mapping.get("nsfw_threshold"), config.nsfw_threshold, 0.0, 1.0)
        config.require_person = _coerce_bool(mapping.get("require_person"), config.require_person)
        config.person_threshold = _coerce_float(mapping.get("person_threshold"), config.person_threshold, 0.0, 1.0)
        config.enable_face_detection = _coerce_bool(mapping.get("enable_face_detection"), config.enable_face_detection)
        config.pipeline = _coerce_choice(mapping.get("pipeline"), config.pipeline, {"cascade", "legacy"})
        config.deep_scan = _coerce_choice(mapping.get("deep_scan"), config.deep_scan, {"candidates", "always", "off"})
        config.person_gate_threshold = _coerce_float(
            mapping.get("person_gate_threshold"), config.person_gate_threshold, 0.0, 1.0
        )
        config.require_female = _coerce_bool(mapping.get("require_female"), config.require_female)
        config.female_threshold = _coerce_float(mapping.get("female_threshold"), config.female_threshold, 0.0, 1.0)
        config.exclude_minors = _coerce_bool(mapping.get("exclude_minors"), config.exclude_minors)
        # Floored above zero on purpose: at 0 the age gate matches essentially every
        # image, which looks like the scanner finding nothing rather than a bad setting.
        config.minor_threshold = _coerce_float(mapping.get("minor_threshold"), config.minor_threshold, 0.01, 1.0)
        config.min_adult_confidence = _coerce_float(
            mapping.get("min_adult_confidence"), config.min_adult_confidence, 0.0, 1.0
        )
        config.max_faces = _coerce_int(mapping.get("max_faces"), config.max_faces, minimum=1, maximum=32)
        # Age-gate constants: minimums prevent a profile from weakening the safety gate
        # to the point of uselessness. Upper bound is 1.0 for all (evidence space).
        config.child_adult_margin = _coerce_float(
            mapping.get("child_adult_margin"), config.child_adult_margin, 0.05, 1.0
        )
        config.strongly_minor_threshold = _coerce_float(
            mapping.get("strongly_minor_threshold"), config.strongly_minor_threshold, 0.50, 1.0
        )
        config.face_anchored_margin = _coerce_float(
            mapping.get("face_anchored_margin"), config.face_anchored_margin, 0.02, 1.0
        )
        config.weak_adult_detail = _coerce_float(mapping.get("weak_adult_detail"), config.weak_adult_detail, 0.20, 1.0)
        config.refine_model = _coerce_str(mapping.get("refine_model"), config.refine_model)
        config.refine_band = _coerce_float(mapping.get("refine_band"), config.refine_band, 0.0, 1.0)
        config.refine_max_images = _coerce_int(
            mapping.get("refine_max_images"), config.refine_max_images, minimum=0, maximum=1_000_000
        )
        config.refine_weight = _coerce_float(mapping.get("refine_weight"), config.refine_weight, 0.0, 1.0)
        config.vlm_enabled = _coerce_bool(mapping.get("vlm_enabled"), config.vlm_enabled)
        config.vlm_base_url = _coerce_str(mapping.get("vlm_base_url"), config.vlm_base_url)
        config.vlm_model = _coerce_str(mapping.get("vlm_model"), config.vlm_model)
        config.vlm_api_key = _coerce_str(mapping.get("vlm_api_key"), config.vlm_api_key)
        config.vlm_band = _coerce_float(mapping.get("vlm_band"), config.vlm_band, 0.0, 1.0)
        config.vlm_max_images = _coerce_int(mapping.get("vlm_max_images"), config.vlm_max_images, 0, 1_000_000)
        config.vlm_concurrency = _coerce_int(mapping.get("vlm_concurrency"), config.vlm_concurrency, 1, 64)
        config.vlm_timeout = _coerce_float(mapping.get("vlm_timeout"), config.vlm_timeout, 1.0, 600.0)
        config.vlm_weight = _coerce_float(mapping.get("vlm_weight"), config.vlm_weight, 0.0, 1.0)
        config.enable_plugins = _coerce_bool(mapping.get("enable_plugins"), config.enable_plugins)
        config.global_learning = _coerce_bool(mapping.get("global_learning"), config.global_learning)
        config.max_learning_weight = _coerce_float(
            mapping.get("max_learning_weight"), config.max_learning_weight, 0.0, 0.95
        )
        config.detail_strongest_weight = _coerce_float(
            mapping.get("detail_strongest_weight"), config.detail_strongest_weight, 0.0, 1.0
        )
        config.detail_average_weight = _coerce_float(
            mapping.get("detail_average_weight"), config.detail_average_weight, 0.0, 1.0
        )
        config.detail_weights = _coerce_weights(mapping.get("detail_weights"), config.detail_weights)
        return config


def _coerce_str(value: Any, default: str) -> str:
    return default if not isinstance(value, str) or not value else value


def _coerce_choice(value: Any, default: str, choices: set[str]) -> str:
    if not isinstance(value, str):
        return default
    value = value.strip()
    return value if value in choices else default


def _coerce_list(value: Any, default: list[str]) -> list[str]:
    if not isinstance(value, list):
        return list(default)
    items = [item for item in value if isinstance(item, str) and item]
    return items or list(default)


def _coerce_prompt_specs(value: Any, default: list[PromptSpec]) -> list[PromptSpec]:
    if not isinstance(value, list):
        return list(default)
    result: list[PromptSpec] = []
    for item in value:
        if isinstance(item, str) and item:
            result.append(item)
        elif isinstance(item, (list, tuple)) and len(item) == 2 and isinstance(item[0], str) and item[0]:
            try:
                result.append((item[0], float(item[1])))
            except (TypeError, ValueError):
                continue
    return result


def _coerce_weights(value: Any, default: dict[str, float]) -> dict[str, float]:
    """Detail-axis weights.

    A supplied mapping replaces the defaults outright rather than merging into them:
    merging made it impossible to drop an axis from the score, because the default
    weight came straight back.
    """
    if not isinstance(value, Mapping):
        return dict(default)
    weights: dict[str, float] = {}
    for name, weight in value.items():
        try:
            parsed = float(weight)
        except (TypeError, ValueError):
            continue
        if parsed >= 0:
            weights[str(name)] = parsed
    return weights or dict(default)


def _coerce_int(value: Any, default: int, minimum: int = 0, maximum: int | None = None) -> int:
    try:
        parsed = int(value)
    except Exception:  # noqa: BLE001
        return default
    if parsed < minimum or (maximum is not None and parsed > maximum):
        LOGGER.warning("Ignoring out-of-range setting %r; using %r", parsed, default)
        return default
    return parsed


def _coerce_float(value: Any, default: float, minimum: float | None = None, maximum: float | None = None) -> float:
    """Parse a float setting, rejecting values the rest of the app cannot cope with.

    The Settings dialog range-checks every one of these, but from_mapping is the funnel
    for input the dialog never sees — imported settings files, saved profiles and
    per-folder overrides. Without the same bounds here a hand-edited or corrupted file
    could set a NaN threshold (nothing ever matches, silently), a zero prompt scale
    (every axis flattens to "no opinion"), or a zero minor_threshold (the age gate
    excludes nearly everything) with no indication anything was wrong.
    """
    try:
        parsed = float(value)
    except Exception:  # noqa: BLE001
        return default
    if not math.isfinite(parsed):
        LOGGER.warning("Ignoring non-finite setting %r; using %r", value, default)
        return default
    clamped = parsed
    if minimum is not None:
        clamped = max(clamped, minimum)
    if maximum is not None:
        clamped = min(clamped, maximum)
    if clamped != parsed:
        LOGGER.warning("Clamped out-of-range setting %r to %r", parsed, clamped)
    return clamped


def _coerce_bool(value: Any, default: bool) -> bool:
    if isinstance(value, bool):
        return value
    return default


# --- Folder-scoped override policy ------------------------------------------
# A per-folder `config_override.json` lives *inside the folder being scanned*, so it is
# only as trustworthy as that folder. Scanning a downloaded or shared folder must not
# let the folder reconfigure the scanner. Only keys that tune local ranking and
# performance may come from one; everything with a blast radius beyond this machine or
# beyond ranking is refused and the global setting is kept.
#
# Deliberately excluded, and why:
#   vlm_*            - can redirect every scanned image to an arbitrary remote endpoint
#   enable_plugins   - executes arbitrary Python from the plugins directory
#   backend,
#   model_name,
#   refine_model     - names a model to fetch and load from a remote hub
#   pipeline         - "legacy" skips the cascade, and with it the age gate
#   exclude_minors,
#   minor_threshold,
#   min_adult_confidence,
#   child_adult_margin,
#   strongly_minor_threshold,
#   face_anchored_margin,
#   weak_adult_detail - the age gate itself
#   prompt_set       - swaps the age gate's prompts wholesale
#   axis_prompts     - can redefine the "child"/"adult" axes and hollow out the age gate
#   global_learning,
#   max_learning_weight - writes to cross-folder state shared with every other folder
FOLDER_OVERRIDE_ALLOWED_KEYS: frozenset[str] = frozenset(
    {
        "device",
        "precision",
        "quantize_cpu",
        "preload_backend",
        "positive_prompts",
        "negative_prompts",
        "batch_size",
        "threshold",
        "zero_shot_scale",
        "classifier_weight",
        "zero_shot_weight",
        "nsfw_filter",
        "nsfw_threshold",
        "require_person",
        "person_threshold",
        "enable_face_detection",
        "deep_scan",
        "person_gate_threshold",
        "require_female",
        "female_threshold",
        "max_faces",
        "refine_band",
        "refine_max_images",
        "refine_weight",
        "detail_strongest_weight",
        "detail_average_weight",
        "detail_weights",
    }
)


def filter_folder_override(mapping: Mapping[str, Any] | None) -> tuple[dict[str, Any], list[str]]:
    """Split a folder override into the keys it may set and the keys it may not.

    Returns (accepted, refused_key_names). Refused keys are reported rather than
    dropped quietly, so a folder that tries to reconfigure the scanner is visible to
    the user instead of silently effective.
    """
    if not mapping:
        return {}, []
    accepted: dict[str, Any] = {}
    refused: list[str] = []
    for key, value in mapping.items():
        if str(key) in FOLDER_OVERRIDE_ALLOWED_KEYS:
            accepted[str(key)] = value
        else:
            refused.append(str(key))
    return accepted, sorted(refused)
