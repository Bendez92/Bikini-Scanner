"""Packaged prompt sets are the reviewable record of the app's accuracy model."""

from __future__ import annotations

import copy
import json
import math
from dataclasses import dataclass
from functools import cache
from importlib.resources import files
from typing import TypeAlias

PromptSpec: TypeAlias = str | tuple[str, float]

DEFAULT_PROMPT_SET = "default"
REQUIRED_AXES: frozenset[str] = frozenset(
    {
        "bikini",
        "bikini_top",
        "bikini_bottom",
        "midriff",
        "cleavage",
        "nsfw",
        "person",
        "female",
        "child",
        "adult",
    }
)


@dataclass(slots=True)
class AxisConfig:
    """Prompt ensembles for one cascade axis.

    Each axis is scored as (positive evidence - negative evidence), so the negatives
    matter as much as the positives: they are the confusions this axis must not make.
    More prompts per axis is generally better — CLIP zero-shot accuracy improves
    measurably with ensembling, and the cost is one text embedding computed at startup.
    """

    positive: list[PromptSpec]
    negative: list[PromptSpec]
    aggregation: str = "weighted_mean"


@dataclass(frozen=True, slots=True)
class PromptSet:
    name: str
    version: int
    positive: list[str]
    negative: list[str]
    axes: dict[str, AxisConfig]


class PromptSetError(ValueError):
    """Raised when a packaged prompt set cannot be loaded safely."""


def available_prompt_sets() -> tuple[str, ...]:
    prompt_dir = files("bikini_scanner").joinpath("data", "prompts")
    try:
        return tuple(
            sorted(path.name.removesuffix(".json") for path in prompt_dir.iterdir() if path.name.endswith(".json"))
        )
    except OSError as error:
        raise PromptSetError(f"Could not enumerate packaged prompt sets: {error}") from error


def _parse_prompt_entry(raw: object, location: str) -> PromptSpec:
    if isinstance(raw, str) and raw:
        return raw
    if isinstance(raw, list) and len(raw) == 2 and isinstance(raw[0], str) and raw[0]:
        weight = raw[1]
        if not isinstance(weight, (int, float, str)):
            raise PromptSetError(f"{location}: prompt weight must be numeric")
        try:
            parsed_weight = float(weight)
        except (TypeError, ValueError):
            raise PromptSetError(f"{location}: prompt weight must be numeric") from None
        if not math.isfinite(parsed_weight):
            raise PromptSetError(f"{location}: prompt weight must be finite")
        return raw[0], parsed_weight
    raise PromptSetError(f"{location}: prompt must be a non-empty string or [text, weight]")


def _parse_axis_prompts(raw: object, axis_name: str) -> AxisConfig:
    if not isinstance(raw, dict):
        raise PromptSetError(f"axis {axis_name!r} must be an object")
    positive_raw = raw.get("positive")
    negative_raw = raw.get("negative")
    if not isinstance(positive_raw, list) or not positive_raw:
        raise PromptSetError(f"axis {axis_name!r} must have a non-empty positive list")
    if not isinstance(negative_raw, list) or not negative_raw:
        raise PromptSetError(f"axis {axis_name!r} must have a non-empty negative list")
    positive = [_parse_prompt_entry(item, f"axis {axis_name!r} positive") for item in positive_raw]
    negative = [_parse_prompt_entry(item, f"axis {axis_name!r} negative") for item in negative_raw]
    aggregation = raw.get("aggregation", "weighted_mean")
    if aggregation not in {"weighted_mean", "max"}:
        raise PromptSetError(f"axis {axis_name!r}: aggregation must be 'weighted_mean' or 'max'")
    return AxisConfig(positive=positive, negative=negative, aggregation=aggregation)


def _parse_top_level_prompts(raw: object, key: str) -> list[str]:
    if not isinstance(raw, list) or not raw:
        raise PromptSetError(f"prompt set must have a non-empty {key!r} list")
    parsed = [_parse_prompt_entry(item, key) for item in raw]
    result: list[str] = []
    for item in parsed:
        if not isinstance(item, str):
            raise PromptSetError(f"{key!r} prompts must be strings")
        result.append(item)
    return result


@cache
def _load_prompt_set(name: str) -> PromptSet:
    if name not in available_prompt_sets():
        raise PromptSetError(f"Unknown prompt set {name!r}")
    resource = files("bikini_scanner").joinpath("data", "prompts", f"{name}.json")
    try:
        document = json.loads(resource.read_text(encoding="utf-8"))
    except (OSError, UnicodeDecodeError, json.JSONDecodeError) as error:
        raise PromptSetError(f"Could not load prompt set {name!r}: {error}") from error
    if not isinstance(document, dict):
        raise PromptSetError(f"Prompt set {name!r} must contain a JSON object")
    if document.get("name") != name:
        raise PromptSetError(f"Prompt set {name!r} has mismatched name")
    version = document.get("version")
    if not isinstance(version, int) or isinstance(version, bool) or version < 1:
        raise PromptSetError(f"Prompt set {name!r} must have a positive integer version")
    axes_raw = document.get("axes")
    if not isinstance(axes_raw, dict):
        raise PromptSetError(f"Prompt set {name!r} must have an axes object")
    missing_axes = REQUIRED_AXES - {str(axis) for axis in axes_raw}
    if missing_axes:
        # Missing child/adult prompts could hollow out the age gate while appearing to work.
        raise PromptSetError(f"Prompt set {name!r} is missing required axes: {', '.join(sorted(missing_axes))}")
    axes = {str(axis_name): _parse_axis_prompts(raw_axis, str(axis_name)) for axis_name, raw_axis in axes_raw.items()}
    return PromptSet(
        name=name,
        version=version,
        positive=_parse_top_level_prompts(document.get("positive_prompts"), "positive_prompts"),
        negative=_parse_top_level_prompts(document.get("negative_prompts"), "negative_prompts"),
        axes=axes,
    )


def load_prompt_set(name: str = DEFAULT_PROMPT_SET) -> PromptSet:
    """Load a packaged prompt set and return independent mutable data."""
    return copy.deepcopy(_load_prompt_set(name))


__all__ = [
    "DEFAULT_PROMPT_SET",
    "REQUIRED_AXES",
    "AxisConfig",
    "PromptSet",
    "PromptSetError",
    "PromptSpec",
    "available_prompt_sets",
    "load_prompt_set",
]
