"""Headless regression tests for scalar configuration metadata."""

from __future__ import annotations

import os
import tempfile
from dataclasses import fields
from pathlib import Path

_STATE_DIR = Path(tempfile.mkdtemp(prefix="bikini_schema_state_"))
os.environ["BIKINI_SCANNER_TEST_SQLITE_PRAGMAS"] = "1"
os.environ["APPDATA"] = str(_STATE_DIR)
os.environ["LOCALAPPDATA"] = str(_STATE_DIR)
os.environ["XDG_CONFIG_HOME"] = str(_STATE_DIR)
os.environ["XDG_STATE_HOME"] = str(_STATE_DIR)
os.environ["HOME"] = str(_STATE_DIR)

import pytest

from bikini_scanner.config import FOLDER_OVERRIDE_ALLOWED_KEYS, ScannerConfig, filter_folder_override
from bikini_scanner.config_schema import (
    FIELD_SPECS,
    SPECS_BY_KEY,
    FieldError,
    coerce_choice,
    coerce_float,
    coerce_int,
    parse_choice_entry,
    parse_float_entry,
    parse_int_entry,
)

NON_SCALAR_FIELDS = {"positive_prompts", "negative_prompts", "axis_prompts", "detail_weights"}
# Current allow-list, spelled out so a change to the trust boundary has to be deliberate.
EXPECTED_FOLDER_KEYS = {
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
NON_OVERRIDABLE_KEYS = {
    "backend",
    "model_name",
    "refine_model",
    "pipeline",
    "vlm_enabled",
    "vlm_base_url",
    "vlm_model",
    "vlm_api_key",
    "vlm_band",
    "vlm_max_images",
    "vlm_concurrency",
    "vlm_timeout",
    "vlm_weight",
    "enable_plugins",
    "exclude_minors",
    "minor_threshold",
    "min_adult_confidence",
    "child_adult_margin",
    "strongly_minor_threshold",
    "face_anchored_margin",
    "weak_adult_detail",
    "axis_prompts",
    "global_learning",
    "max_learning_weight",
}


def test_schema_covers_exactly_scalar_config_fields() -> None:
    scalar_fields = {item.name for item in fields(ScannerConfig)} - NON_SCALAR_FIELDS
    assert set(SPECS_BY_KEY) == scalar_fields
    assert len(FIELD_SPECS) == len(scalar_fields)


def test_folder_override_boundary_is_exact() -> None:
    assert FOLDER_OVERRIDE_ALLOWED_KEYS == EXPECTED_FOLDER_KEYS
    assert not FOLDER_OVERRIDE_ALLOWED_KEYS & NON_OVERRIDABLE_KEYS


def test_age_gate_bounds_are_literal_safety_floors() -> None:
    assert SPECS_BY_KEY["minor_threshold"].minimum == 0.01
    assert SPECS_BY_KEY["child_adult_margin"].minimum == 0.05
    assert SPECS_BY_KEY["strongly_minor_threshold"].minimum == 0.50
    assert SPECS_BY_KEY["face_anchored_margin"].minimum == 0.02
    assert SPECS_BY_KEY["weak_adult_detail"].minimum == 0.20
    assert SPECS_BY_KEY["max_learning_weight"].maximum == 0.95


def test_float_clamps_and_warns(caplog: pytest.LogCaptureFixture) -> None:
    with caplog.at_level("WARNING"):
        assert coerce_float("threshold", 2.0, 0.35) == 1.0
    assert "Clamped out-of-range setting" in caplog.text


def test_int_out_of_range_falls_back_and_warns(caplog: pytest.LogCaptureFixture) -> None:
    with caplog.at_level("WARNING"):
        assert coerce_int("batch_size", 0, 16) == 16
    assert "Ignoring out-of-range setting" in caplog.text


def test_invalid_values_fall_back_and_warn(caplog: pytest.LogCaptureFixture) -> None:
    with caplog.at_level("WARNING"):
        assert coerce_float("threshold", float("nan"), 0.35) == 0.35
        assert coerce_float("threshold", "garbage", 0.35) == 0.35
        assert coerce_int("batch_size", "garbage", 16) == 16
        assert coerce_choice("device", "invalid", "auto") == "auto"
    assert caplog.text.count("Ignoring") == 4


def test_from_mapping_invalid_values_fall_back_and_warn(caplog: pytest.LogCaptureFixture) -> None:
    defaults = ScannerConfig()
    with caplog.at_level("WARNING"):
        config = ScannerConfig.from_mapping({"threshold": "garbage", "deep_scan": "invalid"})
    assert config.threshold == defaults.threshold
    assert config.deep_scan == defaults.deep_scan
    assert caplog.text.count("Ignoring invalid setting") == 2


def test_age_gate_weakening_is_floored() -> None:
    config = ScannerConfig.from_mapping(
        {
            "minor_threshold": 0.0,
            "child_adult_margin": 0.0,
            "strongly_minor_threshold": 0.0,
            "face_anchored_margin": 0.0,
            "weak_adult_detail": 0.0,
            "max_learning_weight": 1.0,
            "pipeline": "nonsense",
        }
    )
    assert config.minor_threshold == 0.01
    assert config.child_adult_margin == 0.05
    assert config.strongly_minor_threshold == 0.50
    assert config.face_anchored_margin == 0.02
    assert config.weak_adult_detail == 0.20
    assert config.max_learning_weight == 0.95
    assert config.pipeline == "cascade"


def test_filter_folder_override_refuses_denied_keys_sorted() -> None:
    accepted, refused = filter_folder_override(dict.fromkeys(NON_OVERRIDABLE_KEYS, 1))
    assert accepted == {}
    assert refused == sorted(NON_OVERRIDABLE_KEYS)


def test_entry_parsers_round_trip_valid_values() -> None:
    assert parse_int_entry("batch_size", " 16 ") == 16
    assert parse_int_entry("vlm_concurrency", "4") == 4
    assert parse_float_entry("threshold", "0.35") == 0.35
    assert parse_float_entry("zero_shot_scale", "40") == 40.0
    assert parse_choice_entry("device", "cuda") == "cuda"
    assert parse_choice_entry("deep_scan", "always") == "always"


def test_entry_parsers_reject_invalid_values() -> None:
    with pytest.raises(FieldError, match="Batch size"):
        parse_int_entry("batch_size", "nope")
    with pytest.raises(FieldError, match="Batch size"):
        parse_int_entry("batch_size", "0")
    with pytest.raises(FieldError, match="Threshold"):
        parse_float_entry("threshold", "1.5")
    with pytest.raises(FieldError, match="VLM concurrency"):
        parse_int_entry("vlm_concurrency", "65")
    with pytest.raises(FieldError, match="Device"):
        parse_choice_entry("device", "nope")
    with pytest.raises(FieldError) as excinfo:
        parse_float_entry("zero_shot_scale", "0")
    assert str(excinfo.value)


def test_minor_threshold_entry_matches_dialog_behaviour() -> None:
    assert parse_float_entry("minor_threshold", "0.005") == 0.005
    with pytest.raises(FieldError, match="Lower is stricter"):
        parse_float_entry("minor_threshold", "0.0")
