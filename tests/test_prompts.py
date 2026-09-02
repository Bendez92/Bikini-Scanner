from __future__ import annotations

import json
from pathlib import Path

import pytest

import bikini_scanner.prompts as prompts
from bikini_scanner.config import ScannerConfig, filter_folder_override
from bikini_scanner.prompts import (
    REQUIRED_AXES,
    PromptSetError,
    available_prompt_sets,
    load_prompt_set,
)


def _default_document() -> dict[str, object]:
    path = Path(__file__).parents[1] / "bikini_scanner" / "data" / "prompts" / "default.json"
    return json.loads(path.read_text(encoding="utf-8"))


def _use_temp_sets(monkeypatch: pytest.MonkeyPatch, tmp_path: Path, *documents: tuple[str, object]) -> None:
    prompt_dir = tmp_path / "data" / "prompts"
    prompt_dir.mkdir(parents=True)
    (prompt_dir / "default.json").write_text(json.dumps(_default_document()), encoding="utf-8")
    for name, document in documents:
        (prompt_dir / f"{name}.json").write_text(json.dumps(document), encoding="utf-8")
    monkeypatch.setattr(prompts, "files", lambda _: tmp_path)
    prompts._load_prompt_set.cache_clear()


def test_default_prompt_set_preserves_accuracy_model_literals() -> None:
    prompt_set = load_prompt_set()

    assert prompt_set.axes["bikini"].positive == [
        "a person wearing a bikini",
        "a woman in a bikini top and bikini bottom",
        "a person with a bare midriff and exposed stomach",
        "a person in swimwear at the beach or pool",
    ]
    assert prompt_set.axes["bikini"].aggregation == "max"
    assert prompt_set.axes["child"].positive == [
        "a photo of a young child",
        "a photo of a little kid",
        "a photo of a toddler",
        "a photo of a baby",
        "a photo of a preteen child",
        "a primary school child",
    ]


def test_default_prompt_set_has_every_required_axis() -> None:
    assert set(load_prompt_set().axes) >= REQUIRED_AXES
    assert available_prompt_sets() == ("default",)


def test_weighted_prompt_is_parsed_in_packaged_loader(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    document = _default_document()
    document["name"] = "weighted"
    axes = document["axes"]
    assert isinstance(axes, dict)
    axes["bikini"]["positive"][0] = ["weighted prompt", 0.5]
    _use_temp_sets(monkeypatch, tmp_path, ("weighted", document))

    assert load_prompt_set("weighted").axes["bikini"].positive[0] == ("weighted prompt", 0.5)


def test_missing_required_axis_raises(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    document = _default_document()
    axes = document["axes"]
    assert isinstance(axes, dict)
    del axes["child"]
    _use_temp_sets(monkeypatch, tmp_path, ("missing-child", document))

    with pytest.raises(PromptSetError, match="child"):
        load_prompt_set("missing-child")


def test_unknown_prompt_set_raises() -> None:
    with pytest.raises(PromptSetError, match="Unknown prompt set"):
        load_prompt_set("does-not-exist")


def test_malformed_weight_raises(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    document = _default_document()
    document["name"] = "bad-weight"
    axes = document["axes"]
    assert isinstance(axes, dict)
    axes["bikini"]["positive"][0] = ["weighted prompt", "not-a-number"]
    _use_temp_sets(monkeypatch, tmp_path, ("bad-weight", document))

    with pytest.raises(PromptSetError, match="weight must be numeric"):
        load_prompt_set("bad-weight")


def test_loaded_prompt_data_is_independent() -> None:
    config = ScannerConfig()
    config.axis_prompts["bikini"].positive.append("mutated")

    fresh = load_prompt_set()
    assert "mutated" not in fresh.axes["bikini"].positive


def test_invalid_prompt_set_falls_back_with_warning(caplog: pytest.LogCaptureFixture) -> None:
    with caplog.at_level("WARNING"):
        config = ScannerConfig.from_mapping({"prompt_set": "nonsense"})

    assert config.prompt_set == "default"
    assert "nonsense" in caplog.text


def test_explicit_axis_prompts_override_named_set(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    document = _default_document()
    document["name"] = "alternate"
    axes = document["axes"]
    assert isinstance(axes, dict)
    axes["child"]["positive"] = ["alternate child"]
    _use_temp_sets(monkeypatch, tmp_path, ("alternate", document))

    config = ScannerConfig.from_mapping(
        {
            "prompt_set": "alternate",
            "axis_prompts": {"child": {"positive": ["explicit child"], "negative": ["explicit negative"]}},
        }
    )

    assert config.prompt_set == "alternate"
    assert config.axis_prompts["child"].positive == ["explicit child"]


def test_prompt_set_is_not_folder_overridable() -> None:
    accepted, refused = filter_folder_override({"prompt_set": "alternate"})

    assert accepted == {}
    assert refused == ["prompt_set"]
