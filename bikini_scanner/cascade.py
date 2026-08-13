"""The staged scan pipeline.

Instead of asking one question of a shrunken whole image ("does this look like a
bikini photo?"), the scan asks a sequence of cheap questions and only spends effort
where the answers keep pointing at a candidate:

    1. people      - is there a person here at all?
    2. sex         - is the subject female?
    3. age         - does anyone here read as a minor?  (exclusion gate)
    4. detail      - bikini / cleavage / midriff, scored on body-region crops

Stages 1-3 are gates. Each one is applied twice: softly, as a multiplier on the final
score, and hard, as an exclusion when it falls below its threshold. The age gate is
the strict one - anything it flags is forced to zero and dropped, never merely ranked
lower.

Axis scores arrive as sigmoids centred on 0.5 (0.5 = "no evidence either way"), so
before combining them into one number they are re-centred into 0..1 *evidence*.
Without that, five neutral 0.5 axes would soft-OR their way to a near-perfect score.
"""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from dataclasses import dataclass, field

import numpy as np

from .config import ScannerConfig
from .regions import (
    KIND_BAND,
    KIND_BAND_LOWER,
    KIND_BAND_MID,
    KIND_BAND_UPPER,
    KIND_CHEST,
    KIND_FACE,
    KIND_FULL,
    KIND_TORSO,
    KIND_WAIST,
    UNANCHORED_KINDS,
)

# Which region kinds may contribute to each axis. Faces do not vote on swimwear, and
# chest/waist bands do not vote on age.
#
# The fallback bands are listed by position, because where a crop sits in the frame
# decides what it can be evidence of. Letting every band vote on every axis was the
# single biggest source of false positives: with no face model installed the bottom
# 55% of a photo would "detect cleavage", and one lucky band was enough to carry an
# image over the threshold.
AXIS_REGION_KINDS: dict[str, frozenset[str]] = {
    "person": frozenset({KIND_FULL, KIND_TORSO, KIND_BAND, KIND_BAND_UPPER, KIND_BAND_MID, KIND_BAND_LOWER}),
    "female": frozenset({KIND_FULL, KIND_FACE, KIND_TORSO, KIND_BAND, KIND_BAND_UPPER, KIND_BAND_MID}),
    "child": frozenset({KIND_FACE, KIND_FULL}),
    "adult": frozenset({KIND_FACE, KIND_FULL}),
    "bikini": frozenset(
        {KIND_FULL, KIND_TORSO, KIND_CHEST, KIND_WAIST, KIND_BAND, KIND_BAND_UPPER, KIND_BAND_MID, KIND_BAND_LOWER}
    ),
    "bikini_top": frozenset({KIND_FULL, KIND_TORSO, KIND_CHEST, KIND_BAND, KIND_BAND_UPPER, KIND_BAND_MID}),
    "bikini_bottom": frozenset({KIND_FULL, KIND_TORSO, KIND_WAIST, KIND_BAND, KIND_BAND_MID, KIND_BAND_LOWER}),
    # Cleavage sits above the waist, so the lower band is not allowed to claim it.
    "cleavage": frozenset({KIND_FULL, KIND_CHEST, KIND_TORSO, KIND_BAND, KIND_BAND_UPPER, KIND_BAND_MID}),
    "midriff": frozenset({KIND_FULL, KIND_WAIST, KIND_TORSO, KIND_BAND, KIND_BAND_MID, KIND_BAND_LOWER}),
    "nsfw": frozenset(
        {KIND_FULL, KIND_TORSO, KIND_CHEST, KIND_WAIST, KIND_BAND, KIND_BAND_UPPER, KIND_BAND_MID, KIND_BAND_LOWER}
    ),
}

DETAIL_AXES = ("bikini", "cleavage", "midriff", "bikini_top", "bikini_bottom")

# How much of an unanchored band's excess over the full frame counts. A face-anchored
# chest crop is where the geometry says it is and gets a full vote; a band is a guess,
# and taking a plain max over four guesses inflates every image. Measured on a real
# folder with no face model: images above the threshold fell from 12 to 7 while every
# accepted image stayed above it.
UNANCHORED_CROP_SHARE = 0.5

STAGE_SCORED = "scored"
STAGE_NO_PERSON = "no_person"
STAGE_NOT_FEMALE = "not_female"
STAGE_MINOR = "minor"

STAGE_REASONS = {
    STAGE_SCORED: "",
    STAGE_NO_PERSON: "no person detected",
    STAGE_NOT_FEMALE: "subject does not read as female",
    STAGE_MINOR: "excluded: subject may be a minor",
}


def evidence(scores: np.ndarray) -> np.ndarray:
    """Re-centre a sigmoid axis score into 0..1 evidence (0.5 and below means none)."""
    return np.clip((np.asarray(scores, dtype=np.float32) - 0.5) * 2.0, 0.0, 1.0)


def _ramp(scores: np.ndarray, low: float, high: float) -> np.ndarray:
    """Smooth 0..1 ramp used for soft gating; below `low` is 0, above `high` is 1."""
    if high <= low:
        high = low + 1e-3
    return np.clip((np.asarray(scores, dtype=np.float32) - low) / (high - low), 0.0, 1.0)


@dataclass(slots=True)
class RegionScoreTable:
    """Per-region axis scores for a whole batch, flattened.

    Regions vary per image, so rows are (image, region) pairs and `owner` says which
    image each row belongs to. Row 0 of each image is always its full frame.
    """

    owner: np.ndarray
    kinds: np.ndarray
    axis_scores: dict[str, np.ndarray]
    image_count: int
    full_row: np.ndarray = field(default_factory=lambda: np.empty((0,), dtype=np.int64))

    def aggregate(self, axis: str) -> np.ndarray:
        """Best evidence for one axis per image, over that axis's eligible regions.

        Anchored crops win outright when they beat the full frame. Unanchored bands only
        get `UNANCHORED_CROP_SHARE` of the distance they claim above the full frame, so a
        single lucky slice of a photo can nudge the score without deciding it.
        """
        scores = self.axis_scores.get(axis)
        if scores is None or self.image_count == 0:
            return np.zeros((self.image_count,), dtype=np.float32)
        # Always seed with the full-frame score so every image has a value even when no
        # eligible crop exists.
        full = np.zeros((self.image_count,), dtype=np.float32)
        if self.full_row.size:
            full[:] = scores[self.full_row]
        eligible = AXIS_REGION_KINDS.get(axis)
        if eligible is None:
            return full.astype(np.float32)

        anchored = full.copy()
        unanchored = full.copy()
        anchored_mask = np.array([kind in eligible and kind not in UNANCHORED_KINDS for kind in self.kinds], dtype=bool)
        unanchored_mask = np.array([kind in eligible and kind in UNANCHORED_KINDS for kind in self.kinds], dtype=bool)
        if anchored_mask.any():
            np.maximum.at(anchored, self.owner[anchored_mask], scores[anchored_mask])
        if unanchored_mask.any():
            np.maximum.at(unanchored, self.owner[unanchored_mask], scores[unanchored_mask])
        discounted = full + UNANCHORED_CROP_SHARE * np.maximum(unanchored - full, 0.0)
        return np.maximum(anchored, discounted).astype(np.float32)


@dataclass(slots=True)
class CascadeResult:
    stage: list[str]
    reason: list[str]
    excluded: np.ndarray
    gate_factor: np.ndarray
    detail: np.ndarray
    score: np.ndarray
    axis_scores: dict[str, np.ndarray]

    def stage_counts(self) -> dict[str, int]:
        counts: dict[str, int] = {}
        for stage in self.stage:
            counts[stage] = counts.get(stage, 0) + 1
        return counts


def combine_detail(
    axis_scores: Mapping[str, np.ndarray],
    weights: Mapping[str, float],
    axes: Sequence[str] = DETAIL_AXES,
    strongest_weight: float = 0.65,
    average_weight: float = 0.35,
) -> np.ndarray:
    """Combine the detail axes into one score.

    Mostly the strongest single axis, plus a share of the average so that corroborating
    signals count. Measured against real labels this beat both a plain max and a soft
    OR: the max alone throws away agreement between axes, and the soft OR saturates -
    once several axes are confident everything ties at 1.0 and the ranking collapses.

    A corroboration floor prevents a single weak axis from carrying an image to a
    match: if fewer than 2 axes show evidence above 0.15 AND the strongest is below
    0.55, the score is dampened by half. This stops the crop-top false positive
    (midriff fires alone at ~0.4, no other axis corroborates) without affecting
    genuine matches where multiple axes agree.
    """
    columns: list[np.ndarray] = []
    for axis in axes:
        scores = axis_scores.get(axis)
        if scores is None:
            continue
        weight = float(weights.get(axis, 0.0))
        if weight <= 0:
            continue
        columns.append(np.clip(evidence(scores) * min(weight, 1.0), 0.0, 1.0))
    if not columns:
        return np.zeros((0,), dtype=np.float32)
    stack = np.vstack(columns)
    strongest = stack.max(axis=0)
    average = stack.mean(axis=0)
    raw = (strongest_weight * strongest + average_weight * average).astype(np.float32)
    # Corroboration floor: dampen images where only one axis fires weakly.
    corroborated = (stack >= 0.15).sum(axis=0)
    uncorroborated_weak = (corroborated < 2) & (strongest < 0.55)
    raw = np.where(uncorroborated_weak, raw * 0.5, raw)
    return raw.astype(np.float32)


def combine_detail_rows(
    axis_scores: Mapping[str, np.ndarray],
    weights: Mapping[str, float],
    kinds: Sequence[str],
) -> np.ndarray:
    """Per-region detail score, used to pick the crop that best represents an image.

    Same combination as `combine_detail`, but each region only votes on the axes its
    position allows — otherwise the crop chosen to represent an image (and to train the
    learned model) can be a bottom-of-frame band that scored highly on cleavage.
    """
    masked: dict[str, np.ndarray] = {}
    for axis, scores in axis_scores.items():
        eligible = AXIS_REGION_KINDS.get(axis)
        if eligible is None:
            masked[axis] = np.asarray(scores, dtype=np.float32)
            continue
        allowed = np.array([kind in eligible for kind in kinds], dtype=bool)
        # 0.5 is the neutral point of the sigmoid axes, i.e. "no evidence either way".
        masked[axis] = np.where(allowed, np.asarray(scores, dtype=np.float32), 0.5).astype(np.float32)
    return combine_detail(masked, weights)


def evaluate(
    table: RegionScoreTable,
    config: ScannerConfig,
    face_counts: np.ndarray | None = None,
) -> CascadeResult:
    """Run the gates and produce a final zero-shot score per image."""
    count = table.image_count
    if count == 0:
        return CascadeResult(
            stage=[],
            reason=[],
            excluded=np.empty((0,), dtype=bool),
            gate_factor=np.empty((0,), dtype=np.float32),
            detail=np.empty((0,), dtype=np.float32),
            score=np.empty((0,), dtype=np.float32),
            axis_scores={},
        )

    aggregated = {axis: table.aggregate(axis) for axis in table.axis_scores}
    # Every gate compares *evidence*, not the raw sigmoid: a raw 0.5 means the axis saw
    # nothing either way, so thresholds applied to raw scores would fire on every image.
    person = evidence(aggregated.get("person", np.full((count,), 0.5, dtype=np.float32)))
    female = evidence(aggregated.get("female", np.full((count,), 0.5, dtype=np.float32)))
    child = evidence(aggregated.get("child", np.full((count,), 0.5, dtype=np.float32)))
    adult = evidence(aggregated.get("adult", np.full((count,), 0.5, dtype=np.float32)))

    # A detected face is direct evidence of a person that does not depend on the prompt.
    has_face = np.zeros((count,), dtype=bool)
    if face_counts is not None and len(face_counts) == count:
        has_face = np.asarray(face_counts, dtype=np.int32) > 0

    # The person stage decides where to spend the crop pass, but it does NOT exclude by
    # default. Measured on real photos, "a photo of a person" scores near zero on exactly
    # the close-up torso shots this tool is looking for, so gating on it hurt ranking
    # badly (AUC 0.39 gated vs 0.93 ungated). Users who want the old hard filter can
    # still switch require_person on.
    if config.require_person:
        person_pass = (person >= float(config.person_gate_threshold)) | has_face
    else:
        person_pass = np.ones((count,), dtype=bool)
    female_pass = (
        np.ones((count,), dtype=bool) if not config.require_female else female >= float(config.female_threshold)
    )

    detail = combine_detail(
        aggregated,
        config.detail_weights,
        strongest_weight=float(config.detail_strongest_weight),
        average_weight=float(config.detail_average_weight),
    )
    if detail.size == 0:
        detail = np.zeros((count,), dtype=np.float32)

    # Three ways to fail the age gate, in decreasing order of how much evidence they
    # need. The last one only binds on images that would otherwise be surfaced as
    # matches, which is where being wrong actually costs something.
    if config.exclude_minors:
        threshold = float(config.minor_threshold)
        # Absolute child evidence that also out-argues the adult reading.
        looks_minor = (child >= threshold) & (child > adult + float(config.child_adult_margin))
        # Overwhelming on its own, whatever the adult axis says.
        strongly_minor = child >= float(config.strongly_minor_threshold)
        # With a real face crop the age read is far more reliable, so a smaller margin
        # is enough to act on.
        reads_younger = has_face & (child > adult + float(config.face_anchored_margin)) & (child >= threshold * 0.4)
        # Positive adult evidence is required before surfacing anything as a match.
        weak_adult = (
            has_face & (detail >= float(config.weak_adult_detail)) & (adult < float(config.min_adult_confidence))
        )
        age_fail = looks_minor | strongly_minor | reads_younger | weak_adult
    else:
        age_fail = np.zeros((count,), dtype=bool)

    if config.require_person:
        person_conf = _ramp(person, float(config.person_gate_threshold), float(config.person_gate_threshold) + 0.2)
        person_conf = np.maximum(person_conf, has_face.astype(np.float32) * 0.85)
    else:
        person_conf = np.ones((count,), dtype=np.float32)
    if config.require_female:
        female_conf = _ramp(
            female, max(0.0, float(config.female_threshold) - 0.05), float(config.female_threshold) + 0.35
        )
        # Never zero out on the sex axis alone: it is the least reliable read on a crop
        # that does not include a face.
        female_conf = np.maximum(female_conf, 0.25)
    else:
        female_conf = np.ones((count,), dtype=np.float32)
    # Age confidence falls off before the hard threshold, so a borderline image is
    # already ranked down by the time it is excluded outright.
    age_conf = 1.0 - _ramp(child, float(config.minor_threshold) * 0.6, float(config.minor_threshold))

    gate_factor = (person_conf * female_conf * age_conf).astype(np.float32)
    score = (detail * gate_factor).astype(np.float32)

    stage: list[str] = []
    reason: list[str] = []
    excluded = np.zeros((count,), dtype=bool)
    for index in range(count):
        if age_fail[index]:
            current = STAGE_MINOR
        elif not person_pass[index]:
            current = STAGE_NO_PERSON
        elif not female_pass[index]:
            current = STAGE_NOT_FEMALE
        else:
            current = STAGE_SCORED
        stage.append(current)
        reason.append(STAGE_REASONS[current])
        excluded[index] = current != STAGE_SCORED

    # Anything the age gate flags is forced to zero, not merely ranked down, so no
    # threshold or filter setting can surface it later.
    score[age_fail] = 0.0

    aggregated["gate_person"] = person_conf
    aggregated["gate_female"] = female_conf
    aggregated["gate_age"] = age_conf
    aggregated["detail"] = detail
    # Evidence-space copies, which is what the gates and the UI thresholds talk about.
    aggregated["evidence_person"] = person
    aggregated["evidence_female"] = female
    aggregated["evidence_child"] = child
    aggregated["evidence_adult"] = adult
    return CascadeResult(
        stage=stage,
        reason=reason,
        excluded=excluded,
        gate_factor=gate_factor,
        detail=detail,
        score=score,
        axis_scores=aggregated,
    )
