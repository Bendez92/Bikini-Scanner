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
from typing import cast

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

# The crops that show one detected person's body. Their age comes from their face
# crop; these are what their swimwear evidence is read from.
SUBJECT_BODY_KINDS = frozenset({KIND_CHEST, KIND_WAIST, KIND_TORSO})

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
    # Which detected person each row belongs to (-1 for the full frame and the
    # unanchored bands). Empty for tables built before subjects existed, which
    # `subject_ids` reads as "nothing is attributed".
    subject: np.ndarray = field(default_factory=lambda: np.empty((0,), dtype=np.int64))
    _axis_masks: dict[str, tuple[np.ndarray, np.ndarray]] | None = field(
        init=False, default=None, repr=False
    )

    def __post_init__(self) -> None:
        """`owner`, `kinds`, `subject` and every axis are indexed by row, together.

        `aggregate` scatters scores through `owner` and masks them through `kinds`, so a
        length mismatch does not raise - it reads the wrong rows, or silently ignores
        the tail. Checking at construction is what makes that a reportable error rather
        than a quietly wrong score.
        """
        rows = int(self.owner.shape[0]) if self.owner.ndim else 0
        problems: list[str] = []
        if len(self.kinds) != rows:
            problems.append(f"kinds has {len(self.kinds)} entries, expected {rows}")
        # subject is empty for tables built before per-person attribution existed.
        if self.subject.size and int(self.subject.shape[0]) != rows:
            problems.append(f"subject has {int(self.subject.shape[0])} entries, expected {rows}")
        if self.full_row.size and int(self.full_row.shape[0]) != int(self.image_count):
            problems.append(
                f"full_row has {int(self.full_row.shape[0])} entries, expected one per image "
                f"({int(self.image_count)})"
            )
        for axis, values in self.axis_scores.items():
            found = int(np.asarray(values).shape[0]) if np.asarray(values).ndim else 0
            if found != rows:
                problems.append(f"axis_scores[{axis!r}] has {found} rows, expected {rows}")
        if problems:
            raise ValueError(f"RegionScoreTable is inconsistent for {rows} row(s): " + "; ".join(problems))

    def subject_ids(self) -> np.ndarray:
        """Per-row subject index, or all -1 when this table has no attribution."""
        if self.subject.size == self.owner.size:
            return np.asarray(self.subject, dtype=np.int64)
        return np.full((self.owner.size,), -1, dtype=np.int64)

    def _ensure_masks(self) -> dict[str, tuple[np.ndarray, np.ndarray]]:
        if self._axis_masks is None:
            self._axis_masks = {}
            kinds = np.asarray(self.kinds)
            for axis in self.axis_scores:
                eligible = AXIS_REGION_KINDS.get(axis)
                if eligible is None:
                    self._axis_masks[axis] = (
                        np.zeros(len(kinds), dtype=bool),
                        np.zeros(len(kinds), dtype=bool),
                    )
                    continue
                allowed = np.isin(kinds, list(eligible))
                unanchored = np.isin(kinds, list(UNANCHORED_KINDS))
                self._axis_masks[axis] = (
                    allowed & ~unanchored,
                    allowed & unanchored,
                )
        return self._axis_masks

    def aggregate(
        self,
        axis: str,
        row_filter: np.ndarray | None = None,
        full_allowed: np.ndarray | None = None,
    ) -> np.ndarray:
        """Best evidence for one axis per image, over that axis's eligible regions.

        Anchored crops win outright when they beat the full frame. Unanchored bands only
        get `UNANCHORED_CROP_SHARE` of the distance they claim above the full frame, so a
        single lucky slice of a photo can nudge the score without deciding it.

        `row_filter` drops individual rows from consideration and `full_allowed` drops
        the full-frame row for individual images. Both exist for the age gate: once a
        detected person reads as a minor, none of their crops — and not the whole frame
        that contains them — may contribute evidence to the image's score.
        """
        scores = self.axis_scores.get(axis)
        if scores is None or self.image_count == 0:
            return np.zeros((self.image_count,), dtype=np.float32)
        # Seed with the full-frame score so every image has a value even when no
        # eligible crop exists. 0.5 is the sigmoid's neutral point, so an image whose
        # full frame is suppressed starts from "no evidence" rather than from evidence
        # against.
        full: np.ndarray = np.full((self.image_count,), 0.5, dtype=np.float32)
        if self.full_row.size:
            full[:] = scores[self.full_row]
        if full_allowed is not None:
            full = np.where(np.asarray(full_allowed, dtype=bool), full, 0.5).astype(np.float32)
        if AXIS_REGION_KINDS.get(axis) is None:
            return full.astype(np.float32)

        anchored_mask, unanchored_mask = self._ensure_masks()[axis]
        if row_filter is not None:
            keep = np.asarray(row_filter, dtype=bool)
            anchored_mask = anchored_mask & keep
            unanchored_mask = unanchored_mask & keep
        if full_allowed is not None:
            # Several axes list KIND_FULL among their eligible kinds, so the full-frame
            # row also travels the anchored path. Suppressing it in the seed above is
            # not enough - it has to be dropped as a row too, or it comes straight back.
            allowed = np.asarray(full_allowed, dtype=bool)
            suppressed_full = (np.asarray(self.kinds) == KIND_FULL) & ~allowed[self.owner]
            anchored_mask = anchored_mask & ~suppressed_full
            unanchored_mask = unanchored_mask & ~suppressed_full
        anchored = full.copy()
        unanchored = full.copy()
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
    kinds_array = np.asarray(kinds)
    masked: dict[str, np.ndarray] = {}
    for axis, scores in axis_scores.items():
        eligible = AXIS_REGION_KINDS.get(axis)
        if eligible is None:
            masked[axis] = np.asarray(scores, dtype=np.float32)
            continue
        allowed = np.isin(kinds_array, list(eligible))
        # 0.5 is the neutral point of the sigmoid axes, i.e. "no evidence either way".
        masked[axis] = np.where(allowed, np.asarray(scores, dtype=np.float32), 0.5).astype(np.float32)
    return combine_detail(masked, weights)


@dataclass(slots=True)
class SubjectAnalysis:
    """Who is in an image, which of them read as minors, and what that suppresses.

    The age gate used to ask "does this photo contain a minor?", scored on the whole
    frame. That is the wrong question for an ordinary family photo: an adult in swimwear
    standing next to her children was hard-excluded because the frame, taken as a whole,
    read as "children". The evidence for the match sat on her chest crop; the evidence
    that binned it came from two toddlers three feet away.

    With face-anchored crops the right question is answerable. Every crop derived from
    one detected face shares a subject index, so each person's age can be read from
    their own face and each person's swimwear evidence from their own body. Then:

      * a subject who reads as a minor contributes nothing - none of their crops may
        score, and the full frame is suppressed too, because it contains them;
      * an image where *every* detected person reads as a minor is excluded outright;
      * an image with at least one adult subject is scored from the adult subjects
        alone, so a photo only surfaces on the strength of an adult's own crops.

    This is strictly narrower than the old whole-frame rule in what it lets through: a
    minor's body can no longer contribute to any score, where previously a minor's crop
    could win the detail slot outright on an image the whole-frame gate happened to
    clear. It is wider only in that other people's ages no longer condemn the subject.

    Images with no detected faces have no subjects and are not covered here at all -
    they keep the whole-frame gate unchanged, because without attribution there is no
    honest way to tell whose body the evidence belongs to.

    "Reads as a minor" needs a face crop to read, and a subject can exist without one:
    `plan_regions` drops any crop that clamps below _MIN_CROP_PX, so a face under about
    28px yields waist and torso crops with no face crop at all. Such a subject's age is
    *unknown*, not adult, and `all_minor` must not count them as evidence that an adult
    is present - that let a photo whose only readable subject was a child surface on the
    unaged subject's body crops, scoring exactly as if a confirmed adult were there.
    So the verdict is taken over the subjects whose age could actually be read, and when
    none could, the whole-frame gate is left in charge (`has_readable_subjects`).
    """

    row_minor: np.ndarray  # per row: belongs to a subject that reads as a minor
    full_allowed: np.ndarray  # per image: may the full frame still contribute evidence
    has_subjects: np.ndarray  # per image: at least one face-anchored subject exists
    # per image: at least one subject has a face crop, so an age reading exists at all
    has_readable_subjects: np.ndarray
    all_minor: np.ndarray  # per image: every subject whose age could be read is a minor
    subject_child: np.ndarray  # (image, subject) child evidence, from face crops
    subject_adult: np.ndarray  # (image, subject) adult evidence, from face crops
    subject_detail: np.ndarray  # (image, subject) detail evidence, from body crops
    subject_minor: np.ndarray  # (image, subject) minor verdict
    subject_present: np.ndarray  # (image, subject) this subject exists in this image


def analyse_subjects(table: RegionScoreTable, config: ScannerConfig) -> SubjectAnalysis | None:
    """Per-person age and detail evidence. None when no image has attributed crops."""
    count = table.image_count
    subject = table.subject_ids()
    if count == 0 or subject.size == 0 or int(subject.max(initial=-1)) < 0:
        return None

    owner = np.asarray(table.owner, dtype=np.int64)
    kinds = np.asarray(table.kinds)
    slots = int(subject.max()) + 1
    shape = (count, slots)
    attributed = subject >= 0
    face_rows = attributed & (kinds == KIND_FACE)
    body_rows = attributed & np.isin(kinds, list(SUBJECT_BODY_KINDS))

    present: np.ndarray = np.zeros(shape, dtype=bool)
    present[owner[attributed], subject[attributed]] = True
    has_face: np.ndarray = np.zeros(shape, dtype=bool)
    has_face[owner[face_rows], subject[face_rows]] = True

    def _best(rows: np.ndarray, axis: str) -> np.ndarray:
        """Strongest raw score for one axis over the given rows, per (image, subject)."""
        # 0.5 is the sigmoid's neutral point: a subject with no eligible crop for this
        # axis must read as "no evidence", not as evidence against.
        out: np.ndarray = np.full(shape, 0.5, dtype=np.float32)
        scores = table.axis_scores.get(axis)
        if scores is not None and rows.any():
            np.maximum.at(out, (owner[rows], subject[rows]), np.asarray(scores, dtype=np.float32)[rows])
        return out

    child = evidence(_best(face_rows, "child"))
    adult = evidence(_best(face_rows, "adult"))
    detail = combine_detail(
        {axis: _best(body_rows, axis).reshape(-1) for axis in DETAIL_AXES if axis in table.axis_scores},
        config.detail_weights,
        strongest_weight=float(config.detail_strongest_weight),
        average_weight=float(config.detail_average_weight),
    )
    detail = (
        detail.reshape(shape) if detail.size == count * slots else np.zeros(shape, dtype=np.float32)
    )

    if config.exclude_minors:
        threshold = float(config.minor_threshold)
        # The same four tests the whole-frame gate applies, but read off one person's own
        # face crop. All four are the face-anchored variants: a subject only exists here
        # because a face was detected for them.
        looks_minor = (child >= threshold) & (child > adult + float(config.child_adult_margin))
        strongly_minor = child >= float(config.strongly_minor_threshold)
        reads_younger = (child > adult + float(config.face_anchored_margin)) & (child >= threshold * 0.4)
        # `weak_adult` - "strong swimwear evidence but no positive adult evidence" - is
        # deliberately NOT applied per subject, though it still guards the whole-frame
        # path in `evaluate`. There it is the only thing standing between an unaged
        # subject and a match. Here the three tests above already read a real face crop,
        # and requiring positive adult evidence on top of them excluded adults whose
        # face is simply turned away or shadowed: the crop yields neither child nor
        # adult evidence, and "no signal" was being treated as "minor".
        #
        # The trade-off is real: a subject whose face reads ambiguously now passes on
        # the absence of child evidence rather than on the presence of adult evidence.
        # The three tests that remain all fire on positive child evidence, so a subject
        # who reads even slightly young is still suppressed.
        minor = has_face & (looks_minor | strongly_minor | reads_younger)
    else:
        minor = np.zeros(shape, dtype=bool)
    minor &= present

    has_subjects = present.any(axis=1)
    any_minor = minor.any(axis=1)
    # Only subjects with a face crop have an age reading at all. Counting a subject
    # whose face was too small to crop as "not a minor" is what let an unaged person
    # stand in for an adult and carry an image past the gate.
    readable = present & has_face
    has_readable_subjects = readable.any(axis=1)
    all_minor = has_readable_subjects & ~(readable & ~minor).any(axis=1)

    row_minor: np.ndarray = np.zeros(owner.size, dtype=bool)
    row_minor[attributed] = minor[owner[attributed], subject[attributed]]
    # The full frame contains everyone, so it stops being usable evidence the moment any
    # subject in it reads as a minor. Images with no minor keep it.
    full_allowed = ~(has_subjects & any_minor)

    return SubjectAnalysis(
        row_minor=row_minor,
        full_allowed=full_allowed,
        has_subjects=has_subjects,
        has_readable_subjects=has_readable_subjects,
        all_minor=all_minor,
        subject_child=child,
        subject_adult=adult,
        subject_detail=detail,
        subject_minor=minor,
        subject_present=present,
    )


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

    # Per-person age reading, when face-anchored crops exist. Everything downstream is
    # then computed with minors' crops (and the frame that contains them) suppressed, so
    # no part of the score can be traced back to a minor's body.
    subjects = analyse_subjects(table, config)
    if subjects is None:
        aggregated = {axis: table.aggregate(axis) for axis in table.axis_scores}
    else:
        keep_rows = ~subjects.row_minor
        aggregated = {
            axis: table.aggregate(axis, row_filter=keep_rows, full_allowed=subjects.full_allowed)
            for axis in table.axis_scores
        }
    # Every gate compares *evidence*, not the raw sigmoid: a raw 0.5 means the axis saw
    # nothing either way, so thresholds applied to raw scores would fire on every image.
    person = evidence(aggregated.get("person", np.full((count,), 0.5, dtype=np.float32)))
    female = evidence(aggregated.get("female", np.full((count,), 0.5, dtype=np.float32)))
    child = evidence(aggregated.get("child", np.full((count,), 0.5, dtype=np.float32)))
    adult = evidence(aggregated.get("adult", np.full((count,), 0.5, dtype=np.float32)))

    # A detected face is direct evidence of a person that does not depend on the prompt.
    has_face: np.ndarray = np.zeros((count,), dtype=bool)
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
        # The two whole-frame tests a per-subject reading may never overrule. The other
        # two are comparative — they weigh child evidence against adult evidence across
        # the whole frame — and reading one person's own face is strictly better
        # evidence than that, which is what the per-subject verdict is for. These two
        # are not comparative: one is overwhelming child evidence on its own terms, and
        # the other is the guard that stops an unaged subject riding out on somebody
        # else's adult reading.
        frame_veto = strongly_minor | weak_adult
    else:
        age_fail = np.zeros((count,), dtype=bool)
        frame_veto = np.zeros((count,), dtype=bool)

    if subjects is not None:
        # Where a face was actually read, that reading decides. The whole-frame tests
        # above still apply to images with no attributed subject - and to images whose
        # subjects all lost their face crop to the minimum-size rule, because there the
        # per-subject verdict has nothing to go on and handing it the decision would
        # replace a real answer with an empty one.
        # The per-subject verdict may *add* exclusions, never remove one. It used to
        # replace the whole-frame answer outright, so a single readable adult face
        # switched the frame veto off: measured on one frame with child evidence 0.98,
        # adding a second subject whose face read clearly adult flipped it from
        # excluded to scored, with the child evidence unchanged. The soft gate still
        # zeroed its score, so it could not surface as a match — but it stopped being
        # filtered out and became visible in the results.
        age_fail = np.where(subjects.has_readable_subjects, subjects.all_minor, age_fail) | frame_veto

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

    stage_arr = np.select(
        [age_fail, ~person_pass, ~female_pass],
        [STAGE_MINOR, STAGE_NO_PERSON, STAGE_NOT_FEMALE],
        default=STAGE_SCORED,
    )
    stage = stage_arr.tolist()
    reason = [STAGE_REASONS[cast(str, value)] for value in stage_arr]
    excluded = stage_arr != STAGE_SCORED

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
