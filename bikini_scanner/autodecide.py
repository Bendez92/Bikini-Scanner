"""Deciding the easy photos automatically, at a stated error rate.

The scanner already puts a number on every photo. What it has never done is say how
much that number can be trusted, so every one of them came back to the reviewer —
including the thousands the model was never in any doubt about. On a folder of 80 000
that is the whole problem: the work is not the hard photos, it is being asked about the
easy ones.

This finds the score above which the model has *demonstrably* been right, and the score
below which it has demonstrably been wrong, measured against the reviewer's own labels.
Everything past those two cuts can be decided without asking; only the middle is worth
a person's time.

The measurement is out-of-fold on purpose. Asking a model how well it does on the
photos it was fitted to gives an answer that is far too flattering, and here that
answer would be used to auto-label tens of thousands of images.
"""

from __future__ import annotations

import logging
from collections.abc import Sequence
from dataclasses import dataclass, field

import numpy as np

LOGGER = logging.getLogger(__name__)

# Below this many labelled photos on the confident side of a cut, the measured error
# rate is noise. At the 99% target one mistake in fewer than a hundred already misses,
# so a cut resting on a handful of labels is not evidence of anything.
MIN_SUPPORT = 40
# Confidence levels offered, as the share of auto-decisions expected to be correct.
TARGETS = (0.999, 0.99, 0.95, 0.90)
DEFAULT_TARGET = 0.99


# Confidence used for the lower bound on a measured accuracy. 1.96 is the two-sided
# 95% normal quantile, so a cut is only taken when its accuracy would still clear the
# target on a pessimistic reading of the evidence behind it.
Z = 1.96


def wilson_lower_bound(correct: int, total: int, z: float = Z) -> float:
    """Pessimistic estimate of an accuracy, given how little was measured.

    Twenty out of twenty is not evidence of 99.9% accuracy — it is evidence of about
    83%. Using the raw fraction instead let a cut resting on a couple of hundred labels
    claim a perfect record and promise nobody would ever be wrong, which on 80 000
    photos is a promise about tens of thousands of images that nothing supports. The
    Wilson score interval is the standard answer and behaves at the extremes, where the
    textbook normal approximation returns a bound of exactly 1.0 and is useless.
    """
    if total <= 0:
        return 0.0
    observed = float(correct) / float(total)
    denominator = 1.0 + z * z / total
    centre = (observed + z * z / (2.0 * total)) / denominator
    spread = z / denominator * float(
        np.sqrt(observed * (1.0 - observed) / total + z * z / (4.0 * total * total))
    )
    return max(0.0, centre - spread)


@dataclass(slots=True)
class Cut:
    """One end of the score range, and how well it did on held-out labels."""

    threshold: float | None = None
    # What was actually observed on the held-out photos past this threshold...
    precision: float | None = None
    # ...and what that observation can actually support, which is the number the cut
    # is chosen on and the number the mistake estimate uses.
    confident_precision: float | None = None
    support: int = 0

    @property
    def found(self) -> bool:
        return self.threshold is not None


@dataclass(slots=True)
class AutoDecidePlan:
    """What would be decided automatically, and what it is expected to cost."""

    target: float = DEFAULT_TARGET
    measured_on: int = 0
    accept: Cut = field(default_factory=Cut)
    reject: Cut = field(default_factory=Cut)
    accept_paths: list[str] = field(default_factory=list)
    reject_paths: list[str] = field(default_factory=list)
    remaining: int = 0
    reason: str = ""

    @property
    def decided(self) -> int:
        return len(self.accept_paths) + len(self.reject_paths)

    @property
    def expected_mistakes(self) -> float:
        """Rough count of auto-decisions that will be wrong, from the measured rates."""
        total = 0.0
        for cut, paths in ((self.accept, self.accept_paths), (self.reject, self.reject_paths)):
            rate = cut.confident_precision if cut.confident_precision is not None else cut.precision
            if rate is not None:
                total += (1.0 - float(rate)) * len(paths)
        return total

    @property
    def usable(self) -> bool:
        return self.decided > 0


def _cut(scores: np.ndarray, wanted: np.ndarray, target: float, min_support: int, high: bool) -> Cut:
    """The most inclusive threshold whose held-out accuracy still meets `target`.

    `high` picks the top of the range (auto-accept); otherwise the bottom
    (auto-reject). Candidates are the distinct labelled scores, so a threshold never
    lands in the middle of a run of ties and claims an accuracy it was not measured at.
    """
    if scores.size == 0:
        return Cut()
    order = np.argsort(scores, kind="stable")
    if high:
        order = order[::-1]
    ranked_scores = scores[order]
    ranked_wanted = wanted[order].astype(np.int64)
    # Accuracy of every prefix, i.e. of every candidate threshold in turn.
    correct: np.ndarray = np.cumsum(ranked_wanted)
    counts: np.ndarray = np.arange(1, len(ranked_wanted) + 1, dtype=np.int64)
    precision = correct / counts
    # Only the last row of a run of equal scores is a real candidate: including half a
    # tied run is not a rule that can be applied to an unlabelled photo.
    last_of_run = np.r_[ranked_scores[1:] != ranked_scores[:-1], True]
    possible = last_of_run & (counts >= int(min_support)) & (precision >= float(target))
    if not possible.any():
        return Cut()
    # Judge candidates on what their evidence supports, not on the raw fraction, and
    # take the most inclusive that survives — the one furthest down the ranking.
    for index in np.flatnonzero(possible)[::-1]:
        position = int(index)
        bound = wilson_lower_bound(int(correct[position]), int(counts[position]))
        if bound >= float(target):
            return Cut(
                threshold=float(ranked_scores[position]),
                precision=float(precision[position]),
                confident_precision=bound,
                support=int(counts[position]),
            )
    return Cut()


def plan_auto_decisions(
    undecided_paths: Sequence[str],
    undecided_scores: Sequence[float],
    labelled_scores: Sequence[float],
    labelled_labels: Sequence[int],
    target: float = DEFAULT_TARGET,
    min_support: int = MIN_SUPPORT,
) -> AutoDecidePlan:
    """Work out which undecided photos the model can be trusted to decide alone.

    `labelled_scores` must be **out-of-fold** — each one produced by a model that had
    not seen that photo — or the error rate this reports is the one the model achieves
    on its own training data, which is not a rate at all.
    """
    plan = AutoDecidePlan(target=float(target))
    scores: np.ndarray = np.asarray(labelled_scores, dtype=np.float64)
    truth: np.ndarray = np.asarray(labelled_labels, dtype=np.int64)
    keep = np.isfinite(scores)
    scores, truth = scores[keep], truth[keep]
    plan.measured_on = int(scores.size)
    if scores.size == 0:
        plan.reason = "Nothing has been labelled yet, so there is no accuracy to measure."
        return plan
    if int((truth == 1).sum()) == 0 or int((truth == 0).sum()) == 0:
        plan.reason = "Every label so far is the same verdict, so there is no boundary to find."
        return plan
    if scores.size < min_support:
        plan.reason = (
            f"{scores.size} labelled photos is too few to measure an error rate; "
            f"about {min_support} are needed."
        )
        return plan

    plan.accept = _cut(scores, truth == 1, target, min_support, high=True)
    plan.reject = _cut(scores, truth == 0, target, min_support, high=False)

    candidates: np.ndarray = np.asarray(undecided_scores, dtype=np.float64)
    paths = [str(path) for path in undecided_paths]
    if candidates.size != len(paths):
        raise ValueError("undecided_paths and undecided_scores must be the same length")
    taken: np.ndarray = np.zeros(len(paths), dtype=bool)
    if plan.accept.found:
        chosen = candidates >= float(plan.accept.threshold or 0.0)
        plan.accept_paths = [path for path, hit in zip(paths, chosen, strict=True) if hit]
        taken |= chosen
    if plan.reject.found:
        # `& ~taken` matters only if the two cuts cross, which they can when the labels
        # are contradictory. An accept and a reject on one photo is not a decision.
        chosen = (candidates <= float(plan.reject.threshold or 0.0)) & ~taken
        plan.reject_paths = [path for path, hit in zip(paths, chosen, strict=True) if hit]
        taken |= chosen
    plan.remaining = int((~taken).sum())
    if not plan.usable:
        plan.reason = (
            f"No part of the score range has been right {target * 100:.1f}% of the time yet. "
            "Judge more photos, or accept a lower confidence."
        )
    return plan
