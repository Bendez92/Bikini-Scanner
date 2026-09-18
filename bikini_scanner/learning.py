"""Learning from Accept/REJECT decisions.

Three things make labels count for more than they used to:

* **Prototypes work from label one.** A logistic model needs a handful of examples of
  each class before it means anything. The centroid of what you accepted versus what
  you rejected is useful immediately, so early labels change the ranking straight away.
* **The blend is earned, not fixed.** The old pipeline always mixed 70% classifier with
  30% zero-shot, whether the classifier had seen six labels or six hundred. Here the
  weight comes from the model's own cross-validated AUC and how much data it has, so a
  weak model stays out of the way and a strong one takes over.
* **Regularisation is tuned, not assumed.** C is picked by cross-validation once there
  is enough data to measure it.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass
from typing import Any

import numpy as np

from .linear_model import (
    LogisticRegression,
    PlattCalibrator,
    cross_val_scores,
    roc_auc,
    sigmoid,
    stratified_split,
)

LOGGER = logging.getLogger(__name__)

# Below this many labels of a class, only the prototype model is used.
MIN_CLASSIFIER_LABELS = 6
# The classifier reaches its full share of the blend at this many labels.
FULL_TRUST_LABELS = 40
_C_GRID = (0.05, 0.25, 1.0, 4.0, 16.0)


def _l2_normalize(matrix: np.ndarray) -> np.ndarray:
    matrix = np.asarray(matrix, dtype=np.float32)
    if matrix.ndim == 1:
        matrix = matrix[None, :]
    norms = np.linalg.norm(matrix, axis=1, keepdims=True)
    norms[norms <= 0] = 1.0
    return (matrix / norms).astype(np.float32)


@dataclass(slots=True)
class PrototypeModel:
    """Centroid of accepted minus centroid of rejected. Usable from the first label."""

    positive: np.ndarray
    negative: np.ndarray
    scale: float = 8.0

    def score(self, features: np.ndarray) -> np.ndarray:
        normalized = _l2_normalize(features)
        positive_similarity = normalized @ self.positive
        negative_similarity = normalized @ self.negative
        return sigmoid((positive_similarity - negative_similarity) * self.scale)


@dataclass(slots=True)
class LearningOutcome:
    classifier: Any | None = None
    prototype: PrototypeModel | None = None
    label_count: int = 0
    positive_count: int = 0
    negative_count: int = 0
    # Of `label_count`, how many came from the folder on screen. The rest were pooled
    # from other folders by global learning. Reported separately because a reviewer
    # comparing the status bar against their own tally needs the two to reconcile.
    # `fit` sets these to the whole training set, so a caller that never pools reads
    # correctly by default; only a pooling caller narrows them.
    local_count: int = 0
    local_positive: int = 0
    local_negative: int = 0
    cv_auc: float | None = None
    weight: float = 0.0
    chosen_c: float | None = None

    @property
    def trained(self) -> bool:
        return self.classifier is not None or self.prototype is not None

    def summary(self) -> str:
        """One line the reviewer can read to see their labels landing.

        Spelling out the accepted/rejected split matters: the most common reason a
        run of Accepts changes nothing is that no REJECTs exist yet, and a count of
        labels alone hides that.
        """
        if not self.trained:
            return "not trained"
        pooled = max(0, self.label_count - self.local_count)
        if pooled:
            parts = [
                f"{self.local_count} labels here "
                f"({self.local_positive} accepted / {self.local_negative} rejected) "
                f"+ {pooled} pooled from other folders"
            ]
        else:
            parts = [f"{self.label_count} labels ({self.positive_count} accepted / {self.negative_count} rejected)"]
        if self.cv_auc is not None:
            parts.append(f"AUC {self.cv_auc:.2f}")
        if self.classifier is None:
            parts.append("prototype only")
        parts.append(f"influence {self.weight * 100:.0f}%")
        if self.weight <= 0.0:
            parts.append("not yet steering the ranking")
        elif self.label_count < FULL_TRUST_LABELS:
            parts.append(f"grows to full strength at {FULL_TRUST_LABELS}")
        return ", ".join(parts)

    def score(self, features: np.ndarray) -> np.ndarray | None:
        """Learned score in 0..1, or None when nothing has been learned yet."""
        features = np.asarray(features, dtype=np.float32)
        if features.size == 0:
            return np.empty((0,), dtype=np.float32)
        prototype_scores = self.prototype.score(features) if self.prototype is not None else None
        classifier_scores = None
        if self.classifier is not None:
            try:
                classifier_scores = self.classifier.predict_proba(features)[:, 1].astype(np.float32)
            except Exception:
                LOGGER.exception("Learned classifier failed to score; falling back to the prototype")
                classifier_scores = None
        if classifier_scores is None:
            return prototype_scores
        if prototype_scores is None:
            return classifier_scores
        # Hand over from prototype to classifier as evidence accumulates.
        share = float(np.clip(self.label_count / FULL_TRUST_LABELS, 0.0, 1.0))
        return (share * classifier_scores + (1.0 - share) * prototype_scores).astype(np.float32)


def _fit_prototype(features: np.ndarray, labels: np.ndarray) -> PrototypeModel | None:
    positives = features[labels == 1]
    negatives = features[labels == 0]
    if positives.shape[0] == 0 or negatives.shape[0] == 0:
        return None
    positive_centroid = _l2_normalize(_l2_normalize(positives).mean(axis=0))[0]
    negative_centroid = _l2_normalize(_l2_normalize(negatives).mean(axis=0))[0]
    return PrototypeModel(positive=positive_centroid, negative=negative_centroid)


def _make_estimator(c_value: float) -> Any:
    # Feature standardisation is built into the model, so no pipeline wrapper is needed.
    return LogisticRegression(C=float(c_value))


def warm_start_of(model: Any) -> tuple[np.ndarray | None, float | None]:
    """Coefficients to start the next fit from, unwrapping a calibrator if present."""
    inner = getattr(model, "model", model)
    if not isinstance(inner, LogisticRegression) or not inner.fitted or inner.coef.size == 0:
        return None, None
    return inner.coef, float(inner.intercept)


def _cross_validated_auc(
    features: np.ndarray,
    labels: np.ndarray,
    c_value: float,
    init_coef: np.ndarray | None = None,
    init_intercept: float | None = None,
) -> float | None:
    counts = np.bincount(labels, minlength=2)
    folds = int(min(5, counts.min()))
    if folds < 3:
        return None
    try:
        predictions = cross_val_scores(features, labels, float(c_value), folds, init_coef, init_intercept)
        scored = np.isfinite(predictions)
        if not scored.all():
            # A fold that could not be trained against contributes no measurement.
            if int((labels[scored] == 1).sum()) == 0 or int((labels[scored] == 0).sum()) == 0:
                return None
            return float(roc_auc(labels[scored], predictions[scored]))
        return float(roc_auc(labels, predictions))
    except Exception as exc:  # noqa: BLE001
        # An unmeasurable AUC is normal early on (too few labels of one class), but it
        # also decides how much influence the learned model earns, so a fit that fails
        # for some *other* reason quietly caps the model at the unmeasured default.
        LOGGER.debug("Cross-validated AUC unavailable at C=%s: %s", c_value, exc)
        return None


def _select_c(
    features: np.ndarray,
    labels: np.ndarray,
    only: float | None = None,
    init_coef: np.ndarray | None = None,
    init_intercept: float | None = None,
) -> tuple[float, float | None]:
    """Pick C by cross-validated AUC; fall back to a middling value when unmeasurable.

    `only` restricts the search to a single value - used when the caller has already
    chosen C on more data than this retrain added. Sweeping the whole grid costs folds
    x grid fits, which is the bulk of the work in a retrain, and the answer does not
    move for the sake of three new labels.
    """
    counts = np.bincount(labels, minlength=2)
    if len(labels) < 12 or counts.min() < 3:
        return 1.0, None
    grid = (float(only),) if only is not None else _C_GRID
    best_c = float(only) if only is not None else 1.0
    best_auc: float | None = None
    for candidate in grid:
        auc = _cross_validated_auc(features, labels, candidate, init_coef, init_intercept)
        if auc is None:
            continue
        if best_auc is None or auc > best_auc:
            best_auc = auc
            best_c = candidate
    return best_c, best_auc


def _calibrate(
    estimator: Any,
    features: np.ndarray,
    labels: np.ndarray,
    init_coef: np.ndarray | None = None,
    init_intercept: float | None = None,
) -> Any:
    """Hold out a slice to calibrate probabilities when there is enough data."""
    counts = np.bincount(labels, minlength=2)
    if counts.min() < 4 or len(labels) < 16:
        estimator.fit(features, labels, init_coef=init_coef, init_intercept=init_intercept)
        return estimator
    try:
        test_size = max(4, int(round(len(labels) * 0.25)))
        if len(labels) - test_size < 8:
            estimator.fit(features, labels, init_coef=init_coef, init_intercept=init_intercept)
            return estimator
        train_index, calibration_index = stratified_split(labels, test_size)
        y_train = labels[train_index]
        y_calibration = labels[calibration_index]
        if len(set(y_train.tolist())) < 2 or len(set(y_calibration.tolist())) < 2:
            estimator.fit(features, labels, init_coef=init_coef, init_intercept=init_intercept)
            return estimator
        estimator.fit(features[train_index], y_train, init_coef=init_coef, init_intercept=init_intercept)
        return PlattCalibrator(model=estimator).fit(features[calibration_index], y_calibration)
    except Exception as exc:  # noqa: BLE001
        # Falling back to the uncalibrated model is the right behaviour, but it changes
        # what the scores mean, so it should not happen invisibly.
        LOGGER.info("Probability calibration failed; using the uncalibrated model: %s", exc)
        estimator.fit(features, labels, init_coef=init_coef, init_intercept=init_intercept)
        return estimator


def fit(
    features: np.ndarray,
    labels: np.ndarray,
    max_weight: float = 0.85,
    warm_start: Any | None = None,
    reuse_c: float | None = None,
) -> LearningOutcome:
    """Train from labelled feature vectors. Never raises: a failure means no learning.

    `warm_start` is the previously fitted model; its coefficients seed this fit, which
    is what makes a retrain after a few new labels cheap. `reuse_c` skips the
    regularisation sweep and takes the value the caller already settled on. Both are
    optimisations only - omit them and this is the same cold fit it always was.
    """
    features = np.asarray(features, dtype=np.float32)
    labels = np.asarray(labels, dtype=np.int64)
    if features.size == 0 or labels.size == 0 or features.shape[0] != labels.shape[0]:
        return LearningOutcome()
    keep = np.isin(labels, (0, 1))
    features = features[keep]
    labels = labels[keep]
    if labels.size == 0:
        return LearningOutcome()

    counts = np.bincount(labels, minlength=2)
    outcome = LearningOutcome(
        label_count=int(labels.size),
        positive_count=int(counts[1]),
        negative_count=int(counts[0]),
        local_count=int(labels.size),
        local_positive=int(counts[1]),
        local_negative=int(counts[0]),
    )
    outcome.prototype = _fit_prototype(features, labels)

    if counts.min() >= 3 and labels.size >= MIN_CLASSIFIER_LABELS:
        init_coef, init_intercept = warm_start_of(warm_start) if warm_start is not None else (None, None)
        chosen_c, auc = _select_c(features, labels, only=reuse_c, init_coef=init_coef, init_intercept=init_intercept)
        outcome.chosen_c = chosen_c
        outcome.cv_auc = auc
        try:
            outcome.classifier = _calibrate(
                _make_estimator(chosen_c), features, labels, init_coef=init_coef, init_intercept=init_intercept
            )
        except Exception:
            LOGGER.exception("Classifier training failed; keeping the prototype model")
            outcome.classifier = None

    outcome.weight = _blend_weight(outcome, max_weight=max_weight)
    return outcome


def _blend_weight(outcome: LearningOutcome, max_weight: float = 0.85) -> float:
    """How much of the final score the learned model earns, in 0..max_weight.

    Driven by measured separability (AUC) and volume of evidence. A model that cannot
    beat a coin flip on held-out data gets no influence at all. Capped below 1.0 so
    zero-shot prompts always retain a voice.
    """
    if not outcome.trained:
        return 0.0
    volume = float(np.clip(outcome.label_count / FULL_TRUST_LABELS, 0.0, 1.0))
    if outcome.cv_auc is None:
        # Unmeasured: allow a modest influence that still scales with evidence.
        quality = 0.5 if outcome.classifier is not None else 0.35
    else:
        quality = float(np.clip((outcome.cv_auc - 0.5) * 2.0, 0.0, 1.0))
    return float(np.clip(quality * volume, 0.0, 1.0) * float(max_weight))


def blend(zero_shot: np.ndarray, learned: np.ndarray | None, weight: float) -> np.ndarray:
    zero_shot = np.asarray(zero_shot, dtype=np.float32)
    if learned is None or weight <= 0 or learned.shape != zero_shot.shape:
        return zero_shot
    weight = float(np.clip(weight, 0.0, 1.0))
    return ((1.0 - weight) * zero_shot + weight * np.asarray(learned, dtype=np.float32)).astype(np.float32)
