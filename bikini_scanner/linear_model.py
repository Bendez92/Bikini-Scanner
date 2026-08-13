"""Small numpy implementations of the few learning primitives this app needs.

scikit-learn and scipy together weigh ~96 MB of compiled binaries in the packaged
build, and this app used a thin slice of them: one L2 logistic regression, feature
standardisation, Platt calibration, stratified splits, and ROC AUC. Everything here is
a direct replacement for that slice, verified against the sklearn versions on real
labels before the dependency was dropped.

Deliberately not a general ML library - it does exactly what the scanner asks for, on
the shapes the scanner produces (a few hundred labelled rows, ~1k features).
"""

from __future__ import annotations

import logging
from dataclasses import dataclass, field

import numpy as np

LOGGER = logging.getLogger(__name__)


def sigmoid(values: np.ndarray) -> np.ndarray:
    """Numerically stable logistic function (replaces scipy.special.expit)."""
    values = np.asarray(values, dtype=np.float64)
    out = np.empty_like(values)
    positive = values >= 0
    out[positive] = 1.0 / (1.0 + np.exp(-values[positive]))
    exp_values = np.exp(values[~positive])
    out[~positive] = exp_values / (1.0 + exp_values)
    return out.astype(np.float32)


def logit(values: np.ndarray) -> np.ndarray:
    """Inverse of sigmoid, clamped to keep finite values away from 0 and 1."""
    values = np.asarray(values, dtype=np.float64)
    clamped = np.clip(values, 1e-7, 1 - 1e-7)
    return (np.log(clamped) - np.log1p(-clamped)).astype(np.float32)


def roc_auc(labels: np.ndarray, scores: np.ndarray) -> float:
    """Rank-based ROC AUC with tie handling (replaces sklearn.metrics.roc_auc_score)."""
    labels = np.asarray(labels).astype(np.int64).ravel()
    scores = np.asarray(scores, dtype=np.float64).ravel()
    positives = int((labels == 1).sum())
    negatives = int((labels == 0).sum())
    if len(labels) < 4 or positives == 0 or negatives == 0:
        # Too few points or a single class make the AUC an unreliable model-selection
        # signal; leave it to the caller to fall back to an uninformative score.
        raise ValueError("ROC AUC needs at least four samples from both classes")
    order = np.argsort(scores, kind="mergesort")
    ranked = scores[order]
    ranks = np.empty(len(scores), dtype=np.float64)
    index = 0
    # Average the ranks inside each tie group, exactly as the standard definition does.
    while index < len(ranked):
        end = index
        while end + 1 < len(ranked) and ranked[end + 1] == ranked[index]:
            end += 1
        ranks[order[index : end + 1]] = 0.5 * (index + end) + 1.0
        index = end + 1
    positive_rank_sum = float(ranks[labels == 1].sum())
    return (positive_rank_sum - positives * (positives + 1) / 2.0) / (positives * negatives)


def average_precision(labels: np.ndarray, scores: np.ndarray) -> float:
    """Area under the precision-recall curve, step-wise (sklearn's definition)."""
    labels = np.asarray(labels).astype(np.int64).ravel()
    scores = np.asarray(scores, dtype=np.float64).ravel()
    positives = int((labels == 1).sum())
    if positives == 0:
        return 0.0
    order = np.argsort(-scores, kind="mergesort")
    hits = labels[order] == 1
    cumulative_hits = np.cumsum(hits)
    precision = cumulative_hits / np.arange(1, len(hits) + 1)
    return float((precision * hits).sum() / positives)


@dataclass(slots=True)
class StandardScaler:
    """Zero-mean, unit-variance per feature; constant features are left alone."""

    mean: np.ndarray = field(default_factory=lambda: np.zeros(0, dtype=np.float32))
    scale: np.ndarray = field(default_factory=lambda: np.ones(0, dtype=np.float32))

    def fit(self, features: np.ndarray) -> StandardScaler:
        features = np.asarray(features, dtype=np.float64)
        self.mean = features.mean(axis=0).astype(np.float32)
        deviation = features.std(axis=0)
        deviation[deviation < 1e-8] = 1.0
        self.scale = deviation.astype(np.float32)
        return self

    def transform(self, features: np.ndarray) -> np.ndarray:
        features = np.asarray(features, dtype=np.float32)
        if self.mean.size == 0:
            return features
        return ((features - self.mean) / self.scale).astype(np.float32)


@dataclass(slots=True)
class LogisticRegression:
    """L2-regularised binary logistic regression with balanced class weights.

    Fitted with full-batch Adam on standardised features. The data here is small and
    wide (hundreds of rows, ~1k columns), where this converges in well under a second
    and matches liblinear closely enough that ranking is unaffected.
    """

    C: float = 1.0
    max_iter: int = 600
    tol: float = 1e-6
    scaler: StandardScaler = field(default_factory=StandardScaler)
    coef: np.ndarray = field(default_factory=lambda: np.zeros(0, dtype=np.float32))
    intercept: float = 0.0
    fitted: bool = False
    converged: bool = field(init=False, default=False)

    def fit(self, features: np.ndarray, labels: np.ndarray) -> LogisticRegression:
        features = np.asarray(features, dtype=np.float32)
        labels = np.asarray(labels).astype(np.float64).ravel()
        if features.ndim != 2 or features.shape[0] != labels.shape[0]:
            raise ValueError("feature/label shape mismatch")
        self.scaler = StandardScaler().fit(features)
        x = self.scaler.transform(features).astype(np.float64)
        n, d = x.shape

        # class_weight="balanced": each class contributes equally regardless of size.
        weights = np.ones(n, dtype=np.float64)
        for value in (0, 1):
            mask = labels == value
            count = int(mask.sum())
            if count:
                weights[mask] = n / (2.0 * count)
        weights /= weights.mean()

        penalty = 1.0 / max(float(self.C), 1e-6)
        coef = np.zeros(d, dtype=np.float64)
        intercept = 0.0
        m_coef = np.zeros(d, dtype=np.float64)
        v_coef = np.zeros(d, dtype=np.float64)
        m_int = 0.0
        v_int = 0.0
        beta1, beta2, epsilon, step = 0.9, 0.999, 1e-8, 0.1
        previous_loss = np.inf
        last_loss = np.nan
        converged = False
        for iteration in range(1, int(self.max_iter) + 1):
            logits = x @ coef + intercept
            probabilities = sigmoid(logits).astype(np.float64)
            residual = (probabilities - labels) * weights
            grad_coef = x.T @ residual / n + penalty * coef / n
            grad_int = float(residual.sum() / n)

            m_coef = beta1 * m_coef + (1 - beta1) * grad_coef
            v_coef = beta2 * v_coef + (1 - beta2) * grad_coef**2
            m_int = beta1 * m_int + (1 - beta1) * grad_int
            v_int = beta2 * v_int + (1 - beta2) * grad_int**2
            bias1 = 1 - beta1**iteration
            bias2 = 1 - beta2**iteration
            coef -= step * (m_coef / bias1) / (np.sqrt(v_coef / bias2) + epsilon)
            intercept -= step * (m_int / bias1) / (np.sqrt(v_int / bias2) + epsilon)

            if iteration % 25 == 0:
                clipped = np.clip(probabilities, 1e-9, 1 - 1e-9)
                loss = float(
                    -(weights * (labels * np.log(clipped) + (1 - labels) * np.log(1 - clipped))).mean()
                    + 0.5 * penalty * float(coef @ coef) / n
                )
                last_loss = loss
                if abs(previous_loss - loss) < self.tol:
                    converged = True
                    break
                previous_loss = loss

        self.coef = coef.astype(np.float32)
        self.intercept = float(intercept)
        self.converged = converged
        self.fitted = True
        if not converged:
            LOGGER.warning(
                "Logistic regression did not converge within %d iterations (final loss change %.3g)",
                self.max_iter,
                abs(previous_loss - last_loss) if previous_loss != np.inf else np.nan,
            )
        return self

    def decision_function(self, features: np.ndarray) -> np.ndarray:
        if not self.fitted:
            raise RuntimeError("model is not fitted")
        x = self.scaler.transform(features)
        return (x @ self.coef + self.intercept).astype(np.float32)

    def predict_proba(self, features: np.ndarray) -> np.ndarray:
        scores = sigmoid(self.decision_function(features))
        return np.column_stack([1.0 - scores, scores]).astype(np.float32)


@dataclass(slots=True)
class PlattCalibrator:
    """Maps raw decision values onto calibrated probabilities (sigmoid calibration)."""

    model: LogisticRegression
    slope: float = 1.0
    bias: float = 0.0

    def fit(self, features: np.ndarray, labels: np.ndarray, min_samples: int = 8) -> PlattCalibrator:
        labels = np.asarray(labels).astype(np.int64).ravel()
        if labels.size < min_samples or len(set(labels.tolist())) < 2:
            raise ValueError("Platt calibration needs at least min_samples from both classes")
        scores = self.model.decision_function(features).astype(np.float64).reshape(-1, 1)
        calibrator = LogisticRegression(C=1e6, max_iter=400)
        calibrator.fit(scores.astype(np.float32), labels)
        # The 1-D calibrator standardises its input, so fold that back into slope/bias.
        scale = float(calibrator.scaler.scale[0]) or 1.0
        mean = float(calibrator.scaler.mean[0])
        self.slope = float(calibrator.coef[0]) / scale
        self.bias = float(calibrator.intercept) - float(calibrator.coef[0]) * mean / scale
        return self

    def predict_proba(self, features: np.ndarray) -> np.ndarray:
        raw = self.model.decision_function(features).astype(np.float64)
        positive = sigmoid(self.slope * raw + self.bias)
        return np.column_stack([1.0 - positive, positive]).astype(np.float32)


def stratified_folds(labels: np.ndarray, folds: int, seed: int = 42) -> list[np.ndarray]:
    """Indices for each fold, keeping the class balance (replaces StratifiedKFold)."""
    labels = np.asarray(labels).astype(np.int64).ravel()
    rng = np.random.default_rng(seed)
    buckets: list[list[int]] = [[] for _ in range(folds)]
    for value in (0, 1):
        members = np.nonzero(labels == value)[0]
        rng.shuffle(members)
        for position, index in enumerate(members):
            buckets[position % folds].append(int(index))
    return [np.asarray(sorted(bucket), dtype=np.int64) for bucket in buckets]


def stratified_split(labels: np.ndarray, test_size: int, seed: int = 42) -> tuple[np.ndarray, np.ndarray]:
    """Train/test indices keeping both classes on both sides where possible."""
    labels = np.asarray(labels).astype(np.int64).ravel()
    rng = np.random.default_rng(seed)
    test: list[int] = []
    total = len(labels)
    for value in (0, 1):
        members = np.nonzero(labels == value)[0]
        rng.shuffle(members)
        share = max(1, int(round(test_size * len(members) / max(total, 1))))
        # Leave at least one member of the class in the training set if possible,
        # so a tiny class does not disappear from the training fold entirely.
        share = min(share, max(0, len(members) - 1))
        test.extend(int(index) for index in members[:share])
    test_index = np.asarray(sorted(set(test)), dtype=np.int64)
    mask = np.ones(total, dtype=bool)
    mask[test_index] = False
    return np.nonzero(mask)[0], test_index


def cross_val_scores(features: np.ndarray, labels: np.ndarray, c_value: float, folds: int) -> np.ndarray:
    """Out-of-fold probabilities (replaces cross_val_predict with predict_proba).

    Rows in a fold that could not be trained against come back as NaN rather than 0.0:
    a silent zero reads as "confidently negative" and drags the measured AUC around
    instead of being recognised as a missing prediction.
    """
    features = np.asarray(features, dtype=np.float32)
    labels = np.asarray(labels).astype(np.int64).ravel()
    out = np.full(len(labels), np.nan, dtype=np.float32)
    for held_out in stratified_folds(labels, folds):
        mask = np.ones(len(labels), dtype=bool)
        mask[held_out] = False
        if len(set(labels[mask].tolist())) < 2 or held_out.size == 0:
            continue
        model = LogisticRegression(C=c_value).fit(features[mask], labels[mask])
        out[held_out] = model.predict_proba(features[held_out])[:, 1]
    return out
