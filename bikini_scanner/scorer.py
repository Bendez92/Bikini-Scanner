from __future__ import annotations

import hashlib
import inspect
import json
import logging
import re
import threading
from collections.abc import Callable, Iterable, Sequence
from dataclasses import dataclass, field
from datetime import datetime, timezone
from pathlib import Path
from typing import TYPE_CHECKING

import numpy as np
from PIL import Image

from . import cascade as cascade_module
from . import learning
from .cascade import RegionScoreTable
from .config import (
    DEFAULT_CLASSIFIER_WEIGHT,
    DEFAULT_ZERO_SHOT_SCALE,
    DEFAULT_ZERO_SHOT_WEIGHT,
    AxisConfig,
    ScannerConfig,
)
from .global_store import GlobalLearningStore
from .image_formats import open_oriented
from .linear_model import LogisticRegression, PlattCalibrator, logit, roc_auc, stratified_split
from .linear_model import sigmoid as expit
from .logging_setup import configure_logging
from .regions import (
    FULL_REGION,
    KIND_FACE,
    KIND_FULL,
    REGION_GEOMETRY_VERSION,
    crop_regions,
    plan_regions,
    region_kind,
)
from .skin import skin_fraction
from .store import FolderStore, collect_image_paths, safe_stat
from .vision_analysis import detect_face_boxes, detect_face_count

LOGGER = logging.getLogger(__name__)

# Axis order in the learned feature vector. Fixed, because features are pooled across
# folders and scans - appending is safe, reordering is not.
FEATURE_AXIS_ORDER = (
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
    "gate_person",
    "gate_female",
    "gate_age",
    "detail",
)

if TYPE_CHECKING:
    from .backend_utils import ImageEmbeddingBackend


@dataclass(slots=True)
class ScoreState:
    paths: list[str]
    embeddings: np.ndarray
    zero_shot_scores: np.ndarray
    scores: np.ndarray
    axis_scores: dict[str, np.ndarray]
    face_counts: np.ndarray | None
    classifier_trained: bool
    classifier_label_count: int
    scan_timestamp: str = ""
    # Cascade output. detail_embeddings holds the best body-region crop per image (the
    # full frame when no crop was taken), which is what the learned model trains on.
    detail_embeddings: np.ndarray | None = None
    region_table: RegionScoreTable | None = None
    cascade_stage: list[str] = field(default_factory=list)
    cascade_reason: list[str] = field(default_factory=list)
    excluded: np.ndarray | None = None
    features: np.ndarray | None = None
    learning_summary: str = ""
    deep_scanned: int = 0
    detail_regions: list[str] = field(default_factory=list)
    refine: RefineResult | None = None


class ScanCancelled(Exception):
    """Raised when a cooperative scan cancellation was requested."""


# A scan is not one loop: images are embedded, then candidates get their region crops,
# then a second-opinion model may run, then everything is scored. Each phase knows its
# own item count, and each owns a fixed slice of the overall bar so the percentage only
# ever moves forwards.
PHASE_EMBED = "embed"
PHASE_DETAIL = "detail"
PHASE_REFINE = "refine"
PHASE_SCORE = "score"

PHASE_LABELS = {
    PHASE_EMBED: "Reading images",
    PHASE_DETAIL: "Checking body regions",
    PHASE_REFINE: "Second opinion",
    PHASE_SCORE: "Scoring",
}
# Shares of the overall bar. Deliberately not equal: embedding dominates a cold scan.
PHASE_SHARES = {PHASE_EMBED: 0.65, PHASE_DETAIL: 0.30, PHASE_REFINE: 0.03, PHASE_SCORE: 0.01}
# Shares when no region pass will run, so the bar still reaches 100%.
PHASE_SHARES_NO_DETAIL = {PHASE_EMBED: 0.95, PHASE_DETAIL: 0.0, PHASE_REFINE: 0.03, PHASE_SCORE: 0.01}


@dataclass(slots=True)
class ScanProgress:
    """One progress tick. `done`/`total` count items in the current phase only."""

    phase: str
    label: str
    done: int
    total: int
    rate: float
    eta_seconds: float | None
    fraction: float  # overall 0..1 across the whole scan

    @property
    def percent(self) -> float:
        return float(self.fraction) * 100.0

    def text(self) -> str:
        """The 'x / N files' readout shown next to the bar."""
        if self.total <= 0:
            return self.label
        noun = "files" if self.phase == PHASE_EMBED else "images"
        return f"{self.label} — {self.done:,} / {self.total:,} {noun}"


class _ProgressReporter:
    """Turns per-phase item counts into monotonic overall progress.

    Ticks are throttled: a 20k-image folder would otherwise post 20k callbacks into the
    Tk event loop, which costs more than the scan.
    """

    _MIN_INTERVAL_SECONDS = 0.06

    def __init__(
        self,
        callback: Callable[..., None] | None,
        shares: dict[str, float] | None = None,
    ) -> None:
        self._callback = callback
        self._shares = dict(shares or PHASE_SHARES)
        self._arity: int | None = None
        if callback is not None:
            try:
                self._arity = len(inspect.signature(callback).parameters)
            except Exception:  # noqa: BLE001
                self._arity = None
        self._phase = PHASE_EMBED
        self._total = 0
        self._done = 0
        self._base = 0.0
        self._fraction = 0.0
        self._markers: list[tuple[int, float]] = []
        self._last_emit = 0.0

    @property
    def active(self) -> bool:
        return self._callback is not None

    def start_phase(self, phase: str, total: int) -> None:
        # Everything before this phase is finished, so its share is banked.
        self._base = self._fraction
        self._phase = phase
        self._total = max(int(total), 0)
        self._done = 0
        self._markers = []
        self.emit(0, force=True)

    def complete_phase(self) -> None:
        """Bank the current phase's full share, whether or not it emitted a final tick.

        A phase with no work to do (an all-cached folder, no deep-scan candidates) still
        has to hand its slice of the bar to the next one.
        """
        if self._callback is None:
            return
        share = float(self._shares.get(self._phase, 0.0))
        self._fraction = max(self._fraction, min(1.0, self._base + share))
        self.emit(self._total, force=True)

    def finish(self) -> None:
        """Land the bar on 100%, including the shares of phases that never ran."""
        if self._callback is None:
            return
        self._base = 1.0
        self._fraction = 1.0
        self.emit(self._total, force=True)

    def emit(self, done: int, force: bool = False) -> None:
        if self._callback is None:
            return
        self._done = max(0, min(int(done), self._total) if self._total else int(done))
        now = datetime.now(timezone.utc).timestamp()
        finished = self._total > 0 and self._done >= self._total
        if not force and not finished and (now - self._last_emit) < self._MIN_INTERVAL_SECONDS:
            return
        self._last_emit = now

        self._markers.append((self._done, now))
        if len(self._markers) > 12:
            del self._markers[:-12]
        rate = 0.0
        if len(self._markers) >= 2:
            first_done, first_time = self._markers[0]
            last_done, last_time = self._markers[-1]
            elapsed = max(last_time - first_time, 1e-6)
            rate = max(last_done - first_done, 0) / elapsed
        eta = None
        if rate > 0 and self._total:
            eta = max(self._total - self._done, 0) / rate

        share = float(self._shares.get(self._phase, 0.0))
        within = (self._done / self._total) if self._total else 1.0
        # Never let a phase boundary or a re-estimated total walk the bar backwards.
        self._fraction = max(self._fraction, min(1.0, self._base + share * within))
        progress = ScanProgress(
            phase=self._phase,
            label=PHASE_LABELS.get(self._phase, self._phase),
            done=self._done,
            total=self._total,
            rate=rate,
            eta_seconds=eta,
            fraction=self._fraction,
        )
        try:
            # One parameter means the caller wants the whole ScanProgress; the two- and
            # four-argument forms are the older signatures, kept working on purpose.
            if self._arity == 2:
                self._callback(self._done, self._total)
            elif self._arity in (3, 4):
                self._callback(self._done, self._total, rate, eta)
            else:
                self._callback(progress)
        except Exception:
            LOGGER.exception("Progress callback failed; continuing the scan")
            self._callback = None


@dataclass(slots=True)
class AxisEmbeddings:
    positive: np.ndarray
    negative: np.ndarray
    aggregation: str
    positive_weights: np.ndarray
    negative_weights: np.ndarray


@dataclass(slots=True)
class BikiniScorer:
    backend: ImageEmbeddingBackend
    config: ScannerConfig = field(default_factory=ScannerConfig)
    axis_embeddings: dict[str, AxisEmbeddings] = field(init=False, repr=False)
    classifier: LogisticRegression | None = field(init=False, default=None, repr=False)
    zero_shot_scale: float = field(init=False, default=DEFAULT_ZERO_SHOT_SCALE, repr=False)
    classifier_weight: float = field(init=False, default=DEFAULT_CLASSIFIER_WEIGHT, repr=False)
    zero_shot_weight: float = field(init=False, default=DEFAULT_ZERO_SHOT_WEIGHT, repr=False)
    learning_outcome: learning.LearningOutcome = field(init=False, default_factory=learning.LearningOutcome, repr=False)
    _global_cache: GlobalLearningStore | None = field(init=False, default=None, repr=False)
    _global_signature: str = field(init=False, default="", repr=False)

    def __post_init__(self) -> None:
        self.axis_embeddings = {
            axis_name: self._build_axis_embeddings(axis_config)
            for axis_name, axis_config in self.config.axis_prompts.items()
        }
        self.zero_shot_scale = float(self.config.zero_shot_scale)
        self.classifier_weight = float(self.config.classifier_weight)
        self.zero_shot_weight = float(self.config.zero_shot_weight)

    def _build_axis_embeddings(self, axis_config: AxisConfig) -> AxisEmbeddings:
        positive_texts, positive_weights = self._normalize_prompts(axis_config.positive)
        negative_texts, negative_weights = self._normalize_prompts(axis_config.negative)
        positive = self.backend.embed_texts(positive_texts)
        negative = self.backend.embed_texts(negative_texts)
        return AxisEmbeddings(
            positive=positive,
            negative=negative,
            aggregation=axis_config.aggregation,
            positive_weights=positive_weights,
            negative_weights=negative_weights,
        )

    def _classifier_signature(
        self, labeled_paths: Sequence[tuple[str, int]], feature_width: int | None = None
    ) -> str:
        labeled_entries: list[dict[str, object]] = []
        for path, label in sorted((str(path), int(label)) for path, label in labeled_paths if label in (0, 1)):
            entry: dict[str, object] = {"path": path, "label": int(label)}
            try:
                stat = Path(path).stat()
            except Exception:  # noqa: BLE001
                stat = None
            if stat is not None:
                entry["mtime_ns"] = int(stat.st_mtime_ns)
                entry["size"] = int(stat.st_size)
            labeled_entries.append(entry)
        payload: dict[str, object] = {
            "version": 1,
            "backend": getattr(self.backend, "__class__", type(self.backend)).__name__,
            "model_name": self.config.model_name,
            "feature_width": feature_width,
            "feature_layout": list(FEATURE_AXIS_ORDER),
            "labels": labeled_entries,
        }
        return hashlib.sha1(json.dumps(payload, sort_keys=True, separators=(",", ":")).encode("utf-8")).hexdigest()

    @staticmethod
    def _normalize_prompts(prompts: Sequence[str | tuple[str, float]]) -> tuple[list[str], np.ndarray]:
        texts: list[str] = []
        weights: list[float] = []
        for prompt in prompts:
            if isinstance(prompt, str):
                text = prompt.strip()
                weight = 1.0
            else:
                text = str(prompt[0]).strip()
                try:
                    weight = float(prompt[1])
                except Exception:  # noqa: BLE001
                    weight = 1.0
            if not text:
                continue
            texts.append(text)
            weights.append(weight if weight > 0 else 1.0)
        if not texts:
            texts = [""]
            weights = [1.0]
        return texts, np.asarray(weights, dtype=np.float32)

    @staticmethod
    def _weighted_mean(similarities: np.ndarray, weights: np.ndarray) -> np.ndarray:
        if similarities.size == 0:
            return np.empty((0,), dtype=np.float32)
        if similarities.ndim == 1:
            similarities = similarities[:, None]
        normalized_weights = weights / max(float(weights.sum()), 1e-8)
        return (similarities * normalized_weights[None, :]).sum(axis=1).astype(np.float32)

    def _axis_zero_shot_scores(self, embeddings: np.ndarray, axis: AxisEmbeddings) -> np.ndarray:
        if embeddings.size == 0:
            return np.empty((0,), dtype=np.float32)
        positive = embeddings @ axis.positive.T
        negative = embeddings @ axis.negative.T
        if axis.aggregation == "max":
            pos_score = positive.max(axis=1)
            neg_score = negative.max(axis=1)
        else:
            pos_score = self._weighted_mean(positive, axis.positive_weights)
            neg_score = self._weighted_mean(negative, axis.negative_weights)
        raw = (pos_score - neg_score) * self.zero_shot_scale
        return expit(raw).astype(np.float32)

    def axis_zero_shot_scores(self, embeddings: np.ndarray) -> dict[str, np.ndarray]:
        return {
            axis_name: self._axis_zero_shot_scores(embeddings, axis) for axis_name, axis in self.axis_embeddings.items()
        }

    def zero_shot_scores(self, embeddings: np.ndarray) -> np.ndarray:
        return self.axis_zero_shot_scores(embeddings).get("bikini", np.empty((0,), dtype=np.float32))

    def train_classifier(
        self,
        embeddings_by_path: dict[str, np.ndarray],
        labels: dict[str, int],
        store: FolderStore | None = None,
    ) -> int:
        labeled_pairs = [
            (path, label) for path, label in labels.items() if path in embeddings_by_path and label in (0, 1)
        ]
        label_count = len(labeled_pairs)
        label_values = {label for _, label in labeled_pairs}
        if label_count < 6 or label_values != {0, 1}:
            self.classifier = None
            return label_count

        first_embedding = next(iter(embeddings_by_path.values()))
        feature_width = int(first_embedding.shape[0]) * 2 + len(FEATURE_AXIS_ORDER)
        signature = self._classifier_signature(labeled_pairs, feature_width)
        if store is not None:
            cached = store.load_classifier_cache()
            if cached is not None and cached.get("signature") == signature:
                classifier = cached.get("classifier")
                if classifier is not None:
                    self.classifier = classifier
                    return label_count

        X = np.vstack([embeddings_by_path[path] for path, _ in labeled_pairs]).astype(np.float32)
        y = np.asarray([label for _, label in labeled_pairs], dtype=np.int64)
        classifier = self._fit_classifier(X, y)
        self.classifier = classifier
        if store is not None and classifier is not None:
            try:
                store.save_classifier_cache(
                    {
                        "signature": signature,
                        "classifier": classifier,
                    }
                )
            except Exception:  # noqa: BLE001
                pass
        return label_count

    @staticmethod
    def label_counts(labels: dict[str, int]) -> dict[str, int]:
        counts = {"good": 0, "bad": 0, "skip": 0, "unlabeled": 0}
        for value in labels.values():
            if value == 1:
                counts["good"] += 1
            elif value == 0:
                counts["bad"] += 1
            elif value == 2:
                counts["skip"] += 1
        return counts

    def estimate_quality(self, embeddings_by_path: dict[str, np.ndarray], labels: dict[str, int]) -> float | None:
        """How well the learned model separates your Accept/REJECT calls (0.5 = chance)."""
        if self.learning_outcome.cv_auc is not None:
            return float(self.learning_outcome.cv_auc)
        labeled_pairs = [
            (path, label) for path, label in labels.items() if path in embeddings_by_path and label in (0, 1)
        ]
        if len(labeled_pairs) < 6:
            return None
        X = np.vstack([embeddings_by_path[path] for path, _ in labeled_pairs]).astype(np.float32)
        y = np.asarray([label for _, label in labeled_pairs], dtype=np.int64)
        class_counts = np.bincount(y, minlength=2)
        if class_counts.min() < 3:
            return None
        test_size = max(2, int(round(len(y) * 0.25)))
        if len(y) - test_size < 4:
            return None
        try:
            train_index, test_index = stratified_split(y, test_size)
            y_train = y[train_index]
            y_test = y[test_index]
            if len(set(y_train.tolist())) < 2 or len(set(y_test.tolist())) < 2:
                return None
            estimator = LogisticRegression().fit(X[train_index], y_train)
            return float(roc_auc(y_test, estimator.predict_proba(X[test_index])[:, 1]))
        except Exception:  # noqa: BLE001
            return None

    def _fit_classifier(self, X: np.ndarray, y: np.ndarray):
        base_classifier = LogisticRegression()
        try:
            class_counts = np.bincount(y, minlength=2)
            if class_counts.min() < 3 or len(y) < 8:
                base_classifier.fit(X, y)
                return base_classifier
            test_size = max(2, int(round(len(y) * 0.25)))
            if len(y) - test_size < 4:
                base_classifier.fit(X, y)
                return base_classifier
            train_index, calibration_index = stratified_split(y, test_size)
            y_train = y[train_index]
            y_calibration = y[calibration_index]
            if len(set(y_train.tolist())) < 2 or len(set(y_calibration.tolist())) < 2:
                base_classifier.fit(X, y)
                return base_classifier
            base_classifier.fit(X[train_index], y_train)
            return PlattCalibrator(model=base_classifier).fit(X[calibration_index], y_calibration)
        except Exception:  # noqa: BLE001
            base_classifier.fit(X, y)
            return base_classifier

    def score_prompt_similarity(
        self,
        embeddings: np.ndarray,
        positive_prompts: Sequence[str | tuple[str, float]],
        negative_prompts: Sequence[str | tuple[str, float]] | None = None,
        scale: float | None = None,
    ) -> np.ndarray:
        positive_texts, positive_weights = self._normalize_prompts(positive_prompts)
        if negative_prompts:
            negative_texts, negative_weights = self._normalize_prompts(negative_prompts)
        else:
            negative_texts = []
            negative_weights = np.asarray([], dtype=np.float32)
        positive = self.backend.embed_texts(positive_texts)
        if negative_texts:
            negative = self.backend.embed_texts(negative_texts)
        else:
            negative = np.empty((0, self.backend.image_embedding_dim), dtype=np.float32)
        axis = AxisEmbeddings(
            positive=positive,
            negative=negative,
            aggregation="weighted_mean",
            positive_weights=positive_weights,
            negative_weights=negative_weights,
        )
        previous_scale = self.zero_shot_scale
        try:
            if scale is not None:
                self.zero_shot_scale = float(scale)
            return self._axis_zero_shot_scores(embeddings, axis)
        finally:
            self.zero_shot_scale = previous_scale

    def final_scores(self, embeddings: np.ndarray, axis_scores: dict[str, np.ndarray] | None = None) -> np.ndarray:
        """Legacy single-embedding scoring, kept for the "legacy" pipeline setting."""
        if axis_scores is None:
            axis_scores = self.axis_zero_shot_scores(embeddings)
        zero_shot = axis_scores.get("bikini", np.empty((0,), dtype=np.float32))
        if (
            self.classifier is None
            or embeddings.size == 0
            or getattr(self.classifier, "coef", None) is None
            or int(embeddings.shape[1]) != int(self.classifier.coef.shape[0])
        ):
            return zero_shot
        classifier_scores = self.classifier.predict_proba(embeddings)[:, 1].astype(np.float32)
        return (self.classifier_weight * classifier_scores + self.zero_shot_weight * zero_shot).astype(np.float32)

    # --- cascade plumbing ---------------------------------------------------
    def full_region_table(self, embeddings: np.ndarray) -> RegionScoreTable:
        """A region table with one full-frame row per image (no deep scan)."""
        embeddings = np.asarray(embeddings, dtype=np.float32)
        count = int(embeddings.shape[0]) if embeddings.ndim == 2 else 0
        rows = np.arange(count, dtype=np.int64)
        return RegionScoreTable(
            owner=rows,
            kinds=np.array([KIND_FULL] * count, dtype=object),
            axis_scores=self.axis_zero_shot_scores(embeddings),
            image_count=count,
            full_row=rows,
        )

    def build_region_table(
        self,
        row_embeddings: np.ndarray,
        owner: Sequence[int],
        region_keys: Sequence[str],
        image_count: int,
    ) -> RegionScoreTable:
        """Score every (image, region) row in one pass and index it by image."""
        row_embeddings = np.asarray(row_embeddings, dtype=np.float32)
        owner_array = np.asarray(owner, dtype=np.int64)
        kinds = np.array([region_kind(key) for key in region_keys], dtype=object)
        full_row = np.zeros((image_count,), dtype=np.int64)
        for row, (image_index, key) in enumerate(zip(owner_array, region_keys, strict=False)):
            if key == FULL_REGION:
                full_row[int(image_index)] = row
        return RegionScoreTable(
            owner=owner_array,
            kinds=kinds,
            axis_scores=self.axis_zero_shot_scores(row_embeddings),
            image_count=int(image_count),
            full_row=full_row,
        )

    def build_features(
        self,
        embeddings: np.ndarray,
        detail_embeddings: np.ndarray | None,
        axis_scores: dict[str, np.ndarray],
    ) -> np.ndarray:
        """Feature vector the learned model trains on.

        Whole-image embedding, best body-crop embedding, and the axis/gate scores. The
        crop half is what lets a label like "yes, this one" teach the model about the
        swimwear rather than about the beach behind it.
        """
        embeddings = np.asarray(embeddings, dtype=np.float32)
        if embeddings.ndim != 2 or embeddings.shape[0] == 0:
            return np.empty((0, 0), dtype=np.float32)
        count = int(embeddings.shape[0])
        if detail_embeddings is None or len(detail_embeddings) != count:
            detail = embeddings
        else:
            detail = np.asarray(detail_embeddings, dtype=np.float32)
        axis_block = np.column_stack(
            [
                np.asarray(axis_scores.get(axis, np.full((count,), 0.5, dtype=np.float32)), dtype=np.float32)
                for axis in FEATURE_AXIS_ORDER
            ]
        )
        return np.hstack([embeddings, detail, axis_block]).astype(np.float32)

    def _global_store(self) -> GlobalLearningStore | None:
        if not self.config.global_learning:
            return None
        if self._global_cache is None:
            try:
                self._global_cache = GlobalLearningStore(model_name=self.config.model_name)
            except Exception:
                LOGGER.exception("Global learning memory unavailable; using folder labels only")
                return None
        return self._global_cache

    def learn(
        self,
        paths: Sequence[str],
        features: np.ndarray,
        labels: dict[str, int],
    ) -> learning.LearningOutcome:
        """Train on this folder's labels plus everything learned in other folders."""
        features = np.asarray(features, dtype=np.float32)
        if features.size == 0:
            self.learning_outcome = learning.LearningOutcome()
            return self.learning_outcome

        local_rows: list[np.ndarray] = []
        local_labels: list[int] = []
        local_paths: list[str] = []
        for index, path in enumerate(paths):
            value = labels.get(str(path))
            if value in (0, 1):
                local_rows.append(features[index])
                local_labels.append(int(value))
                local_paths.append(str(path))

        store = self._global_store()
        if store is not None:
            signature = hashlib.sha1(
                json.dumps(sorted(zip(local_paths, local_labels, strict=False)), separators=(",", ":")).encode("utf-8")
            ).hexdigest()
            if signature != self._global_signature:
                try:
                    store.record(
                        zip(local_paths, local_labels, local_rows, strict=False),
                        sequence=int(datetime.now(timezone.utc).timestamp()),
                    )
                    # A cleared label must stop teaching the model.
                    store.forget(str(path) for path in paths if labels.get(str(path)) not in (0, 1))
                except Exception:
                    LOGGER.exception("Could not update global learning memory")
                self._global_signature = signature

        train_rows = list(local_rows)
        train_labels = list(local_labels)
        if store is not None:
            known = set(local_paths)
            pooled = store.training_set(expected_dim=int(features.shape[1]))
            for row, label, path in zip(pooled.features, pooled.labels, pooled.paths, strict=False):
                if str(path) in known:
                    continue
                train_rows.append(np.asarray(row, dtype=np.float32))
                train_labels.append(int(label))

        if not train_rows:
            self.learning_outcome = learning.LearningOutcome()
            return self.learning_outcome
        outcome = learning.fit(
            np.vstack(train_rows).astype(np.float32),
            np.asarray(train_labels, dtype=np.int64),
            max_weight=float(self.config.max_learning_weight),
        )
        self.learning_outcome = outcome
        self.classifier = outcome.classifier
        LOGGER.info("Learning: %s", outcome.summary())
        return outcome

    def visibility_mask(
        self,
        axis_scores: dict[str, np.ndarray],
        face_counts: np.ndarray | None = None,
        excluded: np.ndarray | None = None,
    ) -> np.ndarray:
        bikini_scores = axis_scores.get("bikini")
        if bikini_scores is None:
            return np.empty((0,), dtype=bool)
        mask = np.ones_like(bikini_scores, dtype=bool)
        nsfw_scores = axis_scores.get("nsfw")
        if nsfw_scores is not None:
            if self.config.nsfw_filter == "exclude":
                mask &= nsfw_scores < float(self.config.nsfw_threshold)
            elif self.config.nsfw_filter == "only":
                mask &= nsfw_scores >= float(self.config.nsfw_threshold)
        person_scores = axis_scores.get("person")
        if self.config.require_person and person_scores is not None:
            # The cascade uses evidence-space values, but person_scores are sigmoid outputs.
            # Compare in evidence space so the same threshold maps to the same gate.
            mask &= logit(np.asarray(person_scores, dtype=np.float32)) >= logit(
                float(self.config.person_threshold)
            )
        if excluded is not None and len(excluded) == len(mask):
            mask &= ~np.asarray(excluded, dtype=bool)
        return mask

    def state_visibility(self, state: ScoreState) -> np.ndarray:
        """Visibility for a scored state, including the cascade's own exclusions."""
        return self.visibility_mask(state.axis_scores, state.face_counts, state.excluded)

    def score_state(
        self,
        paths: Sequence[str],
        embeddings: np.ndarray,
        labels: dict[str, int],
        face_counts: np.ndarray | None = None,
        scan_timestamp: str | None = None,
        store: FolderStore | None = None,
        region_table: RegionScoreTable | None = None,
        detail_embeddings: np.ndarray | None = None,
        deep_scanned: int = 0,
        detail_regions: Sequence[str] | None = None,
        refine: RefineResult | None = None,
    ) -> ScoreState:
        embeddings = np.asarray(embeddings, dtype=np.float32)
        if face_counts is not None:
            face_counts = np.asarray(face_counts, dtype=np.int32)
        count = len(list(paths))

        if self.config.pipeline == "legacy":
            embeddings_by_path = dict(zip(paths, embeddings, strict=False))
            label_count = self.train_classifier(embeddings_by_path, labels, store=store)
            axis_scores = self.axis_zero_shot_scores(embeddings)
            scores = self.final_scores(embeddings, axis_scores=axis_scores)
            return ScoreState(
                paths=list(paths),
                embeddings=embeddings,
                zero_shot_scores=axis_scores.get("bikini", np.empty((0,), dtype=np.float32)),
                scores=scores,
                axis_scores=axis_scores,
                face_counts=face_counts,
                classifier_trained=self.classifier is not None,
                classifier_label_count=label_count,
                scan_timestamp=scan_timestamp or datetime.now(timezone.utc).isoformat(),
            )

        if region_table is None:
            region_table = self.full_region_table(embeddings)
        result = cascade_module.evaluate(region_table, self.config, face_counts)
        axis_scores = result.axis_scores
        zero_shot = np.asarray(result.score, dtype=np.float32)

        if refine is not None and refine.scores.shape[0] == count:
            refined_rows = np.isfinite(refine.scores)
            if refined_rows.any():
                # The larger model gets the louder vote, but not the only one. The
                # legacy CLIP refine path uses refine_weight (default 0.65); the VLM
                # adjudication path uses vlm_weight. They are separate knobs now.
                zero_shot = zero_shot.copy()
                refine_weight = float(self.config.refine_weight)
                refine_weight = float(np.clip(refine_weight, 0.0, 1.0))
                zero_shot[refined_rows] = (
                    (1.0 - refine_weight) * zero_shot[refined_rows] + refine_weight * refine.scores[refined_rows]
                ).astype(np.float32)
            for index in np.nonzero(refine.minor)[0]:
                position = int(index)
                result.stage[position] = cascade_module.STAGE_MINOR
                result.reason[position] = cascade_module.STAGE_REASONS[cascade_module.STAGE_MINOR]
                result.excluded[position] = True

        features = self.build_features(embeddings, detail_embeddings, axis_scores)
        outcome = self.learn(list(paths), features, labels)
        learned = outcome.score(features) if outcome.trained else None
        scores = learning.blend(zero_shot, learned, outcome.weight)
        # No amount of learned confidence may resurrect an age-gated image.
        if result.stage:
            minor_rows = np.asarray([stage == cascade_module.STAGE_MINOR for stage in result.stage], dtype=bool)
            if minor_rows.any():
                scores = np.asarray(scores, dtype=np.float32)
                scores[minor_rows] = 0.0

        return ScoreState(
            paths=list(paths),
            embeddings=embeddings,
            zero_shot_scores=zero_shot,
            scores=np.asarray(scores, dtype=np.float32),
            axis_scores=axis_scores,
            face_counts=face_counts,
            classifier_trained=outcome.classifier is not None,
            classifier_label_count=outcome.label_count,
            scan_timestamp=scan_timestamp or datetime.now(timezone.utc).isoformat(),
            detail_embeddings=detail_embeddings if detail_embeddings is not None else embeddings,
            region_table=region_table,
            cascade_stage=list(result.stage),
            cascade_reason=list(result.reason),
            excluded=result.excluded,
            features=features,
            learning_summary=outcome.summary(),
            deep_scanned=int(deep_scanned)
            or (0 if region_table is None else int(max(0, region_table.owner.size - count))),
            detail_regions=list(detail_regions) if detail_regions else [FULL_REGION] * count,
            refine=refine,
        )

    def rescore_state(
        self,
        state: ScoreState,
        labels: dict[str, int],
        threshold: float = 0.5,
        store: FolderStore | None = None,
        cancel_event: threading.Event | None = None,
    ) -> tuple[ScoreState, list[dict[str, object]]]:
        if cancel_event is not None and cancel_event.is_set():
            raise ScanCancelled
        # Re-uses the cached region scores, so relabelling never re-embeds anything.
        new_state = self.score_state(
            state.paths,
            state.embeddings,
            labels,
            face_counts=state.face_counts,
            store=store,
            region_table=state.region_table,
            detail_embeddings=state.detail_embeddings,
            deep_scanned=state.deep_scanned,
            detail_regions=state.detail_regions,
            refine=state.refine,
        )
        if cancel_event is not None and cancel_event.is_set():
            raise ScanCancelled
        visible_mask = self.state_visibility(new_state)
        samples = bucketed_sampling(
            [path for path, include in zip(new_state.paths, visible_mask, strict=False) if include],
            [score for score, include in zip(new_state.scores, visible_mask, strict=False) if include],
            labels.keys(),
            embeddings=[
                embedding for embedding, include in zip(new_state.embeddings, visible_mask, strict=False) if include
            ],
            threshold=threshold,
            disagreement=state_disagreement(new_state, visible_mask),
        )
        return new_state, samples


def state_disagreement(state: ScoreState, visible_mask: np.ndarray) -> list[float]:
    """Per visible image, how far the learned model moved the prompt-only score.

    Public because the GUI rebuilds the review queue too — without this the
    "Model disagrees" bucket vanished the moment a filter or sort changed.
    """
    scores = np.asarray(state.scores, dtype=np.float32)
    zero_shot = np.asarray(state.zero_shot_scores, dtype=np.float32)
    if scores.shape != zero_shot.shape:
        return []
    gaps = np.abs(scores - zero_shot)
    return [float(gap) for gap, include in zip(gaps, visible_mask, strict=False) if include]


@dataclass(slots=True)
class DeepPassResult:
    region_table: RegionScoreTable
    detail_embeddings: np.ndarray
    face_counts: np.ndarray | None
    deep_scanned: int
    detail_regions: list[str] = field(default_factory=list)


def _region_namespace(config: ScannerConfig) -> str:
    slug = re.sub(r"[^A-Za-z0-9]+", "_", str(config.model_name or "default")).strip("_").lower()
    return f"{slug}__g{REGION_GEOMETRY_VERSION}"


def _candidate_mask(
    base_table: RegionScoreTable,
    config: ScannerConfig,
    face_counts: np.ndarray | None,
) -> np.ndarray:
    """Which images earn the expensive crop pass.

    Anything with a person, a detected face, or a near-miss person score. Images the
    whole-frame pass thinks might contain a minor are deliberately *included*: the face
    crops are what turn that guess into a reliable decision.
    """
    count = base_table.image_count
    if config.deep_scan == "always":
        return np.ones((count,), dtype=bool)
    person = cascade_module.evidence(base_table.aggregate("person"))
    mask = person >= max(0.0, float(config.person_gate_threshold) - 0.15)
    child = cascade_module.evidence(base_table.aggregate("child"))
    mask |= child >= float(config.minor_threshold) * 0.7
    if face_counts is not None and len(face_counts) == count:
        mask |= np.asarray(face_counts, dtype=np.int32) > 0
    return mask


def run_deep_pass(
    backend: ImageEmbeddingBackend,
    scorer: BikiniScorer,
    store: FolderStore | None,
    paths: Sequence[str],
    embeddings: np.ndarray,
    face_counts: np.ndarray | None = None,
    cancel_event: threading.Event | None = None,
    on_progress: Callable[[int, int], None] | None = None,
    content_hashes: Sequence[str | None] | None = None,
) -> DeepPassResult:
    """Embed face and body-region crops for candidate images.

    Region embeddings are cached per (content hash, model, geometry version), so a
    rescan of the same folder skips this entirely.
    """
    embeddings = np.asarray(embeddings, dtype=np.float32)
    base_table = scorer.full_region_table(embeddings)
    count = len(list(paths))
    detail_embeddings = embeddings.copy()
    detail_regions = [FULL_REGION] * count
    if scorer.config.deep_scan == "off" or count == 0:
        return DeepPassResult(base_table, detail_embeddings, face_counts, 0, detail_regions)

    candidates = _candidate_mask(base_table, scorer.config, face_counts)
    candidate_indices = [int(index) for index in np.nonzero(candidates)[0]]
    LOGGER.info("Deep pass: %d/%d images qualify for region crops", len(candidate_indices), count)
    if on_progress is not None:
        # Announce the total before any work, so a caller driving a progress bar can size
        # this phase even when nothing qualifies.
        on_progress(0, len(candidate_indices))

    namespace = _region_namespace(scorer.config)
    owner: list[int] = list(range(count))
    region_keys: list[str] = [FULL_REGION] * count
    row_embeddings: list[np.ndarray] = [embeddings[index] for index in range(count)]
    pending_cache: dict[str, np.ndarray] = {}
    updated_faces = (
        np.asarray(face_counts, dtype=np.int32).copy()
        if face_counts is not None and len(face_counts) == count
        else np.full((count,), -1, dtype=np.int32)
    )
    # Row index of each image's candidate detail regions, resolved after scoring.
    detail_rows: dict[int, list[int]] = {}
    processed = 0

    for index in candidate_indices:
        if cancel_event is not None and cancel_event.is_set():
            raise ScanCancelled
        path = str(paths[index])
        content_hash = None
        if content_hashes is not None and index < len(content_hashes):
            content_hash = content_hashes[index]
        cached: dict[str, np.ndarray] = {}
        if store is not None and content_hash:
            cached = store.lookup_region_embeddings(str(content_hash), namespace)
        crops: list[tuple[str, np.ndarray]] = []
        if cached:
            # "__faces__" is a sidecar entry, not a region: it must never become a row.
            crops = [(key, value) for key, value in cached.items() if key != FULL_REGION and not key.startswith("__")]
            face_marker = cached.get("__faces__")
            if face_marker is not None and np.asarray(face_marker).size:
                updated_faces[index] = int(np.asarray(face_marker).ravel()[0])
        else:
            try:
                # Full resolution on purpose: this pass crops before the model sees
                # anything, so a scaled decode would throw away the very detail the
                # crops exist to recover.
                image = open_oriented(path)
                faces = detect_face_boxes(image)
                updated_faces[index] = len(faces)
                planned = [
                    region
                    for region in plan_regions(image.size, faces, max_faces=int(scorer.config.max_faces))
                    if region.key != FULL_REGION
                ]
                materialised = crop_regions(image, planned)
                if materialised:
                    vectors = backend.embed_pil_images([crop for _, crop in materialised])
                    crops = [
                        (key, np.asarray(vector, dtype=np.float32))
                        for (key, _), vector in zip(materialised, vectors, strict=False)
                    ]
            except Exception as exc:  # noqa: BLE001
                LOGGER.warning("Deep pass skipped %s: %s", path, exc)
                crops = []
            if store is not None and content_hash:
                for key, vector in crops:
                    pending_cache[store.region_cache_key(str(content_hash), namespace, key)] = vector
                pending_cache[store.region_cache_key(str(content_hash), namespace, "__faces__")] = np.asarray(
                    [max(int(updated_faces[index]), 0)], dtype=np.float32
                )

        for key, vector in crops:
            if vector is None or np.asarray(vector).size == 0:
                continue
            owner.append(index)
            region_keys.append(key)
            row_embeddings.append(np.asarray(vector, dtype=np.float32))
            if region_kind(key) != KIND_FACE:
                detail_rows.setdefault(index, []).append(len(row_embeddings) - 1)

        processed += 1
        if on_progress is not None:
            on_progress(processed, len(candidate_indices))
        if store is not None and len(pending_cache) >= 256:
            store.save_region_embeddings(pending_cache)
            pending_cache = {}

    if store is not None and pending_cache:
        store.save_region_embeddings(pending_cache)

    matrix = np.vstack(row_embeddings).astype(np.float32)
    table = scorer.build_region_table(matrix, owner, region_keys, count)

    # Keep, per image, the crop with the strongest detail evidence: that is the view the
    # learned model should train on. Each row only votes on the axes its position allows,
    # so a bottom-of-frame band cannot win the slot by "detecting cleavage".
    row_detail = cascade_module.combine_detail_rows(table.axis_scores, scorer.config.detail_weights, list(table.kinds))
    if row_detail.size == matrix.shape[0]:
        for index, rows in detail_rows.items():
            if not rows:
                continue
            best_row = max(rows, key=lambda row: float(row_detail[row]))
            if float(row_detail[best_row]) > float(row_detail[index]):
                detail_embeddings[index] = matrix[best_row]
                detail_regions[index] = region_keys[best_row]

    resolved_faces: np.ndarray | None = None
    if (updated_faces >= 0).any():
        resolved_faces = updated_faces
    elif face_counts is not None:
        resolved_faces = np.asarray(face_counts, dtype=np.int32)
    return DeepPassResult(table, detail_embeddings, resolved_faces, len(candidate_indices), detail_regions)


@dataclass(slots=True)
class RefineResult:
    """Second-opinion scores, NaN where an image was not re-scored."""

    scores: np.ndarray
    minor: np.ndarray

    def any(self) -> bool:
        return bool(np.isfinite(self.scores).any())


def compute_vlm_scores(
    scorer: BikiniScorer,
    state: ScoreState,
    threshold: float = 0.5,
    cancel_event: threading.Event | None = None,
    on_progress: Callable[[int, int], None] | None = None,
    store: FolderStore | None = None,
) -> RefineResult | None:
    """Adjudicate borderline images with a local vision model.

    Only images near the threshold, plus the ones whose age reading is unsettled, are
    sent. Skin exposure orders that band so the most likely matches are judged first;
    it never removes an image from the band, though the `vlm_max_images` cap can.
    """
    if not scorer.config.vlm_enabled or scorer.config.pipeline == "legacy" or not state.paths:
        return None
    limit = int(scorer.config.vlm_max_images)
    if limit <= 0:
        return None
    from .vlm_backend import VLM_AXES, VLM_PROMPT_VERSION, VLMCancelled, VLMClient

    client = VLMClient(
        scorer.config.vlm_base_url,
        scorer.config.vlm_model,
        scorer.config.vlm_api_key,
        scorer.config.vlm_timeout,
        scorer.config.vlm_concurrency,
    )
    if not client.probe():
        return None
    band = float(scorer.config.vlm_band)
    child = cascade_module.evidence(
        np.asarray(state.axis_scores.get("child", np.full(len(state.paths), 0.5)), dtype=np.float32)
    )
    adult = cascade_module.evidence(
        np.asarray(state.axis_scores.get("adult", np.full(len(state.paths), 0.5)), dtype=np.float32)
    )
    age_margin = max(0.05, band * 0.65)
    candidates: list[tuple[float, int]] = []
    for index, score in enumerate(np.asarray(state.scores, dtype=np.float32)):
        if state.excluded is not None and bool(state.excluded[index]):
            continue
        distance = abs(float(score) - float(threshold))
        age_uncertain = (
            abs(float(child[index]) - float(scorer.config.minor_threshold)) <= age_margin
            or abs(float(adult[index]) - float(scorer.config.min_adult_confidence)) <= age_margin
        )
        if distance <= band or age_uncertain:
            candidates.append((distance, index))
    if not candidates:
        return None
    detail_regions = state.detail_regions or [FULL_REGION] * len(state.paths)
    ranked: list[tuple[float, int, list[Image.Image]]] = []
    for distance, index in candidates:
        try:
            image = open_oriented(state.paths[index])
            views = [image]
            wanted = detail_regions[index] if index < len(detail_regions) else FULL_REGION
            if wanted != FULL_REGION:
                faces = detect_face_boxes(image)
                planned = {
                    region.key: region
                    for region in plan_regions(image.size, faces, max_faces=int(scorer.config.max_faces))
                }
                region = planned.get(wanted)
                if region is not None and region.box is not None:
                    views.append(image.crop(region.box))
            skin = skin_fraction(views[-1])
            # Skin only breaks ties within the eligible band; it never filters.
            priority = 0 if distance <= band else 1
            normalized = distance / max(band, 1e-6)
            ranked.append((priority * 10.0 + normalized - 0.5 * skin, index, views))
        except (OSError, ValueError) as exc:
            LOGGER.warning("VLM pass skipped %s: %s", state.paths[index], exc)
    ranked.sort(key=lambda item: item[0])
    ranked = ranked[:limit]
    if not ranked:
        return None
    cached: dict[int, dict[str, float]] = {}
    pending: dict[str, dict[str, float]] = {}
    uncached_images: list[list[Image.Image]] = []
    uncached_positions: list[int] = []
    # content_hash_for_path reads and SHA-1s the whole file. It was being called twice
    # per candidate (once for the cache lookup, once for the save), which doubles the
    # disk I/O for every VLM judgment. Compute it once per candidate and reuse it.
    position_hashes: dict[int, str | None] = {}
    for position, (_, index, views) in enumerate(ranked):
        content_hash = store.content_hash_for_path(state.paths[index]) if store is not None else None
        position_hashes[position] = content_hash
        if content_hash and store is not None:
            key = store.vlm_cache_key(content_hash, scorer.config.vlm_model, VLM_PROMPT_VERSION)
            verdict = store.lookup_vlm_verdict(key)
            if verdict is not None:
                cached[position] = verdict
                continue
        uncached_images.append(views)
        uncached_positions.append(position)
    if uncached_images:
        try:
            responses = client.score_images(uncached_images, cancel_event=cancel_event, on_progress=on_progress)
        except VLMCancelled:
            # A cancellation is the user's own signal, not an error worth chaining.
            raise ScanCancelled from None
        for position, response in zip(uncached_positions, responses, strict=False):
            if response is None:
                continue
            cached[position] = response
            content_hash = position_hashes.get(position)
            if content_hash and store is not None:
                key = store.vlm_cache_key(content_hash, scorer.config.vlm_model, VLM_PROMPT_VERSION)
                pending[key] = response
    if store is not None and pending:
        store.save_vlm_verdicts(pending)
    if on_progress is not None and not uncached_images:
        on_progress(len(ranked), len(ranked))
    scores = np.full((len(state.paths),), np.nan, dtype=np.float32)
    minor = np.zeros((len(state.paths),), dtype=bool)
    config = scorer.config
    for position, (_, index, _views) in enumerate(ranked):
        values = cached.get(position)
        if values is None:
            continue
        matrix = np.asarray([[values.get(axis, 0.5) for axis in VLM_AXES]], dtype=np.float32)
        axis_scores = {axis: matrix[:, offset] for offset, axis in enumerate(VLM_AXES)}
        table = RegionScoreTable(
            owner=np.array([0], dtype=np.int64),
            kinds=np.array([KIND_FULL], dtype=object),
            axis_scores=axis_scores,
            image_count=1,
            full_row=np.array([0], dtype=np.int64),
        )
        face_count = None
        if state.face_counts is not None and index < len(state.face_counts):
            face_count = np.asarray([state.face_counts[index]], dtype=np.int32)
        result = cascade_module.evaluate(table, config, face_count)
        if result.score.size:
            scores[index] = float(result.score[0])
            minor[index] = bool(result.stage and result.stage[0] == cascade_module.STAGE_MINOR)
    return RefineResult(scores=scores, minor=minor)


def compute_refine_scores(
    scorer: BikiniScorer,
    state: ScoreState,
    threshold: float = 0.5,
    cancel_event: threading.Event | None = None,
    on_progress: Callable[[int, int], None] | None = None,
) -> RefineResult | None:
    """Second opinion from a larger model on the images closest to the threshold.

    Off unless a refine model is configured. Only borderline, non-excluded images are
    re-scored, and only on two views each (the full frame and its best body crop), which
    is what keeps a much slower model affordable.
    """
    model_name = str(scorer.config.refine_model or "").strip()
    if not model_name or scorer.config.pipeline == "legacy" or not state.paths:
        return None
    limit = int(scorer.config.refine_max_images)
    if limit <= 0:
        return None

    band = float(scorer.config.refine_band)
    distances: list[tuple[float, int]] = []
    for index, score in enumerate(np.asarray(state.scores, dtype=np.float32)):
        if state.excluded is not None and bool(state.excluded[index]):
            continue
        distance = abs(float(score) - float(threshold))
        if distance <= band:
            distances.append((distance, index))
    if not distances:
        return None
    distances.sort()
    targets = [index for _, index in distances[:limit]]
    LOGGER.info("Refine pass: re-scoring %d borderline images with %s", len(targets), model_name)

    try:
        from dataclasses import replace as dataclass_replace

        from .clip_backend import get_backend

        refine_config = dataclass_replace(scorer.config, model_name=model_name, refine_model="")
        refine_backend = get_backend(refine_config)
        refine_scorer = BikiniScorer(backend=refine_backend, config=refine_config)
    except Exception:
        LOGGER.exception("Refine model %s could not be loaded; keeping the base scores", model_name)
        return None

    detail_regions = state.detail_regions or [FULL_REGION] * len(state.paths)
    refine_scores = np.full((len(state.paths),), np.nan, dtype=np.float32)
    refine_minor = np.zeros((len(state.paths),), dtype=bool)
    done = 0
    for index in targets:
        if cancel_event is not None and cancel_event.is_set():
            raise ScanCancelled
        path = str(state.paths[index])
        try:
            image = open_oriented(path)
            views: list[tuple[str, Image.Image]] = [(FULL_REGION, image)]
            wanted = detail_regions[index] if index < len(detail_regions) else FULL_REGION
            if wanted != FULL_REGION:
                faces = detect_face_boxes(image)
                planned = {
                    region.key: region
                    for region in plan_regions(image.size, faces, max_faces=int(scorer.config.max_faces))
                }
                region = planned.get(wanted)
                if region is not None and region.box is not None:
                    views.append((wanted, image.crop(region.box)))
            vectors = refine_backend.embed_pil_images([view for _, view in views])
        except Exception as exc:  # noqa: BLE001
            LOGGER.warning("Refine pass skipped %s: %s", path, exc)
            continue
        if vectors is None or len(vectors) == 0:
            continue
        table = refine_scorer.build_region_table(
            np.asarray(vectors, dtype=np.float32),
            owner=[0] * len(views),
            region_keys=[key for key, _ in views],
            image_count=1,
        )
        face_count = None
        if state.face_counts is not None and index < len(state.face_counts):
            face_count = np.asarray([state.face_counts[index]], dtype=np.int32)
        result = cascade_module.evaluate(table, refine_config, face_count)
        if result.score.size:
            refine_scores[index] = float(result.score[0])
            if result.stage and result.stage[0] == cascade_module.STAGE_MINOR:
                refine_minor[index] = True
        done += 1
        if on_progress is not None:
            on_progress(done, len(targets))

    return RefineResult(scores=refine_scores, minor=refine_minor)


def bucketed_sampling(
    paths: Sequence[str],
    scores: Sequence[float],
    labeled_paths: Iterable[str],
    embeddings: Sequence[Sequence[float]] | np.ndarray | None = None,
    threshold: float = 0.5,
    per_bucket: int = 6,
    margin: float = 0.15,
    disagreement: Sequence[float] | None = None,
) -> list[dict[str, object]]:
    """Choose what to put in front of the reviewer next.

    Ordinary score bands plus, when the learned model is running, the images where it
    most disagrees with the prompts. Those are worth more than another easy example:
    labelling one resolves a conflict instead of confirming what is already known.
    """
    labeled_set = {str(path) for path in labeled_paths}
    disagreement_by_index = (
        {index: float(value) for index, value in enumerate(disagreement)} if disagreement is not None else {}
    )
    scored = [
        {"path": str(path), "score": float(score), "index": index}
        for index, (path, score) in enumerate(zip(paths, scores, strict=False))
        if str(path) not in labeled_set
    ]

    used_paths: set[str] = set()
    embedding_array = None if embeddings is None else np.asarray(embeddings, dtype=np.float32)
    if embedding_array is not None and embedding_array.shape[0] != len(paths):
        embedding_array = None

    def _cosine_distance(a: np.ndarray, b: np.ndarray) -> float:
        a_norm = float(np.linalg.norm(a))
        b_norm = float(np.linalg.norm(b))
        if a_norm <= 0 or b_norm <= 0:
            return 0.0
        similarity = float(np.dot(a, b) / (a_norm * b_norm))
        return 1.0 - max(-1.0, min(1.0, similarity))

    def _take(
        items: list[dict[str, object]], bucket_name: str, sort_key, reverse: bool = False
    ) -> list[dict[str, object]]:
        ranked = sorted(items, key=sort_key, reverse=reverse)
        selected: list[dict[str, object]] = []
        selected_vectors: list[np.ndarray] = []
        if embedding_array is None:
            for item in ranked:
                path = str(item["path"])
                if path in used_paths:
                    continue
                used_paths.add(path)
                selected.append({k: v for k, v in item.items() if k != "index"} | {"bucket": bucket_name})
                if len(selected) >= per_bucket:
                    break
            return selected

        candidates = [item for item in ranked if str(item["path"]) not in used_paths]
        while candidates and len(selected) < per_bucket:
            if not selected_vectors:
                best_index = 0
            else:
                best_index = max(
                    range(len(candidates)),
                    key=lambda idx: (
                        min(
                            _cosine_distance(embedding_array[int(candidates[idx]["index"])], vector)
                            for vector in selected_vectors
                        ),
                        -abs(float(candidates[idx]["score"]) - threshold),
                        float(candidates[idx]["score"]),
                    ),
                )
            item = candidates.pop(best_index)
            path = str(item["path"])
            if path in used_paths:
                continue
            used_paths.add(path)
            selected.append({k: v for k, v in item.items() if k != "index"} | {"bucket": bucket_name})
            selected_vectors.append(embedding_array[int(item["index"])])
        return selected

    contested = _take(
        [item for item in scored if disagreement_by_index.get(int(item["index"]), 0.0) >= 0.08],
        "Model disagrees",
        sort_key=lambda item: disagreement_by_index.get(int(item["index"]), 0.0),
        reverse=True,
    )
    likely_good = _take(
        [item for item in scored if item["score"] >= threshold + margin],
        "Likely match",
        sort_key=lambda item: item["score"],
        reverse=True,
    )
    likely_false_positives = _take(
        [item for item in scored if threshold <= item["score"] <= threshold + margin],
        "Likely false positive",
        sort_key=lambda item: item["score"],
    )
    likely_false_negatives = _take(
        [item for item in scored if threshold - margin <= item["score"] < threshold],
        "Likely false negative",
        sort_key=lambda item: item["score"],
        reverse=True,
    )

    uncertain_candidates = [item for item in scored if str(item["path"]) not in used_paths]
    uncertain = _take(
        uncertain_candidates,
        "Uncertain",
        sort_key=lambda item: (abs(item["score"] - threshold), item["score"]),
    )

    if likely_good:
        return [*likely_good, *contested, *likely_false_positives, *likely_false_negatives, *uncertain]
    return [*contested, *uncertain, *likely_false_positives, *likely_false_negatives]


def scan_and_score_folder(
    backend: ImageEmbeddingBackend,
    store: FolderStore,
    scorer: BikiniScorer,
    threshold: float = 0.5,
    progress_callback: Callable[[int, int, float, float | None], None] | None = None,
    batch_size: int = 16,
    cancel_event: threading.Event | None = None,
) -> tuple[ScoreState, list[dict[str, object]]]:
    configure_logging()
    paths = collect_image_paths(store.folder)
    if len(paths) >= 10_000:
        LOGGER.warning("Large folder detected: %d images in %s", len(paths), store.folder)
    else:
        LOGGER.info("Starting scan of %d images in %s", len(paths), store.folder)
    cached_records = store.get_cached_image_records(paths)
    content_embeddings: dict[str, np.ndarray] = {}
    content_face_counts: dict[str, int] = {}
    path_records: dict[Path, dict[str, int | str]] = {}
    image_records: list[dict[str, object]] = []
    skipped_records: list[dict[str, object]] = []
    all_embeddings: list[np.ndarray] = []
    ordered_paths: list[str] = []
    face_counts: list[int | None] = []
    uncached_paths = [path for path in paths if path not in cached_records]
    LOGGER.info(
        "Scan cache contains %d/%d images; %d require embedding", len(cached_records), len(paths), len(uncached_paths)
    )
    processed = 0
    scan_timestamp = datetime.now(timezone.utc).isoformat()
    runs_detail_pass = scorer.config.pipeline != "legacy" and scorer.config.deep_scan != "off"
    reporter = _ProgressReporter(progress_callback, PHASE_SHARES if runs_detail_pass else PHASE_SHARES_NO_DETAIL)
    reporter.start_phase(PHASE_EMBED, len(paths))

    def _notify() -> None:
        reporter.emit(processed)

    def _check_cancelled() -> None:
        if cancel_event is not None and cancel_event.is_set():
            store.save_scan_cache(content_embeddings, path_records, content_face_counts)
            LOGGER.info("Scan cancellation requested after %d processed images", processed)
            raise ScanCancelled

    def _record_vanished(path: Path) -> None:
        """A file listed at the start of the scan is no longer there."""
        LOGGER.warning("Skipping %s: it disappeared during the scan", path)
        skipped_records.append(
            {
                "filename": path.name,
                "path": str(path),
                "error": "file disappeared during the scan",
                "timestamp": scan_timestamp,
            }
        )

    def _add_record(path: Path, embedding: np.ndarray, content_hash: str, face_count: int | None) -> bool:
        stat = safe_stat(path)
        if stat is None:
            _record_vanished(path)
            return False
        path_records[path] = {
            "mtime_ns": stat.st_mtime_ns,
            "size": stat.st_size,
            "content_hash": content_hash,
        }
        ordered_paths.append(str(path))
        all_embeddings.append(np.asarray(embedding, dtype=np.float32))
        face_counts.append(face_count)
        image_records.append(
            {
                "filename": path.name,
                "path": str(path.resolve()),
                "score": None,
                "zero_shot_score": None,
                "axis_scores": {},
                "face_count": int(face_count) if face_count is not None else None,
                "matched": None,
                "timestamp": scan_timestamp,
            }
        )
        return True

    _notify()
    for path in paths:
        _check_cancelled()
        cached = cached_records.get(path)
        if cached is None:
            continue
        embedding = np.asarray(cached["embedding"], dtype=np.float32)
        content_hash = str(cached.get("content_hash") or "")
        face_count: int | None = None
        if content_hash:
            stat = safe_stat(path)
            if stat is None:
                _record_vanished(path)
                processed += 1
                _notify()
                continue
            content_embeddings[content_hash] = embedding
            path_records[path] = {
                "mtime_ns": int(stat.st_mtime_ns),
                "size": int(stat.st_size),
                "content_hash": content_hash,
            }
            cached_face_count = store.lookup_face_count(content_hash)
            if cached_face_count is not None:
                face_count = cached_face_count
                content_face_counts[content_hash] = cached_face_count
            elif scorer.config.enable_face_detection:
                try:
                    face_count = detect_face_count(open_oriented(path))
                except Exception:  # noqa: BLE001
                    face_count = None
                if face_count is not None:
                    content_face_counts[content_hash] = face_count
        _add_record(path, embedding, content_hash, face_count)
        processed += 1
        _notify()

    pending_flush_images = 0
    for decoded_images in backend.iter_image_batches(uncached_paths, batch_size=batch_size):
        _check_cancelled()
        valid_records = [record for record in decoded_images if record.image is not None]
        hash_to_image: dict[str, Image.Image] = {}
        hash_to_records: dict[str, list[Path]] = {}
        for record in valid_records:
            assert record.content_hash is not None
            stat = safe_stat(record.path)
            if stat is None:
                _record_vanished(record.path)
                continue
            cached_embedding = content_embeddings.get(record.content_hash)
            if cached_embedding is None:
                cached_embedding = store.lookup_content_embedding(record.content_hash)
            face_count: int | None = content_face_counts.get(record.content_hash)
            if face_count is None and scorer.config.enable_face_detection:
                face_count = detect_face_count(record.image)
                if face_count is not None:
                    content_face_counts[record.content_hash] = face_count
            if cached_embedding is not None:
                content_embeddings.setdefault(record.content_hash, cached_embedding)
                path_records[record.path] = {
                    "mtime_ns": stat.st_mtime_ns,
                    "size": stat.st_size,
                    "content_hash": record.content_hash,
                }
                ordered_paths.append(str(record.path))
                all_embeddings.append(np.asarray(cached_embedding, dtype=np.float32))
                face_counts.append(face_count)
                image_records.append(
                    {
                        "filename": record.path.name,
                        "path": str(record.path.resolve()),
                        "score": None,
                        "zero_shot_score": None,
                        "axis_scores": {},
                        "face_count": int(face_count) if face_count is not None else None,
                        "matched": None,
                        "timestamp": scan_timestamp,
                    }
                )
                continue
            if record.content_hash not in hash_to_image:
                hash_to_image[record.content_hash] = record.image
                hash_to_records[record.content_hash] = []
            hash_to_records[record.content_hash].append(record.path)
        if hash_to_image:
            _check_cancelled()
            batch_embeddings = backend.embed_pil_images(list(hash_to_image.values()))
            for content_hash, embedding in zip(hash_to_image.keys(), batch_embeddings, strict=False):
                content_embeddings[content_hash] = embedding
                if scorer.config.enable_face_detection and content_hash not in content_face_counts:
                    first_path = hash_to_records.get(content_hash, [None])[0]
                    face_count = None
                    if first_path is not None:
                        try:
                            face_count = detect_face_count(open_oriented(first_path))
                        except Exception:  # noqa: BLE001
                            face_count = None
                    if face_count is not None:
                        content_face_counts[content_hash] = face_count
                face_count = content_face_counts.get(content_hash)
                for path in hash_to_records.get(content_hash, []):
                    stat = safe_stat(path)
                    if stat is None:
                        _record_vanished(path)
                        continue
                    path_records[path] = {
                        "mtime_ns": stat.st_mtime_ns,
                        "size": stat.st_size,
                        "content_hash": content_hash,
                    }
                    ordered_paths.append(str(path))
                    all_embeddings.append(np.asarray(embedding, dtype=np.float32))
                    face_counts.append(face_count)
                    image_records.append(
                        {
                            "filename": path.name,
                            "path": str(path.resolve()),
                            "score": None,
                            "zero_shot_score": None,
                            "axis_scores": {},
                            "face_count": int(face_count) if face_count is not None else None,
                            "matched": None,
                            "timestamp": scan_timestamp,
                        }
                    )
        for record in decoded_images:
            processed += 1
            if record.image is None:
                LOGGER.warning("Skipped unreadable image %s: %s", record.path, record.error)
                skipped_records.append(
                    {
                        "filename": record.path.name,
                        "path": str(record.path.resolve()),
                        "error": record.error,
                        "timestamp": scan_timestamp,
                    }
                )
            _notify()
        pending_flush_images += len(valid_records)
        if pending_flush_images >= max(batch_size * 4, 32):
            store.save_scan_cache(content_embeddings, path_records, content_face_counts)
            LOGGER.info("Flushed scan cache after %d processed images", processed)
            pending_flush_images = 0
        _check_cancelled()
    if not all_embeddings:
        empty = np.empty((0, backend.image_embedding_dim), dtype=np.float32)
        empty_faces = np.empty((0,), dtype=np.int32)
        state = scorer.score_state(
            [],
            empty,
            store.load_labels(),
            face_counts=empty_faces,
            scan_timestamp=scan_timestamp,
            store=store,
        )
        store.save_scan_metadata(store.build_scan_metadata(image_records, skipped_records, scan_timestamp))
        reporter.finish()
        LOGGER.info("Completed empty scan of %s; %d skipped", store.folder, len(skipped_records))
        return state, []

    reporter.complete_phase()
    embeddings = np.vstack(all_embeddings).astype(np.float32)
    face_array = (
        np.asarray([int(face_count) if face_count is not None else -1 for face_count in face_counts], dtype=np.int32)
        if any(face_count is not None for face_count in face_counts)
        else None
    )
    store.save_scan_cache(content_embeddings, path_records, content_face_counts)
    labels = store.load_labels()

    # --- deep pass: face and body-region crops for the candidates -------------
    ordered_hashes = [str(path_records.get(Path(path), {}).get("content_hash") or "") or None for path in ordered_paths]
    deep_started = False

    def _deep_progress(done: int, total: int) -> None:
        # The candidate count is only known once the deep pass has planned its work, so
        # the phase opens on the first tick rather than up front.
        nonlocal deep_started
        if not deep_started:
            deep_started = True
            reporter.start_phase(PHASE_DETAIL, total)
        reporter.emit(done)

    if scorer.config.pipeline == "legacy":
        deep = DeepPassResult(scorer.full_region_table(embeddings), embeddings, face_array, 0)
    else:
        deep = run_deep_pass(
            backend,
            scorer,
            store,
            ordered_paths,
            embeddings,
            face_counts=face_array,
            cancel_event=cancel_event,
            on_progress=_deep_progress,
            content_hashes=ordered_hashes,
        )
        if deep.face_counts is not None:
            face_array = deep.face_counts
    reporter.complete_phase()

    reporter.start_phase(PHASE_SCORE, len(ordered_paths))
    state = scorer.score_state(
        ordered_paths,
        embeddings,
        labels,
        face_counts=face_array,
        scan_timestamp=scan_timestamp,
        store=store,
        region_table=deep.region_table,
        detail_embeddings=deep.detail_embeddings,
        deep_scanned=deep.deep_scanned,
        detail_regions=deep.detail_regions,
    )

    refine_started = False

    def _refine_progress(done: int, total: int) -> None:
        nonlocal refine_started
        if not refine_started:
            refine_started = True
            reporter.start_phase(PHASE_REFINE, total)
        reporter.emit(done)

    refine = None
    if scorer.config.vlm_enabled:
        refine = compute_vlm_scores(
            scorer,
            state,
            threshold=threshold,
            cancel_event=cancel_event,
            on_progress=_refine_progress,
            store=store,
        )
    if refine is None:
        refine = compute_refine_scores(
            scorer, state, threshold=threshold, cancel_event=cancel_event, on_progress=_refine_progress
        )
    if refine_started:
        reporter.complete_phase()
        reporter.start_phase(PHASE_SCORE, len(ordered_paths))
    if refine is not None and refine.any():
        state = scorer.score_state(
            ordered_paths,
            embeddings,
            labels,
            face_counts=face_array,
            scan_timestamp=scan_timestamp,
            store=store,
            region_table=deep.region_table,
            detail_embeddings=deep.detail_embeddings,
            deep_scanned=deep.deep_scanned,
            detail_regions=deep.detail_regions,
            refine=refine,
        )
    visible_mask = scorer.state_visibility(state)
    visible_paths = [path for path, include in zip(state.paths, visible_mask, strict=False) if include]
    visible_scores = [score for score, include in zip(state.scores, visible_mask, strict=False) if include]
    samples = bucketed_sampling(
        visible_paths,
        visible_scores,
        labels.keys(),
        embeddings=[embedding for embedding, include in zip(state.embeddings, visible_mask, strict=False) if include],
        threshold=threshold,
        disagreement=state_disagreement(state, visible_mask),
    )
    for idx, (image_record, score, zero_shot_score, path) in enumerate(
        zip(image_records, state.scores, state.zero_shot_scores, state.paths, strict=False)
    ):
        image_record.update(
            {
                "score": float(score),
                "zero_shot_score": float(zero_shot_score),
                "axis_scores": {
                    axis_name: float(axis_scores[idx]) for axis_name, axis_scores in state.axis_scores.items()
                },
                "face_count": int(state.face_counts[idx])
                if state.face_counts is not None and state.face_counts[idx] >= 0
                else None,
                "label_state": int(labels.get(path)) if path in labels else None,
                "matched": bool(score >= threshold and visible_mask[idx]),
                "cascade_stage": state.cascade_stage[idx] if idx < len(state.cascade_stage) else "",
                "cascade_reason": state.cascade_reason[idx] if idx < len(state.cascade_reason) else "",
                "detail_region": state.detail_regions[idx] if idx < len(state.detail_regions) else "",
            }
        )
    store.save_scan_metadata(store.build_scan_metadata(image_records, skipped_records, scan_timestamp))
    reporter.finish()
    LOGGER.info("Completed scan of %d images in %s; %d skipped", len(state.paths), store.folder, len(skipped_records))
    return state, samples
