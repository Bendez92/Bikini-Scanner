"""Cross-folder learning memory.

Per-folder caches make every new folder start from zero, so the same Accept/REJECT
decisions get made over and over. This store keeps the labelled feature vectors in the
user data directory instead, so teaching the scanner in one folder improves the next.

Entries are namespaced by (model, feature version): features from ViT-B/32 and ViT-L/14
are not comparable, and changing the feature layout invalidates old rows rather than
silently mixing them.
"""

from __future__ import annotations

import json
import logging
import pickle
import re
import threading
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Iterable, Mapping

import numpy as np

from .safe_io import atomic_replace, atomic_write_json, quarantine_broken_file
from .user_prefs import prefs_path

LOGGER = logging.getLogger(__name__)

# Bump when the feature vector layout changes; old rows are then ignored, not reused.
FEATURE_VERSION = 1
# Keeps the memory bounded. Oldest rows are dropped first.
MAX_ENTRIES = 20000

_LOCK = threading.Lock()


def global_dir() -> Path:
    return prefs_path().parent / "learning"


def _namespace(model_name: str) -> str:
    slug = re.sub(r"[^A-Za-z0-9]+", "_", str(model_name or "default")).strip("_").lower()
    return f"{slug}__v{FEATURE_VERSION}"


@dataclass(slots=True)
class TrainingSet:
    features: np.ndarray
    labels: np.ndarray
    paths: list[str]

    def __len__(self) -> int:
        return int(self.labels.shape[0])

    def class_counts(self) -> dict[int, int]:
        if self.labels.size == 0:
            return {0: 0, 1: 0}
        counts = np.bincount(self.labels.astype(np.int64), minlength=2)
        return {0: int(counts[0]), 1: int(counts[1])}


@dataclass(slots=True)
class GlobalLearningStore:
    """Labelled features and the model trained from them, shared across folders."""

    model_name: str
    root: Path = None  # type: ignore[assignment]

    def __post_init__(self) -> None:
        self.root = global_dir() / _namespace(self.model_name)
        try:
            self.root.mkdir(parents=True, exist_ok=True)
        except Exception:  # noqa: BLE001
            LOGGER.warning("Global learning directory unavailable: %s", self.root)

    @property
    def features_path(self) -> Path:
        return self.root / "features.npz"

    @property
    def index_path(self) -> Path:
        return self.root / "index.json"

    @property
    def classifier_path(self) -> Path:
        return self.root / "classifier.pkl"

    # --- persistence --------------------------------------------------------
    def _load_index(self) -> dict[str, dict[str, Any]]:
        if not self.index_path.exists():
            return {}
        try:
            payload = json.loads(self.index_path.read_text(encoding="utf-8"))
        except Exception as exc:  # noqa: BLE001
            LOGGER.warning("Ignoring unreadable global label index %s: %s", self.index_path, exc)
            quarantine_broken_file(self.index_path, LOGGER, "invalid JSON")
            return {}
        if not isinstance(payload, dict):
            return {}
        return {str(key): value for key, value in payload.items() if isinstance(value, dict)}

    def _load_features(self) -> dict[str, np.ndarray]:
        if not self.features_path.exists():
            return {}
        try:
            with np.load(self.features_path, allow_pickle=False) as archive:
                return {str(key): archive[key].astype(np.float32) for key in archive.files}
        except Exception as exc:  # noqa: BLE001
            LOGGER.warning("Ignoring unreadable global feature cache %s: %s", self.features_path, exc)
            quarantine_broken_file(self.features_path, LOGGER, "invalid NPZ")
            return {}

    def record(self, entries: Iterable[tuple[str, int, np.ndarray]], sequence: int) -> int:
        """Add or update labelled examples. Returns the total kept afterwards."""
        entries = [
            (str(path), int(label), np.asarray(feature, dtype=np.float32))
            for path, label, feature in entries
            if label in (0, 1) and feature is not None and np.asarray(feature).size
        ]
        if not entries:
            return 0
        with _LOCK:
            index = self._load_index()
            features = self._load_features()
            for path, label, feature in entries:
                key = _key_for(path)
                index[key] = {"path": path, "label": int(label), "seq": int(sequence)}
                features[key] = feature
            if len(index) > MAX_ENTRIES:
                ordered = sorted(index.items(), key=lambda item: int(item[1].get("seq", 0)))
                for key, _ in ordered[: len(index) - MAX_ENTRIES]:
                    index.pop(key, None)
                    features.pop(key, None)
            # Only keep features that still have a label, and vice versa.
            for key in list(features):
                if key not in index:
                    features.pop(key, None)
            try:
                atomic_write_json(self.index_path, index)

                def write_npz(tmp: Path) -> None:
                    with tmp.open("wb") as handle:
                        np.savez(handle, **features)

                atomic_replace(self.features_path, write_npz)
            except Exception:  # noqa: BLE001
                LOGGER.exception("Could not persist global learning memory")
            return len(index)

    def forget(self, paths: Iterable[str]) -> None:
        """Drop examples (used when a label is cleared)."""
        keys = {_key_for(str(path)) for path in paths}
        if not keys:
            return
        with _LOCK:
            index = self._load_index()
            if not any(key in index for key in keys):
                return
            features = self._load_features()
            for key in keys:
                index.pop(key, None)
                features.pop(key, None)
            try:
                atomic_write_json(self.index_path, index)

                def write_npz(tmp: Path) -> None:
                    with tmp.open("wb") as handle:
                        np.savez(handle, **features)

                atomic_replace(self.features_path, write_npz)
            except Exception:  # noqa: BLE001
                LOGGER.exception("Could not update global learning memory")

    def training_set(self, expected_dim: int | None = None) -> TrainingSet:
        index = self._load_index()
        features = self._load_features()
        rows: list[np.ndarray] = []
        labels: list[int] = []
        paths: list[str] = []
        vanished: list[str] = []
        for key, entry in index.items():
            feature = features.get(key)
            if feature is None:
                continue
            path = str(entry.get("path", key))
            # A label on a file that no longer exists should stop teaching: otherwise a
            # scan of a temporary folder trains this model forever.
            try:
                if not Path(path).exists():
                    vanished.append(path)
                    continue
            except OSError:
                vanished.append(path)
                continue
            if expected_dim is not None and int(feature.shape[-1]) != int(expected_dim):
                continue
            rows.append(np.asarray(feature, dtype=np.float32).ravel())
            labels.append(int(entry.get("label", 0)))
            paths.append(path)
        if vanished:
            LOGGER.info("Dropping %d global label(s) whose files are gone", len(vanished))
            self.forget(vanished)
        if not rows:
            dim = int(expected_dim or 0)
            return TrainingSet(
                features=np.empty((0, dim), dtype=np.float32),
                labels=np.empty((0,), dtype=np.int64),
                paths=[],
            )
        return TrainingSet(
            features=np.vstack(rows).astype(np.float32),
            labels=np.asarray(labels, dtype=np.int64),
            paths=paths,
        )

    def load_classifier(self) -> dict[str, Any] | None:
        if not self.classifier_path.exists():
            return None
        try:
            with self.classifier_path.open("rb") as handle:
                payload = pickle.load(handle)
        except Exception:  # noqa: BLE001
            return None
        if not isinstance(payload, dict) or payload.get("classifier") is None:
            return None
        if int(payload.get("feature_version", -1)) != FEATURE_VERSION:
            return None
        return payload

    def save_classifier(self, payload: Mapping[str, Any]) -> None:
        data = dict(payload)
        data["feature_version"] = FEATURE_VERSION
        try:
            atomic_replace(self.classifier_path, lambda tmp: tmp.write_bytes(pickle.dumps(data)))
        except Exception:  # noqa: BLE001
            LOGGER.exception("Could not persist the global classifier")

    def stats(self) -> dict[str, int]:
        index = self._load_index()
        good = sum(1 for entry in index.values() if int(entry.get("label", -1)) == 1)
        bad = sum(1 for entry in index.values() if int(entry.get("label", -1)) == 0)
        return {"total": len(index), "accepted": good, "rejected": bad}

    def clear(self) -> None:
        with _LOCK:
            for path in (self.index_path, self.features_path, self.classifier_path):
                try:
                    path.unlink()
                except FileNotFoundError:
                    continue
                except Exception:  # noqa: BLE001
                    LOGGER.warning("Could not remove %s", path)


def _key_for(path: str) -> str:
    """NPZ keys cannot contain arbitrary path characters."""
    import hashlib

    return hashlib.sha1(str(path).encode("utf-8")).hexdigest()
