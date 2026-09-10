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
from collections.abc import Iterable, Mapping
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import numpy as np

from .safe_io import atomic_replace, atomic_write_json, quarantine_broken_file
from .store import RestrictedUnpickler
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
    # Everything below is an in-process read cache, stamped with the (mtime, size) of
    # the file it came from. A retrain used to re-read and re-parse the whole index and
    # the whole feature archive, then stat every labelled path, to add three rows. The
    # stamp is two stats; a mismatch (another instance wrote) falls back to a real read.
    _index_cache: dict = field(init=False, default_factory=dict, repr=False)
    _index_stamp: tuple = field(init=False, default=(), repr=False)
    _features_cache: dict = field(init=False, default_factory=dict, repr=False)
    _features_stamp: tuple = field(init=False, default=(), repr=False)
    _training_cache: dict = field(init=False, default_factory=dict, repr=False)

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

    @property
    def retained_path(self) -> Path:
        return self.root / "retained.json"

    # --- deliberately removed examples --------------------------------------
    def _load_retained(self) -> set[str]:
        try:
            payload = json.loads(self.retained_path.read_text(encoding="utf-8"))
        except FileNotFoundError:
            return set()
        except Exception as exc:  # noqa: BLE001
            LOGGER.warning("Ignoring unreadable retained-label list %s: %s", self.retained_path, exc)
            return set()
        return {str(key) for key in payload} if isinstance(payload, list) else set()

    def retain(self, paths: Iterable[str]) -> None:
        """Keep teaching from these examples after their files are gone.

        `training_set` drops a label whose file has vanished, so that scanning a
        temporary folder does not train this model forever. Deleting a photo *because*
        you rejected it is the opposite case: the whole point of "reject and delete"
        is that the scanner learns what you did not want, and dropping the label on the
        next pass would quietly undo the training the action was named for.

        Keys are derived from the path, so this can be called before the retrain that
        records the features — which is what the delete flow does, since the files are
        gone by the time that retrain runs.
        """
        keys = {_key_for(str(path)) for path in paths}
        if not keys:
            return
        with _LOCK:
            merged = self._load_retained() | keys
            try:
                atomic_write_json(self.retained_path, sorted(merged))
            except Exception:
                LOGGER.exception("Could not persist the retained-label list")

    # --- persistence --------------------------------------------------------
    @staticmethod
    def _stamp(path: Path) -> tuple:
        """Cheap identity for a file: modification time and size, or () if absent."""
        try:
            info = path.stat()
        except OSError:
            return ()
        return (info.st_mtime_ns, info.st_size)

    def _load_index(self) -> dict[str, dict[str, Any]]:
        stamp = self._stamp(self.index_path)
        if stamp and stamp == self._index_stamp:
            return self._index_cache
        if not self.index_path.exists():
            self._index_cache, self._index_stamp = {}, stamp
            return {}
        try:
            payload = json.loads(self.index_path.read_text(encoding="utf-8"))
        except Exception as exc:  # noqa: BLE001
            LOGGER.warning("Ignoring unreadable global label index %s: %s", self.index_path, exc)
            quarantine_broken_file(self.index_path, LOGGER, "invalid JSON")
            self._index_cache, self._index_stamp = {}, ()
            return {}
        if not isinstance(payload, dict):
            self._index_cache, self._index_stamp = {}, stamp
            return {}
        parsed = {str(key): value for key, value in payload.items() if isinstance(value, dict)}
        self._index_cache, self._index_stamp = parsed, stamp
        return parsed

    def _load_features(self) -> dict[str, np.ndarray]:
        stamp = self._stamp(self.features_path)
        if stamp and stamp == self._features_stamp:
            return self._features_cache
        if not self.features_path.exists():
            self._features_cache, self._features_stamp = {}, stamp
            return {}
        try:
            with np.load(self.features_path, allow_pickle=False) as archive:
                loaded = {str(key): archive[key].astype(np.float32) for key in archive.files}
        except Exception as exc:  # noqa: BLE001
            LOGGER.warning("Ignoring unreadable global feature cache %s: %s", self.features_path, exc)
            quarantine_broken_file(self.features_path, LOGGER, "invalid NPZ")
            self._features_cache, self._features_stamp = {}, ()
            return {}
        self._features_cache, self._features_stamp = loaded, stamp
        return loaded

    def _adopt(self, index: dict[str, dict[str, Any]], features: dict[str, np.ndarray]) -> None:
        """Take the just-written state as the cache, and drop derived results."""
        self._index_cache, self._index_stamp = index, self._stamp(self.index_path)
        self._features_cache, self._features_stamp = features, self._stamp(self.features_path)
        self._training_cache = {}

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
                evicted = {key for key, _ in ordered[: len(index) - MAX_ENTRIES]}
                for key in evicted:
                    index.pop(key, None)
                    features.pop(key, None)
                # An evicted row can never be consulted again, so its retained mark is
                # dead weight. Without this the list grew by one key per deleted photo
                # for the life of the install and was rewritten in full on every retain.
                retained = self._load_retained()
                if retained & evicted:
                    try:
                        atomic_write_json(self.retained_path, sorted(retained - evicted))
                    except Exception:
                        LOGGER.exception("Could not prune the retained-label list")
            # Only keep features that still have a label, and vice versa.
            for key in list(features):
                if key not in index:
                    features.pop(key, None)
            try:
                atomic_write_json(self.index_path, index)

                def write_npz(tmp: Path) -> None:
                    # No allow_pickle kwarg here on purpose: it only became a real
                    # savez parameter in recent numpy, and on the older 2.x releases
                    # this project still allows it would be swallowed as an *array*
                    # named "allow_pickle" instead. Every value is already a plain
                    # float32 ndarray, which savez never pickles, and the load side
                    # passes allow_pickle=False, which is where it is enforced.
                    with tmp.open("wb") as handle:
                        # numpy 2.5 types savez's second
                        # parameter as allow_pickle, so **features trips the stub.
                        # Keys here are sha1 hexdigests and can never collide with it.
                        np.savez(handle, **features)  # type: ignore[arg-type]

                atomic_replace(self.features_path, write_npz)
                self._adopt(index, features)
            except Exception:
                LOGGER.exception("Could not persist global learning memory")
            return len(index)

    def forget(self, paths: Iterable[str]) -> None:
        """Drop examples (used when a label is cleared)."""
        keys = {_key_for(str(path)) for path in paths}
        if not keys:
            return
        with _LOCK:
            # Clearing a label retires it completely, retained or not; otherwise the
            # list would keep growing with keys nothing refers to any more.
            retained = self._load_retained()
            if retained & keys:
                try:
                    atomic_write_json(self.retained_path, sorted(retained - keys))
                except Exception:
                    LOGGER.exception("Could not persist the retained-label list")
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
                    # No allow_pickle kwarg here on purpose: it only became a real
                    # savez parameter in recent numpy, and on the older 2.x releases
                    # this project still allows it would be swallowed as an *array*
                    # named "allow_pickle" instead. Every value is already a plain
                    # float32 ndarray, which savez never pickles, and the load side
                    # passes allow_pickle=False, which is where it is enforced.
                    with tmp.open("wb") as handle:
                        # numpy 2.5 types savez's second
                        # parameter as allow_pickle, so **features trips the stub.
                        # Keys here are sha1 hexdigests and can never collide with it.
                        np.savez(handle, **features)  # type: ignore[arg-type]

                atomic_replace(self.features_path, write_npz)
                self._adopt(index, features)
            except Exception:
                LOGGER.exception("Could not update global learning memory")

    def training_set(self, expected_dim: int | None = None) -> TrainingSet:
        with _LOCK:
            # Snapshots, not the live caches. _load_index hands back the cache dict
            # itself, and record()/forget() mutate it in place, so iterating the
            # original raised "dictionary changed size during iteration" the moment
            # anything else touched the store.
            index = dict(self._load_index())
            features = dict(self._load_features())
        # Building this walks every labelled row and stats every labelled path to drop
        # ones whose file is gone. That answer only changes when the store changes, so
        # it is derived once per (index, features, dim) rather than once per retrain.
        # The retained list is part of the answer, so it has to be part of the key:
        # without it, retaining an example returned the cached set that had just
        # dropped it.
        cache_key = (self._index_stamp, self._features_stamp, self._stamp(self.retained_path), expected_dim)
        cached = self._training_cache.get(cache_key)
        if cached is not None:
            return cached
        retained = self._load_retained()
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
            # scan of a temporary folder trains this model forever. A file the reviewer
            # deleted *because* they rejected it is the exception — see `retain`.
            if key not in retained:
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
            empty = TrainingSet(
                features=np.empty((0, dim), dtype=np.float32),
                labels=np.empty((0,), dtype=np.int64),
                paths=[],
            )
            return self._remember(cache_key, empty)
        return self._remember(
            cache_key,
            TrainingSet(
                features=np.vstack(rows).astype(np.float32),
                labels=np.asarray(labels, dtype=np.int64),
                paths=paths,
            ),
        )

    def _remember(self, cache_key: tuple, result: TrainingSet) -> TrainingSet:
        # One entry per shape is plenty: callers ask with a single expected_dim.
        self._training_cache = {cache_key: result}
        return result

    def load_classifier(self) -> dict[str, Any] | None:
        if not self.classifier_path.exists():
            return None
        try:
            # Same restricted unpickler the per-folder classifier cache uses: a
            # classifier pickle is data, and nothing in one legitimately needs to
            # import outside numpy and this package's own model classes.
            with self.classifier_path.open("rb") as handle:
                payload = RestrictedUnpickler(handle).load()
        except Exception as exc:  # noqa: BLE001
            LOGGER.warning("Ignoring unreadable global classifier %s: %s", self.classifier_path, exc)
            return None
        if not isinstance(payload, dict) or payload.get("classifier") is None:
            return None
        if int(payload.get("feature_version", -1)) != FEATURE_VERSION:
            return None
        return payload

    def save_classifier(self, payload: Mapping[str, Any]) -> None:
        data = dict(payload)
        data["feature_version"] = FEATURE_VERSION

        def write_classifier(tmp: Path) -> None:
            tmp.write_bytes(pickle.dumps(data))

        try:
            atomic_replace(self.classifier_path, write_classifier)
        except Exception:
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
