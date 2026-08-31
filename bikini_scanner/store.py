from __future__ import annotations

import contextlib
import hashlib
import json
import logging
import os
import pickle
import shutil
from collections.abc import Iterable, Mapping
from dataclasses import dataclass, field
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, cast

import numpy as np

try:
    from filelock import FileLock

    _FILELOCK_AVAILABLE = True
except Exception:  # noqa: BLE001
    _FILELOCK_AVAILABLE = False

from .image_formats import DECODE_VERSION
from .safe_io import atomic_replace, atomic_write_json, quarantine_broken_file
from .sqlite_cache import SQLiteCache

# Pickle is used only for the classifier cache. Rather than deleting legacy caches, a
# restricted unpickler limits what can be loaded to the few scanner-owned classes and
# basic building blocks a trained model legitimately contains.
_CLASSIFIER_PICKLE_ALLOWLIST: set[str] = {
    "builtins",
    "collections.abc",
    "copyreg",
    "numpy",
    "numpy.core.multiarray",
    "numpy.core.numeric",
    "numpy.dtypes",
    "bikini_scanner.linear_model",
}


class RestrictedUnpickler(pickle.Unpickler):
    def find_class(self, module: str, name: str):
        if module not in _CLASSIFIER_PICKLE_ALLOWLIST and not module.startswith("numpy."):
            raise pickle.UnpicklingError(f"Refusing to unpickle {module}.{name}")
        return super().find_class(module, name)

SUPPORTED_IMAGE_SUFFIXES = {".jpg", ".jpeg", ".png", ".bmp", ".gif", ".webp", ".tif", ".tiff", ".heic", ".heif"}
MATCHES_DIR_NAME = "bikini_matches"
SCAN_METADATA_FILENAME = "scan_metadata.json"
FACE_COUNTS_FILENAME = "face_counts.json"
CLASSIFIER_FILENAME = "classifier.pkl"
CONFIG_OVERRIDE_FILENAME = "config_override.json"
REGION_EMBEDDINGS_FILENAME = "region_embeddings.npz"
VLM_VERDICTS_FILENAME = "vlm_verdicts.json"
CACHE_META_FILENAME = "cache_meta.json"
CLASSIFIER_CACHE_VERSION = 1
# Directories the app creates for its own output carry this marker so a later scan of
# the parent folder does not re-ingest, re-rank and re-copy its own copies.
IGNORE_MARKER_FILENAME = ".bikini_scanner_ignore"
LOGGER = logging.getLogger(__name__)


def _is_scanner_owned_directory(path: Path, cache_dir: Path, matches_dir: Path) -> bool:
    return path in (cache_dir, matches_dir) or (path / IGNORE_MARKER_FILENAME).is_file()


def collect_image_paths(folder: str | Path) -> list[Path]:
    """Walk the folder once, pruning directories the scanner owns or has marked."""
    root = Path(folder).expanduser().resolve()
    cache_dir = root / ".bikini_scanner_cache"
    matches_dir = root / MATCHES_DIR_NAME
    paths: list[Path] = []
    for parent, dirnames, filenames in os.walk(root):
        parent_path = Path(parent)
        if _is_scanner_owned_directory(parent_path, cache_dir, matches_dir):
            dirnames.clear()
            continue
        # Prune ignored and app-owned child directories in place so they are never descended.
        kept: list[str] = []
        for name in dirnames:
            child = parent_path / name
            if _is_scanner_owned_directory(child, cache_dir, matches_dir):
                continue
            kept.append(name)
        dirnames[:] = kept
        for name in filenames:
            if Path(name).suffix.lower() in SUPPORTED_IMAGE_SUFFIXES:
                paths.append((parent_path / name).resolve())
    return sorted(paths)


def safe_stat(path: Path) -> os.stat_result | None:
    """stat() that tolerates the file having gone.

    A scan lists the folder once and then stats each file several times over. Anything
    can remove a file in between — watch mode, a sync client, the user tidying up — and
    an unguarded stat turned that into a FileNotFoundError that killed the whole scan
    and threw away every image already embedded.
    """
    try:
        return path.stat()
    except OSError:
        return None


def content_hash_for_path(path: str | Path, chunk_size: int = 1024 * 1024) -> str | None:
    """Hash file contents using the same SHA-1 identity as the scan cache."""
    digest = hashlib.sha1()
    try:
        with Path(path).open("rb") as handle:
            while chunk := handle.read(chunk_size):
                digest.update(chunk)
    except OSError:
        return None
    return digest.hexdigest()


@dataclass(slots=True)
class FolderStore:
    folder: Path
    cache_dir: Path = field(init=False)
    embeddings_path: Path = field(init=False)
    index_path: Path = field(init=False)
    labels_path: Path = field(init=False)
    metadata_path: Path = field(init=False)
    face_counts_path: Path = field(init=False)
    classifier_path: Path = field(init=False)
    config_override_path: Path = field(init=False)
    review_session_path: Path = field(init=False)
    region_embeddings_path: Path = field(init=False)
    vlm_verdicts_path: Path = field(init=False)
    cache_meta_path: Path = field(init=False)
    cache_db_path: Path = field(init=False)
    lock_path: Path = field(init=False)
    sqlite_cache: SQLiteCache | None = field(init=False, default=None, repr=False)
    _labels_cache: dict[str, int] | None = field(init=False, default=None, repr=False)
    _path_index_cache: dict[str, dict[str, int | str]] | None = field(init=False, default=None, repr=False)
    _embedding_cache: dict[str, np.ndarray] | None = field(init=False, default=None, repr=False)
    _face_count_cache: dict[str, int] | None = field(init=False, default=None, repr=False)
    _region_cache: dict[str, np.ndarray] | None = field(init=False, default=None, repr=False)
    _vlm_cache: dict[str, dict[str, float]] | None = field(init=False, default=None, repr=False)
    _lock: Any = field(init=False, default=None, repr=False)

    def __post_init__(self) -> None:
        self.folder = self.folder.expanduser().resolve()
        self.cache_dir = self.folder / ".bikini_scanner_cache"
        self.cache_dir.mkdir(parents=True, exist_ok=True)
        self.embeddings_path = self.cache_dir / "embeddings.npz"
        self.index_path = self.cache_dir / "embeddings_index.json"
        self.labels_path = self.cache_dir / "labels.json"
        self.metadata_path = self.cache_dir / SCAN_METADATA_FILENAME
        self.face_counts_path = self.cache_dir / FACE_COUNTS_FILENAME
        self.classifier_path = self.cache_dir / CLASSIFIER_FILENAME
        self.config_override_path = self.cache_dir / CONFIG_OVERRIDE_FILENAME
        self.review_session_path = self.cache_dir / "review_session.json"
        self.region_embeddings_path = self.cache_dir / REGION_EMBEDDINGS_FILENAME
        self.vlm_verdicts_path = self.cache_dir / VLM_VERDICTS_FILENAME
        self.cache_meta_path = self.cache_dir / CACHE_META_FILENAME
        self.cache_db_path = self.cache_dir / "cache.db"
        self.lock_path = self.cache_dir / ".bikini_scanner.lock"
        if _FILELOCK_AVAILABLE:
            self._lock = FileLock(str(self.lock_path))
        self._discard_stale_derived_caches()
        self.sqlite_cache = SQLiteCache(self.cache_db_path)
        self.sqlite_cache.migrate_from_legacy(
            self.embeddings_path,
            self.index_path,
            self.region_embeddings_path,
            self.face_counts_path,
        )

    def lock(self, timeout: float = -1) -> Any:
        """Advisory lock for this folder. Use in a `with` statement."""
        if self._lock is None:
            return contextlib.nullcontext()
        return self._lock.acquire(timeout=timeout)

    def _discard_stale_derived_caches(self) -> None:
        """Drop cached work that an older build derived from different pixels.

        Everything here is keyed by content hash, which does not change when the way we
        *decode* an image changes. So when the decoder changes - EXIF orientation being
        the case that prompted this - a stale entry would be silently reused and the
        whole folder would keep scoring as though the fix had never landed.

        Labels, per-folder config overrides and the review session are the user's own
        work and are never touched; only derived artefacts are discarded.
        """
        try:
            recorded = 0
            if self.cache_meta_path.is_file():
                payload = json.loads(self.cache_meta_path.read_text(encoding="utf-8"))
                if isinstance(payload, Mapping):
                    recorded = int(payload.get("decode_version", 0) or 0)
        except (OSError, TypeError, ValueError, json.JSONDecodeError):
            recorded = 0
        if recorded == DECODE_VERSION:
            return

        derived = (
            self.embeddings_path,
            self.index_path,
            self.region_embeddings_path,
            self.face_counts_path,
            self.classifier_path,
            self.vlm_verdicts_path,
            self.metadata_path,
        )
        discarded = [path.name for path in derived if path.exists()]
        for path in derived:
            try:
                path.unlink(missing_ok=True)
            except OSError as exc:
                LOGGER.warning("Could not discard stale cache %s: %s", path, exc)
        if self.sqlite_cache is not None:
            self.sqlite_cache.clear()
        for path in (self.cache_db_path, self.cache_dir / "cache.db-wal", self.cache_dir / "cache.db-shm"):
            try:
                if path.exists():
                    path.unlink(missing_ok=True)
                    discarded.append(path.name)
            except OSError as exc:
                LOGGER.warning("Could not discard stale cache file %s: %s", path, exc)
        if discarded:
            LOGGER.info(
                "Image decoding changed (v%d -> v%d); discarded %s in %s so they are rebuilt "
                "from correctly oriented pixels. Your labels were kept.",
                recorded,
                DECODE_VERSION,
                ", ".join(sorted(discarded)),
                self.cache_dir,
            )
        self._embedding_cache = None
        self._path_index_cache = None
        self._face_count_cache = None
        self._region_cache = None
        self._vlm_cache = None
        try:
            atomic_write_json(self.cache_meta_path, {"decode_version": DECODE_VERSION})
        except OSError as exc:
            LOGGER.warning("Could not record the cache version in %s: %s", self.cache_meta_path, exc)

    def _load_vlm_cache(self) -> dict[str, dict[str, float]]:
        if self._vlm_cache is None:
            if not self.vlm_verdicts_path.exists():
                self._vlm_cache = {}
            else:
                try:
                    payload = json.loads(self.vlm_verdicts_path.read_text(encoding="utf-8"))
                    self._vlm_cache = {
                        str(key): {str(axis): float(value) for axis, value in values.items()}
                        for key, values in payload.items()
                        if isinstance(values, dict)
                    }
                except Exception as exc:  # noqa: BLE001
                    LOGGER.warning("Ignoring unreadable VLM verdict cache %s: %s", self.vlm_verdicts_path, exc)
                    self._vlm_cache = {}
        return self._vlm_cache

    @staticmethod
    def vlm_cache_key(content_hash: str, model: str, prompt_version: str) -> str:
        return f"{content_hash}|{model}|{prompt_version}"

    def lookup_vlm_verdict(self, key: str) -> dict[str, float] | None:
        value = self._load_vlm_cache().get(str(key))
        return dict(value) if value is not None else None

    def save_vlm_verdicts(self, entries: Mapping[str, Mapping[str, float]]) -> None:
        if not entries:
            return
        cache = self._load_vlm_cache()
        for key, value in entries.items():
            cache[str(key)] = {str(axis): float(score) for axis, score in value.items()}
        atomic_write_json(self.vlm_verdicts_path, cache)

    def load_labels(self) -> dict[str, int]:
        if self._labels_cache is None:
            if not self.labels_path.exists():
                self._labels_cache = {}
            else:
                try:
                    with self.labels_path.open("r", encoding="utf-8") as handle:
                        data = json.load(handle)
                    if isinstance(data, dict):
                        self._labels_cache = {str(path): int(label) for path, label in data.items()}
                    else:
                        raise TypeError("invalid payload type")
                except (OSError, TypeError, ValueError, json.JSONDecodeError) as exc:
                    LOGGER.warning("Ignoring unreadable labels cache %s: %s", self.labels_path, exc)
                    quarantine_broken_file(self.labels_path, LOGGER, "invalid JSON")
                    self._labels_cache = {}
        return dict(self._labels_cache)

    def save_labels(self, labels: dict[str, int]) -> None:
        """Persist labels, then adopt them in memory.

        Order matters: updating the cache first meant that a failed write (read-only
        folder, full disk, disconnected network drive) left the app believing a label
        was saved when it was not, and the loss only showed up after a restart.
        """
        normalized = {str(path): int(label) for path, label in labels.items()}
        atomic_write_json(self.labels_path, dict(sorted(normalized.items())))
        self._labels_cache = normalized

    def load_config_override(self) -> dict[str, Any] | None:
        if not self.config_override_path.exists():
            return None
        try:
            payload = json.loads(self.config_override_path.read_text(encoding="utf-8"))
        except Exception as exc:  # noqa: BLE001
            LOGGER.warning("Ignoring unreadable config override %s: %s", self.config_override_path, exc)
            quarantine_broken_file(self.config_override_path, LOGGER, "invalid JSON")
            return None
        if isinstance(payload, dict):
            return payload
        LOGGER.warning("Ignoring unreadable config override %s: invalid payload type", self.config_override_path)
        quarantine_broken_file(self.config_override_path, LOGGER, "invalid payload type")
        return None

    def save_config_override(self, payload: Mapping[str, Any]) -> None:
        atomic_write_json(self.config_override_path, dict(payload))

    def clear_config_override(self) -> None:
        try:
            self.config_override_path.unlink()
        except FileNotFoundError:
            pass

    def _load_path_index(self) -> dict[str, dict[str, int | str]]:
        if self._path_index_cache is None:
            sqlite = self.sqlite_cache
            assert sqlite is not None
            records = sqlite._load_all_image_records()
            self._path_index_cache = {
                str(path): {
                    "path": str(path),
                    "mtime_ns": int(record["mtime_ns"]),
                    "size": int(record["size"]),
                    "content_hash": str(record["content_hash"]),
                }
                for path, record in records.items()
            }
        return self._path_index_cache

    def _load_embedding_cache(self) -> dict[str, np.ndarray]:
        if self._embedding_cache is None:
            sqlite = self.sqlite_cache
            assert sqlite is not None
            self._embedding_cache = sqlite._load_all_embeddings()
        return self._embedding_cache

    def _load_face_count_cache(self) -> dict[str, int]:
        if self._face_count_cache is None:
            sqlite = self.sqlite_cache
            assert sqlite is not None
            self._face_count_cache = sqlite.load_face_counts()
        return self._face_count_cache

    def get_cached_image_records(self, paths: Iterable[Path]) -> dict[Path, dict[str, object]]:
        sqlite = self.sqlite_cache
        assert sqlite is not None
        return sqlite.get_cached_image_records(list(paths))

    def get_cached_embeddings(self, paths: Iterable[Path]) -> dict[Path, np.ndarray]:
        return {path: cast(np.ndarray, record["embedding"]) for path, record in self.get_cached_image_records(paths).items()}

    def lookup_content_embedding(self, content_hash: str) -> np.ndarray | None:
        sqlite = self.sqlite_cache
        assert sqlite is not None
        return sqlite.lookup_content_embedding(content_hash)

    def content_hash_for_path(self, path: str | Path) -> str | None:
        return content_hash_for_path(path)

    def lookup_face_count(self, content_hash: str) -> int | None:
        sqlite = self.sqlite_cache
        assert sqlite is not None
        return sqlite.lookup_face_count(content_hash)

    def save_embeddings(self, embeddings_by_path: dict[Path, np.ndarray]) -> None:
        if not embeddings_by_path or self.sqlite_cache is None:
            return
        content_embeddings: dict[str, np.ndarray] = {}
        path_records: dict[Path, dict[str, int | str]] = {}
        for path, embedding in embeddings_by_path.items():
            content_hash = content_hash_for_path(path)
            if content_hash is None:
                continue
            content_embeddings[content_hash] = np.asarray(embedding, dtype=np.float32)
            stat = safe_stat(path)
            if stat is None:
                continue
            path_records[path] = {
                "content_hash": content_hash,
                "mtime_ns": stat.st_mtime_ns,
                "size": stat.st_size,
            }
        self.sqlite_cache.save_scan_cache(content_embeddings, path_records)
        if self._embedding_cache is None:
            self._embedding_cache = {}
        self._embedding_cache.update(content_embeddings)
        if self._path_index_cache is None:
            self._path_index_cache = {}
        for path, record in path_records.items():
            key = str(path.resolve())
            self._path_index_cache[key] = {
                "path": key,
                "content_hash": str(record["content_hash"]),
                "mtime_ns": int(record["mtime_ns"]),
                "size": int(record["size"]),
            }

    def save_scan_cache(
        self,
        content_embeddings: dict[str, np.ndarray],
        path_records: dict[Path, dict[str, int | str]],
        face_counts_by_content_hash: dict[str, int] | None = None,
    ) -> None:
        if self.sqlite_cache is None:
            return
        if not content_embeddings and not path_records and not face_counts_by_content_hash:
            return
        normalized_embeddings = {str(k): np.asarray(v, dtype=np.float32) for k, v in content_embeddings.items()}
        normalized_records: dict[Path, dict[str, int | str]] = {
            path: {
                "content_hash": str(record["content_hash"]),
                "mtime_ns": int(record["mtime_ns"]),
                "size": int(record["size"]),
            }
            for path, record in path_records.items()
        }
        normalized_faces = {str(k): int(v) for k, v in (face_counts_by_content_hash or {}).items()}
        self.sqlite_cache.save_scan_cache(
            normalized_embeddings,
            normalized_records,
            normalized_faces,
        )
        if self._embedding_cache is None:
            self._embedding_cache = {}
        self._embedding_cache.update(normalized_embeddings)
        if self._path_index_cache is None:
            self._path_index_cache = {}
        for path, record in normalized_records.items():
            self._path_index_cache[str(path.resolve())] = {
                "path": str(path.resolve()),
                "content_hash": record["content_hash"],
                "mtime_ns": record["mtime_ns"],
                "size": record["size"],
            }
        if self._face_count_cache is None:
            self._face_count_cache = {}
        self._face_count_cache.update(normalized_faces)

    # --- region (deep scan) embeddings --------------------------------------
    def _load_region_cache(self) -> dict[str, np.ndarray]:
        if self._region_cache is None:
            self._region_cache = {}
        return self._region_cache

    @staticmethod
    def region_cache_key(content_hash: str, namespace: str, region_key: str) -> str:
        # Namespaced by model + geometry version: crops from a different model or a
        # different crop layout are not interchangeable.
        return f"{content_hash}|{namespace}|{region_key}"

    def lookup_region_embeddings(self, content_hash: str, namespace: str) -> dict[str, np.ndarray]:
        sqlite = self.sqlite_cache
        assert sqlite is not None
        return sqlite.lookup_region_embeddings(str(content_hash), str(namespace))

    def save_region_embeddings(self, entries: Mapping[str, np.ndarray]) -> None:
        """Persist region embeddings keyed by region_cache_key()."""
        if not entries or self.sqlite_cache is None:
            return
        normalized = {str(k): np.asarray(v, dtype=np.float32) for k, v in entries.items()}
        self.sqlite_cache.save_region_embeddings(normalized)
        if self._region_cache is None:
            self._region_cache = {}
        self._region_cache.update(normalized)

    def save_scan_metadata(self, payload: Mapping[str, object]) -> None:
        atomic_write_json(self.metadata_path, payload)

    def load_classifier_cache(self) -> dict[str, Any] | None:
        if not self.classifier_path.exists():
            return None
        try:
            with self.classifier_path.open("rb") as handle:
                payload = RestrictedUnpickler(handle).load()
        except Exception:  # noqa: BLE001
            return None
        if not isinstance(payload, dict):
            return None
        if int(payload.get("version", 0)) != CLASSIFIER_CACHE_VERSION:
            return None
        classifier = payload.get("classifier")
        if classifier is None:
            return None
        return payload

    def save_classifier_cache(self, payload: Mapping[str, Any]) -> None:
        data = dict(payload)
        data["version"] = CLASSIFIER_CACHE_VERSION

        def write_classifier(tmp: Path) -> None:
            tmp.write_bytes(pickle.dumps(data))

        atomic_replace(self.classifier_path, write_classifier)

    def load_review_session(self) -> dict[str, Any] | None:
        if not self.review_session_path.exists():
            return None
        try:
            with self.review_session_path.open("r", encoding="utf-8") as handle:
                payload = json.load(handle)
        except Exception as exc:  # noqa: BLE001
            LOGGER.warning("Ignoring unreadable review session %s: %s", self.review_session_path, exc)
            quarantine_broken_file(self.review_session_path, LOGGER, "invalid JSON")
            return None
        if not isinstance(payload, dict):
            quarantine_broken_file(self.review_session_path, LOGGER, "invalid payload type")
            return None
        return payload

    def save_review_session(self, payload: Mapping[str, Any]) -> None:
        atomic_write_json(self.review_session_path, dict(payload))

    def duplicate_groups(self, paths: Iterable[Path] | None = None) -> dict[str, list[str]]:
        index = self._load_path_index()
        allowed = {str(path.resolve()) for path in paths} if paths is not None else None
        groups: dict[str, list[str]] = {}
        for path, record in index.items():
            if allowed is not None and path not in allowed:
                continue
            content_hash = record.get("content_hash")
            if content_hash:
                groups.setdefault(str(content_hash), []).append(path)
        return {key: sorted(values) for key, values in groups.items() if len(values) > 1}

    def cache_size_bytes(self) -> int:
        if not self.cache_dir.exists():
            return 0
        total = 0
        for path in self.cache_dir.rglob("*"):
            if path.is_file():
                try:
                    total += path.stat().st_size
                except OSError:
                    continue
        return total

    def clear_cache(self) -> None:
        if self.sqlite_cache is not None:
            try:
                self.sqlite_cache.close()
            except Exception as exc:  # noqa: BLE001
                # rmtree below is what actually matters; a close failure only risks a
                # locked file on Windows, and knowing that is why it is logged.
                LOGGER.warning("Could not close the SQLite cache before clearing it: %s", exc)
        if self.cache_dir.exists():
            shutil.rmtree(self.cache_dir)
        self.cache_dir.mkdir(parents=True, exist_ok=True)
        self.embeddings_path = self.cache_dir / "embeddings.npz"
        self.index_path = self.cache_dir / "embeddings_index.json"
        self.labels_path = self.cache_dir / "labels.json"
        self.metadata_path = self.cache_dir / SCAN_METADATA_FILENAME
        self.face_counts_path = self.cache_dir / FACE_COUNTS_FILENAME
        self.classifier_path = self.cache_dir / CLASSIFIER_FILENAME
        self.config_override_path = self.cache_dir / CONFIG_OVERRIDE_FILENAME
        self.review_session_path = self.cache_dir / "review_session.json"
        self.region_embeddings_path = self.cache_dir / REGION_EMBEDDINGS_FILENAME
        self.cache_db_path = self.cache_dir / "cache.db"
        self.sqlite_cache = SQLiteCache(self.cache_db_path)
        self._labels_cache = None
        self._path_index_cache = None
        self._embedding_cache = None
        self._face_count_cache = None
        self._region_cache = None
        self._vlm_cache = None

    def build_scan_metadata(
        self,
        image_records: list[dict[str, object]],
        skipped_records: list[dict[str, object]],
        scan_timestamp: str | None = None,
    ) -> dict[str, object]:
        timestamp = scan_timestamp or datetime.now(timezone.utc).isoformat()
        return {
            "scanned_at": timestamp,
            "images": image_records,
            "skipped": skipped_records,
        }
