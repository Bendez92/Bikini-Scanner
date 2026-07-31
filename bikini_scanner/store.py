from __future__ import annotations

import hashlib
import json
import logging
import os
import pickle
import shutil
from dataclasses import dataclass, field
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Iterable, Mapping

import numpy as np

from .safe_io import atomic_replace, atomic_write_json, quarantine_broken_file

SUPPORTED_IMAGE_SUFFIXES = {".jpg", ".jpeg", ".png", ".bmp", ".gif", ".webp", ".tif", ".tiff", ".heic", ".heif"}
MATCHES_DIR_NAME = "bikini_matches"
SCAN_METADATA_FILENAME = "scan_metadata.json"
FACE_COUNTS_FILENAME = "face_counts.json"
CLASSIFIER_FILENAME = "classifier.pkl"
CONFIG_OVERRIDE_FILENAME = "config_override.json"
REGION_EMBEDDINGS_FILENAME = "region_embeddings.npz"
VLM_VERDICTS_FILENAME = "vlm_verdicts.json"
CLASSIFIER_CACHE_VERSION = 1
LOGGER = logging.getLogger(__name__)


def collect_image_paths(folder: str | Path) -> list[Path]:
    root = Path(folder).expanduser().resolve()
    cache_dir = root / ".bikini_scanner_cache"
    matches_dir = root / MATCHES_DIR_NAME
    paths: list[Path] = []
    for path in root.rglob("*"):
        if not path.is_file():
            continue
        if cache_dir in path.parents:
            continue
        if matches_dir in path.parents:
            continue
        if path.suffix.lower() in SUPPORTED_IMAGE_SUFFIXES:
            paths.append(path.resolve())
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


def _legacy_cache_key(path: Path) -> str | None:
    stat = safe_stat(path)
    if stat is None:
        return None
    # path.resolve() is part of the key: changing it would invalidate every legacy cache.
    token = f"{path.resolve()}|{stat.st_mtime_ns}|{stat.st_size}".encode("utf-8")
    return hashlib.sha1(token).hexdigest()


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
    _labels_cache: dict[str, int] | None = field(init=False, default=None, repr=False)
    _path_index_cache: dict[str, dict[str, int | str]] | None = field(init=False, default=None, repr=False)
    _embedding_cache: dict[str, np.ndarray] | None = field(init=False, default=None, repr=False)
    _face_count_cache: dict[str, int] | None = field(init=False, default=None, repr=False)
    _region_cache: dict[str, np.ndarray] | None = field(init=False, default=None, repr=False)
    _vlm_cache: dict[str, dict[str, float]] | None = field(init=False, default=None, repr=False)

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
            if not self.index_path.exists():
                self._path_index_cache = {}
            else:
                try:
                    with self.index_path.open("r", encoding="utf-8") as handle:
                        raw = json.load(handle)
                    if isinstance(raw, dict) and raw and all(isinstance(value, dict) for value in raw.values()):
                        self._path_index_cache = {}
                        for key, value in raw.items():
                            path = str(value.get("path", key))
                            self._path_index_cache[path] = {
                                "path": path,
                                "mtime_ns": int(value.get("mtime_ns", 0)),
                                "size": int(value.get("size", 0)),
                                **({"content_hash": str(value["content_hash"])} if "content_hash" in value else {}),
                            }
                    else:
                        quarantine_broken_file(self.index_path, LOGGER, "invalid payload type")
                        self._path_index_cache = {}
                except (OSError, TypeError, ValueError, json.JSONDecodeError) as exc:
                    LOGGER.warning("Ignoring unreadable embedding index %s: %s", self.index_path, exc)
                    quarantine_broken_file(self.index_path, LOGGER, "invalid JSON")
                    self._path_index_cache = {}
        return self._path_index_cache

    def _load_embedding_cache(self) -> dict[str, np.ndarray]:
        if self._embedding_cache is None:
            if not self.embeddings_path.exists():
                self._embedding_cache = {}
            else:
                try:
                    with np.load(self.embeddings_path, allow_pickle=False) as archive:
                        self._embedding_cache = {str(key): archive[key].astype(np.float32) for key in archive.files}
                except (OSError, ValueError, TypeError) as exc:
                    LOGGER.warning("Ignoring unreadable embedding cache %s: %s", self.embeddings_path, exc)
                    quarantine_broken_file(self.embeddings_path, LOGGER, "invalid NPZ")
                    self._embedding_cache = {}
        return self._embedding_cache

    def _load_face_count_cache(self) -> dict[str, int]:
        if self._face_count_cache is None:
            if not self.face_counts_path.exists():
                self._face_count_cache = {}
            else:
                try:
                    with self.face_counts_path.open("r", encoding="utf-8") as handle:
                        data = json.load(handle)
                    if isinstance(data, dict):
                        self._face_count_cache = {str(key): int(value) for key, value in data.items()}
                    else:
                        raise TypeError("invalid payload type")
                except (OSError, TypeError, ValueError, json.JSONDecodeError) as exc:
                    LOGGER.warning("Ignoring unreadable face-count cache %s: %s", self.face_counts_path, exc)
                    quarantine_broken_file(self.face_counts_path, LOGGER, "invalid JSON")
                    self._face_count_cache = {}
        return self._face_count_cache

    def get_cached_image_records(self, paths: Iterable[Path]) -> dict[Path, dict[str, object]]:
        index = self._load_path_index()
        embeddings = self._load_embedding_cache()
        cached: dict[Path, dict[str, object]] = {}
        for path in paths:
            entry = index.get(str(path))
            if not entry:
                continue
            stat = safe_stat(path)
            if stat is None:
                # Listed a moment ago, gone now. Treat it as uncached rather than fatal.
                continue
            if int(entry.get("mtime_ns", -1)) != stat.st_mtime_ns or int(entry.get("size", -1)) != stat.st_size:
                continue
            content_hash = entry.get("content_hash")
            embedding: np.ndarray | None = None
            if content_hash and str(content_hash) in embeddings:
                embedding = embeddings[str(content_hash)]
            else:
                legacy_key = _legacy_cache_key(path)
                if legacy_key is not None and legacy_key in embeddings:
                    embedding = embeddings[legacy_key]
            if embedding is not None:
                cached[path] = {
                    "path": path,
                    "content_hash": str(content_hash) if content_hash else None,
                    "embedding": embedding,
                    "mtime_ns": stat.st_mtime_ns,
                    "size": stat.st_size,
                }
        return cached

    def get_cached_embeddings(self, paths: Iterable[Path]) -> dict[Path, np.ndarray]:
        return {path: record["embedding"] for path, record in self.get_cached_image_records(paths).items()}

    def lookup_content_embedding(self, content_hash: str) -> np.ndarray | None:
        embeddings = self._load_embedding_cache()
        embedding = embeddings.get(str(content_hash))
        if embedding is None:
            return None
        return embedding.astype(np.float32)

    def content_hash_for_path(self, path: str | Path) -> str | None:
        return content_hash_for_path(path)

    def lookup_face_count(self, content_hash: str) -> int | None:
        return self._load_face_count_cache().get(str(content_hash))

    def save_embeddings(self, embeddings_by_path: dict[Path, np.ndarray]) -> None:
        if not embeddings_by_path:
            return
        archive: dict[str, np.ndarray] = {}
        if self.embeddings_path.exists():
            with np.load(self.embeddings_path, allow_pickle=False) as existing_archive:
                for key in existing_archive.files:
                    archive[key] = existing_archive[key].astype(np.float32)
        dirty = False
        for path, embedding in embeddings_by_path.items():
            key = _legacy_cache_key(path)
            value = np.asarray(embedding, dtype=np.float32)
            if key not in archive or not np.array_equal(archive[key], value):
                dirty = True
            archive[key] = value
        if not dirty:
            return
        def write_npz(tmp: Path) -> None:
            with tmp.open("wb") as handle:
                np.savez(handle, **archive)

        atomic_replace(self.embeddings_path, write_npz)
        self._embedding_cache = archive

    def save_scan_cache(
        self,
        content_embeddings: dict[str, np.ndarray],
        path_records: dict[Path, dict[str, int | str]],
        face_counts_by_content_hash: dict[str, int] | None = None,
    ) -> None:
        if not content_embeddings and not path_records and not face_counts_by_content_hash:
            return
        embeddings = self._load_embedding_cache()
        index = self._load_path_index()
        face_counts = self._load_face_count_cache()
        dirty_embeddings = False
        dirty_index = False
        dirty_faces = False
        for content_hash, embedding in content_embeddings.items():
            key = str(content_hash)
            value = np.asarray(embedding, dtype=np.float32)
            if key not in embeddings or not np.array_equal(embeddings[key], value):
                dirty_embeddings = True
            embeddings[key] = value
        for path, record in path_records.items():
            key = str(path.resolve())
            current = {
                "path": key,
                "mtime_ns": int(record["mtime_ns"]),
                "size": int(record["size"]),
                "content_hash": str(record["content_hash"]),
            }
            if index.get(key) != current:
                dirty_index = True
            index[key] = current
        if face_counts_by_content_hash:
            for content_hash, face_count in face_counts_by_content_hash.items():
                key = str(content_hash)
                value = int(face_count)
                if face_counts.get(key) != value:
                    dirty_faces = True
                face_counts[key] = value
        if dirty_embeddings:
            def write_npz(tmp: Path) -> None:
                with tmp.open("wb") as handle:
                    np.savez(handle, **embeddings)

            atomic_replace(self.embeddings_path, write_npz)
            self._embedding_cache = embeddings
        if dirty_index:
            atomic_write_json(self.index_path, index)
            self._path_index_cache = index
        if dirty_faces:
            atomic_write_json(self.face_counts_path, face_counts)
            self._face_count_cache = face_counts

    # --- region (deep scan) embeddings --------------------------------------
    def _load_region_cache(self) -> dict[str, np.ndarray]:
        if self._region_cache is None:
            if not self.region_embeddings_path.exists():
                self._region_cache = {}
            else:
                try:
                    with np.load(self.region_embeddings_path, allow_pickle=False) as archive:
                        self._region_cache = {str(key): archive[key].astype(np.float32) for key in archive.files}
                except (OSError, ValueError, TypeError) as exc:
                    LOGGER.warning("Ignoring unreadable region cache %s: %s", self.region_embeddings_path, exc)
                    quarantine_broken_file(self.region_embeddings_path, LOGGER, "invalid NPZ")
                    self._region_cache = {}
        return self._region_cache

    @staticmethod
    def region_cache_key(content_hash: str, namespace: str, region_key: str) -> str:
        # Namespaced by model + geometry version: crops from a different model or a
        # different crop layout are not interchangeable.
        return f"{content_hash}|{namespace}|{region_key}"

    def lookup_region_embeddings(self, content_hash: str, namespace: str) -> dict[str, np.ndarray]:
        prefix = f"{content_hash}|{namespace}|"
        cache = self._load_region_cache()
        return {key[len(prefix) :]: value for key, value in cache.items() if key.startswith(prefix)}

    def save_region_embeddings(self, entries: Mapping[str, np.ndarray]) -> None:
        """Persist region embeddings keyed by region_cache_key()."""
        if not entries:
            return
        cache = self._load_region_cache()
        dirty = False
        for key, value in entries.items():
            array = np.asarray(value, dtype=np.float32)
            if key not in cache or not np.array_equal(cache[key], array):
                dirty = True
            cache[key] = array
        if not dirty:
            return

        def write_npz(tmp: Path) -> None:
            with tmp.open("wb") as handle:
                np.savez(handle, **cache)

        atomic_replace(self.region_embeddings_path, write_npz)
        self._region_cache = cache

    def save_scan_metadata(self, payload: Mapping[str, object]) -> None:
        atomic_write_json(self.metadata_path, payload)

    def load_classifier_cache(self) -> dict[str, Any] | None:
        if not self.classifier_path.exists():
            return None
        try:
            with self.classifier_path.open("rb") as handle:
                payload = pickle.load(handle)
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
        atomic_replace(self.classifier_path, lambda tmp: tmp.write_bytes(pickle.dumps(data)))

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
