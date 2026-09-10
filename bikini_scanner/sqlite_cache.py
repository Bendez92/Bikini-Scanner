"""SQLite-backed replacement for monolithic NPZ/JSON per-folder caches.

The legacy cache stored every embedding, the path index, face counts and region
embeddings in a few large files. Rewriting those files on every incremental save
gave O(n^2) write amplification for large folders. This module keeps the same data
in ordinary SQLite tables, so an incremental update touches only the changed rows.

Migration from the old NPZ/JSON files is automatic and one-shot. Once the SQLite
file exists, the legacy files are removed so there is no ambiguity about which
store is authoritative.
"""

from __future__ import annotations

import json
import logging
import os
import sqlite3
import threading
from io import BytesIO
from pathlib import Path
from typing import Any

import numpy as np

from .safe_io import resolved_str

LOGGER = logging.getLogger(__name__)


class _SQLiteTransaction:
    """Hold the cache lock across a BEGIN ... COMMIT/ROLLBACK sequence."""

    # `lock` is the cache's re-entrant lock; typed loosely because threading.RLock is a
    # factory, so its instance type is private in typeshed.
    def __init__(self, lock: Any, connection_factory: Any) -> None:
        self._lock = lock
        self._connection_factory = connection_factory
        self._connection: sqlite3.Connection | None = None

    def __enter__(self) -> sqlite3.Connection:
        self._lock.acquire()
        self._connection = self._connection_factory()
        self._connection.execute("BEGIN")
        return self._connection

    def __exit__(self, exc_type: Any, exc: Any, tb: Any) -> None:
        try:
            if self._connection is not None:
                if exc_type is None:
                    self._connection.commit()
                else:
                    self._connection.rollback()
        finally:
            self._lock.release()


class SQLiteCache:
    """Thread-safe per-folder cache for embeddings, index, regions and face counts."""

    def __init__(self, db_path: Path) -> None:
        self.db_path = db_path
        self.db_path.parent.mkdir(parents=True, exist_ok=True)
        # Re-entrant: clear() has to hold the lock across purge + close + unlink +
        # schema rebuild, and both purge() and _transaction() take it themselves.
        # With a plain Lock the schema rebuild happened *outside* the lock, so a
        # concurrent reader could connect to a database that had no tables yet.
        self._lock = threading.RLock()
        self._connection: sqlite3.Connection | None = None
        self._ensure_tables()

    def _connect(self) -> sqlite3.Connection:
        # check_same_thread is disabled because we guard the connection with our own
        # lock. This is simpler than a connection-per-thread pool for a single local DB.
        if self._connection is None:
            self._connection = sqlite3.connect(
                str(self.db_path), check_same_thread=False, isolation_level=None
            )
            if os.environ.get("BIKINI_SCANNER_TEST_SQLITE_PRAGMAS") == "1":
                self._connection.execute("PRAGMA journal_mode=MEMORY")
                self._connection.execute("PRAGMA synchronous=OFF")
            else:
                self._connection.execute("PRAGMA journal_mode=WAL")
                self._connection.execute("PRAGMA synchronous=NORMAL")
        return self._connection

    def _transaction(self) -> _SQLiteTransaction:
        return _SQLiteTransaction(self._lock, self._connect)

    def _execute(
        self,
        sql: str,
        parameters: tuple[Any, ...] | list[Any] = (),
        conn: sqlite3.Connection | None = None,
    ) -> sqlite3.Cursor:
        if conn is not None:
            return conn.execute(sql, parameters)
        with self._lock:
            return self._connect().execute(sql, parameters)

    # Reads must fetch inside the lock. _execute released it as soon as the statement
    # was issued and handed the cursor back, so the caller's .fetchall() ran unguarded
    # on a connection another thread could be using — the scan worker writes while the
    # main thread answers Tools > Duplicate groups, which shares one connection. That
    # interleaving returned truncated rows (np.load raising "No data left in file") and
    # dropped writes.
    def _fetchall(
        self, sql: str, parameters: tuple[Any, ...] | list[Any] = ()
    ) -> list[tuple[Any, ...]]:
        with self._lock:
            return self._connect().execute(sql, parameters).fetchall()

    def _fetchone(
        self, sql: str, parameters: tuple[Any, ...] | list[Any] = ()
    ) -> tuple[Any, ...] | None:
        with self._lock:
            return self._connect().execute(sql, parameters).fetchone()

    def _executemany(
        self,
        sql: str,
        parameters: list[tuple[Any, ...]],
        conn: sqlite3.Connection | None = None,
    ) -> sqlite3.Cursor:
        if conn is not None:
            return conn.executemany(sql, parameters)
        with self._lock:
            return self._connect().executemany(sql, parameters)

    def _ensure_tables(self) -> None:
        with self._transaction() as conn:
            conn.execute(
                """
                CREATE TABLE IF NOT EXISTS embeddings (
                    content_hash TEXT PRIMARY KEY,
                    embedding BLOB NOT NULL
                )
                """
            )
            conn.execute(
                """
                CREATE TABLE IF NOT EXISTS image_records (
                    path TEXT PRIMARY KEY,
                    content_hash TEXT NOT NULL,
                    mtime_ns INTEGER NOT NULL,
                    size INTEGER NOT NULL
                )
                """
            )
            conn.execute(
                """
                CREATE INDEX IF NOT EXISTS idx_image_records_hash
                ON image_records(content_hash)
                """
            )
            conn.execute(
                """
                CREATE TABLE IF NOT EXISTS face_counts (
                    content_hash TEXT PRIMARY KEY,
                    face_count INTEGER NOT NULL
                )
                """
            )
            conn.execute(
                """
                CREATE TABLE IF NOT EXISTS region_embeddings (
                    content_hash TEXT NOT NULL,
                    namespace TEXT NOT NULL,
                    region_key TEXT NOT NULL,
                    embedding BLOB NOT NULL,
                    PRIMARY KEY (content_hash, namespace, region_key)
                )
                """
            )

    @staticmethod
    def _array_to_blob(array: np.ndarray) -> bytes:
        buffer = BytesIO()
        np.save(buffer, array, allow_pickle=False)
        return buffer.getvalue()

    @staticmethod
    def _blob_to_array(blob: bytes) -> np.ndarray:
        return np.load(BytesIO(blob), allow_pickle=False).astype(np.float32)

    def _has_data(self) -> bool:
        for table in ("embeddings", "image_records", "face_counts", "region_embeddings"):
            row = self._fetchone(f"SELECT COUNT(*) FROM {table}")
            if row is not None and row[0]:
                return True
        return False

    def migrate_from_legacy(
        self,
        embeddings_path: Path,
        index_path: Path,
        region_path: Path,
        face_counts_path: Path,
    ) -> bool:
        """Import existing NPZ/JSON caches once, then remove them."""
        has_legacy = any(
            path.is_file()
            for path in (embeddings_path, index_path, region_path, face_counts_path)
        )
        if not has_legacy:
            return False
        if self._has_data():
            # Prefer the existing database; clean up any leftover legacy files.
            for path in (embeddings_path, index_path, region_path, face_counts_path):
                try:
                    path.unlink(missing_ok=True)
                except OSError:
                    pass
            return False

        LOGGER.info("Migrating legacy NPZ/JSON caches to %s", self.db_path)
        try:
            with self._transaction() as conn:
                if embeddings_path.is_file():
                    with np.load(embeddings_path, allow_pickle=False) as archive:
                        rows: list[tuple[Any, ...]] = [
                            (str(key), self._array_to_blob(archive[key].astype(np.float32)))
                            for key in archive.files
                        ]
                    if rows:
                        self._executemany(
                            "INSERT OR REPLACE INTO embeddings(content_hash, embedding) VALUES (?, ?)",
                            rows,
                            conn,
                        )

                if index_path.is_file():
                    payload = json.loads(index_path.read_text(encoding="utf-8"))
                    if isinstance(payload, dict):
                        rows = [
                            (
                                str(path),
                                str(record.get("content_hash", "")),
                                int(record.get("mtime_ns", 0)),
                                int(record.get("size", 0)),
                            )
                            for path, record in payload.items()
                            if isinstance(record, dict)
                        ]  # type: ignore[misc]
                        if rows:
                            self._executemany(
                                """
                                INSERT OR REPLACE INTO image_records(path, content_hash, mtime_ns, size)
                                VALUES (?, ?, ?, ?)
                                """,
                                rows,
                                conn,
                            )

                if face_counts_path.is_file():
                    payload = json.loads(face_counts_path.read_text(encoding="utf-8"))
                    if isinstance(payload, dict):
                        rows = [
                            (str(content_hash), int(value))
                            for content_hash, value in payload.items()
                        ]
                        if rows:
                            self._executemany(
                                "INSERT OR REPLACE INTO face_counts(content_hash, face_count) VALUES (?, ?)",
                                rows,
                                conn,
                            )

                if region_path.is_file():
                    with np.load(region_path, allow_pickle=False) as archive:
                        rows = []
                        for key in archive.files:
                            parts = str(key).split("|", 2)
                            if len(parts) != 3:
                                continue
                            content_hash, namespace, region_key = parts
                            rows.append(
                                (
                                    content_hash,
                                    namespace,
                                    region_key,
                                    self._array_to_blob(archive[key].astype(np.float32)),
                                )
                            )
                        if rows:
                            self._executemany(
                                """
                                INSERT OR REPLACE INTO region_embeddings(content_hash, namespace, region_key, embedding)
                                VALUES (?, ?, ?, ?)
                                """,
                                rows,
                                conn,
                            )
        except Exception as exc:  # noqa: BLE001
            LOGGER.warning("Legacy cache migration failed, starting fresh: %s", exc)
            return False

        for path in (embeddings_path, index_path, region_path, face_counts_path):
            try:
                path.unlink(missing_ok=True)
            except OSError:
                pass
        return True

    def save_scan_cache(
        self,
        content_embeddings: dict[str, np.ndarray],
        path_records: dict[Path, dict[str, int | str]],
        face_counts_by_content_hash: dict[str, int] | None = None,
    ) -> None:
        if not content_embeddings and not path_records and not face_counts_by_content_hash:
            return
        with self._transaction() as conn:
            if content_embeddings:
                rows: list[tuple[Any, ...]] = [
                    (str(content_hash), self._array_to_blob(np.asarray(embedding, dtype=np.float32)))
                    for content_hash, embedding in content_embeddings.items()
                ]
                self._executemany(
                    "INSERT OR REPLACE INTO embeddings(content_hash, embedding) VALUES (?, ?)",
                    rows,
                    conn,
                )
            if path_records:
                rows = [
                    (
                        resolved_str(path),
                        str(record["content_hash"]),
                        int(record["mtime_ns"]),
                        int(record["size"]),
                    )
                    for path, record in path_records.items()
                ]
                self._executemany(
                    """
                    INSERT OR REPLACE INTO image_records(path, content_hash, mtime_ns, size)
                    VALUES (?, ?, ?, ?)
                    """,
                    rows,
                    conn,
                )
            if face_counts_by_content_hash:
                rows = [
                    (str(content_hash), int(face_count))
                    for content_hash, face_count in face_counts_by_content_hash.items()
                ]
                self._executemany(
                    "INSERT OR REPLACE INTO face_counts(content_hash, face_count) VALUES (?, ?)",
                    rows,
                    conn,
                )

    # SQLite caps the number of bound parameters per statement (999 on older builds,
    # 32766 on current ones). A folder large enough to exceed it — and this app warns
    # about folders of 20 000 — used to fail the very first cache lookup with
    # "too many SQL variables" before any work had been done, so the IN clause is
    # issued in chunks well under the smallest of those limits.
    _QUERY_CHUNK = 500

    def get_cached_image_records(
        self, paths: list[Path]
    ) -> dict[Path, dict[str, object]]:
        if not paths:
            return {}
        path_strings = [resolved_str(path) for path in paths]
        by_path: dict[str, tuple[str, int, int, np.ndarray]] = {}
        for start in range(0, len(path_strings), self._QUERY_CHUNK):
            chunk = path_strings[start : start + self._QUERY_CHUNK]
            placeholders = ",".join("?" * len(chunk))
            rows = self._fetchall(
                f"""
                SELECT path, content_hash, mtime_ns, size, embedding
                FROM image_records
                JOIN embeddings USING (content_hash)
                WHERE path IN ({placeholders})
                """,
                tuple(chunk),
            )
            for row in rows:
                by_path[row[0]] = (row[1], row[2], row[3], self._blob_to_array(row[4]))
        cached: dict[Path, dict[str, object]] = {}
        # Zipped against the strings computed above rather than resolving a second time.
        for path, resolved in zip(paths, path_strings, strict=True):
            entry = by_path.get(resolved)
            if entry is None:
                continue
            content_hash, mtime_ns, size, embedding = entry
            try:
                stat = path.stat()
            except OSError:
                continue
            if stat.st_mtime_ns != mtime_ns or stat.st_size != size:
                continue
            cached[path] = {
                "path": path,
                "content_hash": content_hash,
                "embedding": embedding,
                "mtime_ns": stat.st_mtime_ns,
                "size": stat.st_size,
            }
        return cached

    def _load_all_embeddings(self) -> dict[str, np.ndarray]:
        rows = self._fetchall("SELECT content_hash, embedding FROM embeddings")
        return {str(row[0]): self._blob_to_array(row[1]) for row in rows}

    def _load_all_image_records(self) -> dict[Path, dict[str, int | str]]:
        rows = self._fetchall("SELECT path, content_hash, mtime_ns, size FROM image_records")
        return {
            Path(str(row[0])): {
                "path": str(row[0]),
                "content_hash": str(row[1]),
                "mtime_ns": int(row[2]),
                "size": int(row[3]),
            }
            for row in rows
        }

    def lookup_content_embedding(self, content_hash: str) -> np.ndarray | None:
        row = self._fetchone(
            "SELECT embedding FROM embeddings WHERE content_hash = ?",
            (str(content_hash),),
        )
        if row is None:
            return None
        return self._blob_to_array(row[0])

    def lookup_face_count(self, content_hash: str) -> int | None:
        row = self._fetchone(
            "SELECT face_count FROM face_counts WHERE content_hash = ?",
            (str(content_hash),),
        )
        if row is None:
            return None
        return int(row[0])

    def load_face_counts(self) -> dict[str, int]:
        rows = self._fetchall("SELECT content_hash, face_count FROM face_counts")
        return {str(row[0]): int(row[1]) for row in rows}

    def save_region_embeddings(self, entries: dict[str, np.ndarray]) -> None:
        if not entries:
            return
        rows = []
        for key, value in entries.items():
            parts = str(key).split("|", 2)
            if len(parts) != 3:
                continue
            content_hash, namespace, region_key = parts
            rows.append(
                (
                    content_hash,
                    namespace,
                    region_key,
                    self._array_to_blob(np.asarray(value, dtype=np.float32)),
                )
            )
        if not rows:
            return
        with self._transaction() as conn:
            self._executemany(
                """
                INSERT OR REPLACE INTO region_embeddings(content_hash, namespace, region_key, embedding)
                VALUES (?, ?, ?, ?)
                """,
                rows,
                conn,
            )

    def lookup_region_embeddings(self, content_hash: str, namespace: str) -> dict[str, np.ndarray]:
        rows = self._fetchall(
            """
            SELECT region_key, embedding FROM region_embeddings
            WHERE content_hash = ? AND namespace = ?
            """,
            (str(content_hash), str(namespace)),
        )
        return {str(row[0]): self._blob_to_array(row[1]) for row in rows}

    def purge(self) -> None:
        """Empty every table without needing the file to be deletable.

        Deleting the database is the tidier reset, but it only works while nothing else
        holds the file open - on Windows an open handle in another FolderStore makes the
        unlink fail, and a cache that was supposed to be discarded would survive with a
        warning. Emptying the tables first means the discard is real either way.
        """
        with self._lock:
            try:
                conn = self._connect()
                with conn:
                    # Table names are literals from this tuple, never caller input.
                    for table in ("embeddings", "image_records", "face_counts", "region_embeddings"):
                        conn.execute(f"DELETE FROM {table}")
            except sqlite3.Error as exc:
                LOGGER.warning("Could not empty the cache DB %s: %s", self.db_path, exc)

    def clear(self) -> None:
        """Empty the cache and leave a usable, empty database behind.

        The caller keeps using this object — FolderStore discards derived caches from
        `ensure_embedding_namespace` in the middle of a live scan. Without the schema
        rebuild at the end, the next `_connect()` opened a brand-new file that
        `_ensure_tables` had never run against, and every following query died with
        "no such table: image_records". Switching model in Settings and rescanning the
        same folder hit exactly that, and left the folder's cache broken for the rest
        of the session.

        The whole sequence is held under the lock. Rebuilding the schema outside it left
        a window in which a reader on another thread could connect to a database with no
        tables in it - reachable because Tools > Clear cached scan data is not blocked
        while a scan is running.
        """
        with self._lock:
            self.purge()
            try:
                if self._connection is not None:
                    self._connection.close()
                    self._connection = None
            except sqlite3.Error:
                pass
            try:
                self.db_path.unlink(missing_ok=True)
            except OSError as exc:
                LOGGER.warning("Could not remove cache DB %s: %s", self.db_path, exc)
            # Safe under the lock: it is re-entrant, and purge()/_transaction() take it.
            self._ensure_tables()

    def close(self) -> None:
        with self._lock:
            if self._connection is not None:
                try:
                    self._connection.close()
                except sqlite3.Error:
                    pass
                self._connection = None
