from __future__ import annotations

import json
import logging
import os
import tempfile
from collections.abc import Callable
from pathlib import Path
from typing import Any


def _fsync_and_replace(tmp_path: Path, destination: Path) -> None:
    """Flush the temporary file to disk, then atomically move it into place."""
    try:
        with tmp_path.open("r+b") as handle:
            os.fsync(handle.fileno())
    except OSError:
        pass
    os.replace(tmp_path, destination)


def atomic_write_text(path: str | Path, text: str, encoding: str = "utf-8") -> None:
    destination = Path(path)
    destination.parent.mkdir(parents=True, exist_ok=True)
    fd, tmp_name = tempfile.mkstemp(prefix=f".{destination.name}.", suffix=".tmp", dir=str(destination.parent))
    os.close(fd)
    tmp_path = Path(tmp_name)
    try:
        tmp_path.write_text(text, encoding=encoding)
        _fsync_and_replace(tmp_path, destination)
    except Exception:
        try:
            tmp_path.unlink()
        except OSError:
            pass
        raise


def atomic_write_json(path: str | Path, payload: Any) -> None:
    atomic_write_text(path, json.dumps(payload, indent=2, sort_keys=True))


def atomic_replace(path: str | Path, writer: Callable[[Path], None]) -> None:
    destination = Path(path)
    destination.parent.mkdir(parents=True, exist_ok=True)
    fd, tmp_name = tempfile.mkstemp(prefix=f".{destination.name}.", suffix=".tmp", dir=str(destination.parent))
    os.close(fd)
    tmp_path = Path(tmp_name)
    try:
        writer(tmp_path)
        _fsync_and_replace(tmp_path, destination)
    except Exception:
        try:
            tmp_path.unlink()
        except OSError:
            pass
        raise


def quarantine_broken_file(path: str | Path, logger: logging.Logger, reason: str) -> Path | None:
    source = Path(path)
    if not source.exists():
        return None
    suffix = ".broken"
    candidate = source.with_name(f"{source.name}{suffix}")
    counter = 1
    while candidate.exists():
        candidate = source.with_name(f"{source.name}{suffix}.{counter}")
        counter += 1
    try:
        source.replace(candidate)
        logger.warning("Preserved broken file %s as %s (%s)", source, candidate, reason)
        return candidate
    except OSError as exc:
        logger.warning("Failed to preserve broken file %s (%s): %s", source, reason, exc)
        return None
