from __future__ import annotations

import logging
import os
import sys
from logging.handlers import RotatingFileHandler
from pathlib import Path

APP_NAME = "bikini-scanner"
LOG_FILENAME = "bikini_scanner.log"


class RedactingFormatter(logging.Formatter):
    """Replace user home and well-known profile paths with '~' in log lines."""

    REPLACEMENT = "~"

    def __init__(self, fmt: str | None = None, datefmt: str | None = None) -> None:
        super().__init__(fmt, datefmt)
        self._roots: list[Path] = []
        home = Path.home()
        self._roots.append(home)
        for env in ("HOME", "USERPROFILE", "APPDATA", "LOCALAPPDATA", "XDG_CONFIG_HOME", "XDG_STATE_HOME"):
            value = os.environ.get(env)
            if value:
                try:
                    self._roots.append(Path(value).expanduser().resolve())
                except (OSError, ValueError):
                    self._roots.append(Path(value))
        # Longest first so a more specific path is replaced before its parent.
        self._roots.sort(key=lambda path: len(str(path)), reverse=True)

    def format(self, record: logging.LogRecord) -> str:
        message = super().format(record)
        for root in self._roots:
            text = str(root)
            if text and text in message:
                message = message.replace(text, self.REPLACEMENT)
        return message


def user_data_dir() -> Path:
    if os.name == "nt":
        base = Path(os.environ.get("LOCALAPPDATA") or Path.home() / "AppData" / "Local")
    elif sys.platform.startswith("darwin"):
        base = Path.home() / "Library" / "Logs"
    else:
        base = Path(os.environ.get("XDG_STATE_HOME") or Path.home() / ".local" / "state")
    return base / APP_NAME


def log_path() -> Path:
    return user_data_dir() / LOG_FILENAME


def configure_logging() -> Path:
    path = log_path()
    path.parent.mkdir(parents=True, exist_ok=True)
    root = logging.getLogger()
    if root.level == logging.NOTSET or root.level > logging.INFO:
        root.setLevel(logging.INFO)
    for handler in root.handlers:
        if isinstance(handler, RotatingFileHandler) and Path(handler.baseFilename) == path:
            return path
    handler = RotatingFileHandler(path, maxBytes=2 * 1024 * 1024, backupCount=3, encoding="utf-8")
    handler.setFormatter(RedactingFormatter("%(asctime)s %(levelname)s %(name)s: %(message)s"))
    root.addHandler(handler)
    return path


def read_log_tail(max_bytes: int = 200_000) -> str:
    path = log_path()
    if not path.exists():
        return ""
    try:
        with path.open("rb") as handle:
            handle.seek(0, 2)
            size = handle.tell()
            handle.seek(max(0, size - max_bytes))
            return handle.read().decode("utf-8", errors="replace")
    except OSError:
        return ""
