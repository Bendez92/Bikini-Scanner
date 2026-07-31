from __future__ import annotations

import json
import logging
from urllib.error import URLError
from urllib.request import Request, urlopen

from .__version__ import __version__

LOGGER = logging.getLogger(__name__)


def _version_tuple(value: str) -> tuple[int, ...]:
    parts: list[int] = []
    for part in value.strip().lstrip("v").split("."):
        try:
            parts.append(int(part))
        except ValueError:
            parts.append(0)
    return tuple(parts)


def check_for_update(url: str, timeout: float = 3.0) -> dict[str, str] | None:
    if not url.strip():
        return None
    try:
        request = Request(url, headers={"Accept": "application/json", "User-Agent": "BikiniScanner"})
        with urlopen(request, timeout=timeout) as response:
            payload = json.loads(response.read().decode("utf-8"))
        latest = str(payload.get("latest_version", "")).strip()
        download_url = str(payload.get("download_url", "")).strip()
        if not latest or _version_tuple(latest) <= _version_tuple(__version__):
            return None
        return {"latest_version": latest, "download_url": download_url}
    except (OSError, URLError, ValueError, TypeError) as exc:
        LOGGER.warning("Update check failed: %s", exc)
        return None
