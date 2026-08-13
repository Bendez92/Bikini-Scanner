from __future__ import annotations

import base64
import html
import logging
import os
import re
import shutil
import tempfile
from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from datetime import datetime, timezone
from io import BytesIO
from pathlib import Path

from PIL import Image

from .image_formats import apply_orientation, open_oriented
from .store import IGNORE_MARKER_FILENAME

LABEL_NAMES = {1: "good", 0: "bad", 2: "skip"}


def _mark_ignored_directory(directory: Path) -> None:
    """Drop a marker so a future scan does not ingest its own output copies."""
    try:
        directory.mkdir(parents=True, exist_ok=True)
        (directory / IGNORE_MARKER_FILENAME).touch(exist_ok=True)
    except OSError:
        pass
OUTPUT_ORGANIZATIONS = {"flat", "score_band", "label", "score_band_label"}
DUPLICATE_POLICIES = {"skip", "rename", "overwrite"}
DEFAULT_HTML_EMBED_LIMIT = 512
LOGGER = logging.getLogger(__name__)


@dataclass(slots=True)
class OutputOptions:
    organization: str = "flat"
    score_band_low: float = 0.35
    score_band_high: float = 0.7
    filename_template: str = "{stem}"
    duplicate_policy: str = "rename"


@dataclass(slots=True)
class PlannedTransfer:
    source: Path
    destination: Path
    action: str
    label: str
    score: float
    band: str
    collision: bool = False
    reason: str = ""
    source_removed: bool = True
    # Set when this item failed during execution; the rest of the batch still runs.
    error: str = ""


def label_name(label: int | None) -> str:
    return LABEL_NAMES.get(int(label) if label is not None else -1, "unlabeled")


def score_band(score: float, low: float, high: float) -> str:
    if score >= high:
        return "high"
    if score >= low:
        return "medium"
    return "low"


def organization_parts(organization: str, score: float, label: str, low: float, high: float) -> list[str]:
    organization = organization if organization in OUTPUT_ORGANIZATIONS else "flat"
    band = score_band(score, low, high)
    if organization == "flat":
        return []
    if organization == "score_band":
        return [band]
    if organization == "label":
        return [label]
    return [label, band]


def _sanitize_filename(value: str) -> str:
    value = re.sub(r'[<>:"/\\|?*\x00]', "_", value).strip().strip(".")
    return value or "output"


def format_output_name(
    source: Path,
    score: float,
    label: str,
    index: int,
    timestamp: str | None,
    template: str,
) -> str:
    stamp = _parse_timestamp(timestamp)
    context = {
        "stem": source.stem,
        "name": source.name,
        "ext": source.suffix,
        "score": f"{score:.3f}",
        "score_pct": f"{score * 100:.1f}",
        "index": f"{index:04d}",
        "date": stamp.strftime("%Y%m%d"),
        "timestamp": stamp.strftime("%Y%m%d_%H%M%S"),
        "label": label,
        "label_name": label,
    }
    try:
        rendered = template.format_map(_SafeDict(context))
    except Exception:  # noqa: BLE001
        rendered = source.stem
    rendered = _sanitize_filename(rendered)
    if not rendered.lower().endswith(source.suffix.lower()):
        rendered = f"{rendered}{source.suffix}"
    return rendered


def _parse_timestamp(timestamp: str | None) -> datetime:
    if not timestamp:
        return datetime.now(timezone.utc)
    try:
        return datetime.fromisoformat(timestamp.replace("Z", "+00:00"))
    except Exception:  # noqa: BLE001
        return datetime.now(timezone.utc)


class _SafeDict(dict):
    def __missing__(self, key: str) -> str:
        return ""


def build_transfer_plan(
    sources: Sequence[str | Path],
    destination_root: str | Path,
    scores: Mapping[str, float],
    labels: Mapping[str, int],
    options: OutputOptions,
    timestamp: str | None = None,
    move: bool = False,
) -> list[PlannedTransfer]:
    root = Path(destination_root)
    plan: list[PlannedTransfer] = []
    seen: set[Path] = set()
    for index, raw_source in enumerate(sources):
        source = Path(raw_source)
        score = float(scores.get(str(source), 0.0))
        label = label_name(labels.get(str(source)))
        band = score_band(score, options.score_band_low, options.score_band_high)
        parts = organization_parts(options.organization, score, label, options.score_band_low, options.score_band_high)
        filename = format_output_name(source, score, label, index, timestamp, options.filename_template)
        destination_dir = root.joinpath(*parts) if parts else root
        destination = destination_dir / filename
        duplicate_policy = "rename" if options.duplicate_policy not in DUPLICATE_POLICIES else options.duplicate_policy
        collision = destination in seen or destination.exists()
        action = "move" if move else "copy"
        reason = ""
        if source.resolve() == destination.resolve():
            action = "skip"
            reason = "source and destination are identical"
        if action == "skip":
            pass
        elif duplicate_policy == "skip" and collision:
            action = "skip"
            reason = "duplicate exists"
        elif duplicate_policy == "rename":
            destination = _unique_destination(destination_dir, filename, seen)
            collision = collision or destination.exists()
            if collision:
                reason = "renamed"
        elif duplicate_policy == "overwrite" and collision:
            action = "overwrite"
        seen.add(destination)
        plan.append(
            PlannedTransfer(
                source=source,
                destination=destination,
                action=action,
                label=label,
                score=score,
                band=band,
                collision=collision,
                reason=reason,
            )
        )
    return plan


def _unique_destination(destination_dir: Path, filename: str, reserved: set[Path] | None = None) -> Path:
    destination = destination_dir / filename
    reserved = reserved or set()
    if not destination.exists() and destination not in reserved:
        return destination
    stem = Path(filename).stem
    suffix = Path(filename).suffix
    counter = 1
    while destination.exists() or destination in reserved:
        destination = destination_dir / f"{stem}_{counter}{suffix}"
        counter += 1
    return destination


def execute_transfer_plan(plan: Sequence[PlannedTransfer], move: bool = False) -> tuple[int, int, int, int]:
    """Run a transfer plan. Returns (processed, skipped, retained_sources, failed).

    One unreadable file, one locked destination or one full disk must not abandon the
    rest of the batch half-done with nothing said about it: every file is attempted
    independently, a failure is recorded on the item and counted, and the run continues.
    """
    processed = 0
    skipped = 0
    retained_sources = 0
    failed = 0
    destination_roots: set[Path] = set()
    for item in plan:
        if item.action == "skip":
            skipped += 1
            continue
        try:
            item.destination.parent.mkdir(parents=True, exist_ok=True)
            destination_roots.add(item.destination.parent)
            if item.destination.exists() and item.action == "overwrite":
                try:
                    item.destination.unlink()
                except Exception:  # noqa: BLE001
                    pass
            if move:
                try:
                    if item.source.parent == item.destination.parent:
                        os.replace(item.source, item.destination)
                    else:
                        shutil.move(str(item.source), str(item.destination))
                except Exception as exc:  # noqa: BLE001
                    LOGGER.warning(
                        "Move failed for %s -> %s; falling back to copy+delete: %s",
                        item.source,
                        item.destination,
                        exc,
                    )
                    # If this copy also fails it raises out to the per-item handler
                    # below, so the source is left untouched and the batch carries on.
                    shutil.copy2(item.source, item.destination)
                    try:
                        item.source.unlink()
                    except Exception as unlink_exc:  # noqa: BLE001
                        item.source_removed = False
                        retained_sources += 1
                        LOGGER.warning(
                            "Copied %s to %s but could not remove the source: %s",
                            item.source,
                            item.destination,
                            unlink_exc,
                        )
            else:
                shutil.copy2(item.source, item.destination)
        except Exception as exc:  # noqa: BLE001
            failed += 1
            item.error = str(exc)
            item.source_removed = False
            LOGGER.warning("Transfer failed for %s -> %s: %s", item.source, item.destination, exc)
            continue
        processed += 1
    for root in destination_roots:
        _mark_ignored_directory(root)
    return processed, skipped, retained_sources, failed


def build_html_report(
    output_path: str | Path,
    samples: Sequence[Mapping[str, object]],
    labels: Mapping[str, int],
    scores: Mapping[str, float],
    axis_scores: Mapping[str, Mapping[str, float]] | None = None,
    title: str = "Bikini Scanner report",
    thumb_size: int = 240,
    max_embedded_thumbnails: int | None = DEFAULT_HTML_EMBED_LIMIT,
    assets_dir: str | Path | None = None,
) -> Path:
    output = Path(output_path)
    use_assets = assets_dir is not None or (
        max_embedded_thumbnails is not None and len(samples) > max_embedded_thumbnails
    )
    assets_root = Path(assets_dir) if assets_dir is not None else output.with_name(f"{output.stem}_assets")
    if use_assets:
        assets_root.mkdir(parents=True, exist_ok=True)
        _mark_ignored_directory(assets_root)
    rows: list[str] = []
    for index, sample in enumerate(samples):
        path = Path(str(sample["path"]))
        if not path.exists():
            continue
        score = float(scores.get(str(path), float(sample.get("score", 0.0))))
        label = label_name(labels.get(str(path)))
        if use_assets:
            asset_path = assets_root / f"thumb_{index:06d}.jpg"
            _write_thumbnail(asset_path, path, thumb_size)
            relative_asset = os.path.relpath(asset_path, output.parent).replace(os.sep, "/")
            image_html = f"<img src='{html.escape(relative_asset)}' alt='{html.escape(path.name)}'>"
        else:
            image_html = _thumbnail_data_uri(path, thumb_size)
        axes_html = ""
        if axis_scores is not None:
            key_scores = axis_scores.get(str(path), {})
            axis_parts = [f"{html.escape(name)}: {float(value):.3f}" for name, value in key_scores.items()]
            if axis_parts:
                axes_html = f"<div class='axes'>{' • '.join(axis_parts)}</div>"
        rows.append(
            f"""
            <article class="card">
              <div class="thumb">{image_html}</div>
              <div class="meta">
                <div class="name">{html.escape(path.name)}</div>
                <div class="score">Score {score:.3f} • Label {html.escape(label)}</div>
                {axes_html}
                <div class="path">{html.escape(str(path))}</div>
              </div>
            </article>
            """
        )
    html_text = f"""<!doctype html>
<html>
<head>
<meta charset="utf-8">
<title>{html.escape(title)}</title>
<style>
body {{ font-family: sans-serif; background: #111; color: #eee; margin: 0; padding: 18px; }}
.grid {{ display: grid; grid-template-columns: repeat(auto-fill, minmax(280px, 1fr)); gap: 14px; }}
.card {{ background: #1f1f1f; border: 1px solid #333; border-radius: 10px; padding: 10px; display: flex; gap: 12px; }}
.thumb img {{ width: 120px; height: 120px; object-fit: cover; border-radius: 6px; }}
.name {{ font-weight: 700; margin-bottom: 4px; }}
.score, .axes, .path {{ font-size: 12px; color: #cfcfcf; margin-top: 4px; word-break: break-word; }}
</style>
</head>
<body>
<h1>{html.escape(title)}</h1>
<div class="grid">
{"".join(rows)}
</div>
</body>
</html>"""
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(html_text, encoding="utf-8")
    return output


def _thumbnail_data_uri(path: Path, thumb_size: int) -> str:
    buffer = _thumbnail_bytes(path, thumb_size)
    encoded = base64.b64encode(buffer).decode("ascii")
    return f"<img src='data:image/jpeg;base64,{encoded}' alt='{html.escape(path.name)}'>"


def _write_thumbnail(destination: Path, path: Path, thumb_size: int) -> None:
    destination.parent.mkdir(parents=True, exist_ok=True)
    destination.write_bytes(_thumbnail_bytes(path, thumb_size))


def _thumbnail_bytes(path: Path, thumb_size: int) -> bytes:
    image = open_oriented(path)
    image.thumbnail((thumb_size, thumb_size))
    buffer = BytesIO()
    image.save(buffer, format="JPEG", quality=75)
    return buffer.getvalue()


def write_image_metadata(path: str | Path, keyword: str, score: float | None = None) -> bool:
    source = Path(path)
    if not source.exists():
        return False
    suffix = source.suffix.lower()
    if suffix not in {".jpg", ".jpeg", ".tif", ".tiff", ".png", ".webp"}:
        return False
    if suffix in {".tif", ".tiff"} and _write_pyexiv2_metadata(source, keyword, score):
        return True
    tmp = None
    try:
        with Image.open(source) as handle:
            image = apply_orientation(handle)
            exif = image.getexif()
            exif[40094] = keyword.encode("utf-16le")
            if score is not None:
                exif[270] = f"score={score:.3f}"
            fd, tmp_name = tempfile.mkstemp(suffix=source.suffix, dir=str(source.parent))
            os.close(fd)
            tmp = Path(tmp_name)
            save_kwargs: dict[str, object] = {"exif": exif.tobytes()}
            if suffix == ".png":
                save_kwargs.pop("exif", None)
                pnginfo = _pnginfo(keyword, score)
                save_kwargs["pnginfo"] = pnginfo
            image.save(tmp, **save_kwargs)
        if suffix in {".jpg", ".jpeg"}:
            _inject_jpeg_xmp(tmp, keyword, score)
        os.replace(tmp, source)
        return True
    except Exception:  # noqa: BLE001
        if tmp is not None and tmp.exists():
            try:
                tmp.unlink()
            except Exception:  # noqa: BLE001
                pass
        return False


def _xmp_packet(keyword: str, score: float | None) -> bytes:
    escaped_keyword = html.escape(keyword, quote=True)
    score_property = f"<xmp:Rating>{score:.3f}</xmp:Rating>" if score is not None else ""
    xml = (
        '<?xpacket begin="\ufeff" id="W5M0MpCehiHzreSzNTczkc9d"?>'
        '<x:xmpmeta xmlns:x="adobe:ns:meta/">'
        '<rdf:RDF xmlns:rdf="http://www.w3.org/1999/02/22-rdf-syntax-ns#">'
        '<rdf:Description xmlns:dc="http://purl.org/dc/elements/1.1/" '
        'xmlns:lr="http://ns.adobe.com/lightroom/1.0/" '
        'xmlns:xmp="http://ns.adobe.com/xap/1.0/">'
        f"<dc:subject><rdf:Bag><rdf:li>{escaped_keyword}</rdf:li></rdf:Bag></dc:subject>"
        f"<lr:hierarchicalSubject><rdf:Bag><rdf:li>{escaped_keyword}</rdf:li></rdf:Bag></lr:hierarchicalSubject>"
        f"{score_property}"
        '</rdf:Description></rdf:RDF></x:xmpmeta><?xpacket end="w"?>'
    )
    return b"http://ns.adobe.com/xap/1.0/\x00" + xml.encode("utf-8")


def _inject_jpeg_xmp(path: Path, keyword: str, score: float | None) -> None:
    data = path.read_bytes()
    if not data.startswith(b"\xff\xd8"):
        return
    payload = _xmp_packet(keyword, score)
    segment_length = len(payload) + 2
    if segment_length > 0xFFFF:
        return
    segment = b"\xff\xe1" + segment_length.to_bytes(2, "big") + payload
    path.write_bytes(data[:2] + segment + data[2:])


def _write_pyexiv2_metadata(source: Path, keyword: str, score: float | None) -> bool:
    try:
        import pyexiv2
    except ImportError:
        # pyexiv2 is an optional extra; its absence is the normal case, not an error.
        return False
    tmp = None
    try:
        fd, tmp_name = tempfile.mkstemp(suffix=source.suffix, dir=str(source.parent))
        os.close(fd)
        tmp = Path(tmp_name)
        shutil.copyfile(source, tmp)
        metadata = pyexiv2.ImageMetadata(str(tmp))
        metadata.read()
        metadata["Xmp.dc.subject"] = [keyword]
        metadata["Xmp.lr.hierarchicalSubject"] = [keyword]
        if score is not None:
            metadata["Xmp.Rating"] = float(score)
        metadata.write()
        os.replace(tmp, source)
        return True
    except Exception as exc:  # noqa: BLE001
        LOGGER.warning("Optional pyexiv2 metadata write failed for %s: %s", source, exc)
        if tmp is not None and tmp.exists():
            try:
                tmp.unlink()
            except Exception:  # noqa: BLE001
                pass
        return False


def _pnginfo(keyword: str, score: float | None):
    from PIL import PngImagePlugin

    info = PngImagePlugin.PngInfo()
    info.add_text("Keywords", keyword)
    if score is not None:
        info.add_text("Score", f"{score:.3f}")
    return info


def trash_files(paths: Sequence[str | Path]) -> tuple[bool, str]:
    try:
        from send2trash import send2trash
    except Exception as exc:  # noqa: BLE001
        return False, str(exc)
    try:
        for path in paths:
            send2trash(str(path))
        return True, ""
    except Exception as exc:  # noqa: BLE001
        return False, str(exc)
