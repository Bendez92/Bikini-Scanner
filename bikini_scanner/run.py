from __future__ import annotations

import argparse
import csv
import json
import logging
import signal
import sys
import threading
from pathlib import Path
from typing import cast

from .config import ScannerConfig
from .config_profiles import profile_config
from .logging_setup import configure_logging
from .output_ops import (
    OutputOptions,
    build_html_report,
    build_transfer_plan,
    execute_transfer_plan,
    write_image_metadata,
)
from .plugins import apply_plugins
from .safe_io import atomic_write_json
from .scorer import BikiniScorer, ScanCancelled, scan_and_score_folder
from .store import FolderStore, collect_image_paths

LOGGER = logging.getLogger(__name__)


def emit(message: str) -> None:
    """Report progress to whatever the caller can actually see.

    A windowed PyInstaller build has no stdout, so a bare print() raises and kills
    headless mode outright. Everything goes to the log either way, which is where a
    packaged run can be inspected afterwards.
    """
    LOGGER.info("%s", message)
    stream = sys.stdout
    if stream is None:
        return
    try:
        print(message)
    except Exception:  # noqa: BLE001
        # A failed write to stdout (closed pipe, dead console) has nowhere left to be
        # reported to; the log file already has the same information.
        pass


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(prog="bikini_scanner", description="Local bikini-content scanner")
    parser.add_argument("--folder", help="Preselect a folder in the GUI", default="")
    parser.add_argument("--threshold", type=float, default=None, help="Initial match threshold")
    parser.add_argument("--headless", action="store_true", help="Scan and export without launching the GUI")
    parser.add_argument("--output", default="", help="CSV/JSON output file or transfer destination")
    parser.add_argument("--format", choices=("csv", "json"), default="csv", help="Headless result format")
    parser.add_argument("--organization", choices=("flat", "score_band", "label", "score_band_label"), default="flat")
    parser.add_argument("--copy", action="store_true", help="Copy matches to the output destination")
    parser.add_argument("--move", action="store_true", help="Move matches to the output destination")
    parser.add_argument("--html-report", default="", help="Write an HTML report to this path")
    parser.add_argument(
        "--dry-run", action="store_true", help="Print the transfer plan without copying or moving files"
    )
    parser.add_argument(
        "--write-metadata", action="store_true", help="Write bikini keyword metadata into matched images"
    )
    parser.add_argument("--profile", default="", help="Saved or built-in profile name")
    # Headless and interactive review could not previously feed each other: a scan run
    # from a script had no way to take the decisions someone made in the GUI, and
    # decisions made headlessly had no way out. These two make them compose.
    parser.add_argument(
        "--import-labels",
        default="",
        help="Merge a JSON map of {image path: 0|1|2} into this folder's labels before scoring",
    )
    parser.add_argument(
        "--export-labels", default="", help="Write this folder's labels to a JSON file after scoring"
    )
    parser.add_argument(
        "--rerank",
        action="store_true",
        help="Re-rank using cached embeddings only; fail rather than embed images that are not cached yet",
    )
    parser.add_argument(
        "--estimate",
        action="store_true",
        help="Report what a scan of this folder would involve, then exit without scanning",
    )
    return parser


def main(
    argv: list[str] | None = None,
    config_override: ScannerConfig | None = None,
    enforced_backend: str | None = None,
) -> int:
    parser = build_parser()
    args = parser.parse_args(argv)
    config = profile_config(args.profile) if args.profile else (config_override or ScannerConfig())
    if config is None:
        parser.error(f"Unknown profile: {args.profile}")
    if args.threshold is not None:
        config.threshold = float(args.threshold)
    if enforced_backend is not None:
        config.backend = enforced_backend
    if args.headless:
        if not args.folder:
            parser.error("--folder is required with --headless")
        return run_headless(args, config)
    from .gui import launch_gui

    launch_gui(config=config, initial_folder=args.folder)
    return 0


def main_onnx(argv: list[str] | None = None, config_override: ScannerConfig | None = None) -> int:
    args = list(sys.argv[1:] if argv is None else argv)
    config = config_override or ScannerConfig(backend="clip-onnx")
    config.backend = "clip-onnx"
    if "--headless" in args:
        return main(args, config_override=config, enforced_backend="clip-onnx")
    folder = ""
    threshold = None
    if "--folder" in args:
        index = args.index("--folder")
        if index + 1 < len(args):
            folder = args[index + 1]
    if "--threshold" in args:
        index = args.index("--threshold")
        if index + 1 < len(args):
            threshold = float(args[index + 1])
    from .gui import launch_gui

    launch_gui(config=config, initial_folder=folder, initial_threshold=threshold)
    return 0


def _merge_labels(store: FolderStore, source: Path) -> int:
    """Fold an external label map into the folder's own, newest wins.

    Only 0/1/2 are accepted and every key is resolved against this folder, so a file
    exported from one machine still applies on another where the paths differ.
    """
    payload = json.loads(source.read_text(encoding="utf-8"))
    if not isinstance(payload, dict):
        raise ValueError("expected a JSON object of {path: 0|1|2}")
    labels = store.load_labels()
    merged = 0
    for raw_path, raw_label in payload.items():
        try:
            value = int(raw_label)
        except (TypeError, ValueError):
            continue
        if value not in (0, 1, 2):
            continue
        candidate = Path(str(raw_path))
        # Accept a bare filename or a path from another machine: what identifies the
        # image here is its location in the folder being scanned.
        resolved = candidate if candidate.is_absolute() and candidate.exists() else store.folder / candidate.name
        if not resolved.exists():
            continue
        labels[str(resolved.resolve())] = value
        merged += 1
    if merged:
        store.save_labels(labels)
    return merged


def _estimate_scan(folder: Path, store: FolderStore) -> int:
    """Say what a scan would involve without doing any of it.

    --dry-run already covered transfers; there was no equivalent for the scan itself,
    so the only way to find out how long a folder would take was to start it.
    """
    paths = collect_image_paths(folder)
    if not paths:
        emit(f"No supported images found in {folder}")
        return 0
    cached = 0
    try:
        cached = store.cached_path_count(paths)
    except Exception:  # noqa: BLE001
        cached = 0
    labels = store.load_labels()
    emit(f"Folder:            {folder}")
    emit(f"Images found:      {len(paths):,}")
    emit(f"Already embedded:  {cached:,}  (these are read from the cache)")
    emit(f"Need embedding:    {len(paths) - cached:,}")
    emit(f"Decisions on file: {len(labels):,}")
    if cached >= len(paths):
        emit("Everything is cached, so a scan would only re-rank — expect seconds, not minutes.")
    else:
        emit("Embedding is the slow part; the rest of the scan is comparatively quick.")
    return 0


def run_headless(args: argparse.Namespace, config: ScannerConfig) -> int:
    configure_logging()
    folder = Path(args.folder).expanduser().resolve()
    if not folder.is_dir():
        emit(f"Folder does not exist: {folder}")
        return 2
    store = FolderStore(folder)
    if args.import_labels:
        try:
            merged = _merge_labels(store, Path(args.import_labels).expanduser())
        except (OSError, ValueError) as exc:
            emit(f"Could not import labels from {args.import_labels}: {exc}")
            return 2
        emit(f"Imported {merged} label(s) from {args.import_labels}")
    if args.estimate:
        return _estimate_scan(folder, store)
    if args.rerank and not store.cache_db_path.exists():
        emit(f"--rerank needs an existing scan cache in {folder}; run a scan first.")
        return 2
    from .clip_backend import get_backend
    backend = get_backend(config)
    scorer = BikiniScorer(backend, config)
    cancel_event = threading.Event()
    previous_sigint = signal.getsignal(signal.SIGINT)

    def request_cancel(_signum, _frame) -> None:
        cancel_event.set()

    signal.signal(signal.SIGINT, request_cancel)
    try:
        state, samples = scan_and_score_folder(
            backend,
            store,
            scorer,
            threshold=float(config.threshold),
            batch_size=config.batch_size,
            cancel_event=cancel_event,
        )
    except ScanCancelled:
        emit("Scan cancelled; cached work was flushed. Run the scan again to resume.")
        return 130
    finally:
        signal.signal(signal.SIGINT, previous_sigint)
    samples = apply_plugins(state, samples, enabled=config.enable_plugins)
    threshold = float(config.threshold)
    # state_visibility also applies the cascade exclusions (age gate, sex gate).
    visible_mask = scorer.state_visibility(state)
    visible_matches = [
        path
        for path, score, include in zip(state.paths, state.scores, visible_mask, strict=False)
        if include and float(score) >= threshold
    ]
    visible_match_set = set(visible_matches)
    output = Path(args.output).expanduser() if args.output else folder / f"bikini_results.{args.format}"
    labels = store.load_labels()
    if args.export_labels:
        destination = Path(args.export_labels).expanduser()
        try:
            atomic_write_json(destination, dict(sorted(labels.items())))
            emit(f"Wrote {len(labels)} label(s) to {destination}")
        except OSError as exc:
            emit(f"Could not write labels to {destination}: {exc}")
            return 2
    notes = store.load_notes()
    path_index = {path: index for index, path in enumerate(state.paths)}
    records = []
    for sample in samples:
        path = str(sample.get("path", ""))
        if not path:
            continue
        state_index = path_index.get(path, -1)
        axes = (
            {name: float(values[state_index]) for name, values in state.axis_scores.items()} if state_index >= 0 else {}
        )
        records.append(
            {
                "path": path,
                "filename": Path(path).name,
                "score": float(cast(float, sample.get("score", state.scores[state_index] if state_index >= 0 else 0.0))),
                "zero_shot_score": float(state.zero_shot_scores[state_index] if state_index >= 0 else 0.0),
                "axis_scores": axes,
                "label": labels.get(path),
                "note": notes.get(path, ""),
                "matched": path in visible_match_set,
                "bucket": str(sample.get("bucket", "")),
            }
        )
    output.parent.mkdir(parents=True, exist_ok=True)
    if args.format == "json":
        output.write_text(json.dumps({"folder": str(folder), "images": records}, indent=2), encoding="utf-8")
    else:
        with output.open("w", newline="", encoding="utf-8") as handle:
            # "note" travels with the row here for the same reason it does in the GUI
            # export: a record of the decision without the reasoning is half the answer.
            writer = csv.DictWriter(handle, fieldnames=("path", "filename", "score", "matched", "label", "note"))
            writer.writeheader()
            for record in records:
                writer.writerow({key: record[key] for key in writer.fieldnames})
    scores = {path: float(score) for path, score in zip(state.paths, state.scores, strict=False)}
    if args.html_report:
        axis_scores = {
            path: {
                axis_name: float(values[index])
                for axis_name, values in state.axis_scores.items()
                if index < len(values)
            }
            for index, path in enumerate(state.paths)
        }
        report_path = build_html_report(
            Path(args.html_report).expanduser(),
            samples,
            labels,
            scores,
            axis_scores=axis_scores,
            title="Bikini Scanner report",
            match_threshold=threshold,
        )
        emit(f"HTML report written to {report_path}")
    if args.write_metadata:
        written = sum(1 for path in visible_matches if write_image_metadata(path, "bikini", score=scores.get(path)))
        emit(f"Metadata written to {written}/{len(visible_matches)} matched files.")
    if args.copy or args.move or args.dry_run:
        if args.copy and args.move:
            emit("Choose only one of --copy or --move")
            return 2
        options = OutputOptions(organization=args.organization)
        plan = build_transfer_plan(visible_matches, output.parent / "matches", scores, labels, options, move=args.move)
        if args.dry_run:
            emit(f"Dry-run transfer plan ({len(plan)} items):")
            for item in plan:
                emit(f"{item.action}: {item.source} -> {item.destination} ({item.reason})")
        else:
            processed, skipped, retained, failed = execute_transfer_plan(plan, move=args.move)
            if retained:
                emit(f"Warning: {retained} source file(s) could not be removed after move fallback.")
            for item in plan:
                if item.error:
                    emit(f"Failed: {item.source} -> {item.destination}: {item.error}")
            emit(f"Transfer complete: {processed} processed, {skipped} skipped, {failed} failed.")
            if failed:
                return 1
    emit(f"Scanned {len(state.paths)} images; {len(visible_match_set)} matches; wrote {output}")
    return 0
