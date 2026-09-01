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
from .scorer import BikiniScorer, ScanCancelled, scan_and_score_folder
from .store import FolderStore

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


def run_headless(args: argparse.Namespace, config: ScannerConfig) -> int:
    configure_logging()
    folder = Path(args.folder).expanduser().resolve()
    if not folder.is_dir():
        emit(f"Folder does not exist: {folder}")
        return 2
    store = FolderStore(folder)
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
                "score": float(
                    cast(float, sample.get("score", state.scores[state_index] if state_index >= 0 else 0.0))
                ),
                "zero_shot_score": float(state.zero_shot_scores[state_index] if state_index >= 0 else 0.0),
                "axis_scores": axes,
                "label": labels.get(path),
                "matched": path in visible_match_set,
                "bucket": str(sample.get("bucket", "")),
            }
        )
    output.parent.mkdir(parents=True, exist_ok=True)
    if args.format == "json":
        output.write_text(json.dumps({"folder": str(folder), "images": records}, indent=2), encoding="utf-8")
    else:
        with output.open("w", newline="", encoding="utf-8") as handle:
            writer = csv.DictWriter(handle, fieldnames=("path", "filename", "score", "matched", "label"))
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
