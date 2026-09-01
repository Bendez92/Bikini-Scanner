"""Measure scan accuracy against your own labelled images.

`score_baseline.py` answers "did anything move?". This answers "is it any good?", which
is a different question and the one every threshold in `cascade.py` and `config.py` was
tuned against by hand. It runs the real pipeline over images you have labelled yourself
and reports ranking quality (ROC AUC, average precision), the operating point actually
shipped (precision/recall at the configured threshold), where the cascade gates dropped
labelled positives, and the individual images the scanner got most wrong.

The manifest is a CSV with a `path` column and a `label` column::

    path,label
    holiday/DSC_0001.jpg,match
    holiday/DSC_0002.jpg,no_match

Relative paths resolve against the manifest's own directory. Labels may be written as
`match`/`no_match`, `1`/`0`, `true`/`false`, `yes`/`no`, or `accept`/`reject`. Keep the
manifest and the images outside the repository: they are yours, and nothing here writes
them anywhere.

    python scripts/evaluate.py --manifest labels.csv
    python scripts/evaluate.py --manifest labels.csv --output run.json
    python scripts/evaluate.py --manifest labels.csv --compare run.json
    python scripts/evaluate.py --manifest labels.csv --sweep person_threshold --sweep-values 0.3,0.4,0.5

Learning is disabled by default (`global_learning=False`, `max_learning_weight=0`) so a
run measures the prompt-and-cascade pipeline itself and repeats identically. Pass
`--use-learning` to measure what a user with labels actually sees instead.

Scanning writes the ordinary `.bikini_scanner_cache/` next to your images, so a second
run over the same corpus reuses the embeddings and is much faster than the first.
"""

from __future__ import annotations

import argparse
import csv
import json
import sys
from collections import defaultdict
from collections.abc import Callable
from dataclasses import dataclass, field
from pathlib import Path

import numpy as np

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from bikini_scanner.backend_utils import ImageEmbeddingBackend
from bikini_scanner.cascade import STAGE_MINOR, STAGE_NO_PERSON, STAGE_NOT_FEMALE, STAGE_SCORED
from bikini_scanner.config import ScannerConfig
from bikini_scanner.config_profiles import profile_config
from bikini_scanner.linear_model import average_precision, roc_auc
from bikini_scanner.scorer import BikiniScorer, ScanProgress, scan_and_score_folder
from bikini_scanner.store import FolderStore

POSITIVE_LABELS = {"1", "match", "true", "yes", "accept", "accepted", "positive", "good"}
NEGATIVE_LABELS = {"0", "no_match", "nomatch", "false", "no", "reject", "rejected", "negative", "bad"}
STAGES = (STAGE_SCORED, STAGE_NO_PERSON, STAGE_NOT_FEMALE, STAGE_MINOR)
CURVE_POINTS = 21

# The report is written straight out as JSON, so it is modelled as JSON rather than as
# classes: `Metrics` holds the numbers, `Report` is the whole document.
JsonValue = str | int | float | bool | None | list["JsonValue"] | dict[str, "JsonValue"]
Metrics = dict[str, float | int | None]
Case = dict[str, JsonValue]
Report = dict[str, JsonValue]


@dataclass(slots=True)
class Row:
    """One labelled image after it has been through the pipeline."""

    path: str
    name: str
    label: int
    score: float
    zero_shot: float
    stage: str
    visible: bool
    detail_region: str
    axes: dict[str, float] = field(default_factory=dict)

    def predicted(self, threshold: float) -> bool:
        """What the app would actually show: an excluded image is never a match."""
        return self.visible and self.score >= threshold


def _parse_label(raw: str, source: str, line_number: int) -> int:
    value = raw.strip().lower()
    if value in POSITIVE_LABELS:
        return 1
    if value in NEGATIVE_LABELS:
        return 0
    raise ValueError(f"{source}:{line_number}: unrecognised label {raw!r}; use match/no_match, 1/0, true/false")


def load_manifest(path: Path) -> dict[str, int]:
    """Read the labelled corpus. Returns resolved path -> label."""
    base = path.resolve().parent
    labels: dict[str, int] = {}
    with path.open(newline="", encoding="utf-8") as handle:
        reader = csv.DictReader(handle)
        if reader.fieldnames is None or "path" not in reader.fieldnames or "label" not in reader.fieldnames:
            raise ValueError(f"{path}: expected a CSV header with 'path' and 'label' columns")
        for line_number, record in enumerate(reader, 2):
            raw_path = (record.get("path") or "").strip()
            if not raw_path:
                continue
            image = Path(raw_path).expanduser()
            if not image.is_absolute():
                image = base / image
            labels[str(image.resolve())] = _parse_label(record.get("label") or "", path.name, line_number)
    if not labels:
        raise ValueError(f"{path}: no labelled rows found")
    return labels


def build_config(args: argparse.Namespace) -> ScannerConfig:
    config = profile_config(args.profile) if args.profile else ScannerConfig()
    if config is None:
        raise ValueError(f"Unknown profile: {args.profile}")
    if args.threshold is not None:
        config.threshold = float(args.threshold)
    if args.backend:
        config.backend = args.backend
    config.preload_backend = False
    if not args.use_learning:
        # Isolate the prompt-and-cascade pipeline: a learned model trained on whatever
        # labels happen to sit beside the images would make the numbers unrepeatable.
        config.global_learning = False
        config.max_learning_weight = 0.0
    return config


def _as_bool(raw: str) -> bool:
    value = raw.strip().lower()
    if value in {"1", "true", "yes", "on"}:
        return True
    if value in {"0", "false", "no", "off"}:
        return False
    raise ValueError(f"Expected a true/false value, got {raw!r}")


def _set_threshold(config: ScannerConfig, raw: str) -> None:
    config.threshold = float(raw)


def _set_zero_shot_scale(config: ScannerConfig, raw: str) -> None:
    config.zero_shot_scale = float(raw)


def _set_zero_shot_weight(config: ScannerConfig, raw: str) -> None:
    config.zero_shot_weight = float(raw)


def _set_classifier_weight(config: ScannerConfig, raw: str) -> None:
    config.classifier_weight = float(raw)


def _set_person_threshold(config: ScannerConfig, raw: str) -> None:
    config.person_threshold = float(raw)


def _set_person_gate_threshold(config: ScannerConfig, raw: str) -> None:
    config.person_gate_threshold = float(raw)


def _set_female_threshold(config: ScannerConfig, raw: str) -> None:
    config.female_threshold = float(raw)


def _set_nsfw_threshold(config: ScannerConfig, raw: str) -> None:
    config.nsfw_threshold = float(raw)


def _set_refine_band(config: ScannerConfig, raw: str) -> None:
    config.refine_band = float(raw)


def _set_refine_weight(config: ScannerConfig, raw: str) -> None:
    config.refine_weight = float(raw)


def _set_detail_strongest_weight(config: ScannerConfig, raw: str) -> None:
    config.detail_strongest_weight = float(raw)


def _set_detail_average_weight(config: ScannerConfig, raw: str) -> None:
    config.detail_average_weight = float(raw)


def _set_max_learning_weight(config: ScannerConfig, raw: str) -> None:
    config.max_learning_weight = float(raw)


def _set_batch_size(config: ScannerConfig, raw: str) -> None:
    config.batch_size = int(raw)


def _set_max_faces(config: ScannerConfig, raw: str) -> None:
    config.max_faces = int(raw)


def _set_refine_max_images(config: ScannerConfig, raw: str) -> None:
    config.refine_max_images = int(raw)


def _set_require_person(config: ScannerConfig, raw: str) -> None:
    config.require_person = _as_bool(raw)


def _set_require_female(config: ScannerConfig, raw: str) -> None:
    config.require_female = _as_bool(raw)


def _set_enable_face_detection(config: ScannerConfig, raw: str) -> None:
    config.enable_face_detection = _as_bool(raw)


def _set_deep_scan(config: ScannerConfig, raw: str) -> None:
    config.deep_scan = raw.strip()


# The keys a sweep may touch, each with the setter that knows its type. This is a
# deliberate allowlist rather than a generic attribute write: it keeps the scoring knobs
# sweepable while leaving the backend, model, prompts, pipeline, VLM networking, plugin
# execution and the age gate out of reach of a command-line string.
SWEEPABLE: dict[str, Callable[[ScannerConfig, str], None]] = {
    "threshold": _set_threshold,
    "zero_shot_scale": _set_zero_shot_scale,
    "zero_shot_weight": _set_zero_shot_weight,
    "classifier_weight": _set_classifier_weight,
    "person_threshold": _set_person_threshold,
    "person_gate_threshold": _set_person_gate_threshold,
    "female_threshold": _set_female_threshold,
    "nsfw_threshold": _set_nsfw_threshold,
    "refine_band": _set_refine_band,
    "refine_weight": _set_refine_weight,
    "refine_max_images": _set_refine_max_images,
    "detail_strongest_weight": _set_detail_strongest_weight,
    "detail_average_weight": _set_detail_average_weight,
    "max_learning_weight": _set_max_learning_weight,
    "batch_size": _set_batch_size,
    "max_faces": _set_max_faces,
    "require_person": _set_require_person,
    "require_female": _set_require_female,
    "enable_face_detection": _set_enable_face_detection,
    "deep_scan": _set_deep_scan,
}


def apply_override(config: ScannerConfig, key: str, raw_value: str) -> None:
    """Set one sweepable config field from a command-line string."""
    setter = SWEEPABLE.get(key)
    if setter is None:
        raise ValueError(f"{key} cannot be swept; choose one of: {', '.join(sorted(SWEEPABLE))}")
    setter(config, raw_value)


def _make_backend(config: ScannerConfig, fake: bool) -> ImageEmbeddingBackend:
    if fake:
        # Only useful for checking that this script itself runs; the fake backend
        # invents embeddings, so its numbers mean nothing.
        tests_dir = REPO_ROOT / "tests"
        if str(tests_dir) not in sys.path:
            sys.path.insert(0, str(tests_dir))
        from fake_backend import FakeBackend

        return FakeBackend()
    from bikini_scanner.clip_backend import get_backend

    return get_backend(config)


def _progress_printer(folder: Path, quiet: bool) -> Callable[[ScanProgress], None]:
    """Print at most one line per phase per 10% so a long real scan is not silent."""
    seen: dict[str, int] = {}

    def report(progress: ScanProgress) -> None:
        if quiet:
            return
        step = int(progress.percent // 10)
        if seen.get(progress.phase) == step:
            return
        seen[progress.phase] = step
        print(f"  {folder.name}: {progress.text()} ({progress.percent:.0f}%)")

    return report


def score_corpus(labels: dict[str, int], config: ScannerConfig, fake_backend: bool, quiet: bool) -> list[Row]:
    """Run the real scan over every folder the manifest mentions and keep labelled rows."""
    by_folder: dict[Path, list[str]] = defaultdict(list)
    for image in labels:
        by_folder[Path(image).parent].append(image)

    backend = _make_backend(config, fake_backend)
    rows: list[Row] = []
    found: set[str] = set()
    for folder in sorted(by_folder):
        if not folder.is_dir():
            print(f"WARNING: folder does not exist, skipping: {folder}", file=sys.stderr)
            continue
        if not quiet:
            print(f"Scanning {folder} ({len(by_folder[folder])} labelled images)")
        scorer = BikiniScorer(backend, config)
        store = FolderStore(folder)
        state, _samples = scan_and_score_folder(
            backend,
            store,
            scorer,
            threshold=float(config.threshold),
            batch_size=int(config.batch_size),
            progress_callback=_progress_printer(folder, quiet),
        )
        visible = scorer.state_visibility(state)
        for index, scanned_path in enumerate(state.paths):
            key = str(Path(scanned_path).resolve())
            if key not in labels:
                continue
            found.add(key)
            rows.append(
                Row(
                    path=key,
                    name=Path(key).name,
                    label=labels[key],
                    score=float(state.scores[index]),
                    zero_shot=float(state.zero_shot_scores[index]),
                    stage=state.cascade_stage[index] if index < len(state.cascade_stage) else STAGE_SCORED,
                    visible=bool(visible[index]) if index < len(visible) else True,
                    detail_region=state.detail_regions[index] if index < len(state.detail_regions) else "full",
                    axes={
                        axis: float(values[index])
                        for axis, values in sorted(state.axis_scores.items())
                        if index < len(values)
                    },
                )
            )
    for missing in sorted(set(labels) - found):
        print(f"WARNING: labelled image was not scored (unreadable or moved): {missing}", file=sys.stderr)
    rows.sort(key=lambda row: row.path)
    return rows


def confusion(rows: list[Row], threshold: float) -> dict[str, int]:
    counts = {"tp": 0, "fp": 0, "tn": 0, "fn": 0}
    for row in rows:
        predicted = row.predicted(threshold)
        if row.label == 1:
            counts["tp" if predicted else "fn"] += 1
        else:
            counts["fp" if predicted else "tn"] += 1
    return counts


def _ratio(numerator: int, denominator: int) -> float:
    return float(numerator) / float(denominator) if denominator else 0.0


def curve(rows: list[Row], points: int = CURVE_POINTS) -> list[dict[str, float]]:
    """Precision/recall across the threshold range, so the shipped one can be judged."""
    if not rows:
        return []
    scores = sorted({round(row.score, 4) for row in rows})
    if len(scores) > points:
        indices = np.linspace(0, len(scores) - 1, points).round().astype(int)
        scores = [scores[index] for index in dict.fromkeys(indices.tolist())]
    curve_rows: list[dict[str, float]] = []
    for threshold in scores:
        counts = confusion(rows, threshold)
        curve_rows.append(
            {
                "threshold": float(threshold),
                "precision": _ratio(counts["tp"], counts["tp"] + counts["fp"]),
                "recall": _ratio(counts["tp"], counts["tp"] + counts["fn"]),
                "matches": float(counts["tp"] + counts["fp"]),
            }
        )
    return curve_rows


def gate_table(rows: list[Row]) -> dict[str, dict[str, int]]:
    """Where each labelled image ended up. A positive in a non-`scored` stage is a
    false negative the gates caused, which no threshold change can recover."""
    table = {stage: {"positive": 0, "negative": 0} for stage in STAGES}
    for row in rows:
        bucket = table.setdefault(row.stage, {"positive": 0, "negative": 0})
        bucket["positive" if row.label == 1 else "negative"] += 1
    return table


def evaluate(rows: list[Row], threshold: float) -> Metrics:
    labels = np.asarray([row.label for row in rows], dtype=np.int64)
    scores = np.asarray([row.score for row in rows], dtype=np.float32)
    zero_shot = np.asarray([row.zero_shot for row in rows], dtype=np.float32)
    counts = confusion(rows, threshold)
    precision = _ratio(counts["tp"], counts["tp"] + counts["fp"])
    recall = _ratio(counts["tp"], counts["tp"] + counts["fn"])
    # roc_auc refuses corpora it cannot rank meaningfully; report that as "no number"
    # rather than a misleading one, and say so in the printed report.
    rankable = bool(labels.size >= 4 and 0 < int(labels.sum()) < int(labels.size))
    return {
        "threshold": float(threshold),
        "images": len(rows),
        "positives": int(labels.sum()),
        "negatives": int(labels.size - labels.sum()),
        "auc": roc_auc(labels, scores) if rankable else None,
        "auc_zero_shot": roc_auc(labels, zero_shot) if rankable else None,
        "average_precision": average_precision(labels, scores) if rankable else None,
        "precision": precision,
        "recall": recall,
        "f1": (2 * precision * recall / (precision + recall)) if precision + recall else 0.0,
        "accuracy": _ratio(counts["tp"] + counts["tn"], len(rows)),
        **counts,
    }


def worst_cases(rows: list[Row], threshold: float, top: int) -> dict[str, list[Case]]:
    """The images to look at first: confident mistakes in both directions."""
    false_positives = sorted(
        (row for row in rows if row.label == 0 and row.predicted(threshold)),
        key=lambda row: row.score,
        reverse=True,
    )
    false_negatives = sorted(
        (row for row in rows if row.label == 1 and not row.predicted(threshold)),
        key=lambda row: row.score,
    )

    def describe(row: Row) -> Case:
        return {
            "name": row.name,
            "path": row.path,
            "score": round(row.score, 4),
            "zero_shot": round(row.zero_shot, 4),
            "stage": row.stage,
            "detail_region": row.detail_region,
            "axes": {axis: round(value, 4) for axis, value in row.axes.items()},
        }

    return {
        "false_positives": [describe(row) for row in false_positives[:top]],
        "false_negatives": [describe(row) for row in false_negatives[:top]],
    }


def build_report(rows: list[Row], config: ScannerConfig, args: argparse.Namespace) -> Report:
    threshold = float(config.threshold)
    return {
        "manifest": str(Path(args.manifest).resolve()),
        "settings": {
            "profile": args.profile or "",
            "backend": "fake" if args.fake_backend else config.backend,
            "model_name": config.model_name,
            "threshold": threshold,
            "pipeline": config.pipeline,
            "exclude_minors": bool(config.exclude_minors),
            "learning": bool(args.use_learning),
            "fake_backend": bool(args.fake_backend),
        },
        "metrics": evaluate(rows, threshold),
        "gates": gate_table(rows),
        "curve": curve(rows),
        "worst": worst_cases(rows, threshold, args.top),
        "rows": [
            {
                "name": row.name,
                "label": row.label,
                "score": round(row.score, 6),
                "zero_shot": round(row.zero_shot, 6),
                "stage": row.stage,
                "visible": row.visible,
            }
            for row in rows
        ],
    }


def _format_metric(value: float | None) -> str:
    return "n/a" if value is None else f"{value:.4f}"


def print_report(report: Report) -> None:
    metrics = report["metrics"]
    settings = report["settings"]
    print()
    print(f"Corpus: {metrics['images']} labelled images ({metrics['positives']} match, {metrics['negatives']} not)")
    print(
        f"Backend {settings['backend']}, threshold {metrics['threshold']:.3f}, learning {'on' if settings['learning'] else 'off'}"
    )
    print()
    print("Ranking quality (threshold-independent)")
    print(f"  ROC AUC            {_format_metric(metrics['auc'])}")
    print(f"  ROC AUC zero-shot  {_format_metric(metrics['auc_zero_shot'])}")
    print(f"  Average precision  {_format_metric(metrics['average_precision'])}")
    if metrics["auc"] is None:
        print("  (needs at least four labelled images with both labels present)")
    print()
    print(f"Operating point at threshold {metrics['threshold']:.3f}")
    print(f"  precision {metrics['precision']:.4f}   recall {metrics['recall']:.4f}   F1 {metrics['f1']:.4f}")
    print(f"  TP {metrics['tp']}  FP {metrics['fp']}  FN {metrics['fn']}  TN {metrics['tn']}")
    print()
    print("Cascade stages (a labelled match outside 'scored' is a gate-caused miss)")
    for stage, counts in report["gates"].items():
        if counts["positive"] or counts["negative"]:
            print(f"  {stage:<12} match {counts['positive']:>5}   not-match {counts['negative']:>5}")
    print()
    print("Precision / recall across thresholds")
    for point in report["curve"]:
        print(
            f"  {point['threshold']:.3f}  precision {point['precision']:.3f}  "
            f"recall {point['recall']:.3f}  ({int(point['matches'])} shown)"
        )
    for title, key in (("Worst false positives", "false_positives"), ("Worst false negatives", "false_negatives")):
        entries = report["worst"][key]
        if not entries:
            continue
        print()
        print(title)
        for entry in entries:
            axes = ", ".join(
                f"{axis} {value:.2f}" for axis, value in entry["axes"].items() if axis.startswith("evidence_")
            )
            print(f"  {entry['score']:.3f}  {entry['name']}  [{entry['stage']}/{entry['detail_region']}]  {axes}")


def compare(current: Report, previous: Report, tolerance: float) -> int:
    """Report what a change did to the numbers. Non-zero exit means something moved."""
    moved: list[str] = []
    for key in ("auc", "auc_zero_shot", "average_precision", "precision", "recall", "f1"):
        new_value = current["metrics"].get(key)
        old_value = previous["metrics"].get(key)
        if new_value is None or old_value is None:
            if new_value != old_value:
                moved.append(f"  ~ {key}: {old_value} -> {new_value}")
            continue
        if abs(float(new_value) - float(old_value)) > tolerance:
            moved.append(f"  ~ {key}: {float(old_value):.4f} -> {float(new_value):.4f}")
    for key in ("tp", "fp", "fn", "tn"):
        if current["metrics"][key] != previous["metrics"][key]:
            moved.append(f"  ~ {key}: {previous['metrics'][key]} -> {current['metrics'][key]}")
    current_rows = {row["name"]: row for row in current["rows"]}
    previous_rows = {row["name"]: row for row in previous["rows"]}
    for name in sorted(set(current_rows) | set(previous_rows)):
        if name not in previous_rows:
            moved.append(f"  + {name} (new)")
        elif name not in current_rows:
            moved.append(f"  - {name} (gone)")
        elif abs(current_rows[name]["score"] - previous_rows[name]["score"]) > tolerance:
            moved.append(f"  ~ {name}: {previous_rows[name]['score']:.4f} -> {current_rows[name]['score']:.4f}")
    print()
    if not moved:
        print(f"No change beyond tolerance {tolerance}.")
        return 0
    print(f"Changed in {len(moved)} place(s) vs the previous run:")
    for line in moved:
        print(line)
    return 1


def run_sweep(labels: dict[str, int], args: argparse.Namespace) -> int:
    values = [value.strip() for value in args.sweep_values.split(",") if value.strip()]
    if not values:
        print("--sweep-values must list at least one value", file=sys.stderr)
        return 2
    print(f"Sweeping {args.sweep} over {values}")
    results: list[tuple[str, Metrics]] = []
    for value in values:
        config = build_config(args)
        apply_override(config, args.sweep, value)
        rows = score_corpus(labels, config, args.fake_backend, quiet=True)
        results.append((value, evaluate(rows, float(config.threshold))))
    print()
    print(f"{args.sweep:<24} {'AUC':>8} {'AP':>8} {'prec':>8} {'recall':>8} {'F1':>8}   TP/FP/FN")
    for value, metrics in results:
        print(
            f"{value:<24} {_format_metric(metrics['auc']):>8} {_format_metric(metrics['average_precision']):>8} "
            f"{metrics['precision']:>8.4f} {metrics['recall']:>8.4f} {metrics['f1']:>8.4f}   "
            f"{metrics['tp']}/{metrics['fp']}/{metrics['fn']}"
        )
    return 0


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="evaluate.py",
        description="Measure scan accuracy against a labelled corpus.",
        epilog=(
            "examples:\n"
            "  python scripts/evaluate.py --manifest labels.csv --output run.json\n"
            "  python scripts/evaluate.py --manifest labels.csv --compare run.json\n"
            "  python scripts/evaluate.py --manifest labels.csv --sweep threshold --sweep-values 0.4,0.5,0.6\n"
        ),
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    parser.add_argument("--manifest", required=True, help="CSV with 'path' and 'label' columns")
    parser.add_argument("--profile", default="", help="Settings profile to evaluate (default: built-in defaults)")
    parser.add_argument("--threshold", type=float, default=None, help="Override the match threshold")
    parser.add_argument("--backend", default="", help="Override the backend, e.g. clip-onnx")
    parser.add_argument(
        "--use-learning",
        action="store_true",
        help="Keep active learning on (default: off, so runs are repeatable)",
    )
    parser.add_argument("--top", type=int, default=10, help="How many worst false positives/negatives to list")
    parser.add_argument("--output", default="", help="Write the full report as JSON to this path")
    parser.add_argument("--compare", default="", help="Compare this run against a JSON report written earlier")
    parser.add_argument("--tolerance", type=float, default=0.005, help="Absolute movement tolerated by --compare")
    parser.add_argument(
        "--sweep",
        default="",
        choices=sorted(SWEEPABLE),
        metavar="KEY",
        help="Scoring setting to sweep; one of: " + ", ".join(sorted(SWEEPABLE)),
    )
    parser.add_argument("--sweep-values", default="", help="Comma-separated values for --sweep")
    parser.add_argument(
        "--fake-backend",
        action="store_true",
        help="Use the deterministic test backend to smoke-test this script (the numbers are meaningless)",
    )
    parser.add_argument("--quiet", action="store_true", help="Suppress per-phase scan progress")
    return parser


def main(argv: list[str] | None = None) -> int:
    parser = build_parser()
    args = parser.parse_args(argv)
    manifest = Path(args.manifest).expanduser()
    if not manifest.is_file():
        print(f"No manifest at {manifest}", file=sys.stderr)
        return 2
    try:
        labels = load_manifest(manifest)
    except (OSError, ValueError) as error:
        print(f"ERROR: {error}", file=sys.stderr)
        return 2

    if args.sweep:
        try:
            return run_sweep(labels, args)
        except ValueError as error:
            print(f"ERROR: {error}", file=sys.stderr)
            return 2

    try:
        config = build_config(args)
    except ValueError as error:
        print(f"ERROR: {error}", file=sys.stderr)
        return 2
    rows = score_corpus(labels, config, args.fake_backend, args.quiet)
    if not rows:
        print("No labelled image could be scored.", file=sys.stderr)
        return 2
    report = build_report(rows, config, args)
    print_report(report)
    if args.output:
        destination = Path(args.output).expanduser()
        destination.parent.mkdir(parents=True, exist_ok=True)
        destination.write_text(json.dumps(report, indent=2, sort_keys=True), encoding="utf-8")
        print(f"\nWrote {destination}")
    if args.compare:
        source = Path(args.compare).expanduser()
        if not source.is_file():
            print(f"No report at {source}; run with --output first.", file=sys.stderr)
            return 2
        return compare(report, json.loads(source.read_text(encoding="utf-8")), float(args.tolerance))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
