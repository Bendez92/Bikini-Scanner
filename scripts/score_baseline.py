"""Record and compare a deterministic scoring baseline.

Refactoring a scoring pipeline is only safe if you can tell an intended change from an
accidental one. This walks a fixed synthetic corpus through the real scan pipeline with
the deterministic test backend and writes every score, axis value and cascade stage to
JSON. Re-run it after a change and diff:

    python scripts/score_baseline.py --write  tests/baseline_scores.json
    python scripts/score_baseline.py --compare tests/baseline_scores.json

`--compare` exits non-zero when anything moved by more than --tolerance, and prints the
rows that moved, so an unexpected change is visible immediately and an expected one can
be reviewed line by line before the baseline is regenerated.

The corpus deliberately includes EXIF-rotated copies: orientation handling is easy to
break silently and it changes results for a large share of real photographs.
"""

from __future__ import annotations

import argparse
import json
import os
import shutil
import sys
import tempfile
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO_ROOT))
sys.path.insert(0, str(REPO_ROOT / "tests"))

# Keep user state out of the way exactly as the test suite does, so a baseline run
# never reads or writes the real preferences, log or cross-folder learning store.
_STATE_DIR = Path(tempfile.mkdtemp(prefix="bikini_baseline_state_"))
for _variable in ("APPDATA", "LOCALAPPDATA", "XDG_CONFIG_HOME", "XDG_STATE_HOME", "HOME", "USERPROFILE"):
    os.environ[_variable] = str(_STATE_DIR)

from PIL import Image

from bikini_scanner.config import ScannerConfig
from bikini_scanner.scorer import BikiniScorer, scan_and_score_folder
from bikini_scanner.store import FolderStore

CORPUS_SIZE = 12
# Orientation 6 is "rotate 90 CW to display": the single most common value produced by
# phones held in portrait, and the case that breaks a pipeline which ignores EXIF.
EXIF_ORIENTATIONS = (1, 3, 6, 8)


def _gradient_image(index: int, width: int, height: int) -> Image.Image:
    image = Image.new("RGB", (width, height))
    pixels = image.load()
    for y in range(height):
        for x in range(width):
            value = (x * (index + 1) + y * (index + 2)) % 255
            pixels[x, y] = (value, (value * 3 + index * 17) % 255, (index * 30 + y) % 255)
    return image


def build_corpus(folder: Path) -> list[Path]:
    """A fixed, reproducible set of images including EXIF-rotated variants."""
    folder.mkdir(parents=True, exist_ok=True)
    sizes = [(320, 240), (240, 320), (400, 150), (150, 400), (256, 256), (300, 200)]
    paths: list[Path] = []
    for index in range(CORPUS_SIZE):
        width, height = sizes[index % len(sizes)]
        image = _gradient_image(index, width, height)
        path = folder / f"corpus_{index:02d}.jpg"
        orientation = EXIF_ORIENTATIONS[index % len(EXIF_ORIENTATIONS)]
        exif = image.getexif()
        exif[0x0112] = orientation
        image.save(path, quality=70, exif=exif)
        paths.append(path)
    duplicate = folder / "corpus_duplicate.jpg"
    shutil.copyfile(paths[0], duplicate)
    paths.append(duplicate)
    return sorted(paths)


def collect_baseline() -> dict[str, object]:
    from fake_backend import FakeBackend

    folder = Path(tempfile.mkdtemp(prefix="bikini_baseline_corpus_"))
    try:
        build_corpus(folder)
        config = ScannerConfig()
        config.preload_backend = False
        config.enable_face_detection = False
        backend = FakeBackend()
        scorer = BikiniScorer(backend, config)
        store = FolderStore(folder)
        state, samples = scan_and_score_folder(
            backend, store, scorer, threshold=float(config.threshold), batch_size=8
        )
        visible = scorer.state_visibility(state)
        rows = []
        for index, path in enumerate(state.paths):
            rows.append(
                {
                    "name": Path(path).name,
                    "score": round(float(state.scores[index]), 6),
                    "zero_shot": round(float(state.zero_shot_scores[index]), 6),
                    "stage": state.cascade_stage[index] if index < len(state.cascade_stage) else "",
                    "visible": bool(visible[index]) if index < len(visible) else True,
                    "detail_region": (
                        state.detail_regions[index] if index < len(state.detail_regions) else "full"
                    ),
                    "axes": {
                        axis: round(float(values[index]), 6)
                        for axis, values in sorted(state.axis_scores.items())
                        if index < len(values)
                    },
                }
            )
        rows.sort(key=lambda row: row["name"])
        return {
            "corpus_size": len(state.paths),
            "buckets": [str(sample.get("bucket", "")) for sample in samples],
            "rows": rows,
        }
    finally:
        shutil.rmtree(folder, ignore_errors=True)


def compare(current: dict[str, object], previous: dict[str, object], tolerance: float) -> int:
    differences: list[str] = []
    current_rows = {str(row["name"]): row for row in current["rows"]}  # type: ignore[index]
    previous_rows = {str(row["name"]): row for row in previous["rows"]}  # type: ignore[index]
    for name in sorted(set(current_rows) | set(previous_rows)):
        if name not in previous_rows:
            differences.append(f"  + {name} (new)")
            continue
        if name not in current_rows:
            differences.append(f"  - {name} (gone)")
            continue
        new_row, old_row = current_rows[name], previous_rows[name]
        for field in ("score", "zero_shot"):
            delta = abs(float(new_row[field]) - float(old_row[field]))
            if delta > tolerance:
                differences.append(
                    f"  ~ {name}.{field}: {old_row[field]:.6f} -> {new_row[field]:.6f}  (Δ {delta:.6f})"
                )
        for field in ("stage", "visible", "detail_region"):
            if new_row[field] != old_row[field]:
                differences.append(f"  ~ {name}.{field}: {old_row[field]!r} -> {new_row[field]!r}")
        for axis in sorted(set(new_row["axes"]) | set(old_row["axes"])):  # type: ignore[arg-type]
            new_value = float(new_row["axes"].get(axis, float("nan")))  # type: ignore[union-attr]
            old_value = float(old_row["axes"].get(axis, float("nan")))  # type: ignore[union-attr]
            if new_value != new_value or old_value != old_value:  # NaN means the axis appeared/vanished
                differences.append(f"  ~ {name}.axes.{axis}: {old_value} -> {new_value}")
            elif abs(new_value - old_value) > tolerance:
                differences.append(
                    f"  ~ {name}.axes.{axis}: {old_value:.6f} -> {new_value:.6f}"
                    f"  (Δ {abs(new_value - old_value):.6f})"
                )
    if current["buckets"] != previous["buckets"]:
        differences.append(f"  ~ review buckets: {previous['buckets']} -> {current['buckets']}")
    if not differences:
        print(f"Baseline matches ({current['corpus_size']} images, tolerance {tolerance}).")
        return 0
    print(f"Baseline differs in {len(differences)} place(s):")
    for line in differences:
        print(line)
    print("\nIf these changes are intended, re-run with --write to update the baseline.")
    return 1


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Record or compare a scoring baseline")
    group = parser.add_mutually_exclusive_group(required=True)
    group.add_argument("--write", metavar="PATH", help="Write a fresh baseline to PATH")
    group.add_argument("--compare", metavar="PATH", help="Compare the current pipeline against PATH")
    parser.add_argument(
        "--tolerance",
        type=float,
        default=1e-5,
        help="Absolute difference tolerated per value (default: 1e-5)",
    )
    args = parser.parse_args(argv)

    current = collect_baseline()
    if args.write:
        destination = Path(args.write)
        destination.parent.mkdir(parents=True, exist_ok=True)
        destination.write_text(json.dumps(current, indent=2, sort_keys=True), encoding="utf-8")
        print(f"Wrote baseline for {current['corpus_size']} images to {destination}")
        return 0
    source = Path(args.compare)
    if not source.is_file():
        print(f"No baseline at {source}; run with --write first.", file=sys.stderr)
        return 2
    previous = json.loads(source.read_text(encoding="utf-8"))
    return compare(current, previous, float(args.tolerance))


if __name__ == "__main__":
    raise SystemExit(main())
