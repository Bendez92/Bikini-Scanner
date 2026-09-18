"""Explain, for one or more images, exactly why the scanner did or did not match them.

A score of zero has several very different causes - the age gate fired, the detail axes
saw nothing, the corroboration floor halved a lone weak axis - and the GUI only shows
the number. This runs the same region planning, axis scoring and cascade evaluation a
real scan runs, then prints every intermediate value and names the specific condition
that produced the outcome.

    python scripts/explain_image.py "C:/photos/pool.jpg" [more.jpg ...]

Nothing here writes to the folder store, the preferences file or the learning store, so
it is safe to point at any image.
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import numpy as np

REPO_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO_ROOT))

from bikini_scanner import cascade as cascade_module
from bikini_scanner.clip_backend import get_backend
from bikini_scanner.config import ScannerConfig
from bikini_scanner.image_formats import open_oriented
from bikini_scanner.regions import FULL_REGION, crop_regions, plan_regions
from bikini_scanner.scorer import BikiniScorer
from bikini_scanner.vision_analysis import detect_face_boxes

AXES = (
    "bikini",
    "bikini_top",
    "bikini_bottom",
    "midriff",
    "cleavage",
    "nsfw",
    "person",
    "female",
    "child",
    "adult",
)


def build_table(backend, scorer, paths, deep: bool):
    """Region table for these images, mirroring run_deep_pass without any caching."""
    count = len(paths)
    full_vectors = backend.embed_images([str(path) for path in paths], batch_size=count or 1)
    owner = list(range(count))
    keys = [FULL_REGION] * count
    rows = [np.asarray(full_vectors[index], dtype=np.float32) for index in range(count)]
    face_counts = np.zeros((count,), dtype=np.int32)
    region_report: dict[int, list[str]] = {index: [] for index in range(count)}

    if deep:
        pending: list[tuple[int, str]] = []
        crops = []
        for index, path in enumerate(paths):
            image = open_oriented(str(path))
            faces = detect_face_boxes(image)
            face_counts[index] = len(faces)
            planned = [
                region
                for region in plan_regions(image.size, faces, max_faces=int(scorer.config.max_faces))
                if region.key != FULL_REGION
            ]
            for key, crop in crop_regions(image, planned):
                pending.append((index, key))
                crops.append(crop)
                region_report[index].append(key)
        if crops:
            vectors = backend.embed_pil_images(crops)
            for (index, key), vector in zip(pending, vectors, strict=False):
                owner.append(index)
                keys.append(key)
                rows.append(np.asarray(vector, dtype=np.float32))

    matrix = np.vstack(rows).astype(np.float32)
    return scorer.build_region_table(matrix, owner, keys, count), face_counts, region_report


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("images", nargs="+")
    parser.add_argument("--model", default=None, help="override model_name")
    parser.add_argument("--no-deep", action="store_true", help="full frame only, no region crops")
    args = parser.parse_args()

    config = ScannerConfig()
    if args.model:
        config.model_name = args.model
    paths = [Path(image) for image in args.images]
    missing = [path for path in paths if not path.is_file()]
    if missing:
        for path in missing:
            print(f"not a file: {path}")
        return 2

    backend = get_backend(config)
    scorer = BikiniScorer(backend=backend, config=config)
    table, face_counts, region_report = build_table(backend, scorer, paths, deep=not args.no_deep)
    result = cascade_module.evaluate(table, config, face_counts)

    # These are the post-suppression values the pipeline actually gated on, not a fresh
    # unfiltered aggregation - otherwise the report disagrees with the decision.
    aggregated = result.axis_scores
    child = aggregated["evidence_child"]
    adult = aggregated["evidence_adult"]
    has_face = face_counts > 0
    subjects = cascade_module.analyse_subjects(table, config)

    for index, path in enumerate(paths):
        print("=" * 78)
        print(path)
        print(f"  faces detected : {int(face_counts[index])}")
        print(f"  regions        : {len(region_report[index])} crops {region_report[index] or '(full frame only)'}")
        print("  --- axis scores (raw sigmoid / evidence) ---")
        for axis in AXES:
            if axis not in aggregated:
                continue
            raw = float(aggregated[axis][index])
            print(f"    {axis:<15} {raw:6.3f}  ->  {float(cascade_module.evidence(np.array([raw]))[0]):6.3f}")
        print("  --- gates ---")
        for name in ("gate_person", "gate_female", "gate_age"):
            print(f"    {name:<15} {float(result.axis_scores[name][index]):6.3f}")
        print(f"    detail          {float(result.detail[index]):6.3f}")
        print(f"    final score     {float(result.score[index]):6.3f}   (threshold {config.threshold})")
        print(f"    stage           {result.stage[index]}  {result.reason[index] or ''}")

        if subjects is not None and bool(subjects.has_subjects[index]):
            print("  --- detected subjects (age from their own face crop) ---")
            for slot in range(subjects.subject_present.shape[1]):
                if not subjects.subject_present[index, slot]:
                    continue
                verdict = "MINOR - suppressed" if subjects.subject_minor[index, slot] else "adult"
                print(
                    f"    subject {slot}: child {subjects.subject_child[index, slot]:.3f}"
                    f"  adult {subjects.subject_adult[index, slot]:.3f}"
                    f"  detail {subjects.subject_detail[index, slot]:.3f}   -> {verdict}"
                )
            print(f"    full frame usable: {bool(subjects.full_allowed[index])}")

        if config.exclude_minors and (subjects is None or not subjects.has_subjects[index]):
            c = float(child[index])
            a = float(adult[index])
            d = float(result.detail[index])
            face = bool(has_face[index])
            checks = [
                (
                    "looks_minor",
                    (c >= config.minor_threshold) and (c > a + config.child_adult_margin),
                    f"child {c:.3f} >= {config.minor_threshold} and child > adult {a:.3f}"
                    f" + {config.child_adult_margin}",
                ),
                (
                    "strongly_minor",
                    c >= config.strongly_minor_threshold,
                    f"child {c:.3f} >= {config.strongly_minor_threshold}",
                ),
                (
                    "reads_younger",
                    face and (c > a + config.face_anchored_margin) and (c >= config.minor_threshold * 0.4),
                    f"face={face} and child {c:.3f} > adult {a:.3f} + {config.face_anchored_margin}"
                    f" and child >= {config.minor_threshold * 0.4:.3f}",
                ),
                (
                    "weak_adult",
                    face and (d >= config.weak_adult_detail) and (a < config.min_adult_confidence),
                    f"face={face} and detail {d:.3f} >= {config.weak_adult_detail}"
                    f" and adult {a:.3f} < {config.min_adult_confidence}",
                ),
            ]
            print("  --- whole-frame age gate (no subjects attributed) ---")
            for name, fired, detail_text in checks:
                print(f"    [{'X' if fired else ' '}] {name:<16} {detail_text}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
