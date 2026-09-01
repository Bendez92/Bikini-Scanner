"""Checks for the accuracy harness in `scripts/evaluate.py`.

The harness produces the numbers a tuning change is judged by, so its arithmetic is
worth pinning: a confusion matrix that quietly counts an age-excluded positive as a
true negative would flatter every future change. The end-to-end case runs the real scan
pipeline over generated images with the fake backend, the same way the rest of the suite
does, so no model weights are needed.
"""

from __future__ import annotations

import json
import os
import sys
import tempfile
import unittest
from pathlib import Path

import numpy as np
from PIL import Image

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "scripts"))

_STATE_DIR = Path(tempfile.mkdtemp(prefix="bikini_eval_state_"))
os.environ["BIKINI_SCANNER_TEST_SQLITE_PRAGMAS"] = "1"
os.environ["APPDATA"] = str(_STATE_DIR)
os.environ["LOCALAPPDATA"] = str(_STATE_DIR)
os.environ["XDG_CONFIG_HOME"] = str(_STATE_DIR)
os.environ["XDG_STATE_HOME"] = str(_STATE_DIR)
os.environ["HOME"] = str(_STATE_DIR)
os.environ["USERPROFILE"] = str(_STATE_DIR)

import evaluate

from bikini_scanner.cascade import STAGE_MINOR, STAGE_SCORED


def make_row(
    name: str,
    label: int,
    score: float,
    *,
    stage: str = STAGE_SCORED,
    visible: bool = True,
) -> evaluate.Row:
    return evaluate.Row(
        path=f"/corpus/{name}",
        name=name,
        label=label,
        score=score,
        zero_shot=score,
        stage=stage,
        visible=visible,
        detail_region="full",
        axes={"evidence_person": score},
    )


class ManifestTests(unittest.TestCase):
    def setUp(self) -> None:
        self.folder = Path(tempfile.mkdtemp(prefix="bikini_manifest_"))

    def write(self, text: str) -> Path:
        path = self.folder / "labels.csv"
        path.write_text(text, encoding="utf-8")
        return path

    def test_relative_paths_resolve_against_the_manifest(self) -> None:
        manifest = self.write("path,label\nimages/a.jpg,match\n")
        labels = evaluate.load_manifest(manifest)
        expected = str((self.folder / "images" / "a.jpg").resolve())
        self.assertEqual(labels, {expected: 1})

    def test_label_spellings(self) -> None:
        manifest = self.write("path,label\na.jpg,MATCH\nb.jpg,no_match\nc.jpg,1\nd.jpg,false\ne.jpg,Reject\n")
        self.assertEqual(sorted(evaluate.load_manifest(manifest).values()), [0, 0, 0, 1, 1])

    def test_unknown_label_is_rejected_with_its_line_number(self) -> None:
        manifest = self.write("path,label\na.jpg,match\nb.jpg,maybe\n")
        with self.assertRaises(ValueError) as caught:
            evaluate.load_manifest(manifest)
        self.assertIn("labels.csv:3", str(caught.exception))

    def test_missing_columns_are_rejected(self) -> None:
        manifest = self.write("file,verdict\na.jpg,match\n")
        with self.assertRaises(ValueError):
            evaluate.load_manifest(manifest)

    def test_empty_manifest_is_rejected(self) -> None:
        manifest = self.write("path,label\n")
        with self.assertRaises(ValueError):
            evaluate.load_manifest(manifest)


class MetricTests(unittest.TestCase):
    def test_excluded_positive_counts_as_a_false_negative(self) -> None:
        """An age-excluded image scores 0 and is invisible; it is a miss, not a pass."""
        rows = [
            make_row("gated.jpg", 1, 0.0, stage=STAGE_MINOR, visible=False),
            make_row("shown.jpg", 1, 0.9),
        ]
        counts = evaluate.confusion(rows, 0.5)
        self.assertEqual((counts["tp"], counts["fn"], counts["fp"], counts["tn"]), (1, 1, 0, 0))

    def test_a_high_score_that_a_gate_hid_is_not_a_false_positive(self) -> None:
        rows = [make_row("gated.jpg", 0, 0.95, stage=STAGE_MINOR, visible=False)]
        counts = evaluate.confusion(rows, 0.5)
        self.assertEqual((counts["fp"], counts["tn"]), (0, 1))

    def test_metrics_on_a_perfect_ranking(self) -> None:
        rows = [
            make_row("a.jpg", 1, 0.9),
            make_row("b.jpg", 1, 0.8),
            make_row("c.jpg", 0, 0.1),
            make_row("d.jpg", 0, 0.05),
        ]
        metrics = evaluate.evaluate(rows, 0.5)
        self.assertAlmostEqual(metrics["auc"] or 0.0, 1.0, places=6)
        self.assertAlmostEqual(metrics["precision"], 1.0, places=6)
        self.assertAlmostEqual(metrics["recall"], 1.0, places=6)
        self.assertAlmostEqual(metrics["f1"], 1.0, places=6)
        self.assertEqual(metrics["images"], 4)
        self.assertEqual((metrics["positives"], metrics["negatives"]), (2, 2))

    def test_auc_is_none_when_the_corpus_cannot_be_ranked(self) -> None:
        """One class, or fewer than four images, has no meaningful AUC."""
        single_class = [make_row(f"{index}.jpg", 1, 0.9) for index in range(6)]
        self.assertIsNone(evaluate.evaluate(single_class, 0.5)["auc"])
        too_small = [make_row("a.jpg", 1, 0.9), make_row("b.jpg", 0, 0.1)]
        metrics = evaluate.evaluate(too_small, 0.5)
        self.assertIsNone(metrics["auc"])
        self.assertIsNone(metrics["average_precision"])
        self.assertAlmostEqual(metrics["precision"], 1.0, places=6)

    def test_zero_division_is_reported_as_zero_not_raised(self) -> None:
        rows = [make_row(f"{index}.jpg", index % 2, index / 100.0) for index in range(6)]
        metrics = evaluate.evaluate(rows, 0.9)
        self.assertEqual((metrics["precision"], metrics["recall"], metrics["f1"]), (0.0, 0.0, 0.0))

    def test_recall_falls_as_the_threshold_rises(self) -> None:
        rows = [make_row(f"{index}.jpg", index % 2, index / 10.0) for index in range(10)]
        recalls = [point["recall"] for point in evaluate.curve(rows)]
        self.assertEqual(recalls, sorted(recalls, reverse=True))

    def test_curve_is_capped_at_the_requested_number_of_points(self) -> None:
        rows = [make_row(f"{index}.jpg", index % 2, index / 500.0) for index in range(500)]
        self.assertLessEqual(len(evaluate.curve(rows)), evaluate.CURVE_POINTS)

    def test_gate_table_separates_labels_by_stage(self) -> None:
        rows = [
            make_row("a.jpg", 1, 0.9),
            make_row("b.jpg", 1, 0.0, stage=STAGE_MINOR, visible=False),
            make_row("c.jpg", 0, 0.0, stage=STAGE_MINOR, visible=False),
        ]
        table = evaluate.gate_table(rows)
        self.assertEqual(table[STAGE_SCORED], {"positive": 1, "negative": 0})
        self.assertEqual(table[STAGE_MINOR], {"positive": 1, "negative": 1})

    def test_worst_cases_are_ordered_by_confidence_of_the_mistake(self) -> None:
        rows = [
            make_row("fp_high.jpg", 0, 0.95),
            make_row("fp_low.jpg", 0, 0.55),
            make_row("fn_low.jpg", 1, 0.02),
            make_row("fn_high.jpg", 1, 0.45),
        ]
        worst = evaluate.worst_cases(rows, 0.5, top=5)
        self.assertEqual([entry["name"] for entry in worst["false_positives"]], ["fp_high.jpg", "fp_low.jpg"])
        self.assertEqual([entry["name"] for entry in worst["false_negatives"]], ["fn_low.jpg", "fn_high.jpg"])

    def test_worst_cases_honour_top(self) -> None:
        rows = [make_row(f"{index}.jpg", 0, 0.9) for index in range(5)]
        self.assertEqual(len(evaluate.worst_cases(rows, 0.5, top=2)["false_positives"]), 2)


class OverrideTests(unittest.TestCase):
    def setUp(self) -> None:
        from bikini_scanner.config import ScannerConfig

        self.config = ScannerConfig()

    def test_numeric_and_boolean_keys_are_coerced(self) -> None:
        evaluate.apply_override(self.config, "threshold", "0.42")
        evaluate.apply_override(self.config, "batch_size", "8")
        evaluate.apply_override(self.config, "require_female", "false")
        self.assertAlmostEqual(self.config.threshold, 0.42, places=6)
        self.assertEqual(self.config.batch_size, 8)
        self.assertFalse(self.config.require_female)

    def test_unknown_key_is_rejected(self) -> None:
        with self.assertRaises(ValueError):
            evaluate.apply_override(self.config, "not_a_setting", "1")

    def test_settings_outside_the_allowlist_cannot_be_swept(self) -> None:
        """A sweep tunes scoring; it must not reach the backend, prompts, VLM, plugins
        or the age gate."""
        for key in (
            "backend",
            "model_name",
            "pipeline",
            "positive_prompts",
            "vlm_enabled",
            "vlm_base_url",
            "enable_plugins",
            "exclude_minors",
            "minor_threshold",
        ):
            with self.subTest(key=key), self.assertRaises(ValueError):
                evaluate.apply_override(self.config, key, "1")

    def test_a_non_boolean_value_for_a_boolean_key_is_rejected(self) -> None:
        with self.assertRaises(ValueError):
            evaluate.apply_override(self.config, "require_person", "maybe")


class CompareTests(unittest.TestCase):
    def report(self, auc: float, score: float) -> dict[str, object]:
        return {
            "metrics": {"auc": auc, "precision": 1.0, "recall": 1.0, "f1": 1.0, "tp": 1, "fp": 0, "fn": 0, "tn": 1},
            "rows": [{"name": "a.jpg", "score": score}],
        }

    def test_identical_runs_compare_clean(self) -> None:
        self.assertEqual(evaluate.compare(self.report(0.9, 0.5), self.report(0.9, 0.5), 0.005), 0)

    def test_movement_within_tolerance_is_ignored(self) -> None:
        self.assertEqual(evaluate.compare(self.report(0.9, 0.5), self.report(0.902, 0.502), 0.005), 0)

    def test_movement_beyond_tolerance_is_reported(self) -> None:
        self.assertEqual(evaluate.compare(self.report(0.95, 0.5), self.report(0.9, 0.5), 0.005), 1)

    def test_a_row_appearing_or_vanishing_is_reported(self) -> None:
        current = self.report(0.9, 0.5)
        previous = self.report(0.9, 0.5)
        previous["rows"] = [{"name": "b.jpg", "score": 0.5}]
        self.assertEqual(evaluate.compare(current, previous, 0.005), 1)


class EndToEndTests(unittest.TestCase):
    """Drive the whole harness over generated images through the fake backend."""

    def setUp(self) -> None:
        self.folder = Path(tempfile.mkdtemp(prefix="bikini_eval_corpus_"))
        images = self.folder / "images"
        images.mkdir()
        rows = []
        for index in range(6):
            name = f"img_{index}.jpg"
            generator = np.random.default_rng(index)
            Image.fromarray((generator.random((48, 48, 3)) * 255).astype("uint8")).save(images / name)
            rows.append(f"images/{name},{'match' if index % 2 == 0 else 'no_match'}")
        self.manifest = self.folder / "labels.csv"
        self.manifest.write_text("path,label\n" + "\n".join(rows) + "\n", encoding="utf-8")

    def test_run_writes_a_report_and_compares_clean_against_itself(self) -> None:
        output = self.folder / "run.json"
        exit_code = evaluate.main(
            ["--manifest", str(self.manifest), "--fake-backend", "--quiet", "--output", str(output)]
        )
        self.assertEqual(exit_code, 0)
        report = json.loads(output.read_text(encoding="utf-8"))
        self.assertEqual(report["metrics"]["images"], 6)
        self.assertEqual(report["metrics"]["positives"], 3)
        self.assertEqual(report["settings"]["backend"], "fake")
        self.assertFalse(report["settings"]["learning"])
        self.assertEqual(len(report["rows"]), 6)
        counts = report["metrics"]
        self.assertEqual(counts["tp"] + counts["fp"] + counts["fn"] + counts["tn"], 6)
        self.assertEqual(
            evaluate.main(["--manifest", str(self.manifest), "--fake-backend", "--quiet", "--compare", str(output)]),
            0,
        )

    def test_sweep_runs_every_value(self) -> None:
        exit_code = evaluate.main(
            [
                "--manifest",
                str(self.manifest),
                "--fake-backend",
                "--quiet",
                "--sweep",
                "threshold",
                "--sweep-values",
                "0.3,0.6",
            ]
        )
        self.assertEqual(exit_code, 0)

    def test_a_missing_manifest_is_an_error_not_a_traceback(self) -> None:
        self.assertEqual(evaluate.main(["--manifest", str(self.folder / "nope.csv")]), 2)


if __name__ == "__main__":
    unittest.main(verbosity=2)
