"""End-to-end functional checks for Bikini Scanner.

Runs against a throwaway folder of generated images and never touches real user data:
the user preferences path, the last-folder marker, and the cross-folder learning store
are all redirected into a temporary directory first.

    PYTHONPATH=. python tests/test_functional.py            # everything
    PYTHONPATH=. python tests/test_functional.py Scoring    # one class

By default the suite runs against `tests.fake_backend.FakeBackend`: deterministic,
content-derived embeddings that need no model weights and no network, so the whole run
costs seconds instead of minutes. Set BIKINI_SCANNER_REAL_BACKEND=1 to run the same
tests against the real CLIP model (slow, downloads ~600 MB on first use) when you want
to check that a change behaves the same way against real embeddings.
"""

from __future__ import annotations

import io
import json
import logging
import os
import pickle
import shutil
import sys
import tempfile
import textwrap
import threading
import types
import unittest
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path

import numpy as np
from PIL import Image

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
sys.path.insert(0, str(Path(__file__).resolve().parent))

# Redirect every user-state path before importing anything that reads them.
_STATE_DIR = Path(tempfile.mkdtemp(prefix="bikini_state_"))
os.environ["BIKINI_SCANNER_TEST_SQLITE_PRAGMAS"] = "1"
os.environ["APPDATA"] = str(_STATE_DIR)
os.environ["LOCALAPPDATA"] = str(_STATE_DIR)  # the log lives here on Windows
os.environ["XDG_CONFIG_HOME"] = str(_STATE_DIR)
os.environ["XDG_STATE_HOME"] = str(_STATE_DIR)
os.environ["HOME"] = str(_STATE_DIR)
os.environ["USERPROFILE"] = str(_STATE_DIR)

from bikini_scanner import (
    cascade,
    config_profiles,
    image_formats,
    learning,
    linear_model,
    logging_setup,
    output_ops,
    plugins,
    regions,
    safe_io,
    update_checker,
    user_prefs,
)
from bikini_scanner import scorer as scorer_module
from bikini_scanner import store as store_module
from bikini_scanner.__version__ import __version__
from bikini_scanner.config import ScannerConfig, filter_folder_override
from bikini_scanner.config_profiles import BUILTIN_PROFILES, profile_config, profile_names
from bikini_scanner.global_store import GlobalLearningStore
from bikini_scanner.regions import plan_regions
from bikini_scanner.scorer import (
    BikiniScorer,
    RefineResult,
    ScoreState,
    bucketed_sampling,
    compute_vlm_scores,
    scan_and_score_folder,
)
from bikini_scanner.skin import skin_fraction
from bikini_scanner.store import FolderStore, collect_image_paths, content_hash_for_path
from bikini_scanner.vision_analysis import FaceBox, detect_face_count
from bikini_scanner.vlm_backend import VLMCancelled, VLMClient, is_local_endpoint, parse_axis_json

IMAGE_COUNT = 8
_SHARED: dict[str, object] = {}


def _make_image_bytes(size: tuple[int, int] = (64, 64)) -> bytes:
    buffer = io.BytesIO()
    Image.new("RGB", size, color=(120, 90, 60)).save(buffer, format="JPEG")
    return buffer.getvalue()


def _make_images(folder: Path, count: int = IMAGE_COUNT) -> list[Path]:
    folder.mkdir(parents=True, exist_ok=True)
    paths = []
    sizes = [(640, 480), (480, 640), (800, 300), (300, 800), (512, 512)]
    for index in range(count):
        path = folder / f"sample_{index:02d}.jpg"
        width, height = sizes[index % len(sizes)]
        image = Image.new("RGB", (width, height))
        pixels = image.load()
        for y in range(0, height, 2):
            for x in range(0, width, 2):
                value = (x + y * (index + 1)) % 255
                for dy in range(2):
                    for dx in range(2):
                        if x + dx < width and y + dy < height:
                            pixels[x + dx, y + dy] = (value, (value * 3) % 255, (index * 30) % 255)
        image.save(path, quality=60)
        paths.append(path)
    # A byte-identical copy exercises content-hash de-duplication.
    duplicate = folder / "duplicate_of_00.jpg"
    shutil.copyfile(paths[0], duplicate)
    paths.append(duplicate)
    return paths


def _use_real_backend() -> bool:
    return os.environ.get("BIKINI_SCANNER_REAL_BACKEND", "").strip().lower() in {"1", "true", "yes"}


def _build_backend(config: ScannerConfig):
    """The real CLIP backend only when asked for; torch is imported lazily either way."""
    if _use_real_backend():
        from bikini_scanner.clip_backend import get_backend

        return get_backend(config)
    from fake_backend import FakeBackend

    return FakeBackend()


def _shared():
    if "backend" not in _SHARED:
        config = ScannerConfig()
        config.preload_backend = False
        _SHARED["config"] = config
        _SHARED["backend"] = _build_backend(config)
        root = Path(tempfile.mkdtemp(prefix="bikini_scan_"))
        _SHARED["root"] = root
        _make_images(root)
    return _SHARED


class ScanPipeline(unittest.TestCase):
    """A scan produces a usable state, and the cache makes the second one cheap."""

    @classmethod
    def setUpClass(cls) -> None:
        shared = _shared()
        cls.config = ScannerConfig()
        cls.backend = shared["backend"]
        cls.folder = Path(str(shared["root"]))
        cls.store = FolderStore(cls.folder)
        cls.scorer = BikiniScorer(backend=cls.backend, config=cls.config)
        cls.state, cls.samples = scan_and_score_folder(
            cls.backend, cls.store, cls.scorer, threshold=cls.config.threshold
        )

    def test_every_image_scored(self) -> None:
        found = collect_image_paths(self.folder)
        self.assertEqual(len(self.state.paths), len(found))
        self.assertEqual(len(self.state.scores), len(self.state.paths))
        self.assertTrue(np.isfinite(self.state.scores).all())
        self.assertTrue(((self.state.scores >= 0) & (self.state.scores <= 1)).all())

    def test_cascade_produced_stages_and_axes(self) -> None:
        self.assertEqual(len(self.state.cascade_stage), len(self.state.paths))
        for axis in ("bikini", "cleavage", "midriff", "person", "female", "child", "adult", "detail"):
            self.assertIn(axis, self.state.axis_scores, f"missing axis {axis}")
            self.assertEqual(len(self.state.axis_scores[axis]), len(self.state.paths))

    def test_deep_pass_scored_region_crops(self) -> None:
        table = self.state.region_table
        self.assertIsNotNone(table)
        # More rows than images means crops were actually embedded and scored.
        self.assertGreater(table.owner.size, len(self.state.paths))

    def test_duplicate_images_share_an_embedding(self) -> None:
        groups = self.store.duplicate_groups()
        self.assertTrue(groups, "the identical copy should be detected as a duplicate")
        for members in groups.values():
            first = self.state.paths.index(members[0])
            second = self.state.paths.index(members[1])
            np.testing.assert_allclose(self.state.embeddings[first], self.state.embeddings[second])

    def test_rescan_reuses_cache(self) -> None:
        state, _ = scan_and_score_folder(self.backend, self.store, self.scorer, threshold=self.config.threshold)
        self.assertEqual(len(state.paths), len(self.state.paths))
        np.testing.assert_allclose(state.scores, self.state.scores, atol=1e-5)

    def test_scan_metadata_written(self) -> None:
        payload = json.loads(self.store.metadata_path.read_text(encoding="utf-8"))
        self.assertEqual(len(payload["images"]), len(self.state.paths))
        record = payload["images"][0]
        for key in ("filename", "path", "score", "axis_scores", "matched", "cascade_stage"):
            self.assertIn(key, record)


class VLMAdjudication(unittest.TestCase):
    class Handler(BaseHTTPRequestHandler):
        calls = 0
        lock = threading.Lock()

        def do_GET(self):
            if self.path == "/v1/models":
                self.send_response(200)
                self.send_header("Content-Type", "application/json")
                self.end_headers()
                self.wfile.write(b'{"data":[]}')
                return
            self.send_error(404)

        def do_POST(self):
            if self.path != "/v1/chat/completions":
                self.send_error(404)
                return
            length = int(self.headers.get("Content-Length", "0"))
            self.rfile.read(length)
            with self.lock:
                type(self).calls += 1
            body = json.dumps(
                {"choices": [{"message": {"content": '```json\n{"bikini": 1.4, "child": 0.1, "adult": 0.9}\n```'}}]}
            ).encode()
            self.send_response(200)
            self.send_header("Content-Type", "application/json")
            self.send_header("Content-Length", str(len(body)))
            self.end_headers()
            self.wfile.write(body)

        def log_message(self, *_args):
            return

    @classmethod
    def setUpClass(cls):
        cls.Handler.calls = 0
        cls.server = ThreadingHTTPServer(("127.0.0.1", 0), cls.Handler)
        cls.thread = threading.Thread(target=cls.server.serve_forever, daemon=True)
        cls.thread.start()
        cls.url = f"http://127.0.0.1:{cls.server.server_port}/v1"

    @classmethod
    def tearDownClass(cls):
        cls.server.shutdown()
        cls.server.server_close()
        cls.thread.join(timeout=2)

    def test_parser_fences_missing_and_clamps(self):
        values = parse_axis_json('```json\n{"bikini": 2, "child": -1}\n```')
        self.assertEqual(values["bikini"], 1.0)
        self.assertEqual(values["child"], 0.0)
        self.assertNotIn("adult", values)
        with self.assertRaises((ValueError, TypeError, json.JSONDecodeError)):
            parse_axis_json("not JSON")

    def test_skin_fraction_is_bounded(self):
        for color in ((0, 0, 0), (255, 255, 255), (180, 120, 90)):
            value = skin_fraction(Image.new("RGB", (300, 200), color))
            self.assertGreaterEqual(value, 0.0)
            self.assertLessEqual(value, 1.0)

    def test_concurrency_and_cancel(self):
        client = VLMClient(self.url, "test", concurrency=2)
        images = [[Image.new("RGB", (16, 16), "white")] for _ in range(3)]
        progress = []
        results = client.score_images(images, on_progress=lambda done, total: progress.append((done, total)))
        self.assertEqual(len(results), 3)
        self.assertTrue(progress)
        cancel = threading.Event()
        cancel.set()
        with self.assertRaises(VLMCancelled):
            client.score_images(images, cancel_event=cancel)

    def test_cache_hit_and_target_band(self):
        class Backend:
            image_embedding_dim = 2

            def embed_texts(self, prompts):
                return np.ones((len(prompts), 2), dtype=np.float32)

        root = Path(tempfile.mkdtemp(prefix="vlm_test_"))
        paths = []
        for index in range(3):
            path = root / f"{index}.jpg"
            Image.new("RGB", (80, 80), (180, 120, 90)).save(path)
            paths.append(str(path))
        config = ScannerConfig(vlm_enabled=True, vlm_base_url=self.url, vlm_model="test", vlm_max_images=2)
        scorer = BikiniScorer(Backend(), config)
        state = ScoreState(
            paths=paths,
            embeddings=np.ones((3, 2), dtype=np.float32),
            zero_shot_scores=np.array([0.34, 0.9, 0.36], dtype=np.float32),
            scores=np.array([0.34, 0.9, 0.36], dtype=np.float32),
            axis_scores={
                "child": np.array([0.5, 0.5, 0.5], dtype=np.float32),
                "adult": np.array([0.5, 0.5, 0.5], dtype=np.float32),
            },
            face_counts=None,
            classifier_trained=False,
            classifier_label_count=0,
            excluded=np.array([False, False, False]),
        )
        store = FolderStore(root)
        before = self.Handler.calls
        result = compute_vlm_scores(scorer, state, threshold=0.35, store=store)
        self.assertIsInstance(result, RefineResult)
        self.assertEqual(np.isfinite(result.scores).sum(), 2)
        first_calls = self.Handler.calls - before
        self.assertEqual(first_calls, 2)
        compute_vlm_scores(scorer, state, threshold=0.35, store=store)
        self.assertEqual(self.Handler.calls - before, first_calls)

    def test_unreachable_probe_skips(self):
        client = VLMClient("http://127.0.0.1:1/v1", "test", timeout=0.1)
        self.assertFalse(client.probe())


class AgeGate(unittest.TestCase):
    """The age gate must exclude, force the score to zero, and never be overridable."""

    def setUp(self) -> None:
        self.config = ScannerConfig()
        self.count = 4

    def _table(self, child: float, adult: float, detail: float) -> cascade.RegionScoreTable:
        rows = np.arange(self.count, dtype=np.int64)
        axis = {
            "child": np.full(self.count, child, dtype=np.float32),
            "adult": np.full(self.count, adult, dtype=np.float32),
            "person": np.full(self.count, 0.9, dtype=np.float32),
            "female": np.full(self.count, 0.9, dtype=np.float32),
            "bikini": np.full(self.count, detail, dtype=np.float32),
            "cleavage": np.full(self.count, detail, dtype=np.float32),
            "midriff": np.full(self.count, detail, dtype=np.float32),
            "bikini_top": np.full(self.count, detail, dtype=np.float32),
            "bikini_bottom": np.full(self.count, detail, dtype=np.float32),
            "nsfw": np.full(self.count, 0.5, dtype=np.float32),
        }
        return cascade.RegionScoreTable(
            owner=rows,
            kinds=np.array(["full"] * self.count, dtype=object),
            axis_scores=axis,
            image_count=self.count,
            full_row=rows,
        )

    def test_strong_child_evidence_excludes_and_zeroes(self) -> None:
        result = cascade.evaluate(self._table(child=0.99, adult=0.5, detail=0.99), self.config)
        self.assertTrue(result.excluded.all())
        self.assertEqual(list(set(result.stage)), [cascade.STAGE_MINOR])
        np.testing.assert_allclose(result.score, 0.0)

    def test_adult_subject_is_not_gated(self) -> None:
        result = cascade.evaluate(self._table(child=0.5, adult=0.95, detail=0.95), self.config)
        self.assertFalse(result.excluded.any())
        self.assertTrue((result.score > 0).all())

    def test_gate_can_be_switched_off(self) -> None:
        config = ScannerConfig()
        config.exclude_minors = False
        result = cascade.evaluate(self._table(child=0.99, adult=0.5, detail=0.99), config)
        self.assertFalse(result.excluded.any())

    def test_visibility_mask_drops_excluded_images(self) -> None:
        scorer = BikiniScorer(backend=_shared()["backend"], config=self.config)
        result = cascade.evaluate(self._table(child=0.99, adult=0.5, detail=0.9), self.config)
        mask = scorer.visibility_mask(result.axis_scores, None, result.excluded)
        self.assertFalse(mask.any())

    def test_the_gate_is_reachable_from_the_settings_dialog(self) -> None:
        """The toggle exists in code; it also has to be visible and clickable.

        It was gridded into the same cell as the 'Minor sensitivity' caption, so the two
        were drawn on top of each other and the box could not be seen or clicked.
        """
        tkinter = __import__("tkinter")
        try:
            root = tkinter.Tk()
        except Exception as exc:  # noqa: BLE001
            self.skipTest(f"no display available: {exc}")
        try:
            from bikini_scanner import gui as gui_module

            for name in ("_resume_last_folder_if_any", "_maybe_preload_backend", "_maybe_show_first_run_guide"):
                setattr(gui_module.BikiniScannerApp, name, lambda self: None)
            app = gui_module.BikiniScannerApp(root, config=ScannerConfig(preload_backend=False))
            before = set(root.winfo_children())
            app.open_settings_dialog()
            root.update()
            dialog = next(w for w in root.winfo_children() if w not in before)

            def walk(widget):
                yield widget
                for child in widget.winfo_children():
                    yield from walk(child)

            cells: dict[tuple[int, int], int] = {}
            age_box = None
            for widget in walk(dialog):
                info = widget.grid_info() if hasattr(widget, "grid_info") else None
                if info:
                    cells[(int(info["row"]), int(info["column"]))] = (
                        cells.get((int(info["row"]), int(info["column"])), 0) + 1
                    )
                try:
                    if "may show a minor" in str(widget.cget("text")):
                        age_box = widget
                except Exception:  # noqa: BLE001
                    continue

            self.assertIsNotNone(age_box, "the age-gate checkbox is missing from Settings")
            self.assertTrue(age_box.winfo_ismapped(), "the age-gate checkbox is not displayed")
            position = (int(age_box.grid_info()["row"]), int(age_box.grid_info()["column"]))
            self.assertEqual(cells[position], 1, "another widget shares the checkbox's grid cell")
            variable = str(age_box.cget("variable"))
            initial = root.getvar(variable)
            age_box.invoke()
            self.assertNotEqual(str(initial), str(root.getvar(variable)), "the checkbox does not toggle")
            app._closing = True
        finally:
            root.destroy()


class RegionAggregation(unittest.TestCase):
    """Where a crop sits decides what it can be evidence of, and how loudly."""

    @staticmethod
    def _table(region_key: str, region_score: float, full_score: float = 0.5) -> cascade.RegionScoreTable:
        keys = ["full", region_key]
        return cascade.RegionScoreTable(
            owner=np.array([0, 0], dtype=np.int64),
            kinds=np.array([regions.region_kind(key) for key in keys], dtype=object),
            axis_scores={"cleavage": np.array([full_score, region_score], dtype=np.float32)},
            image_count=1,
            full_row=np.array([0], dtype=np.int64),
        )

    def test_lower_band_cannot_claim_cleavage(self) -> None:
        # The bottom of a frame is not evidence of cleavage, however it scores.
        aggregated = self._table("bandlow", 0.99).aggregate("cleavage")
        self.assertAlmostEqual(float(aggregated[0]), 0.5, places=5)

    def test_unanchored_band_only_gets_a_partial_vote(self) -> None:
        aggregated = self._table("bandmid", 0.9).aggregate("cleavage")
        expected = 0.5 + cascade.UNANCHORED_CROP_SHARE * (0.9 - 0.5)
        self.assertAlmostEqual(float(aggregated[0]), expected, places=5)

    def test_face_anchored_crop_gets_a_full_vote(self) -> None:
        aggregated = self._table("chest0", 0.9).aggregate("cleavage")
        self.assertAlmostEqual(float(aggregated[0]), 0.9, places=5)

    def test_a_weak_crop_never_drags_the_full_frame_down(self) -> None:
        aggregated = self._table("chest0", 0.1, full_score=0.8).aggregate("cleavage")
        self.assertAlmostEqual(float(aggregated[0]), 0.8, places=5)

    def test_band_kinds_survive_a_round_trip_through_the_cache_key(self) -> None:
        planned = regions.plan_regions((800, 900), [])
        by_key = {region.key: region.kind for region in planned}
        self.assertTrue({"bandtop", "bandmid", "bandlow"} <= set(by_key))
        for key, kind in by_key.items():
            self.assertEqual(regions.region_kind(key), kind, f"{key} reclassified on reload")


class Learning(unittest.TestCase):
    """Labels have to change the ranking, survive a rescore, and be forgettable."""

    @classmethod
    def setUpClass(cls) -> None:
        shared = _shared()
        cls.config = ScannerConfig()
        cls.scorer = BikiniScorer(backend=shared["backend"], config=cls.config)
        cls.folder = Path(str(shared["root"]))
        cls.store = FolderStore(cls.folder)
        cls.state, _ = scan_and_score_folder(shared["backend"], cls.store, cls.scorer, threshold=cls.config.threshold)

    def _labels(self) -> dict[str, int]:
        ranked = np.argsort(-np.asarray(self.state.zero_shot_scores))
        return {str(self.state.paths[int(index)]): 1 if rank % 2 == 0 else 0 for rank, index in enumerate(ranked[:6])}

    def test_labels_move_the_scores(self) -> None:
        before = np.asarray(self.state.scores).copy()
        new_state, _ = self.scorer.rescore_state(self.state, self._labels(), threshold=self.config.threshold)
        self.assertGreater(float(np.abs(np.asarray(new_state.scores) - before).max()), 0.0)
        self.assertTrue(new_state.learning_summary)

    def test_state_disagreement_is_public_and_matches_the_scores(self) -> None:
        """The GUI rebuilds the review queue from this, so it has to be importable.

        It was private, so the GUI could not pass it and the "Model disagrees" bucket
        vanished whenever a filter or sort rebuilt the queue.
        """
        mask = np.ones(len(self.state.paths), dtype=bool)
        gaps = scorer_module.state_disagreement(self.state, mask)
        self.assertEqual(len(gaps), len(self.state.paths))
        expected = np.abs(np.asarray(self.state.scores) - np.asarray(self.state.zero_shot_scores))
        np.testing.assert_allclose(np.asarray(gaps), expected, atol=1e-6)
        mask[0] = False
        self.assertEqual(len(scorer_module.state_disagreement(self.state, mask)), len(self.state.paths) - 1)

    def test_rescore_does_not_reembed(self) -> None:
        new_state, _ = self.scorer.rescore_state(self.state, self._labels(), threshold=self.config.threshold)
        np.testing.assert_allclose(new_state.embeddings, self.state.embeddings)
        self.assertIs(new_state.region_table, self.state.region_table)

    def test_prototype_works_from_two_labels(self) -> None:
        # The prototype compares directions, so the classes need different directions
        # (magnitude alone carries no information for cosine similarity).
        accepted = np.array([1.0, 0.0, 0.0, 0.0], dtype=np.float32)
        rejected = np.array([0.0, 1.0, 0.0, 0.0], dtype=np.float32)
        features = np.vstack([accepted, rejected, accepted * 0.5, rejected * 2.0]).astype(np.float32)
        outcome = learning.fit(features, np.array([1, 0, 1, 0], dtype=np.int64))
        self.assertIsNotNone(outcome.prototype)
        scores = outcome.score(features)
        self.assertGreater(float(scores[0]), float(scores[1]))
        self.assertGreater(float(scores[2]), float(scores[3]))

    def test_weight_reflects_measured_quality(self) -> None:
        rng = np.random.default_rng(0)
        features = rng.normal(size=(60, 12)).astype(np.float32)
        separable = (features[:, 0] > 0).astype(np.int64)
        good = learning.fit(features, separable)
        noise = rng.integers(0, 2, size=60)
        bad = learning.fit(features, noise)
        self.assertGreater(good.weight, bad.weight)

    def test_global_store_records_prunes_and_clears(self) -> None:
        store = GlobalLearningStore(model_name="test-model")
        store.clear()
        vanishing = Path(tempfile.mkdtemp(prefix="bikini_gone_")) / "x.jpg"
        vanishing.parent.mkdir(parents=True, exist_ok=True)
        vanishing.write_bytes(b"x")
        real = Path(str(_shared()["root"])) / "sample_00.jpg"
        store.record([(str(real), 1, np.ones(4, dtype=np.float32))], sequence=1)
        store.record([(str(vanishing), 0, np.zeros(4, dtype=np.float32))], sequence=2)
        self.assertEqual(store.stats()["total"], 2)
        shutil.rmtree(vanishing.parent, ignore_errors=True)
        # A label whose file is gone must stop training the model.
        self.assertEqual(len(store.training_set(expected_dim=4)), 1)
        self.assertEqual(store.stats()["total"], 1)
        store.clear()
        self.assertEqual(store.stats()["total"], 0)


class ScanProgressReporting(unittest.TestCase):
    """The bar has to move forwards only, and finish, whatever phases actually ran."""

    def _ticks(self, folder: Path, config: ScannerConfig | None = None) -> list:
        config = config or ScannerConfig()
        scorer = BikiniScorer(backend=_shared()["backend"], config=config)
        ticks: list = []
        scan_and_score_folder(
            _shared()["backend"],
            FolderStore(folder),
            scorer,
            threshold=config.threshold,
            progress_callback=ticks.append,
        )
        return ticks

    def test_progress_is_monotonic_and_reaches_one(self) -> None:
        ticks = self._ticks(Path(str(_shared()["root"])))
        self.assertTrue(ticks, "no progress was reported")
        fractions = [tick.fraction for tick in ticks]
        self.assertEqual(fractions, sorted(fractions), "progress went backwards")
        self.assertAlmostEqual(fractions[-1], 1.0, places=6)

    def test_counts_are_reported_per_phase(self) -> None:
        ticks = self._ticks(Path(str(_shared()["root"])))
        embed = [tick for tick in ticks if tick.phase == scorer_module.PHASE_EMBED]
        self.assertTrue(embed)
        self.assertEqual(embed[-1].total, len(collect_image_paths(Path(str(_shared()["root"])))))
        self.assertIn(f"/ {embed[-1].total:,}", embed[-1].text())

    def test_legacy_pipeline_still_finishes_the_bar(self) -> None:
        config = ScannerConfig()
        config.pipeline = "legacy"
        folder = Path(tempfile.mkdtemp(prefix="bikini_progress_legacy_"))
        _make_images(folder, count=2)
        ticks = self._ticks(folder, config)
        self.assertAlmostEqual(ticks[-1].fraction, 1.0, places=6)

    def test_old_four_argument_callback_still_works(self) -> None:
        calls: list[tuple[int, int]] = []
        scan_and_score_folder(
            _shared()["backend"],
            FolderStore(Path(str(_shared()["root"]))),
            BikiniScorer(backend=_shared()["backend"], config=ScannerConfig()),
            threshold=0.35,
            progress_callback=lambda done, total, rate, eta: calls.append((done, total)),
        )
        self.assertTrue(calls, "the legacy progress signature was never called")


class GuiConcurrency(unittest.TestCase):
    """Labelling a run of photos must not start a background pass per click."""

    def setUp(self) -> None:
        tkinter = __import__("tkinter")
        try:
            self.root = tkinter.Tk()
        except Exception as exc:  # noqa: BLE001
            self.skipTest(f"no display available: {exc}")
        from bikini_scanner import gui as gui_module

        for name in ("_resume_last_folder_if_any", "_maybe_preload_backend", "_maybe_show_first_run_guide"):
            setattr(gui_module.BikiniScannerApp, name, lambda self: None)
        self.folder = Path(tempfile.mkdtemp(prefix="bikini_conc_"))
        _make_images(self.folder, count=2)
        self.app = gui_module.BikiniScannerApp(self.root, config=ScannerConfig(preload_backend=False))
        self.app._set_folder(str(self.folder))
        self.app.backend = object()
        self.app._ensure_scorer = lambda: True
        paths = [str(path) for path in collect_image_paths(self.folder)]
        self.app.current_state = scorer_module.ScoreState(
            paths=paths,
            embeddings=np.zeros((len(paths), 4), dtype=np.float32),
            zero_shot_scores=np.full(len(paths), 0.5, dtype=np.float32),
            scores=np.full(len(paths), 0.5, dtype=np.float32),
            axis_scores={"bikini": np.full(len(paths), 0.6, dtype=np.float32)},
            face_counts=None,
            classifier_trained=False,
            classifier_label_count=0,
            excluded=np.zeros(len(paths), dtype=bool),
        )
        self.gate = threading.Event()
        outer = self

        class _Blocking:
            config = ScannerConfig()

            def state_visibility(self, state):
                return np.ones((len(state.paths),), dtype=bool)

            def rescore_state(self, state, labels, threshold=0.5, store=None, cancel_event=None):
                outer.gate.wait(5.0)
                return state, []

        self.app.scorer = _Blocking()

    def tearDown(self) -> None:
        self.gate.set()
        self.app._closing = True
        self.root.destroy()
        shutil.rmtree(self.folder, ignore_errors=True)

    def test_repeated_retrains_do_not_stack_threads(self) -> None:
        spawned: list[threading.Thread] = []
        real_thread = threading.Thread

        class Counting(real_thread):
            def start(self):
                spawned.append(self)
                super().start()

        threading.Thread = Counting
        try:
            for _ in range(4):
                self.app.update_algorithm()
        finally:
            threading.Thread = real_thread
        self.assertEqual(len(spawned), 1, "each Retrain click started its own worker")
        self.assertTrue(self.app._retrain_pending, "the extra clicks should be coalesced, not dropped")

    def test_a_second_scan_is_refused_while_one_runs(self) -> None:
        launched: list[bool] = []
        self.app._launch_background_scan = lambda full_rescan: launched.append(full_rescan)
        self.app._scan_active = True
        from tkinter import messagebox

        original = messagebox.showinfo
        messagebox.showinfo = lambda *args, **kwargs: None
        try:
            self.app.run_scan()
        finally:
            messagebox.showinfo = original
        self.assertEqual(launched, [], "a second scan started on top of a running one")

    def test_the_cancel_token_is_not_orphaned(self) -> None:
        self.app.update_algorithm()
        first = self.app._scan_cancel_event
        self.app.update_algorithm()
        self.assertIs(self.app._scan_cancel_event, first, "Stop would no longer reach the running pass")


class NumericPrimitives(unittest.TestCase):
    """The numpy replacements for scikit-learn have to behave like the originals."""

    def test_sigmoid_is_stable_at_extremes(self) -> None:
        values = np.array([-1000.0, -50.0, 0.0, 50.0, 1000.0])
        out = linear_model.sigmoid(values)
        self.assertTrue(np.isfinite(out).all())
        self.assertAlmostEqual(float(out[2]), 0.5, places=6)
        self.assertAlmostEqual(float(out[0]), 0.0, places=6)
        self.assertAlmostEqual(float(out[4]), 1.0, places=6)

    def test_roc_auc_known_values(self) -> None:
        self.assertAlmostEqual(linear_model.roc_auc([0, 0, 1, 1], [0.1, 0.2, 0.3, 0.4]), 1.0)
        self.assertAlmostEqual(linear_model.roc_auc([0, 0, 1, 1], [0.4, 0.3, 0.2, 0.1]), 0.0)
        # Four samples are required for a reliable estimate; a perfectly tied ranking
        # should still evaluate to chance.
        self.assertAlmostEqual(linear_model.roc_auc([0, 0, 1, 1], [0.5, 0.5, 0.5, 0.5]), 0.5)

    def test_logistic_regression_separates(self) -> None:
        rng = np.random.default_rng(3)
        x = np.vstack([rng.normal(2.0, 0.5, (40, 5)), rng.normal(-2.0, 0.5, (40, 5))]).astype(np.float32)
        y = np.array([1] * 40 + [0] * 40, dtype=np.int64)
        model = linear_model.LogisticRegression().fit(x, y)
        probabilities = model.predict_proba(x)[:, 1]
        self.assertGreater(linear_model.roc_auc(y, probabilities), 0.99)
        np.testing.assert_allclose(model.predict_proba(x).sum(axis=1), 1.0, atol=1e-5)

    def test_stratified_folds_cover_every_row_once(self) -> None:
        labels = np.array([0] * 10 + [1] * 6, dtype=np.int64)
        folds = linear_model.stratified_folds(labels, 3)
        combined = np.concatenate(folds)
        self.assertEqual(sorted(combined.tolist()), list(range(16)))
        for fold in folds:
            self.assertTrue(set(labels[fold].tolist()) <= {0, 1})


class ReviewSampling(unittest.TestCase):
    def test_buckets_exclude_already_labelled(self) -> None:
        paths = [f"p{i}" for i in range(20)]
        scores = list(np.linspace(0, 1, 20))
        samples = bucketed_sampling(paths, scores, {"p0", "p1"}, threshold=0.5)
        chosen = {str(sample["path"]) for sample in samples}
        self.assertNotIn("p0", chosen)
        self.assertNotIn("p1", chosen)

    def test_disagreement_bucket_appears(self) -> None:
        paths = [f"p{i}" for i in range(12)]
        scores = [0.5] * 12
        disagreement = [0.9 if i < 3 else 0.0 for i in range(12)]
        samples = bucketed_sampling(paths, scores, [], threshold=0.5, disagreement=disagreement)
        self.assertIn("Model disagrees", {str(sample["bucket"]) for sample in samples})


class OutputOperations(unittest.TestCase):
    @classmethod
    def setUpClass(cls) -> None:
        cls.folder = Path(str(_shared()["root"]))
        cls.sources = sorted(str(p) for p in collect_image_paths(cls.folder))[:4]
        cls.scores = {path: 0.4 + 0.1 * index for index, path in enumerate(cls.sources)}
        cls.labels = {cls.sources[0]: 1, cls.sources[1]: 0}

    def test_transfer_plan_and_copy(self) -> None:
        destination = Path(tempfile.mkdtemp(prefix="bikini_out_"))
        options = output_ops.OutputOptions()
        plan = output_ops.build_transfer_plan(self.sources, destination, self.scores, self.labels, options)
        self.assertEqual(len(plan), len(self.sources))
        processed, _skipped, _retained, failed = output_ops.execute_transfer_plan(plan, move=False)
        self.assertEqual(processed, len(self.sources))
        self.assertEqual(failed, 0)
        self.assertEqual(len(list(destination.rglob("*.jpg"))), len(self.sources))
        for source in self.sources:
            self.assertTrue(Path(source).exists(), "copy must not remove the original")

    def test_one_failing_file_does_not_abandon_the_batch(self) -> None:
        """A file that cannot be transferred is reported, and the rest still go."""
        destination = Path(tempfile.mkdtemp(prefix="bikini_out_fail_"))
        sources = list(self.sources)
        missing = str(Path(sources[0]).with_name("does_not_exist.jpg"))
        # Put the doomed file first: the old code raised here and never reached the rest.
        plan = output_ops.build_transfer_plan(
            [missing, *sources], destination, self.scores, self.labels, output_ops.OutputOptions()
        )
        processed, _skipped, _retained, failed = output_ops.execute_transfer_plan(plan, move=False)
        self.assertEqual(failed, 1)
        self.assertEqual(processed, len(sources))
        self.assertEqual(len(list(destination.rglob("*.jpg"))), len(sources))
        broken = [item for item in plan if item.error]
        self.assertEqual(len(broken), 1)
        self.assertEqual(Path(broken[0].source).name, "does_not_exist.jpg")

    def test_a_failed_move_leaves_the_source_alone(self) -> None:
        working = Path(tempfile.mkdtemp(prefix="bikini_out_move_"))
        source = working / "keep_me.jpg"
        shutil.copyfile(self.sources[0], source)
        # A plain file where the destination folder should be: creating the destination
        # directory fails, so both the move and its copy fallback fail. That is the case
        # that used to raise straight out of execute_transfer_plan.
        blocker = working / "blocker"
        blocker.write_bytes(b"not a directory")
        plan = output_ops.build_transfer_plan(
            [str(source)], blocker / "out", {str(source): 0.5}, {}, output_ops.OutputOptions(), move=True
        )
        processed, _skipped, _retained, failed = output_ops.execute_transfer_plan(plan, move=True)
        self.assertEqual(failed, 1)
        self.assertEqual(processed, 0)
        self.assertTrue(source.exists(), "a failed move must not delete the source")
        self.assertTrue(plan[0].error)
        self.assertFalse(plan[0].source_removed)

    def test_organisation_by_score_band(self) -> None:
        destination = Path(tempfile.mkdtemp(prefix="bikini_out_band_"))
        options = output_ops.OutputOptions()
        options.organization = "score_band"
        plan = output_ops.build_transfer_plan(self.sources, destination, self.scores, self.labels, options)
        output_ops.execute_transfer_plan(plan, move=False)
        self.assertTrue([p for p in destination.iterdir() if p.is_dir()])

    def test_html_report_is_self_contained(self) -> None:
        destination = Path(tempfile.mkdtemp(prefix="bikini_html_")) / "report.html"
        samples = [{"path": path, "score": self.scores[path], "bucket": "Bikini"} for path in self.sources]
        output_ops.build_html_report(destination, samples, self.labels, self.scores)
        self.assertTrue(destination.exists())
        html = destination.read_text(encoding="utf-8", errors="ignore")
        self.assertIn("<html", html.lower())
        # Small reports embed their thumbnails, so the file stands alone.
        self.assertIn("data:image", html)

    def test_large_html_report_uses_an_assets_folder(self) -> None:
        destination = Path(tempfile.mkdtemp(prefix="bikini_html_big_")) / "report.html"
        samples = [{"path": path, "score": self.scores[path], "bucket": "Bikini"} for path in self.sources]
        output_ops.build_html_report(destination, samples, self.labels, self.scores, max_embedded_thumbnails=1)
        assets = destination.with_name(f"{destination.stem}_assets")
        self.assertTrue(assets.is_dir())
        self.assertTrue(list(assets.glob("*.jpg")))

    def test_metadata_written_to_jpeg(self) -> None:
        working = Path(tempfile.mkdtemp(prefix="bikini_meta_")) / "tagged.jpg"
        shutil.copyfile(self.sources[0], working)
        self.assertTrue(output_ops.write_image_metadata(working, "bikini", 0.87))
        self.assertGreater(working.stat().st_size, 0)
        with Image.open(working) as image:
            image.verify()


class OutputTransferExecution(unittest.TestCase):
    """The move and overwrite execution paths, including the ones that only run once
    something has already gone wrong.

    These are the branches that can lose a file: a rename that fails and falls back to
    copy-then-delete, a source that cannot be removed after that copy, and an overwrite
    that unlinks an existing destination. The happy copy path is covered by
    OutputOperations above; everything here is about what happens when the filesystem
    refuses.
    """

    def setUp(self) -> None:
        self.working = Path(tempfile.mkdtemp(prefix="bikini_exec_"))
        self.source_dir = self.working / "src"
        self.source_dir.mkdir()
        self.source = self.source_dir / "photo.jpg"
        self.source.write_bytes(_make_image_bytes())
        self.payload = self.source.read_bytes()

    def _plan(self, destination_dir: Path, *, move: bool = False, policy: str = "rename"):
        options = output_ops.OutputOptions()
        options.duplicate_policy = policy
        return output_ops.build_transfer_plan(
            [str(self.source)], destination_dir, {str(self.source): 0.5}, {}, options, move=move
        )

    def test_move_within_the_same_folder_relocates_the_file(self) -> None:
        """Same-parent moves take the os.replace path."""
        options = output_ops.OutputOptions()
        options.filename_template = "renamed"
        plan = output_ops.build_transfer_plan(
            [str(self.source)], self.source_dir, {str(self.source): 0.5}, {}, options, move=True
        )
        processed, _skipped, retained, failed = output_ops.execute_transfer_plan(plan, move=True)
        self.assertEqual((processed, retained, failed), (1, 0, 0))
        self.assertFalse(self.source.exists(), "a move must remove the source")
        self.assertEqual((self.source_dir / "renamed.jpg").read_bytes(), self.payload)

    def test_move_across_folders_relocates_the_file(self) -> None:
        """Differing parents take the shutil.move path."""
        destination = self.working / "out"
        plan = self._plan(destination, move=True)
        processed, _skipped, retained, failed = output_ops.execute_transfer_plan(plan, move=True)
        self.assertEqual((processed, retained, failed), (1, 0, 0))
        self.assertFalse(self.source.exists(), "a move must remove the source")
        self.assertEqual(plan[0].destination.read_bytes(), self.payload)

    def test_move_falls_back_to_copy_when_the_rename_fails(self) -> None:
        """A cross-volume move raises out of shutil.move; the copy fallback must still
        deliver the file and remove the source."""
        destination = self.working / "out"
        plan = self._plan(destination, move=True)

        def refuse(*_args, **_kwargs):
            raise OSError("simulated cross-device rename")

        original = shutil.move
        shutil.move = refuse
        try:
            processed, _skipped, retained, failed = output_ops.execute_transfer_plan(plan, move=True)
        finally:
            shutil.move = original

        self.assertEqual((processed, retained, failed), (1, 0, 0))
        self.assertEqual(plan[0].destination.read_bytes(), self.payload)
        self.assertFalse(self.source.exists(), "the fallback must remove the source after copying")
        self.assertTrue(plan[0].source_removed)

    def test_a_fallback_that_cannot_remove_the_source_reports_it(self) -> None:
        """Copy succeeded but the source could not be unlinked, so the file now exists in
        both places. That must be counted and flagged, never reported as a clean move."""
        destination = self.working / "out"
        plan = self._plan(destination, move=True)

        def refuse_move(*_args, **_kwargs):
            raise OSError("simulated cross-device rename")

        def refuse_unlink(_self, *_args, **_kwargs):
            raise PermissionError("simulated locked source")

        original_move = shutil.move
        original_unlink = Path.unlink
        shutil.move = refuse_move
        Path.unlink = refuse_unlink
        try:
            processed, _skipped, retained, failed = output_ops.execute_transfer_plan(plan, move=True)
        finally:
            shutil.move = original_move
            Path.unlink = original_unlink

        self.assertEqual(failed, 0, "the copy succeeded, so this is not a failure")
        self.assertEqual(processed, 1)
        self.assertEqual(retained, 1, "the surviving source must be counted")
        self.assertFalse(plan[0].source_removed, "the item must record that the source is still there")
        self.assertTrue(self.source.exists())
        self.assertEqual(plan[0].destination.read_bytes(), self.payload)

    def test_overwrite_policy_replaces_an_existing_destination(self) -> None:
        """Pins the end result of the overwrite policy: planned as "overwrite", and the
        stale bytes are gone afterwards.

        It deliberately does not claim to cover the pre-emptive unlink in
        execute_transfer_plan. Deleting that unlink leaves this test green, because
        shutil.copy2 truncates the destination by itself -- which is what the comment
        on that branch already says. The unlink is defensive only, so there is no
        observable behaviour left for a test to hold it to.
        """
        destination = self.working / "out"
        destination.mkdir()
        existing = destination / "photo.jpg"
        existing.write_bytes(b"stale contents")
        plan = self._plan(destination, policy="overwrite")
        self.assertEqual(plan[0].action, "overwrite")
        processed, _skipped, _retained, failed = output_ops.execute_transfer_plan(plan, move=False)
        self.assertEqual((processed, failed), (1, 0))
        self.assertEqual(existing.read_bytes(), self.payload, "the stale file must be replaced")

    def test_rename_policy_gives_each_collision_a_fresh_name(self) -> None:
        """Two files already holding the obvious names push the counter loop past its
        first attempt, so the copy lands on photo_2.jpg rather than overwriting."""
        destination = self.working / "out"
        destination.mkdir()
        (destination / "photo.jpg").write_bytes(b"first")
        (destination / "photo_1.jpg").write_bytes(b"second")
        plan = self._plan(destination, policy="rename")
        processed, _skipped, _retained, failed = output_ops.execute_transfer_plan(plan, move=False)
        self.assertEqual((processed, failed), (1, 0))
        self.assertEqual(plan[0].destination.name, "photo_2.jpg")
        self.assertEqual((destination / "photo.jpg").read_bytes(), b"first", "existing files must survive")
        self.assertEqual((destination / "photo_1.jpg").read_bytes(), b"second")

    def test_skip_policy_leaves_an_existing_destination_untouched(self) -> None:
        destination = self.working / "out"
        destination.mkdir()
        existing = destination / "photo.jpg"
        existing.write_bytes(b"do not touch")
        plan = self._plan(destination, policy="skip")
        processed, skipped, _retained, failed = output_ops.execute_transfer_plan(plan, move=False)
        self.assertEqual((processed, skipped, failed), (0, 1, 0))
        self.assertEqual(plan[0].reason, "duplicate exists")
        self.assertEqual(existing.read_bytes(), b"do not touch")

    def test_a_source_that_is_its_own_destination_is_skipped(self) -> None:
        """Scanning a folder and writing output back into it must not copy a file over
        itself, which would truncate it."""
        options = output_ops.OutputOptions()
        options.filename_template = "{stem}"
        plan = output_ops.build_transfer_plan(
            [str(self.source)], self.source_dir, {str(self.source): 0.5}, {}, options, move=True
        )
        self.assertEqual(plan[0].action, "skip")
        self.assertEqual(plan[0].reason, "source and destination are identical")
        processed, skipped, _retained, failed = output_ops.execute_transfer_plan(plan, move=True)
        self.assertEqual((processed, skipped, failed), (0, 1, 0))
        self.assertEqual(self.source.read_bytes(), self.payload, "the original must be intact")


class AtomicWrites(unittest.TestCase):
    """safe_io is what stops a crash mid-write from truncating the label store.

    The guarantee is all-or-nothing: the destination either holds the previous
    contents or the new ones, never a half-written file, and a failed write leaves no
    stray temporary behind.
    """

    def setUp(self) -> None:
        self.working = Path(tempfile.mkdtemp(prefix="bikini_atomic_"))

    def _strays(self) -> list[Path]:
        return list(self.working.rglob("*.tmp"))

    def test_text_is_written_and_no_temporary_survives(self) -> None:
        target = self.working / "labels.json"
        safe_io.atomic_write_text(target, "hello")
        self.assertEqual(target.read_text(encoding="utf-8"), "hello")
        self.assertEqual(self._strays(), [], "the temporary must be moved, not left behind")

    def test_missing_parent_directories_are_created(self) -> None:
        target = self.working / "deep" / "nested" / "labels.json"
        safe_io.atomic_write_text(target, "hello")
        self.assertEqual(target.read_text(encoding="utf-8"), "hello")

    def test_an_existing_file_keeps_its_old_contents_when_the_write_fails(self) -> None:
        """The point of the temp-file dance: a failed write must not damage what is
        already on disk, and must not leave a .tmp lying next to it."""
        target = self.working / "labels.json"
        target.write_text("original", encoding="utf-8")

        def explode(_path: Path) -> None:
            raise RuntimeError("simulated serialisation failure")

        with self.assertRaises(RuntimeError):
            safe_io.atomic_replace(target, explode)

        self.assertEqual(target.read_text(encoding="utf-8"), "original")
        self.assertEqual(self._strays(), [], "a failed write must clean up its temporary")

    def test_a_failing_text_write_cleans_up_and_re_raises(self) -> None:
        target = self.working / "labels.json"
        original_write = Path.write_text

        def explode(self_path, *args, **kwargs):
            if self_path.suffix == ".tmp":
                raise OSError("simulated disk full")
            return original_write(self_path, *args, **kwargs)

        Path.write_text = explode
        try:
            with self.assertRaises(OSError):
                safe_io.atomic_write_text(target, "never lands")
        finally:
            Path.write_text = original_write

        self.assertFalse(target.exists(), "nothing should have been put in place")
        self.assertEqual(self._strays(), [], "a failed write must clean up its temporary")

    def test_a_write_still_lands_when_fsync_is_unavailable(self) -> None:
        """fsync failures are swallowed on purpose: some filesystems refuse it, and the
        replace is still worth doing."""
        target = self.working / "labels.json"
        original_fsync = os.fsync

        def refuse(_fd):
            raise OSError("simulated fsync refusal")

        os.fsync = refuse
        try:
            safe_io.atomic_write_text(target, "written anyway")
        finally:
            os.fsync = original_fsync

        self.assertEqual(target.read_text(encoding="utf-8"), "written anyway")
        self.assertEqual(self._strays(), [])

    def test_json_round_trips(self) -> None:
        target = self.working / "payload.json"
        payload = {"b": 2, "a": [1, 2, 3]}
        safe_io.atomic_write_json(target, payload)
        self.assertEqual(json.loads(target.read_text(encoding="utf-8")), payload)

    def test_atomic_replace_hands_the_writer_a_temporary_not_the_target(self) -> None:
        """The writer must never see the real path, or a crash inside it would damage
        the existing file directly."""
        target = self.working / "labels.json"
        target.write_text("original", encoding="utf-8")
        seen: list[Path] = []

        def writer(path: Path) -> None:
            seen.append(path)
            path.write_text("replacement", encoding="utf-8")

        safe_io.atomic_replace(target, writer)
        self.assertEqual(target.read_text(encoding="utf-8"), "replacement")
        self.assertNotEqual(seen[0], target, "the writer must be given a temporary path")


class QuarantineBrokenFiles(unittest.TestCase):
    def setUp(self) -> None:
        self.working = Path(tempfile.mkdtemp(prefix="bikini_quarantine_"))
        self.logger = logging.getLogger("bikini_scanner.tests.quarantine")

    def test_a_missing_file_quarantines_to_nothing_and_says_nothing(self) -> None:
        """Nothing to quarantine is the ordinary case, not a failure.

        The silence is the assertion that matters. Without the exists() guard the call
        still returns None -- the rename below simply raises and is swallowed -- but it
        logs a warning about a file that was never there.
        """
        with self.assertNoLogs(self.logger, level="WARNING"):
            result = safe_io.quarantine_broken_file(self.working / "absent.json", self.logger, "unreadable")
        self.assertIsNone(result)

    def test_a_broken_file_is_renamed_out_of_the_way(self) -> None:
        source = self.working / "labels.json"
        source.write_bytes(b"corrupt")
        with self.assertLogs(self.logger, level="WARNING"):
            result = safe_io.quarantine_broken_file(source, self.logger, "bad json")
        self.assertIsNotNone(result)
        assert result is not None
        self.assertEqual(result.name, "labels.json.broken")
        self.assertEqual(result.read_bytes(), b"corrupt", "the evidence must be preserved")
        self.assertFalse(source.exists(), "the broken file must be moved aside")

    def test_repeated_breakages_do_not_overwrite_the_earlier_evidence(self) -> None:
        source = self.working / "labels.json"
        (self.working / "labels.json.broken").write_bytes(b"first failure")
        (self.working / "labels.json.broken.1").write_bytes(b"second failure")
        source.write_bytes(b"third failure")
        with self.assertLogs(self.logger, level="WARNING"):
            result = safe_io.quarantine_broken_file(source, self.logger, "bad json")
        assert result is not None
        self.assertEqual(result.name, "labels.json.broken.2")
        self.assertEqual((self.working / "labels.json.broken").read_bytes(), b"first failure")
        self.assertEqual((self.working / "labels.json.broken.1").read_bytes(), b"second failure")

    def test_a_quarantine_that_cannot_rename_reports_none(self) -> None:
        source = self.working / "labels.json"
        source.write_bytes(b"corrupt")
        original_replace = Path.replace

        def refuse(_self, _target):
            raise PermissionError("simulated locked file")

        Path.replace = refuse
        try:
            with self.assertLogs(self.logger, level="WARNING"):
                result = safe_io.quarantine_broken_file(source, self.logger, "bad json")
        finally:
            Path.replace = original_replace

        self.assertIsNone(result, "a failed quarantine must report failure, not a path")
        self.assertTrue(source.exists())


class ClassifierCachePersistence(unittest.TestCase):
    """The per-folder classifier cache is a pickle on disk, so every load is a trust
    decision as well as a compatibility one."""

    def setUp(self) -> None:
        self.folder = Path(tempfile.mkdtemp(prefix="bikini_clf_"))
        self.store = FolderStore(self.folder)
        self.addCleanup(self.store.sqlite_cache.close)

    def test_absent_cache_loads_as_nothing(self) -> None:
        self.assertIsNone(self.store.load_classifier_cache())

    def test_a_saved_cache_round_trips_and_is_stamped_with_its_version(self) -> None:
        self.store.save_classifier_cache({"classifier": {"weights": [1.0, 2.0]}})
        loaded = self.store.load_classifier_cache()
        self.assertIsNotNone(loaded)
        assert loaded is not None
        self.assertEqual(loaded["classifier"], {"weights": [1.0, 2.0]})
        self.assertEqual(loaded["version"], store_module.CLASSIFIER_CACHE_VERSION)

    def test_a_cache_from_another_version_is_ignored(self) -> None:
        """An old pickle must be discarded rather than fed to the current code."""
        self.store.save_classifier_cache({"classifier": {"weights": [1.0]}})
        stale = {"version": store_module.CLASSIFIER_CACHE_VERSION + 1, "classifier": {"weights": [1.0]}}
        self.store.classifier_path.write_bytes(pickle.dumps(stale))
        self.assertIsNone(self.store.load_classifier_cache())

    def test_a_cache_without_a_classifier_is_ignored(self) -> None:
        payload = {"version": store_module.CLASSIFIER_CACHE_VERSION, "classifier": None}
        self.store.classifier_path.write_bytes(pickle.dumps(payload))
        self.assertIsNone(self.store.load_classifier_cache())

    def test_a_cache_that_is_not_a_mapping_is_ignored(self) -> None:
        self.store.classifier_path.write_bytes(pickle.dumps(["not", "a", "dict"]))
        self.assertIsNone(self.store.load_classifier_cache())

    def test_corrupt_bytes_load_as_nothing_rather_than_raising(self) -> None:
        self.store.classifier_path.write_bytes(b"this is not a pickle at all")
        self.assertIsNone(self.store.load_classifier_cache())

    def test_the_unpickler_refuses_a_module_outside_the_allowlist(self) -> None:
        """The guard itself, checked without running anything: find_class must raise
        rather than import."""
        hostile = b"cos\nsystem\n(S'echo pwned'\ntR."
        with self.assertRaises(pickle.UnpicklingError):
            store_module.RestrictedUnpickler(io.BytesIO(hostile)).load()

    def test_loading_the_cache_goes_through_the_restricted_unpickler(self) -> None:
        """That the call site actually uses the guard, not just that the guard exists.

        The payload is a well-formed cache whose classifier is an instance of a class
        from a module outside the allowlist. An unrestricted unpickler would load it
        happily and return a dict; the restricted one refuses and the load yields None.
        A hostile os.system payload would NOT work here -- it returns an int, so the
        isinstance check rejects it either way and the test could not tell the two
        unpicklers apart.
        """
        payload = {
            "version": store_module.CLASSIFIER_CACHE_VERSION,
            "classifier": types.SimpleNamespace(weights=[1.0]),
        }
        self.store.classifier_path.write_bytes(pickle.dumps(payload))
        self.assertIsNone(
            self.store.load_classifier_cache(),
            "a classifier from a non-allowlisted module must not be unpickled",
        )


class ReviewSessionPersistence(unittest.TestCase):
    def setUp(self) -> None:
        self.folder = Path(tempfile.mkdtemp(prefix="bikini_review_"))
        self.store = FolderStore(self.folder)
        self.addCleanup(self.store.sqlite_cache.close)

    def test_absent_session_loads_as_nothing(self) -> None:
        self.assertIsNone(self.store.load_review_session())

    def test_a_saved_session_round_trips(self) -> None:
        self.store.save_review_session({"index": 4, "bucket": "Bikini"})
        self.assertEqual(self.store.load_review_session(), {"index": 4, "bucket": "Bikini"})

    def test_unreadable_json_is_quarantined_rather_than_deleted(self) -> None:
        """A corrupt session must not take the app down, and must not be silently
        destroyed either -- the bytes are kept next to it for inspection."""
        self.store.review_session_path.write_text("{not valid json", encoding="utf-8")
        with self.assertLogs(store_module.LOGGER, level="WARNING"):
            self.assertIsNone(self.store.load_review_session())
        preserved = self.store.review_session_path.with_name(f"{self.store.review_session_path.name}.broken")
        self.assertTrue(preserved.exists(), "the corrupt file must be preserved")
        self.assertEqual(preserved.read_text(encoding="utf-8"), "{not valid json")
        self.assertFalse(self.store.review_session_path.exists())

    def test_a_session_of_the_wrong_shape_is_quarantined(self) -> None:
        self.store.review_session_path.write_text("[1, 2, 3]", encoding="utf-8")
        self.assertIsNone(self.store.load_review_session())
        preserved = self.store.review_session_path.with_name(f"{self.store.review_session_path.name}.broken")
        self.assertTrue(preserved.exists())


class DuplicateGroupsAndCacheSize(unittest.TestCase):
    def setUp(self) -> None:
        self.folder = Path(tempfile.mkdtemp(prefix="bikini_dupes_"))
        payload = _make_image_bytes()
        self.twin_a = self.folder / "a.jpg"
        self.twin_b = self.folder / "b.jpg"
        self.lonely = self.folder / "c.jpg"
        self.twin_a.write_bytes(payload)
        self.twin_b.write_bytes(payload)
        self.lonely.write_bytes(_make_image_bytes(size=(48, 48)))
        self.store = FolderStore(self.folder)
        self.addCleanup(self.store.sqlite_cache.close)
        self.store.save_scan_cache(
            {},
            {
                p: {
                    "content_hash": content_hash_for_path(p),
                    "mtime_ns": p.stat().st_mtime_ns,
                    "size": p.stat().st_size,
                }
                for p in (self.twin_a, self.twin_b, self.lonely)
            },
        )

    def test_identical_files_are_grouped_and_singletons_are_not(self) -> None:
        groups = self.store.duplicate_groups()
        self.assertEqual(len(groups), 1, "only the two identical files form a group")
        members = next(iter(groups.values()))
        self.assertEqual(members, sorted([str(self.twin_a.resolve()), str(self.twin_b.resolve())]))
        self.assertNotIn(str(self.lonely.resolve()), members)

    def test_restricting_to_one_file_leaves_no_group(self) -> None:
        """A group needs two members present; filtering one twin out dissolves it."""
        groups = self.store.duplicate_groups([self.twin_a])
        self.assertEqual(groups, {})

    def test_cache_size_counts_nested_files_too(self) -> None:
        """The cache has subdirectories, so a top-level-only walk would under-report."""
        self.store.save_scan_metadata({"scanned": 3})
        nested_dir = self.store.cache_dir / "thumbs" / "deep"
        nested_dir.mkdir(parents=True)
        (nested_dir / "buried.bin").write_bytes(b"x" * 4096)

        size = self.store.cache_size_bytes()
        expected = sum(p.stat().st_size for p in self.store.cache_dir.rglob("*") if p.is_file())
        self.assertEqual(size, expected)
        self.assertGreaterEqual(size, 4096, "the nested file must be included in the total")

    def test_clearing_the_cache_empties_it_but_keeps_the_images(self) -> None:
        self.store.save_scan_metadata({"scanned": 3})
        self.store.save_review_session({"index": 1})
        self.assertGreater(self.store.cache_size_bytes(), 0)

        self.store.clear_cache()
        self.addCleanup(self.store.sqlite_cache.close)

        self.assertIsNone(self.store.load_review_session(), "cleared state must not come back")
        self.assertTrue(self.store.cache_dir.is_dir(), "the cache directory itself is recreated")
        for image in (self.twin_a, self.twin_b, self.lonely):
            self.assertTrue(image.exists(), "clearing the cache must never touch the user's images")


class PluginLoading(unittest.TestCase):
    """Plugins are arbitrary Python from the user's own plugins folder, executed in
    process. The defaults and the failure handling are the whole safety story."""

    def setUp(self) -> None:
        self.directory = Path(tempfile.mkdtemp(prefix="bikini_plugins_"))
        original = plugins.plugins_dir
        plugins.plugins_dir = lambda: self.directory
        self.addCleanup(setattr, plugins, "plugins_dir", original)
        self.samples: list[dict[str, object]] = [{"path": "a.jpg"}, {"path": "b.jpg"}]

    def _write_plugin(self, name: str, body: str) -> Path:
        path = self.directory / name
        path.write_text(textwrap.dedent(body), encoding="utf-8")
        return path

    def test_plugins_do_not_run_unless_explicitly_enabled(self) -> None:
        """The default must be off: a plugin file sitting in the folder is not consent
        to execute it."""
        marker = self.directory / "executed.txt"
        self._write_plugin(
            "evil.py",
            f"""
            from pathlib import Path
            Path(r"{marker}").write_text("ran", encoding="utf-8")

            def process_results(state, samples):
                return []
            """,
        )
        result = plugins.apply_plugins(None, self.samples)
        self.assertEqual(result, self.samples)
        self.assertFalse(marker.exists(), "a disabled plugin must not be imported at all")

    def test_a_missing_plugins_folder_is_not_an_error(self) -> None:
        """Pins the outcome -- samples come back untouched, nothing raises.

        It does not pin the exists() early-out above the loop: deleting that guard
        leaves this test green, because Path.glob over a missing directory yields
        nothing rather than raising. The guard saves a walk, not a crash.
        """
        shutil.rmtree(self.directory)
        self.assertEqual(plugins.apply_plugins(None, self.samples, enabled=True), self.samples)

    def test_an_enabled_plugin_can_rewrite_the_samples(self) -> None:
        self._write_plugin(
            "keep_first.py",
            """
            def process_results(state, samples):
                return samples[:1]
            """,
        )
        with self.assertLogs(plugins.LOGGER, level="INFO"):
            result = plugins.apply_plugins(None, self.samples, enabled=True)
        self.assertEqual(result, [{"path": "a.jpg"}])

    def test_a_plugin_returning_none_leaves_the_samples_alone(self) -> None:
        self._write_plugin(
            "observer.py",
            """
            def process_results(state, samples):
                return None
            """,
        )
        with self.assertLogs(plugins.LOGGER, level="INFO"):
            result = plugins.apply_plugins(None, self.samples, enabled=True)
        self.assertEqual(result, self.samples)

    def test_a_plugin_without_the_hook_is_reported_and_skipped(self) -> None:
        self._write_plugin("useless.py", "VALUE = 1\n")
        with self.assertLogs(plugins.LOGGER, level="WARNING") as captured:
            result = plugins.apply_plugins(None, self.samples, enabled=True)
        self.assertEqual(result, self.samples)
        self.assertIn("process_results", "".join(captured.output))

    def test_one_broken_plugin_does_not_stop_the_others(self) -> None:
        """A plugin that raises on import must not cost the user the plugins that work."""
        self._write_plugin("a_broken.py", "raise RuntimeError('boom')\n")
        self._write_plugin(
            "b_working.py",
            """
            def process_results(state, samples):
                return samples[:1]
            """,
        )
        with self.assertLogs(plugins.LOGGER, level="WARNING"):
            result = plugins.apply_plugins(None, self.samples, enabled=True)
        self.assertEqual(result, [{"path": "a.jpg"}], "the working plugin must still run")

    def test_a_hook_that_raises_is_caught(self) -> None:
        self._write_plugin(
            "thrower.py",
            """
            def process_results(state, samples):
                raise ValueError("bad plugin")
            """,
        )
        with self.assertLogs(plugins.LOGGER, level="WARNING"):
            result = plugins.apply_plugins(None, self.samples, enabled=True)
        self.assertEqual(result, self.samples)

    def test_plugins_are_chained_in_filename_order(self) -> None:
        self._write_plugin(
            "1_first.py",
            """
            def process_results(state, samples):
                return samples + [{"path": "from_first"}]
            """,
        )
        self._write_plugin(
            "2_second.py",
            """
            def process_results(state, samples):
                return samples + [{"path": "from_second"}]
            """,
        )
        with self.assertLogs(plugins.LOGGER, level="INFO"):
            result = plugins.apply_plugins(None, self.samples, enabled=True)
        self.assertEqual(
            [str(entry["path"]) for entry in result],
            ["a.jpg", "b.jpg", "from_first", "from_second"],
            "each plugin must receive the previous plugin's output",
        )


class VersionComparison(unittest.TestCase):
    def test_a_leading_v_and_whitespace_are_ignored(self) -> None:
        self.assertEqual(update_checker._version_tuple("  v1.4.2 "), (1, 4, 2))

    def test_non_numeric_segments_count_as_zero_rather_than_raising(self) -> None:
        self.assertEqual(update_checker._version_tuple("1.4.2b"), (1, 4, 0))

    def test_ordering_is_numeric_not_lexicographic(self) -> None:
        """The bug this prevents: "1.10.0" sorting below "1.9.0" as text."""
        self.assertGreater(update_checker._version_tuple("1.10.0"), update_checker._version_tuple("1.9.0"))


class UpdateChecking(unittest.TestCase):
    """The update check talks to the network, so every failure mode has to end in a
    quiet None rather than an exception reaching the GUI."""

    def setUp(self) -> None:
        self.original = update_checker.urlopen
        self.addCleanup(setattr, update_checker, "urlopen", self.original)
        self.requested: list[object] = []

    def _serve(self, body: str):
        class FakeResponse:
            def __enter__(self_inner):
                return self_inner

            def __exit__(self_inner, *_exc):
                return False

            def read(self_inner):
                return body.encode("utf-8")

        def fake_urlopen(request, timeout=None):
            self.requested.append(request)
            return FakeResponse()

        update_checker.urlopen = fake_urlopen

    def test_a_blank_url_asks_nothing_and_complains_about_nothing(self) -> None:
        """An unconfigured update URL is a normal state, not a failure to report.

        The silence matters as much as the None. Without the early return the blank URL
        reaches Request(), which raises ValueError, which is caught -- same None, but a
        warning in the log on every startup for a feature the user never enabled.
        """
        self._serve("{}")
        with self.assertNoLogs(update_checker.LOGGER, level="WARNING"):
            self.assertIsNone(update_checker.check_for_update("   "))
        self.assertEqual(self.requested, [], "a blank URL must not reach the network")

    def test_a_newer_version_is_reported(self) -> None:
        current = update_checker._version_tuple(__version__)
        newer = ".".join(str(part) for part in (current[0] + 1, *current[1:]))
        self._serve(json.dumps({"latest_version": newer, "download_url": "https://example.invalid/dl"}))
        result = update_checker.check_for_update("https://example.invalid/latest")
        self.assertEqual(result, {"latest_version": newer, "download_url": "https://example.invalid/dl"})

    def test_the_current_version_is_not_an_update(self) -> None:
        self._serve(json.dumps({"latest_version": __version__, "download_url": "x"}))
        self.assertIsNone(update_checker.check_for_update("https://example.invalid/latest"))

    def test_an_older_version_is_not_an_update(self) -> None:
        self._serve(json.dumps({"latest_version": "0.0.1", "download_url": "x"}))
        self.assertIsNone(update_checker.check_for_update("https://example.invalid/latest"))

    def test_a_payload_without_a_version_is_ignored(self) -> None:
        """Pins the outcome, not the `not latest` guard that produces it.

        That guard is redundant: _version_tuple("") is (0,), which loses the comparison
        below on its own, so deleting it leaves this test green. It would only matter if
        the app's own version were 0.0.0.
        """
        self._serve(json.dumps({"download_url": "x"}))
        self.assertIsNone(update_checker.check_for_update("https://example.invalid/latest"))

    def test_malformed_json_is_swallowed(self) -> None:
        self._serve("not json at all")
        with self.assertLogs(update_checker.LOGGER, level="WARNING"):
            self.assertIsNone(update_checker.check_for_update("https://example.invalid/latest"))

    def test_a_network_failure_is_swallowed(self) -> None:
        def explode(_request, timeout=None):
            raise OSError("simulated connection refused")

        update_checker.urlopen = explode
        with self.assertLogs(update_checker.LOGGER, level="WARNING"):
            self.assertIsNone(update_checker.check_for_update("https://example.invalid/latest"))


class UserPreferences(unittest.TestCase):
    def setUp(self) -> None:
        self.directory = Path(tempfile.mkdtemp(prefix="bikini_prefs_"))
        self.path = self.directory / "prefs.json"
        original = user_prefs.prefs_path
        user_prefs.prefs_path = lambda: self.path
        self.addCleanup(setattr, user_prefs, "prefs_path", original)

    def test_absent_prefs_load_as_an_empty_mapping(self) -> None:
        self.assertEqual(user_prefs.load_user_prefs(), {})

    def test_prefs_round_trip(self) -> None:
        user_prefs.save_user_prefs({"last_folder": "C:/photos", "threshold": 0.4})
        self.assertEqual(user_prefs.load_user_prefs(), {"last_folder": "C:/photos", "threshold": 0.4})

    def test_corrupt_prefs_are_quarantined_not_deleted(self) -> None:
        """Losing preferences is survivable; silently destroying the file is not."""
        self.path.write_text("{broken", encoding="utf-8")
        with self.assertLogs(user_prefs.LOGGER, level="WARNING"):
            self.assertEqual(user_prefs.load_user_prefs(), {})
        preserved = self.path.with_name(f"{self.path.name}.broken")
        self.assertTrue(preserved.exists())
        self.assertEqual(preserved.read_text(encoding="utf-8"), "{broken")

    def test_prefs_of_the_wrong_shape_are_quarantined(self) -> None:
        self.path.write_text("[1, 2, 3]", encoding="utf-8")
        with self.assertLogs(user_prefs.LOGGER, level="WARNING"):
            self.assertEqual(user_prefs.load_user_prefs(), {})
        self.assertTrue(self.path.with_name(f"{self.path.name}.broken").exists())

    def test_saving_creates_missing_parent_directories(self) -> None:
        nested = self.directory / "deep" / "nested" / "prefs.json"
        user_prefs.prefs_path = lambda: nested
        user_prefs.save_user_prefs({"a": 1})
        self.assertEqual(json.loads(nested.read_text(encoding="utf-8")), {"a": 1})


class PromptNormalisation(unittest.TestCase):
    """Prompts arrive from config as bare strings or (text, weight) pairs, so the
    normaliser is the first thing any bad config hits."""

    def test_bare_strings_get_a_unit_weight(self) -> None:
        texts, weights = BikiniScorer._normalize_prompts(["a photo", " a drawing "])
        self.assertEqual(texts, ["a photo", "a drawing"])
        np.testing.assert_allclose(weights, [1.0, 1.0])

    def test_weighted_pairs_keep_their_weight(self) -> None:
        texts, weights = BikiniScorer._normalize_prompts([("a photo", 2.5)])
        self.assertEqual(texts, ["a photo"])
        np.testing.assert_allclose(weights, [2.5])

    def test_an_unparsable_weight_falls_back_to_one(self) -> None:
        _texts, weights = BikiniScorer._normalize_prompts([("a photo", "heavy")])
        np.testing.assert_allclose(weights, [1.0])

    def test_a_non_positive_weight_falls_back_to_one(self) -> None:
        """A zero or negative weight would silently delete or invert a prompt."""
        _texts, weights = BikiniScorer._normalize_prompts([("a", 0.0), ("b", -3.0)])
        np.testing.assert_allclose(weights, [1.0, 1.0])

    def test_blank_prompts_are_dropped(self) -> None:
        texts, weights = BikiniScorer._normalize_prompts(["real", "   ", ""])
        self.assertEqual(texts, ["real"])
        self.assertEqual(len(weights), 1)

    def test_an_entirely_blank_prompt_list_still_yields_one_entry(self) -> None:
        """Downstream code indexes into this, so it must never come back empty."""
        texts, weights = BikiniScorer._normalize_prompts(["  ", ""])
        self.assertEqual(len(texts), 1)
        self.assertEqual(len(weights), 1)


class LabelCounts(unittest.TestCase):
    def test_each_label_value_lands_in_its_own_bucket(self) -> None:
        counts = BikiniScorer.label_counts({"a": 1, "b": 1, "c": 0, "d": 2, "e": 7})
        self.assertEqual(counts["good"], 2)
        self.assertEqual(counts["bad"], 1)
        self.assertEqual(counts["skip"], 1)

    def test_an_empty_mapping_counts_zero_everywhere(self) -> None:
        self.assertEqual(BikiniScorer.label_counts({}), {"good": 0, "bad": 0, "skip": 0, "unlabeled": 0})


class ClassifierTraining(unittest.TestCase):
    """Training is cached against a signature of the labelled set, so the signature has
    to move whenever the training data does -- otherwise a stale model is reused."""

    def setUp(self) -> None:
        self.folder = Path(tempfile.mkdtemp(prefix="bikini_train_"))
        self.scorer = BikiniScorer(backend=_shared()["backend"], config=ScannerConfig())
        self.rng = np.random.default_rng(0)
        self.paths: list[str] = []
        for index in range(10):
            path = self.folder / f"img_{index}.jpg"
            path.write_bytes(_make_image_bytes())
            self.paths.append(str(path))
        # Separable by the first component so the fit is meaningful rather than noise.
        self.embeddings = {
            path: np.concatenate(
                [[1.0 if index % 2 == 0 else -1.0], self.rng.normal(size=7)]
            ).astype(np.float32)
            for index, path in enumerate(self.paths)
        }
        self.labels = {path: (1 if index % 2 == 0 else 0) for index, path in enumerate(self.paths)}

    def test_too_few_labels_leave_no_classifier(self) -> None:
        few = dict(list(self.labels.items())[:4])
        count = self.scorer.train_classifier(self.embeddings, few)
        self.assertEqual(count, 4)
        self.assertIsNone(self.scorer.classifier, "fewer than six labels must not train a model")

    def test_a_single_class_leaves_no_classifier(self) -> None:
        """Eight accepts and no rejects is not something to learn a boundary from."""
        one_sided = dict.fromkeys(self.paths[:8], 1)
        self.scorer.train_classifier(self.embeddings, one_sided)
        self.assertIsNone(self.scorer.classifier)

    def test_skip_labels_are_not_training_data(self) -> None:
        mixed = dict(self.labels)
        for path in self.paths[:2]:
            mixed[path] = 2
        count = self.scorer.train_classifier(self.embeddings, mixed)
        self.assertEqual(count, len(self.paths) - 2, "label 2 (skip) must not be counted or trained on")

    def test_enough_labels_train_a_classifier(self) -> None:
        count = self.scorer.train_classifier(self.embeddings, self.labels)
        self.assertEqual(count, len(self.paths))
        self.assertIsNotNone(self.scorer.classifier)

    def test_the_signature_tracks_the_labelled_set(self) -> None:
        pairs = [(path, self.labels[path]) for path in self.paths]
        base = self.scorer._classifier_signature(pairs, 24)
        self.assertEqual(base, self.scorer._classifier_signature(list(reversed(pairs)), 24),
                         "ordering must not change the signature")
        flipped = [(path, 1 - label) for path, label in pairs]
        self.assertNotEqual(base, self.scorer._classifier_signature(flipped, 24),
                            "changed labels must change the signature")
        self.assertNotEqual(base, self.scorer._classifier_signature(pairs, 48),
                            "a different feature width must change the signature")

    def test_editing_a_labelled_image_changes_the_signature(self) -> None:
        """The signature folds in mtime and size, so a re-saved photo retrains rather
        than reusing a model fitted to the old pixels."""
        pairs = [(path, self.labels[path]) for path in self.paths]
        before = self.scorer._classifier_signature(pairs, 24)
        edited = Path(self.paths[0])
        edited.write_bytes(_make_image_bytes(size=(96, 96)))
        os.utime(edited, (0, 0))
        self.assertNotEqual(before, self.scorer._classifier_signature(pairs, 24))

    def test_a_matching_cached_classifier_is_reused_instead_of_refitting(self) -> None:
        store = FolderStore(self.folder)
        self.addCleanup(store.sqlite_cache.close)
        self.scorer.train_classifier(self.embeddings, self.labels, store=store)
        trained = self.scorer.classifier
        self.assertIsNotNone(trained)

        fresh = BikiniScorer(backend=_shared()["backend"], config=ScannerConfig())
        fresh.train_classifier(self.embeddings, self.labels, store=store)
        self.assertIsNotNone(fresh.classifier)
        self.assertEqual(
            store.load_classifier_cache()["signature"],
            fresh._classifier_signature(
                [(p, self.labels[p]) for p in self.paths],
                int(next(iter(self.embeddings.values())).shape[0]) * 2 + len(scorer_module.FEATURE_AXIS_ORDER),
            ),
        )

    def test_a_cache_from_different_labels_is_not_reused(self) -> None:
        store = FolderStore(self.folder)
        self.addCleanup(store.sqlite_cache.close)
        store.save_classifier_cache({"signature": "not-the-right-signature", "classifier": "a stale object"})
        self.scorer.train_classifier(self.embeddings, self.labels, store=store)
        self.assertNotEqual(self.scorer.classifier, "a stale object", "a stale cache must be refitted, not trusted")
        self.assertIsNotNone(self.scorer.classifier)

    def test_a_cache_that_cannot_be_written_does_not_lose_the_model(self) -> None:
        """A read-only cache folder used to throw the training away on every restart."""
        store = FolderStore(self.folder)
        self.addCleanup(store.sqlite_cache.close)

        def refuse(_self, _payload):
            raise OSError("simulated read-only cache")

        # FolderStore uses __slots__, so the method is patched on the class.
        original = FolderStore.save_classifier_cache
        FolderStore.save_classifier_cache = refuse
        self.addCleanup(setattr, FolderStore, "save_classifier_cache", original)

        with self.assertLogs(scorer_module.LOGGER, level="ERROR"):
            count = self.scorer.train_classifier(self.embeddings, self.labels, store=store)
        self.assertEqual(count, len(self.paths))
        self.assertIsNotNone(self.scorer.classifier, "the model is usable this session even if it cannot be saved")


class QualityEstimation(unittest.TestCase):
    def setUp(self) -> None:
        self.scorer = BikiniScorer(backend=_shared()["backend"], config=ScannerConfig())
        rng = np.random.default_rng(1)
        self.embeddings = {
            f"p{i}": np.concatenate([[1.0 if i % 2 == 0 else -1.0], rng.normal(size=7)]).astype(np.float32)
            for i in range(20)
        }
        self.labels = {f"p{i}": (1 if i % 2 == 0 else 0) for i in range(20)}

    def test_too_few_labels_give_no_estimate(self) -> None:
        """Pins the outcome, not the `< 6` guard that produces it.

        That guard is arithmetically redundant: test_size is max(2, round(n/4)), so for
        every n below 6 the `len(y) - test_size < 4` check below already returns None.
        Deleting it leaves this test green because there is no input that reaches one
        guard without the other.
        """
        few = {f"p{i}": self.labels[f"p{i}"] for i in range(4)}
        self.assertIsNone(self.scorer.estimate_quality(self.embeddings, few))

    def test_a_minority_class_of_two_gives_no_estimate(self) -> None:
        """Two examples of a class cannot honestly be split into train and test.

        The size matters: with 20 labels and a minority of 2 the split does succeed, and
        without the min-class guard this returns a perfect 1.0 computed from a single
        minority example -- a quality figure the user would read as certainty. At 10
        labels the split fails for unrelated reasons and the guard cannot be seen.
        """
        labels = {f"p{i}": (0 if i < 2 else 1) for i in range(20)}
        self.assertIsNone(self.scorer.estimate_quality(self.embeddings, labels))

    def test_a_separable_set_scores_above_chance(self) -> None:
        auc = self.scorer.estimate_quality(self.embeddings, self.labels)
        self.assertIsNotNone(auc)
        assert auc is not None
        self.assertGreaterEqual(auc, 0.5)
        self.assertLessEqual(auc, 1.0)

    def test_a_measured_cv_score_is_preferred_over_a_fresh_split(self) -> None:
        self.scorer.learning_outcome.cv_auc = 0.77
        self.assertAlmostEqual(self.scorer.estimate_quality(self.embeddings, self.labels), 0.77)


class ConfigProfiles(unittest.TestCase):
    def setUp(self) -> None:
        self.directory = Path(tempfile.mkdtemp(prefix="bikini_profiles_"))
        self.path = self.directory / "profiles.json"
        original = config_profiles.profiles_path
        config_profiles.profiles_path = lambda: self.path
        self.addCleanup(setattr, config_profiles, "profiles_path", original)

    def test_absent_profiles_load_as_empty(self) -> None:
        self.assertEqual(config_profiles.load_profiles(), {})

    def test_a_saved_profile_round_trips(self) -> None:
        config = ScannerConfig()
        config.threshold = 0.42
        config_profiles.save_profile("Mine", config)
        self.assertIn("Mine", config_profiles.load_profiles())
        restored = config_profiles.profile_config("Mine")
        self.assertIsNotNone(restored)
        assert restored is not None
        self.assertAlmostEqual(restored.threshold, 0.42)

    def test_a_builtin_name_cannot_be_overwritten(self) -> None:
        with self.assertRaises(ValueError):
            config_profiles.save_profile("Strict", ScannerConfig())

    def test_a_blank_name_is_refused(self) -> None:
        with self.assertRaises(ValueError):
            config_profiles.save_profile("   ", ScannerConfig())

    def test_builtin_profiles_cannot_be_deleted(self) -> None:
        """The profiles file is written by hand often enough to contain a shadowing
        entry. Without the built-in guard that entry would be deletable, and the name
        would then resolve to the built-in again -- a delete that appears to work and
        silently does nothing. An empty profiles file cannot show this: the lookup
        misses and reports False either way.
        """
        self.path.write_text(json.dumps({"Strict": {"threshold": 0.9}}), encoding="utf-8")
        self.assertFalse(config_profiles.delete_profile("Strict"))
        self.assertIn("Strict", config_profiles.load_profiles(), "the shadowing entry must survive")

    def test_deleting_an_unknown_profile_reports_false(self) -> None:
        self.assertFalse(config_profiles.delete_profile("never existed"))

    def test_a_custom_profile_can_be_deleted(self) -> None:
        config_profiles.save_profile("Mine", ScannerConfig())
        self.assertTrue(config_profiles.delete_profile("Mine"))
        self.assertNotIn("Mine", config_profiles.load_profiles())

    def test_builtins_lead_the_name_list_and_customs_are_sorted(self) -> None:
        config_profiles.save_profile("zebra", ScannerConfig())
        config_profiles.save_profile("alpha", ScannerConfig())
        names = config_profiles.profile_names()
        self.assertEqual(names[: len(BUILTIN_PROFILES)], list(BUILTIN_PROFILES))
        self.assertEqual(names[len(BUILTIN_PROFILES):], ["alpha", "zebra"])

    def test_an_unknown_profile_has_no_config(self) -> None:
        self.assertIsNone(config_profiles.profile_config("not a profile"))

    def test_corrupt_profiles_are_quarantined(self) -> None:
        self.path.write_text("{not json", encoding="utf-8")
        with self.assertLogs(config_profiles.LOGGER, level="WARNING"):
            self.assertEqual(config_profiles.load_profiles(), {})
        self.assertTrue(self.path.with_name(f"{self.path.name}.broken").exists())

    def test_a_profile_file_whose_entries_are_not_mappings_is_quarantined(self) -> None:
        """A JSON object of strings would otherwise reach ScannerConfig.from_mapping."""
        self.path.write_text(json.dumps({"Mine": "not a mapping"}), encoding="utf-8")
        with self.assertLogs(config_profiles.LOGGER, level="WARNING"):
            self.assertEqual(config_profiles.load_profiles(), {})
        self.assertTrue(self.path.with_name(f"{self.path.name}.broken").exists())

    def test_a_top_level_list_is_quarantined(self) -> None:
        self.path.write_text(json.dumps([{"threshold": 0.5}]), encoding="utf-8")
        with self.assertLogs(config_profiles.LOGGER, level="WARNING"):
            self.assertEqual(config_profiles.load_profiles(), {})
        self.assertTrue(self.path.with_name(f"{self.path.name}.broken").exists())

    def test_legacy_only_keys_are_reported_as_inert_under_the_cascade_pipeline(self) -> None:
        """A profile setting classifier_weight without pipeline=legacy looks like it is
        tuning the scoring and in fact does nothing."""
        self.assertEqual(
            config_profiles.inert_keys({"classifier_weight": 0.8, "threshold": 0.5}),
            {"classifier_weight"},
        )

    def test_nothing_is_inert_once_the_legacy_pipeline_is_selected(self) -> None:
        self.assertEqual(
            config_profiles.inert_keys({"classifier_weight": 0.8, "pipeline": "legacy"}),
            set(),
        )

    def test_neither_builtin_profile_ships_an_inert_key(self) -> None:
        """Both built-ins used to set weights that the cascade pipeline ignores."""
        for name, mapping in BUILTIN_PROFILES.items():
            with self.subTest(profile=name):
                self.assertEqual(config_profiles.inert_keys(mapping), set())

    def test_neither_builtin_profile_touches_the_age_gate(self) -> None:
        """A profile that quietly loosened the age gate would be an unpleasant surprise."""
        for name, mapping in BUILTIN_PROFILES.items():
            with self.subTest(profile=name):
                self.assertNotIn("exclude_minors", mapping)
                self.assertNotIn("minor_threshold", mapping)


class ExifOrientation(unittest.TestCase):
    """A portrait phone photo is stored landscape with a rotation tag. Ignoring the tag
    feeds the model a sideways image and reports the wrong dimensions."""

    def setUp(self) -> None:
        self.directory = Path(tempfile.mkdtemp(prefix="bikini_exif_"))

    def _write(self, name: str, orientation: int | None) -> Path:
        path = self.directory / name
        image = Image.new("RGB", (80, 40), color=(120, 90, 60))
        if orientation is None:
            image.save(path, format="JPEG")
        else:
            exif = image.getexif()
            exif[0x0112] = orientation
            image.save(path, format="JPEG", exif=exif)
        return path

    def test_an_untagged_image_keeps_its_stored_size(self) -> None:
        path = self._write("plain.jpg", None)
        self.assertEqual(image_formats.oriented_size(path), (80, 40))

    def test_a_transposed_orientation_swaps_the_reported_axes(self) -> None:
        path = self._write("rotated.jpg", 6)
        self.assertEqual(image_formats.oriented_size(path), (40, 80))

    def test_a_non_transposing_orientation_leaves_the_axes_alone(self) -> None:
        path = self._write("mirrored.jpg", 2)
        self.assertEqual(image_formats.oriented_size(path), (80, 40))

    def test_open_oriented_rotates_the_pixels_to_match(self) -> None:
        path = self._write("rotated.jpg", 6)
        with image_formats.open_oriented(path) as image:
            self.assertEqual(image.size, (40, 80))
            self.assertEqual(image.mode, "RGB")

    def test_the_decoded_image_outlives_the_file_handle(self) -> None:
        """Callers open inside a `with`; a handle-backed image would die on close."""
        path = self._write("plain.jpg", None)
        image = image_formats.open_oriented(path)
        path.unlink()
        self.assertEqual(image.size, (80, 40))
        image.load()

    def test_a_malformed_exif_block_does_not_cost_the_image(self) -> None:
        original = image_formats.ImageOps.exif_transpose

        def explode(_image):
            raise ValueError("simulated malformed EXIF")

        image_formats.ImageOps.exif_transpose = explode
        self.addCleanup(setattr, image_formats.ImageOps, "exif_transpose", original)

        path = self._write("plain.jpg", None)
        with Image.open(path) as handle:
            result = image_formats.apply_orientation(handle)
        self.assertEqual(result.size, (80, 40))
        result.load()


class LogRedaction(unittest.TestCase):
    """The log is something users paste into bug reports, so the home directory has to
    come out of it."""

    def test_the_home_directory_is_replaced(self) -> None:
        formatter = logging_setup.RedactingFormatter("%(message)s")
        record = logging.LogRecord(
            "t", logging.INFO, "p", 1, "failed to read %s" % (Path.home() / "photos" / "a.jpg"), None, None
        )
        message = formatter.format(record)
        self.assertNotIn(str(Path.home()), message)
        self.assertIn(logging_setup.RedactingFormatter.REPLACEMENT, message)
        self.assertIn("photos", message, "only the private prefix is removed")

    def test_a_message_with_no_home_path_is_untouched(self) -> None:
        formatter = logging_setup.RedactingFormatter("%(message)s")
        record = logging.LogRecord("t", logging.INFO, "p", 1, "nothing private here", None, None)
        self.assertEqual(formatter.format(record), "nothing private here")

    def test_a_missing_log_reads_as_empty(self) -> None:
        original = logging_setup.log_path
        logging_setup.log_path = lambda: Path(tempfile.mkdtemp(prefix="bikini_nolog_")) / "absent.log"
        self.addCleanup(setattr, logging_setup, "log_path", original)
        self.assertEqual(logging_setup.read_log_tail(), "")

    def test_a_log_that_exists_but_cannot_be_opened_reads_as_empty(self) -> None:
        """Reaches the OSError handler, which the missing-file case never does -- it
        returns on the exists() check well before any open."""
        directory = Path(tempfile.mkdtemp(prefix="bikini_dirlog_"))
        original = logging_setup.log_path
        logging_setup.log_path = lambda: directory  # a directory exists but will not open
        self.addCleanup(setattr, logging_setup, "log_path", original)
        self.assertEqual(logging_setup.read_log_tail(), "")

    def test_only_the_tail_of_a_large_log_is_read(self) -> None:
        directory = Path(tempfile.mkdtemp(prefix="bikini_biglog_"))
        path = directory / "app.log"
        path.write_text("A" * 5000 + "TAIL_MARKER", encoding="utf-8")
        original = logging_setup.log_path
        logging_setup.log_path = lambda: path
        self.addCleanup(setattr, logging_setup, "log_path", original)

        tail = logging_setup.read_log_tail(max_bytes=100)
        self.assertLessEqual(len(tail), 100)
        self.assertIn("TAIL_MARKER", tail, "the tail must be the end of the file, not the start")


class Configuration(unittest.TestCase):
    def test_round_trip_preserves_every_field(self) -> None:
        config = ScannerConfig()
        config.deep_scan = "always"
        config.minor_threshold = 0.22
        config.detail_weights = {"bikini": 0.5, "cleavage": 0.25}
        restored = ScannerConfig.from_mapping(config.to_dict())
        self.assertEqual(restored.to_dict(), config.to_dict())

    def test_invalid_values_fall_back_to_defaults(self) -> None:
        restored = ScannerConfig.from_mapping(
            {"deep_scan": "nonsense", "minor_threshold": "abc", "batch_size": -5, "pipeline": "???"}
        )
        defaults = ScannerConfig()
        self.assertEqual(restored.deep_scan, defaults.deep_scan)
        self.assertEqual(restored.minor_threshold, defaults.minor_threshold)
        self.assertEqual(restored.batch_size, defaults.batch_size)
        self.assertEqual(restored.pipeline, defaults.pipeline)

    def test_builtin_profiles_load(self) -> None:
        self.assertTrue(profile_names())
        for name in BUILTIN_PROFILES:
            self.assertIsInstance(profile_config(name), ScannerConfig)

    def test_builtin_profiles_contain_no_inert_settings(self) -> None:
        """A profile key that does nothing on the active pipeline is a lie to the user."""
        for name, mapping in BUILTIN_PROFILES.items():
            self.assertEqual(config_profiles.inert_keys(mapping), set(), f"{name} sets ignored keys")

    def test_builtin_profiles_actually_differ_in_strictness(self) -> None:
        strict = profile_config("Strict")
        loose = profile_config("Loose")
        self.assertGreater(strict.threshold, loose.threshold)
        self.assertEqual(strict.nsfw_filter, "exclude")
        self.assertEqual(loose.nsfw_filter, "include")

    def test_no_builtin_profile_weakens_the_age_gate(self) -> None:
        defaults = ScannerConfig()
        for name in BUILTIN_PROFILES:
            config = profile_config(name)
            self.assertTrue(config.exclude_minors, f"{name} switched the age gate off")
            self.assertLessEqual(config.minor_threshold, defaults.minor_threshold, f"{name} loosened the age gate")

    def test_folder_override_round_trip(self) -> None:
        store = FolderStore(Path(str(_shared()["root"])))
        config = ScannerConfig()
        config.threshold = 0.77
        store.save_config_override(config.to_dict())
        self.assertEqual(ScannerConfig.from_mapping(store.load_config_override()).threshold, 0.77)
        store.clear_config_override()
        self.assertIsNone(store.load_config_override())


class Robustness(unittest.TestCase):
    def test_unreadable_file_is_skipped_not_fatal(self) -> None:
        folder = Path(tempfile.mkdtemp(prefix="bikini_bad_"))
        good = _make_images(folder, count=2)  # two images plus a duplicate copy
        (folder / "broken.jpg").write_bytes(b"not an image")
        store = FolderStore(folder)
        scorer = BikiniScorer(backend=_shared()["backend"], config=ScannerConfig())
        state, _ = scan_and_score_folder(_shared()["backend"], store, scorer, threshold=0.35)
        self.assertEqual(len(state.paths), len(good))
        payload = json.loads(store.metadata_path.read_text(encoding="utf-8"))
        self.assertEqual(len(payload["skipped"]), 1)

    def test_empty_folder_scans_cleanly(self) -> None:
        folder = Path(tempfile.mkdtemp(prefix="bikini_empty_"))
        store = FolderStore(folder)
        scorer = BikiniScorer(backend=_shared()["backend"], config=ScannerConfig())
        state, samples = scan_and_score_folder(_shared()["backend"], store, scorer, threshold=0.35)
        self.assertEqual(state.paths, [])
        self.assertEqual(samples, [])

    def test_corrupt_json_caches_are_quarantined(self) -> None:
        folder = Path(tempfile.mkdtemp(prefix="bikini_corrupt_"))
        store = FolderStore(folder)
        store.labels_path.write_text("{not json", encoding="utf-8")
        self.assertEqual(store.load_labels(), {})
        store.save_labels({"a": 1})
        self.assertEqual(store.load_labels(), {"a": 1})

    def test_a_file_vanishing_mid_scan_is_survivable(self) -> None:
        """A scan lists the folder once, then stats each file repeatedly.

        Watch mode, a sync client or the user tidying up can remove a file in between.
        That used to raise FileNotFoundError and throw away the whole scan.
        """
        folder = Path(tempfile.mkdtemp(prefix="bikini_vanish_"))
        real = _make_images(folder, count=2)
        store = FolderStore(folder)
        scorer = BikiniScorer(backend=_shared()["backend"], config=ScannerConfig())
        scan_and_score_folder(_shared()["backend"], store, scorer, threshold=0.35)

        ghost = folder / "removed_after_listing.jpg"
        shutil.copyfile(real[0], ghost)
        scan_and_score_folder(_shared()["backend"], store, scorer, threshold=0.35)
        ghost.unlink()

        original = scorer_module.collect_image_paths
        scorer_module.collect_image_paths = lambda target: sorted([*original(target), ghost])
        try:
            state, _ = scan_and_score_folder(_shared()["backend"], store, scorer, threshold=0.35)
        finally:
            scorer_module.collect_image_paths = original
        self.assertNotIn(str(ghost), state.paths)
        self.assertTrue(state.paths, "the surviving images should still have been scored")
        payload = json.loads(store.metadata_path.read_text(encoding="utf-8"))
        self.assertTrue(
            any("removed_after_listing" in str(record.get("path", "")) for record in payload["skipped"]),
            "the vanished file should be recorded as skipped",
        )

    def test_labels_are_not_cached_when_the_write_fails(self) -> None:
        """A failed save must not leave the app thinking the decision was recorded."""
        folder = Path(tempfile.mkdtemp(prefix="bikini_rolabels_"))
        store = FolderStore(folder)
        store.save_labels({"a.jpg": 1})

        def explode(*_args, **_kwargs):
            raise OSError("disk full")

        original = store_module.atomic_write_json
        store_module.atomic_write_json = explode
        try:
            with self.assertRaises(OSError):
                store.save_labels({"a.jpg": 1, "b.jpg": 0})
        finally:
            store_module.atomic_write_json = original
        self.assertEqual(store.load_labels(), {"a.jpg": 1}, "the failed label was cached anyway")

    def test_out_of_range_settings_are_bounded(self) -> None:
        """from_mapping is the funnel for imported files, profiles and folder overrides."""
        defaults = ScannerConfig()
        self.assertEqual(ScannerConfig.from_mapping({"threshold": float("nan")}).threshold, defaults.threshold)
        self.assertEqual(ScannerConfig.from_mapping({"threshold": float("inf")}).threshold, defaults.threshold)
        self.assertEqual(ScannerConfig.from_mapping({"threshold": 99.0}).threshold, 1.0)
        self.assertEqual(ScannerConfig.from_mapping({"threshold": -5.0}).threshold, 0.0)
        self.assertEqual(ScannerConfig.from_mapping({"batch_size": 10**9}).batch_size, defaults.batch_size)
        self.assertGreater(ScannerConfig.from_mapping({"zero_shot_scale": 0.0}).zero_shot_scale, 0.0)
        # At zero the age gate matches nearly everything, which reads as "found nothing".
        self.assertGreater(ScannerConfig.from_mapping({"minor_threshold": 0.0}).minor_threshold, 0.0)

    def test_region_planning_handles_tiny_images(self) -> None:
        self.assertEqual([r.key for r in plan_regions((10, 10), [])], ["full"])
        regions = plan_regions((800, 900), [FaceBox(x=300, y=80, width=120, height=140)])
        keys = {region.key for region in regions}
        self.assertIn("full", keys)
        self.assertTrue({"chest0", "waist0", "torso0"} & keys)

    def test_face_detection_absence_is_unknown_not_zero(self) -> None:
        with Image.open(sorted(collect_image_paths(Path(str(_shared()["root"]))))[0]) as image:
            count = detect_face_count(image)
        self.assertTrue(count is None or isinstance(count, int))


class ImageDecoding(unittest.TestCase):
    """Orientation-aware loading is central to scoring and display consistency."""

    def test_exif_orientation_tag_is_applied(self) -> None:
        """A portrait JPEG stored on its side must report portrait dimensions and pixels."""
        folder = Path(tempfile.mkdtemp(prefix="bikini_exif_"))
        try:
            source = folder / "oriented.jpg"
            # Camera held sideways: raster is 200 wide, 100 tall, but tag says "rotate 90 CW".
            image = Image.new("RGB", (200, 100), color="blue")
            exif = image.getexif()
            exif[0x0112] = 6  # rotate 90 CW -> displayed 100x200
            image.save(source, exif=exif.tobytes())

            self.assertEqual(image_formats.oriented_size(source), (100, 200))

            opened = image_formats.open_oriented(source)
            self.assertEqual(opened.size, (100, 200))
            self.assertEqual(opened.mode, "RGB")
        finally:
            shutil.rmtree(folder, ignore_errors=True)

    def test_decode_version_invalidates_derived_caches(self) -> None:
        """When the decoder changes, derived caches are discarded once per folder."""
        folder = Path(tempfile.mkdtemp(prefix="bikini_inval_"))
        try:
            store = FolderStore(folder)
            # Simulate an old decoder version having been recorded.
            store.cache_meta_path.write_text(json.dumps({"decode_version": 1}), encoding="utf-8")
            # Pretend derived caches exist.
            store.embeddings_path.write_bytes(b"x")
            store.region_embeddings_path.write_bytes(b"x")

            # Creating a new FolderStore should notice the version mismatch and clear
            # stale derived files while keeping user-owned files such as labels.
            store2 = FolderStore(folder)
            self.assertFalse(store2.embeddings_path.exists())
            self.assertFalse(store2.region_embeddings_path.exists())
            payload = json.loads(store2.cache_meta_path.read_text(encoding="utf-8"))
            self.assertEqual(payload["decode_version"], image_formats.DECODE_VERSION)
        finally:
            shutil.rmtree(folder, ignore_errors=True)


class SQLiteCacheMigration(unittest.TestCase):
    """Legacy NPZ/JSON caches are migrated into SQLite once, then removed."""

    def test_legacy_npz_json_cache_migrated_to_sqlite(self) -> None:
        folder = Path(tempfile.mkdtemp(prefix="bikini_sqlite_"))
        try:
            image_path = folder / "a.jpg"
            image_path.write_bytes(_make_image_bytes())
            cache_dir = folder / ".bikini_scanner_cache"
            cache_dir.mkdir()
            (cache_dir / "cache_meta.json").write_text(
                json.dumps({"decode_version": image_formats.DECODE_VERSION}), encoding="utf-8"
            )
            content_hash = content_hash_for_path(image_path)
            embedding = np.arange(4, dtype=np.float32)
            np.savez(cache_dir / "embeddings.npz", **{content_hash: embedding})
            (cache_dir / "embeddings_index.json").write_text(
                json.dumps(
                    {
                        str(image_path): {
                            "path": str(image_path),
                            "mtime_ns": image_path.stat().st_mtime_ns,
                            "size": image_path.stat().st_size,
                            "content_hash": content_hash,
                        }
                    }
                ),
                encoding="utf-8",
            )
            (cache_dir / "face_counts.json").write_text(json.dumps({content_hash: 2}), encoding="utf-8")
            np.savez(
                cache_dir / "region_embeddings.npz",
                **{f"{content_hash}|v1|upper": np.arange(4, dtype=np.float32)},
            )

            store = FolderStore(folder)
            self.assertTrue((cache_dir / "cache.db").exists())
            self.assertFalse((cache_dir / "embeddings.npz").exists())
            self.assertFalse((cache_dir / "embeddings_index.json").exists())
            self.assertFalse((cache_dir / "face_counts.json").exists())
            self.assertFalse((cache_dir / "region_embeddings.npz").exists())
            np.testing.assert_allclose(store.lookup_content_embedding(content_hash), embedding)
            self.assertEqual(store.lookup_face_count(content_hash), 2)
            regions = store.lookup_region_embeddings(content_hash, "v1")
            self.assertIn("upper", regions)
            np.testing.assert_allclose(regions["upper"], embedding)
        finally:
            shutil.rmtree(folder, ignore_errors=True)

    def test_decode_version_bump_deletes_sqlite_and_legacy_caches(self) -> None:
        folder = Path(tempfile.mkdtemp(prefix="bikini_sqlite_inval_"))
        try:
            image_path = folder / "a.jpg"
            image_path.write_bytes(_make_image_bytes())
            cache_dir = folder / ".bikini_scanner_cache"
            cache_dir.mkdir()
            content_hash = content_hash_for_path(image_path)
            (cache_dir / "cache_meta.json").write_text(
                json.dumps({"decode_version": image_formats.DECODE_VERSION - 1}), encoding="utf-8"
            )
            np.savez(cache_dir / "embeddings.npz", **{content_hash: np.arange(4, dtype=np.float32)})
            (cache_dir / "cache.db").write_bytes(b"sqlite")
            (cache_dir / "cache.db-wal").write_bytes(b"wal")

            store = FolderStore(folder)
            self.assertFalse((cache_dir / "embeddings.npz").exists())
            # The stale DB is replaced with a fresh empty one, so lookups return nothing.
            self.assertTrue((cache_dir / "cache.db").exists())
            self.assertIsNone(store.lookup_content_embedding(content_hash))
            meta = json.loads((cache_dir / "cache_meta.json").read_text(encoding="utf-8"))
            self.assertEqual(meta["decode_version"], image_formats.DECODE_VERSION)
        finally:
            shutil.rmtree(folder, ignore_errors=True)


class IgnoreMarkers(unittest.TestCase):
    """Output directories are not re-scanned as input on a later run."""

    def test_ignored_subdirectory_is_skipped(self) -> None:
        folder = Path(tempfile.mkdtemp(prefix="bikini_ignore_"))
        try:
            (folder / "keep.jpg").write_bytes(_make_image_bytes())
            ignored = folder / "already_output"
            ignored.mkdir()
            (ignored / ".bikini_scanner_ignore").touch()
            (ignored / "copy.jpg").write_bytes(_make_image_bytes())
            self.assertEqual({p.name for p in collect_image_paths(folder)}, {"keep.jpg"})
        finally:
            shutil.rmtree(folder, ignore_errors=True)

    def test_output_transfer_writes_ignore_marker(self) -> None:
        destination = Path(tempfile.mkdtemp(prefix="bikini_out_marker_"))
        source = destination.with_name(destination.name + "_src")
        source.mkdir()
        (source / "a.jpg").write_bytes(_make_image_bytes())
        try:
            plan = output_ops.build_transfer_plan(
                [str(source / "a.jpg")], destination, {str(source / "a.jpg"): 0.5}, {}, output_ops.OutputOptions()
            )
            output_ops.execute_transfer_plan(plan, move=False)
            self.assertTrue((destination / ".bikini_scanner_ignore").is_file())
            self.assertEqual({p.name for p in collect_image_paths(destination)}, set())
        finally:
            shutil.rmtree(destination, ignore_errors=True)
            shutil.rmtree(source, ignore_errors=True)

    def test_html_assets_get_ignore_marker(self) -> None:
        destination = Path(tempfile.mkdtemp(prefix="bikini_html_marker_"))
        try:
            output_ops.build_html_report(
                destination / "report.html",
                [{"path": str(destination / "dummy.jpg"), "score": 0.5, "bucket": "Bikini"}],
                {},
                {},
                max_embedded_thumbnails=0,
            )
            assets = destination / "report_assets"
            self.assertTrue((assets / ".bikini_scanner_ignore").is_file())
        finally:
            shutil.rmtree(destination, ignore_errors=True)


class FolderOverrideTrust(unittest.TestCase):
    """A config override ships inside the folder being scanned, so it is untrusted.

    Anything that could reach off this machine, run code, or weaken the age gate must
    be refused; local ranking and performance knobs are still allowed through.
    """

    def test_egress_and_safety_keys_are_refused(self) -> None:
        hostile = {
            "vlm_enabled": True,
            "vlm_base_url": "https://attacker.example/v1",
            "vlm_api_key": "stolen",
            "enable_plugins": True,
            "pipeline": "legacy",
            "exclude_minors": False,
            "minor_threshold": 1.0,
            "axis_prompts": {"child": {"positive": ["x"], "negative": ["y"]}},
            "backend": "clip-onnx",
            "model_name": "attacker/model",
            "refine_model": "attacker/model",
            "global_learning": True,
        }
        accepted, refused = filter_folder_override(hostile)
        self.assertEqual(accepted, {})
        self.assertEqual(sorted(refused), sorted(hostile))

    def test_benign_tuning_keys_are_allowed(self) -> None:
        accepted, refused = filter_folder_override({"threshold": 0.5, "deep_scan": "always", "batch_size": 4})
        self.assertEqual(accepted, {"threshold": 0.5, "deep_scan": "always", "batch_size": 4})
        self.assertEqual(refused, [])

    def test_hostile_override_cannot_reach_the_config(self) -> None:
        folder = Path(tempfile.mkdtemp(prefix="bikini_override_"))
        self.addCleanup(shutil.rmtree, folder, True)
        store = FolderStore(folder)
        store.save_config_override({"vlm_enabled": True, "vlm_base_url": "https://attacker.example/v1"})

        accepted, refused = filter_folder_override(store.load_config_override())
        config = ScannerConfig.from_mapping({**ScannerConfig().to_dict(), **accepted})

        self.assertFalse(config.vlm_enabled)
        self.assertNotIn("attacker.example", config.vlm_base_url)
        self.assertIn("vlm_base_url", refused)

    def test_saved_override_round_trips_without_refusals(self) -> None:
        """What the app writes itself must not warn when it is read back."""
        payload, _ = filter_folder_override(ScannerConfig().to_dict())
        _, refused = filter_folder_override(payload)
        self.assertEqual(refused, [])


class LegacyPipelineAgeGate(unittest.TestCase):
    """`pipeline` is a scoring choice; it is not an opt-out from the age gate."""

    AXES = (
        "person",
        "female",
        "child",
        "adult",
        "bikini",
        "bikini_top",
        "bikini_bottom",
        "cleavage",
        "midriff",
        "nsfw",
    )

    def _scorer(self, pipeline: str, exclude_minors: bool) -> BikiniScorer:
        config = ScannerConfig()
        config.pipeline = pipeline
        config.exclude_minors = exclude_minors
        axes = self.AXES

        class _MinorScorer(BikiniScorer):
            def axis_zero_shot_scores(self, embeddings, *args, **kwargs):  # type: ignore[no-untyped-def]
                count = len(embeddings)
                scores = {axis: np.full((count,), 0.5, dtype=np.float32) for axis in axes}
                scores["person"][:] = 0.99
                scores["female"][:] = 0.99
                scores["child"][:] = 0.99
                scores["adult"][:] = 0.01
                for axis in ("bikini", "bikini_top", "bikini_bottom", "cleavage", "midriff"):
                    scores[axis][:] = 0.99
                return scores

        return _MinorScorer(backend=_shared()["backend"], config=config)

    @staticmethod
    def _embeddings(count: int = 3) -> np.ndarray:
        return np.tile(np.eye(1, 512, dtype=np.float32), (count, 1))

    def test_legacy_pipeline_zeroes_and_hides_minors(self) -> None:
        scorer = self._scorer("legacy", exclude_minors=True)
        state = scorer.score_state(["a.jpg", "b.jpg", "c.jpg"], self._embeddings(), {})
        np.testing.assert_allclose(state.scores, 0.0)
        self.assertIsNotNone(state.excluded)
        self.assertTrue(np.asarray(state.excluded).all())
        self.assertFalse(scorer.state_visibility(state).any())
        self.assertEqual(set(state.cascade_stage), {cascade.STAGE_MINOR})

    def test_legacy_matches_cascade_on_the_age_gate(self) -> None:
        embeddings = self._embeddings()
        results = {}
        for pipeline in ("cascade", "legacy"):
            scorer = self._scorer(pipeline, exclude_minors=True)
            state = scorer.score_state(["a.jpg", "b.jpg", "c.jpg"], embeddings, {})
            results[pipeline] = scorer.state_visibility(state).tolist()
        self.assertEqual(results["cascade"], results["legacy"])

    def test_legacy_without_the_gate_still_scores(self) -> None:
        """Turning the gate off is still the user's call, in either pipeline."""
        scorer = self._scorer("legacy", exclude_minors=False)
        state = scorer.score_state(["a.jpg", "b.jpg", "c.jpg"], self._embeddings(), {})
        self.assertTrue((state.scores > 0).all())
        self.assertIsNone(state.excluded)


class TrashBatching(unittest.TestCase):
    """A failure partway through a trash run is not "nothing happened"."""

    def test_partial_failure_reports_both_halves(self) -> None:
        folder = Path(tempfile.mkdtemp(prefix="bikini_trash_"))
        self.addCleanup(shutil.rmtree, folder, True)
        paths = []
        for index in range(4):
            path = folder / f"f{index}.jpg"
            path.write_bytes(_make_image_bytes())
            paths.append(str(path))

        removed: list[str] = []

        def fake_send2trash(target: str) -> None:
            if target == paths[1]:
                raise OSError("locked by another process")
            removed.append(target)

        module = types.ModuleType("send2trash")
        module.send2trash = fake_send2trash  # type: ignore[attr-defined]
        original = sys.modules.get("send2trash")
        sys.modules["send2trash"] = module
        try:
            outcome = output_ops.trash_files(paths)
        finally:
            if original is None:
                sys.modules.pop("send2trash", None)
            else:
                sys.modules["send2trash"] = original

        self.assertTrue(outcome.available)
        # The batch continued past the failure instead of abandoning the rest.
        self.assertEqual(outcome.trashed_count, 3)
        self.assertEqual(outcome.failed_count, 1)
        self.assertEqual(removed, [paths[0], paths[2], paths[3]])
        self.assertEqual(outcome.failures[0][0], paths[1])

    def test_missing_send2trash_attempts_nothing(self) -> None:
        original = sys.modules.get("send2trash")
        sys.modules["send2trash"] = None  # type: ignore[assignment]
        try:
            outcome = output_ops.trash_files(["a.jpg"])
        finally:
            if original is None:
                sys.modules.pop("send2trash", None)
            else:
                sys.modules["send2trash"] = original
        self.assertFalse(outcome.available)
        self.assertEqual(outcome.trashed_count, 0)


class VLMEndpointTrust(unittest.TestCase):
    def test_non_http_schemes_are_rejected(self) -> None:
        for url in ("ftp://host/v1", "file:///etc/passwd", "not a url", "gopher://host/v1"):
            with self.assertRaises(ValueError):
                VLMClient(url, "model")

    def test_loopback_spellings_are_recognised(self) -> None:
        for url in (
            "http://localhost:11434/v1",
            "http://127.0.0.1:8080/v1",
            "http://127.5.5.5/v1",
            "http://[::1]:8080/v1",
            "http://[::ffff:127.0.0.1]/v1",
        ):
            self.assertTrue(is_local_endpoint(url), url)

    def test_remote_endpoints_are_not_treated_as_local(self) -> None:
        for url in ("https://attacker.example/v1", "http://10.0.0.5/v1", "https://api.openai.com/v1"):
            self.assertFalse(is_local_endpoint(url), url)


class GlobalClassifierUnpickling(unittest.TestCase):
    """The global classifier gets the same restricted unpickler as the per-folder one."""

    def test_disallowed_module_is_refused(self) -> None:
        store = GlobalLearningStore("restricted-unpickler-test")
        store.classifier_path.parent.mkdir(parents=True, exist_ok=True)
        # A payload whose unpickling would import os.system.
        store.classifier_path.write_bytes(pickle.dumps({"classifier": os.system, "feature_version": 0}))
        self.addCleanup(store.classifier_path.unlink, True)
        self.assertIsNone(store.load_classifier())

    def test_restricted_unpickler_blocks_arbitrary_imports(self) -> None:
        blob = pickle.dumps(os.system)
        with self.assertRaises(pickle.UnpicklingError):
            store_module.RestrictedUnpickler(io.BytesIO(blob)).load()


class RefineWeightSelection(unittest.TestCase):
    """vlm_weight and refine_weight are separate knobs, so the blend must tell them apart."""

    def _state(self, refine: RefineResult, vlm_weight: float, refine_weight: float) -> ScoreState:
        config = ScannerConfig()
        config.vlm_weight = vlm_weight
        config.refine_weight = refine_weight
        config.exclude_minors = False
        scorer = BikiniScorer(backend=_shared()["backend"], config=config)
        embeddings = np.tile(np.eye(1, 512, dtype=np.float32), (2, 1))
        return scorer.score_state(["a.jpg", "b.jpg"], embeddings, {}, refine=refine)

    def test_vlm_and_clip_sources_use_different_weights(self) -> None:
        scores = np.array([1.0, 1.0], dtype=np.float32)
        minor = np.zeros(2, dtype=bool)

        vlm = self._state(RefineResult(scores, minor, source="vlm"), vlm_weight=1.0, refine_weight=0.0)
        clip = self._state(RefineResult(scores, minor, source="clip"), vlm_weight=1.0, refine_weight=0.0)

        # vlm_weight=1.0 hands the VLM pass the whole vote; refine_weight=0.0 gives the
        # CLIP pass none. Identical inputs must therefore land on different scores.
        self.assertFalse(np.allclose(vlm.zero_shot_scores, clip.zero_shot_scores))
        np.testing.assert_allclose(vlm.zero_shot_scores, 1.0, atol=1e-6)

    def test_refine_result_defaults_to_clip(self) -> None:
        self.assertEqual(RefineResult(np.zeros(1, np.float32), np.zeros(1, bool)).source, "clip")


class EmbeddingCacheIsolation(unittest.TestCase):
    """Cached full-frame embeddings must not leak between models."""

    def setUp(self) -> None:
        self.folder = Path(tempfile.mkdtemp(prefix="bikini_ns_"))
        self.addCleanup(shutil.rmtree, self.folder, True)
        self.image = self.folder / "one.jpg"
        self.image.write_bytes(_make_image_bytes())

    def test_changing_model_discards_derived_caches(self) -> None:
        store = FolderStore(self.folder)
        store.ensure_embedding_namespace("clip_torch-model_a")
        store.save_embeddings({self.image: np.ones(512, dtype=np.float32)})
        self.assertTrue(store.get_cached_embeddings([self.image]))

        # Same model again: the cache survives.
        store = FolderStore(self.folder)
        store.ensure_embedding_namespace("clip_torch-model_a")
        self.assertTrue(store.get_cached_embeddings([self.image]))

        # Different model: the stale vectors go.
        store = FolderStore(self.folder)
        store.ensure_embedding_namespace("clip_torch-model_b")
        self.assertFalse(store.get_cached_embeddings([self.image]))

    def test_labels_survive_a_model_change(self) -> None:
        store = FolderStore(self.folder)
        store.ensure_embedding_namespace("model_a")
        store.save_labels({str(self.image): 1})
        store = FolderStore(self.folder)
        store.ensure_embedding_namespace("model_b")
        self.assertEqual(store.load_labels().get(str(self.image)), 1)

    def test_wrong_width_embeddings_are_ignored(self) -> None:
        store = FolderStore(self.folder)
        store.save_embeddings({self.image: np.ones(512, dtype=np.float32)})
        self.assertTrue(store.get_cached_embeddings([self.image], 512))
        # A 768-d model must not be handed a 512-d vector.
        self.assertFalse(store.get_cached_embeddings([self.image], 768))

    def test_decode_version_and_namespace_coexist_in_cache_meta(self) -> None:
        store = FolderStore(self.folder)
        store.ensure_embedding_namespace("model_a")
        payload = json.loads(store.cache_meta_path.read_text(encoding="utf-8"))
        self.assertEqual(payload.get("embedding_namespace"), "model_a")
        self.assertIn("decode_version", payload)


def _cleanup() -> None:
    shutil.rmtree(_STATE_DIR, ignore_errors=True)
    root = _SHARED.get("root")
    if root:
        shutil.rmtree(str(root), ignore_errors=True)


if __name__ == "__main__":
    try:
        # unittest.main already resolves bare names against this module; prefixing them
        # with "__main__." made it look up __main__.__main__ and fail on every class.
        unittest.main(argv=sys.argv, exit=False, verbosity=2)
    finally:
        _cleanup()
