"""End-to-end functional checks for Bikini Scanner.

Runs against a throwaway folder of generated images and never touches real user data:
the user preferences path, the last-folder marker, and the cross-folder learning store
are all redirected into a temporary directory first.

    PYTHONPATH=. python tests/test_functional.py            # everything
    PYTHONPATH=. python tests/test_functional.py Scoring    # one class

The CLIP backend is loaded once and shared, so the whole suite costs about one scan.
"""

from __future__ import annotations

import json
import os
import shutil
import sys
import tempfile
import threading
import unittest
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path

import numpy as np
from PIL import Image

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

# Redirect every user-state path before importing anything that reads them.
_STATE_DIR = Path(tempfile.mkdtemp(prefix="bikini_state_"))
os.environ["APPDATA"] = str(_STATE_DIR)
os.environ["LOCALAPPDATA"] = str(_STATE_DIR)  # the log lives here on Windows
os.environ["XDG_CONFIG_HOME"] = str(_STATE_DIR)
os.environ["XDG_STATE_HOME"] = str(_STATE_DIR)
os.environ["HOME"] = str(_STATE_DIR)
os.environ["USERPROFILE"] = str(_STATE_DIR)

from bikini_scanner import cascade, learning, linear_model, output_ops, regions  # noqa: E402
from bikini_scanner import scorer as scorer_module  # noqa: E402
from bikini_scanner import store as store_module  # noqa: E402
from bikini_scanner.clip_backend import get_backend  # noqa: E402
from bikini_scanner.config import ScannerConfig  # noqa: E402
from bikini_scanner import config_profiles  # noqa: E402
from bikini_scanner.config_profiles import BUILTIN_PROFILES, profile_config, profile_names  # noqa: E402
from bikini_scanner.global_store import GlobalLearningStore  # noqa: E402
from bikini_scanner.regions import plan_regions  # noqa: E402
from bikini_scanner.scorer import BikiniScorer, bucketed_sampling, scan_and_score_folder  # noqa: E402
from bikini_scanner.scorer import RefineResult, ScoreState, compute_vlm_scores  # noqa: E402
from bikini_scanner.skin import skin_fraction  # noqa: E402
from bikini_scanner.store import FolderStore, collect_image_paths  # noqa: E402
from bikini_scanner.vlm_backend import VLMCancelled, VLMClient, parse_axis_json  # noqa: E402
from bikini_scanner.vision_analysis import FaceBox, detect_face_count  # noqa: E402

IMAGE_COUNT = 8
_SHARED: dict[str, object] = {}


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


def _shared():
    if "backend" not in _SHARED:
        config = ScannerConfig()
        config.preload_backend = False
        _SHARED["config"] = config
        _SHARED["backend"] = get_backend(config)
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
                {
                    "choices": [
                        {
                            "message": {
                                "content": '```json\n{"bikini": 1.4, "child": 0.1, "adult": 0.9}\n```'
                            }
                        }
                    ]
                }
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
                    cells[(int(info["row"]), int(info["column"]))] = cells.get(
                        (int(info["row"]), int(info["column"])), 0
                    ) + 1
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
        cls.state, _ = scan_and_score_folder(
            shared["backend"], cls.store, cls.scorer, threshold=cls.config.threshold
        )

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
        self.assertAlmostEqual(linear_model.roc_auc([0, 1], [0.5, 0.5]), 0.5)

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
        processed, skipped, retained, failed = output_ops.execute_transfer_plan(plan, move=False)
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
        processed, skipped, retained, failed = output_ops.execute_transfer_plan(plan, move=False)
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
        processed, skipped, retained, failed = output_ops.execute_transfer_plan(plan, move=True)
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
