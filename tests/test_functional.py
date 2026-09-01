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
import os
import pickle
import shutil
import sys
import tempfile
import threading
import time
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
    output_ops,
    regions,
    safe_io,
)
from bikini_scanner import run as run_module
from bikini_scanner import scorer as scorer_module
from bikini_scanner import store as store_module
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

            # Keyed by the containing widget as well as the cell: the dialog is a
            # notebook now, so row 7 column 0 exists once per tab and a bare (row,
            # column) key would count unrelated tabs as a collision.
            cells: dict[tuple[str, int, int], int] = {}
            age_box = None
            for widget in walk(dialog):
                info = widget.grid_info() if hasattr(widget, "grid_info") else None
                if info:
                    key = (str(widget.winfo_parent()), int(info["row"]), int(info["column"]))
                    cells[key] = cells.get(key, 0) + 1
                try:
                    if "may show a minor" in str(widget.cget("text")):
                        age_box = widget
                except Exception:  # noqa: BLE001
                    continue

            self.assertIsNotNone(age_box, "the age-gate checkbox is missing from Settings")
            self.assertTrue(age_box.winfo_ismapped(), "the age-gate checkbox is not displayed")
            position = (
                str(age_box.winfo_parent()),
                int(age_box.grid_info()["row"]),
                int(age_box.grid_info()["column"]),
            )
            self.assertEqual(cells[position], 1, "another widget shares the checkbox's grid cell")
            variable = str(age_box.cget("variable"))
            initial = root.getvar(variable)
            age_box.invoke()
            self.assertNotEqual(str(initial), str(root.getvar(variable)), "the checkbox does not toggle")
            app._closing = True
        finally:
            root.destroy()


class SettingsWindow(unittest.TestCase):
    """The form has to fit its own labels, and the user has to be able to widen it."""

    def setUp(self) -> None:
        tkinter = __import__("tkinter")
        try:
            self.root = tkinter.Tk()
        except Exception as exc:  # noqa: BLE001
            self.skipTest(f"no display available: {exc}")
        from bikini_scanner import gui as gui_module

        for name in ("_resume_last_folder_if_any", "_maybe_preload_backend", "_maybe_show_first_run_guide"):
            setattr(gui_module.BikiniScannerApp, name, lambda self: None)
        self.app = gui_module.BikiniScannerApp(self.root, config=ScannerConfig(preload_backend=False))
        before = set(self.root.winfo_children())
        self.app.open_settings_dialog()
        self.root.update()
        self.dialog = next(w for w in self.root.winfo_children() if w not in before)

    def tearDown(self) -> None:
        self.app._closing = True
        self.root.destroy()

    def test_the_window_can_be_widened(self) -> None:
        # It was fixed-width, so a label wider than the form had nowhere to go.
        self.assertEqual(
            tuple(bool(value) for value in self.dialog.resizable()),
            (True, True),
            "the settings window cannot be resized in both directions",
        )

    def test_it_opens_wide_enough_for_its_own_contents(self) -> None:
        self.root.update_idletasks()
        self.assertGreaterEqual(
            int(self.dialog.winfo_width()),
            int(self.dialog.winfo_reqwidth()),
            "the settings window opens narrower than the widest tab needs, so text is clipped",
        )

    def test_every_group_of_settings_is_reachable(self) -> None:
        notebook = self.app.settings_notebook
        tabs = [str(notebook.tab(index, "text")) for index in range(notebook.index("end"))]
        self.assertEqual(tabs, ["Detection", "Model", "Prompts", "Advanced"])
        # Detection is what a reviewer opens Settings for, so it is the tab on top.
        self.assertEqual(notebook.index(notebook.select()), 0)


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


class GuiReviewQueue(unittest.TestCase):
    """The review loop must count down and never re-serve a decided photo."""

    def setUp(self) -> None:
        tkinter = __import__("tkinter")
        try:
            self.root = tkinter.Tk()
        except Exception as exc:  # noqa: BLE001
            self.skipTest(f"no display available: {exc}")
        from bikini_scanner import gui as gui_module

        for name in ("_resume_last_folder_if_any", "_maybe_preload_backend", "_maybe_show_first_run_guide"):
            setattr(gui_module.BikiniScannerApp, name, lambda self: None)
        self.folder = Path(tempfile.mkdtemp(prefix="bikini_queue_"))
        _make_images(self.folder, count=6)
        self.app = gui_module.BikiniScannerApp(self.root, config=ScannerConfig(preload_backend=False))
        # Both are saved preferences shared by every app built in this run, so a test
        # that changes one would otherwise decide what the next test sees.
        self.app.label_filter_var.set("all")
        self.app.hide_decided_var.set(True)
        self.app._set_folder(str(self.folder))
        self.paths = [str(path) for path in collect_image_paths(self.folder)]
        self.app.current_state = scorer_module.ScoreState(
            paths=self.paths,
            embeddings=np.zeros((len(self.paths), 4), dtype=np.float32),
            zero_shot_scores=np.full(len(self.paths), 0.5, dtype=np.float32),
            scores=np.full(len(self.paths), 0.9, dtype=np.float32),
            axis_scores={"bikini": np.full(len(self.paths), 0.6, dtype=np.float32)},
            face_counts=None,
            classifier_trained=False,
            classifier_label_count=0,
            excluded=np.zeros(len(self.paths), dtype=bool),
        )

        class _Passthrough:
            config = ScannerConfig()

            def state_visibility(self, state):
                return np.ones((len(state.paths),), dtype=bool)

            def label_counts(self, labels):
                values = list(labels.values())
                return {
                    "good": values.count(1),
                    "bad": values.count(0),
                    "skip": values.count(2),
                    "unlabeled": 0,
                }

            def estimate_quality(self, embeddings_by_path, labels):
                return None

        self.app.scorer = _Passthrough()
        self.app.page_samples = [
            {"path": path, "score": 0.9, "bucket": "Bikini"} for path in self.paths
        ]
        self.app.displayed_samples = list(self.app.page_samples)
        self.app.current_samples = list(self.app.page_samples)

    def tearDown(self) -> None:
        self.app._closing = True
        self.root.destroy()
        shutil.rmtree(self.folder, ignore_errors=True)

    def test_focus_skips_photos_that_were_already_decided(self) -> None:
        assert self.app.store is not None
        # Everything but the last photo has been judged already.
        self.app.store.save_labels(dict.fromkeys(self.paths[:-1], 1))
        self.app.focused_path = self.paths[0]
        self.app._advance_focus_after(self.paths[0])
        self.assertEqual(
            self.app.focused_path,
            self.paths[-1],
            "the focus landed on a photo that had already been accepted or rejected",
        )

    def test_focus_stops_instead_of_looping_when_everything_is_decided(self) -> None:
        assert self.app.store is not None
        self.app.store.save_labels(dict.fromkeys(self.paths, 0))
        self.app.focused_path = self.paths[0]
        self.app._advance_focus_after(self.paths[0])
        self.assertEqual(
            self.app.focused_path,
            self.paths[0],
            "the queue wrapped back onto decided photos instead of stopping",
        )
        self.assertIn("decided", self.app.status_var.get().lower())

    def test_progress_note_counts_what_is_left(self) -> None:
        assert self.app.store is not None
        self.app.store.save_labels(dict.fromkeys(self.paths[:2], 1))
        note = self.app._progress_note()
        self.assertIn("2 decided", note)
        self.assertIn(f"{len(self.paths) - 2} left in folder", note)

    def test_undecided_remaining_ignores_labelled_photos(self) -> None:
        assert self.app.store is not None
        self.app.store.save_labels({self.paths[0]: 1, self.paths[1]: 0, self.paths[2]: 2})
        self.assertEqual(self.app._undecided_remaining(), len(self.paths) - 3)

    def test_the_grid_renders_one_page_at_a_time(self) -> None:
        # Synthetic paths: this exercises slicing, not thumbnail decoding.
        synthetic = [str(self.folder / f"page{index:03d}.jpg") for index in range(50)]
        self.app.page_size_var.set(20)
        self.app.sort_var.set("filename")
        self.app.current_samples = [
            {"path": path, "score": 0.9, "bucket": "Bikini"} for path in synthetic
        ]
        self.app._refresh_displayed_results()
        self.assertEqual(len(self.app.displayed_samples), 50, "paging must not drop results")
        self.assertEqual(len(self.app.page_samples), 20, "the grid rendered more than one page of cards")
        self.assertEqual(self.app._page_count(), 3)
        self.app.next_page()
        self.assertEqual(self.app.page_index, 1)
        self.assertEqual(
            [str(sample["path"]) for sample in self.app.page_samples],
            synthetic[20:40],
        )
        self.app.next_page()
        self.assertEqual(len(self.app.page_samples), 10, "the last page should hold the remainder")

    def test_a_decided_photo_leaves_the_grid_immediately(self) -> None:
        """The reported bug: Accept/REJECT re-rendered the same photos.

        The grid only rebuilt when the background retrain landed, and the retrain
        rebuilt the same list, so nothing on screen ever moved.
        """
        self.app.update_algorithm = lambda: None  # type: ignore[method-assign]
        self.app.focused_path = self.paths[0]
        self.app.set_label(self.paths[0], 1)
        shown = [str(sample["path"]) for sample in self.app.page_samples]
        self.assertNotIn(self.paths[0], shown, "the accepted photo is still in the grid")
        self.assertEqual(len(shown), len(self.paths) - 1)
        self.assertEqual(self.app.focused_path, self.paths[1], "the active picture did not move on")

    def test_the_detected_view_advances_too(self) -> None:
        self.app.update_algorithm = lambda: None  # type: ignore[method-assign]
        self.app.view_mode = "detected"
        self.app.current_samples = [
            {"path": path, "score": 0.9, "bucket": "Bikini"} for path in self.paths
        ]
        self.app._refresh_displayed_results()
        self.app.focused_path = self.paths[0]
        self.app.set_label(self.paths[0], 0)
        self.assertNotIn(
            self.paths[0],
            [str(sample["path"]) for sample in self.app.page_samples],
            "the rejected photo stayed in the detected-files grid",
        )

    def test_asking_for_the_labeled_ones_still_shows_them(self) -> None:
        assert self.app.store is not None
        self.app.store.save_labels({self.paths[0]: 1})
        self.app.label_filter_var.set("labeled")
        # Asked in the detected view, which lists everything found. The review queue is
        # assembled out of undecided photos, so a labelled one is never in it to show.
        self.app.view_mode = "detected"
        self.app.current_samples = [
            {"path": path, "score": 0.9, "bucket": "Bikini"} for path in self.paths
        ]
        self.app._refresh_displayed_results()
        self.assertEqual(
            [str(sample["path"]) for sample in self.app.page_samples],
            [self.paths[0]],
            "'Show: labeled' was overruled by the hide-decided rule",
        )

    def test_decided_photos_come_back_when_the_rule_is_switched_off(self) -> None:
        assert self.app.store is not None
        self.app.store.save_labels(dict.fromkeys(self.paths[:3], 1))
        # Asked of the detected view: the review queue is assembled out of undecided
        # photos in the first place, so there is nothing there for the rule to hide.
        self.app.view_mode = "detected"
        self.app.current_samples = [
            {"path": path, "score": 0.9, "bucket": "Bikini"} for path in self.paths
        ]
        self.app._refresh_displayed_results()
        self.assertEqual(len(self.app.page_samples), len(self.paths) - 3)
        self.app.hide_decided_var.set(False)
        self.assertEqual(
            len(self.app.page_samples),
            len(self.paths),
            "unticking 'Hide decided' did not bring the decided photos back",
        )

    def test_a_run_of_decisions_costs_one_retrain(self) -> None:
        """A retrain refits the model and rescores every image in the folder.

        Firing one per click meant a reviewer working at a photo a second queued work
        faster than the machine could clear it — measured at 1.5-1.8 s a click on
        5 000 images. The decisions are saved immediately either way; only the re-rank
        waits for a pause.
        """
        launched: list[bool] = []
        self.app._launch_background_scan = lambda full_rescan: launched.append(full_rescan)  # type: ignore[method-assign]
        self.app._ensure_scorer = lambda: True  # type: ignore[method-assign]
        for path in self.paths[:5]:
            self.app.focused_path = path
            self.app.set_label(path, 1)
        self.assertEqual(launched, [], "labelling started a retrain per click instead of batching them")
        self.assertTrue(self.app._retrain_pending, "the queued labels were forgotten rather than deferred")
        self.assertEqual(self.app._labels_since_retrain, 5)
        # Whatever fires it — the idle timer, the burst cap, or Tools > Update
        # rankings — one pass folds in everything that accumulated.
        self.app._flush_retrain()
        self.assertEqual(launched, [False], "the queued labels never reached the model")
        self.assertFalse(self.app._retrain_pending)
        self.assertEqual(self.app._labels_since_retrain, 0)

    def test_the_grid_keeps_the_cards_that_are_staying(self) -> None:
        """Removing one card must not rebuild the other nineteen.

        A card is ~17 Tk widgets; tearing down and rebuilding a whole page per
        decision profiled at over a second.
        """
        self.app._launch_background_scan = lambda full_rescan: None  # type: ignore[method-assign]
        self.app._ensure_scorer = lambda: True  # type: ignore[method-assign]
        self.app._refresh_displayed_results()
        decided = str(self.app.page_samples[0]["path"])
        survivor = str(self.app.page_samples[1]["path"])
        survivor_widget = self.app.cards[survivor].frame
        self.app.focused_path = decided
        self.app.set_label(decided, 1)
        self.assertNotIn(decided, self.app.cards, "the decided card was left in the grid")
        self.assertIs(
            self.app.cards[survivor].frame,
            survivor_widget,
            "the whole page was torn down and rebuilt to remove a single card",
        )

    def test_a_retrain_does_not_move_the_photo_you_are_about_to_judge(self) -> None:
        """A re-rank must not reorder the page under the reviewer.

        The model re-ranks everything, so without pinning, the photo being lined up
        for an Accept slides elsewhere the moment the retrain lands and the click
        goes to something else.
        """
        self.app._launch_background_scan = lambda full_rescan: None  # type: ignore[method-assign]
        self.app._refresh_displayed_results()
        before = [str(sample["path"]) for sample in self.app.page_samples]
        # The re-rank arrives and reverses the ranking outright.
        for index, sample in enumerate(self.app.current_samples):
            sample["score"] = float(index) / len(self.app.current_samples)
        self.app._refresh_displayed_results(reset_page=False)
        self.assertEqual(
            [str(sample["path"]) for sample in self.app.page_samples],
            before,
            "the re-rank reshuffled the page the reviewer was working through",
        )
        # Turning the page is the boundary where the new ranking is allowed in.
        self.app.page_size_var.set(3)
        self.app._page_order = []
        self.app._refresh_displayed_results(reset_page=True)
        self.app.next_page()
        self.app.previous_page()
        self.assertNotEqual(
            [str(sample["path"]) for sample in self.app.page_samples],
            before[:3],
            "the new ranking never took effect, even after a page turn",
        )

    def test_undo_says_what_it_would_undo(self) -> None:
        self.app._launch_background_scan = lambda full_rescan: None  # type: ignore[method-assign]
        self.assertEqual(self.app.undo_hint_var.get(), "", "there is nothing to undo yet")
        self.app.focused_path = self.paths[0]
        self.app.set_label(self.paths[0], 1)
        self.assertIn("Accept", self.app.undo_hint_var.get())
        self.assertIn(Path(self.paths[0]).name, self.app.undo_hint_var.get())
        # A bulk edit has to be distinguishable from a single misclick before you
        # press Ctrl+Z, which is the whole point.
        self.app._apply_label_batch(dict.fromkeys(self.paths, 0), status="bulk", retrain=False)
        self.assertIn(f"REJECT {len(self.paths)} photos", self.app.undo_hint_var.get())

    def test_focus_mode_decides_without_the_grid(self) -> None:
        self.app._launch_background_scan = lambda full_rescan: None  # type: ignore[method-assign]
        self.app._refresh_displayed_results()
        self.app.focused_path = str(self.app.page_samples[0]["path"])
        first = self.app.focused_path
        self.app.enter_focus_mode()
        self.root.update()
        try:
            self.assertIsNotNone(self.app._focus_window, "focus mode did not open")
            self.app._focus_decide(1)
            self.root.update()
            self.assertNotEqual(self.app.focused_path, first, "the queue did not advance")
            assert self.app.store is not None
            self.assertEqual(self.app.store.load_labels().get(first), 1, "the decision was not recorded")
        finally:
            self.app.exit_focus_mode()
            self.root.update()
        self.assertIsNone(self.app._focus_window, "focus mode did not close")

    def test_a_retrain_reports_what_the_labels_changed(self) -> None:
        state = self.app.current_state
        assert state is not None
        previous = dict.fromkeys(self.paths, 0.1)
        state.classifier_label_count = 8
        state.learning_summary = "8 labels, influence 12%"
        report = self.app._retrain_report(state, previous, threshold=0.5)
        self.assertIn("8 labels", report)
        self.assertIn(f"{len(self.paths)} photos changed side of the threshold", report)

    def test_a_retrain_with_no_labels_says_so(self) -> None:
        state = self.app.current_state
        assert state is not None
        state.classifier_label_count = 0
        report = self.app._retrain_report(state, {}, threshold=0.5)
        self.assertIn("zero-shot", report)


class WorkflowTools(unittest.TestCase):
    """The queue, notes, browse mode and bulk scopes, driven the way a user drives them."""

    def setUp(self) -> None:
        tkinter = __import__("tkinter")
        try:
            self.root = tkinter.Tk()
        except Exception as exc:  # noqa: BLE001
            self.skipTest(f"no display available: {exc}")
        from bikini_scanner import gui as gui_module

        for name in ("_resume_last_folder_if_any", "_maybe_preload_backend", "_maybe_show_first_run_guide"):
            setattr(gui_module.BikiniScannerApp, name, lambda self: None)
        self.folder = Path(tempfile.mkdtemp(prefix="bikini_tools_"))
        _make_images(self.folder, count=5)
        self.app = gui_module.BikiniScannerApp(self.root, config=ScannerConfig(preload_backend=False))
        self.app.label_filter_var.set("all")
        self.app.hide_decided_var.set(True)
        self.app._launch_background_scan = lambda full_rescan: None  # type: ignore[method-assign]

    def tearDown(self) -> None:
        self.app._closing = True
        self.root.destroy()
        shutil.rmtree(self.folder, ignore_errors=True)

    def test_a_folder_can_be_reviewed_without_scanning_it(self) -> None:
        """Judging photos should not require a model download and an embedding pass."""
        self.app.browse_without_scanning(str(self.folder))
        self.root.update()
        self.assertEqual(self.app.view_mode, "browse")
        self.assertTrue(self.app.page_samples, "browsing showed nothing")
        target = str(self.app.page_samples[0]["path"])
        self.app.focused_path = target
        self.app.set_label(target, 1)
        assert self.app.store is not None
        # The decision lands in the same file a scan reads, so a later scan starts
        # already knowing it.
        self.assertEqual(self.app.store.load_labels().get(target), 1)
        self.assertNotIn(target, [str(sample["path"]) for sample in self.app.page_samples])

    def test_the_scan_queue_can_be_reordered_and_pruned(self) -> None:
        self.app.scan_queue = ["/one", "/two", "/three"]
        self.app._reload_queue_listbox()
        self.app.queue_listbox.selection_set(2)
        self.app.move_queue_item(-1)
        self.assertEqual(self.app.scan_queue, ["/one", "/three", "/two"])
        self.app.queue_listbox.selection_clear(0, "end")
        self.app.queue_listbox.selection_set(0)
        self.app.remove_queue_item()
        self.assertEqual(self.app.scan_queue, ["/three", "/two"])

    def test_a_note_survives_and_reaches_the_places_that_show_it(self) -> None:
        self.app.browse_without_scanning(str(self.folder))
        self.root.update()
        assert self.app.store is not None
        target = str(self.app.page_samples[0]["path"])
        self.app.store.save_notes({target: "second opinion needed"})
        self.assertEqual(self.app.note_for(target), "second opinion needed")
        self.assertIn("second opinion needed", self.app._card_label_text(target))
        self.assertIn("second opinion needed", self.app._preview_caption_text(target))
        # Blank clears rather than storing an empty string.
        self.app.store.save_notes({target: "   "})
        self.assertEqual(self.app.note_for(target), "")

    def test_bulk_actions_say_what_they_will_touch(self) -> None:
        self.app.browse_without_scanning(str(self.folder))
        self.root.update()
        paths, scope = self.app._output_scope()
        self.assertEqual(len(paths), len(self.app.displayed_samples))
        self.assertIn("image", scope)
        # "Visible" now depends on hide-decided as well as the filters and paging, so
        # the confirmation has to name that rather than leave it implied.
        self.assertIn("already decided", scope)

    def test_find_similar_can_be_backed_out_of(self) -> None:
        self.app.browse_without_scanning(str(self.folder))
        self.root.update()
        before = [str(sample["path"]) for sample in self.app.page_samples]
        self.app._view_return = (
            self.app.view_mode,
            list(self.app.current_samples),
            self.app.page_index,
            self.app.focused_path,
        )
        self.app.view_mode = "similar"
        self.app.current_samples = [{"path": before[0], "score": 1.0, "bucket": "Similar"}]
        self.app._refresh_displayed_results()
        self.assertEqual(len(self.app.page_samples), 1)
        self.app.go_back()
        self.assertEqual(self.app.view_mode, "browse")
        self.assertEqual([str(sample["path"]) for sample in self.app.page_samples], before)


class CacheAndDecisions(unittest.TestCase):
    """Clearing a cache must not take the work that cannot be recomputed."""

    def setUp(self) -> None:
        self.folder = Path(tempfile.mkdtemp(prefix="bikini_clear_"))
        _make_images(self.folder, count=3)
        self.store = FolderStore(self.folder)
        self.paths = [str(path) for path in collect_image_paths(self.folder)]
        self.store.save_labels({self.paths[0]: 1, self.paths[1]: 0})
        self.store.save_notes({self.paths[0]: "keep this one"})
        self.store.config_override_path.write_text(json.dumps({"threshold": 0.5}), encoding="utf-8")

    def tearDown(self) -> None:
        shutil.rmtree(self.folder, ignore_errors=True)

    def test_clearing_the_cache_keeps_decisions_notes_and_the_override(self) -> None:
        """The old behaviour rmtree'd the lot, including hours of hand-made decisions."""
        self.store.clear_cache()
        self.assertEqual(self.store.load_labels(), {self.paths[0]: 1, self.paths[1]: 0})
        self.assertEqual(self.store.load_notes(), {self.paths[0]: "keep this one"})
        self.assertTrue(self.store.config_override_path.exists(), "the folder's saved settings were destroyed")

    def test_clearing_everything_is_still_possible_when_asked_for(self) -> None:
        self.store.clear_cache(keep_decisions=False)
        self.assertEqual(self.store.load_labels(), {})
        self.assertEqual(self.store.load_notes(), {})

    def test_deleting_decisions_leaves_the_cache_alone(self) -> None:
        marker = self.store.cache_dir / "cache.db"
        self.store.delete_decisions()
        self.assertEqual(self.store.load_labels(), {})
        self.assertEqual(self.store.load_notes(), {})
        self.assertTrue(marker.exists(), "deleting decisions should not touch cached scan data")

    def test_cached_path_count_reports_what_a_scan_would_reuse(self) -> None:
        paths = collect_image_paths(self.folder)
        self.assertEqual(self.store.cached_path_count(paths), 0)


class ResolvedPathMemo(unittest.TestCase):
    """The path memo has to agree with Path.resolve, or the cache keys drift."""

    def setUp(self) -> None:
        self.folder = Path(tempfile.mkdtemp(prefix="bikini_resolve_"))
        _make_images(self.folder, count=2)

    def tearDown(self) -> None:
        shutil.rmtree(self.folder, ignore_errors=True)

    def test_it_matches_path_resolve(self) -> None:
        for path in collect_image_paths(self.folder):
            self.assertEqual(safe_io.resolved_str(path), str(path.resolve()))

    def test_it_agrees_across_equivalent_spellings(self) -> None:
        """A cache keyed on this must not split one file into two entries."""
        path = next(iter(collect_image_paths(self.folder)))
        roundabout = Path(str(path.parent)) / "." / path.name
        self.assertEqual(safe_io.resolved_str(roundabout), safe_io.resolved_str(path))

    def test_a_missing_file_still_resolves(self) -> None:
        # Scan metadata records skipped and vanished files, so this must not raise.
        missing = self.folder / "gone.jpg"
        self.assertEqual(safe_io.resolved_str(missing), str(missing.resolve()))


class GridLayout(unittest.TestCase):
    """Column fitting and the grid's share of the window."""

    def setUp(self) -> None:
        tkinter = __import__("tkinter")
        try:
            self.root = tkinter.Tk()
        except Exception as exc:  # noqa: BLE001
            self.skipTest(f"no display available: {exc}")
        from bikini_scanner import gui as gui_module

        for name in ("_resume_last_folder_if_any", "_maybe_preload_backend", "_maybe_show_first_run_guide"):
            setattr(gui_module.BikiniScannerApp, name, lambda self: None)
        self.gui_module = gui_module
        self.app = gui_module.BikiniScannerApp(self.root, config=ScannerConfig(preload_backend=False))

    def tearDown(self) -> None:
        self.app._closing = True
        self.root.destroy()

    def test_a_wider_window_fits_more_cards_not_bigger_ones(self) -> None:
        """Auto columns; widening used to make two cards larger instead of adding a third."""
        self.app.columns_var.set(0)
        self.app.thumbnail_size_var.set(240)
        self.root.geometry("1500x950")
        self.root.update()
        wide = self.app._grid_columns()
        self.root.geometry("900x700")
        self.root.update()
        narrow = self.app._grid_columns()
        self.assertGreater(wide, narrow, "the column count did not follow the window width")
        self.assertGreaterEqual(wide, 2)

    def test_an_explicit_column_count_is_still_obeyed(self) -> None:
        self.app.columns_var.set(2)
        self.root.geometry("1900x1000")
        self.root.update()
        self.assertEqual(self.app._grid_columns(), 2, "a pinned column count was overridden by auto-fit")

    def test_the_grid_keeps_room_for_a_whole_card(self) -> None:
        """The floor has to cover the thumbnail *and* its action row.

        With a flat floor the Accept/REJECT buttons sat below the fold at 950x700.
        """
        for thumb in (160, 240, 400):
            self.app.thumbnail_size_var.set(thumb)
            self.assertGreater(
                self.app._min_grid_height(),
                thumb,
                "the grid floor must leave room for the card chrome, not just the image",
            )

    def test_the_preview_never_outgrows_its_pane(self) -> None:
        # Rendering taller than the pane clipped the bottom off the photo.
        self.root.geometry("950x700")
        self.root.update()
        _, height = self.app._preview_size()
        self.assertLessEqual(height, 700 - self.app._min_grid_height())


class ThumbnailFraming(unittest.TestCase):
    """Grid thumbnails fill their slot; the preview still shows the whole frame."""

    def test_a_portrait_photo_fills_a_square_slot(self) -> None:
        from bikini_scanner import gui as gui_module

        source = Image.new("RGB", (400, 1200), (10, 20, 30))
        filled = gui_module.BikiniScannerApp._thumbnail_fill(source, 240, 240)
        self.assertEqual(filled.size, (240, 240), "the thumbnail did not fill its slot")

    def test_a_landscape_photo_fills_a_square_slot(self) -> None:
        from bikini_scanner import gui as gui_module

        source = Image.new("RGB", (1600, 500), (10, 20, 30))
        self.assertEqual(gui_module.BikiniScannerApp._thumbnail_fill(source, 240, 240).size, (240, 240))

    def test_the_preview_still_letterboxes(self) -> None:
        """Cropping is right for a contact sheet and wrong for judging a photo."""
        from bikini_scanner import gui as gui_module

        source = Image.new("RGB", (400, 1200), (10, 20, 30))
        boxed = gui_module.BikiniScannerApp._preview_letterbox(source, 800, 300)
        self.assertEqual(boxed.height, 300)
        self.assertLess(boxed.width, 800, "the preview must not crop to fill")


class ProgressWeighting(unittest.TestCase):
    """The bar has to reflect the work this scan actually has to do."""

    def test_a_cached_rescan_does_not_hand_the_bar_to_the_embed_phase(self) -> None:
        cold = scorer_module.phase_shares(uncached=1000, total=1000, runs_detail_pass=True)
        warm = scorer_module.phase_shares(uncached=0, total=1000, runs_detail_pass=True)
        self.assertAlmostEqual(sum(cold.values()), 1.0, places=6)
        self.assertAlmostEqual(sum(warm.values()), 1.0, places=6)
        self.assertGreater(cold[scorer_module.PHASE_EMBED], 0.5, "a cold scan is mostly embedding")
        self.assertLess(
            warm[scorer_module.PHASE_EMBED],
            0.05,
            "a fully cached rescan gave the embed phase 65% of the bar and then crawled",
        )
        self.assertGreater(warm[scorer_module.PHASE_DETAIL], cold[scorer_module.PHASE_DETAIL])

    def test_no_detail_pass_still_reaches_one(self) -> None:
        shares = scorer_module.phase_shares(uncached=10, total=10, runs_detail_pass=False)
        self.assertAlmostEqual(sum(shares.values()), 1.0, places=6)
        self.assertEqual(shares[scorer_module.PHASE_DETAIL], 0.0)


class ShortcutGuard(unittest.TestCase):
    """A single letter must not label a photo from the wrong widget."""

    def setUp(self) -> None:
        tkinter = __import__("tkinter")
        try:
            self.root = tkinter.Tk()
        except Exception as exc:  # noqa: BLE001
            self.skipTest(f"no display available: {exc}")
        from bikini_scanner import gui as gui_module

        for name in ("_resume_last_folder_if_any", "_maybe_preload_backend", "_maybe_show_first_run_guide"):
            setattr(gui_module.BikiniScannerApp, name, lambda self: None)
        self.app = gui_module.BikiniScannerApp(self.root, config=ScannerConfig(preload_backend=False))

    def tearDown(self) -> None:
        self.app._closing = True
        self.root.destroy()

    def test_the_queue_listbox_swallows_its_own_keystrokes(self) -> None:
        """Selecting a queue folder and pressing 'd' used to reject the active photo."""
        # The queue panel is collapsed by default and an unmapped widget cannot take
        # focus, so open it first. focus_force because a suite run has no window
        # manager focus, which would leave focus_get() returning None.
        self.app._toggle_panel("queue")
        self.root.update()
        self.app.queue_listbox.focus_force()
        self.root.update()
        if self.root.focus_get() is not self.app.queue_listbox:
            self.skipTest("the window manager did not grant focus to the test window")
        self.assertTrue(
            self.app._focus_is_text_input(),
            "a Listbox with focus must not let single-letter label shortcuts through",
        )

    def test_the_grid_still_receives_shortcuts(self) -> None:
        self.app.grid_canvas.focus_force()
        self.root.update()
        if self.root.focus_get() is not self.app.grid_canvas:
            self.skipTest("the window manager did not grant focus to the test window")
        self.assertFalse(self.app._focus_is_text_input(), "the results grid must still take shortcuts")


class ProfileContent(unittest.TestCase):
    """A profile should capture a way of working, not one number."""

    def test_built_ins_cover_whole_setups(self) -> None:
        self.assertGreaterEqual(len(BUILTIN_PROFILES), 5)
        thorough = profile_config("Thorough")
        triage = profile_config("Fast triage")
        assert thorough is not None and triage is not None
        self.assertEqual(thorough.deep_scan, "always")
        self.assertTrue(thorough.refine_model)
        self.assertEqual(triage.deep_scan, "off")
        self.assertFalse(triage.refine_model)

    def test_a_saved_profile_round_trips_every_setting(self) -> None:
        config = ScannerConfig()
        config.threshold = 0.42
        config.deep_scan = "always"
        config.positive_prompts = ["a distinctive prompt"]
        config_profiles.save_profile("Round trip", config)
        try:
            restored = profile_config("Round trip")
            assert restored is not None
            self.assertAlmostEqual(restored.threshold, 0.42)
            self.assertEqual(restored.deep_scan, "always")
            self.assertEqual(restored.positive_prompts, ["a distinctive prompt"])
        finally:
            config_profiles.delete_profile("Round trip")


class HeadlessLabelExchange(unittest.TestCase):
    """Batch runs and interactive review have to be able to feed each other."""

    def setUp(self) -> None:
        self.folder = Path(tempfile.mkdtemp(prefix="bikini_exchange_"))
        _make_images(self.folder, count=3)
        self.store = FolderStore(self.folder)
        self.paths = [str(path) for path in collect_image_paths(self.folder)]

    def tearDown(self) -> None:
        shutil.rmtree(self.folder, ignore_errors=True)

    def test_labels_import_by_filename_and_from_another_machine(self) -> None:
        source = self.folder / "labels.json"
        source.write_text(
            json.dumps(
                {
                    Path(self.paths[0]).name: 1,
                    "/somewhere/else/" + Path(self.paths[1]).name: 0,
                    "not_in_this_folder.jpg": 1,
                }
            ),
            encoding="utf-8",
        )
        merged = run_module._merge_labels(self.store, source)
        self.assertEqual(merged, 2, "a label for a file that is not here should be skipped")
        labels = self.store.load_labels()
        self.assertEqual(labels.get(self.paths[0]), 1)
        self.assertEqual(labels.get(self.paths[1]), 0)

    def test_out_of_range_labels_are_refused(self) -> None:
        source = self.folder / "labels.json"
        source.write_text(json.dumps({Path(self.paths[0]).name: 9, Path(self.paths[1]).name: "x"}), encoding="utf-8")
        self.assertEqual(run_module._merge_labels(self.store, source), 0)
        self.assertEqual(self.store.load_labels(), {})

    def test_a_non_object_payload_is_rejected(self) -> None:
        source = self.folder / "labels.json"
        source.write_text(json.dumps([1, 2, 3]), encoding="utf-8")
        with self.assertRaises(ValueError):
            run_module._merge_labels(self.store, source)


class LogisticConvergence(unittest.TestCase):
    """The optimiser has to recognise when it has finished."""

    @staticmethod
    def _separable(rows: int, columns: int, seed: int = 3):
        rng = np.random.default_rng(seed)
        features = rng.normal(size=(rows, columns)).astype(np.float32)
        labels = (rng.random(rows) < 0.5).astype(np.int64)
        features[labels == 1] += 0.6
        return features, labels

    def test_an_ordinary_fit_stops_well_before_the_cap(self) -> None:
        features, labels = self._separable(80, 40)
        model = linear_model.LogisticRegression().fit(features, labels)
        self.assertTrue(model.converged, "a well-posed fit still reported non-convergence")
        self.assertLess(model.n_iter, model.max_iter, "the fit burned every iteration it was allowed")

    def test_a_warm_start_reaches_the_same_answer_sooner(self) -> None:
        features, labels = self._separable(80, 40)
        cold = linear_model.LogisticRegression().fit(features, labels)
        warm = linear_model.LogisticRegression().fit(
            features, labels, init_coef=cold.coef, init_intercept=cold.intercept
        )
        self.assertLess(warm.n_iter, cold.n_iter, "warm starting did not save any work")
        # Same optimum, so the ranking a reviewer sees is unchanged.
        np.testing.assert_allclose(
            cold.decision_function(features), warm.decision_function(features), atol=2e-2
        )

    def test_the_regularisation_sweep_can_be_skipped(self) -> None:
        features, labels = self._separable(60, 20)
        swept = learning.fit(features, labels)
        self.assertIsNotNone(swept.chosen_c)
        reused = learning.fit(features, labels, reuse_c=swept.chosen_c)
        self.assertEqual(reused.chosen_c, swept.chosen_c, "reuse_c did not pin the value it was given")


class QueueSelection(unittest.TestCase):
    """Building the review queue must stay exact while being fast enough to run."""

    def test_diverse_selection_handles_ties_and_empty_vectors(self) -> None:
        rng = np.random.default_rng(11)
        count = 120
        paths = [f"p{index:03d}.jpg" for index in range(count)]
        scores = rng.random(count)
        scores[:40] = 0.5  # exact ties on the secondary key
        embeddings = rng.normal(size=(count, 32)).astype(np.float32)
        embeddings[:10] = embeddings[0]  # duplicates
        embeddings[10:13] = 0.0  # zero-length vectors
        samples = bucketed_sampling(
            paths, scores, [], embeddings=embeddings, threshold=0.35, disagreement=rng.random(count) * 0.3
        )
        chosen = [str(sample["path"]) for sample in samples]
        self.assertTrue(samples, "the queue came back empty")
        self.assertEqual(len(chosen), len(set(chosen)), "the same photo was queued twice")

    def test_the_queue_is_built_without_scanning_every_pair(self) -> None:
        """Guards the vectorised selection: this shape took over a second per retrain."""
        rng = np.random.default_rng(5)
        count = 4000
        paths = [f"p{index:05d}.jpg" for index in range(count)]
        scores = rng.random(count)
        embeddings = rng.normal(size=(count, 256)).astype(np.float32)
        started = time.perf_counter()
        bucketed_sampling(
            paths, scores, [], embeddings=embeddings, threshold=0.35, disagreement=rng.random(count) * 0.3
        )
        elapsed = time.perf_counter() - started
        self.assertLess(elapsed, 1.0, f"building the review queue took {elapsed:.2f}s for {count} images")


class LearningReadout(unittest.TestCase):
    """The label counts on screen have to reconcile with the reviewer's own tally."""

    def test_pooled_labels_are_named_as_pooled(self) -> None:
        outcome = learning.LearningOutcome(
            prototype=object(),
            label_count=128,
            positive_count=14,
            negative_count=114,
            local_count=14,
            local_positive=7,
            local_negative=7,
            weight=0.82,
        )
        summary = outcome.summary()
        # Without the split this read "128 labels (14 accepted / 114 rejected)" beside a
        # stats panel saying "Accepted 7 | Rejected 7" - two answers to one question.
        self.assertIn("14 labels here", summary)
        self.assertIn("7 accepted / 7 rejected", summary)
        self.assertIn("114 pooled from other folders", summary)

    def test_folder_only_labels_are_not_described_as_pooled(self) -> None:
        outcome = learning.LearningOutcome(
            prototype=object(),
            label_count=14,
            positive_count=7,
            negative_count=7,
            local_count=14,
            local_positive=7,
            local_negative=7,
            weight=0.12,
        )
        summary = outcome.summary()
        self.assertIn("14 labels (7 accepted / 7 rejected)", summary)
        self.assertNotIn("pooled", summary)

    def test_fit_defaults_to_treating_every_label_as_local(self) -> None:
        # A caller that does not pool must not read as "0 labels here + N pooled".
        rng = np.random.default_rng(2)
        features = np.vstack(
            [rng.normal(1.0, 0.2, (8, 6)), rng.normal(-1.0, 0.2, (8, 6))]
        ).astype(np.float32)
        labels = np.array([1] * 8 + [0] * 8, dtype=np.int64)
        outcome = learning.fit(features, labels)
        self.assertEqual(outcome.local_count, outcome.label_count)
        self.assertNotIn("pooled", outcome.summary())

    def test_an_untrusted_model_says_it_is_not_steering(self) -> None:
        outcome = learning.LearningOutcome(prototype=object(), label_count=4, weight=0.0)
        self.assertIn("not yet steering", outcome.summary())

    def test_the_scorer_reports_the_local_share_of_a_pooled_fit(self) -> None:
        shared = _shared()
        folder = Path(tempfile.mkdtemp(prefix="bikini_pooled_"))
        try:
            paths = [str(path) for path in _make_images(folder, count=4)]
            scorer = scorer_module.BikiniScorer(shared["backend"], ScannerConfig())
            features = np.random.default_rng(5).random((len(paths), 8)).astype(np.float32)
            labels = {paths[0]: 1, paths[1]: 1, paths[2]: 0, paths[3]: 0}
            outcome = scorer.learn(paths, features, labels)
            self.assertEqual(outcome.local_count, 4)
            self.assertEqual(outcome.local_positive, 2)
            self.assertEqual(outcome.local_negative, 2)
        finally:
            shutil.rmtree(folder, ignore_errors=True)


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
