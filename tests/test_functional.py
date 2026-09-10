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

import ast
import contextlib
import io
import json
import logging
import os
import pickle
import shutil
import sys
import tempfile
import threading
import time
import traceback
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
    autodecide,
    cascade,
    config_profiles,
    duplicates,
    image_formats,
    learning,
    linear_model,
    output_ops,
    regions,
    safe_io,
    vision_analysis,
)
from bikini_scanner import global_store as global_store_module
from bikini_scanner import gui as gui_module
from bikini_scanner import run as run_module
from bikini_scanner import scorer as scorer_module
from bikini_scanner import store as store_module
from bikini_scanner.config import ScannerConfig, filter_folder_override


class _NonBlockingDialogs:
    """Stands in for tkinter.messagebox and answers instead of waiting.

    A real message box blocks until somebody presses OK, and in a test nobody ever
    does. One unstubbed dialog therefore does not fail the run, it stops it: a single
    test that drove a full scan took this suite from 61 s to 1 490 s, and its own
    runtime swung between 21 s and 877 s depending on what else happened to be on
    screen. Stubbing each call site by hand only ever fixed the ones already found.

    So every dialog the app can raise is answered here for the whole suite. Questions
    answer `answer` (False by default, i.e. decline, which is the safe reply to
    "delete these?"), and a test that needs otherwise uses `answering(True)`. Every
    call is recorded, so a test can also assert on what it was asked.
    """

    def __init__(self) -> None:
        self.calls: list[tuple[str, str, str]] = []
        self.answer = False
        for name in ("showinfo", "showwarning", "showerror"):
            setattr(self, name, self._telling(name))
        for name in ("askyesno", "askokcancel", "askretrycancel"):
            setattr(self, name, self._asking(name))

    def _telling(self, kind: str):
        def call(title: str = "", message: str = "", **_kwargs: object) -> None:
            self.calls.append((kind, str(title), str(message)))
            return None

        return call

    def _asking(self, kind: str):
        def call(title: str = "", message: str = "", **_kwargs: object) -> bool:
            self.calls.append((kind, str(title), str(message)))
            return bool(self.answer)

        return call

    @contextlib.contextmanager
    def answering(self, answer: bool):
        """Answer every question with `answer` for the duration."""
        previous, self.answer = self.answer, bool(answer)
        try:
            yield self
        finally:
            self.answer = previous

    @contextlib.contextmanager
    def recording(self):
        """Watch only the dialogs raised inside the block."""
        start = len(self.calls)
        try:
            yield self
        finally:
            self.seen = self.calls[start:]

    def titles(self) -> list[str]:
        return [title for _kind, title, _message in self.calls]


# Installed once, for every test in the file. gui.py does `from tkinter import
# messagebox`, so this replaces the name it actually calls through.
DIALOGS = _NonBlockingDialogs()
gui_module.messagebox = DIALOGS  # type: ignore[assignment]


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

    # --- per-subject gating ------------------------------------------------
    # A photo of an adult in swimwear standing beside her children used to be excluded
    # outright: the age axes read the whole frame, the frame contained children, and the
    # score was forced to zero at any threshold. These cover the subject-attributed gate
    # that replaced that behaviour when face-anchored crops are available.

    def _subject_table(self, people: list[tuple[float, float, float]]) -> cascade.RegionScoreTable:
        """One image, one face+chest row pair per (child, adult, detail) person."""
        owner: list[int] = [0]
        kinds: list[str] = ["full"]
        subject: list[int] = [-1]
        child: list[float] = [0.9]  # the whole frame reads "child", as a group photo does
        adult: list[float] = [0.1]
        detail: list[float] = [0.5]
        for index, (person_child, person_adult, person_detail) in enumerate(people):
            owner += [0, 0]
            kinds += ["face", "chest"]
            subject += [index, index]
            child += [person_child, 0.5]
            adult += [person_adult, 0.5]
            detail += [0.5, person_detail]
        axis = {name: np.asarray(detail, dtype=np.float32) for name in cascade.DETAIL_AXES}
        axis["child"] = np.asarray(child, dtype=np.float32)
        axis["adult"] = np.asarray(adult, dtype=np.float32)
        axis["person"] = np.full(len(owner), 0.9, dtype=np.float32)
        axis["female"] = np.full(len(owner), 0.9, dtype=np.float32)
        axis["nsfw"] = np.full(len(owner), 0.5, dtype=np.float32)
        return cascade.RegionScoreTable(
            owner=np.asarray(owner, dtype=np.int64),
            kinds=np.array(kinds, dtype=object),
            axis_scores=axis,
            image_count=1,
            full_row=np.zeros((1,), dtype=np.int64),
            subject=np.asarray(subject, dtype=np.int64),
        )

    def test_adult_beside_children_is_scored_not_excluded(self) -> None:
        """The regression this whole mechanism exists for."""
        table = self._subject_table([(0.05, 0.95, 0.99), (0.99, 0.05, 0.6), (0.95, 0.05, 0.6)])
        result = cascade.evaluate(table, self.config)
        self.assertFalse(result.excluded.any())
        self.assertGreater(float(result.score[0]), 0.0)

    def test_only_the_adults_crops_contribute_to_the_score(self) -> None:
        """A suppressed minor's body must not raise the score, even when it scores high."""
        weak_adult = self._subject_table([(0.05, 0.95, 0.30), (0.99, 0.05, 0.99)])
        strong_adult = self._subject_table([(0.05, 0.95, 0.99), (0.99, 0.05, 0.99)])
        self.assertLess(
            float(cascade.evaluate(weak_adult, self.config).score[0]),
            float(cascade.evaluate(strong_adult, self.config).score[0]),
        )

    def test_every_subject_a_minor_is_still_excluded(self) -> None:
        table = self._subject_table([(0.99, 0.05, 0.99), (0.95, 0.05, 0.9)])
        result = cascade.evaluate(table, self.config)
        self.assertTrue(result.excluded.all())
        self.assertEqual(result.stage[0], cascade.STAGE_MINOR)
        np.testing.assert_allclose(result.score, 0.0)

    def test_a_minors_crops_cannot_win_the_detail_slot(self) -> None:
        table = self._subject_table([(0.05, 0.95, 0.4), (0.99, 0.05, 0.99)])
        subjects = cascade.analyse_subjects(table, self.config)
        assert subjects is not None
        # Rows 3 and 4 are the minor's face and chest.
        self.assertTrue(subjects.row_minor[3] and subjects.row_minor[4])
        self.assertFalse(subjects.row_minor[1] or subjects.row_minor[2])

    def test_images_without_faces_keep_the_whole_frame_gate(self) -> None:
        """No attribution means no change in behaviour: the old rules still apply."""
        self.assertIsNone(cascade.analyse_subjects(self._table(child=0.99, adult=0.5, detail=0.9), self.config))
        result = cascade.evaluate(self._table(child=0.99, adult=0.5, detail=0.99), self.config)
        self.assertTrue(result.excluded.all())

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

    def test_a_retained_label_keeps_teaching_after_its_file_is_deleted(self) -> None:
        """"Reject and delete" would otherwise undo its own training.

        A label whose file is gone normally stops teaching, so that scanning a
        temporary folder does not train the model forever. A photo deleted *because*
        it was rejected is the opposite case: dropping it on the next pass would
        quietly reverse the very training the action is named for.
        """
        store = GlobalLearningStore(model_name="retain-test")
        store.clear()
        doomed = Path(tempfile.mkdtemp(prefix="bikini_retain_"))
        self.addCleanup(shutil.rmtree, doomed, True)
        kept, dropped = doomed / "kept.jpg", doomed / "dropped.jpg"
        for path in (kept, dropped):
            path.write_bytes(b"x")
        store.record([(str(kept), 0, np.ones(4, dtype=np.float32))], sequence=1)
        store.record([(str(dropped), 0, np.zeros(4, dtype=np.float32))], sequence=2)
        # Marked before the files go, which is the order the delete flow uses.
        store.retain([str(kept)])
        kept.unlink()
        dropped.unlink()
        training = store.training_set(expected_dim=4)
        self.assertEqual([str(path) for path in training.paths], [str(kept)])
        self.assertEqual(store.stats()["total"], 1, "the retained label was pruned away")
        # Clearing the label retires it for good, retained or not.
        store.forget([str(kept)])
        self.assertEqual(len(store.training_set(expected_dim=4)), 0)
        store.clear()

    def test_evicting_an_entry_drops_its_retained_mark_too(self) -> None:
        """Otherwise the list grows by a key per deleted photo, for good.

        `record` prunes the oldest rows once the store passes MAX_ENTRIES. A retained
        key whose row has gone can never be consulted again, so leaving it behind is a
        leak in a file that is rewritten in full on every retain.
        """
        store = GlobalLearningStore(model_name="retain-evict-test")
        store.clear()
        folder = Path(tempfile.mkdtemp(prefix="bikini_evict_"))
        self.addCleanup(shutil.rmtree, folder, True)
        old = folder / "old.jpg"
        old.write_bytes(b"x")
        store.record([(str(old), 0, np.ones(4, dtype=np.float32))], sequence=1)
        store.retain([str(old)])
        self.assertIn(str(old), [str(path) for path in store.training_set(expected_dim=4).paths])
        # Push it out with newer rows, more than the store will keep.
        original_max = global_store_module.MAX_ENTRIES
        global_store_module.MAX_ENTRIES = 2
        try:
            for index in range(3):
                fresh = folder / f"new{index}.jpg"
                fresh.write_bytes(b"x")
                store.record([(str(fresh), 1, np.ones(4, dtype=np.float32))], sequence=10 + index)
        finally:
            global_store_module.MAX_ENTRIES = original_max
        retained = json.loads(store.retained_path.read_text(encoding="utf-8"))
        self.assertEqual(retained, [], "the evicted row kept its retained mark")
        store.clear()

    def test_retaining_is_visible_to_a_training_set_already_cached(self) -> None:
        store = GlobalLearningStore(model_name="retain-cache-test")
        store.clear()
        folder = Path(tempfile.mkdtemp(prefix="bikini_retain_cache_"))
        self.addCleanup(shutil.rmtree, folder, True)
        target = folder / "a.jpg"
        target.write_bytes(b"x")
        store.record([(str(target), 1, np.ones(4, dtype=np.float32))], sequence=1)
        self.assertEqual(len(store.training_set(expected_dim=4)), 1)
        store.retain([str(target)])
        target.unlink()
        # Without the retained list in the cache key this returned the stale answer.
        self.assertEqual(len(store.training_set(expected_dim=4)), 1)
        store.clear()


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

    def _decide_one_of_each(self) -> None:
        """One true positive, false positive, false negative, true negative and skip."""
        assert self.app.store is not None
        state = self.app.current_state
        assert state is not None
        self.app.threshold_var.set(0.5)
        state.scores[2] = 0.1
        state.scores[3] = 0.1
        self.app.store.save_labels(
            {self.paths[0]: 1, self.paths[1]: 0, self.paths[2]: 1, self.paths[3]: 0, self.paths[4]: 2}
        )

    def test_the_decisions_view_lists_every_decision_by_what_it_says_about_the_scanner(self) -> None:
        """A decided photo the scanner missed was shown in no view at all.

        The review queue is built from undecided photos and the detected list from
        what scored above the threshold, so a false negative had nowhere to appear
        and "what did it get wrong?" could not be answered from the screen.
        """
        self._decide_one_of_each()
        self.app.hide_decided_var.set(True)
        self.app.show_decisions()
        self.assertEqual(self.app.view_mode, "decided")
        buckets = {str(sample["path"]): str(sample["bucket"]) for sample in self.app.page_samples}
        self.assertEqual(
            buckets,
            {
                self.paths[0]: "True positives",
                self.paths[1]: "False positives",
                self.paths[2]: "False negatives",
                self.paths[3]: "True negatives",
                self.paths[4]: "Skipped",
            },
            "decided photos were not grouped by outcome, or 'Hide decided' hid them",
        )
        # The scanner's mistakes come first; the undecided photo is not a decision.
        order = [str(sample["bucket"]) for sample in self.app.page_samples]
        self.assertEqual(order[:2], ["False positives", "False negatives"])
        self.assertNotIn(self.paths[5], buckets)

    def test_a_card_says_whether_it_is_detected_and_whether_the_scanner_was_right(self) -> None:
        self._decide_one_of_each()
        self.app.show_decisions()
        false_positive = self.app.cards[self.paths[1]]
        self.assertTrue(str(false_positive.score_label.cget("text")).startswith("DETECTED"))
        self.assertIn("FALSE POSITIVE", str(false_positive.verdict_label.cget("text")))
        self.assertEqual(str(false_positive.verdict_label.cget("style")), "FalsePositive.TLabel")
        false_negative = self.app.cards[self.paths[2]]
        self.assertTrue(str(false_negative.score_label.cget("text")).startswith("not detected"))
        self.assertIn("FALSE NEGATIVE", str(false_negative.verdict_label.cget("text")))
        self.assertIn("True positive", str(self.app.cards[self.paths[0]].verdict_label.cget("text")))
        # The undecided photo, back in the review queue, carries no verdict at all.
        self.app.review_samples = [
            {"path": path, "score": 0.9, "bucket": "Likely match"} for path in self.paths
        ]
        self.app.restore_review_view()
        undecided = self.app.cards[self.paths[5]]
        self.assertEqual(undecided.verdict_label.grid_info(), {}, "an undecided card showed a verdict line")
        self.assertIn("DETECTED", str(undecided.score_label.cget("text")))

    def test_the_card_names_the_verdict_and_leaves_the_explaining_to_the_preview(self) -> None:
        """The card said one fact three times over.

        "DETECTED", then "REJECTED", then "detected, but you rejected it" — the third
        line restated the two above it on every card in the grid, and it was the line
        that wrapped. The term is named on the card and glossed in the preview, which
        has a full row to itself.
        """
        self._decide_one_of_each()
        self.app.show_decisions()
        verdict = str(self.app.cards[self.paths[1]].verdict_label.cget("text"))
        self.assertEqual(verdict, "✘ FALSE POSITIVE")
        self.assertNotIn("but you rejected it", verdict)
        self.assertIn("but you rejected it", self.app._preview_caption_text(self.paths[1]))

    def test_the_preview_caption_carries_the_verdict(self) -> None:
        self._decide_one_of_each()
        caption = self.app._preview_caption_text(self.paths[1])
        self.assertIn("DETECTED", caption)
        self.assertIn("REJECTED", caption)
        self.assertIn("FALSE POSITIVE", caption)
        self.assertIn("not detected", self.app._preview_caption_text(self.paths[2]))

    def test_the_stats_line_separates_what_you_did_from_what_the_scanner_got_wrong(self) -> None:
        """Eight pipe-separated numbers in a row answered two questions at once.

        "Accepted 2" and "False positives 1" are counts of different things — one is
        the reviewer's tally, the other the scanner's error rate — and read as one
        undifferentiated run they could not be told apart at a glance.
        """
        self._decide_one_of_each()
        self.app._update_stats_panel()
        stats = self.app.stats_var.get()
        self.assertIn("You: 2 accepted", stats)
        self.assertIn("1 skipped", stats)
        self.assertIn("Scanner: 1 false positive · 1 false negative", stats)

    def test_the_stats_line_says_so_when_the_scanner_has_made_no_mistakes(self) -> None:
        assert self.app.store is not None
        state = self.app.current_state
        assert state is not None
        self.app.threshold_var.set(0.5)
        # Both accepted, both detected: two true positives and nothing wrong.
        self.app.store.save_labels({self.paths[0]: 1, self.paths[1]: 1})
        self.app._update_stats_panel()
        self.assertIn("Scanner: no mistakes so far", self.app.stats_var.get())

    def test_opening_the_decisions_view_with_nothing_decided_raises_no_dialog(self) -> None:
        """The empty-state panel already says it; a modal on top was one more click."""
        assert self.app.store is not None
        self.app.store.save_labels({})
        with DIALOGS.recording() as dialogs:
            self.app.show_decisions()
        self.assertEqual(dialogs.seen, [], "opening an empty Decisions view raised a dialog")
        self.assertEqual(self.app.view_mode, "decided")
        self.assertEqual(self.app.page_samples, [])
        self.assertIn("Nothing decided yet", self.app.status_var.get())

    def _band_of(self) -> dict[str, str]:
        return {str(sample["path"]): str(sample["bucket"]) for sample in self.app.page_samples}

    def test_all_results_bands_every_photo_by_how_sure_the_scanner_is(self) -> None:
        """One view for the whole folder, rather than a tab per question.

        The detected list showed only what cleared the threshold and the review queue
        only what was undecided, so no single screen ever showed the folder. The bands
        sub-divide the scanner's own answer, so a card and its band never disagree.
        """
        state = self.app.current_state
        assert state is not None
        self.app.threshold_var.set(0.5)
        for index, score in enumerate([0.95, 0.82, 0.48, 0.40, 0.20, 0.05]):
            state.scores[index] = score
        self.app.show_all_results()
        self.assertEqual(self.app.view_mode, "triage")
        bands = self._band_of()
        self.assertEqual(bands[self.paths[0]], "Detected")
        self.assertEqual(bands[self.paths[1]], "Detected")
        # Within the margin below the line: a near miss worth a look.
        self.assertEqual(bands[self.paths[2]], "Possible")
        self.assertEqual(bands[self.paths[3]], "Possible")
        self.assertEqual(bands[self.paths[4]], "Probable reject")
        self.assertEqual(bands[self.paths[5]], "Probable reject")
        # Every photo is present exactly once — this view hides nothing.
        self.assertEqual(len(bands), len(self.paths))
        self.assertEqual(
            [str(sample["bucket"]) for sample in self.app.page_samples][:2],
            ["Detected", "Detected"],
            "the bands are not in their fixed top-to-bottom order",
        )

    def test_a_card_never_disagrees_with_the_band_it_is_in(self) -> None:
        state = self.app.current_state
        assert state is not None
        self.app.threshold_var.set(0.5)
        for index, score in enumerate([0.95, 0.82, 0.48, 0.40, 0.20, 0.05]):
            state.scores[index] = score
        self.app.show_all_results()
        for sample in self.app.page_samples:
            card = self.app.cards[str(sample["path"])]
            detected = str(card.score_label.cget("text")).startswith("DETECTED")
            self.assertEqual(
                detected,
                str(sample["bucket"]) == "Detected",
                f"{sample['bucket']} band disagreed with the card's own detection line",
            )

    def test_the_first_page_carries_every_band_not_just_the_top_one(self) -> None:
        """Cutting one band-sorted list into pages showed one band at a time.

        With 300 detected photos and a 20-card page, pages 1 to 15 were Detected and
        nothing else — so the view whose whole point is that the three bands sit on
        one screen delivered exactly what the separate tabs already did.
        """
        self.app.view_mode = "triage"
        self.app.page_size_var.set(20)
        self.app.displayed_samples = (
            [{"path": f"/d{i}.jpg", "score": 0.9, "bucket": "Detected"} for i in range(300)]
            + [{"path": f"/p{i}.jpg", "score": 0.4, "bucket": "Possible"} for i in range(5)]
            + [{"path": f"/r{i}.jpg", "score": 0.1, "bucket": "Probable reject"} for i in range(900)]
        )
        self.app.page_index = 0
        page = self.app._page_slice()
        bands = {str(sample["bucket"]) for sample in page}
        self.assertEqual(bands, {"Detected", "Possible", "Probable reject"})
        # The small band spends what it has and hands the rest of its share back,
        # rather than leaving a fifth of the page empty.
        self.assertEqual(sum(1 for s in page if s["bucket"] == "Possible"), 5)
        self.assertGreater(sum(1 for s in page if s["bucket"] == "Detected"), 5)
        # Paging advances all three bands together, and never repeats a photo.
        self.app.page_index = 1
        second = self.app._page_slice()
        self.assertTrue({"Detected", "Probable reject"} <= {str(s["bucket"]) for s in second})
        self.assertFalse(
            {str(s["path"]) for s in page} & {str(s["path"]) for s in second},
            "a photo appeared on two pages",
        )

    def test_a_band_too_small_to_spend_its_share_hands_the_rest_back(self) -> None:
        quotas = self.app._band_quotas({"Detected": 300, "Possible": 5, "Probable reject": 900}, 120)
        self.assertEqual(quotas["Possible"], 5)
        self.assertGreater(sum(quotas.values()), 110, "the page was left mostly empty")
        # One band on its own takes the whole page.
        self.assertEqual(self.app._band_quotas({"Detected": 500}, 120), {"Detected": 120})
        self.assertEqual(self.app._band_quotas({"Detected": 0}, 120), {})

    def test_deciding_a_photo_leaves_it_where_it_is_and_fades_it(self) -> None:
        """Rows that empty as you work move the next photo under your cursor."""
        state = self.app.current_state
        assert state is not None
        self.app.threshold_var.set(0.5)
        for index, score in enumerate([0.95, 0.82, 0.48, 0.40, 0.20, 0.05]):
            state.scores[index] = score
        self.app.hide_decided_var.set(True)
        self.app.show_all_results()
        before = [str(sample["path"]) for sample in self.app.page_samples]
        target = self.paths[1]
        self.app.set_label(target, 0)
        self.assertEqual(
            [str(sample["path"]) for sample in self.app.page_samples],
            before,
            "deciding a photo reordered or removed it, despite 'hide decided'",
        )
        card = self.app.cards[target]
        self.assertEqual(card.style, "DimCard.TFrame", "a decided photo kept its bright match border")
        self.assertIn("REJECTED", str(card.label_label.cget("text")))
        # An undecided detection is what must still stand out.
        self.assertEqual(self.app.cards[self.paths[0]].style, "MatchCard.TFrame")

    def test_a_finished_scan_lands_on_all_results(self) -> None:
        """The landing view has to be the one that shows the whole folder."""

        state = self.app.current_state
        assert state is not None
        self.app.view_mode = "review"
        # A finished full scan reports itself with a modal; DIALOGS answers it.
        self.app._scan_completed(self.app._refresh_generation, state, [], full_rescan=True)
        self.assertEqual(self.app.view_mode, "triage")
        self.assertEqual(
            len(self.app.page_samples), len(self.paths), "the landing view did not show the whole folder"
        )

    def test_a_retrain_keeps_the_view_the_reviewer_was_on(self) -> None:
        state = self.app.current_state
        assert state is not None
        self.app.review_samples = [
            {"path": path, "score": 0.9, "bucket": "Likely match"} for path in self.paths
        ]
        self.app.restore_review_view()
        self.app._scan_completed(self.app._refresh_generation, state, [], full_rescan=False)
        self.assertEqual(self.app.view_mode, "review", "a re-rank threw the reviewer into another view")

    def test_moving_the_sensitivity_slider_rebands_everything(self) -> None:
        state = self.app.current_state
        assert state is not None
        self.app.threshold_var.set(0.5)
        for index, score in enumerate([0.95, 0.82, 0.48, 0.40, 0.20, 0.05]):
            state.scores[index] = score
        self.app.show_all_results()
        self.assertEqual(self._band_of()[self.paths[2]], "Possible")
        self.app.threshold_var.set(0.3)
        self.app._after_threshold_settles()
        bands = self._band_of()
        self.assertEqual(bands[self.paths[2]], "Detected", "0.48 stayed below a 0.30 threshold")
        self.assertEqual(bands[self.paths[4]], "Possible")
        self.assertEqual(bands[self.paths[5]], "Probable reject")

    def _spread_scores(self) -> dict[str, str]:
        """Score every path across the three bands; return the band each belongs in.

        Every path, not the first six: the folder fixture carries a duplicate as well
        as the samples, and leaving its score alone silently put an extra photo in a
        band the test then disagreed with.
        """
        state = self.app.current_state
        assert state is not None
        self.app.threshold_var.set(0.5)
        pattern = [0.95, 0.82, 0.48, 0.40, 0.20, 0.05]
        expected: dict[str, str] = {}
        for index, path in enumerate(state.paths):
            score = pattern[index % len(pattern)]
            state.scores[index] = score
            expected[str(path)] = (
                "Detected" if score >= 0.5 else "Possible" if score >= 0.35 else "Probable reject"
            )
        return expected

    @contextlib.contextmanager
    def _bin_available(self, available: bool = True):
        """Pin whether the recycle bin can be used, whatever this machine has."""
        original = gui_module.trash_available
        gui_module.trash_available = lambda: (  # type: ignore[assignment]
            (True, "") if available else (False, "send2trash is not installed")
        )
        try:
            yield
        finally:
            gui_module.trash_available = original  # type: ignore[assignment]

    def _band_buttons(self, band: str) -> list[str]:
        from bikini_scanner import gui as gui_module

        frame = self.app._bucket_headings[band]
        return [
            str(child.cget("text"))
            for child in frame.winfo_children()
            if isinstance(child, gui_module.ttk.Button)
        ]

    def test_each_band_offers_the_bulk_action_that_belongs_to_it(self) -> None:
        """Doing these a card at a time is the work the bands exist to avoid."""
        self._spread_scores()
        self.app.show_all_results()
        self.assertEqual(self._band_buttons("Detected"), ["Export these…"])
        self.assertEqual(self._band_buttons("Probable reject"), ["Reject all & delete…"])
        # The possible band is the one that is meant to be judged by hand, so it gets
        # no bulk action to fire off by accident.
        self.assertEqual(self._band_buttons("Possible"), [])

    def test_the_other_views_carry_no_band_actions(self) -> None:
        self.app.review_samples = [
            {"path": path, "score": 0.9, "bucket": "Likely match"} for path in self.paths
        ]
        self.app.restore_review_view()
        self.assertEqual(self.app._band_actions("Likely match"), [])

    def test_a_band_action_covers_the_whole_band_not_the_page_on_screen(self) -> None:
        expected = self._spread_scores()
        self.app.show_all_results()
        for band in ("Detected", "Possible", "Probable reject"):
            self.assertEqual(
                set(self.app._band_paths(band)),
                {path for path, name in expected.items() if name == band},
                f"the {band} band action would not have covered the whole band",
            )
        # Between them the three bands account for every photo, exactly once.
        covered = [path for band in ("Detected", "Possible", "Probable reject") for path in self.app._band_paths(band)]
        self.assertEqual(sorted(covered), sorted(expected))

    def test_reject_and_delete_labels_retains_and_bins_in_that_order(self) -> None:
        """The labels are the training, so they are written before the files go.

        The retrain that folds them in runs off cached embeddings a moment later, by
        which time the photos no longer exist — so the labels are also marked to
        survive their files, or the delete would undo its own training.
        """

        self._spread_scores()
        self.app.show_all_results()
        targets = self.app._band_paths("Probable reject")
        self.assertTrue(targets)
        order: list[str] = []
        retained: list[str] = []
        trashed: list[str] = []
        self.app._retain_global_labels = lambda paths: (  # type: ignore[method-assign]
            order.append("retain"),
            retained.extend(paths),
        )
        self.app._trash_and_report = lambda paths: (  # type: ignore[method-assign]
            order.append("trash"),
            trashed.extend(paths),
        )
        with self._bin_available(), DIALOGS.answering(True):
            self.app.reject_and_delete_band("Probable reject")
        assert self.app.store is not None
        labels = self.app.store.load_labels()
        self.assertTrue(all(labels.get(path) == 0 for path in targets), "the band was not rejected")
        self.assertEqual(sorted(retained), sorted(targets))
        self.assertEqual(sorted(trashed), sorted(targets))
        self.assertEqual(order, ["retain", "trash"], "the files went before the labels were secured")

    def test_reject_and_delete_leaves_photos_you_already_judged_alone(self) -> None:
        """It used to overwrite an Accept and bin the file, while promising not to.

        The all-results view deliberately keeps decided photos on screen, so a band
        action reached everything the reviewer had already judged — including a photo
        they had rescued from the reject band by accepting it.
        """

        self._spread_scores()
        self.app.show_all_results()
        band = self.app._band_paths("Probable reject")
        self.assertGreaterEqual(len(band), 2)
        rescued, skipped = band[0], band[1]
        assert self.app.store is not None
        self.app.store.save_labels({rescued: 1, skipped: 2})
        self.app.show_all_results()
        trashed: list[str] = []
        self.app._trash_and_report = lambda paths: trashed.extend(paths)  # type: ignore[method-assign]
        self.app._retain_global_labels = lambda paths: None  # type: ignore[method-assign]
        with self._bin_available(), DIALOGS.answering(True):
            self.app.reject_and_delete_band("Probable reject")
        labels = self.app.store.load_labels()
        self.assertEqual(labels.get(rescued), 1, "an accepted photo was re-labelled REJECT")
        self.assertEqual(labels.get(skipped), 2, "a skipped photo was re-labelled REJECT")
        self.assertNotIn(rescued, trashed, "an accepted photo was moved to the recycle bin")
        self.assertNotIn(skipped, trashed, "a skipped photo was moved to the recycle bin")

    def test_reject_and_delete_writes_nothing_when_the_bin_is_unavailable(self) -> None:
        """Both halves or neither: the labels are permanent and keep training."""

        self._spread_scores()
        self.app.show_all_results()
        band = self.app._band_paths("Probable reject")
        retained: list[str] = []
        self.app._retain_global_labels = lambda paths: retained.extend(paths)  # type: ignore[method-assign]
        with self._bin_available(False), DIALOGS.answering(True):
            self.app.reject_and_delete_band("Probable reject")
        assert self.app.store is not None
        labels = self.app.store.load_labels()
        self.assertTrue(
            all(labels.get(path) is None for path in band),
            "the band was rejected even though nothing could be deleted",
        )
        self.assertEqual(retained, [], "labels were retained for a deletion that never happened")

    def test_a_decision_still_refreshes_when_a_label_filter_is_hiding_it(self) -> None:
        """The fast path's premise fails when a filter answers on the label itself."""
        self._spread_scores()
        self.app.label_filter_var.set("unlabeled")
        self.app.show_all_results()
        target = str(self.app.page_samples[0]["path"])
        self.app.set_label(target, 1)
        self.assertNotIn(
            target,
            [str(sample["path"]) for sample in self.app.page_samples],
            "a decided photo stayed on screen while 'Labels: unlabeled' was set",
        )

    def test_deciding_one_selected_card_clears_the_selection(self) -> None:
        """It kept its selection border and joined the next click."""
        self._spread_scores()
        self.app.show_all_results()
        order = [str(sample["path"]) for sample in self.app.page_samples]
        self.app.click_card(order[2], toggle=True)
        self.assertEqual(self.app.selected_paths, {order[2]})
        self.app.label_selection(0)
        self.assertEqual(self.app.selected_paths, set(), "a one-card selection survived its decision")
        self.assertEqual(self.app.cards[order[2]].style, "DimCard.TFrame")

    def test_deciding_a_selected_card_keeps_the_progress_count_on_screen(self) -> None:
        """The counts are the per-click feedback; "Selection cleared." replaced them."""
        self._spread_scores()
        self.app.show_all_results()
        order = [str(sample["path"]) for sample in self.app.page_samples]
        self.app.click_card(order[2], toggle=True)
        self.app.label_selection(0)
        status = self.app.status_var.get()
        self.assertIn("left in folder", status, f"the decision's own status was overwritten: {status!r}")
        self.assertNotEqual(status, "Selection cleared.")
        # And a later Escape must not announce a selection nobody still had.
        self.app.clear_selection()
        self.assertEqual(self.app.status_var.get(), status, "Escape announced a stale selection")

    def test_a_bulk_decision_also_leaves_its_own_status_standing(self) -> None:
        self._spread_scores()
        self.app.show_all_results()
        order = [str(sample["path"]) for sample in self.app.page_samples]
        self.app.click_card(order[0])
        self.app.click_card(order[2], extend=True)
        self.app.label_selection(1)
        self.assertIn("3 photos", self.app.status_var.get())
        self.app.clear_selection()
        self.assertIn("3 photos", self.app.status_var.get(), "Escape overwrote the batch status")

    def test_a_measurement_overtaken_by_a_retrain_measures_the_new_scan(self) -> None:
        """Handing back a stale answer told a well-labelled folder to go and label.

        The dialog would recompute, find no measurement and nothing in flight, and
        fall through to the "judge about forty photos first" message.
        """
        self._spread_scores()
        self.app.show_all_results()
        seen: list[object] = []
        self.app._measure_out_of_fold = lambda state: seen.append(state) or None  # type: ignore[method-assign]
        landed: list[int] = []
        self.app._oof_busy = True
        self.app.ensure_out_of_fold(lambda: landed.append(1))
        stale = self.app.current_state
        assert stale is not None
        # The retrain lands first and swaps the state under the worker.
        self.app.current_state = scorer_module.ScoreState(
            paths=list(stale.paths),
            embeddings=stale.embeddings,
            zero_shot_scores=stale.zero_shot_scores,
            scores=stale.scores,
            axis_scores=stale.axis_scores,
            face_counts=None,
            classifier_trained=False,
            classifier_label_count=0,
            excluded=stale.excluded,
        )
        self.app._out_of_fold_ready(stale, None)
        self.assertEqual(landed, [], "a stale answer was handed back as if it were current")
        self.assertEqual(seen[-1], self.app.current_state, "the replacement scan was not measured")

    def test_the_retry_chase_gives_up_rather_than_spawning_forever(self) -> None:
        self._spread_scores()
        self.app.show_all_results()
        self.app._measure_out_of_fold = lambda state: None  # type: ignore[method-assign]
        landed: list[int] = []
        self.app._oof_busy = True
        self.app.ensure_out_of_fold(lambda: landed.append(1))
        for _ in range(gui_module._OOF_MAX_RETRIES + 2):
            stale = self.app.current_state
            assert stale is not None
            self.app.current_state = scorer_module.ScoreState(
                paths=list(stale.paths),
                embeddings=stale.embeddings,
                zero_shot_scores=stale.zero_shot_scores,
                scores=stale.scores,
                axis_scores=stale.axis_scores,
                face_counts=None,
                classifier_trained=False,
                classifier_label_count=0,
                excluded=stale.excluded,
            )
            self.app._out_of_fold_ready(stale, None)
            if landed:
                break
        self.assertEqual(landed, [1], "the chase never ended")
        self.assertIn("changed", self.app.auto_decide_plan(0.95).reason)

    def test_clearing_the_selection_says_so(self) -> None:
        self._spread_scores()
        self.app.show_all_results()
        self.app.select_all_on_page()
        self.assertIn("selected", self.app.status_var.get())
        self.app.clear_selection()
        self.assertNotIn("selected —", self.app.status_var.get())

    def test_burst_groups_are_dropped_once_they_describe_another_scan(self) -> None:
        """A rescan replaces the state; the old grouping must stop expanding."""
        self._spread_scores()
        self.app.show_all_results()
        group = self._link_a_burst()
        self.app.near_duplicate_var.set(True)
        self.assertEqual(self.app._expand_to_duplicates([group[0]]), group)
        # A rescan hands over a new state object; nothing has regrouped it yet.
        state = self.app.current_state
        assert state is not None
        self.app.current_state = scorer_module.ScoreState(
            paths=list(state.paths),
            embeddings=state.embeddings,
            zero_shot_scores=state.zero_shot_scores,
            scores=state.scores,
            axis_scores=state.axis_scores,
            face_counts=None,
            classifier_trained=False,
            classifier_label_count=0,
            excluded=state.excluded,
        )
        self.assertEqual(
            self.app._expand_to_duplicates([group[0]]),
            [group[0]],
            "a decision expanded through groups built for a previous scan",
        )

    def test_declining_the_reject_and_delete_prompt_changes_nothing(self) -> None:

        self._spread_scores()
        self.app.show_all_results()
        targets = self.app._band_paths("Probable reject")
        trashed: list[str] = []
        self.app._trash_and_report = lambda paths: trashed.extend(paths)  # type: ignore[method-assign]
        with self._bin_available(), DIALOGS.answering(False):
            self.app.reject_and_delete_band("Probable reject")
        assert self.app.store is not None
        labels = self.app.store.load_labels()
        self.assertEqual(trashed, [], "files were binned after the prompt was declined")
        self.assertTrue(all(labels.get(path) is None for path in targets))

    def test_a_decision_in_the_all_results_view_does_not_resort_the_folder(self) -> None:
        """Re-sorting the whole folder on every click is the cost that scales.

        The bands are cut off the score and a decided photo keeps its place, so a
        decision provably cannot change the displayed list or its order. Doing the
        filter-and-sort pass anyway cost about 400 ms per Accept at 80 000 photos —
        on the main thread, on a job that is already 80 000 photos long.
        """
        self._spread_scores()
        self.app.show_all_results()
        calls: list[int] = []
        original = self.app._apply_display_filters
        self.app._apply_display_filters = lambda samples: (  # type: ignore[method-assign]
            calls.append(1),
            original(samples),
        )[1]
        before = [str(sample["path"]) for sample in self.app.page_samples]
        self.app.set_label(before[0], 0)
        self.assertEqual(calls, [], "one decision re-filtered and re-sorted the whole folder")
        self.assertEqual([str(s["path"]) for s in self.app.page_samples], before)
        # The card itself still had to be repainted.
        self.assertEqual(self.app.cards[before[0]].style, "DimCard.TFrame")

    def test_a_decision_in_a_view_that_hides_it_still_refreshes(self) -> None:
        """The shortcut above must not leak into the views that do drop the photo."""
        self.app.review_samples = [
            {"path": path, "score": 0.9, "bucket": "Likely match"} for path in self.paths
        ]
        self.app.hide_decided_var.set(True)
        self.app.restore_review_view()
        target = str(self.app.page_samples[0]["path"])
        self.app.set_label(target, 0)
        self.assertNotIn(
            target,
            [str(sample["path"]) for sample in self.app.page_samples],
            "a decided photo stayed in a view that is meant to hide it",
        )

    def test_the_page_plan_is_built_once_per_refresh(self) -> None:
        """One refresh asked for it three times, each a pass over the whole folder."""
        self._spread_scores()
        self.app.show_all_results()
        built: list[int] = []
        original = self.app._band_quotas
        self.app._band_quotas = lambda totals, size: (  # type: ignore[method-assign]
            built.append(1),
            original(totals, size),
        )[1]
        self.app._page_count()
        self.app._page_slice()
        self.app._sync_pager()
        self.assertEqual(built, [], "the cached page plan was rebuilt")
        # A new displayed list must invalidate it.
        self.app._refresh_displayed_results(reset_page=False)
        self.app._page_count()
        self.assertEqual(len(built), 1, "the page plan survived a rebuild of the displayed list")

    def test_the_headline_counts_are_batched_not_recomputed_per_click(self) -> None:
        """They walk every label in the folder and cannot move by more than one."""
        self._spread_scores()
        self.app.show_all_results()
        counted: list[int] = []
        original = self.app._outcome_counts
        self.app._outcome_counts = lambda labels=None: (  # type: ignore[method-assign]
            counted.append(1),
            original(labels),
        )[1]
        self.app.set_label(str(self.app.page_samples[0]["path"]), 0)
        self.assertEqual(counted, [], "the stats line was recomputed inline on a decision")
        # Queued, not dropped: flushing gives the real numbers.
        self.app._flush_summary_refresh()
        self.assertEqual(len(counted), 1)
        self.assertIn("You:", self.app.stats_var.get())

    def test_shift_click_selects_a_range_and_one_key_decides_it(self) -> None:
        """Between "decide this photo" and "decide all 5 000" there was nothing."""
        self._spread_scores()
        self.app.show_all_results()
        order = [str(sample["path"]) for sample in self.app.page_samples]
        self.app.click_card(order[1])
        self.app.click_card(order[3], extend=True)
        self.assertEqual(self.app.selected_paths, set(order[1:4]))
        for path in order[1:4]:
            self.assertEqual(self.app.cards[path].style, "SelectedCard.TFrame")
        self.app.label_selection(0)
        assert self.app.store is not None
        labels = self.app.store.load_labels()
        self.assertTrue(all(labels.get(path) == 0 for path in order[1:4]))
        self.assertIsNone(labels.get(order[0]), "the decision reached past the selection")
        self.assertEqual(self.app.selected_paths, set(), "the selection outlived the decision")

    def test_control_click_toggles_one_card_and_a_plain_click_clears(self) -> None:
        self._spread_scores()
        self.app.show_all_results()
        order = [str(sample["path"]) for sample in self.app.page_samples]
        self.app.click_card(order[0])
        self.app.click_card(order[2], toggle=True)
        self.app.click_card(order[4], toggle=True)
        self.assertEqual(self.app.selected_paths, {order[2], order[4]})
        self.app.click_card(order[2], toggle=True)
        self.assertEqual(self.app.selected_paths, {order[4]})
        self.app.click_card(order[1])
        self.assertEqual(self.app.selected_paths, set(), "a plain click left the selection standing")

    def test_a_decision_aimed_outside_the_selection_takes_only_that_card(self) -> None:
        """The file-manager rule: dragging a file outside a highlight takes just it."""
        self._spread_scores()
        self.app.show_all_results()
        order = [str(sample["path"]) for sample in self.app.page_samples]
        self.app.click_card(order[1])
        self.app.click_card(order[3], extend=True)
        self.app.label_selection(1, order[5])
        assert self.app.store is not None
        labels = self.app.store.load_labels()
        self.assertEqual(labels.get(order[5]), 1)
        self.assertTrue(all(labels.get(path) is None for path in order[1:4]))

    def test_a_selection_never_outlives_the_page_it_was_made_on(self) -> None:
        """A bulk decision must not reach photos the reviewer can no longer see."""
        self._spread_scores()
        self.app.page_size_var.set(20)
        self.app.show_all_results()
        self.app.select_all_on_page()
        self.assertTrue(self.app.selected_paths)
        self.app.current_samples = [
            sample for sample in self.app.current_samples if str(sample["bucket"]) == "Detected"
        ]
        self.app._refresh_displayed_results()
        on_page = {str(sample["path"]) for sample in self.app.page_samples}
        self.assertTrue(self.app.selected_paths <= on_page, "the selection kept photos that left the page")

    def test_escape_clears_the_selection(self) -> None:
        self._spread_scores()
        self.app.show_all_results()
        self.app.select_all_on_page()
        self.app._handle_clear_selection_shortcut(None)
        self.assertEqual(self.app.selected_paths, set())

    def _link_a_burst(self) -> list[str]:
        """Pretend the grouping pass found one burst among the folder's photos."""
        group = self.paths[1:4]
        self.app._near_duplicate_index = dict.fromkeys(group, group)
        self.app._near_duplicate_state = self.app.current_state
        return group

    def test_deciding_one_of_a_burst_decides_the_burst_when_asked(self) -> None:
        self._spread_scores()
        self.app.show_all_results()
        group = self._link_a_burst()
        self.app.near_duplicate_var.set(True)
        self.app.click_card(group[0])
        self.app.label_selection(1)
        assert self.app.store is not None
        labels = self.app.store.load_labels()
        self.assertTrue(all(labels.get(path) == 1 for path in group), "the burst was not decided together")
        self.assertIn("near-identical", self.app.status_var.get())

    def test_the_burst_expansion_is_off_unless_switched_on(self) -> None:
        """One click deciding eight photos has to be something you turned on."""
        self._spread_scores()
        self.app.show_all_results()
        group = self._link_a_burst()
        self.app.near_duplicate_var.set(False)
        self.app.click_card(group[0])
        self.app.label_selection(0)
        assert self.app.store is not None
        labels = self.app.store.load_labels()
        self.assertEqual(labels.get(group[0]), 0)
        self.assertTrue(all(labels.get(path) is None for path in group[1:]))

    def test_a_card_says_it_is_one_of_a_burst_either_way(self) -> None:
        """It is the reason to switch the expansion on, and the warning once it is."""
        self._spread_scores()
        self.app.show_all_results()
        group = self._link_a_burst()
        self.app.near_duplicate_var.set(False)
        self.assertIn("not linked", self.app._axis_details_text(group[0]))
        self.app.near_duplicate_var.set(True)
        details = self.app._axis_details_text(group[0])
        self.assertIn(f"1 of {len(group)} near-identical", details)
        self.assertNotIn("not linked", details)

    def test_no_dialog_can_stall_the_suite(self) -> None:
        """The safety net itself, because forgetting it is what cost the time.

        Every message box blocks until someone presses OK. Stubbing them one call
        site at a time only ever covered the ones already discovered, and the one that
        was missed took the suite from 61 s to 1 490 s. This drives three dialogs with
        nothing stubbed locally and expects to come straight back.
        """
        started = time.perf_counter()
        with DIALOGS.recording() as dialogs:
            # Info, from a view with nothing to show.
            self.app.current_state = None
            self.app.show_all_results()
            # A question, declined by default, so nothing destructive follows.
            self.app.mark_all_shown(0)
        self.assertLess(time.perf_counter() - started, 5.0, "a dialog blocked the test")
        self.assertTrue(dialogs.seen, "the dialogs were never raised, so nothing was proved")
        self.assertIn("No results", [title for _kind, title, _message in dialogs.seen])

    def test_a_question_is_declined_unless_a_test_says_otherwise(self) -> None:
        """False is the safe default: it is the answer to 'delete these?'."""
        self._spread_scores()
        self.app.show_all_results()
        band = self.app._band_paths("Probable reject")
        trashed: list[str] = []
        self.app._trash_and_report = lambda paths: trashed.extend(paths)  # type: ignore[method-assign]
        self.app._retain_global_labels = lambda paths: None  # type: ignore[method-assign]
        with self._bin_available():
            self.app.reject_and_delete_band("Probable reject")
        assert self.app.store is not None
        self.assertEqual(trashed, [], "an unanswered question was taken as yes")
        self.assertTrue(all(self.app.store.load_labels().get(path) is None for path in band))

    def test_auto_decide_needs_labels_before_it_will_decide_anything(self) -> None:
        """It writes labels in bulk, so it may not guess from nothing."""
        self._spread_scores()
        self.app.show_all_results()
        plan = self.app.auto_decide_plan(0.99)
        self.assertFalse(plan.usable)
        self.assertEqual(plan.accept_paths, [])
        self.assertEqual(plan.reject_paths, [])
        self.assertTrue(plan.reason, "the refusal gave no reason")

    def test_the_held_out_measurement_runs_off_the_main_thread(self) -> None:
        """It fits a model per fold over every label, so it cannot run inline.

        After a first auto-decide pass writes tens of thousands of labels, doing this
        on the main thread froze the window for minutes with the dialog already
        painted — and repeated the whole cost on every change of confidence.
        """
        self._spread_scores()
        self.app.show_all_results()
        ran_on: list[str] = []
        measured: list[tuple[list[float], list[int]] | None] = [([0.9] * 60, [1] * 30 + [0] * 30)]

        def fake_measure(state: object) -> tuple[list[float], list[int]] | None:
            ran_on.append(threading.current_thread().name)
            return measured[0]

        self.app._measure_out_of_fold = fake_measure  # type: ignore[method-assign]
        landed: list[int] = []
        self.assertFalse(
            self.app.ensure_out_of_fold(lambda: landed.append(1)),
            "the first call claimed the answer was already in hand",
        )
        # Wait for the worker itself. The hand-back goes through root.after from that
        # thread, which needs a running mainloop the harness does not have, so the
        # delivery half is driven directly below rather than polled for.
        for _ in range(400):
            if ran_on:
                break
            time.sleep(0.01)
        self.assertTrue(ran_on, "the measurement never ran")
        self.assertNotEqual(ran_on[0], "MainThread", "the measurement ran on the UI thread")
        self.app._out_of_fold_ready(self.app.current_state, measured[0])  # type: ignore[arg-type]
        self.assertEqual(landed, [1], "the waiter was never called back")
        # Measured once per state: changing the confidence must not re-run it.
        self.assertTrue(self.app.ensure_out_of_fold(lambda: landed.append(2)))
        self.assertEqual(len(ran_on), 1, "the measurement was repeated for the same state")

    def test_a_second_waiter_is_not_forgotten_while_a_measurement_runs(self) -> None:
        self._spread_scores()
        self.app.show_all_results()
        self.app._measure_out_of_fold = lambda state: None  # type: ignore[method-assign]
        landed: list[int] = []
        self.app._oof_busy = True
        self.assertFalse(self.app.ensure_out_of_fold(lambda: landed.append(1)))
        self.app._out_of_fold_ready(self.app.current_state, None)  # type: ignore[arg-type]
        self.assertEqual(landed, [1], "a caller arriving mid-measurement never heard back")

    def test_auto_decide_declines_when_the_state_carries_no_features(self) -> None:
        """An older cached state has no feature matrix, so nothing can be measured."""
        self._spread_scores()
        state = self.app.current_state
        assert state is not None
        state.features = None
        self.assertIsNone(self.app._out_of_fold_scores())
        self.assertFalse(self.app.auto_decide_plan(0.95).usable)

    def test_the_three_views_can_be_switched_from_the_keyboard(self) -> None:
        """Every other step of the review loop has a key; switching view did not."""
        self._decide_one_of_each()
        self.app.review_samples = [
            {"path": path, "score": 0.9, "bucket": "Likely match"} for path in self.paths
        ]
        self.app._handle_view_shortcut(None, self.app.show_decisions)
        self.assertEqual(self.app.view_mode, "decided")
        self.app._handle_view_shortcut(None, self.app.restore_review_view)
        self.assertEqual(self.app.view_mode, "review")
        self.app._handle_view_shortcut(None, self.app.show_detected_files)
        self.assertEqual(self.app.view_mode, "detected")

    def test_a_view_shortcut_does_not_fire_while_typing_in_a_filter(self) -> None:
        self.app.view_mode = "review"
        self.app._focus_is_text_input = lambda: True  # type: ignore[method-assign]
        self.app._handle_view_shortcut(None, self.app.show_decisions)
        self.assertEqual(self.app.view_mode, "review", "typing '3' in a filter box switched view")

    def test_changing_a_decision_in_the_decisions_view_moves_the_photo_to_its_new_group(self) -> None:
        self._decide_one_of_each()
        self.app.show_decisions()
        self.app.set_label(self.paths[1], 1)
        buckets = {str(sample["path"]): str(sample["bucket"]) for sample in self.app.page_samples}
        self.assertEqual(buckets[self.paths[1]], "True positives")
        self.assertEqual(self.app.view_mode, "decided", "changing a decision threw the reviewer out of the view")

    def test_the_decisions_view_follows_the_sensitivity_slider(self) -> None:
        self._decide_one_of_each()
        self.app.show_decisions()
        # Dragged below every score: nothing is "missed" any more, so the accepted
        # photo that was a false negative becomes a true positive.
        self.app.threshold_var.set(0.05)
        self.app._after_threshold_settles()
        buckets = {str(sample["path"]): str(sample["bucket"]) for sample in self.app.page_samples}
        self.assertEqual(buckets[self.paths[2]], "True positives")
        self.assertEqual(buckets[self.paths[3]], "False positives")


class AgeGateFrameVeto(unittest.TestCase):
    """A per-subject reading may add exclusions, never cancel the frame's own."""

    @staticmethod
    def _frame(with_readable_adult: bool) -> cascade.RegionScoreTable:
        kinds: list[str] = []
        subject: list[int] = []
        rows: list[dict[str, float]] = []

        def add(kind: str, who: int, child: float, adult: float, detail: float) -> None:
            kinds.append(kind)
            subject.append(who)
            rows.append(
                {
                    "child": child, "adult": adult, "person": 0.95, "female": 0.95, "nsfw": 0.5,
                    "bikini": detail, "cleavage": detail, "midriff": detail,
                    "bikini_top": detail, "bikini_bottom": detail,
                }
            )

        # Whole frame: overwhelming child evidence, strong swimwear evidence.
        add(regions.KIND_FULL, -1, 0.99, 0.02, 0.95)
        # An unaged subject: body crops, but a face too small to crop, so never `minor`.
        for kind in (regions.KIND_CHEST, regions.KIND_WAIST, regions.KIND_TORSO):
            add(kind, 0, 0.5, 0.5, 0.97)
        if with_readable_adult:
            add(regions.KIND_FACE, 1, 0.02, 0.99, 0.5)
            add(regions.KIND_TORSO, 1, 0.5, 0.5, 0.55)
        count = len(kinds)
        return cascade.RegionScoreTable(
            owner=np.zeros(count, dtype=np.int64),
            kinds=np.array(kinds),
            axis_scores={
                axis: np.array([row[axis] for row in rows], dtype=np.float32) for axis in rows[0]
            },
            image_count=1,
            full_row=np.array([0], dtype=np.int64),
            subject=np.array(subject, dtype=np.int64),
        )

    def test_a_readable_adult_face_cannot_switch_off_the_frame_veto(self) -> None:
        """One adult in shot used to un-exclude a frame that screamed minor.

        The per-subject verdict replaced the whole-frame answer outright, so adding a
        second subject whose face read clearly adult flipped the identical frame from
        excluded to scored with its child evidence unchanged at 0.98. The soft gate
        still zeroed the score, so it could not surface as a match, but it stopped
        being filtered out of the results.
        """
        config = ScannerConfig()
        self.assertTrue(config.exclude_minors, "the gate under test is off by default")
        for readable in (False, True):
            result = cascade.evaluate(
                self._frame(readable),
                config,
                face_counts=np.array([1 if readable else 0], dtype=np.int32),
            )
            self.assertEqual(
                result.stage[0],
                cascade.STAGE_MINOR,
                f"a frame with 0.98 child evidence was not age-gated (readable adult={readable})",
            )
            self.assertTrue(bool(result.excluded[0]))
            self.assertEqual(float(result.score[0]), 0.0)

    def test_a_frame_with_no_child_evidence_is_still_scored(self) -> None:
        """The veto must not become a blanket exclusion."""
        table = self._frame(True)
        table.axis_scores["child"] = np.full(len(table.kinds), 0.02, dtype=np.float32)
        table.axis_scores["adult"] = np.full(len(table.kinds), 0.98, dtype=np.float32)
        result = cascade.evaluate(table, ScannerConfig(), face_counts=np.array([1], dtype=np.int32))
        self.assertEqual(result.stage[0], cascade.STAGE_SCORED)
        self.assertFalse(bool(result.excluded[0]))


class AutoDecideCalibration(unittest.TestCase):
    """The scanner may only decide alone where it has earned the right to."""

    def test_a_perfect_run_of_twenty_does_not_prove_a_perfect_model(self) -> None:
        """The bound is what stops a small sample promising nobody will be wrong."""
        self.assertLess(autodecide.wilson_lower_bound(20, 20), 0.9)
        self.assertLess(autodecide.wilson_lower_bound(200, 200), 0.99)
        # Evidence still moves it the right way as it accumulates.
        self.assertGreater(
            autodecide.wilson_lower_bound(2000, 2000), autodecide.wilson_lower_bound(200, 200)
        )
        self.assertEqual(autodecide.wilson_lower_bound(0, 0), 0.0)
        self.assertLess(autodecide.wilson_lower_bound(90, 100), 0.9)

    def _separated(self, count: int = 400, noise: float = 0.0) -> tuple[list[float], list[int]]:
        rng = np.random.default_rng(21)
        truth = (rng.random(count) < 0.5).astype(np.int64)
        scores = np.where(truth == 1, 0.9, 0.1) + rng.normal(0, noise, count) if noise else np.where(
            truth == 1, 0.9, 0.1
        )
        return [float(value) for value in scores], [int(value) for value in truth]

    def test_a_clean_separation_hands_back_both_cuts(self) -> None:
        scores, truth = self._separated()
        plan = autodecide.plan_auto_decisions(
            ["hi.jpg", "lo.jpg", "mid.jpg"], [0.95, 0.05, 0.5], scores, truth, target=0.95
        )
        self.assertTrue(plan.usable)
        self.assertEqual(plan.accept_paths, ["hi.jpg"])
        self.assertEqual(plan.reject_paths, ["lo.jpg"])
        self.assertEqual(plan.remaining, 1, "the uncertain photo was decided anyway")

    def test_a_refused_plan_still_reports_everything_as_remaining(self) -> None:
        """It decided nothing, so nothing can have stopped remaining.

        Found by fuzzing: 532 of 4000 random trials broke
        decided + remaining == len(undecided_paths), because every early return left
        `remaining` at its default of 0. Nothing renders it on a refused plan today,
        but the obvious next use would report "0 left for you" for a folder of 80,000
        undecided photos.
        """
        paths = [f"p{index}.jpg" for index in range(25)]
        scores = [0.5] * 25
        for labelled, truth in (
            ([], []),                                   # nothing judged yet
            ([0.9] * 60, [1] * 60),                     # all one verdict
            ([0.9, 0.1], [1, 0]),                       # too few to measure
        ):
            plan = autodecide.plan_auto_decisions(paths, scores, labelled, truth)
            self.assertFalse(plan.usable)
            self.assertEqual(
                plan.decided + plan.remaining, len(paths), f"partition broken for {plan.reason!r}"
            )
            self.assertEqual(plan.remaining, len(paths))

    def test_too_few_labels_decides_nothing_and_says_why(self) -> None:
        plan = autodecide.plan_auto_decisions(["a.jpg"], [0.9], [0.9, 0.1], [1, 0])
        self.assertFalse(plan.usable)
        self.assertIn("too few", plan.reason)
        self.assertEqual(plan.accept_paths, [])
        self.assertEqual(plan.reject_paths, [])

    def test_labels_that_are_all_one_verdict_decide_nothing(self) -> None:
        scores = [0.5 + index / 1000 for index in range(60)]
        plan = autodecide.plan_auto_decisions(["a.jpg"], [0.9], scores, [1] * 60)
        self.assertFalse(plan.usable)
        self.assertIn("same verdict", plan.reason)

    def test_an_unreachable_target_is_refused_rather_than_approximated(self) -> None:
        scores, truth = self._separated()
        plan = autodecide.plan_auto_decisions(["a.jpg"], [0.95], scores, truth, target=0.99999)
        self.assertFalse(plan.usable)
        self.assertIn("right", plan.reason)

    def test_the_mistake_estimate_is_pessimistic_not_optimistic(self) -> None:
        """It is the number the reviewer decides on; it must not flatter the model."""
        scores, truth = self._separated()
        plan = autodecide.plan_auto_decisions(
            [f"p{i}.jpg" for i in range(100)], [0.95] * 100, scores, truth, target=0.95
        )
        self.assertTrue(plan.usable)
        self.assertIsNotNone(plan.accept.confident_precision)
        self.assertLess(
            plan.accept.confident_precision or 1.0,
            plan.accept.precision or 0.0,
            "the bound was not below the raw observation",
        )
        self.assertGreater(plan.expected_mistakes, 0.0, "a perfect record was promised")

    def test_no_photo_is_both_accepted_and_rejected(self) -> None:
        """Contradictory labels can make the two cuts cross; a photo still gets one."""
        rng = np.random.default_rng(22)
        scores = [float(value) for value in rng.random(300)]
        truth = [int(value) for value in (rng.random(300) < 0.5)]
        for target in (0.5, 0.6):
            plan = autodecide.plan_auto_decisions(
                [f"p{i}.jpg" for i in range(50)],
                [float(value) for value in rng.random(50)],
                scores,
                truth,
                target=target,
            )
            self.assertEqual(
                set(plan.accept_paths) & set(plan.reject_paths), set(), "a photo got both verdicts"
            )
            self.assertEqual(plan.decided + plan.remaining, 50)

    def test_the_inputs_must_line_up(self) -> None:
        scores, truth = self._separated()
        with self.assertRaises(ValueError):
            autodecide.plan_auto_decisions(["a.jpg", "b.jpg"], [0.9], scores, truth)


class NearDuplicateGrouping(unittest.TestCase):
    """Burst shots are one decision, not eight."""

    @staticmethod
    def _burst(base: np.ndarray, count: int, rng, spread: float = 0.005) -> list[np.ndarray]:
        frames = [base]
        for _ in range(count - 1):
            noisy = base + rng.normal(scale=spread, size=base.shape).astype(np.float32)
            frames.append(noisy / np.linalg.norm(noisy))
        return frames

    def _folder(self, sizes: list[int], rng, dim: int = 64) -> tuple[list[str], np.ndarray, dict[str, int]]:
        paths: list[str] = []
        rows: list[np.ndarray] = []
        shot_of: dict[str, int] = {}
        for shot, count in enumerate(sizes):
            base = rng.normal(size=dim).astype(np.float32)
            base /= np.linalg.norm(base)
            for frame, vector in enumerate(self._burst(base, count, rng)):
                path = f"shot{shot:02d}_{frame}.jpg"
                paths.append(path)
                rows.append(vector)
                shot_of[path] = shot
        return paths, np.vstack(rows).astype(np.float32), shot_of

    def test_a_burst_becomes_one_group_and_singles_are_left_alone(self) -> None:
        rng = np.random.default_rng(11)
        paths, embeddings, shot_of = self._folder([4, 3, 1, 1, 2], rng)
        groups = duplicates.near_duplicate_groups(paths, embeddings)
        self.assertEqual(sorted(len(group) for group in groups), [2, 3, 4])
        for group in groups:
            self.assertEqual(
                len({shot_of[path] for path in group}),
                1,
                "two different shots were merged into one group",
            )
        # A photo that is nobody's near-duplicate is not returned at all.
        grouped = {path for group in groups for path in group}
        self.assertNotIn("shot02_0.jpg", grouped)

    def test_photos_below_the_threshold_are_not_grouped(self) -> None:
        rng = np.random.default_rng(12)
        paths, embeddings, _ = self._folder([2, 2], rng)
        # Far above any real burst's similarity: nothing should survive it.
        self.assertEqual(duplicates.near_duplicate_groups(paths, embeddings, threshold=0.999999), [])

    def test_an_unusable_embedding_is_never_anybodys_duplicate(self) -> None:
        """inf and NaN rows used to be divided through, warning and yielding NaN units."""
        import warnings

        paths = ["inf.jpg", "nan.jpg", "a.jpg", "b.jpg"]
        embeddings = np.array(
            [[np.inf, 1.0, 0.0, 0.0], [np.nan, 0.0, 1.0, 0.0], [1.0, 0.0, 0.0, 0.0], [1.0, 0.0, 0.0, 0.0]],
            dtype=np.float32,
        )
        with warnings.catch_warnings():
            warnings.simplefilter("error", RuntimeWarning)
            groups = duplicates.near_duplicate_groups(paths, embeddings)
        self.assertEqual(groups, [["a.jpg", "b.jpg"]], "the genuine pair was not grouped")
        grouped = {path for group in groups for path in group}
        self.assertFalse({"inf.jpg", "nan.jpg"} & grouped, "a non-finite row was grouped")

    def test_a_zero_vector_is_never_anybodys_duplicate(self) -> None:
        """It has no direction, so it is not similar to anything, including itself."""
        rng = np.random.default_rng(13)
        paths, embeddings, _ = self._folder([2], rng)
        paths = [*paths, "blank_a.jpg", "blank_b.jpg"]
        embeddings = np.vstack([embeddings, np.zeros((2, embeddings.shape[1]), dtype=np.float32)])
        groups = duplicates.near_duplicate_groups(paths, embeddings)
        grouped = {path for group in groups for path in group}
        self.assertNotIn("blank_a.jpg", grouped)
        self.assertNotIn("blank_b.jpg", grouped)

    def test_mismatched_inputs_are_refused_rather_than_guessed_at(self) -> None:
        rng = np.random.default_rng(14)
        paths, embeddings, _ = self._folder([2], rng)
        self.assertEqual(duplicates.near_duplicate_groups(paths[:1], embeddings), [])
        self.assertEqual(duplicates.near_duplicate_groups([], np.empty((0, 4), dtype=np.float32)), [])

    def test_the_candidate_pairs_are_capped_rather_than_left_to_grow(self) -> None:
        """Bucket size is bounded; the total was not, and it grows quadratically.

        A folder of one subject buckets far more unevenly than random vectors, so the
        pair set — the peak memory of the whole pass — could reach tens of millions of
        entries. Past the ceiling the grouping degrades by finding fewer bursts, which
        is the right way for it to fail.
        """
        rng = np.random.default_rng(16)
        paths, embeddings, shot_of = self._folder([4] * 20, rng)
        capped = duplicates.near_duplicate_groups(paths, embeddings, max_pairs=1)
        # Whatever survives the cap must still be correct, never a wrong merge.
        for group in capped:
            self.assertEqual(len({shot_of[path] for path in group}), 1)
        full = duplicates.near_duplicate_groups(paths, embeddings)
        self.assertGreaterEqual(len(full), len(capped))
        self.assertEqual(sum(len(group) for group in full), 80)

    def test_grouping_is_transitive_across_a_long_burst(self) -> None:
        """Frame 1 and frame 8 may not pair directly; the union-find still joins them."""
        rng = np.random.default_rng(15)
        base = rng.normal(size=64).astype(np.float32)
        base /= np.linalg.norm(base)
        frames = self._burst(base, 8, rng)
        paths = [f"burst_{i}.jpg" for i in range(8)]
        groups = duplicates.near_duplicate_groups(paths, np.vstack(frames).astype(np.float32))
        self.assertEqual(len(groups), 1)
        self.assertEqual(len(groups[0]), 8)


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

    def test_clearing_the_cache_never_touches_what_it_promises_to_keep(self) -> None:
        """It used to carry them through memory, and either end could lose them.

        The old path read each irreplaceable file, rmtree'd the whole directory, then
        wrote them back. A read failure was logged at WARNING and the file was deleted
        anyway — routine on Windows, where antivirus and OneDrive hold files open — and
        a failed write-back afterwards had nothing left to restore from. Measured on
        500 labels, either fault left zero behind while notes.json survived, so the
        loss was silent and partial. Nothing is read or moved now, so there is no
        window at all: this asserts the files are not so much as opened.
        """
        guarded = {
            self.store.labels_path.name,
            self.store.notes_path.name,
            self.store.config_override_path.name,
            self.store.review_session_path.name,
        }
        touched: list[str] = []
        real_unlink, real_read, real_rename = Path.unlink, Path.read_bytes, Path.rename

        def watch(real, verb):
            def guard(target, *args, **kwargs):
                if target.name in guarded:
                    touched.append(f"{verb} {target.name}")
                return real(target, *args, **kwargs)

            return guard

        Path.unlink = watch(real_unlink, "deleted")  # type: ignore[method-assign]
        Path.read_bytes = watch(real_read, "read")  # type: ignore[method-assign]
        Path.rename = watch(real_rename, "moved")  # type: ignore[method-assign]
        try:
            self.store.clear_cache(keep_decisions=True)
        finally:
            Path.unlink = real_unlink  # type: ignore[method-assign]
            Path.read_bytes = real_read  # type: ignore[method-assign]
            Path.rename = real_rename  # type: ignore[method-assign]
        self.assertEqual(touched, [], f"clear_cache put the irreplaceable files at risk: {touched}")
        self.assertEqual(self.store.load_labels(), {self.paths[0]: 1, self.paths[1]: 0})
        self.assertEqual(self.store.load_notes(), {self.paths[0]: "keep this one"})

    def test_an_undeletable_recomputable_file_does_not_abort_the_clear(self) -> None:
        """One stubborn file must not leave the rest of the cache behind."""
        stubborn = self.store.cache_dir / "scan_metadata.json"
        stubborn.write_text("{}", encoding="utf-8")
        doomed = self.store.cache_dir / "face_counts.json"
        doomed.write_text("{}", encoding="utf-8")
        real_unlink = Path.unlink

        def flaky(target, *args, **kwargs):
            if target.name == stubborn.name:
                raise PermissionError(32, "The process cannot access the file")
            return real_unlink(target, *args, **kwargs)

        Path.unlink = flaky  # type: ignore[method-assign]
        try:
            self.store.clear_cache(keep_decisions=True)
        finally:
            Path.unlink = real_unlink  # type: ignore[method-assign]
        self.assertFalse(doomed.exists(), "the clear stopped at the first failure")
        self.assertEqual(self.store.load_labels(), {self.paths[0]: 1, self.paths[1]: 0})

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
        raw = working.read_bytes()
        self.assertIn(b"http://ns.adobe.com/xap/1.0/", raw, "the XMP packet was not written")
        self.assertIn("bikini".encode("utf-16le"), raw, "the EXIF keyword was not written")

    def test_tagging_a_jpeg_does_not_re_encode_it(self) -> None:
        """Writing a keyword must annotate the photo, not recompress it.

        This used to decode the image, apply its EXIF orientation and save it again at
        Pillow's default JPEG quality, so an action described as "write keyword tags"
        permanently degraded every original it touched.
        """
        working = Path(tempfile.mkdtemp(prefix="bikini_meta_lossless_")) / "tagged.jpg"
        shutil.copyfile(self.sources[0], working)
        with Image.open(working) as before:
            before_pixels = np.asarray(before.convert("RGB"))
            before_size = before.size
        self.assertTrue(output_ops.write_image_metadata(working, "bikini", 0.87))
        with Image.open(working) as after:
            after_pixels = np.asarray(after.convert("RGB"))
            self.assertEqual(after.size, before_size, "the frame was rotated or resized")
        self.assertTrue(
            np.array_equal(before_pixels, after_pixels), "the photo was re-encoded rather than tagged"
        )

    def test_tagging_a_png_keeps_its_existing_text_chunks(self) -> None:
        from PIL import PngImagePlugin

        working = Path(tempfile.mkdtemp(prefix="bikini_meta_png_")) / "tagged.png"
        info = PngImagePlugin.PngInfo()
        info.add_text("Author", "someone else")
        Image.new("RGB", (32, 24), color=(10, 200, 30)).save(working, pnginfo=info)
        with Image.open(working) as before:
            before_pixels = np.asarray(before.convert("RGB"))
        self.assertTrue(output_ops.write_image_metadata(working, "bikini", 0.5))
        with Image.open(working) as after:
            after_pixels = np.asarray(after.convert("RGB"))
            text = dict(after.text)
        # PNG is lossless, so a re-save costs nothing — but it must not silently drop
        # whatever else the file was annotated with.
        self.assertTrue(np.array_equal(before_pixels, after_pixels))
        self.assertEqual(text.get("Author"), "someone else")
        self.assertEqual(text.get("Keywords"), "bikini")

    def test_an_unsupported_format_is_left_alone_rather_than_recompressed(self) -> None:
        """WebP cannot be tagged losslessly here, so it must not be tagged at all."""
        working = Path(tempfile.mkdtemp(prefix="bikini_meta_webp_")) / "tagged.webp"
        Image.new("RGB", (32, 24), color=(200, 40, 40)).save(working, quality=40)
        original = working.read_bytes()
        try:
            import pyexiv2  # noqa: F401
        except ImportError:
            self.assertFalse(output_ops.write_image_metadata(working, "bikini", 0.5))
            self.assertEqual(working.read_bytes(), original, "the file was rewritten anyway")
        else:
            self.skipTest("pyexiv2 is installed, so WebP is tagged in place")


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


class UpdateCheckTrust(unittest.TestCase):
    """The update URL is fetched automatically, so it gets the same guard as the VLM."""

    def test_only_http_urls_are_ever_fetched(self) -> None:
        """urlopen also speaks file: and ftp:, and neither is a release manifest."""
        import bikini_scanner.update_checker as update_checker

        opened: list[str] = []
        original = update_checker.urlopen
        update_checker.urlopen = lambda *a, **k: opened.append("fetched")  # type: ignore[assignment]
        try:
            for url in (
                "file:///C:/Windows/win.ini",
                "ftp://example.invalid/latest.json",
                "javascript:alert(1)",
                "not-a-url",
                "http://",
                "   ",
            ):
                self.assertIsNone(update_checker.check_for_update(url), f"{url!r} was accepted")
        finally:
            update_checker.urlopen = original
        self.assertEqual(opened, [], "a non-http(s) update URL was actually fetched")

    def test_an_ordinary_https_url_is_still_allowed_through(self) -> None:
        import bikini_scanner.update_checker as update_checker

        reached: list[str] = []

        def fake_urlopen(request, timeout=None):
            reached.append(request.full_url)
            raise OSError("no network in tests")

        original = update_checker.urlopen
        update_checker.urlopen = fake_urlopen  # type: ignore[assignment]
        try:
            self.assertIsNone(update_checker.check_for_update("https://example.invalid/latest.json"))
        finally:
            update_checker.urlopen = original
        self.assertEqual(reached, ["https://example.invalid/latest.json"])


class SQLiteReadsStayLocked(unittest.TestCase):
    """The cursor-outside-the-lock hazard must not be reintroduced."""

    def test_there_is_no_execute_that_hands_back_a_cursor(self) -> None:
        """It had no callers left, but it sat there inviting the original bug back.

        Returning a cursor after releasing the lock is what returned truncated blobs
        (np.load raising "No data left in file") and dropped writes, because the scan
        worker writes while the main thread reads on the same connection.
        """
        from bikini_scanner.sqlite_cache import SQLiteCache

        self.assertFalse(
            hasattr(SQLiteCache, "_execute"),
            "SQLiteCache._execute is back; reads must go through _fetchall/_fetchone",
        )
        for name in ("_fetchall", "_fetchone"):
            self.assertTrue(hasattr(SQLiteCache, name), f"{name} is missing")


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


class RegionCacheAlignment(unittest.TestCase):
    """An image the scan could not hash must not shift anyone else's cache entries."""

    def setUp(self) -> None:
        self.config = ScannerConfig(deep_scan="always", enable_face_detection=False)
        self.backend = _build_backend(self.config)
        self.scorer = BikiniScorer(self.backend, self.config)
        self.folder = Path(tempfile.mkdtemp(prefix="bikini_regioncache_"))
        self.paths = [str(path) for path in _make_images(self.folder, count=3)]
        self.hashes = [content_hash_for_path(path) for path in self.paths]

    def tearDown(self) -> None:
        shutil.rmtree(self.folder, ignore_errors=True)

    def _cached_regions(self, content_hashes: list[str | None]) -> dict[str, dict[str, np.ndarray]]:
        cache_dir = Path(tempfile.mkdtemp(prefix="bikini_regionstore_"))
        store = FolderStore(cache_dir)
        embeddings = np.vstack(
            [self.backend.embed_pil_images([Image.open(path).convert("RGB")])[0] for path in self.paths]
        ).astype(np.float32)
        scorer_module.run_deep_pass(
            self.backend,
            self.scorer,
            store,
            self.paths,
            embeddings,
            content_hashes=content_hashes,
        )
        namespace = scorer_module._region_namespace(self.config)
        saved = {
            str(content_hash): store.lookup_region_embeddings(str(content_hash), namespace)
            for content_hash in self.hashes
            if content_hash
        }
        shutil.rmtree(cache_dir, ignore_errors=True)
        return saved

    def test_an_unhashable_image_does_not_corrupt_the_others(self) -> None:
        """pending_meta used to be filtered while pending_crops was not.

        The two were zipped against the embedding vectors, so one image with no content
        hash shifted the alignment and every later crop was written to the cache under
        another image's key — persisted, and reused by every later scan.
        """
        everything = self._cached_regions(list(self.hashes))
        with_a_gap = self._cached_regions([None, *self.hashes[1:]])
        self.assertTrue(everything, "the deep pass cached nothing to compare")
        # The first image is unhashable in the second run, so it is expected to be
        # absent. Every other image must be cached exactly as it was before.
        for content_hash in self.hashes[1:]:
            assert content_hash is not None
            expected = everything[str(content_hash)]
            actual = with_a_gap[str(content_hash)]
            self.assertEqual(sorted(expected), sorted(actual), f"{content_hash} lost or gained regions")
            for key, vector in expected.items():
                self.assertTrue(
                    np.allclose(vector, actual[key]),
                    f"{content_hash}/{key} was cached with another image's embedding",
                )

    def test_the_decoded_frames_are_only_retained_for_the_vlm_pass(self) -> None:
        """They are full-resolution images; holding one per candidate is the whole scan."""
        embeddings = np.vstack(
            [self.backend.embed_pil_images([Image.open(path).convert("RGB")])[0] for path in self.paths]
        ).astype(np.float32)

        def deep_pass(config: ScannerConfig) -> scorer_module.DeepPassResult:
            # A fresh store each time: a warm region cache skips the decode entirely,
            # which is exactly the path that populates decoded_images.
            cache_dir = Path(tempfile.mkdtemp(prefix="bikini_decoded_"))
            try:
                return scorer_module.run_deep_pass(
                    self.backend,
                    BikiniScorer(self.backend, config),
                    FolderStore(cache_dir),
                    self.paths,
                    embeddings,
                    content_hashes=list(self.hashes),
                )
            finally:
                shutil.rmtree(cache_dir, ignore_errors=True)

        without_vlm = deep_pass(ScannerConfig(deep_scan="always", enable_face_detection=False))
        self.assertEqual(without_vlm.decoded_images, {}, "frames retained with the VLM pass switched off")
        with_vlm = deep_pass(
            ScannerConfig(deep_scan="always", enable_face_detection=False, vlm_enabled=True)
        )
        self.assertTrue(with_vlm.decoded_images, "the VLM pass needs the frames it was going to judge")


class SQLiteParameterLimits(unittest.TestCase):
    """SQLite caps bound parameters per statement; a big folder must not trip it."""

    def test_the_chunk_stays_under_the_oldest_sqlite_limit(self) -> None:
        from bikini_scanner.sqlite_cache import SQLiteCache

        # 999 is the limit on pre-3.32 builds. Anything at or above it fails there with
        # "too many SQL variables" before a single image has been read.
        self.assertLess(SQLiteCache._QUERY_CHUNK, 999)

    def test_more_paths_than_one_chunk_are_all_returned(self) -> None:
        from bikini_scanner.sqlite_cache import SQLiteCache

        folder = Path(tempfile.mkdtemp(prefix="bikini_sqlchunk_"))
        cache = SQLiteCache(folder / "cache.db")
        count = SQLiteCache._QUERY_CHUNK * 2 + 7
        paths: list[Path] = []
        records: dict[Path, dict[str, int | str]] = {}
        embeddings: dict[str, np.ndarray] = {}
        for index in range(count):
            path = folder / f"image_{index:05d}.jpg"
            path.write_bytes(b"x" * (index % 7 + 1))
            stat = path.stat()
            content_hash = f"hash{index:05d}"
            paths.append(path)
            embeddings[content_hash] = np.full(4, index, dtype=np.float32)
            records[path] = {"content_hash": content_hash, "mtime_ns": stat.st_mtime_ns, "size": stat.st_size}
        cache.save_scan_cache(embeddings, records)
        found = cache.get_cached_image_records(paths)
        self.assertEqual(len(found), count, "the chunked IN clause lost rows at a boundary")
        # Spot-check that each path kept its own embedding across the chunk boundaries.
        for index in (0, SQLiteCache._QUERY_CHUNK - 1, SQLiteCache._QUERY_CHUNK, count - 1):
            record = found[paths[index]]
            self.assertEqual(record["content_hash"], f"hash{index:05d}")
            self.assertTrue(np.allclose(record["embedding"], float(index)))  # type: ignore[arg-type]
        cache.close()
        shutil.rmtree(folder, ignore_errors=True)


class SettingsPersistence(unittest.TestCase):
    """Settings changed in the app have to still be there next time it opens."""

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

    def test_the_scanner_settings_are_written_to_the_preferences_file(self) -> None:
        from bikini_scanner.user_prefs import load_user_prefs

        self.app.global_config.model_name = "openai/clip-vit-large-patch14"
        self.app.global_config.deep_scan = "always"
        self.app.global_config.minor_threshold = 0.22
        self.app._save_user_prefs()
        stored = load_user_prefs().get("scanner_config")
        self.assertIsInstance(stored, dict, "nothing from Tools > Settings was persisted")
        restored = ScannerConfig.from_mapping(stored)
        # The whole config round-trips, not just the few fields touched above.
        self.assertEqual(restored.to_dict(), self.app.global_config.to_dict())

    def test_a_folder_override_is_not_promoted_into_the_global_settings(self) -> None:
        from bikini_scanner.user_prefs import load_user_prefs

        self.app.global_config.threshold = 0.4
        self.app.config.threshold = 0.9  # as a folder override would leave it
        self.app.folder_override_active = True
        self.app._save_user_prefs()
        restored = ScannerConfig.from_mapping(load_user_prefs().get("scanner_config"))
        self.assertAlmostEqual(restored.threshold, 0.4)

    def test_the_vlm_api_key_is_not_written_to_the_preferences_file(self) -> None:
        from bikini_scanner.user_prefs import load_user_prefs, prefs_path

        self.app.global_config.vlm_api_key = "sk-SECRET-do-not-store"
        self.app._save_user_prefs()
        self.assertNotIn("sk-SECRET-do-not-store", prefs_path().read_text(encoding="utf-8"))
        restored = ScannerConfig.from_mapping(load_user_prefs().get("scanner_config"))
        self.assertEqual(restored.vlm_api_key, "")

    def test_the_number_beside_the_slider_follows_the_slider(self) -> None:
        # Tk only fires a Scale's command for user interaction, so a programmatic set
        # used to leave the entry showing the previous value after Save, Import or a
        # profile change.
        self.app._set_threshold(0.72)
        self.assertAlmostEqual(float(self.app.threshold_var.get()), 0.72)
        self.assertEqual(self.app.threshold_text_var.get(), "0.72")


class ScanFailureRecovery(unittest.TestCase):
    """A folder that fails must not take the queue, or every later retrain, with it."""

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
        self.errors: list[str] = []
        gui_module.messagebox.showerror = lambda title, message, **kwargs: self.errors.append(str(message))
        self.started: list[str] = []
        self.app._start_next_queue_item = lambda: self.started.append("next")  # type: ignore[method-assign]

    def tearDown(self) -> None:
        self.app._closing = True
        self.root.destroy()

    def test_a_failed_folder_moves_the_queue_on(self) -> None:
        self.app.scan_queue = ["/one", "/two"]
        self.app.queue_active = True
        self.app.queue_index = 0
        self.app._scan_failed(RuntimeError("boom"), self.app._refresh_generation)
        self.root.update()
        self.assertEqual(self.app.queue_index, 1)
        self.assertTrue(self.app.queue_active, "there was another folder to scan")
        self.assertEqual(self.started, ["next"])
        self.assertIn("1 left in the queue", self.errors[0])

    def test_the_last_folder_failing_ends_the_queue(self) -> None:
        self.app.scan_queue = ["/only"]
        self.app.queue_active = True
        self.app.queue_index = 0
        self.app._scan_failed(RuntimeError("boom"), self.app._refresh_generation)
        self.root.update()
        # queue_active left set was what stalled everything: run_queue refused to start
        # again, and _flush_retrain returns early while a queue is running, so no label
        # was ever folded into the model for the rest of the session.
        self.assertFalse(self.app.queue_active)
        self.assertEqual(self.started, [])


class BrowseModeIsModelFree(unittest.TestCase):
    """Browsing without scanning must never pull the model in behind the user's back."""

    def setUp(self) -> None:
        tkinter = __import__("tkinter")
        try:
            self.root = tkinter.Tk()
        except Exception as exc:  # noqa: BLE001
            self.skipTest(f"no display available: {exc}")
        from bikini_scanner import gui as gui_module

        for name in ("_resume_last_folder_if_any", "_maybe_preload_backend", "_maybe_show_first_run_guide"):
            setattr(gui_module.BikiniScannerApp, name, lambda self: None)
        self.folder = Path(tempfile.mkdtemp(prefix="bikini_browse_"))
        _make_images(self.folder, count=4)
        self.app = gui_module.BikiniScannerApp(self.root, config=ScannerConfig(preload_backend=False))
        self.app.label_filter_var.set("all")

    def tearDown(self) -> None:
        self.app._closing = True
        self.root.destroy()
        shutil.rmtree(self.folder, ignore_errors=True)

    def test_a_decision_does_not_queue_a_retrain(self) -> None:
        """The state here has no embeddings, so a re-rank has nothing to work from.

        Queueing one anyway loaded the model on the main thread — a ~600 MB download on
        a cold install — and then failed on the zero-width embeddings, so the first
        Accept in a model-free browse ended in a "Scan failed" dialog.
        """
        self.app.browse_without_scanning(str(self.folder))
        self.root.update()
        target = str(self.app.page_samples[0]["path"])
        self.app.set_label(target, 1)
        self.root.update()
        assert self.app.store is not None
        self.assertEqual(self.app.store.load_labels().get(target), 1, "the decision must still be saved")
        self.assertFalse(self.app._retrain_pending)
        self.assertIsNone(self.app._retrain_after_id)
        self.assertIsNone(self.app.scorer, "browsing loaded a model it was supposed to avoid")


class GridArrowNavigation(unittest.TestCase):
    """Up and Down move by a row, including on the default 'fit to window' setting."""

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
        self.app._apply_focus_visuals = lambda: None  # type: ignore[method-assign]
        self.app.page_samples = [{"path": f"/photo_{index}.jpg", "score": 0.5} for index in range(9)]

    def tearDown(self) -> None:
        self.app._closing = True
        self.root.destroy()

    def test_down_steps_a_whole_row_on_auto_columns(self) -> None:
        # columns_var is 0 for "fit to window", which is the default. Reading it
        # directly gave a column count of 1, so Down behaved exactly like Right.
        self.app.columns_var.set(0)
        self.app._grid_columns = lambda: 3  # type: ignore[method-assign]
        self.app.focused_path = "/photo_0.jpg"
        self.app.move_focus_grid(1, 0)
        self.assertEqual(self.app.focused_path, "/photo_3.jpg")
        self.app.move_focus_grid(-1, 0)
        self.assertEqual(self.app.focused_path, "/photo_0.jpg")

    def test_an_explicit_column_count_is_still_honoured(self) -> None:
        self.app.columns_var.set(4)
        self.app.focused_path = "/photo_0.jpg"
        self.app.move_focus_grid(1, 0)
        self.assertEqual(self.app.focused_path, "/photo_4.jpg")


class LiveStoreModelSwitch(unittest.TestCase):
    """Changing the model mid-session must not break the folder's cache database.

    The GUI only builds a new FolderStore when the *folder* changes. Changing the model
    in Settings clears the backend and the scorer and keeps the store, so the next scan
    calls ensure_embedding_namespace on a store whose SQLite connection is already open.
    """

    def setUp(self) -> None:
        self.folder = Path(tempfile.mkdtemp(prefix="bikini_liveswitch_"))
        self.addCleanup(shutil.rmtree, self.folder, True)
        self.image = self.folder / "one.jpg"
        self.image.write_bytes(_make_image_bytes())

    def test_the_cache_survives_a_switch_on_a_live_store(self) -> None:
        store = FolderStore(self.folder)  # the one instance the GUI keeps for this folder
        store.ensure_embedding_namespace("clip_torch-model_a")
        store.save_embeddings({self.image: np.ones(512, dtype=np.float32)})
        self.assertEqual(len(store.get_cached_image_records([self.image])), 1)

        # clear() used to delete cache.db and leave the object pointing at a file that
        # _ensure_tables had never run against, so this raised
        # "no such table: image_records" and every later query in the session did too.
        store.ensure_embedding_namespace("clip_torch-model_b")
        self.assertEqual(store.get_cached_image_records([self.image]), {})

        # And it must still be usable: the scan that triggered the switch carries on.
        store.save_embeddings({self.image: np.ones(512, dtype=np.float32)})
        self.assertEqual(len(store.get_cached_image_records([self.image])), 1)
        self.assertEqual(len(FolderStore(self.folder).get_cached_image_records([self.image])), 1)

    def test_a_full_scan_survives_a_model_switch(self) -> None:
        shared = _shared()
        backend = shared["backend"]
        store = FolderStore(self.folder)
        for model in ("model-a", "model-b", "model-a"):
            config = ScannerConfig(model_name=model, preload_backend=False)
            state, _ = scan_and_score_folder(backend, store, BikiniScorer(backend, config), threshold=0.35)
            self.assertEqual(len(state.paths), 1, f"scan with {model} produced no results")


class RegionCacheNamespace(unittest.TestCase):
    def test_max_faces_is_part_of_the_crop_layout_identity(self) -> None:
        """Otherwise lowering it silently reuses the extra subjects' cached crops.

        max_faces is also a key a folder override may set, so a folder that had already
        been scanned would ignore the new value entirely.
        """
        one = scorer_module._region_namespace(ScannerConfig(max_faces=1))
        three = scorer_module._region_namespace(ScannerConfig(max_faces=3))
        self.assertNotEqual(one, three)

    def test_the_model_still_separates_namespaces(self) -> None:
        a = scorer_module._region_namespace(ScannerConfig(model_name="a", max_faces=2))
        b = scorer_module._region_namespace(ScannerConfig(model_name="b", max_faces=2))
        self.assertNotEqual(a, b)


class VLMUnusableVerdicts(unittest.TestCase):
    """A reply that names none of the axes is not a judgment of zero."""

    def test_a_reply_with_no_axes_is_refused_by_the_parser(self) -> None:
        self.assertEqual(parse_axis_json('{"nudity": 0.9, "note": "high"}'), {})

    def test_an_empty_verdict_is_not_scored_as_a_confident_zero(self) -> None:
        # Every axis defaulting to 0.5 is "no evidence either way", which the cascade
        # scores 0.0 - and that used to be blended in at vlm_weight, pushing a genuine
        # match down on the strength of a reply the model never made.
        from bikini_scanner.regions import KIND_FULL
        from bikini_scanner.vlm_backend import VLM_AXES

        neutral = np.asarray([[0.5] * len(VLM_AXES)], dtype=np.float32)
        table = cascade.RegionScoreTable(
            owner=np.array([0], dtype=np.int64),
            kinds=np.array([KIND_FULL], dtype=object),
            axis_scores={axis: neutral[:, i] for i, axis in enumerate(VLM_AXES)},
            image_count=1,
            full_row=np.array([0], dtype=np.int64),
        )
        self.assertAlmostEqual(float(cascade.evaluate(table, ScannerConfig(), None).score[0]), 0.0)

    def test_the_client_returns_none_for_an_axis_free_reply(self) -> None:
        class Handler(BaseHTTPRequestHandler):
            def do_GET(self) -> None:
                self.send_response(200)
                self.end_headers()
                self.wfile.write(b'{"data": []}')

            def do_POST(self) -> None:
                self.send_response(200)
                self.end_headers()
                self.wfile.write(b'{"choices":[{"message":{"content":"{\\"unrelated\\": 1}"}}]}')

            def log_message(self, *args: object) -> None:
                pass

        server = ThreadingHTTPServer(("127.0.0.1", 0), Handler)
        threading.Thread(target=server.serve_forever, daemon=True).start()
        self.addCleanup(server.shutdown)
        client = VLMClient(f"http://127.0.0.1:{server.server_address[1]}/v1", "m", timeout=5.0, concurrency=1)
        results = client.score_images([[Image.new("RGB", (32, 32))]])
        # None, not {}: the caller skips None and would have cached and scored {}.
        self.assertEqual(results, [None])


class VLMCancellationIsPrompt(unittest.TestCase):
    def test_stop_does_not_wait_out_the_request_timeout(self) -> None:
        """Every worker sitting in a slow request must not hide the cancel flag.

        as_completed only comes back when a request finishes, so Stop used to do nothing
        for a full timeout period - a minute on the default setting - with the button
        already greyed out.
        """

        class SlowHandler(BaseHTTPRequestHandler):
            def do_GET(self) -> None:
                self.send_response(200)
                self.end_headers()
                self.wfile.write(b'{"data": []}')

            def do_POST(self) -> None:
                time.sleep(20)

            def log_message(self, *args: object) -> None:
                pass

        server = ThreadingHTTPServer(("127.0.0.1", 0), SlowHandler)
        threading.Thread(target=server.serve_forever, daemon=True).start()
        self.addCleanup(server.shutdown)
        client = VLMClient(f"http://127.0.0.1:{server.server_address[1]}/v1", "m", timeout=20.0, concurrency=2)
        cancel = threading.Event()
        threading.Timer(0.5, cancel.set).start()
        started = time.monotonic()
        with self.assertRaises(VLMCancelled):
            client.score_images([[Image.new("RGB", (32, 32))] for _ in range(4)], cancel_event=cancel)
        elapsed = time.monotonic() - started
        self.assertLess(elapsed, 8.0, f"cancelling took {elapsed:.1f}s")


class BoundedBackendCache(unittest.TestCase):
    """A CLIP model is 600 MB and the large one 1.7 GB; the cache has to have a lid."""

    def test_the_least_recently_used_backend_is_evicted(self) -> None:
        from collections import OrderedDict

        from bikini_scanner.backend_utils import remember_bounded

        cache: OrderedDict[str, str] = OrderedDict()
        for name in ("a", "b", "c"):
            remember_bounded(cache, name, f"model-{name}", limit=2)
        self.assertEqual(list(cache), ["b", "c"], "the cache grew past its limit")

    def test_a_hit_keeps_a_backend_alive(self) -> None:
        from collections import OrderedDict

        from bikini_scanner.backend_utils import remember_bounded

        cache: OrderedDict[str, str] = OrderedDict()
        remember_bounded(cache, "scan", "m1", limit=2)
        remember_bounded(cache, "refine", "m2", limit=2)
        cache.move_to_end("scan")  # what a cache hit does
        remember_bounded(cache, "third", "m3", limit=2)
        self.assertEqual(list(cache), ["scan", "third"], "the recently used backend was evicted")

    def test_the_limit_is_never_zero(self) -> None:
        from collections import OrderedDict

        from bikini_scanner.backend_utils import remember_bounded

        cache: OrderedDict[str, str] = OrderedDict()
        remember_bounded(cache, "only", "m", limit=0)
        self.assertEqual(list(cache), ["only"], "a zero limit would evict the model being loaded")


class CredentialsStayOutOfSharedFiles(unittest.TestCase):
    """A settings file or a profile is meant to be copied around; a bearer token is not."""

    def setUp(self) -> None:
        self.config = ScannerConfig()
        self.config.vlm_api_key = "sk-SECRET-do-not-store"
        self.config.vlm_base_url = "https://vision.example.com/v1"

    def test_a_saved_profile_omits_the_api_key(self) -> None:
        config_profiles.save_profile("secret carrier", self.config)
        self.addCleanup(config_profiles.delete_profile, "secret carrier")
        raw = config_profiles.profiles_path().read_text(encoding="utf-8")
        self.assertNotIn("sk-SECRET-do-not-store", raw)
        restored = config_profiles.profile_config("secret carrier")
        assert restored is not None
        self.assertEqual(restored.vlm_api_key, "")
        # Everything that is not a credential still round-trips.
        self.assertEqual(restored.vlm_base_url, "https://vision.example.com/v1")

    def test_without_secrets_leaves_the_rest_alone(self) -> None:
        stripped = config_profiles.without_secrets(self.config.to_dict())
        self.assertEqual(stripped["vlm_api_key"], "")
        self.assertEqual(stripped["vlm_base_url"], "https://vision.example.com/v1")
        self.assertEqual(stripped["threshold"], self.config.threshold)


class DetailWeightsMustScoreSomething(unittest.TestCase):
    def test_an_all_zero_mapping_is_refused(self) -> None:
        """It is not a weighting, it is an off switch, and a folder override may set it."""
        defaults = ScannerConfig().detail_weights
        config = ScannerConfig.from_mapping(
            {"detail_weights": dict.fromkeys(defaults, 0.0)}
        )
        self.assertEqual(config.detail_weights, defaults)

    def test_a_hostile_folder_override_cannot_switch_scoring_off(self) -> None:
        accepted, _refused = filter_folder_override({"detail_weights": {"bikini": 0, "cleavage": 0}})
        merged = ScannerConfig.from_mapping({**ScannerConfig().to_dict(), **accepted})
        self.assertTrue(any(weight > 0 for weight in merged.detail_weights.values()))

    def test_dropping_one_axis_is_still_allowed(self) -> None:
        config = ScannerConfig.from_mapping({"detail_weights": {"bikini": 1.0, "cleavage": 0.0}})
        self.assertEqual(config.detail_weights, {"bikini": 1.0, "cleavage": 0.0})

    def test_a_non_finite_weight_is_dropped(self) -> None:
        config = ScannerConfig.from_mapping({"detail_weights": {"bikini": float("nan"), "midriff": 0.5}})
        self.assertEqual(config.detail_weights, {"midriff": 0.5})


class PluginReturnValues(unittest.TestCase):
    """Everything downstream indexes sample["path"]; a plugin must not break that."""

    def setUp(self) -> None:
        from bikini_scanner import plugins

        self.plugins = plugins
        self.good = [{"path": "/a.jpg", "score": 0.5}, {"path": "/b.jpg", "score": 0.4}]

    def test_a_list_of_strings_is_rejected(self) -> None:
        self.assertIsNone(self.plugins._usable_samples(["/a.jpg", "/b.jpg"], "bad.py"))

    def test_dicts_without_a_path_are_rejected(self) -> None:
        self.assertIsNone(self.plugins._usable_samples([{"score": 0.5}], "bad.py"))

    def test_a_non_iterable_return_is_rejected(self) -> None:
        self.assertIsNone(self.plugins._usable_samples(42, "bad.py"))

    def test_none_means_leave_the_list_alone(self) -> None:
        self.assertIsNone(self.plugins._usable_samples(None, "quiet.py"))

    def test_a_well_formed_return_is_accepted(self) -> None:
        self.assertEqual(self.plugins._usable_samples(self.good, "good.py"), self.good)

    def test_a_bad_plugin_does_not_replace_the_samples(self) -> None:
        directory = self.plugins.plugins_dir()
        directory.mkdir(parents=True, exist_ok=True)
        script = directory / "breaks_the_grid.py"
        script.write_text("def process_results(state, samples):\n    return ['not a dict']\n", encoding="utf-8")
        self.addCleanup(script.unlink, True)
        result = self.plugins.apply_plugins(None, list(self.good), enabled=True)
        self.assertEqual(result, self.good, "the malformed return reached the grid")


class FaceModelLocationIsMemoised(unittest.TestCase):
    def setUp(self) -> None:
        vision_analysis.forget_model_location()
        self.addCleanup(vision_analysis.forget_model_location)

    def test_the_answer_is_reused_rather_than_re_stated(self) -> None:
        calls: list[int] = []
        original = vision_analysis._locate_model

        def counting() -> object:
            calls.append(1)
            return original()

        vision_analysis._locate_model = counting  # type: ignore[assignment]
        self.addCleanup(setattr, vision_analysis, "_locate_model", original)
        for _ in range(20):
            vision_analysis.resolve_model()
            vision_analysis.face_detection_available()
        self.assertEqual(len(calls), 1, f"the model path was looked up {len(calls)} times")

    def test_forgetting_re_checks(self) -> None:
        vision_analysis.resolve_model()
        vision_analysis.forget_model_location()
        original = vision_analysis._locate_model
        calls: list[int] = []

        def counting() -> object:
            calls.append(1)
            return original()

        vision_analysis._locate_model = counting  # type: ignore[assignment]
        self.addCleanup(setattr, vision_analysis, "_locate_model", original)
        vision_analysis.resolve_model()
        self.assertEqual(len(calls), 1)


class ClearCacheKeepsYourPlace(unittest.TestCase):
    def test_the_review_session_survives_clearing_the_cache(self) -> None:
        """_discard_derived_caches already keeps it; the two paths must not disagree."""
        folder = Path(tempfile.mkdtemp(prefix="bikini_place_"))
        self.addCleanup(shutil.rmtree, folder, True)
        (folder / "one.jpg").write_bytes(_make_image_bytes())
        store = FolderStore(folder)
        store.save_labels({str(folder / "one.jpg"): 1})
        store.save_review_session({"page_index": 7, "view_mode": "detected"})
        store.clear_cache(keep_decisions=True)
        restored = store.load_review_session()
        self.assertIsNotNone(restored)
        assert restored is not None
        self.assertEqual(restored.get("page_index"), 7)
        self.assertEqual(store.load_labels().get(str(folder / "one.jpg")), 1)


class ReportNamesWhatItLeftOut(unittest.TestCase):
    def test_missing_images_are_counted_in_the_report(self) -> None:
        folder = Path(tempfile.mkdtemp(prefix="bikini_report_missing_"))
        self.addCleanup(shutil.rmtree, folder, True)
        present = folder / "here.jpg"
        present.write_bytes(_make_image_bytes())
        gone = folder / "moved_after_the_scan.jpg"
        destination = folder / "report.html"
        output_ops.build_html_report(
            destination,
            [
                {"path": str(present), "score": 0.9, "bucket": "Bikini"},
                {"path": str(gone), "score": 0.8, "bucket": "Bikini"},
            ],
            {},
            {},
        )
        text = destination.read_text(encoding="utf-8")
        self.assertIn("no longer exist", text)
        self.assertIn("1 image(s)", text)

    def test_a_complete_report_says_nothing_about_missing_files(self) -> None:
        folder = Path(tempfile.mkdtemp(prefix="bikini_report_ok_"))
        self.addCleanup(shutil.rmtree, folder, True)
        present = folder / "here.jpg"
        present.write_bytes(_make_image_bytes())
        destination = folder / "report.html"
        output_ops.build_html_report(
            destination, [{"path": str(present), "score": 0.9, "bucket": "Bikini"}], {}, {}
        )
        self.assertNotIn("no longer exist", destination.read_text(encoding="utf-8"))


class LogFileStaysReadable(unittest.TestCase):
    """The log viewer is offered to the user, so it must not be third-party INFO noise."""

    def setUp(self) -> None:
        from bikini_scanner import logging_setup

        self.filter = logging_setup._OwnInfoOthersWarnings()

    def _record(self, name: str, level: int) -> logging.LogRecord:
        return logging.LogRecord(name, level, __file__, 1, "msg", None, None)

    def test_our_own_info_is_kept(self) -> None:
        self.assertTrue(self.filter.filter(self._record("bikini_scanner.scorer", logging.INFO)))
        self.assertTrue(self.filter.filter(self._record("bikini_scanner", logging.INFO)))

    def test_third_party_info_is_dropped(self) -> None:
        for name in ("transformers.modeling_utils", "urllib3.connectionpool", "PIL.Image"):
            self.assertFalse(self.filter.filter(self._record(name, logging.INFO)))

    def test_third_party_warnings_and_errors_are_kept(self) -> None:
        self.assertTrue(self.filter.filter(self._record("torch", logging.WARNING)))
        self.assertTrue(self.filter.filter(self._record("torch", logging.ERROR)))

    def test_a_lookalike_package_is_not_treated_as_ours(self) -> None:
        self.assertFalse(self.filter.filter(self._record("bikini_scanner_plugin_x", logging.INFO)))


class VanishedFilesReadAsVanished(unittest.TestCase):
    def test_a_file_removed_mid_scan_is_not_listed_as_unreadable(self) -> None:
        """"[Errno 2] No such file" next to genuinely corrupt photos misleads about both."""
        shared = _shared()
        backend = shared["backend"]
        folder = Path(tempfile.mkdtemp(prefix="bikini_vanish_"))
        self.addCleanup(shutil.rmtree, folder, True)
        _make_images(folder, count=3)
        doomed = sorted(folder.glob("*.jpg"))[0]
        original = scorer_module.collect_image_paths

        def collect_then_delete(target: Path) -> list[Path]:
            paths = original(target)
            doomed.unlink()
            return paths

        scorer_module.collect_image_paths = collect_then_delete
        self.addCleanup(setattr, scorer_module, "collect_image_paths", original)
        store = FolderStore(folder)
        scan_and_score_folder(backend, store, BikiniScorer(backend, ScannerConfig()), threshold=0.35)
        metadata = json.loads(store.metadata_path.read_text(encoding="utf-8"))
        skipped = metadata["skipped"]
        self.assertEqual(len(skipped), 1)
        self.assertEqual(skipped[0]["error"], "file disappeared during the scan")


class UnagedSubjectsAreNotAdults(unittest.TestCase):
    """A subject with no face crop has an unknown age, not an adult one.

    plan_regions drops any crop that clamps below _MIN_CROP_PX, so a face under about
    28px yields waist and torso crops and no face crop at all. That subject is
    `present` but can never be `minor`, and counting them as evidence that an adult is
    present let a photo whose only readable subject was a child surface on the unaged
    subject's body crops, scoring exactly as if a confirmed adult were there.
    """

    NEUTRAL = 0.5

    def _table(self, rows: list[tuple[str, float, float, float]]) -> cascade.RegionScoreTable:
        """rows = (region key, child, adult, detail); row 0 is the full frame."""
        kinds = []
        for key, *_ in rows:
            for prefix, kind in (
                ("full", "full"), ("face", "face"), ("chest", "chest"),
                ("waist", "waist"), ("torso", "torso"),
            ):
                if key.startswith(prefix):
                    kinds.append(kind)
                    break
            else:
                kinds.append("band")
        child = np.asarray([r[1] for r in rows], dtype=np.float32)
        adult = np.asarray([r[2] for r in rows], dtype=np.float32)
        detail = np.asarray([r[3] for r in rows], dtype=np.float32)
        axes = {
            "child": child,
            "adult": adult,
            "person": np.full(len(rows), 0.9, dtype=np.float32),
            "female": np.full(len(rows), 0.9, dtype=np.float32),
            "nsfw": np.full(len(rows), 0.5, dtype=np.float32),
        }
        for axis in ("bikini", "cleavage", "midriff", "bikini_top", "bikini_bottom"):
            axes[axis] = detail
        return cascade.RegionScoreTable(
            owner=np.zeros(len(rows), dtype=np.int64),
            kinds=np.asarray(kinds, dtype=object),
            axis_scores=axes,
            image_count=1,
            full_row=np.asarray([0], dtype=np.int64),
            subject=np.asarray([regions.region_subject(r[0]) for r in rows], dtype=np.int64),
        )

    def _child_plus(self, second: list[tuple[str, float, float, float]]) -> cascade.CascadeResult:
        rows = [
            ("full", self.NEUTRAL, self.NEUTRAL, 0.95),
            ("face0", 0.99, 0.05, self.NEUTRAL),
            ("chest0", self.NEUTRAL, self.NEUTRAL, 0.95),
            *second,
        ]
        return cascade.evaluate(self._table(rows), ScannerConfig(), np.asarray([2], dtype=np.int32))

    def test_a_child_plus_an_unaged_subject_is_excluded(self) -> None:
        result = self._child_plus(
            [
                ("waist1", self.NEUTRAL, self.NEUTRAL, 0.95),
                ("torso1", self.NEUTRAL, self.NEUTRAL, 0.95),
            ]
        )
        self.assertEqual(result.stage[0], cascade.STAGE_MINOR)
        self.assertEqual(float(result.score[0]), 0.0)

    def test_a_child_plus_a_confirmed_adult_is_still_scored(self) -> None:
        """The whole point of per-subject reasoning; it must survive the fix."""
        result = self._child_plus(
            [
                ("face1", 0.05, 0.95, self.NEUTRAL),
                ("waist1", self.NEUTRAL, self.NEUTRAL, 0.95),
                ("torso1", self.NEUTRAL, self.NEUTRAL, 0.95),
            ]
        )
        self.assertEqual(result.stage[0], cascade.STAGE_SCORED)
        self.assertGreater(float(result.score[0]), 0.0)

    def test_unaged_and_adult_are_not_the_same_verdict(self) -> None:
        unaged = self._child_plus(
            [("waist1", self.NEUTRAL, self.NEUTRAL, 0.95), ("torso1", self.NEUTRAL, self.NEUTRAL, 0.95)]
        )
        adult = self._child_plus(
            [
                ("face1", 0.05, 0.95, self.NEUTRAL),
                ("waist1", self.NEUTRAL, self.NEUTRAL, 0.95),
                ("torso1", self.NEUTRAL, self.NEUTRAL, 0.95),
            ]
        )
        self.assertNotEqual(unaged.stage[0], adult.stage[0])

    def test_subjects_with_no_readable_face_fall_back_to_the_whole_frame_gate(self) -> None:
        """Not excluded outright - that would bin every distant subject in the folder."""
        table = self._table(
            [
                ("full", self.NEUTRAL, 0.95, 0.95),
                ("waist0", self.NEUTRAL, self.NEUTRAL, 0.95),
                ("torso0", self.NEUTRAL, self.NEUTRAL, 0.95),
            ]
        )
        analysis = cascade.analyse_subjects(table, ScannerConfig())
        assert analysis is not None
        self.assertTrue(bool(analysis.has_subjects[0]))
        self.assertFalse(bool(analysis.has_readable_subjects[0]))
        result = cascade.evaluate(table, ScannerConfig(), np.asarray([1], dtype=np.int32))
        self.assertEqual(result.stage[0], cascade.STAGE_SCORED)

    def test_the_same_frame_reading_as_a_child_is_still_gated(self) -> None:
        """The whole-frame fallback has to keep working, not just pass everything."""
        table = self._table(
            [
                ("full", 0.99, 0.05, 0.95),
                ("waist0", self.NEUTRAL, self.NEUTRAL, 0.95),
                ("torso0", self.NEUTRAL, self.NEUTRAL, 0.95),
            ]
        )
        result = cascade.evaluate(table, ScannerConfig(), np.asarray([1], dtype=np.int32))
        self.assertEqual(result.stage[0], cascade.STAGE_MINOR)

    def test_a_small_face_really_does_lose_its_face_crop(self) -> None:
        """The geometry this whole class is about, asserted directly."""
        from bikini_scanner.vision_analysis import FaceBox

        small = regions.plan_regions((1200, 900), [FaceBox(x=400, y=200, width=22, height=29)])
        keys = [r.key for r in small if r.key != regions.FULL_REGION]
        self.assertTrue(keys, "a small face produced no regions at all")
        self.assertFalse(any(k.startswith("face") for k in keys), keys)
        self.assertTrue(any(k.startswith(("waist", "torso")) for k in keys), keys)

        big = regions.plan_regions((1200, 900), [FaceBox(x=400, y=200, width=60, height=78)])
        self.assertTrue(any(r.key.startswith("face") for r in big))


class CacheIsThreadSafe(unittest.TestCase):
    """The scan worker writes while the main thread reads the same connection.

    Tools > Duplicate groups and Tools > Clear cached scan data are both reachable
    during a scan and both go through SQLiteCache, so this is not a theoretical race.
    """

    def setUp(self) -> None:
        from bikini_scanner.sqlite_cache import SQLiteCache

        self.folder = Path(tempfile.mkdtemp(prefix="bikini_threads_"))
        self.addCleanup(shutil.rmtree, self.folder, True)
        self.cache = SQLiteCache(self.folder / "cache.db")
        self.addCleanup(self.cache.close)

    def test_concurrent_writes_all_land_and_stay_intact(self) -> None:
        threads_count, per_thread = 6, 40
        errors: list[str] = []

        def writer(tid: int) -> None:
            try:
                for i in range(per_thread):
                    content_hash = f"t{tid}_{i}"
                    path = self.folder / f"t{tid}_{i}.jpg"
                    path.write_bytes(b"x" * (i % 5 + 1))
                    stat = path.stat()
                    self.cache.save_scan_cache(
                        {content_hash: np.full(8, tid, dtype=np.float32)},
                        {path: {"content_hash": content_hash, "mtime_ns": stat.st_mtime_ns, "size": stat.st_size}},
                        {content_hash: i % 3},
                    )
                    self.cache.lookup_content_embedding(content_hash)
                    self.cache.load_face_counts()
            except Exception:  # noqa: BLE001
                errors.append(traceback.format_exc())

        workers = [threading.Thread(target=writer, args=(t,)) for t in range(threads_count)]
        for worker in workers:
            worker.start()
        for worker in workers:
            worker.join()
        self.assertEqual(errors, [], errors[0] if errors else "")
        rows = self.cache._load_all_embeddings()
        # .fetchall() used to run outside the lock, so rows came back truncated
        # (np.load raising "No data left in file") and writes were dropped.
        self.assertEqual(len(rows), threads_count * per_thread)
        for key, vector in rows.items():
            self.assertEqual(float(vector[0]), float(key.split("_")[0][1:]), f"{key} holds another thread's vector")

    def test_clearing_under_concurrent_readers_never_loses_the_schema(self) -> None:
        self.cache.save_scan_cache({"h": np.ones(4, dtype=np.float32)}, {})
        errors: list[str] = []
        stop = threading.Event()

        def reader() -> None:
            try:
                while not stop.is_set():
                    self.cache.lookup_content_embedding("h")
                    self.cache.load_face_counts()
            except Exception:  # noqa: BLE001
                errors.append(traceback.format_exc())

        readers = [threading.Thread(target=reader) for _ in range(3)]
        for r in readers:
            r.start()
        try:
            for _ in range(5):
                self.cache.clear()
                self.cache.save_scan_cache({"h": np.ones(4, dtype=np.float32)}, {})
        finally:
            stop.set()
            for r in readers:
                r.join()
        # Rebuilding the schema outside the lock left a window where a reader could
        # connect to a database with no tables in it.
        self.assertEqual(errors, [], errors[0] if errors else "")


class GlobalStoreIsThreadSafe(unittest.TestCase):
    def test_reading_the_training_set_while_recording(self) -> None:
        """_load_index hands back the live cache dict, which record() mutates in place."""
        store = GlobalLearningStore(model_name="thread-probe")
        self.addCleanup(store.clear)
        folder = Path(tempfile.mkdtemp(prefix="bikini_globthreads_"))
        self.addCleanup(shutil.rmtree, folder, True)
        errors: list[str] = []

        def learner(tid: int) -> None:
            try:
                for i in range(10):
                    path = folder / f"g{tid}_{i}.jpg"
                    path.write_bytes(b"y")
                    store.record([(str(path), i % 2, np.full(6, tid, dtype=np.float32))], sequence=i)
                    store.training_set(expected_dim=6)
            except Exception:  # noqa: BLE001
                errors.append(traceback.format_exc())

        workers = [threading.Thread(target=learner, args=(t,)) for t in range(4)]
        for worker in workers:
            worker.start()
        for worker in workers:
            worker.join()
        self.assertEqual(errors, [], errors[0] if errors else "")
        self.assertEqual(store.stats()["total"], 4 * 10)


class AtomicWriteRetriesOnWindows(unittest.TestCase):
    def test_a_transient_permission_error_is_retried(self) -> None:
        """os.replace fails on Windows if anything has the destination open for a moment."""
        target = Path(tempfile.mkdtemp(prefix="bikini_retry_")) / "labels.json"
        self.addCleanup(shutil.rmtree, target.parent, True)
        real_replace = os.replace
        attempts: list[int] = []

        def flaky(src: object, dst: object) -> None:
            attempts.append(1)
            if len(attempts) < 3:
                raise PermissionError(5, "Access is denied")
            real_replace(src, dst)  # type: ignore[arg-type]

        safe_io.os.replace = flaky  # type: ignore[attr-defined]
        self.addCleanup(setattr, safe_io.os, "replace", real_replace)
        safe_io.atomic_write_json(target, {"kept": True})
        self.assertEqual(len(attempts), 3)
        self.assertEqual(json.loads(target.read_text(encoding="utf-8")), {"kept": True})

    def test_a_persistent_failure_still_raises(self) -> None:
        target = Path(tempfile.mkdtemp(prefix="bikini_retry_fail_")) / "labels.json"
        self.addCleanup(shutil.rmtree, target.parent, True)
        real_replace = os.replace

        def always_denied(src: object, dst: object) -> None:
            raise PermissionError(5, "Access is denied")

        safe_io.os.replace = always_denied  # type: ignore[attr-defined]
        self.addCleanup(setattr, safe_io.os, "replace", real_replace)
        with self.assertRaises(PermissionError):
            safe_io.atomic_write_json(target, {"lost": True})
        # The temp file must not be left lying beside the destination.
        self.assertEqual(list(target.parent.glob(".*tmp")), [])


class SourcesAreCleanUtf8(unittest.TestCase):
    def test_no_byte_order_marks_or_mojibake(self) -> None:
        """A stray BOM makes a file unparseable to tools that read it as plain UTF-8."""
        root = Path(__file__).resolve().parents[1]
        # Built rather than written literally: spelling the replacement character out
        # would put it in this very file and make the check fail on itself.
        replacement = chr(0xFFFD)
        damaged: list[str] = []
        for folder in ("bikini_scanner", "tests", "scripts"):
            for path in sorted((root / folder).rglob("*.py")):
                raw = path.read_bytes()
                if raw.startswith(b"\xef\xbb\xbf"):
                    damaged.append(f"{path.name}: BOM")
                if replacement in raw.decode("utf-8", errors="replace"):
                    damaged.append(f"{path.name}: mojibake")
        self.assertEqual(damaged, [])


class ParallelSequencesFailLoudly(unittest.TestCase):
    """The structural fix: a length mismatch must raise, not truncate.

    The core types are bundles of positionally-coupled sequences. Every consumer zips
    them together, so a short one used to quietly drop images, pair a path with another
    image's score, or - as happened in the region cache - write one image's embedding
    under another's key and persist it.
    """

    def test_every_zip_in_the_package_is_strict(self) -> None:
        package = Path(__file__).resolve().parents[1] / "bikini_scanner"
        lax: list[str] = []
        for path in sorted(package.rglob("*.py")):
            tree = ast.parse(path.read_text(encoding="utf-8"))
            for node in ast.walk(tree):
                if not (isinstance(node, ast.Call) and isinstance(node.func, ast.Name)):
                    continue
                if node.func.id != "zip":
                    continue
                strict = next((k for k in node.keywords if k.arg == "strict"), None)
                ok = (
                    strict is not None
                    and isinstance(strict.value, ast.Constant)
                    and strict.value.value is True
                )
                if not ok:
                    lax.append(f"{path.name}:{node.lineno}")
        self.assertEqual(lax, [], "zip() without strict=True silently truncates on mismatch")

    def test_score_state_rejects_a_short_array(self) -> None:
        with self.assertRaises(ValueError) as caught:
            ScoreState(
                paths=["a", "b", "c"],
                embeddings=np.zeros((3, 4), dtype=np.float32),
                zero_shot_scores=np.zeros(3, dtype=np.float32),
                scores=np.zeros(2, dtype=np.float32),  # one short
                axis_scores={},
                face_counts=None,
                classifier_trained=False,
                classifier_label_count=0,
            )
        self.assertIn("scores", str(caught.exception))

    def test_score_state_rejects_a_mismatched_axis(self) -> None:
        with self.assertRaises(ValueError) as caught:
            ScoreState(
                paths=["a", "b"],
                embeddings=np.zeros((2, 4), dtype=np.float32),
                zero_shot_scores=np.zeros(2, dtype=np.float32),
                scores=np.zeros(2, dtype=np.float32),
                axis_scores={"bikini": np.zeros(5, dtype=np.float32)},
                face_counts=None,
                classifier_trained=False,
                classifier_label_count=0,
            )
        self.assertIn("bikini", str(caught.exception))

    def test_score_state_still_allows_the_legitimate_shapes(self) -> None:
        """Optional fields absent, list fields empty, and the browse-mode zero-width case."""
        ScoreState(
            paths=["a", "b"],
            embeddings=np.zeros((2, 0), dtype=np.float32),  # browse mode: no embeddings
            zero_shot_scores=np.zeros(2, dtype=np.float32),
            scores=np.zeros(2, dtype=np.float32),
            axis_scores={},
            face_counts=None,
            classifier_trained=False,
            classifier_label_count=0,
            excluded=np.zeros(2, dtype=bool),
        )
        ScoreState(
            paths=[],
            embeddings=np.zeros((0, 512), dtype=np.float32),
            zero_shot_scores=np.zeros(0, dtype=np.float32),
            scores=np.zeros(0, dtype=np.float32),
            axis_scores={},
            face_counts=None,
            classifier_trained=False,
            classifier_label_count=0,
        )

    def test_region_table_rejects_mismatched_rows(self) -> None:
        with self.assertRaises(ValueError) as caught:
            cascade.RegionScoreTable(
                owner=np.zeros(3, dtype=np.int64),
                kinds=np.array(["full", "full"], dtype=object),  # one short
                axis_scores={"bikini": np.zeros(3, dtype=np.float32)},
                image_count=1,
            )
        self.assertIn("kinds", str(caught.exception))

    def test_region_table_rejects_a_mismatched_axis(self) -> None:
        with self.assertRaises(ValueError) as caught:
            cascade.RegionScoreTable(
                owner=np.zeros(2, dtype=np.int64),
                kinds=np.array(["full", "full"], dtype=object),
                axis_scores={"bikini": np.zeros(7, dtype=np.float32)},
                image_count=1,
            )
        self.assertIn("bikini", str(caught.exception))

    def test_region_table_allows_a_table_with_no_subject_attribution(self) -> None:
        """Tables built before per-person attribution leave `subject` empty."""
        cascade.RegionScoreTable(
            owner=np.zeros(2, dtype=np.int64),
            kinds=np.array(["full", "face"], dtype=object),
            axis_scores={"bikini": np.zeros(2, dtype=np.float32)},
            image_count=1,
            full_row=np.zeros(1, dtype=np.int64),
        )


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
