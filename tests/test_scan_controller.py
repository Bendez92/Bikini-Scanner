from __future__ import annotations

import os
import tempfile
import unittest
from collections.abc import Callable
from pathlib import Path
from typing import Any

import numpy as np

_STATE_DIR = Path(tempfile.mkdtemp(prefix="bikini_controller_state_"))
os.environ["APPDATA"] = str(_STATE_DIR)
os.environ["LOCALAPPDATA"] = str(_STATE_DIR)
os.environ["XDG_CONFIG_HOME"] = str(_STATE_DIR)
os.environ["XDG_STATE_HOME"] = str(_STATE_DIR)
os.environ["HOME"] = str(_STATE_DIR)
os.environ["USERPROFILE"] = str(_STATE_DIR)

from bikini_scanner.scan_controller import (
    ScanCallbacks,
    ScanController,
    ScanRequest,
)
from bikini_scanner.scorer import ScanCancelled, ScanProgress, ScoreState


def _state() -> ScoreState:
    return ScoreState(
        paths=[],
        embeddings=np.empty((0, 1), dtype=np.float32),
        zero_shot_scores=np.empty(0, dtype=np.float32),
        scores=np.empty(0, dtype=np.float32),
        axis_scores={},
        face_counts=None,
        classifier_trained=False,
        classifier_label_count=0,
    )


class FakeStore:
    def __init__(self) -> None:
        self.labels = {"image.jpg": 1}

    def load_labels(self) -> dict[str, int]:
        return self.labels


class FakeScorer:
    def __init__(self) -> None:
        self.calls: list[tuple[Any, ...]] = []

    def rescore_state(self, *args: Any, **kwargs: Any) -> tuple[ScoreState, list[dict[str, object]]]:
        self.calls.append((args, kwargs))
        return _state(), [{"kind": "rescore"}]


class ScanControllerTests(unittest.TestCase):
    def setUp(self) -> None:
        self.spawned: list[Callable[[], None]] = []
        self.dispatched: list[Callable[[], None]] = []
        self.progress: list[ScanProgress] = []
        self.completed: list[tuple[ScoreState, list[dict[str, object]], bool]] = []
        self.failed: list[Exception] = []
        self.cancelled = 0
        self.scan_cancel_event = None
        self.store = FakeStore()
        self.scorer = FakeScorer()

        def spawn(thunk: Callable[[], None]) -> None:
            self.spawned.append(thunk)

        self.controller = ScanController(
            dispatch=self.dispatched.append,
            callbacks=ScanCallbacks(
                progress=self.progress.append,
                completed=lambda state, samples, full: self.completed.append((state, samples, full)),
                failed=self.failed.append,
                cancelled=lambda: setattr(self, "cancelled", self.cancelled + 1),
            ),
            spawn=spawn,
            scan_function=self.scan,
        )

    def scan(self, *args: Any, **kwargs: Any) -> tuple[ScoreState, list[dict[str, object]]]:
        self.scan_cancel_event = kwargs["cancel_event"]
        progress_callback = kwargs["progress_callback"]
        progress_callback(ScanProgress("embed", "Reading images", 1, 1, 1.0, None, 1.0))
        return _state(), [{"kind": "scan"}]

    def request(self, *, full_rescan: bool = True, source_state: ScoreState | None = None) -> ScanRequest:
        return ScanRequest(
            store=self.store,  # type: ignore[arg-type]
            backend=object(),  # type: ignore[arg-type]
            scorer=self.scorer,  # type: ignore[arg-type]
            threshold=0.5,
            batch_size=4,
            full_rescan=full_rescan,
            source_state=source_state,
        )

    def drain(self) -> None:
        while self.dispatched:
            self.dispatched.pop(0)()

    def test_start_refuses_second_active_pass_and_allows_restart_after_completion(self) -> None:
        self.assertTrue(self.controller.start(self.request()))
        self.assertFalse(self.controller.start(self.request()))
        self.assertEqual(len(self.spawned), 1)
        self.spawned[0]()
        self.assertTrue(self.controller.active)
        self.assertFalse(self.controller.start(self.request()))
        self.drain()
        self.assertFalse(self.controller.active)
        self.assertTrue(self.controller.start(self.request()))
        self.assertEqual(len(self.spawned), 2)

    def test_callbacks_are_only_invoked_through_dispatch(self) -> None:
        self.controller.start(self.request())
        self.spawned[0]()
        self.assertEqual(self.progress, [])
        self.assertEqual(self.completed, [])
        self.assertTrue(self.controller.active)
        self.drain()
        self.assertEqual(len(self.progress), 1)
        self.assertEqual(len(self.completed), 1)
        self.assertFalse(self.controller.active)

    def test_stale_closure_is_dropped_without_state_change(self) -> None:
        self.controller.start(self.request())
        self.spawned[0]()
        terminal = self.dispatched.pop()
        terminal()
        self.assertFalse(self.controller.active)
        self.controller.start(self.request())
        completed_before = list(self.completed)
        active_before = self.controller.active
        self.drain()
        self.assertEqual(self.completed, completed_before)
        self.assertEqual(self.controller.active, active_before)

    def test_cancelled_and_failed_clear_pending_retrain(self) -> None:
        def cancelled_scan(*args: Any, **kwargs: Any) -> tuple[ScoreState, list[dict[str, object]]]:
            raise ScanCancelled()

        controller = ScanController(
            dispatch=self.dispatched.append,
            callbacks=ScanCallbacks(
                progress=self.progress.append,
                completed=lambda *_args: None,
                failed=self.failed.append,
                cancelled=lambda: setattr(self, "cancelled", self.cancelled + 1),
            ),
            spawn=lambda thunk: self.spawned.append(thunk),
            scan_function=cancelled_scan,
        )
        controller.queue_retrain()
        controller.start(self.request())
        self.spawned[0]()
        self.assertTrue(controller.retrain_pending)
        self.drain()
        self.assertFalse(controller.retrain_pending)
        self.assertEqual(self.cancelled, 1)

        def failed_scan(*args: Any, **kwargs: Any) -> tuple[ScoreState, list[dict[str, object]]]:
            raise RuntimeError("broken")

        controller = ScanController(
            dispatch=self.dispatched.append,
            callbacks=ScanCallbacks(
                progress=self.progress.append,
                completed=lambda *_args: None,
                failed=self.failed.append,
                cancelled=lambda: None,
            ),
            spawn=lambda thunk: self.spawned.append(thunk),
            scan_function=failed_scan,
        )
        controller.queue_retrain()
        controller.start(self.request())
        self.spawned[1]()
        self.assertTrue(controller.retrain_pending)
        self.drain()
        self.assertFalse(controller.retrain_pending)
        self.assertEqual(str(self.failed[-1]), "broken")

    def test_completed_leaves_pending_retrain_set(self) -> None:
        self.controller.queue_retrain()
        self.controller.start(self.request())
        self.spawned[0]()
        self.assertTrue(self.controller.active)
        self.assertTrue(self.controller.retrain_pending)
        self.drain()
        self.assertTrue(self.controller.retrain_pending)

    def test_cancel_sets_the_scan_event_without_clearing_active(self) -> None:
        self.controller.start(self.request())
        event = self.controller.cancel_event
        self.assertIsNotNone(event)
        self.assertTrue(self.controller.cancel())
        self.assertIs(event, self.controller.cancel_event)
        self.assertTrue(event.is_set())
        self.assertTrue(self.controller.active)
        self.spawned[0]()
        self.assertIs(self.scan_cancel_event, event)
        self.assertTrue(self.controller.cancel())
        self.drain()
        self.assertFalse(self.controller.active)
        self.assertFalse(self.controller.cancel())

    def test_retrain_queue_operations(self) -> None:
        self.assertFalse(self.controller.claim_retrain())
        self.controller.queue_retrain()
        self.assertTrue(self.controller.retrain_pending)
        self.assertTrue(self.controller.claim_retrain())
        self.assertFalse(self.controller.claim_retrain())
        self.controller.queue_retrain()
        self.controller.drop_retrain()
        self.assertFalse(self.controller.retrain_pending)

    def test_rescore_and_full_rescan_dispatch_the_expected_path(self) -> None:
        source_state = _state()
        self.controller.start(self.request(full_rescan=False, source_state=source_state))
        self.spawned[0]()
        self.assertEqual(len(self.scorer.calls), 1)
        args, kwargs = self.scorer.calls[0]
        self.assertIs(args[0], source_state)
        self.assertEqual(args[1], self.store.labels)
        self.assertEqual(kwargs["threshold"], 0.5)
        self.assertIs(kwargs["store"], self.store)
        self.drain()

        self.controller.start(self.request(full_rescan=True, source_state=source_state))
        self.spawned[1]()
        self.assertEqual(len(self.scorer.calls), 1)


if __name__ == "__main__":
    unittest.main()
