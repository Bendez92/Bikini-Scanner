"""Coordinate background scans without depending on the graphical user interface."""

from __future__ import annotations

import threading
import time
from collections.abc import Callable
from dataclasses import dataclass

from .backend_utils import ImageEmbeddingBackend
from .scorer import BikiniScorer, ScanCancelled, ScanProgress, ScoreState, scan_and_score_folder
from .store import FolderStore


def _spawn_daemon_thread(target: Callable[[], None]) -> None:
    threading.Thread(target=target, daemon=True).start()


@dataclass(slots=True, frozen=True)
class ScanRequest:
    store: FolderStore
    backend: ImageEmbeddingBackend
    scorer: BikiniScorer
    threshold: float
    batch_size: int
    full_rescan: bool
    source_state: ScoreState | None = None


@dataclass(slots=True, frozen=True)
class ScanCallbacks:
    progress: Callable[[ScanProgress], None]
    completed: Callable[[ScoreState, list[dict[str, object]], bool], None]
    failed: Callable[[Exception], None]
    cancelled: Callable[[], None]


class ScanController:
    def __init__(
        self,
        dispatch: Callable[[Callable[[], None]], None],
        callbacks: ScanCallbacks,
        spawn: Callable[[Callable[[], None]], None] = _spawn_daemon_thread,
        scan_function: Callable[..., tuple[ScoreState, list[dict[str, object]]]] = scan_and_score_folder,
    ) -> None:
        self._dispatch = dispatch
        self._callbacks = callbacks
        self._spawn = spawn
        self._scan_function = scan_function
        self._lock = threading.Lock()
        self._active = False
        self._generation = 0
        self._retrain_pending = False
        self._cancel_event: threading.Event | None = None
        self._started_at: float | None = None

    @property
    def active(self) -> bool:
        with self._lock:
            return self._active

    @property
    def generation(self) -> int:
        with self._lock:
            return self._generation

    @property
    def retrain_pending(self) -> bool:
        with self._lock:
            return self._retrain_pending

    @property
    def cancel_event(self) -> threading.Event | None:
        with self._lock:
            return self._cancel_event

    @property
    def started_at(self) -> float | None:
        with self._lock:
            return self._started_at

    def start(self, request: ScanRequest) -> bool:
        with self._lock:
            if self._active:
                return False
            self._generation += 1
            generation = self._generation
            cancel_event = threading.Event()
            self._cancel_event = cancel_event
            self._started_at = time.monotonic()
            self._active = True
        self._spawn(lambda: self._run(request, generation, cancel_event))
        return True

    def cancel(self) -> bool:
        with self._lock:
            if not self._active or self._cancel_event is None:
                return False
            self._cancel_event.set()
            return True

    def queue_retrain(self) -> None:
        with self._lock:
            self._retrain_pending = True

    def claim_retrain(self) -> bool:
        with self._lock:
            if not self._retrain_pending:
                return False
            self._retrain_pending = False
            return True

    def drop_retrain(self) -> None:
        with self._lock:
            self._retrain_pending = False

    def _run(self, request: ScanRequest, generation: int, cancel_event: threading.Event) -> None:
        def report_progress(progress: ScanProgress) -> None:
            if self._is_current(generation):
                self._dispatch(lambda: self._deliver_progress(generation, progress))

        try:
            if request.full_rescan or request.source_state is None:
                state, samples = self._scan_function(
                    request.backend,
                    request.store,
                    request.scorer,
                    threshold=request.threshold,
                    batch_size=request.batch_size,
                    cancel_event=cancel_event,
                    progress_callback=report_progress,
                )
            else:
                state, samples = request.scorer.rescore_state(
                    request.source_state,
                    request.store.load_labels(),
                    threshold=request.threshold,
                    store=request.store,
                    cancel_event=cancel_event,
                )
        except ScanCancelled:
            if self._is_current(generation):
                self._dispatch(lambda: self._deliver_cancelled(generation))
            return
        except Exception as exc:  # noqa: BLE001
            error = exc
            if self._is_current(generation):
                self._dispatch(lambda: self._deliver_failed(generation, error))
            return
        if self._is_current(generation):
            self._dispatch(lambda: self._deliver_completed(generation, state, samples, request.full_rescan))

    def _is_current(self, generation: int) -> bool:
        with self._lock:
            return generation == self._generation

    def _deliver_progress(self, generation: int, progress: ScanProgress) -> None:
        if self._is_current(generation):
            self._callbacks.progress(progress)

    def _deliver_completed(
        self,
        generation: int,
        state: ScoreState,
        samples: list[dict[str, object]],
        full_rescan: bool,
    ) -> None:
        with self._lock:
            if generation != self._generation:
                return
            self._active = False
            self._cancel_event = None
        self._callbacks.completed(state, samples, full_rescan)

    def _deliver_failed(self, generation: int, exc: Exception) -> None:
        with self._lock:
            if generation != self._generation:
                return
            self._active = False
            self._cancel_event = None
            self._retrain_pending = False
        self._callbacks.failed(exc)

    def _deliver_cancelled(self, generation: int) -> None:
        with self._lock:
            if generation != self._generation:
                return
            self._active = False
            self._cancel_event = None
            self._retrain_pending = False
        self._callbacks.cancelled()
