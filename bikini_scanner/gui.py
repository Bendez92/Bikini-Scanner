from __future__ import annotations

import csv
import json
import logging
import os
import platform
import subprocess
import sys
import threading
import time
from collections import OrderedDict, deque
from dataclasses import dataclass
from pathlib import Path
from tkinter import (
    BOTH,
    BOTTOM,
    END,
    LEFT,
    RIGHT,
    TOP,
    BooleanVar,
    Canvas,
    DoubleVar,
    IntVar,
    Listbox,
    Menu,
    StringVar,
    Text,
    Tk,
    Toplevel,
    filedialog,
    messagebox,
    ttk,
)
from tkinter import font as tkfont
from typing import Any, cast

import numpy as np
from PIL import Image, ImageTk

from . import cascade, vision_analysis
from .__version__ import __version__
from .backend_utils import ImageEmbeddingBackend
from .config import HIGH_ACCURACY_MODEL, ScannerConfig, filter_folder_override
from .config_profiles import BUILTIN_PROFILES, delete_profile, profile_config, profile_names, save_profile
from .config_schema import FieldError, parse_choice_entry, parse_float_entry, parse_int_entry
from .global_store import GlobalLearningStore
from .image_formats import open_oriented, oriented_size
from .logging_setup import configure_logging, log_path, read_log_tail
from .output_ops import (
    OutputOptions,
    PlannedTransfer,
    build_html_report,
    build_transfer_plan,
    execute_transfer_plan,
    trash_files,
    write_image_metadata,
)
from .plugins import apply_plugins
from .safe_io import atomic_write_json, quarantine_broken_file
from .scorer import (
    PHASE_EMBED,
    BikiniScorer,
    ScanCancelled,
    ScanProgress,
    ScoreState,
    bucketed_sampling,
    scan_and_score_folder,
    state_disagreement,
)
from .store import MATCHES_DIR_NAME, SUPPORTED_IMAGE_SUFFIXES, FolderStore
from .update_checker import check_for_update
from .user_prefs import load_user_prefs, prefs_path, save_user_prefs
from .vlm_backend import is_local_endpoint

try:
    import psutil
except Exception:  # noqa: BLE001
    psutil = None

try:
    from tkinterdnd2 import DND_FILES, TkinterDnD
except Exception:  # noqa: BLE001
    DND_FILES = None
    TkinterDnD = None


LAST_FOLDER_STATE_PATH = Path.home() / ".bikini_scanner_last_folder.json"
LOGGER = logging.getLogger(__name__)

# Every result card reserves exactly this much width for its text column so one long
# filename cannot make its card a different size from the cards around it.
CARD_INFO_WIDTH = 340
# Buckets used by the detected-files view, in the order they are listed.
DETECTED_BUCKETS = ("Cleavage", "Bikini", "Midriff", "Explicit (NSFW)", "Other detections")


@dataclass(slots=True)
class ResultCard:
    frame: ttk.Frame
    path: str
    name_label: ttk.Label
    score_label: ttk.Label
    label_label: ttk.Label
    details_label: ttk.Label
    image_ref: ImageTk.PhotoImage
    score: float = 0.0


class BikiniScannerApp:
    def __init__(
        self,
        root: Tk,
        config: ScannerConfig | None = None,
        initial_folder: str = "",
    ) -> None:
        configure_logging()
        self.root = root
        self.root.title(f"Bikini Scanner {__version__}")
        self.config = config or ScannerConfig()
        self.global_config = ScannerConfig.from_mapping(self.config.to_dict())
        self.folder_override_active = False
        self.override_var = StringVar(value="")
        self.user_prefs = load_user_prefs()
        self.update_url_var = StringVar(value=str(self.user_prefs.get("update_url", "")))
        self.folder_var = StringVar(value=initial_folder)
        self.status_var = StringVar(value="Choose a folder to begin.")
        self.summary_var = StringVar(value="")
        self.threshold_var = DoubleVar(value=float(self.config.threshold))
        self.theme_var = StringVar(value=str(self.user_prefs.get("theme", "dark")))
        self.font_size_var = IntVar(value=int(self.user_prefs.get("font_size", 10)))
        self.columns_var = IntVar(value=int(self.user_prefs.get("columns", 2)))
        self.thumbnail_size_var = IntVar(value=int(self.user_prefs.get("thumbnail_size", 320)))
        try:
            thumbnail_cache_size = int(self.user_prefs.get("thumbnail_cache_size", 512))
        except (TypeError, ValueError):
            thumbnail_cache_size = 512
        self.thumbnail_cache_size_var = IntVar(value=max(32, min(2048, thumbnail_cache_size)))
        self.search_var = StringVar(value=str(self.user_prefs.get("search", "")))
        self.sort_var = StringVar(value=str(self.user_prefs.get("sort", "score")))
        self.match_filter_var = StringVar(value=str(self.user_prefs.get("match_filter", "all")))
        self.label_filter_var = StringVar(value=str(self.user_prefs.get("label_filter", "all")))
        self.score_min_var = StringVar(value=str(self.user_prefs.get("score_min", "")))
        self.score_max_var = StringVar(value=str(self.user_prefs.get("score_max", "")))
        self.output_organization_var = StringVar(value=str(self.user_prefs.get("output_organization", "flat")))
        self.output_template_var = StringVar(value=str(self.user_prefs.get("output_template", "{stem}")))
        self.output_duplicate_var = StringVar(value=str(self.user_prefs.get("output_duplicate", "rename")))
        self.output_score_low_var = DoubleVar(value=float(self.user_prefs.get("output_score_low", 0.35)))
        self.output_score_high_var = DoubleVar(value=float(self.user_prefs.get("output_score_high", 0.7)))
        self.recent_folders: list[str] = [
            str(item) for item in self.user_prefs.get("recent_folders", []) if isinstance(item, str)
        ]
        self.move_files_var = BooleanVar(value=False)
        self.nsfw_only_var = BooleanVar(value=self.config.nsfw_filter == "only")
        self.hardware_var = StringVar(value="")
        self.progress_var = DoubleVar(value=0.0)
        self.progress_text_var = StringVar(value="")
        self.progress_detail_var = StringVar(value="")
        self.loading_var = StringVar(value="")
        self.stats_var = StringVar(value="")
        self.notice_var = StringVar(value="")
        self.cards: dict[str, ResultCard] = {}
        self.displayed_samples: list[dict[str, object]] = []
        self.review_samples: list[dict[str, object]] = []
        self.photo_refs: list[ImageTk.PhotoImage] = []
        self.thumbnail_cache: OrderedDict[tuple[str, int], ImageTk.PhotoImage] = OrderedDict()
        self.preview_caption_var = StringVar(value="")
        self.preview_cache: OrderedDict[tuple[str, int, int], ImageTk.PhotoImage] = OrderedDict()
        self._preview_render_size: tuple[int, int] = (0, 0)
        self._preview_resize_after_id: str | None = None
        self._threshold_refresh_after_id: str | None = None
        # Collapsible chrome: advanced controls stay out of the way until asked for.
        self._panels: dict[str, ttk.Frame] = {}
        self._panel_open: dict[str, bool] = {"filters": False, "queue": False}
        self.current_state: ScoreState | None = None
        self.current_samples: list[dict[str, object]] = []
        self.view_mode = "review"
        self.similar_anchor_path: str | None = None
        self.focused_path: str | None = None
        self.undo_stack: list[dict[str, object]] = []
        self.redo_stack: list[dict[str, object]] = []
        self.quality_history: deque[float] = deque(maxlen=6)
        self.scan_queue: list[str] = []
        self.queue_active = False
        self.queue_index = 0
        self.watch_enabled_var = BooleanVar(value=False)
        self._watch_snapshot: dict[str, tuple[int, int]] = {}
        self._watch_after_id: str | None = None
        self._hardware_after_id: str | None = None
        self.store: FolderStore | None = None
        self.backend: ImageEmbeddingBackend | None = None
        self.scorer: BikiniScorer | None = None
        self._refresh_generation = 0
        self._scan_start_monotonic: float | None = None
        self._scan_cancel_event: threading.Event | None = None
        self._backend_preload_started = False
        self._scan_active = False
        # Set when a retrain is asked for while one is already running, so labelling a
        # run of photos quickly produces one retrain at the end rather than a thread per
        # click, all racing each other over the same scorer and label store.
        self._retrain_pending = False
        self._closing = False
        self._first_run_guide_shown = bool(self.user_prefs.get("first_run_guide_shown", False))
        try:
            self._psutil_process = psutil.Process() if psutil is not None else None
        except Exception:  # noqa: BLE001
            self._psutil_process = None
        self._psutil_cpu_primed = False
        self._build_ui()
        self._enable_drag_and_drop()
        self._bind_ui_prefs()
        self._restore_ui_prefs()
        self._apply_theme()
        self._bind_shortcuts()
        self.root.protocol("WM_DELETE_WINDOW", self._on_close)
        self._apply_window_icon()
        # Resolve the folder before preloading: _set_folder applies that folder's
        # config override and clears self.backend, which would race the preload thread.
        # That applies to --folder too, which used to be bound after construction and so
        # could wipe a backend the preload thread was still loading.
        if initial_folder.strip():
            self._set_folder(initial_folder)
        else:
            self._resume_last_folder_if_any()
        self._maybe_preload_backend()
        self._refresh_hardware_status()
        self._after(750, self._maybe_show_first_run_guide)

    def _after(self, delay_ms: int, callback, *args):
        if self._closing:
            return None
        try:
            return self.root.after(delay_ms, callback, *args)
        except Exception:  # noqa: BLE001
            return None

    def _create_modal(
        self,
        title: str,
        *,
        padding: int = 12,
        geometry: str | None = None,
        resizable: tuple[bool, bool] | None = None,
    ) -> tuple[Toplevel, ttk.Frame]:
        dialog = Toplevel(self.root)
        dialog.title(title)
        dialog.transient(self.root)
        dialog.grab_set()

        # Every modal grabs pointer events. If a dialog is closed while an exception is
        # in flight (or via a bare dialog.destroy button that bypasses a custom close
        # handler), the grab can be left active and the whole app stays modal-locked.
        # Attach a safe close to every modal and wire WM_DELETE_WINDOW to it; individual
        # dialogs that already define their own close handler override this protocol and
        # are unaffected.
        def _safe_close() -> None:
            try:
                dialog.grab_release()
            except Exception:  # noqa: BLE001
                pass
            dialog.destroy()

        dialog._safe_close = _safe_close  # type: ignore[attr-defined]
        dialog.protocol("WM_DELETE_WINDOW", _safe_close)

        if geometry:
            dialog.geometry(geometry)
        if resizable is not None:
            dialog.resizable(*resizable)
        dialog.configure(bg=self._palette()["bg"])
        outer = ttk.Frame(dialog, padding=padding)
        outer.pack(fill=BOTH, expand=True)
        return dialog, outer

    @staticmethod
    def _modal_button_row(parent: ttk.Frame, *, pady: tuple[int, int] = (10, 0)) -> ttk.Frame:
        row = ttk.Frame(parent)
        row.pack(side=TOP, fill="x", pady=pady)
        return row

    @staticmethod
    def _modal_scroll_frame(parent: ttk.Frame) -> tuple[Canvas, ttk.Frame, int]:
        canvas = Canvas(parent, highlightthickness=0)
        scrollbar = ttk.Scrollbar(parent, orient="vertical", command=canvas.yview)
        scroll_frame = ttk.Frame(canvas)
        scroll_window = canvas.create_window((0, 0), window=scroll_frame, anchor="nw")
        canvas.configure(yscrollcommand=scrollbar.set)
        canvas.pack(side=LEFT, fill=BOTH, expand=True)
        scrollbar.pack(side=RIGHT, fill="y")
        scroll_frame.bind("<Configure>", lambda _event: canvas.configure(scrollregion=canvas.bbox("all")))
        return canvas, scroll_frame, scroll_window

    def _build_ui(self) -> None:
        self._build_menu_bar()
        # Bottom first: it claims the bottom edge before the results area expands.
        self._build_status_bar()
        self._build_command_bar()
        self._build_filter_panel()
        self._build_queue_panel()
        self._build_threshold_row()
        self._build_view_switch()

        self.canvas = ttk.Frame(self.root)
        self.canvas.pack(side=TOP, fill=BOTH, expand=True)
        self.scroll = ttk.Scrollbar(self.canvas, orient="vertical")
        self.scroll.pack(side=RIGHT, fill="y")
        self.grid_canvas = Canvas(self.canvas, yscrollcommand=self.scroll.set, highlightthickness=0)
        self.grid_canvas.pack(side=LEFT, fill=BOTH, expand=True)
        self.scroll.config(command=self.grid_canvas.yview)
        self.grid_inner = ttk.Frame(self.grid_canvas)
        self.grid_window = self.grid_canvas.create_window((0, 0), window=self.grid_inner, anchor="nw")
        self.grid_inner.bind(
            "<Configure>", lambda _event: self.grid_canvas.configure(scrollregion=self.grid_canvas.bbox("all"))
        )
        self.grid_canvas.bind(
            "<Configure>",
            lambda event: self.grid_canvas.itemconfigure(self.grid_window, width=event.width),
        )

        # Active-picture preview; packed above the grid (before=self.canvas) whenever a
        # card is focused. Everything here is packed with side=TOP and no fill so it
        # stays horizontally centred no matter how wide the window gets.
        self.preview_frame = ttk.Frame(self.root, padding=(10, 4, 10, 6))
        self.preview_image_label = ttk.Label(self.preview_frame, anchor="center")
        self.preview_image_label.pack(side=TOP)
        ttk.Label(
            self.preview_frame,
            textvariable=self.preview_caption_var,
            anchor="center",
            justify="center",
        ).pack(side=TOP, pady=(4, 0))
        preview_buttons = ttk.Frame(self.preview_frame)
        preview_buttons.pack(side=TOP, pady=(6, 0))
        ttk.Button(preview_buttons, text="Accept (A)", width=14, command=lambda: self.label_focused_card(1)).pack(
            side=LEFT, padx=(0, 8)
        )
        ttk.Button(preview_buttons, text="REJECT (D)", width=14, command=lambda: self.label_focused_card(0)).pack(
            side=LEFT
        )
        self.preview_image_label.bind(
            "<Double-Button-1>",
            lambda _event: self.view_image(self.focused_path) if self.focused_path else None,
        )
        # Resize the active picture with the window, debounced so dragging the frame
        # does not rebuild the image on every <Configure>.
        self.root.bind("<Configure>", self._on_root_configure, add="+")
        self._build_empty_state()
        self._sync_panel_buttons()
        self._refresh_empty_state()

    # --- chrome -------------------------------------------------------------
    def _build_command_bar(self) -> None:
        """One row for the things you actually do: pick a folder, scan it, review."""
        bar = ttk.Frame(self.root, padding=(12, 10, 12, 8), style="Toolbar.TFrame")
        bar.pack(side=TOP, fill="x")
        self.command_bar = bar

        # Right-hand side is packed first so it keeps its width when the path is long.
        self.theme_button = ttk.Button(bar, text=self._theme_button_text(), command=self.cycle_theme, width=11)
        self.theme_button.pack(side=RIGHT)
        self._tooltip(self.theme_button, "Switch between dark, light, and following Windows")
        settings_button = ttk.Button(bar, text="Settings", command=self.open_settings_dialog)
        settings_button.pack(side=RIGHT, padx=(0, 6))
        help_button = ttk.Button(bar, text="?", command=self.show_shortcuts, width=3)
        help_button.pack(side=RIGHT, padx=(0, 6))
        self._tooltip(help_button, "Keyboard shortcuts and a quick tour")

        self.queue_toggle = ttk.Button(bar, text="Queue", command=lambda: self._toggle_panel("queue"))
        self.queue_toggle.pack(side=RIGHT, padx=(0, 6))
        self._tooltip(self.queue_toggle, "Scan several folders in a row, or watch one for changes")
        self.filter_toggle = ttk.Button(bar, text="Filters & view", command=lambda: self._toggle_panel("filters"))
        self.filter_toggle.pack(side=RIGHT, padx=(0, 6))
        self._tooltip(self.filter_toggle, "Search, sort, score range, thumbnail size")

        ttk.Label(bar, text="Folder", style="Muted.TLabel").pack(side=LEFT, padx=(0, 8))
        folder_entry = ttk.Entry(bar, textvariable=self.folder_var)
        folder_entry.pack(side=LEFT, fill="x", expand=True, padx=(0, 6))
        self._tooltip(folder_entry, "The folder to scan. You can also drag a folder onto this window.")
        choose_button = ttk.Button(bar, text="Choose…", command=self.choose_folder)
        choose_button.pack(side=LEFT, padx=(0, 12))

        self.run_button = ttk.Button(bar, text="Run scan", command=self.run_scan, style="Accent.TButton")
        self.run_button.pack(side=LEFT)
        self._tooltip(self.run_button, "Scan this folder for bikini, cleavage and midriff photos")
        self.stop_scan_button = ttk.Button(bar, text="Stop", command=self.cancel_scan, state="disabled", width=7)
        self.stop_scan_button.pack(side=LEFT, padx=6)
        self.update_button = ttk.Button(bar, text="Update rankings", command=self.update_algorithm)
        self.update_button.pack(side=LEFT, padx=(0, 6))
        self._tooltip(self.update_button, "Re-rank using everything you have accepted and rejected so far")
        ttk.Label(bar, textvariable=self.override_var, style="Muted.TLabel").pack(side=LEFT, padx=(4, 0))

    def _build_filter_panel(self) -> None:
        panel = ttk.Frame(self.root, padding=(12, 4, 12, 10), style="Toolbar.TFrame")
        self._panels["filters"] = panel

        row_one = ttk.Frame(panel, style="Toolbar.TFrame")
        row_one.pack(side=TOP, fill="x")
        ttk.Label(row_one, text="Search", style="Muted.TLabel").pack(side=LEFT)
        search_entry = ttk.Entry(row_one, textvariable=self.search_var, width=20)
        search_entry.pack(side=LEFT, padx=(4, 12))
        ttk.Label(row_one, text="Sort", style="Muted.TLabel").pack(side=LEFT)
        sort_combo = ttk.Combobox(
            row_one, textvariable=self.sort_var, values=("score", "filename", "date"), state="readonly", width=10
        )
        sort_combo.pack(side=LEFT, padx=(4, 12))
        ttk.Label(row_one, text="Show", style="Muted.TLabel").pack(side=LEFT)
        match_combo = ttk.Combobox(
            row_one,
            textvariable=self.match_filter_var,
            values=("all", "matched", "unmatched"),
            state="readonly",
            width=10,
        )
        match_combo.pack(side=LEFT, padx=(4, 6))
        label_combo = ttk.Combobox(
            row_one,
            textvariable=self.label_filter_var,
            values=("all", "unlabeled", "labeled", "skipped"),
            state="readonly",
            width=10,
        )
        label_combo.pack(side=LEFT, padx=(0, 12))
        ttk.Label(row_one, text="Score", style="Muted.TLabel").pack(side=LEFT)
        ttk.Entry(row_one, textvariable=self.score_min_var, width=5).pack(side=LEFT, padx=(4, 2))
        # An en dash is the correct typography for a numeric range separator here.
        ttk.Label(row_one, text="–", style="Muted.TLabel").pack(side=LEFT)  # noqa: RUF001
        ttk.Entry(row_one, textvariable=self.score_max_var, width=5).pack(side=LEFT, padx=(2, 12))
        ttk.Checkbutton(row_one, text="Only NSFW", variable=self.nsfw_only_var, command=self._toggle_nsfw_only).pack(
            side=LEFT
        )
        clear_button = ttk.Button(row_one, text="Clear filters", command=self.clear_filters)
        clear_button.pack(side=RIGHT)

        row_two = ttk.Frame(panel, style="Toolbar.TFrame")
        row_two.pack(side=TOP, fill="x", pady=(8, 0))
        ttk.Label(row_two, text="Columns", style="Muted.TLabel").pack(side=LEFT)
        columns_spin = ttk.Spinbox(row_two, from_=1, to=6, textvariable=self.columns_var, width=4)
        columns_spin.pack(side=LEFT, padx=(4, 12))
        ttk.Label(row_two, text="Text size", style="Muted.TLabel").pack(side=LEFT)
        font_spin = ttk.Spinbox(row_two, from_=8, to=16, textvariable=self.font_size_var, width=4)
        font_spin.pack(side=LEFT, padx=(4, 12))
        ttk.Label(row_two, text="Thumbnail size", style="Muted.TLabel").pack(side=LEFT)
        thumb_scale = ttk.Scale(row_two, from_=120, to=520, variable=self.thumbnail_size_var, orient="horizontal")
        thumb_scale.pack(side=LEFT, fill="x", expand=True, padx=(4, 12))
        ttk.Button(row_two, text="Prompt tester", command=self.open_prompt_tester_dialog).pack(side=RIGHT)

    def _build_queue_panel(self) -> None:
        panel = ttk.Frame(self.root, padding=(12, 4, 12, 10), style="Toolbar.TFrame")
        self._panels["queue"] = panel
        row = ttk.Frame(panel, style="Toolbar.TFrame")
        row.pack(side=TOP, fill="x")
        ttk.Button(row, text="Add folder", command=self.add_folder_to_queue).pack(side=LEFT)
        ttk.Button(row, text="Run queue", command=self.run_queue).pack(side=LEFT, padx=6)
        ttk.Button(row, text="Stop queue", command=self.stop_queue).pack(side=LEFT)
        ttk.Checkbutton(
            row,
            text="Watch this folder for new photos",
            variable=self.watch_enabled_var,
            command=self._toggle_watch_mode,
        ).pack(side=LEFT, padx=16)
        self.queue_listbox = Listbox(panel, height=3)
        self.queue_listbox.pack(side=TOP, fill="x", pady=(8, 0))

    def _build_view_switch(self) -> None:
        """Detected list vs review queue — the two ways to look at a scan."""
        row = ttk.Frame(self.root, padding=(12, 0, 12, 6))
        row.pack(side=TOP, fill="x")
        self.view_switch_row = row
        self.detected_button = ttk.Button(
            row, text="Detected files", command=self.show_detected_files, style="Accent.TButton"
        )
        self.detected_button.pack(side=LEFT)
        self._tooltip(self.detected_button, "Every photo found, grouped by what was detected")
        self.review_button = ttk.Button(row, text="Review queue", command=self.restore_review_view)
        self.review_button.pack(side=LEFT, padx=6)
        self._tooltip(self.review_button, "A curated shortlist to Accept or REJECT so the scanner learns")
        self.view_hint = ttk.Label(row, text="", style="Muted.TLabel")
        self.view_hint.pack(side=LEFT, padx=(12, 0))

    def _sync_view_switch(self) -> None:
        if not hasattr(self, "detected_button"):
            return
        detected = self.view_mode == "detected"
        hints = {
            "detected": "Everything above the sensitivity threshold.",
            "review": "A shortlist chosen to teach the scanner fastest.",
            "similar": "Images similar to the one you picked.",
        }
        try:
            self.detected_button.configure(style="Accent.TButton" if detected else "TButton")
            self.review_button.configure(style="TButton" if detected else "Accent.TButton")
            self.view_hint.configure(text=hints.get(self.view_mode, ""))
        except Exception:  # noqa: BLE001
            pass

    def _build_threshold_row(self) -> None:
        row = ttk.Frame(self.root, padding=(12, 0, 12, 8))
        row.pack(side=TOP, fill="x")
        ttk.Label(row, text="Sensitivity", style="Muted.TLabel").pack(side=LEFT)
        slider = ttk.Scale(
            row,
            from_=0.0,
            to=1.0,
            orient="horizontal",
            variable=self.threshold_var,
            command=self._on_threshold_change,
        )
        slider.pack(side=LEFT, fill="x", expand=True, padx=10)
        self._tooltip(slider, "Left shows more photos (and more false alarms); right shows only the surest matches")
        ttk.Label(row, textvariable=self.summary_var, style="Muted.TLabel").pack(side=LEFT)

    def _build_status_bar(self) -> None:
        outer = ttk.Frame(self.root, padding=(12, 6, 12, 8), style="Toolbar.TFrame")
        outer.pack(side=BOTTOM, fill="x")

        self.progress_row = ttk.Frame(outer, style="Toolbar.TFrame")
        # The count line sits above the bar rather than beside it, so "1,204 / 4,096
        # files" is readable at a glance and never squeezes the bar as the digits grow.
        ttk.Label(
            self.progress_row,
            textvariable=self.progress_detail_var,
            style="Toolbar.TLabel",
            anchor="w",
        ).pack(side=TOP, fill="x", pady=(0, 3))
        bar_row = ttk.Frame(self.progress_row, style="Toolbar.TFrame")
        bar_row.pack(side=TOP, fill="x")
        self.progress_bar = ttk.Progressbar(
            bar_row,
            orient="horizontal",
            maximum=100,
            variable=self.progress_var,
            mode="determinate",
        )
        self.progress_bar.pack(side=LEFT, fill="x", expand=True)
        # Fixed width so the bar does not twitch as the percentage text changes length.
        ttk.Label(bar_row, textvariable=self.progress_text_var, width=6, anchor="e", style="Toolbar.TLabel").pack(
            side=LEFT, padx=(8, 0)
        )

        status_row = ttk.Frame(outer, style="Toolbar.TFrame")
        status_row.pack(side=TOP, fill="x")
        self._status_row = status_row
        ttk.Label(status_row, textvariable=self.status_var, style="Toolbar.TLabel").pack(
            side=LEFT, fill="x", expand=True
        )
        # VLM badge: visible only when VLM adjudication is enabled, so the user knows
        # the scan includes a second-opinion stage.
        self.vlm_badge = ttk.Label(status_row, text="VLM", style="Accent.TLabel", padding=(4, 0))
        self.vlm_badge.pack(side=LEFT, padx=(6, 0))
        self._refresh_vlm_badge()
        ttk.Label(status_row, textvariable=self.loading_var, style="Toolbar.TLabel").pack(side=LEFT, padx=(10, 0))
        self.hardware_label = ttk.Label(status_row, textvariable=self.hardware_var, style="Muted.TLabel")
        if psutil is not None:
            self.hardware_label.pack(side=RIGHT)

        stats_row = ttk.Frame(outer, style="Toolbar.TFrame")
        stats_row.pack(side=TOP, fill="x", pady=(2, 0))
        ttk.Label(stats_row, textvariable=self.stats_var, style="Muted.TLabel").pack(side=LEFT, fill="x", expand=True)
        ttk.Label(stats_row, textvariable=self.notice_var, style="Muted.TLabel").pack(side=RIGHT)

    def _build_empty_state(self) -> None:
        """Guidance in the results area instead of a bare 'no samples' line."""
        self.empty_state = ttk.Frame(self.canvas, padding=48, style="Surface.TFrame")
        inner = ttk.Frame(self.empty_state, style="Surface.TFrame")
        inner.pack(expand=True)
        self.empty_title = ttk.Label(inner, text="", style="Heading.TLabel", anchor="center", justify="center")
        self.empty_title.pack(side=TOP)
        self.empty_body = ttk.Label(
            inner, text="", style="SurfaceMuted.TLabel", anchor="center", justify="center", wraplength=520
        )
        self.empty_body.pack(side=TOP, pady=(10, 16))
        self.empty_button = ttk.Button(inner, text="", style="Accent.TButton")
        self.empty_button.pack(side=TOP)

    def _refresh_empty_state(self) -> None:
        """Swap the results grid for advice whenever there is nothing to show."""
        if not hasattr(self, "empty_state"):
            return
        folder = self.folder_var.get().strip()
        if self.current_state is None and not folder:
            title = "Choose a folder to get started"
            body = (
                "Pick a folder of photos and press Run scan. Everything happens on this "
                "computer — no images are uploaded anywhere."
            )
            action = ("Choose folder…", self.choose_folder)
        elif self.current_state is None:
            title = "Ready to scan"
            body = f"{folder}\n\nPress Run scan to look through this folder."
            action = ("Run scan", self.run_scan)
        elif not self.displayed_samples:
            if self._filters_active():
                title = "Nothing matches these filters"
                body = "Your search, score range, or label filter is hiding every result."
                action = ("Clear filters", self.clear_filters)
            else:
                title = "No matches at this sensitivity"
                body = (
                    "Nothing scored above the current sensitivity. Drag the sensitivity "
                    "slider left to see near misses, or scan a different folder."
                )
                action = ("Show detected files", self.show_detected_files)
        else:
            if self.empty_state.winfo_ismapped():
                self.empty_state.pack_forget()
                self.scroll.pack(side=RIGHT, fill="y")
                self.grid_canvas.pack(side=LEFT, fill=BOTH, expand=True)
            return

        self.empty_title.configure(text=title)
        self.empty_body.configure(text=body)
        self.empty_button.configure(text=action[0], command=action[1])
        if not self.empty_state.winfo_ismapped():
            self.grid_canvas.pack_forget()
            self.scroll.pack_forget()
            self.empty_state.pack(fill=BOTH, expand=True)

    def _refresh_vlm_badge(self) -> None:
        """Show or hide the VLM badge in the status bar based on config."""
        if not hasattr(self, "vlm_badge"):
            return
        try:
            if self.config.vlm_enabled:
                self.vlm_badge.pack(side=LEFT, padx=(6, 0))
            else:
                self.vlm_badge.pack_forget()
        except Exception:  # noqa: BLE001
            pass

    def _show_progress(self, visible: bool) -> None:
        """The progress bar only occupies space while something is running."""
        row = getattr(self, "progress_row", None)
        if row is None:
            return
        try:
            if visible and not row.winfo_ismapped():
                row.pack(side=TOP, fill="x", pady=(0, 4), before=self._status_row)
            elif not visible and row.winfo_ismapped():
                row.pack_forget()
        except Exception:  # noqa: BLE001
            pass

    def _filters_active(self) -> bool:
        return bool(
            self.search_var.get().strip()
            or self.score_min_var.get().strip()
            or self.score_max_var.get().strip()
            or self.match_filter_var.get().strip() not in ("", "all")
            or self.label_filter_var.get().strip() not in ("", "all")
        )

    def clear_filters(self) -> None:
        self.search_var.set("")
        self.score_min_var.set("")
        self.score_max_var.set("")
        self.match_filter_var.set("all")
        self.label_filter_var.set("all")
        self.status_var.set("Filters cleared.")

    def _toggle_panel(self, name: str) -> None:
        panel = self._panels.get(name)
        if panel is None:
            return
        # Tracked state, not winfo_ismapped(): mapping is only settled after the event
        # loop runs, so two toggles in a row would read each other's stale answer.
        wanted = not self._panel_open.get(name, False)
        # Re-pack every open panel in a fixed order, otherwise the second one opened
        # jumps above the first.
        for key in ("filters", "queue"):
            other = self._panels.get(key)
            if other is not None:
                other.pack_forget()
        for key in ("filters", "queue"):
            other = self._panels.get(key)
            if other is None:
                continue
            should_show = wanted if key == name else self._panel_open.get(key, False)
            if should_show:
                other.pack(side=TOP, fill="x", after=self.command_bar)
            self._panel_open[key] = should_show
        self._sync_panel_buttons()

    def _sync_panel_buttons(self) -> None:
        for name, button in (
            ("filters", getattr(self, "filter_toggle", None)),
            ("queue", getattr(self, "queue_toggle", None)),
        ):
            if button is None:
                continue
            label = "Filters & view" if name == "filters" else "Queue"
            # Show a dot when filters are active so the user knows something is hidden.
            if name == "filters" and self._filters_active():
                label = f"{label} ●"
            try:
                button.configure(text=f"{label} ▴" if self._panel_open.get(name) else f"{label} ▾")
            except Exception:  # noqa: BLE001
                continue

    def _theme_button_text(self) -> str:
        return {"dark": "Theme: Dark", "light": "Theme: Light"}.get(self.theme_var.get().strip().lower(), "Theme: Auto")

    def cycle_theme(self) -> None:
        order = ["dark", "light", "system"]
        current = self.theme_var.get().strip().lower()
        self.theme_var.set(order[(order.index(current) + 1) % len(order)] if current in order else "dark")

    def _tooltip(self, widget, text: str) -> None:
        """Plain hover help. Tk has none built in, and these controls need explaining."""
        state: dict[str, object] = {"window": None, "after": None}

        def show() -> None:
            state["after"] = None
            if state["window"] is not None:
                return
            try:
                x = widget.winfo_rootx() + 12
                y = widget.winfo_rooty() + widget.winfo_height() + 6
            except Exception:  # noqa: BLE001
                return
            palette = self._palette()
            window = Toplevel(widget)
            window.wm_overrideredirect(True)
            window.wm_geometry(f"+{x}+{y}")
            window.configure(bg=palette["border"])
            ttk.Label(
                window,
                text=text,
                style="Tooltip.TLabel",
                wraplength=320,
                justify="left",
                padding=(8, 5),
            ).pack()
            state["window"] = window

        def enter(_event=None) -> None:
            if state["after"] is None:
                state["after"] = self._after(600, show)

        def leave(_event=None) -> None:
            if state["after"] is not None:
                try:
                    self.root.after_cancel(state["after"])  # type: ignore[arg-type]
                except Exception:  # noqa: BLE001
                    pass
                state["after"] = None
            window = state["window"]
            if window is not None:
                try:
                    window.destroy()  # type: ignore[attr-defined]
                except Exception:  # noqa: BLE001
                    pass
                state["window"] = None

        widget.bind("<Enter>", enter, add="+")
        widget.bind("<Leave>", leave, add="+")
        widget.bind("<ButtonPress>", leave, add="+")

    def show_shortcuts(self) -> None:
        messagebox.showinfo(
            "Shortcuts and tips",
            "Reviewing\n"
            "  j / k or arrow keys   move between photos\n"
            "  a or g                Accept the active photo\n"
            "  d or b                REJECT the active photo\n"
            "  s                     Skip\n"
            "  Ctrl+Z / Ctrl+Y       undo / redo a decision\n"
            "  double-click          open the full-size viewer\n\n"
            "Tips\n"
            "  • The big picture at the top is the active one; Accept and REJECT apply to it.\n"
            "  • Every Accept and REJECT trains the scanner, and now carries over to other folders.\n"
            "  • 'Detected files' lists everything found, grouped by what was detected.\n"
            "  • Drag the sensitivity slider left to see near misses.",
        )

    def _palette(self) -> dict[str, str]:
        if self._theme_is_dark():
            # Three distinct depths (app < toolbar/card < input) so panels read as
            # surfaces instead of one flat grey, with text at ~13:1 contrast.
            return {
                "bg": "#17181c",
                "toolbar": "#1f2126",
                "panel": "#24262c",
                "fg": "#eceef2",
                "muted": "#9aa0ab",
                "entry_bg": "#2c2f36",
                "button_bg": "#31343c",
                "button_active": "#3d414b",
                "scale_trough": "#2c2f36",
                "border": "#3d414b",
                "arrow": "#c9ced8",
                "select_bg": "#2f4f7f",
                "select_fg": "#ffffff",
                "accent": "#4c8dff",
                "accent_fg": "#ffffff",
                "accent_active": "#639bff",
                "tooltip_bg": "#31343c",
            }
        return {
            "bg": "#eef0f4",
            "toolbar": "#f7f8fa",
            "panel": "#ffffff",
            "fg": "#1a1c20",
            "muted": "#5a6270",
            "entry_bg": "#ffffff",
            "button_bg": "#f0f1f4",
            "button_active": "#e2e5ea",
            "scale_trough": "#d5d8de",
            "border": "#c2c7d0",
            "arrow": "#1a1c20",
            "select_bg": "#d7e6ff",
            "select_fg": "#0d1a2b",
            "accent": "#2f6fdd",
            "accent_fg": "#ffffff",
            "accent_active": "#255cbd",
            "tooltip_bg": "#ffffff",
        }

    def _theme_is_dark(self) -> bool:
        theme = self.theme_var.get().strip().lower()
        if theme == "dark":
            return True
        if theme == "light":
            return False
        if theme == "system":
            return self._detect_system_dark_mode()
        return False

    @staticmethod
    def _detect_system_dark_mode() -> bool:
        if os.name == "nt":
            try:
                import winreg

                with winreg.OpenKey(
                    winreg.HKEY_CURRENT_USER,
                    r"Software\Microsoft\Windows\CurrentVersion\Themes\Personalize",
                ) as key:
                    value, _ = winreg.QueryValueEx(key, "AppsUseLightTheme")
                    return int(value) == 0
            except Exception:  # noqa: BLE001
                return False
        if platform.system() == "Darwin":
            try:
                result = subprocess.run(
                    ["defaults", "read", "-g", "AppleInterfaceStyle"],
                    capture_output=True,
                    text=True,
                    check=False,
                )
                return "Dark" in result.stdout
            except Exception:  # noqa: BLE001
                return False
        for name in ("GTK_THEME", "QT_STYLE_OVERRIDE"):
            value = os.environ.get(name, "").lower()
            if "dark" in value:
                return True
        return False

    def _apply_theme(self) -> None:
        palette = self._palette()
        style = ttk.Style()
        try:
            style.theme_use("clam")
        except Exception:  # noqa: BLE001
            pass
        self.root.configure(bg=palette["bg"])
        style.configure("TFrame", background=palette["panel"])
        style.configure("TLabel", background=palette["panel"], foreground=palette["fg"])
        style.configure(
            "TButton",
            background=palette["button_bg"],
            foreground=palette["fg"],
            borderwidth=1,
            focusthickness=2,
            focuscolor=palette["fg"],
            padding=(8, 5),
            bordercolor=palette["border"],
            # clam fakes 3D with light/dark edges; left at defaults they ring every
            # control in near-white. Matching them to the fill keeps controls flat.
            lightcolor=palette["button_bg"],
            darkcolor=palette["button_bg"],
        )
        style.map(
            "TButton",
            background=[("active", palette["button_active"]), ("pressed", palette["button_active"])],
            foreground=[("disabled", palette["muted"]), ("active", palette["fg"])],
            lightcolor=[("active", palette["button_active"]), ("pressed", palette["button_active"])],
            darkcolor=[("active", palette["button_active"]), ("pressed", palette["button_active"])],
        )
        style.configure(
            "TCheckbutton",
            background=palette["panel"],
            foreground=palette["fg"],
            # clam's indicator uses indicatorbackground/-foreground, not indicatorcolor.
            indicatorbackground=palette["entry_bg"],
            indicatorforeground=palette["fg"],
            upperbordercolor=palette["border"],
            lowerbordercolor=palette["border"],
            bordercolor=palette["border"],
            lightcolor=palette["panel"],
            darkcolor=palette["panel"],
        )
        style.map(
            "TCheckbutton",
            foreground=[("disabled", palette["muted"])],
            background=[("active", palette["panel"])],
            indicatorbackground=[
                ("disabled", palette["panel"]),
                ("selected", palette["accent"]),
                ("active", palette["button_active"]),
                ("!selected", palette["entry_bg"]),
            ],
            indicatorforeground=[("selected", palette["select_fg"])],
        )
        style.configure(
            "TEntry",
            fieldbackground=palette["entry_bg"],
            foreground=palette["fg"],
            insertcolor=palette["fg"],
            bordercolor=palette["border"],
            lightcolor=palette["entry_bg"],
            darkcolor=palette["entry_bg"],
        )
        style.map(
            "TEntry", fieldbackground=[("disabled", palette["panel"])], foreground=[("disabled", palette["muted"])]
        )
        # Combobox: the field, the arrow, and the popdown list are three separate surfaces.
        style.configure(
            "TCombobox",
            fieldbackground=palette["entry_bg"],
            background=palette["button_bg"],
            foreground=palette["fg"],
            arrowcolor=palette["arrow"],
            bordercolor=palette["border"],
            insertcolor=palette["fg"],
            lightcolor=palette["entry_bg"],
            darkcolor=palette["entry_bg"],
        )
        style.map(
            "TCombobox",
            fieldbackground=[("readonly", palette["entry_bg"]), ("disabled", palette["panel"])],
            foreground=[("disabled", palette["muted"])],
            background=[("active", palette["button_active"])],
            selectbackground=[("!focus", palette["entry_bg"])],
            selectforeground=[("!focus", palette["fg"])],
        )
        style.configure(
            "TSpinbox",
            fieldbackground=palette["entry_bg"],
            background=palette["button_bg"],
            foreground=palette["fg"],
            arrowcolor=palette["arrow"],
            bordercolor=palette["border"],
            insertcolor=palette["fg"],
            lightcolor=palette["entry_bg"],
            darkcolor=palette["entry_bg"],
        )
        style.map(
            "TSpinbox", fieldbackground=[("disabled", palette["panel"])], foreground=[("disabled", palette["muted"])]
        )
        for scale_style in ("Horizontal.TScale", "Vertical.TScale"):
            # `background` is the draggable thumb here, not the strip behind it.
            style.configure(
                scale_style,
                background=palette["button_active"],
                troughcolor=palette["scale_trough"],
                bordercolor=palette["border"],
                lightcolor=palette["button_active"],
                darkcolor=palette["button_active"],
            )
            style.map(scale_style, background=[("active", palette["accent"])])
        for bar_style in ("Vertical.TScrollbar", "Horizontal.TScrollbar"):
            style.configure(
                bar_style,
                background=palette["button_bg"],
                troughcolor=palette["scale_trough"],
                arrowcolor=palette["arrow"],
                bordercolor=palette["border"],
                lightcolor=palette["button_bg"],
                darkcolor=palette["button_bg"],
            )
            style.map(
                bar_style,
                background=[("active", palette["button_active"])],
                lightcolor=[("active", palette["button_active"])],
                darkcolor=[("active", palette["button_active"])],
            )
        for pbar_style in ("TProgressbar", "Horizontal.TProgressbar"):
            style.configure(
                pbar_style,
                background=palette["accent"],
                troughcolor=palette["scale_trough"],
                bordercolor=palette["border"],
                lightcolor=palette["accent"],
                darkcolor=palette["accent"],
            )
        style.configure("Card.TFrame", background=palette["panel"])
        style.configure("FocusedCard.TFrame", background=palette["select_bg"])
        # Match cards get an accent border so matches stand out from non-matches at a glance.
        style.configure(
            "MatchCard.TFrame",
            background=palette["panel"],
            bordercolor=palette["accent"],
            relief="solid",
            borderwidth=2,
        )
        style.configure(
            "FocusedMatchCard.TFrame",
            background=palette["select_bg"],
            bordercolor=palette["accent"],
            relief="solid",
            borderwidth=2,
        )
        # VLM badge in the status bar: accent background so it reads as an active feature.
        style.configure("Accent.TLabel", background=palette["accent"], foreground=palette["accent_fg"])
        # Chrome surfaces: toolbars sit a step above the app background.
        style.configure("Toolbar.TFrame", background=palette["toolbar"])
        style.configure("Toolbar.TLabel", background=palette["toolbar"], foreground=palette["fg"])
        style.configure("Muted.TLabel", background=palette["toolbar"], foreground=palette["muted"])
        # The results area sits on the app background, so anything drawn over it needs
        # matching styles or it shows up as a pale rectangle.
        style.configure("Surface.TFrame", background=palette["bg"])
        style.configure("SurfaceMuted.TLabel", background=palette["bg"], foreground=palette["muted"])
        style.configure(
            "Heading.TLabel",
            background=palette["bg"],
            foreground=palette["fg"],
            font=("TkDefaultFont", int(self.font_size_var.get()) + 6, "bold"),
        )
        style.configure(
            "Tooltip.TLabel",
            background=palette["tooltip_bg"],
            foreground=palette["fg"],
            borderwidth=0,
        )
        style.configure(
            "Accent.TButton",
            background=palette["accent"],
            foreground=palette["accent_fg"],
            bordercolor=palette["accent"],
            lightcolor=palette["accent"],
            darkcolor=palette["accent"],
            focuscolor=palette["accent_fg"],
            padding=(14, 5),
        )
        style.map(
            "Accent.TButton",
            background=[
                ("active", palette["accent_active"]),
                ("pressed", palette["accent_active"]),
                ("disabled", palette["button_bg"]),
            ],
            foreground=[("disabled", palette["muted"]), ("active", palette["accent_fg"])],
            lightcolor=[("active", palette["accent_active"]), ("pressed", palette["accent_active"])],
            darkcolor=[("active", palette["accent_active"]), ("pressed", palette["accent_active"])],
        )
        style.configure("MenuBar.TFrame", background=palette["bg"])
        style.configure(
            "MenuBar.TMenubutton",
            background=palette["bg"],
            foreground=palette["fg"],
            arrowsize=0,
            borderwidth=0,
            relief="flat",
            padding=(8, 3),
        )
        style.map(
            "MenuBar.TMenubutton",
            background=[("active", palette["button_active"]), ("pressed", palette["button_active"])],
            foreground=[("active", palette["fg"])],
        )
        self._apply_option_db(palette)
        self._apply_classic_widget_theme(palette)
        self._apply_dark_titlebar()
        self.grid_inner.configure(style="TFrame")
        if hasattr(self, "theme_button"):
            try:
                self.theme_button.configure(text=self._theme_button_text())
            except Exception:  # noqa: BLE001
                pass
        self._apply_ui_scale()
        self.status_var.set(self.status_var.get())
        self.summary_var.set(self.summary_var.get())
        self.root.update_idletasks()

    def _apply_option_db(self, palette: dict[str, str]) -> None:
        """Seed defaults for classic tk widgets, which ignore ttk styles.

        The option database only reaches widgets built after this runs, which covers
        every dialog; long-lived widgets are handled in _apply_classic_widget_theme.
        """
        options = {
            "*Menu.background": palette["panel"],
            "*Menu.foreground": palette["fg"],
            "*Menu.activeBackground": palette["select_bg"],
            "*Menu.activeForeground": palette["select_fg"],
            "*Menu.selectColor": palette["fg"],
            "*Menu.relief": "flat",
            "*Menu.borderWidth": 1,
            "*Text.background": palette["entry_bg"],
            "*Text.foreground": palette["fg"],
            "*Text.insertBackground": palette["fg"],
            "*Text.selectBackground": palette["select_bg"],
            "*Text.selectForeground": palette["select_fg"],
            "*Text.highlightBackground": palette["border"],
            "*Text.highlightColor": palette["border"],
            "*Listbox.background": palette["entry_bg"],
            "*Listbox.foreground": palette["fg"],
            "*Listbox.selectBackground": palette["select_bg"],
            "*Listbox.selectForeground": palette["select_fg"],
            "*Listbox.highlightBackground": palette["border"],
            "*Canvas.background": palette["bg"],
            "*Canvas.highlightBackground": palette["bg"],
            # The combobox dropdown is a Listbox inside a popdown toplevel and is not
            # reachable through ttk styling.
            "*TCombobox*Listbox.background": palette["entry_bg"],
            "*TCombobox*Listbox.foreground": palette["fg"],
            "*TCombobox*Listbox.selectBackground": palette["select_bg"],
            "*TCombobox*Listbox.selectForeground": palette["select_fg"],
        }
        for pattern, value in options.items():
            try:
                self.root.option_add(pattern, value)
            except Exception:  # noqa: BLE001
                continue

    def _apply_classic_widget_theme(self, palette: dict[str, str]) -> None:
        """Recolour already-built classic tk widgets, which the option DB cannot reach."""
        try:
            self.grid_canvas.configure(bg=palette["bg"], highlightbackground=palette["bg"])
        except Exception:  # noqa: BLE001
            pass
        try:
            self.queue_listbox.configure(
                bg=palette["entry_bg"],
                fg=palette["fg"],
                selectbackground=palette["select_bg"],
                selectforeground=palette["select_fg"],
                highlightbackground=palette["border"],
                highlightcolor=palette["border"],
                borderwidth=1,
                relief="flat",
            )
        except Exception:  # noqa: BLE001
            pass
        for menu in getattr(self, "menus", []):
            self._recolour_menu_tree(menu, palette)

    def _recolour_menu_tree(self, menu: Menu, palette: dict[str, str]) -> None:
        try:
            menu.configure(
                bg=palette["panel"],
                fg=palette["fg"],
                activebackground=palette["select_bg"],
                activeforeground=palette["select_fg"],
                selectcolor=palette["fg"],
                relief="flat",
                borderwidth=1,
            )
        except Exception:  # noqa: BLE001
            return
        # Cascades are separate Menu widgets; walk them so submenus match.
        try:
            end = menu.index("end")
        except Exception:  # noqa: BLE001
            return
        if end is None:
            return
        for index in range(int(end) + 1):
            try:
                if menu.type(index) != "cascade":
                    continue
                child_name = menu.entrycget(index, "menu")
            except Exception:  # noqa: BLE001
                continue
            child = self.root.nametowidget(child_name) if child_name else None
            if isinstance(child, Menu):
                self._recolour_menu_tree(child, palette)

    def _apply_dark_titlebar(self) -> None:
        """Ask DWM for a dark title bar; Tk does not opt in on its own."""
        if os.name != "nt":
            return
        try:
            import ctypes

            self.root.update_idletasks()
            hwnd = ctypes.windll.user32.GetParent(self.root.winfo_id())
            if not hwnd:
                return
            value = ctypes.c_int(1 if self._theme_is_dark() else 0)
            # 20 = DWMWA_USE_IMMERSIVE_DARK_MODE on Windows 10 1903+/11;
            # 19 was the pre-release attribute id on older builds.
            for attribute in (20, 19):
                result = ctypes.windll.dwmapi.DwmSetWindowAttribute(
                    hwnd, ctypes.c_int(attribute), ctypes.byref(value), ctypes.sizeof(value)
                )
                if result == 0:
                    break
        except Exception:  # noqa: BLE001
            return

    def _apply_ui_scale(self) -> None:
        scale = max(0.75, min(1.5, float(self.font_size_var.get()) / 10.0))
        try:
            self.root.tk.call("tk", "scaling", scale)
        except Exception:  # noqa: BLE001
            pass
        base_size = max(8, int(self.font_size_var.get()))
        for name in ("TkDefaultFont", "TkTextFont", "TkMenuFont", "TkHeadingFont", "TkCaptionFont"):
            try:
                tkfont.nametofont(name).configure(size=base_size)
            except Exception:  # noqa: BLE001
                continue

    def _bind_ui_prefs(self) -> None:
        for variable in (
            self.theme_var,
            self.font_size_var,
            self.columns_var,
            self.thumbnail_size_var,
            self.thumbnail_cache_size_var,
            self.search_var,
            self.sort_var,
            self.match_filter_var,
            self.label_filter_var,
            self.score_min_var,
            self.score_max_var,
            self.update_url_var,
        ):
            try:
                variable.trace_add("write", lambda *_args: self._on_ui_pref_change())
            except Exception:  # noqa: BLE001
                continue

    def _on_ui_pref_change(self) -> None:
        self._apply_theme()
        self._trim_thumbnail_cache()
        self._save_user_prefs()
        if self.current_state is not None:
            self._refresh_active_view()

    def _refresh_active_view(self) -> None:
        """Re-render results without knocking the user out of the current view.

        Changing a filter, the sort, the column count or the thumbnail size used to
        force the bucketed review view back on screen, which threw away the detected
        list every time.
        """
        if self.view_mode == "detected":
            samples = self._detected_samples()
            if samples:
                self.current_samples = samples
                self._refresh_displayed_results()
                return
        elif self.view_mode == "similar" and self.similar_anchor_path:
            self._refresh_displayed_results()
            return
        self._refresh_current_results()

    def _restore_ui_prefs(self) -> None:
        geometry = self.user_prefs.get("window_geometry")
        if isinstance(geometry, str) and geometry:
            try:
                self.root.geometry(geometry)
            except Exception:  # noqa: BLE001
                pass

    def _save_user_prefs(self) -> None:
        payload = {
            "theme": self.theme_var.get().strip(),
            "font_size": int(self.font_size_var.get()),
            "columns": int(self.columns_var.get()),
            "thumbnail_size": int(self.thumbnail_size_var.get()),
            "thumbnail_cache_size": self._thumbnail_cache_limit(),
            "search": self.search_var.get(),
            "sort": self.sort_var.get(),
            "match_filter": self.match_filter_var.get(),
            "label_filter": self.label_filter_var.get(),
            "score_min": self.score_min_var.get().strip(),
            "score_max": self.score_max_var.get().strip(),
            "update_url": self.update_url_var.get().strip(),
            "output_organization": self.output_organization_var.get(),
            "output_template": self.output_template_var.get(),
            "output_duplicate": self.output_duplicate_var.get(),
            "output_score_low": float(self.output_score_low_var.get()),
            "output_score_high": float(self.output_score_high_var.get()),
            "recent_folders": self.recent_folders[:10],
            "first_run_guide_shown": self._first_run_guide_shown,
            "window_geometry": self.root.geometry(),
        }
        self.user_prefs = payload
        try:
            save_user_prefs(payload)
        except Exception:  # noqa: BLE001
            pass

    def _thumbnail_cache_limit(self) -> int:
        try:
            return max(32, min(2048, int(self.thumbnail_cache_size_var.get())))
        except Exception:  # noqa: BLE001
            return 512

    def _trim_thumbnail_cache(self) -> None:
        limit = self._thumbnail_cache_limit()
        while len(self.thumbnail_cache) > limit:
            self.thumbnail_cache.popitem(last=False)

    def _maybe_show_first_run_guide(self) -> None:
        if self._first_run_guide_shown:
            return
        self._first_run_guide_shown = True
        self._save_user_prefs()
        self.show_guide()

    def _build_menu_bar(self) -> None:
        # Windows draws a root `menu=` menubar itself and ignores Tk colours, leaving a
        # white strip. Menubuttons in a ttk frame are ours to style.
        self.menu_bar_frame = ttk.Frame(self.root, style="MenuBar.TFrame", padding=(4, 2))
        self.menu_bar_frame.pack(side=TOP, fill="x")
        file_menu = Menu(self.menu_bar_frame, tearoff=False)
        view_menu = Menu(self.menu_bar_frame, tearoff=False)
        tools_menu = Menu(self.menu_bar_frame, tearoff=False)
        help_menu = Menu(self.menu_bar_frame, tearoff=False)
        recent_menu = Menu(file_menu, tearoff=False)

        file_menu.add_command(label="Open folder...", command=self.choose_folder)
        file_menu.add_command(label="Resume last scan", command=self.resume_last_scan)
        file_menu.add_cascade(label="Recent folders", menu=recent_menu)
        file_menu.add_separator()
        file_menu.add_command(label="Exit", command=self._on_close)

        view_menu.add_command(label="Detected files", command=self.show_detected_files)
        view_menu.add_command(label="Review queue", command=self.restore_review_view)
        view_menu.add_separator()
        view_menu.add_command(label="Light theme", command=lambda: self.theme_var.set("light"))
        view_menu.add_command(label="Dark theme", command=lambda: self.theme_var.set("dark"))
        view_menu.add_command(label="System theme", command=lambda: self.theme_var.set("system"))

        tools_menu.add_command(label="Settings", command=self.open_settings_dialog)
        tools_menu.add_command(label="Settings profiles", command=self.open_profiles_dialog)
        tools_menu.add_command(label="Import settings", command=self.import_settings)
        tools_menu.add_command(label="Export settings", command=self.export_settings)
        tools_menu.add_command(label="Output options", command=self.open_output_options_dialog)
        tools_menu.add_command(label="Prompt tester", command=self.open_prompt_tester_dialog)
        tools_menu.add_separator()
        tools_menu.add_command(label="Copy matches to subfolder", command=self.copy_matches_to_subfolder)
        tools_menu.add_command(label="Export matches (CSV)", command=self.export_matches)
        tools_menu.add_command(label="Export HTML report", command=self.export_html_report)
        tools_menu.add_command(label="Write metadata tags", command=self.write_metadata_to_visible)
        tools_menu.add_command(label="Accept everything shown", command=lambda: self.mark_all_shown(1))
        tools_menu.add_command(label="Reject everything shown", command=lambda: self.mark_all_shown(0))
        tools_menu.add_separator()
        tools_menu.add_command(label="Clear cache", command=self.clear_cache)
        tools_menu.add_command(label="Reset cross-folder learning", command=self.reset_global_learning)
        tools_menu.add_separator()
        tools_menu.add_command(label="Add folder to queue", command=self.add_folder_to_queue)
        tools_menu.add_command(label="Run queue", command=self.run_queue)
        tools_menu.add_command(label="Stop queue", command=self.stop_queue)
        tools_menu.add_separator()
        tools_menu.add_command(label="Trash visible files", command=self.trash_visible_files)
        tools_menu.add_command(label="Duplicate groups", command=self.show_duplicate_groups)

        help_menu.add_command(label="Guide", command=self.show_guide)
        help_menu.add_command(label="About", command=self.show_about)
        help_menu.add_command(label="Log viewer", command=self.show_log_viewer)
        help_menu.add_command(label="Check for updates", command=self.check_for_updates)

        self.menus = [file_menu, view_menu, tools_menu, help_menu, recent_menu]
        for label, menu in (
            ("File", file_menu),
            ("View", view_menu),
            ("Tools", tools_menu),
            ("Help", help_menu),
        ):
            button = ttk.Menubutton(
                self.menu_bar_frame, text=label, menu=menu, direction="below", style="MenuBar.TMenubutton"
            )
            button.pack(side=LEFT, padx=(0, 2))
        self.recent_menu = recent_menu
        self._rebuild_recent_menu()

    def _apply_window_icon(self) -> None:
        try:
            icon_path = Path(__file__).resolve().parents[1] / "assets" / "bikini_scanner.png"
            if icon_path.exists():
                with Image.open(icon_path) as image:
                    icon = ImageTk.PhotoImage(image.copy())
                self._window_icon = icon
                self.root.iconphoto(True, icon)  # type: ignore[arg-type]
        except Exception:  # noqa: BLE001
            return

    def _enable_drag_and_drop(self) -> None:
        if TkinterDnD is None or DND_FILES is None:
            return
        try:
            self.root.drop_target_register(DND_FILES)  # type: ignore[attr-defined]
            self.root.dnd_bind("<<Drop>>", self._on_drop_files)  # type: ignore[attr-defined]
        except Exception:  # noqa: BLE001
            return

    def _on_drop_files(self, event) -> str:
        data = getattr(event, "data", "")
        if not data:
            return "break"
        paths = self._parse_dropped_paths(data)
        # A directory drop wins outright: scan that folder. When the user drops image
        # files instead (e.g. straight from a file manager), open the folder that
        # contains them so the scan picks them up. Without this the drop did nothing
        # and gave no feedback, which read as "drag-and-drop is broken".
        folders = [path for path in paths if Path(path).is_dir()]
        if folders:
            self.open_folder(folders[0], scan=True)
            return "break"
        image_suffixes = {suffix.lower() for suffix in SUPPORTED_IMAGE_SUFFIXES}
        image_paths = [path for path in paths if Path(path).is_file() and Path(path).suffix.lower() in image_suffixes]
        if image_paths:
            self.open_folder(str(Path(image_paths[0]).parent), scan=True)
            return "break"
        self.status_var.set("Drop a folder or image files to scan them.")
        return "break"

    @staticmethod
    def _parse_dropped_paths(data: str) -> list[str]:
        items = []
        token = ""
        quoted = False
        for char in data.strip():
            if char == "{":
                quoted = True
                token = ""
            elif char == "}":
                quoted = False
                if token:
                    items.append(token)
                token = ""
            elif char.isspace() and not quoted:
                if token:
                    items.append(token)
                    token = ""
            else:
                token += char
        if token:
            items.append(token)
        return items

    def _rebuild_recent_menu(self) -> None:
        if not hasattr(self, "recent_menu"):
            return
        self.recent_menu.delete(0, END)
        if not self.recent_folders:
            self.recent_menu.add_command(label="No recent folders", state="disabled")
            return
        for folder in self.recent_folders[:10]:
            self.recent_menu.add_command(
                label=folder, command=lambda value=folder: self._open_recent_folder(value)  # type: ignore[misc]
            )

    def _open_recent_folder(self, folder: str) -> None:
        self.open_folder(folder, scan=True)

    def _add_recent_folder(self, folder: str) -> None:
        folder = str(Path(folder).expanduser().resolve())
        self.recent_folders = [item for item in self.recent_folders if item != folder]
        self.recent_folders.insert(0, folder)
        self.recent_folders = self.recent_folders[:10]
        self._rebuild_recent_menu()
        self._save_user_prefs()

    def choose_folder(self) -> None:
        folder = filedialog.askdirectory(title="Choose image folder")
        if folder:
            self.open_folder(folder, scan=False)

    def _backend_summary(self) -> str:
        if self.backend is None:
            return f"{self.config.backend} pending"
        device = getattr(self.backend, "active_device", "cpu")
        precision = getattr(self.backend, "active_precision", "fp32")
        return f"{self.config.backend} {device}/{precision}"

    def _ensure_backend(self, show_errors: bool = True) -> bool:
        if self.backend is not None and self.scorer is not None:
            return True
        try:
            from .clip_backend import get_backend
            self.backend = get_backend(self.config)
            return True
        except Exception as exc:
            LOGGER.exception("Backend load failed")
            if show_errors:
                messagebox.showerror("Could not load the scanning model", self._model_error_text(exc))
            return False

    def _model_error_text(self, exc: Exception) -> str:
        """Explain a model load failure in terms the user can act on.

        The first launch of a packaged build has to download the CLIP weights, so the
        common failure here is 'no internet', not a real bug — and the raw exception
        for that is a wall of Hugging Face stack text.
        """
        detail = str(exc).strip()
        lowered = detail.lower()
        offline_markers = (
            "connection",
            "connect",
            "offline",
            "timed out",
            "timeout",
            "network",
            "resolve",
            "proxy",
            "ssl",
            "max retries",
        )
        if any(marker in lowered for marker in offline_markers):
            return (
                f"The scanning model could not be downloaded.\n\n"
                f"The first scan needs internet access to fetch the CLIP model "
                f"({self.config.model_name}). After that it is cached and the app works offline.\n\n"
                f"Check your connection or proxy settings and try again.\n\nDetails: {detail[:300]}"
            )
        if "not a local folder" in lowered or "repo" in lowered or "404" in lowered:
            return (
                f"The model name in Settings does not appear to exist:\n\n{self.config.model_name}\n\n"
                f"Restore it to openai/clip-vit-base-patch32 in Tools > Settings.\n\nDetails: {detail[:300]}"
            )
        return f"The scanning model could not be loaded.\n\nDetails: {detail[:500]}"

    def _ensure_scorer(self) -> bool:
        if self.scorer is not None:
            return True
        if self.backend is None and not self._ensure_backend():
            return False
        assert self.backend is not None
        try:
            self.scorer = BikiniScorer(self.backend, self.config)
            return True
        except Exception as exc:  # noqa: BLE001
            messagebox.showerror("Model load error", str(exc))
            return False

    def _maybe_preload_backend(self) -> None:
        if self._backend_preload_started or not self.config.preload_backend:
            return
        self._backend_preload_started = True
        self.loading_var.set("Loading model...")
        self._show_progress(True)
        self._start_indeterminate_progress("Loading the scanning model…")

        def worker() -> None:
            loaded = self._ensure_backend(show_errors=False)

            def finished() -> None:
                self.loading_var.set("")
                # Either way the bar must leave indeterminate mode, or a later scan
                # keeps animating instead of showing real percentages. A scan started
                # while the model was still loading already owns the bar, so leave it be.
                if not self._scan_active:
                    self._reset_progress_bar()
                    self._show_progress(False)
                if loaded:
                    self._refresh_hardware_status()

            if not loaded:
                self._backend_preload_started = False
            self._after(0, finished)

        threading.Thread(target=worker, daemon=True).start()

    def _reset_progress_bar(self, text: str = "", detail: str = "") -> None:
        """Return the bar to a determinate zero; the preloader leaves it spinning."""
        try:
            self.progress_bar.stop()
            self.progress_bar.configure(mode="determinate")
        except Exception:  # noqa: BLE001
            pass
        self.progress_var.set(0.0)
        self.progress_text_var.set(text)
        self.progress_detail_var.set(detail)

    def _start_indeterminate_progress(self, detail: str) -> None:
        """For work whose size is not known in advance (model load, retrain)."""
        self.progress_detail_var.set(detail)
        self.progress_text_var.set("")
        try:
            self.progress_bar.configure(mode="indeterminate")
            self.progress_bar.start(12)
        except Exception:  # noqa: BLE001
            pass

    def _scan_progress_update(self, progress: ScanProgress) -> None:
        percent = progress.percent
        self.progress_var.set(percent)
        self.progress_text_var.set(f"{percent:.0f}%")

        parts = [progress.text()]
        if progress.rate > 0:
            parts.append(f"{progress.rate:.1f}/s")
        if progress.eta_seconds is not None and progress.eta_seconds >= 0:
            parts.append(f"ETA {time.strftime('%M:%S', time.gmtime(min(progress.eta_seconds, 359_999)))}")
        self.progress_detail_var.set("   ·   ".join(parts))

        if progress.phase == PHASE_EMBED and progress.total >= 10_000 and progress.done == 0:
            self.status_var.set(
                f"Large folder detected ({progress.total:,} images); using incremental cache flushing..."
            )
            return
        self.status_var.set(f"Scanning... {percent:.0f}%")

    def _load_last_folder(self) -> str:
        if not LAST_FOLDER_STATE_PATH.exists():
            return ""
        try:
            with LAST_FOLDER_STATE_PATH.open("r", encoding="utf-8") as handle:
                payload = json.load(handle)
            folder = str(payload.get("folder", ""))
            return folder if folder else ""
        except Exception as exc:  # noqa: BLE001
            LOGGER.warning("Ignoring unreadable last-folder state %s: %s", LAST_FOLDER_STATE_PATH, exc)
            quarantine_broken_file(LAST_FOLDER_STATE_PATH, LOGGER, "invalid JSON")
            return ""

    def _save_last_folder(self, folder: str) -> None:
        try:
            atomic_write_json(LAST_FOLDER_STATE_PATH, {"folder": folder})
        except Exception:  # noqa: BLE001
            pass

    def _resume_last_folder_if_any(self) -> None:
        if self.folder_var.get().strip():
            return
        folder = self._load_last_folder()
        if folder and Path(folder).exists():
            # Preselect the folder only. Scanning is an explicit user action, so
            # _set_folder (not folder_var.set) also primes self.store for Run scan.
            self._set_folder(folder)
            self.status_var.set(f"Ready — {folder}. Press Run scan to start.")

    def resume_last_scan(self) -> None:
        folder = self._load_last_folder()
        if not folder or not Path(folder).is_dir():
            messagebox.showinfo("Resume scan", "No existing last-scanned folder was found.")
            return
        self.open_folder(folder, scan=True)

    def _session_payload(self) -> dict[str, object]:
        return {
            "focused_path": self.focused_path,
            "view_mode": self.view_mode,
            "current_samples": self.current_samples,
            "review_samples": self.review_samples,
            "quality_history": list(self.quality_history),
        }

    def _save_review_session(self) -> None:
        if self.store is None:
            return
        try:
            self.store.save_review_session(self._session_payload())
        except Exception:  # noqa: BLE001
            pass

    def _restore_review_session(self) -> None:
        if self.store is None:
            return
        payload = self.store.load_review_session()
        if not payload:
            return
        focused_path = payload.get("focused_path")
        if isinstance(focused_path, str) and focused_path:
            self.focused_path = focused_path
        quality_history = payload.get("quality_history")
        if isinstance(quality_history, list):
            self.quality_history.clear()
            for value in quality_history:
                try:
                    self.quality_history.append(float(value))
                except Exception:  # noqa: BLE001
                    continue

    def _update_stats_panel(self, record_history: bool = False) -> None:
        if self.current_state is None:
            self.stats_var.set("")
            self.notice_var.set("")
            return
        labels = self.store.load_labels() if self.store is not None else {}
        counts = (
            self.scorer.label_counts(labels)
            if self.scorer is not None
            else {"good": 0, "bad": 0, "skip": 0, "unlabeled": 0}
        )
        unlabeled = sum(1 for path in self.current_state.paths if path not in labels)
        counts["unlabeled"] = unlabeled
        quality = None
        if self.scorer is not None:
            embeddings_by_path = dict(zip(self.current_state.paths, self.current_state.embeddings, strict=False))
            quality = self.scorer.estimate_quality(embeddings_by_path, labels)
        if record_history and quality is not None:
            self.quality_history.append(quality)
        plateau = ""
        if len(self.quality_history) >= 3:
            recent = list(self.quality_history)[-3:]
            if max(recent) - min(recent) < 0.015:
                plateau = " Labeling may be plateauing."
        quality_text = f"quality {quality:.3f}" if quality is not None else "quality n/a"
        learning_text = self.current_state.learning_summary or "no model yet"
        excluded_text = ""
        if self.current_state.excluded is not None:
            excluded = int(np.count_nonzero(self.current_state.excluded))
            if excluded:
                gated = sum(1 for stage in self.current_state.cascade_stage if stage == "minor")
                excluded_text = f" | {excluded} filtered out" + (f" ({gated} age-gated)" if gated else "")
        self.stats_var.set(
            f"Accepted {counts['good']} | Rejected {counts['bad']} | Skipped {counts['skip']} | "
            f"Unlabeled {counts['unlabeled']} | {quality_text} | learning: {learning_text}{excluded_text}"
        )
        self.notice_var.set(plateau)

    def _session_focus(self) -> None:
        # Always keep one card active so the enlarged preview has something to show.
        if self.displayed_samples and (not self.focused_path or self.focused_path not in self.cards):
            self.focused_path = str(self.displayed_samples[0]["path"])
        self._apply_focus_visuals()

    def _append_queue_item(self, folder: str) -> None:
        folder = str(Path(folder).expanduser().resolve())
        if folder in self.scan_queue:
            return
        self.scan_queue.append(folder)
        self.queue_listbox.insert(END, folder)

    def add_folder_to_queue(self) -> None:
        folder = filedialog.askdirectory(title="Add folder to scan queue")
        if folder:
            self._append_queue_item(folder)

    def queue_current_folder(self) -> None:
        folder = self.folder_var.get().strip()
        if folder:
            self._append_queue_item(folder)

    def run_queue(self) -> None:
        if self.queue_active:
            return
        if not self.scan_queue and self.folder_var.get().strip():
            self.queue_current_folder()
        if not self.scan_queue:
            messagebox.showinfo("Queue empty", "Add at least one folder to the queue.")
            return
        self.queue_active = True
        self.queue_index = 0
        self.status_var.set(f"Queue started with {len(self.scan_queue)} folders.")
        self._start_next_queue_item()

    def _start_next_queue_item(self) -> None:
        if not self.queue_active:
            return
        if self.queue_index >= len(self.scan_queue):
            self._finish_queue()
            return
        folder = self.scan_queue[self.queue_index]
        self.open_folder(folder, scan=False)
        self.status_var.set(f"Queue item {self.queue_index + 1}/{len(self.scan_queue)}: {Path(folder).name}")
        # _set_folder clears the scorer along with the backend (a folder override can
        # change the model), so both have to be rebuilt before the scan is launched.
        if not self._ensure_scorer():
            self.queue_active = False
            return
        self._launch_background_scan(full_rescan=True)

    def _finish_queue(self) -> None:
        self.queue_active = False
        if self.scan_queue:
            self.root.bell()
            self.status_var.set(f"Queue complete — {len(self.scan_queue)} folders processed.")
            messagebox.showinfo("Queue complete", f"Completed {len(self.scan_queue)} queued folders.")

    def stop_queue(self) -> None:
        self.queue_active = False
        self.status_var.set("Queue stopped.")

    def _toggle_watch_mode(self) -> None:
        if self.watch_enabled_var.get():
            if self.current_state is not None:
                self._watch_snapshot = self._collect_watch_snapshot()
            self._schedule_watch_poll()
            self.status_var.set("Watch mode enabled.")
        else:
            self.status_var.set("Watch mode disabled.")
            self._cancel_watch_poll()

    def _cancel_watch_poll(self) -> None:
        """Cancel the pending poll as well as forgetting it.

        Only clearing the id left the timer running, so turning watch mode off and back
        on again started a second polling chain on top of the first.
        """
        if self._watch_after_id is not None:
            try:
                self.root.after_cancel(self._watch_after_id)
            except Exception:  # noqa: BLE001
                pass
            self._watch_after_id = None

    def _schedule_watch_poll(self) -> None:
        if self._closing or not self.watch_enabled_var.get():
            return
        self._cancel_watch_poll()
        self._watch_after_id = self._after(5000, self._watch_poll)

    def _collect_watch_snapshot(self) -> dict[str, tuple[int, int]]:
        if self.store is None:
            return {}
        root = self.store.folder
        cache_dir = root / ".bikini_scanner_cache"
        matches_dir = root / MATCHES_DIR_NAME
        snapshot: dict[str, tuple[int, int]] = {}
        pending = [root]
        while pending:
            directory = pending.pop()
            try:
                entries = os.scandir(directory)
            except OSError:
                continue
            with entries:
                for entry in entries:
                    entry_path = Path(entry.path)
                    try:
                        if entry.is_dir(follow_symlinks=False):
                            if entry_path not in {cache_dir, matches_dir}:
                                pending.append(entry_path)
                            continue
                        if not entry.is_file(follow_symlinks=False):
                            continue
                        if entry_path.suffix.lower() not in SUPPORTED_IMAGE_SUFFIXES:
                            continue
                        stat = entry.stat(follow_symlinks=False)
                    except OSError:
                        continue
                    snapshot[str(entry_path.resolve())] = (stat.st_mtime_ns, stat.st_size)
        return snapshot

    def _watch_poll(self) -> None:
        self._watch_after_id = None
        if not self.watch_enabled_var.get() or self.queue_active or self._scan_active or self.current_state is None:
            self._schedule_watch_poll()
            return
        if self.store is None:
            self._schedule_watch_poll()
            return
        try:
            current_snapshot = self._collect_watch_snapshot()
        except Exception:  # noqa: BLE001
            self._schedule_watch_poll()
            return
        if current_snapshot != self._watch_snapshot:
            self._watch_snapshot = current_snapshot
            self.status_var.set("Watch folder detected changes — rescanning...")
            self.open_folder(self.folder_var.get(), scan=True)
        self._schedule_watch_poll()

    def _on_close(self) -> None:
        self._closing = True
        if self._scan_cancel_event is not None:
            self._scan_cancel_event.set()
        self.queue_active = False
        # Every scheduled after() callback must be cancelled before destroy(), or it
        # fires into a half-torn-down interpreter and raises TclError. _watch_after_id
        # was the only one cancelled here for a long time; the hardware-status,
        # threshold-refresh, and preview-resize timers leaked and could crash on exit.
        for after_id in (
            self._watch_after_id,
            self._preview_resize_after_id,
            self._threshold_refresh_after_id,
            self._hardware_after_id,
        ):
            if after_id is not None:
                try:
                    self.root.after_cancel(after_id)
                except Exception:  # noqa: BLE001
                    pass
        self._watch_after_id = None
        self._preview_resize_after_id = None
        self._threshold_refresh_after_id = None
        self._hardware_after_id = None
        self._save_review_session()
        self._save_user_prefs()
        self.root.destroy()

    def open_folder(self, folder: str, *, scan: bool = False) -> None:
        self._set_folder(folder)
        if scan:
            self.run_scan()

    def _set_folder(self, folder: str, scan: bool = False) -> None:
        folder = str(Path(folder).expanduser().resolve())
        self.config = ScannerConfig.from_mapping(self.global_config.to_dict())
        folder_store = FolderStore(Path(folder))
        self.store = folder_store
        # The override file lives inside the scanned folder, so it is only as
        # trustworthy as that folder: anything that could reach off this machine, run
        # code, or weaken the age gate is refused and reported rather than applied.
        override, refused = filter_folder_override(folder_store.load_config_override())
        self.folder_override_active = bool(override)
        if override:
            self.config = ScannerConfig.from_mapping({**self.global_config.to_dict(), **override})
        if refused:
            LOGGER.warning(
                "Ignored %d restricted key(s) in the folder override for %s: %s",
                len(refused),
                folder,
                ", ".join(refused),
            )
            messagebox.showwarning(
                "Folder override restricted",
                f"{folder}\n\nThis folder carries a settings override that tried to change "
                f"{len(refused)} setting(s) a folder is not allowed to change:\n\n"
                f"{', '.join(refused)}\n\n"
                "Those were ignored and your global settings kept. The remaining override "
                "settings were applied.",
            )
        self.backend = None
        self.scorer = None
        self.override_var.set("Folder override active" if self.folder_override_active else "")
        self.thumbnail_cache.clear()
        self.current_state = None
        self.current_samples = []
        self.displayed_samples = []
        self.review_samples = []
        self.undo_stack.clear()
        self.redo_stack.clear()
        self.focused_path = None
        self.folder_var.set(folder)
        self.status_var.set(f"Selected {folder}")
        self._refresh_empty_state()
        self._save_last_folder(folder)
        self._add_recent_folder(folder)
        self._watch_snapshot = {}
        if scan:
            self.run_scan()

    def _save_folder_override(self) -> None:
        folder = self.folder_var.get().strip()
        if not folder:
            messagebox.showinfo("Folder override", "Choose or scan a folder first.")
            return
        self.store = self.store or FolderStore(Path(folder))
        # Only persist what a folder override is allowed to carry, so reopening the
        # folder does not warn about restricted keys this app wrote itself.
        payload, _refused = filter_folder_override(self.config.to_dict())
        self.store.save_config_override(payload)
        self.folder_override_active = True
        self.override_var.set("Folder override active")
        self.status_var.set("Saved settings override for this folder.")

    def _reset_folder_override(self) -> None:
        if self.store is None:
            return
        self.store.clear_config_override()
        self.folder_override_active = False
        self.config = ScannerConfig.from_mapping(self.global_config.to_dict())
        self.override_var.set("")
        self.status_var.set("Folder override reset to global settings.")

    def show_guide(self) -> None:
        messagebox.showinfo(
            "Bikini Scanner guide",
            "1. Choose a folder and run a scan.\n"
            "2. Use the grid to review images with Accept, REJECT, or Skip.\n"
            "3. Use filters, search, sorting, and the image viewer to narrow the set.\n"
            "4. Open Tools > Settings to adjust scoring and hardware options.\n"
            "5. Recently scanned folders appear in File > Recent folders.",
        )

    def show_about(self) -> None:
        messagebox.showinfo(
            "About Bikini Scanner",
            f"Bikini Scanner {__version__}\n"
            "Local CPU-first bikini-content scanner with optional active learning, review tools, and per-folder caches.\n"
            f"User data: {prefs_path().parent}\n"
            f"Log file: {log_path()}",
        )

    def check_for_updates(self) -> None:
        url = self.update_url_var.get().strip()
        if not url:
            messagebox.showinfo("Check for updates", "No update URL is configured. Set one in Settings.")
            return
        self.status_var.set("Checking for updates...")

        def worker() -> None:
            result = check_for_update(url)
            self._after(0, lambda: self._show_update_result(result))

        threading.Thread(target=worker, daemon=True).start()

    def _show_update_result(self, result: dict[str, str] | None) -> None:
        self.status_var.set("Update check complete.")
        if result is None:
            messagebox.showinfo(
                "Check for updates", f"You are up to date ({__version__}), or the update server was unreachable."
            )
            return
        download_url = result.get("download_url", "")
        message = f"Version {result['latest_version']} is available."
        if download_url:
            message += f"\n\nDownload: {download_url}"
        messagebox.showinfo("Update available", message)

    def _toggle_nsfw_only(self) -> None:
        self.config.nsfw_filter = "only" if self.nsfw_only_var.get() else "include"
        self.status_var.set(f"NSFW filter set to {self.config.nsfw_filter}.")
        if self.current_state is not None:
            self._refresh_active_view()

    def _result_visibility_mask(self) -> np.ndarray:
        if self.current_state is None:
            return np.empty((0,), dtype=bool)
        if self.scorer is None:
            return np.ones(len(self.current_state.paths), dtype=bool)
        # state_visibility also applies the cascade's own exclusions (age gate etc).
        return self.scorer.state_visibility(self.current_state)

    def _visible_matches(self) -> list[str]:
        if self.current_state is None:
            return []
        threshold = float(self.threshold_var.get())
        mask = self._result_visibility_mask()
        return [
            path
            for path, score, include in zip(self.current_state.paths, self.current_state.scores, mask, strict=False)
            if include and score >= threshold
        ]

    def _current_samples_from_state(self) -> list[dict[str, object]]:
        if self.current_state is None:
            return []
        if self.scorer is None:
            return self.current_samples
        mask = self._result_visibility_mask()
        paths = [path for path, include in zip(self.current_state.paths, mask, strict=False) if include]
        scores = [score for score, include in zip(self.current_state.scores, mask, strict=False) if include]
        embeddings = [
            embedding for embedding, include in zip(self.current_state.embeddings, mask, strict=False) if include
        ]
        labels = self.store.load_labels().keys() if self.store is not None else []
        return bucketed_sampling(
            paths,
            scores,
            labels,
            embeddings=embeddings,
            threshold=float(self.threshold_var.get()),
            # Same signal the scan itself passes. Leaving it out here meant the
            # "Model disagrees" bucket — the most useful things to label — silently
            # disappeared as soon as a filter, sort or threshold rebuilt the queue.
            disagreement=state_disagreement(self.current_state, mask),
        )

    def _post_process_review_samples(self, samples: list[dict[str, object]]) -> list[dict[str, object]]:
        if self.current_state is None or not samples:
            return list(samples)
        try:
            processed = apply_plugins(
                self.current_state, list(samples), enabled=self.config.enable_plugins
            )
        except Exception:
            LOGGER.exception("Review sample plugin processing failed; using unmodified samples.")
            return list(samples)
        return list(processed)

    def _sample_sort_key(self, sample: dict[str, object]) -> tuple[object, ...]:
        path = str(sample["path"])
        score = float(cast(float, sample.get("score", 0.0)))
        stat = None
        try:
            stat = Path(path).stat()
        except Exception:  # noqa: BLE001
            stat = None
        sort_mode = self.sort_var.get().strip()
        if sort_mode == "filename":
            return (Path(path).name.lower(), -score, path)
        if sort_mode == "date":
            return (-(stat.st_mtime if stat is not None else 0.0), -score, path)
        bucket_order = {
            "Likely match": 0,
            "Likely false positive": 1,
            "Likely false negative": 2,
            "Uncertain": 3,
        }
        # Detected-view buckets sort after the review buckets and in their listed order.
        bucket_order.update({name: 10 + position for position, name in enumerate(DETECTED_BUCKETS)})
        return (bucket_order.get(str(sample.get("bucket", "Uncertain")), 20), -score, Path(path).name.lower(), path)

    def _score_range(self) -> tuple[float | None, float | None]:
        def parse(value: str) -> float | None:
            value = value.strip()
            if not value:
                return None
            try:
                return max(0.0, min(1.0, float(value)))
            except ValueError:
                return None

        return parse(self.score_min_var.get()), parse(self.score_max_var.get())

    def _sample_visible(self, sample: dict[str, object]) -> bool:
        if self.current_state is None:
            return False
        path = str(sample["path"])
        search = self.search_var.get().strip().lower()
        label = self._label_text(path)
        bucket = str(sample.get("bucket", ""))
        if search:
            searchable = f"{path} {label} {bucket}".lower()
            if search not in searchable:
                return False
        label_mode = self.label_filter_var.get().strip()
        if label_mode == "unlabeled" and label != "unlabeled":
            return False
        if label_mode == "labeled" and label == "unlabeled":
            return False
        if label_mode == "skipped" and label != "skip":
            return False
        match_mode = self.match_filter_var.get().strip()
        if match_mode != "all":
            threshold = float(self.threshold_var.get())
            matches = float(cast(float, sample.get("score", 0.0))) >= threshold
            if match_mode == "matched" and not matches:
                return False
            if match_mode == "unmatched" and matches:
                return False
        score = float(cast(float, sample.get("score", 0.0)))
        score_min, score_max = self._score_range()
        if score_min is not None and score < score_min:
            return False
        if score_max is not None and score > score_max:
            return False
        return not (score_min is not None and score_max is not None and score_min > score_max)

    def _apply_display_filters(self, samples: list[dict[str, object]]) -> list[dict[str, object]]:
        filtered = [sample for sample in samples if self._sample_visible(sample)]
        return sorted(filtered, key=self._sample_sort_key)

    def _refresh_current_results(self) -> None:
        if self.current_state is None:
            return
        self.current_samples = self._post_process_review_samples(self._current_samples_from_state())
        self.review_samples = list(self.current_samples)
        self.view_mode = "review"
        self.similar_anchor_path = None
        self._refresh_displayed_results()

    def _refresh_displayed_results(self) -> None:
        self.displayed_samples = self._apply_display_filters(self.current_samples)
        self._refresh_summary()
        self._render_samples()
        self._save_review_session()

    def _axis_details_text(self, path: str) -> str:
        if self.current_state is None:
            return ""
        try:
            index = self.current_state.paths.index(path)
        except ValueError:
            return ""
        state = self.current_state
        parts: list[str] = []
        # Evidence values (0 = the axis saw nothing), which is what the gates compare.
        for label, axis_name in (
            ("bikini", "bikini"),
            ("cleav", "cleavage"),
            ("midriff", "midriff"),
            ("top", "bikini_top"),
            ("btm", "bikini_bottom"),
        ):
            axis_scores = state.axis_scores.get(axis_name)
            if axis_scores is None or index >= len(axis_scores):
                continue
            parts.append(f"{label} {cascade.evidence(np.asarray([axis_scores[index]]))[0]:.2f}")
        for label, axis_name in (("female", "evidence_female"), ("adult", "evidence_adult")):
            axis_scores = state.axis_scores.get(axis_name)
            if axis_scores is not None and index < len(axis_scores):
                parts.append(f"{label} {axis_scores[index]:.2f}")
        if state.face_counts is not None and index < len(state.face_counts):
            value = int(state.face_counts[index])
            if value >= 0:
                parts.append(f"faces {value}")
        if index < len(state.detail_regions) and state.detail_regions[index] not in ("", "full"):
            parts.append(f"crop {state.detail_regions[index]}")
        if index < len(state.cascade_reason) and state.cascade_reason[index]:
            parts.append(f"— {state.cascade_reason[index]}")
        return "  ".join(parts)

    def _match_score_for_path(self, path: str) -> float:
        if self.current_state is None:
            return 0.0
        for candidate, score in zip(self.current_state.paths, self.current_state.scores, strict=False):
            if candidate == path:
                return float(score)
        return 0.0

    def _detected_bucket(self, index: int | None) -> str:
        """Name the strongest thing the scanner detected in one image."""
        state = self.current_state
        if state is None or index is None:
            return DETECTED_BUCKETS[-1]

        def axis(name: str) -> float:
            scores = state.axis_scores.get(name)
            if scores is None or index >= len(scores):
                return 0.0
            # Evidence, not the raw sigmoid: 0.5 raw means the axis saw nothing.
            return float(cascade.evidence(np.asarray([scores[index]]))[0])

        candidates = {
            "Cleavage": axis("cleavage"),
            "Bikini": max(axis("bikini"), axis("bikini_top"), axis("bikini_bottom")),
            "Midriff": axis("midriff"),
            "Explicit (NSFW)": axis("nsfw"),
        }
        best = max(candidates, key=lambda name: candidates[name])
        # Below this the axes are all just noise, so naming one of them would be a lie.
        if candidates[best] < 0.2:
            return DETECTED_BUCKETS[-1]
        return best

    def _detected_samples(self) -> list[dict[str, object]]:
        """Every file above the threshold, grouped by what was detected in it."""
        if self.current_state is None:
            return []
        scores = self._score_map()
        index_by_path = {path: index for index, path in enumerate(self.current_state.paths)}
        return [
            {
                "path": path,
                "score": scores.get(path, 0.0),
                "bucket": self._detected_bucket(index_by_path.get(path)),
            }
            for path in self._visible_matches()
        ]

    def show_detected_files(self) -> None:
        if self.current_state is None:
            messagebox.showinfo("No results", "Run a scan first.")
            return
        samples = self._detected_samples()
        if not samples:
            messagebox.showinfo("No detected files", "No files scored above the current threshold.")
            return
        self.view_mode = "detected"
        self.similar_anchor_path = None
        self.current_samples = samples
        threshold = float(self.threshold_var.get())
        self.status_var.set(
            f"{len(samples)} detected files at threshold {threshold:.3f}, grouped by what was detected. "
            "Switch to 'Review queue' to teach the scanner."
        )
        self._refresh_displayed_results()

    def restore_review_view(self) -> None:
        if self.review_samples:
            self.view_mode = "review"
            self.similar_anchor_path = None
            self.current_samples = list(self.review_samples)
            self.status_var.set("Review view restored.")
            self._refresh_displayed_results()
            self._refresh_summary()
            self._session_focus()
        else:
            messagebox.showinfo("No review view", "Run or update a scan first.")

    def find_similar(self, anchor_path: str) -> None:
        if self.current_state is None:
            messagebox.showinfo("No results", "Run a scan first.")
            return
        ranking = self._rank_similar_paths(anchor_path)
        if not ranking:
            messagebox.showinfo("No results", "No similar images found.")
            return
        self.view_mode = "similar"
        self.similar_anchor_path = anchor_path
        self.current_samples = [
            {"path": path, "score": similarity, "bucket": "Similar"} for path, similarity in ranking
        ]
        self.status_var.set(f"Showing similar images to {Path(anchor_path).name}.")
        self._refresh_displayed_results()
        self._refresh_summary()

    def _rank_similar_paths(self, anchor_path: str) -> list[tuple[str, float]]:
        if self.current_state is None:
            return []
        try:
            anchor_index = self.current_state.paths.index(anchor_path)
        except ValueError:
            return []
        visibility = self._result_visibility_mask()
        anchor_embedding = np.asarray(self.current_state.embeddings[anchor_index], dtype=np.float32)
        anchor_norm = float(np.linalg.norm(anchor_embedding))
        if anchor_norm <= 0:
            return []
        ranked: list[tuple[str, float]] = []
        for _index, (path, embedding, include) in enumerate(
            zip(self.current_state.paths, self.current_state.embeddings, visibility, strict=False)
        ):
            if not include or path == anchor_path:
                continue
            vector = np.asarray(embedding, dtype=np.float32)
            vector_norm = float(np.linalg.norm(vector))
            if vector_norm <= 0:
                continue
            similarity = float(np.dot(anchor_embedding, vector) / (anchor_norm * vector_norm))
            ranked.append((path, similarity))
        return sorted(ranked, key=lambda item: item[1], reverse=True)

    def view_image(self, path: str) -> None:
        if self.current_state is None:
            return
        viewer, outer = self._create_modal(Path(path).name, padding=0, geometry="1000x800")
        try:
            image = open_oriented(path)
        except Exception as exc:  # noqa: BLE001
            messagebox.showerror("View failed", str(exc), parent=viewer)
            viewer.destroy()
            return
        info = self._file_info_text(path)
        top = ttk.Frame(outer, padding=10)
        top.pack(side=TOP, fill=BOTH)
        ttk.Label(top, text=Path(path).name).pack(side=LEFT)
        ttk.Label(top, text=f"{self._match_score_for_path(path):.3f}").pack(side=LEFT, padx=10)
        ttk.Label(top, text=info).pack(side=LEFT, padx=10)
        canvas = Canvas(outer, highlightthickness=0, bg=self._palette()["bg"])
        canvas.pack(fill=BOTH, expand=True)
        overlay = canvas.create_text(
            10, 10, anchor="nw", text=self._axis_details_text(path), fill=self._palette()["fg"]
        )
        state = {"zoom": 1.0, "offset_x": 0, "offset_y": 0}
        photo_ref: list[ImageTk.PhotoImage] = []

        def redraw() -> None:
            width = max(canvas.winfo_width(), 1)
            height = max(canvas.winfo_height(), 1)
            zoom = max(0.1, min(8.0, state["zoom"]))
            size = (max(1, int(image.width * zoom)), max(1, int(image.height * zoom)))
            resized = image.resize(size, Image.Resampling.LANCZOS)
            photo = ImageTk.PhotoImage(resized)
            photo_ref[:] = [photo]
            canvas.delete("image")
            canvas.create_image(state["offset_x"], state["offset_y"], anchor="center", image=photo, tags="image")
            canvas.coords(overlay, 10, height - 30)
            canvas.configure(scrollregion=(0, 0, width, height))

        def zoom(event) -> str:
            delta = 1.1 if getattr(event, "delta", 0) > 0 else 0.9
            state["zoom"] *= delta
            redraw()
            return "break"

        def start_pan(event) -> None:
            canvas.scan_mark(event.x, event.y)

        def move_pan(event) -> None:
            canvas.scan_dragto(event.x, event.y, gain=1)
            state["offset_x"] = canvas.canvasx(event.x)
            state["offset_y"] = canvas.canvasy(event.y)
            redraw()

        def close_viewer() -> None:
            image.close()
            viewer.destroy()

        canvas.bind("<MouseWheel>", zoom)
        canvas.bind("<Button-4>", lambda event: zoom(type("E", (), {"delta": 120})()))
        canvas.bind("<Button-5>", lambda event: zoom(type("E", (), {"delta": -120})()))
        canvas.bind("<ButtonPress-1>", start_pan)
        canvas.bind("<B1-Motion>", move_pan)
        viewer.protocol("WM_DELETE_WINDOW", close_viewer)
        viewer.bind("<Escape>", lambda _event: close_viewer())
        viewer.bind("<Configure>", lambda _event: redraw())
        redraw()

    def _file_info_text(self, path: str) -> str:
        try:
            stat = Path(path).stat()
            # Displayed dimensions, so a portrait phone photo does not report itself as
            # landscape. Reads the header only; the raster is never decoded.
            width, height = oriented_size(path)
            size_text = f"{width}x{height}"
        except Exception:  # noqa: BLE001
            return path
        modified = time.strftime("%Y-%m-%d %H:%M", time.localtime(stat.st_mtime))
        size_mb = stat.st_size / (1024 * 1024)
        return f"{size_text} | {size_mb:.1f} MB | {modified} | {path}"

    def reveal_in_file_manager(self, path: str) -> None:
        command = self._reveal_command(Path(path))
        try:
            subprocess.Popen(command)
        except Exception as exc:  # noqa: BLE001
            messagebox.showerror("Reveal failed", str(exc))

    @staticmethod
    def _reveal_command(path: Path) -> list[str]:
        resolved = path.resolve()
        if platform.system() == "Windows":
            return ["explorer", "/select,", str(resolved)]
        if platform.system() == "Darwin":
            return ["open", "-R", str(resolved)]
        return ["xdg-open", str(resolved.parent)]

    def open_prompt_tester_dialog(self) -> None:
        if self.current_state is None:
            messagebox.showinfo("No results", "Run a scan first.")
            return
        if not self._ensure_scorer():
            return
        dialog, outer = self._create_modal("Prompt tester")
        positive_text = Text(outer, width=58, height=5, wrap="word")
        positive_text.grid(row=0, column=0, columnspan=2, sticky="ew", pady=(0, 8))
        positive_text.insert("1.0", "\n".join(self.config.positive_prompts[:3]))
        negative_text = Text(outer, width=58, height=5, wrap="word")
        negative_text.grid(row=1, column=0, columnspan=2, sticky="ew", pady=(0, 8))
        negative_text.insert("1.0", "\n".join(self.config.negative_prompts[:3]))
        top_n_var = StringVar(value="8")
        ttk.Label(outer, text="Top N").grid(row=2, column=0, sticky="w")
        top_n_entry = ttk.Entry(outer, textvariable=top_n_var, width=10)
        top_n_entry.grid(row=2, column=1, sticky="w", pady=(0, 8))

        results = ttk.Frame(outer)
        results.grid(row=4, column=0, columnspan=2, sticky="nsew")
        outer.columnconfigure(0, weight=1)
        outer.columnconfigure(1, weight=1)
        outer.rowconfigure(4, weight=1)
        results.columnconfigure(0, weight=1)

        canvas, scroll_frame, scroll_window = self._modal_scroll_frame(results)
        canvas.bind("<Configure>", lambda event: canvas.itemconfigure(scroll_window, width=event.width))

        status_label = ttk.Label(outer, text="")
        status_label.grid(row=3, column=0, columnspan=2, sticky="w", pady=(0, 8))
        preview_refs: list[ImageTk.PhotoImage] = []

        def run_test() -> None:
            raw_positive = [line.strip() for line in positive_text.get("1.0", "end").splitlines() if line.strip()]
            raw_negative = [line.strip() for line in negative_text.get("1.0", "end").splitlines() if line.strip()]
            if not raw_positive:
                messagebox.showerror("Invalid prompt", "Enter at least one positive prompt.", parent=dialog)
                return
            try:
                top_n = max(1, int(top_n_var.get().strip()))
            except Exception:  # noqa: BLE001
                messagebox.showerror("Invalid prompt", "Top N must be a positive integer.", parent=dialog)
                return
            if self.scorer is None or self.current_state is None:
                return
            scores = self.scorer.score_prompt_similarity(
                self.current_state.embeddings,
                raw_positive,
                raw_negative,
                scale=self.config.zero_shot_scale,
            )
            visible_mask = self._result_visibility_mask()
            ranked = [
                (path, float(score))
                for path, score, include in zip(self.current_state.paths, scores, visible_mask, strict=False)
                if include
            ]
            ranked.sort(key=lambda item: item[1], reverse=True)
            ranked = ranked[:top_n]
            for child in scroll_frame.winfo_children():
                child.destroy()
            preview_refs.clear()
            if not ranked:
                ttk.Label(scroll_frame, text="No results matched the current filters.").grid(
                    row=0, column=0, padx=10, pady=10
                )
                status_label.configure(text="No results.")
                return
            for row, (path, score) in enumerate(ranked):
                preview_refs.append(
                    self._render_card(scroll_frame, path, score, row, register=False, show_actions=False)
                )
            status_label.configure(text=f"Computed prompt scores for {len(ranked)} images.")

        button_row = ttk.Frame(outer)
        button_row.grid(row=5, column=0, columnspan=2, sticky="e", pady=(8, 0))
        ttk.Button(
            button_row, text="Close", command=dialog._safe_close  # type: ignore[attr-defined]
        ).pack(side=RIGHT, padx=(8, 0))
        ttk.Button(button_row, text="Test", command=run_test).pack(side=RIGHT)
        run_test()

    def reset_global_learning(self) -> None:
        """Forget every Accept/REJECT pooled across folders. Per-folder labels stay."""
        store = GlobalLearningStore(model_name=self.config.model_name)
        stats = store.stats()
        if not stats.get("total"):
            messagebox.showinfo("Cross-folder learning", "There is nothing pooled across folders yet.")
            return
        if not messagebox.askyesno(
            "Reset cross-folder learning",
            f"Forget {stats['total']} pooled decisions "
            f"({stats['accepted']} accepted, {stats['rejected']} rejected)?\n\n"
            "The labels saved inside each scanned folder are not touched, so re-scanning "
            "a folder teaches the scanner again from those.",
        ):
            return
        store.clear()
        if self.scorer is not None:
            self.scorer._global_signature = ""
        self.status_var.set("Cross-folder learning reset.")

    def _face_model_status(self) -> str:
        if vision_analysis.face_detection_available():
            return "Face detection: on (regions anchored to detected faces)"
        if not hasattr(vision_analysis, "cv2") or vision_analysis.cv2 is None:
            return "Face detection: unavailable (OpenCV missing)"
        return "Face detection: off — body bands are used instead"

    def install_face_model(self, parent, status_label: ttk.Label | None = None) -> None:
        """Fetch the YuNet face model, on an explicit click and after confirmation.

        Anchoring crops to real faces makes the age and sex stages markedly more
        reliable than judging a whole frame, but the model is not bundled with
        OpenCV, so it is a deliberate opt-in download.
        """
        if vision_analysis.cv2 is None or not hasattr(vision_analysis.cv2, "FaceDetectorYN"):
            messagebox.showerror(
                "Face detection unavailable",
                "This build of OpenCV has no YuNet detector, so face-anchored regions cannot be enabled.",
                parent=parent,
            )
            return
        size_kb = vision_analysis.MODEL_APPROX_BYTES // 1024
        if not messagebox.askyesno(
            "Download face model",
            f"Download the YuNet face detection model ({size_kb} KB) from:\n\n"
            f"{vision_analysis.MODEL_URL}\n\n"
            f"It is saved to:\n{vision_analysis.model_path()}\n\n"
            "The download is checked against a known SHA-256 before it is installed. Continue?",
            parent=parent,
        ):
            return

        def worker() -> None:
            import urllib.request

            try:
                request = urllib.request.Request(vision_analysis.MODEL_URL, headers={"User-Agent": "bikini-scanner"})
                with urllib.request.urlopen(request, timeout=60) as response:
                    payload = response.read()
                target = vision_analysis.install_model_from_bytes(payload)
            except Exception as exc:
                LOGGER.exception("Face model download failed")
                # Bound as a default: Python unbinds `exc` when the except block ends,
                # so a lambda that closed over it raised NameError instead of showing
                # the error, and a failed download reported nothing at all.
                message = str(exc)
                self._after(0, lambda text=message: messagebox.showerror("Download failed", text, parent=parent))
                return
            LOGGER.info("Installed face model at %s", target)

            def finish() -> None:
                if status_label is not None:
                    try:
                        status_label.configure(text=self._face_model_status())
                    except Exception:  # noqa: BLE001
                        pass
                self.status_var.set("Face model installed. Run a new scan to use face-anchored regions.")
                messagebox.showinfo(
                    "Face model installed",
                    "Face detection is now enabled. Run a new scan so regions can be anchored to faces.",
                    parent=parent,
                )

            self._after(0, finish)

        self.status_var.set("Downloading face model...")
        threading.Thread(target=worker, name="face-model-download", daemon=True).start()

    def open_settings_dialog(self) -> None:
        # This form runs to 40-odd grid rows. In a fixed-size window the lower half,
        # including the Save and Cancel buttons, was simply unreachable on a short
        # screen, so it lives in the same scrollable frame the other tall modals use.
        dialog, outer = self._create_modal("Settings", resizable=(False, True))
        canvas, form, scroll_window = self._modal_scroll_frame(outer)
        canvas.bind("<Configure>", lambda event: canvas.itemconfigure(scroll_window, width=event.width))

        def scroll_settings(event: Any) -> None:
            # Text widgets scroll themselves. Without this the wheel over the prompt
            # boxes would scroll the dialog out from under the pointer instead.
            if isinstance(event.widget, Text):
                return
            canvas.yview_scroll(-int(event.delta / 120), "units")

        canvas.bind("<MouseWheel>", scroll_settings)
        canvas.bind("<Button-4>", lambda _event: canvas.yview_scroll(-1, "units"))
        canvas.bind("<Button-5>", lambda _event: canvas.yview_scroll(1, "units"))
        palette = self._palette()

        backend_var = StringVar(value=self.config.backend)
        model_var = StringVar(value=self.config.model_name)
        device_var = StringVar(value=self.config.device)
        precision_var = StringVar(value=self.config.precision)
        quantize_cpu_var = BooleanVar(value=self.config.quantize_cpu)
        preload_backend_var = BooleanVar(value=self.config.preload_backend)
        nsfw_mode_var = StringVar(value=self.config.nsfw_filter)
        require_person_var = BooleanVar(value=self.config.require_person)
        enable_face_detection_var = BooleanVar(value=self.config.enable_face_detection)
        batch_var = StringVar(value=str(self.config.batch_size))
        nsfw_threshold_var = StringVar(value=str(self.config.nsfw_threshold))
        person_threshold_var = StringVar(value=str(self.config.person_threshold))
        scale_var = StringVar(value=str(self.config.zero_shot_scale))
        classifier_weight_var = StringVar(value=str(self.config.classifier_weight))
        zero_shot_weight_var = StringVar(value=str(self.config.zero_shot_weight))
        threshold_var = StringVar(value=str(self.config.threshold))
        thumbnail_cache_var = StringVar(value=str(self._thumbnail_cache_limit()))
        deep_scan_var = StringVar(value=self.config.deep_scan)
        exclude_minors_var = BooleanVar(value=self.config.exclude_minors)
        minor_threshold_var = StringVar(value=str(self.config.minor_threshold))
        require_female_var = BooleanVar(value=self.config.require_female)
        female_threshold_var = StringVar(value=str(self.config.female_threshold))
        global_learning_var = BooleanVar(value=self.config.global_learning)
        refine_var = BooleanVar(value=bool(self.config.refine_model))
        vlm_enabled_var = BooleanVar(value=self.config.vlm_enabled)
        vlm_base_url_var = StringVar(value=self.config.vlm_base_url)
        vlm_model_var = StringVar(value=self.config.vlm_model)
        vlm_concurrency_var = StringVar(value=str(self.config.vlm_concurrency))
        vlm_band_var = StringVar(value=str(self.config.vlm_band))
        vlm_max_images_var = StringVar(value=str(self.config.vlm_max_images))

        def add_labeled_entry(row: int, label: str, variable: StringVar, width: int = 48, tip: str = "") -> ttk.Entry:
            caption = ttk.Label(form, text=label)
            caption.grid(row=row, column=0, sticky="w", pady=(0, 4))
            entry = ttk.Entry(form, textvariable=variable, width=width)
            entry.grid(row=row, column=1, sticky="ew", pady=(0, 8))
            if tip:
                # Hover help on the caption as well: that is where the eye lands first.
                self._tooltip(caption, tip)
                self._tooltip(entry, tip)
            return entry

        def add_check(parent, row: int, column: int, text: str, variable: BooleanVar, tip: str) -> ttk.Checkbutton:
            box = ttk.Checkbutton(parent, text=text, variable=variable)
            box.grid(row=row, column=column, sticky="w", pady=(0, 6))
            self._tooltip(box, tip)
            return box

        def add_inline_entry(row: int, label: str, variable: StringVar, width: int = 14, tip: str = "") -> ttk.Entry:
            """A caption+entry pair for a row whose column 0 is already a checkbox.

            add_labeled_entry always puts its caption in column 0, so on the two rows
            that pair a checkbox with a number the caption was gridded into the same
            cell as the checkbox and the two were drawn on top of each other — the box
            was invisible and unclickable. Everything here lives in column 1.
            """
            holder = ttk.Frame(form)
            holder.grid(row=row, column=1, sticky="ew", padx=(14, 0), pady=(0, 8))
            caption = ttk.Label(holder, text=label)
            caption.pack(side=LEFT, padx=(0, 8))
            entry = ttk.Entry(holder, textvariable=variable, width=width)
            entry.pack(side=LEFT)
            if tip:
                self._tooltip(caption, tip)
                self._tooltip(entry, tip)
            return entry

        def add_combo(row: int, label: str, variable: StringVar, values: tuple[str, ...], tip: str) -> ttk.Combobox:
            caption = ttk.Label(form, text=label)
            caption.grid(row=row, column=0, sticky="w", pady=(0, 4))
            combo = ttk.Combobox(form, textvariable=variable, values=values, state="readonly")
            combo.grid(row=row, column=1, sticky="ew", pady=(0, 8))
            self._tooltip(caption, tip)
            self._tooltip(combo, tip)
            return combo

        def add_section(row: int, title: str) -> None:
            """A separator plus a bold header to break up the settings form."""
            ttk.Separator(form, orient="horizontal").grid(row=row, column=0, columnspan=2, sticky="ew", pady=(8, 4))
            ttk.Label(form, text=title, font=("TkDefaultFont", 10, "bold")).grid(
                row=row + 1, column=0, sticky="w", pady=(0, 4)
            )

        add_section(0, "Scoring prompts")
        prompt_caption = ttk.Label(form, text="Positive prompts")
        prompt_caption.grid(row=2, column=0, sticky="w")
        ttk.Label(form, text="Primary scoring uses the canonical Bikini axis defaults.", wraplength=360).grid(
            row=2, column=1, sticky="e"
        )
        positive_text = Text(form, width=58, height=6, wrap="word")
        positive_text.grid(row=3, column=0, columnspan=2, sticky="ew", pady=(0, 8))
        positive_text.insert("1.0", "\n".join(self.config.positive_prompts))
        positive_tip = (
            "One phrase per line describing what you WANT found. Each is compared against "
            "the image and the best match wins, so several differently worded lines beat "
            "one clever one. These feed the headline bikini axis; the cleavage, midriff, "
            "age and sex axes have their own built-in wording."
        )
        self._tooltip(prompt_caption, positive_tip)
        self._tooltip(positive_text, positive_tip)

        negative_caption = ttk.Label(form, text="Negative prompts")
        negative_caption.grid(row=4, column=0, sticky="w")
        negative_text = Text(form, width=58, height=6, wrap="word")
        negative_text.grid(row=5, column=0, columnspan=2, sticky="ew", pady=(0, 8))
        negative_text.insert("1.0", "\n".join(self.config.negative_prompts))
        negative_tip = (
            "One phrase per line for what you do NOT want. A score is positive evidence "
            "minus negative evidence, so these matter as much as the positives: list the "
            "things the scanner keeps confusing for a match, like 'a fully clothed person'."
        )
        self._tooltip(negative_caption, negative_tip)
        self._tooltip(negative_text, negative_tip)

        add_section(6, "Model & Hardware")
        backend_combo = add_combo(
            8,
            "Backend",
            backend_var,
            ("clip-torch", "clip-onnx"),
            "Which engine runs the model. clip-torch is the normal choice. clip-onnx is "
            "experimental and needs the optional onnxruntime package installed. Changing "
            "this requires a new scan.",
        )
        add_labeled_entry(
            9,
            "Model name",
            model_var,
            tip="The Hugging Face model used for scoring. openai/clip-vit-base-patch32 is the "
            "default and is downloaded once, then cached. openai/clip-vit-large-patch14 is "
            "noticeably more accurate but a ~1.7 GB download and several times slower per "
            "image. Changing this invalidates cached embeddings and needs a fresh scan.",
        )
        add_combo(
            10,
            "Device",
            device_var,
            ("auto", "cpu", "cuda"),
            "Where the model runs. auto picks your NVIDIA GPU when one is usable and falls "
            "back to the CPU otherwise. Force cpu if a GPU driver misbehaves.",
        )
        add_combo(
            11,
            "Precision",
            precision_var,
            ("auto", "fp32", "fp16"),
            "Numeric precision. fp16 roughly halves GPU memory and speeds scans up, but only "
            "applies on CUDA; on the CPU everything runs fp32 regardless. auto chooses for you.",
        )
        add_check(
            form,
            12,
            0,
            "Quantize CPU (int8)",
            quantize_cpu_var,
            "Compresses the model to 8-bit for CPU scanning. Faster and lighter on memory, at "
            "some cost in accuracy. Worth trying on a slow machine with a large folder.",
        )
        add_check(
            form,
            12,
            1,
            "Preload backend on startup",
            preload_backend_var,
            "Loads the model in the background as soon as the app opens, so your first scan "
            "starts immediately. Turn off if you want the app to open using less memory and "
            "do not mind waiting at the first scan instead.",
        )
        add_labeled_entry(
            13,
            "Batch size",
            batch_var,
            width=18,
            tip="How many images are fed to the model at once. Larger is faster but uses more "
            "memory. 16 suits most machines; drop to 4-8 if scanning runs out of memory.",
        )
        add_labeled_entry(
            14,
            "Zero-shot scale",
            scale_var,
            width=18,
            tip="How sharply prompt scores separate. The model's raw agreement differences are "
            "tiny, so this multiplies them: too low and everything looks like a 50/50 guess, "
            "too high and every score slams to 0 or 1. 40 was chosen by measurement — leave it "
            "unless you are deliberately recalibrating.",
        )
        add_labeled_entry(
            15,
            "Classifier weight",
            classifier_weight_var,
            width=18,
            tip="Legacy blend control, used only when the pipeline is set to 'legacy'. The "
            "cascade pipeline ignores it and instead decides how much to trust what it has "
            "learned from your Accept/REJECT decisions, based on its own measured accuracy.",
        )
        add_labeled_entry(
            16,
            "Zero-shot weight",
            zero_shot_weight_var,
            width=18,
            tip="The other half of the legacy blend: how much the prompt-only score counts. "
            "Also unused by the cascade pipeline.",
        )
        add_section(17, "Detection")
        add_labeled_entry(
            19,
            "Threshold",
            threshold_var,
            width=18,
            tip="The sensitivity a scan starts at, matching the slider on the main window. "
            "Lower shows more photos and more false alarms; higher shows only the surest "
            "matches. 0.35 suits the current scoring.",
        )
        nsfw_combo = add_combo(
            20,
            "NSFW mode",
            nsfw_mode_var,
            ("include", "exclude", "only"),
            "What to do with explicit images. include keeps them alongside everything else, "
            "exclude drops them from results, only shows nothing else.",
        )
        add_labeled_entry(
            21,
            "NSFW threshold",
            nsfw_threshold_var,
            width=18,
            tip="How sure the scanner must be before treating an image as explicit, for the "
            "NSFW mode above. Lower catches more but misjudges more.",
        )
        add_labeled_entry(
            22,
            "Person threshold",
            person_threshold_var,
            width=18,
            tip="How sure the scanner must be that a person is present, used only when "
            "'Require person' is ticked below.",
        )
        add_check(
            form,
            23,
            0,
            "Require person",
            require_person_var,
            "Discard images where no person is detected. Off by default for good reason: "
            "close-up body shots often score very low on 'is this a person', so this filter "
            "throws away real matches. Turn it on only if landscapes are cluttering results.",
        )
        add_check(
            form,
            23,
            1,
            "Enable face detection",
            enable_face_detection_var,
            "Count faces in every image during the scan. Needs the face model installed below. "
            "The deep pass already detects faces for candidate images, so this mainly adds a "
            "face count to the details line.",
        )
        add_section(24, "Detection pipeline")
        add_combo(
            26,
            "Deep scan (body-region crops)",
            deep_scan_var,
            ("candidates", "always", "off"),
            "The model only ever sees a small square, so a bikini top in a full photo is a "
            "handful of pixels. Deep scan re-checks face and body crops separately, which is "
            "what makes cleavage and midriff detectable at all.\n\n"
            "candidates: crop only images that might contain a person (recommended).\n"
            "always: crop everything — best on distant or background subjects, much slower.\n"
            "off: whole frame only — fastest, and noticeably worse.",
        )
        add_check(
            form,
            27,
            0,
            "Exclude images that may show a minor",
            exclude_minors_var,
            "Drops anything that reads as showing a child. Flagged images are forced to a zero "
            "score and hidden from every view, so no threshold or filter can bring them back. "
            "Deliberately errs toward excluding.",
        )
        add_inline_entry(
            27,
            "Minor sensitivity (lower = stricter)",
            minor_threshold_var,
            tip="How much child-like evidence triggers the age exclusion to the left. LOWER IS "
            "STRICTER: 0.30 excludes on modest evidence, 0.60 waits for strong evidence and "
            "therefore excludes less. Age estimates are rough, which is why the default sits "
            "low. To switch the gate off entirely, untick the box rather than raising this.",
        )
        add_check(
            form,
            28,
            0,
            "Prefer female subjects",
            require_female_var,
            "Ranks images with a female subject higher. On its own this only re-orders results; "
            "it discards nothing unless you also raise the cut-off to the right.",
        )
        add_inline_entry(
            28,
            "Female cut-off (0 = rank only)",
            female_threshold_var,
            tip="Leave at 0 to only re-order results. Above 0 it becomes a hard filter that "
            "discards images scoring below it. Kept at 0 by default because on real photos a "
            "hard cut-off silently binned a genuine match whose close-up crop gave the model "
            "nothing to judge sex from.",
        )
        add_check(
            form,
            29,
            0,
            "Learn across all folders",
            global_learning_var,
            "Pool your Accept/REJECT decisions so every folder benefits from all of them. Off "
            "means each folder learns from scratch. Each folder's own labels are saved either "
            "way; Tools > Reset cross-folder learning clears the shared pool.",
        )
        add_check(
            form,
            29,
            1,
            f"High-accuracy re-check of borderline images ({HIGH_ACCURACY_MODEL.split('/')[-1]}, ~1.7 GB download)",
            refine_var,
            "Re-scores only the images sitting closest to your sensitivity setting using a much "
            "larger model, then blends its opinion in. Those borderline cases are where "
            "mistakes live, so the accuracy is worth it there. Costs a one-time ~1.7 GB "
            "download and adds minutes to a scan; the rest of the images are untouched.",
        )
        add_section(30, "VLM adjudication")
        add_check(
            form,
            32,
            0,
            "Use local vision-LLM adjudication",
            vlm_enabled_var,
            "Optional second opinion from a local Ollama or llama.cpp server. It only checks "
            "borderline images and uncertain age calls, in parallel, so the usual CLIP scan "
            "stays fast. The server must already be running.",
        )
        add_labeled_entry(
            33,
            "VLM server URL",
            vlm_base_url_var,
            width=36,
            tip="OpenAI-compatible local endpoint, for example http://localhost:11434/v1. "
            "The stage is skipped if it cannot reach this address.",
        )
        add_labeled_entry(
            34,
            "VLM model",
            vlm_model_var,
            width=36,
            tip="Model name served by Ollama or llama.cpp, for example qwen2.5vl:7b.",
        )
        add_labeled_entry(
            35,
            "VLM concurrency",
            vlm_concurrency_var,
            width=18,
            tip="How many local requests run at once. Higher is faster only when your server "
            "has enough CPU/GPU memory; 4 is a sensible starting point.",
        )
        add_labeled_entry(
            36,
            "VLM borderline band",
            vlm_band_var,
            width=18,
            tip="Only scores within this distance of the threshold are sent to the VLM, plus "
            "uncertain age calls. Wider is more accurate but costs more requests.",
        )
        add_labeled_entry(
            37,
            "VLM maximum images",
            vlm_max_images_var,
            width=18,
            tip="Hard cap on VLM requests per scan. Skin exposure is only used to prioritize "
            "images inside the eligible band; it never excludes an eligible image by itself.",
        )
        face_row = ttk.Frame(form)
        face_row.grid(row=38, column=0, columnspan=2, sticky="ew", pady=(0, 8))
        face_status = ttk.Label(face_row, text=self._face_model_status())
        face_status.pack(side=LEFT)
        face_button = ttk.Button(
            face_row,
            text="Install face model",
            command=lambda: self.install_face_model(dialog, face_status),
        )
        face_button.pack(side=RIGHT)
        self._tooltip(
            face_button,
            "Downloads a 230 KB face detector (shown for confirmation first). With it, crops "
            "are anchored to real faces and the age check reads actual face crops, which is "
            "markedly more reliable than judging a whole photo. Without it the scanner falls "
            "back to fixed body bands and still works.",
        )
        self._tooltip(face_status, "Whether face-anchored crops are available right now.")

        add_section(39, "Advanced")
        add_labeled_entry(
            41,
            "Thumbnail cache entries",
            thumbnail_cache_var,
            width=18,
            tip="How many scaled thumbnails to keep in memory (32-2048). Higher makes scrolling "
            "and resizing smoother at the cost of RAM. Lower it if the app feels heavy.",
        )
        url_caption = ttk.Label(form, text="Update manifest URL (optional)")
        url_caption.grid(row=42, column=0, sticky="w", pady=(0, 4))
        update_url_entry = ttk.Entry(form, textvariable=self.update_url_var, width=48)
        update_url_entry.grid(row=42, column=1, sticky="ew", pady=(0, 8))
        url_tip = (
            "Optional address of a JSON file listing the newest version, used by Help > Check "
            "for updates. Leave empty and the app never contacts anything for updates."
        )
        self._tooltip(url_caption, url_tip)
        self._tooltip(update_url_entry, url_tip)

        button_row = ttk.Frame(form)
        button_row.grid(row=43, column=0, columnspan=2, sticky="e", pady=(10, 0))

        def close_dialog() -> None:
            dialog.grab_release()
            dialog.destroy()

        dialog.protocol("WM_DELETE_WINDOW", close_dialog)

        def save_settings() -> None:
            raw_positive = [line.strip() for line in positive_text.get("1.0", "end").splitlines()]
            raw_negative = [line.strip() for line in negative_text.get("1.0", "end").splitlines()]
            positive_prompts = [line for line in raw_positive if line]
            negative_prompts = [line for line in raw_negative if line]
            if not positive_prompts:
                messagebox.showerror("Invalid settings", "Positive prompts cannot be empty.", parent=dialog)
                return
            if not negative_prompts:
                messagebox.showerror("Invalid settings", "Negative prompts cannot be empty.", parent=dialog)
                return
            backend = backend_var.get().strip() or self.config.backend
            if backend not in {"clip-torch", "clip-onnx"}:
                messagebox.showerror("Invalid settings", "Backend must be clip-torch or clip-onnx.", parent=dialog)
                return
            model_name = model_var.get().strip()
            if not model_name:
                messagebox.showerror("Invalid settings", "Model name cannot be empty.", parent=dialog)
                return
            try:
                batch_size = parse_int_entry("batch_size", batch_var.get())
            except FieldError as exc:
                messagebox.showerror("Invalid settings", str(exc), parent=dialog)
                return
            try:
                thumbnail_cache_size = int(thumbnail_cache_var.get().strip())
            except Exception:  # noqa: BLE001
                messagebox.showerror(
                    "Invalid settings", "Thumbnail cache entries must be an integer from 32 to 2048.", parent=dialog
                )
                return
            if not 32 <= thumbnail_cache_size <= 2048:
                messagebox.showerror(
                    "Invalid settings", "Thumbnail cache entries must be from 32 to 2048.", parent=dialog
                )
                return
            try:
                zero_shot_scale = parse_float_entry("zero_shot_scale", scale_var.get())
                classifier_weight = parse_float_entry("classifier_weight", classifier_weight_var.get())
                zero_shot_weight = parse_float_entry("zero_shot_weight", zero_shot_weight_var.get())
                threshold = parse_float_entry("threshold", threshold_var.get())
                nsfw_threshold = parse_float_entry("nsfw_threshold", nsfw_threshold_var.get())
                person_threshold = parse_float_entry("person_threshold", person_threshold_var.get())
                vlm_concurrency = parse_int_entry("vlm_concurrency", vlm_concurrency_var.get())
                vlm_band = parse_float_entry("vlm_band", vlm_band_var.get())
                vlm_max_images = parse_int_entry("vlm_max_images", vlm_max_images_var.get())
            except FieldError as exc:
                messagebox.showerror("Invalid settings", str(exc), parent=dialog)
                return
            vlm_base_url = vlm_base_url_var.get().strip()
            vlm_model = vlm_model_var.get().strip()
            if bool(vlm_enabled_var.get()) and (not vlm_base_url or not vlm_model):
                messagebox.showerror(
                    "Invalid settings", "VLM server URL and model are required when enabled.", parent=dialog
                )
                return
            # Enabling the VLM pass uploads every adjudicated image to that endpoint.
            # A local server is the documented setup and stays silent; anything else is
            # confirmed once here, because the only other signal is a line in the log.
            if (
                bool(vlm_enabled_var.get())
                and vlm_base_url != self.config.vlm_base_url
                and not is_local_endpoint(vlm_base_url)
                and not messagebox.askyesno(
                    "Send images to a remote server?",
                    f"{vlm_base_url}\n\nThis is not a local address. Every image the VLM pass "
                    "adjudicates will be uploaded to it.\n\nUse this endpoint?",
                    parent=dialog,
                    default="no",
                    icon="warning",
                )
            ):
                return
            try:
                minor_threshold = parse_float_entry("minor_threshold", minor_threshold_var.get())
                female_threshold = parse_float_entry("female_threshold", female_threshold_var.get())
            except FieldError as exc:
                messagebox.showerror("Invalid settings", str(exc), parent=dialog)
                return
            try:
                deep_scan = parse_choice_entry("deep_scan", deep_scan_var.get())
                nsfw_mode = parse_choice_entry("nsfw_filter", nsfw_mode_var.get())
                device = parse_choice_entry("device", device_var.get())
                precision = parse_choice_entry("precision", precision_var.get())
            except FieldError as exc:
                messagebox.showerror("Invalid settings", str(exc), parent=dialog)
                return

            backend_changed = backend != self.config.backend or model_name != self.config.model_name
            backend_changed = backend_changed or device != self.config.device or precision != self.config.precision
            backend_changed = backend_changed or bool(quantize_cpu_var.get()) != self.config.quantize_cpu
            scorer_changed = (
                positive_prompts != self.config.positive_prompts
                or negative_prompts != self.config.negative_prompts
                or zero_shot_scale != self.config.zero_shot_scale
                or classifier_weight != self.config.classifier_weight
                or zero_shot_weight != self.config.zero_shot_weight
            )
            filter_changed = (
                threshold != self.config.threshold
                or nsfw_mode != self.config.nsfw_filter
                or nsfw_threshold != self.config.nsfw_threshold
                or bool(require_person_var.get()) != self.config.require_person
                or person_threshold != self.config.person_threshold
                or bool(enable_face_detection_var.get()) != self.config.enable_face_detection
                or bool(exclude_minors_var.get()) != self.config.exclude_minors
                or minor_threshold != self.config.minor_threshold
                or bool(require_female_var.get()) != self.config.require_female
                or female_threshold != self.config.female_threshold
                or bool(global_learning_var.get()) != self.config.global_learning
            )
            # A deep-scan or refine change needs a fresh scan, not just a re-filter:
            # both change what gets embedded.
            rescan_needed = (
                deep_scan != self.config.deep_scan
                or (HIGH_ACCURACY_MODEL if bool(refine_var.get()) else "") != self.config.refine_model
            )
            vlm_changed = (
                bool(vlm_enabled_var.get()) != self.config.vlm_enabled
                or vlm_base_url != self.config.vlm_base_url
                or vlm_model != self.config.vlm_model
                or vlm_concurrency != self.config.vlm_concurrency
                or vlm_band != self.config.vlm_band
                or vlm_max_images != self.config.vlm_max_images
            )

            self.config.backend = backend
            self.config.model_name = model_name
            self.config.device = device
            self.config.precision = precision
            self.config.quantize_cpu = bool(quantize_cpu_var.get())
            self.config.preload_backend = bool(preload_backend_var.get())
            self.config.positive_prompts = positive_prompts
            self.config.negative_prompts = negative_prompts
            self.config.batch_size = batch_size
            self.config.threshold = threshold
            self.config.zero_shot_scale = zero_shot_scale
            self.config.classifier_weight = classifier_weight
            self.config.zero_shot_weight = zero_shot_weight
            self.config.nsfw_filter = nsfw_mode
            self.config.nsfw_threshold = nsfw_threshold
            self.config.require_person = bool(require_person_var.get())
            self.config.person_threshold = person_threshold
            self.config.enable_face_detection = bool(enable_face_detection_var.get())
            self.config.deep_scan = deep_scan
            self.config.exclude_minors = bool(exclude_minors_var.get())
            self.config.minor_threshold = minor_threshold
            self.config.require_female = bool(require_female_var.get())
            self.config.female_threshold = female_threshold
            self.config.global_learning = bool(global_learning_var.get())
            self.config.refine_model = HIGH_ACCURACY_MODEL if bool(refine_var.get()) else ""
            self.config.vlm_enabled = bool(vlm_enabled_var.get())
            self.config.vlm_base_url = vlm_base_url
            self.config.vlm_model = vlm_model
            self.config.vlm_concurrency = vlm_concurrency
            self.config.vlm_band = vlm_band
            self.config.vlm_max_images = vlm_max_images
            self.thumbnail_cache_size_var.set(thumbnail_cache_size)
            self._trim_thumbnail_cache()
            if not self.folder_override_active:
                self.global_config = ScannerConfig.from_mapping(self.config.to_dict())
            self.threshold_var.set(threshold)
            self.nsfw_only_var.set(self.config.nsfw_filter == "only")
            self._refresh_vlm_badge()
            self._refresh_summary()

            if backend_changed:
                self.backend = None
                self.scorer = None
                self.current_state = None
                self.current_samples = []
                self.review_samples = []
                self._clear_grid()
                self.status_var.set("Settings saved. Run a new scan to apply backend/model changes.")
                self._refresh_summary()
                if self.config.preload_backend:
                    self._backend_preload_started = False
                    self._maybe_preload_backend()
            elif rescan_needed or vlm_changed:
                # Deep scan and refine decide what gets embedded, so a rescore of the
                # cached region table cannot apply them.
                self.scorer = None
                self.status_var.set("Settings saved. Run a new scan to apply them.")
                self._refresh_summary()
            elif scorer_changed:
                self.scorer = None
                self.status_var.set("Settings saved. Click Retrain or run a new scan to apply.")
                self._refresh_summary()
            else:
                self.status_var.set("Settings saved.")
                if filter_changed and self.current_state is not None:
                    self._refresh_active_view()
                if self.config.preload_backend:
                    self._maybe_preload_backend()

            close_dialog()

        reset_override = ttk.Button(button_row, text="Reset folder override", command=self._reset_folder_override)
        reset_override.pack(side=LEFT)
        self._tooltip(
            reset_override,
            "Discard the settings saved just for the current folder, so it goes back to using your normal settings.",
        )
        save_override = ttk.Button(button_row, text="Save folder override", command=self._save_folder_override)
        save_override.pack(side=LEFT, padx=6)
        self._tooltip(
            save_override,
            "Pin these settings to the current folder only. Useful when one folder needs a "
            "different sensitivity or different prompts from everything else. Other folders "
            "keep using your normal settings.",
        )
        cancel_button = ttk.Button(button_row, text="Cancel", command=close_dialog)
        cancel_button.pack(side=RIGHT, padx=(8, 0))
        self._tooltip(cancel_button, "Close without keeping any of the changes above.")
        save_button = ttk.Button(button_row, text="Save", command=save_settings)
        save_button.pack(side=RIGHT)
        self._tooltip(
            save_button,
            "Apply these settings everywhere. Changes to the model, prompts or deep scan need "
            "a new scan to take effect; filters and thresholds re-rank what is already on screen.",
        )
        dialog.columnconfigure(0, weight=1)
        form.columnconfigure(1, weight=1)
        positive_text.configure(
            bg=palette["entry_bg"],
            fg=palette["fg"],
            insertbackground=palette["fg"],
            highlightbackground=palette["panel"],
            relief="solid",
        )
        for widget in (nsfw_combo, backend_combo):
            widget.configure(width=22)
        negative_text.configure(
            bg=palette["entry_bg"],
            fg=palette["fg"],
            insertbackground=palette["fg"],
            highlightbackground=palette["panel"],
            relief="solid",
        )
        # Size the viewport to the form, but never past the screen: that cap is what
        # makes the scrollbar do any work.
        dialog.update_idletasks()
        for child in form.winfo_children():
            child.bind("<MouseWheel>", scroll_settings, add="+")
            child.bind("<Button-4>", lambda _event: canvas.yview_scroll(-1, "units"), add="+")
            child.bind("<Button-5>", lambda _event: canvas.yview_scroll(1, "units"), add="+")
        canvas.configure(
            width=form.winfo_reqwidth(),
            height=min(form.winfo_reqheight(), max(360, int(dialog.winfo_screenheight() * 0.8))),
        )
        backend_combo.focus_set()

    def open_output_options_dialog(self) -> None:
        dialog, form = self._create_modal("Output options")

        organization_var = StringVar(value=self.output_organization_var.get())
        template_var = StringVar(value=self.output_template_var.get())
        duplicate_var = StringVar(value=self.output_duplicate_var.get())
        low_var = StringVar(value=str(self.output_score_low_var.get()))
        high_var = StringVar(value=str(self.output_score_high_var.get()))

        ttk.Label(form, text="Organization").grid(row=0, column=0, sticky="w", pady=(0, 4))
        org_combo = ttk.Combobox(
            form,
            textvariable=organization_var,
            values=("flat", "score_band", "label", "score_band_label"),
            state="readonly",
        )
        org_combo.grid(row=0, column=1, sticky="ew", pady=(0, 8))
        ttk.Label(form, text="Filename template").grid(row=1, column=0, sticky="w", pady=(0, 4))
        template_entry = ttk.Entry(form, textvariable=template_var, width=46)
        template_entry.grid(row=1, column=1, sticky="ew", pady=(0, 8))
        ttk.Label(form, text="Duplicate policy").grid(row=2, column=0, sticky="w", pady=(0, 4))
        dup_combo = ttk.Combobox(
            form, textvariable=duplicate_var, values=("skip", "rename", "overwrite"), state="readonly"
        )
        dup_combo.grid(row=2, column=1, sticky="ew", pady=(0, 8))
        ttk.Label(form, text="Score band low").grid(row=3, column=0, sticky="w", pady=(0, 4))
        low_entry = ttk.Entry(form, textvariable=low_var, width=18)
        low_entry.grid(row=3, column=1, sticky="w", pady=(0, 8))
        ttk.Label(form, text="Score band high").grid(row=4, column=0, sticky="w", pady=(0, 4))
        high_entry = ttk.Entry(form, textvariable=high_var, width=18)
        high_entry.grid(row=4, column=1, sticky="w", pady=(0, 8))
        ttk.Label(
            form,
            text="Tokens: {stem} {name} {ext} {score} {score_pct} {index} {date} {timestamp} {label}",
            wraplength=440,
        ).grid(row=5, column=0, columnspan=2, sticky="w", pady=(0, 8))

        button_row = ttk.Frame(form)
        button_row.grid(row=6, column=0, columnspan=2, sticky="e", pady=(10, 0))

        def close_dialog() -> None:
            dialog.grab_release()
            dialog.destroy()

        def save_output_settings() -> None:
            organization = organization_var.get().strip()
            if organization not in {"flat", "score_band", "label", "score_band_label"}:
                messagebox.showerror("Invalid settings", "Choose a valid organization scheme.", parent=dialog)
                return
            duplicate = duplicate_var.get().strip()
            if duplicate not in {"skip", "rename", "overwrite"}:
                messagebox.showerror("Invalid settings", "Choose a valid duplicate policy.", parent=dialog)
                return
            try:
                low_value = float(low_var.get().strip())
                high_value = float(high_var.get().strip())
            except Exception:  # noqa: BLE001
                messagebox.showerror("Invalid settings", "Score cutoffs must be numeric.", parent=dialog)
                return
            if not (0.0 <= low_value <= 1.0 and 0.0 <= high_value <= 1.0 and low_value <= high_value):
                messagebox.showerror(
                    "Invalid settings", "Score cutoffs must be between 0 and 1 and low <= high.", parent=dialog
                )
                return
            template = template_var.get().strip()
            if not template:
                messagebox.showerror("Invalid settings", "Template cannot be empty.", parent=dialog)
                return
            self.output_organization_var.set(organization)
            self.output_template_var.set(template)
            self.output_duplicate_var.set(duplicate)
            self.output_score_low_var.set(low_value)
            self.output_score_high_var.set(high_value)
            self._save_user_prefs()
            self.status_var.set("Output settings saved.")
            close_dialog()

        ttk.Button(button_row, text="Cancel", command=close_dialog).pack(side=RIGHT, padx=(8, 0))
        ttk.Button(button_row, text="Save", command=save_output_settings).pack(side=RIGHT)
        dialog.protocol("WM_DELETE_WINDOW", close_dialog)
        form.columnconfigure(1, weight=1)
        template_entry.focus_set()

    def export_settings(self) -> None:
        target = filedialog.asksaveasfilename(
            title="Export settings",
            defaultextension=".json",
            filetypes=(("JSON files", "*.json"), ("All files", "*.*")),
        )
        if not target:
            return
        try:
            Path(target).write_text(json.dumps(self.config.to_dict(), indent=2, sort_keys=True), encoding="utf-8")
            self.status_var.set(f"Exported settings to {target}")
        except OSError as exc:
            messagebox.showerror("Export settings failed", str(exc))

    def import_settings(self) -> None:
        source = filedialog.askopenfilename(
            title="Import settings",
            filetypes=(("JSON files", "*.json"), ("All files", "*.*")),
        )
        if not source:
            return
        try:
            payload = json.loads(Path(source).read_text(encoding="utf-8"))
            if not isinstance(payload, dict):
                raise ValueError("Settings file must contain a JSON object.")
            imported = ScannerConfig.from_mapping(payload)
        except Exception as exc:  # noqa: BLE001
            messagebox.showerror("Import settings failed", str(exc))
            return
        self.config = imported
        self.global_config = ScannerConfig.from_mapping(imported.to_dict())
        self.folder_override_active = False
        self.override_var.set("")
        self.backend = None
        self.scorer = None
        self._sync_config_controls()
        self.status_var.set("Settings imported. Run a scan to apply them.")

    def _sync_config_controls(self) -> None:
        """Point the on-screen controls at the config that is actually in force.

        The sensitivity slider is the value a scan is run with, so a config loaded from
        a file or a profile has to move it — otherwise an imported threshold of 0.7 is
        quietly overridden by whatever the slider was left on.
        """
        self.threshold_var.set(float(self.config.threshold))
        self.nsfw_only_var.set(self.config.nsfw_filter == "only")
        self._refresh_summary()

    def open_profiles_dialog(self) -> None:
        dialog, outer = self._create_modal("Settings profiles")
        profile_var = StringVar(value=profile_names()[0] if profile_names() else "")
        combo = ttk.Combobox(outer, textvariable=profile_var, values=profile_names(), state="readonly", width=30)
        combo.pack(side=TOP, anchor="w", pady=(0, 10))
        ttk.Label(
            outer,
            text="Built-in profiles cannot be deleted. Apply a profile, then save it as a custom profile if desired.",
        ).pack(side=TOP, anchor="w", pady=(0, 10))
        buttons = self._modal_button_row(outer, pady=(0, 0))

        def refresh() -> None:
            combo.configure(values=profile_names())

        def apply() -> None:
            selected = profile_config(profile_var.get())
            if selected is None:
                return
            self.config = selected
            self.global_config = ScannerConfig.from_mapping(selected.to_dict())
            self.backend = None
            self.scorer = None
            self._sync_config_controls()
            self.status_var.set(f"Applied profile {profile_var.get()}. Run a scan to apply it.")

        def save_as() -> None:
            name = filedialog.asksaveasfilename(
                title="Save profile as",
                initialfile="profile",
                parent=dialog,
                filetypes=(("Profile names", "*"),),
            )
            if not name:
                return
            try:
                save_profile(Path(name).name, self.config)
                refresh()
                profile_var.set(Path(name).name)
            except ValueError as exc:
                messagebox.showerror("Save profile failed", str(exc), parent=dialog)

        def delete() -> None:
            name = profile_var.get()
            if name in BUILTIN_PROFILES:
                messagebox.showinfo("Delete profile", "Built-in profiles cannot be deleted.", parent=dialog)
                return
            if delete_profile(name):
                refresh()
                profile_var.set("")

        ttk.Button(buttons, text="Apply", command=apply).pack(side=LEFT)
        ttk.Button(buttons, text="Save as...", command=save_as).pack(side=LEFT, padx=6)
        ttk.Button(buttons, text="Delete", command=delete).pack(side=LEFT)
        ttk.Button(
            buttons, text="Close", command=dialog._safe_close  # type: ignore[attr-defined]
        ).pack(side=RIGHT)

    def run_scan(self) -> None:
        if self._scan_active:
            # Two scans of one folder race each other over the same scorer and the same
            # embedding cache, and only the newest could be stopped.
            messagebox.showinfo(
                "Scan in progress",
                "A scan is already running. Press Stop to end it, or wait for it to finish.",
            )
            return
        folder = self.folder_var.get().strip()
        if not folder:
            messagebox.showerror("No folder", "Please choose a folder first.")
            return
        if not Path(folder).is_dir():
            messagebox.showerror("No folder", f"This folder does not exist:\n{folder}")
            return
        # Pre-scan check: count supported images so we don't run a full scan on an
        # empty folder or one with only unsupported files (e.g. all .txt or .pdf).
        image_suffixes = {suffix.lower() for suffix in SUPPORTED_IMAGE_SUFFIXES}
        try:
            image_count = sum(
                1 for entry in Path(folder).rglob("*") if entry.is_file() and entry.suffix.lower() in image_suffixes
            )
        except OSError:
            image_count = 0
        if image_count == 0:
            messagebox.showinfo(
                "No images found",
                f"No supported image files were found in this folder:\n{folder}\n\n"
                "Supported formats: JPG, PNG, BMP, GIF, WEBP, TIFF, HEIC.",
            )
            return
        # The folder box is editable and callers may set it directly, so the store
        # can lag behind the path shown. Bind it here rather than assert later.
        if self.store is None or str(self.store.folder) != str(Path(folder).expanduser().resolve()):
            self._set_folder(folder)
        self._add_recent_folder(folder)
        LOGGER.info("Starting scan requested for %s", folder)
        if not self._ensure_backend():
            return
        self._refresh_hardware_status()
        self.thumbnail_cache.clear()
        self.current_state = None
        self.current_samples = []
        self.review_samples = []
        self.undo_stack.clear()
        self.redo_stack.clear()
        self.focused_path = None
        if not self._ensure_scorer():
            return
        self.status_var.set("Scanning folder...")
        self._reset_progress_bar("0%", "Starting scan…")
        self._launch_background_scan(full_rescan=True)

    def cancel_scan(self) -> None:
        if not self._scan_active or self._scan_cancel_event is None:
            return
        self._scan_cancel_event.set()
        self.status_var.set("Stopping scan after the current batch...")
        self.stop_scan_button.configure(state="disabled")

    def update_algorithm(self) -> None:
        if self.store is None:
            messagebox.showinfo("Not ready", "Run a scan first.")
            return
        if self.current_state is None:
            messagebox.showinfo("Not ready", "Run a scan first.")
            return
        if self._scan_active:
            # Coalesce instead of stacking threads: labelling several photos in a row
            # used to start one rescore per click. They shared one scorer and one label
            # store, and an older one's `forget` pass could drop a label a newer one had
            # just written. One retrain after the current pass gives the same answer.
            self._retrain_pending = True
            self.status_var.set("Retrain queued — it will run when the current pass finishes.")
            return
        if not self._ensure_scorer():
            return
        self.status_var.set("Updating algorithm...")
        self._reset_progress_bar()
        self._launch_background_scan(full_rescan=False)

    def _run_pending_retrain(self) -> None:
        """Start the retrain that was asked for while another pass was running."""
        if not self._retrain_pending or self._scan_active or self.queue_active:
            return
        self._retrain_pending = False
        self.update_algorithm()

    def _launch_background_scan(self, full_rescan: bool) -> None:
        assert self.store is not None
        assert self.backend is not None
        assert self.scorer is not None
        generation = self._refresh_generation = self._refresh_generation + 1
        source_state = self.current_state
        self._scan_start_monotonic = time.monotonic()
        self._scan_active = True
        cancel_event = threading.Event()
        self._scan_cancel_event = cancel_event
        self.stop_scan_button.configure(state="normal")
        self._show_progress(True)
        if full_rescan or source_state is None:
            self._reset_progress_bar("0%", "Starting scan…")
        else:
            # A retrain reuses the cached embeddings and reports no item counts, so an
            # honest bar here is a spinner, not a fake percentage.
            self._start_indeterminate_progress("Re-ranking with your labels…")

        # Read Tk state here, on the main thread. Tk variables belong to the thread that
        # created the interpreter, and the worker used to call threshold_var.get()
        # itself — an unsupported cross-thread call into Tcl that raises outright when
        # the main loop is not currently running.
        threshold = float(self.threshold_var.get())
        batch_size = int(self.config.batch_size)
        store = self.store
        backend = self.backend
        scorer = self.scorer
        assert store is not None and backend is not None and scorer is not None

        def worker() -> None:
            def report_progress(progress: ScanProgress) -> None:
                self._after(0, self._scan_progress_update, progress)

            try:
                if full_rescan or source_state is None:
                    state, samples = scan_and_score_folder(
                        backend,
                        store,
                        scorer,
                        threshold=threshold,
                        batch_size=batch_size,
                        cancel_event=cancel_event,
                        progress_callback=report_progress,
                    )
                else:
                    labels = store.load_labels()
                    state, samples = scorer.rescore_state(
                        source_state,
                        labels,
                        threshold=threshold,
                        store=store,
                        cancel_event=cancel_event,
                    )
            except ScanCancelled:
                self._after(0, lambda token=generation: self._scan_cancelled(token))
                return
            except Exception as exc:  # noqa: BLE001
                self._after(0, lambda error=exc, token=generation: self._scan_failed(error, token))
                return
            self._after(
                0,
                lambda token=generation, new_state=state, new_samples=samples, is_full=full_rescan: (
                    self._scan_completed(
                        token,
                        new_state,
                        new_samples,
                        is_full,
                    )
                ),
            )

        threading.Thread(target=worker, daemon=True).start()

    def _scan_cancelled(self, generation: int) -> None:
        if generation != self._refresh_generation:
            return
        self._scan_active = False
        self._scan_cancel_event = None
        self.stop_scan_button.configure(state="disabled")
        self._reset_progress_bar()
        self._show_progress(False)
        # Stop means stop: a retrain queued behind this pass is dropped, not run.
        self._retrain_pending = False
        self.status_var.set("Scan stopped. Cached work is available for the next scan.")
        if self.queue_active:
            self.queue_active = False
            self.status_var.set("Scan stopped; queue paused.")

    def _scan_failed(self, exc: Exception, generation: int) -> None:
        if generation != self._refresh_generation:
            return
        self._scan_active = False
        self._scan_cancel_event = None
        self.stop_scan_button.configure(state="disabled")
        self._reset_progress_bar()
        self._show_progress(False)
        self._retrain_pending = False
        self.status_var.set("Scan failed.")
        LOGGER.error("Scan failed: %s", exc, exc_info=(type(exc), exc, exc.__traceback__))
        # Map common failures to user-friendly text with a suggested action.
        friendly = self._friendly_error(exc)
        messagebox.showerror("Scan failed", friendly)

    @staticmethod
    def _friendly_error(exc: Exception) -> str:
        """Translate common exceptions into user-readable text with a suggested fix."""
        msg = str(exc).lower()
        if "out of memory" in msg or "cuda oom" in msg or "memoryerror" in msg:
            return (
                "The scanner ran out of memory. Try reducing the batch size in Settings "
                "(e.g. from 16 to 4), or use a smaller model."
            )
        if "connection" in msg or "download" in msg or "timeout" in msg or "urlopen" in msg or "ssl" in msg:
            return (
                "Could not download the model. Check your internet connection and try "
                "again. The model is downloaded once and then cached locally."
            )
        if "no such file" in msg or "filenotfound" in msg or "permission" in msg or "access" in msg:
            return (
                f"A file could not be read. This may be a permission issue or a file that "
                f"was moved during the scan.\n\nDetails: {exc}"
            )
        if "cuda" in msg or "device" in msg or "gpu" in msg:
            return (
                "There was a problem with the GPU. Try switching the device to 'cpu' in "
                f"Settings if this keeps happening.\n\nDetails: {exc}"
            )
        return f"An unexpected error occurred during the scan.\n\nDetails: {exc}"

    def _scan_completed(
        self,
        generation: int,
        state: ScoreState,
        samples: list[dict[str, object]],
        full_rescan: bool,
    ) -> None:
        if generation != self._refresh_generation:
            return
        self._scan_active = False
        self._scan_cancel_event = None
        self.stop_scan_button.configure(state="disabled")
        # Land on a full bar before it goes away, so a fast cached scan does not look
        # like it stopped halfway.
        self._reset_progress_bar()
        self.progress_var.set(100)
        self.progress_text_var.set("100%")
        self._show_progress(False)
        previous_view = self.view_mode
        self.current_state = state
        processed_samples = self._post_process_review_samples(list(samples))
        self.review_samples = list(processed_samples)
        self.current_samples = list(processed_samples)
        self.view_mode = "review"
        self.similar_anchor_path = None
        self._restore_review_session()
        threshold = float(self.threshold_var.get())
        visible_mask = self._result_visibility_mask()
        matches = sum(
            1 for score, include in zip(state.scores, visible_mask, strict=False) if include and score >= threshold
        )
        if full_rescan:
            LOGGER.info(
                "Scan completed for %s: %d images, %d matches", self.folder_var.get().strip(), len(state.paths), matches
            )
            self.root.bell()
            self.status_var.set(
                f"Scan complete — {len(state.paths)} images, {matches} matches. Use J/K to navigate, A to accept, D to reject."
            )
            excluded = int(np.count_nonzero(state.excluded)) if state.excluded is not None else 0
            age_gated = sum(1 for stage in state.cascade_stage if stage == "minor")
            # Read skipped count from the scan metadata so the user knows some files
            # could not be read (corrupt, moved, unsupported encoding).
            skipped_count = 0
            if self.store is not None and self.store.metadata_path.exists():
                try:
                    import json

                    metadata = json.loads(self.store.metadata_path.read_text(encoding="utf-8"))
                    skipped_count = len(metadata.get("skipped", []))
                except Exception:  # noqa: BLE001
                    pass
            filtered_note = ""
            if excluded:
                filtered_note = f"\n{excluded} filtered out by the detection gates"
                filtered_note += f" ({age_gated} as possible minors)." if age_gated else "."
            if skipped_count:
                filtered_note += f"\n{skipped_count} file{'s' if skipped_count != 1 else ''} could not be read (corrupt or unsupported)."
            if matches:
                detail = (
                    "The detected files are listed below, grouped by what was detected.\n"
                    "Switch to 'Review queue' to teach the scanner."
                )
            else:
                detail = "Nothing scored above the threshold. Lower the Threshold slider to see the near misses."
            detail = f"{filtered_note}\n\n{detail}".lstrip()
            messagebox.showinfo(
                "Scan complete",
                f"Scan complete.\n\n{len(state.paths)} images scanned.\n{matches} matches at threshold {threshold:.3f}.\n\n"
                f"{detail}",
            )
        else:
            self.status_var.set("Scan complete.")
        # A fresh scan lands on the full detected-files list; retrains keep whichever view was active.
        if matches and (full_rescan or previous_view == "detected"):
            self.show_detected_files()
        else:
            self._refresh_displayed_results()
        self._update_stats_panel(record_history=True)
        self._save_review_session()
        self._save_last_folder(self.folder_var.get().strip())
        # Must be built the same way the watch poll builds it. Deriving the baseline
        # from state.paths instead left out every file the scan skipped as unreadable,
        # so a folder with one corrupt image looked "changed" on every single poll and
        # rescanned itself forever.
        self._watch_snapshot = self._collect_watch_snapshot()
        self._refresh_hardware_status()
        if self.queue_active and full_rescan:
            self.queue_index += 1
            if self.queue_index < len(self.scan_queue):
                self._after(0, self._start_next_queue_item)
            else:
                self._after(0, self._finish_queue)
        elif self._retrain_pending:
            # Labels arrived while this pass was running; fold them in now.
            self._after(0, self._run_pending_retrain)

    def _refresh_summary(self) -> None:
        if self.current_state is None:
            self.summary_var.set("")
            self.stats_var.set("")
            self.notice_var.set("")
            return
        threshold = float(self.threshold_var.get())
        visible_mask = self._result_visibility_mask()
        matches = sum(
            1
            for score, include in zip(self.current_state.scores, visible_mask, strict=False)
            if include and score >= threshold
        )
        self.summary_var.set(
            f"{len(self.current_state.paths)} images, {matches} above threshold, "
            f"{self.current_state.classifier_label_count} labeled, "
            f"{'classifier on' if self.current_state.classifier_trained else 'zero-shot only'}"
        )
        self._update_stats_panel(record_history=False)

    def _on_threshold_change(self, _value: str) -> None:
        self._refresh_summary()
        if self.view_mode != "detected" or self.current_state is None:
            return
        # Re-list the detected files for the new threshold, debounced so dragging the
        # slider does not rebuild the grid on every pixel.
        if self._threshold_refresh_after_id is not None:
            try:
                self.root.after_cancel(self._threshold_refresh_after_id)
            except Exception:  # noqa: BLE001
                pass
        self._threshold_refresh_after_id = self._after(400, self._relist_detected_files)

    def _relist_detected_files(self) -> None:
        self._threshold_refresh_after_id = None
        if self.view_mode != "detected" or self.current_state is None:
            return
        samples = self._detected_samples()
        self.current_samples = samples
        threshold = float(self.threshold_var.get())
        self.status_var.set(
            f"{len(samples)} detected files at threshold {threshold:.3f}, grouped by what was detected. "
            "Switch to 'Review queue' to teach the scanner."
        )
        self._refresh_displayed_results()

    def _bind_shortcuts(self) -> None:
        self.root.bind_all("<KeyPress-j>", self._handle_next_card)
        self.root.bind_all("<KeyPress-k>", self._handle_previous_card)
        self.root.bind_all("<Up>", self._handle_move_up)
        self.root.bind_all("<Down>", self._handle_move_down)
        self.root.bind_all("<Left>", self._handle_move_left)
        self.root.bind_all("<Right>", self._handle_move_right)
        self.root.bind_all("<KeyPress-g>", lambda event: self._handle_label_shortcut(event, 1))
        self.root.bind_all("<KeyPress-b>", lambda event: self._handle_label_shortcut(event, 0))
        self.root.bind_all("<KeyPress-a>", lambda event: self._handle_label_shortcut(event, 1))
        self.root.bind_all("<KeyPress-d>", lambda event: self._handle_label_shortcut(event, 0))
        self.root.bind_all("<KeyPress-s>", lambda event: self._handle_label_shortcut(event, 2))
        self.root.bind_all("<Control-z>", self._handle_undo_shortcut)
        self.root.bind_all("<Control-y>", self._handle_redo_shortcut)

    def _focus_is_text_input(self) -> bool:
        widget = self.root.focus_get()
        if widget is None:
            return False
        widget_class = widget.winfo_class()
        return widget_class in {"Entry", "Text", "TCombobox"}

    def _handle_next_card(self, event) -> str:
        if self._focus_is_text_input():
            return ""
        self.move_focus(1)
        return "break"

    def _handle_previous_card(self, event) -> str:
        if self._focus_is_text_input():
            return ""
        self.move_focus(-1)
        return "break"

    def _handle_move_up(self, event) -> str:
        if self._focus_is_text_input():
            return ""
        self.move_focus_grid(-1, 0)
        return "break"

    def _handle_move_down(self, event) -> str:
        if self._focus_is_text_input():
            return ""
        self.move_focus_grid(1, 0)
        return "break"

    def _handle_move_left(self, event) -> str:
        if self._focus_is_text_input():
            return ""
        self.move_focus(-1)
        return "break"

    def _handle_move_right(self, event) -> str:
        if self._focus_is_text_input():
            return ""
        self.move_focus(1)
        return "break"

    def _handle_label_shortcut(self, event, label: int) -> str:
        if self._focus_is_text_input():
            return ""
        self.label_focused_card(label)
        return "break"

    def _handle_undo_shortcut(self, event) -> str:
        if self._focus_is_text_input():
            return ""
        self.undo_last_label()
        return "break"

    def _handle_redo_shortcut(self, event) -> str:
        if self._focus_is_text_input():
            return ""
        self.redo_last_label()
        return "break"

    def move_focus(self, delta: int) -> None:
        paths = [str(sample["path"]) for sample in self.displayed_samples or self.current_samples]
        if not paths:
            return
        if self.focused_path not in paths:
            self.focused_path = paths[0]
        else:
            index = paths.index(self.focused_path)
            self.focused_path = paths[(index + delta) % len(paths)]
        self._apply_focus_visuals()

    def move_focus_grid(self, row_delta: int, column_delta: int) -> None:
        paths = [str(sample["path"]) for sample in self.displayed_samples or self.current_samples]
        if not paths:
            return
        columns = max(1, int(self.columns_var.get()))
        if self.focused_path not in paths:
            self.focused_path = paths[0]
        index = paths.index(self.focused_path)
        row = index // columns
        column = index % columns
        row = max(0, row + row_delta)
        column = max(0, column + column_delta)
        target = min(row * columns + column, len(paths) - 1)
        self.focused_path = paths[target]
        self._apply_focus_visuals()

    def focus_path(self, path: str) -> None:
        if self.current_state is None:
            return
        self.focused_path = path
        self._apply_focus_visuals()

    def _apply_focus_visuals(self) -> None:
        threshold = float(self.threshold_var.get())
        for path, card in self.cards.items():
            focused = path == self.focused_path
            is_match = card.score >= threshold
            try:
                if focused:
                    style = "FocusedMatchCard.TFrame" if is_match else "FocusedCard.TFrame"
                else:
                    style = "MatchCard.TFrame" if is_match else "Card.TFrame"
                card.frame.configure(style=style)
            except Exception:  # noqa: BLE001
                pass
            card.name_label.configure(text=f"▶ {Path(path).name}" if focused else Path(path).name)
        if self.focused_path and self.focused_path in self.cards:
            self._scroll_card_into_view(self.focused_path)
        self._update_preview()
        self._save_review_session()

    def _preview_caption_text(self, path: str) -> str:
        return (
            f"{Path(path).name}  —  score {self._match_score_for_path(path):.3f}  —  {self._label_display_text(path)}"
        )

    def _preview_size(self) -> tuple[int, int]:
        """Size of the enlarged active picture, scaled to the current window."""
        try:
            root_width = int(self.root.winfo_width())
            root_height = int(self.root.winfo_height())
        except Exception:  # noqa: BLE001
            root_width, root_height = 0, 0
        if root_width <= 1 or root_height <= 1:
            root_width, root_height = 992, 1041
        # A third of the window: big enough to judge a picture, small enough to leave
        # the result grid usable underneath.
        height = max(260, min(int(root_height * 0.34), 560))
        width = max(420, min(root_width - 60, int(height * 1.9)))
        return width, height

    def _on_root_configure(self, event) -> None:
        if getattr(event, "widget", None) is not self.root:
            return
        if self._preview_resize_after_id is not None:
            try:
                self.root.after_cancel(self._preview_resize_after_id)
            except Exception:  # noqa: BLE001
                pass
        self._preview_resize_after_id = self._after(200, self._resize_preview_if_needed)

    def _resize_preview_if_needed(self) -> None:
        self._preview_resize_after_id = None
        if not self.focused_path:
            return
        width, height = self._preview_size()
        last_width, last_height = self._preview_render_size
        # Ignore small changes: re-rendering costs a full image resize, and a tiny
        # nudge is not worth it.
        if abs(width - last_width) < 24 and abs(height - last_height) < 24:
            return
        self._update_preview()

    def _update_preview(self) -> None:
        path = self.focused_path
        if not path or path not in self.cards:
            self.preview_frame.pack_forget()
            return
        width, height = self._preview_size()
        self._preview_render_size = (width, height)
        cache_key = (path, width, height)
        photo = self.preview_cache.get(cache_key)
        if photo is None:
            try:
                photo = ImageTk.PhotoImage(self._preview_letterbox(open_oriented(path), width, height))
            except Exception:  # noqa: BLE001
                photo = ImageTk.PhotoImage(Image.new("RGB", (width // 2, height), color=self._palette()["button_bg"]))
            self.preview_cache[cache_key] = photo
            while len(self.preview_cache) > 8:
                self.preview_cache.popitem(last=False)
        else:
            self.preview_cache.move_to_end(cache_key)
        self.preview_image_label.configure(image=photo)
        self.preview_image_label.image = photo  # type: ignore[attr-defined]
        self.preview_caption_var.set(self._preview_caption_text(path))
        if not self.preview_frame.winfo_ismapped():
            self.preview_frame.pack(side=TOP, fill="x", before=self.canvas)

    def _scroll_card_into_view(self, path: str) -> None:
        card = self.cards.get(path)
        if card is None:
            return
        self.grid_canvas.update_idletasks()
        try:
            y = card.frame.winfo_y()
            height = max(self.grid_inner.winfo_height(), 1)
            self.grid_canvas.yview_moveto(max(min(y / height, 1.0), 0.0))
        except Exception:  # noqa: BLE001
            return

    def _refresh_hardware_status(self) -> None:
        backend_text = self._backend_summary()
        if psutil is None or self._psutil_process is None:
            self.hardware_var.set(backend_text)
        else:
            try:
                if not self._psutil_cpu_primed:
                    self._psutil_process.cpu_percent(None)
                    self._psutil_cpu_primed = True
                    self.hardware_var.set(f"{backend_text} | CPU -- | RAM -- | RSS --")
                else:
                    cpu_percent = self._psutil_process.cpu_percent(None)
                    memory_percent = psutil.virtual_memory().percent
                    rss_mb = self._psutil_process.memory_info().rss / (1024 * 1024)
                    self.hardware_var.set(
                        f"{backend_text} | CPU {cpu_percent:.0f}% | RAM {memory_percent:.0f}% | RSS {rss_mb:.0f} MB"
                    )
            except Exception:  # noqa: BLE001
                self.hardware_var.set(backend_text)
        # One repeating chain, not one per caller: this is invoked on startup, after the
        # model preload, and at both ends of every scan, and each of those used to start
        # its own self-rescheduling timer.
        if self._hardware_after_id is not None:
            try:
                self.root.after_cancel(self._hardware_after_id)
            except Exception:  # noqa: BLE001
                pass
        self._hardware_after_id = self._after(1500, self._refresh_hardware_status)

    def _clear_grid(self) -> None:
        for child in self.grid_inner.winfo_children():
            child.destroy()
        self.cards.clear()
        self.photo_refs.clear()

    def _card_info_width(self, columns: int) -> int:
        """Width every card reserves for its text column in this render pass.

        Derived from the space a card actually gets, so the text is neither clipped
        nor able to stretch its card wider than the cards beside it.
        """
        width = 0
        for widget in (self.grid_canvas, self.root):
            try:
                width = int(widget.winfo_width())
            except Exception:  # noqa: BLE001
                width = 0
            if width > 1:
                break
        if width <= 1:
            width = 1000
        thumb_size = max(120, int(self.thumbnail_size_var.get()))
        # Per card: padx 10 each side, 8 of frame padding each side, 10 after the image.
        card_width = width // max(1, columns) - 20
        return max(150, min(520, card_width - thumb_size - 36))

    def _render_samples(self) -> None:
        self._clear_grid()
        samples = self.displayed_samples if self.current_samples else []
        if not samples:
            self._update_preview()
            self._refresh_empty_state()
            self._sync_view_switch()
            return
        self._refresh_empty_state()
        self._sync_view_switch()
        columns = max(1, int(self.columns_var.get()))
        # Column config survives _clear_grid, so drop the settings for any column the
        # previous render used and this one does not — a stale weighted empty column
        # shifts every card sideways.
        for column in range(columns, 16):
            self.grid_inner.columnconfigure(column, weight=0, uniform="")
        for column in range(columns):
            self.grid_inner.columnconfigure(column, weight=1, uniform="cards")
        info_width = self._card_info_width(columns)
        buckets: list[str] = []
        grouped: dict[str, list[dict[str, object]]] = {}
        for sample in samples:
            bucket = str(sample["bucket"])
            if bucket not in grouped:
                buckets.append(bucket)
                grouped[bucket] = []
            grouped[bucket].append(sample)
        row = 0
        for bucket in buckets:
            items = grouped[bucket]
            ttk.Label(self.grid_inner, text=f"{bucket} ({len(items)})", font=("TkDefaultFont", 11, "bold")).grid(
                row=row,
                column=0,
                columnspan=columns,
                sticky="w",
                padx=10,
                pady=(10, 4),
            )
            row += 1
            for index, sample in enumerate(items):
                path = str(sample["path"])
                score = float(cast(float, sample["score"]))
                self._render_card(
                    self.grid_inner,
                    path,
                    score,
                    row + index // columns,
                    column=index % columns,
                    columnspan=1,
                    info_width=info_width,
                )
            row += (len(items) + columns - 1) // columns
        self._session_focus()

    def _render_card(
        self,
        container: ttk.Frame,
        path: str,
        score: float,
        row: int,
        column: int = 0,
        columnspan: int = 1,
        register: bool = True,
        show_actions: bool = True,
        info_width: int = CARD_INFO_WIDTH,
    ) -> ImageTk.PhotoImage:
        threshold = float(self.threshold_var.get())
        is_match = score >= threshold
        frame = ttk.Frame(container, padding=8, style="MatchCard.TFrame" if is_match else "Card.TFrame")
        frame.grid(row=row, column=column, columnspan=columnspan, sticky="nsew", padx=10, pady=6)
        frame.columnconfigure(1, weight=1)
        thumb_size = max(120, int(self.thumbnail_size_var.get()))
        # The content row is pinned to the thumbnail height and absorbs any slack, so
        # the action row underneath lands on the same line on every card.
        frame.rowconfigure(0, minsize=thumb_size, weight=1)
        cache_key = (path, thumb_size)
        try:
            photo = self.thumbnail_cache.get(cache_key)
            if photo is None:
                photo = ImageTk.PhotoImage(self._letterbox(open_oriented(path), thumb_size, thumb_size))
                self.thumbnail_cache[cache_key] = photo
            else:
                self.thumbnail_cache.move_to_end(cache_key)
            self._trim_thumbnail_cache()
        except Exception:  # noqa: BLE001
            photo = ImageTk.PhotoImage(Image.new("RGB", (thumb_size, thumb_size), color=self._palette()["button_bg"]))
            self.thumbnail_cache[cache_key] = photo
            self._trim_thumbnail_cache()

        image_label = ttk.Label(frame, image=photo)
        image_label.grid(row=0, column=0, padx=(0, 10), sticky="n")

        # Fixed-size text column with propagation switched off: a long filename or a
        # long details line is clipped instead of stretching the card.
        right = ttk.Frame(frame, width=info_width, height=thumb_size)
        right.grid(row=0, column=1, sticky="nsew")
        right.grid_propagate(False)
        right.columnconfigure(0, weight=1)

        info = ttk.Frame(right)
        info.grid(row=0, column=0, sticky="new")
        info.columnconfigure(0, weight=1)
        wrap_length = max(120, info_width - 12)
        name_label = ttk.Label(info, text=Path(path).name, wraplength=wrap_length, justify="left")
        name_label.grid(row=0, column=0, sticky="ew")
        score_label = ttk.Label(info, text=f"Score: {score:.3f}")
        score_label.grid(row=1, column=0, sticky="w", pady=(2, 0))
        label_label = ttk.Label(info, text=f"Label: {self._label_display_text(path)}")
        label_label.grid(row=2, column=0, sticky="w", pady=(2, 0))
        details_label = ttk.Label(info, text=self._axis_details_text(path), wraplength=wrap_length, justify="left")
        details_label.grid(row=3, column=0, sticky="ew", pady=(2, 0))
        # The card width is fixed above, so wrap the long labels to whatever width they
        # are actually given instead of letting them run past the card edge.
        self._autowrap(name_label)
        self._autowrap(details_label)

        if show_actions:
            # Buttons sit below the fixed-size image so they never get squeezed by the
            # text column and land in the same spot on every card.
            buttons = ttk.Frame(frame)
            buttons.grid(row=1, column=0, columnspan=2, sticky="sew", pady=(8, 0))
            # Accept and REJECT are packed side by side so they always share one
            # horizontal line, whatever the card width.
            primary = ttk.Frame(buttons)
            primary.grid(row=0, column=0, sticky="w", pady=(0, 4))
            ttk.Button(primary, text="Accept", width=12, command=lambda: self.set_label(path, 1)).pack(
                side=LEFT, padx=(0, 6)
            )
            ttk.Button(primary, text="REJECT", width=12, command=lambda: self.set_label(path, 0)).pack(side=LEFT)
            secondary = ttk.Frame(buttons)
            secondary.grid(row=1, column=0, sticky="w")
            actions = [
                ("View", lambda: self.view_image(path)),
                ("Skip", lambda: self.set_label(path, 2)),
                ("Find similar", lambda: self.find_similar(path)),
                ("Reveal", lambda: self.reveal_in_file_manager(path)),
            ]
            for text, command in actions:
                ttk.Button(secondary, text=text, command=command).pack(side=LEFT, padx=(0, 6))
        if register:
            self.cards[path] = ResultCard(
                frame=frame,
                path=path,
                name_label=name_label,
                score_label=score_label,
                label_label=label_label,
                details_label=details_label,
                image_ref=photo,
                score=score,
            )
            self.photo_refs.append(photo)
        def _on_click(_event: object, candidate: str = path) -> None:
            self.focus_path(candidate)

        def _on_double_click(_event: object, candidate: str = path) -> None:
            self.view_image(candidate)

        for widget in (frame, image_label, right, info, name_label, score_label, label_label, details_label):
            widget.bind("<Button-1>", _on_click)
            widget.bind("<Double-Button-1>", _on_double_click)
        return photo

    @staticmethod
    def _autowrap(label: ttk.Label) -> None:
        """Keep a label's wrap width in step with the width it is actually given."""

        def resize(event) -> None:
            wrap = max(80, int(event.width) - 4)
            try:
                current = int(label.cget("wraplength") or 0)
            except Exception:  # noqa: BLE001
                current = 0
            if current != wrap:
                label.configure(wraplength=wrap)

        label.bind("<Configure>", resize)

    @staticmethod
    def _letterbox(image: Image.Image, width: int, height: int) -> Image.Image:
        """Fit the image into a fixed-size transparent tile so widget sizes never vary."""
        thumb = image.convert("RGB")
        thumb.thumbnail((width, height))
        tile = Image.new("RGBA", (width, height), (0, 0, 0, 0))
        tile.paste(thumb, ((width - thumb.width) // 2, (height - thumb.height) // 2))
        return tile

    @staticmethod
    def _preview_letterbox(image: Image.Image, max_width: int, height: int) -> Image.Image:
        """Fit the active picture to a fixed height so the grid below never jumps.

        Only the height is padded: the label is centred by its packer, so padding the
        width too would just waste memory on a large transparent tile.
        """
        thumb = image.convert("RGB")
        thumb.thumbnail((max_width, height))
        tile = Image.new("RGBA", (max(thumb.width, 1), height), (0, 0, 0, 0))
        tile.paste(thumb, (0, (height - thumb.height) // 2))
        return tile

    def _label_display_text(self, path: str) -> str:
        return {
            "good": "Accepted",
            "bad": "REJECTED",
            "skip": "Skipped",
        }.get(self._label_text(path), "Unlabeled")

    def _label_text(self, path: str) -> str:
        if self.store is None:
            return "unlabeled"
        labels = self.store.load_labels()
        value = labels.get(path)
        if value == 1:
            return "good"
        if value == 0:
            return "bad"
        if value == 2:
            return "skip"
        return "unlabeled"

    def set_label(self, path: str, label: int) -> None:
        if self.store is None:
            return
        # Deciding on the active picture moves to the next one. Without this the big
        # preview sits on the image you just judged and the click looks like it did
        # nothing. Captured before labelling, because the retrain re-renders the grid.
        advance_from = path if path == self.focused_path else None
        self._apply_label_batch(
            {path: int(label)}, status=f"Saved label for {Path(path).name}. Retraining...", retrain=True
        )
        if advance_from is not None:
            self._advance_focus_after(advance_from)

    def _advance_focus_after(self, path: str) -> None:
        """Point the active picture at the next image in the list currently shown."""
        order = [str(sample["path"]) for sample in (self.displayed_samples or self.current_samples)]
        if len(order) < 2:
            return
        try:
            index = order.index(path)
        except ValueError:
            index = -1
        # Everything after the current position, then wrap around to what came before.
        for candidate in order[index + 1 :] + order[: max(index, 0)]:
            if candidate != path:
                self.focused_path = candidate
                self._apply_focus_visuals()
                return

    def _apply_label_batch(
        self,
        changes: dict[str, int | None],
        status: str,
        retrain: bool,
        record_undo: bool = True,
    ) -> None:
        if self.store is None:
            return
        labels = self.store.load_labels()
        before = {path: labels.get(path) for path in changes}
        for path, value in changes.items():
            if value is None:
                labels.pop(path, None)
            else:
                labels[path] = int(value)
        try:
            self.store.save_labels(labels)
        except OSError as exc:
            # Losing a decision silently is worse than interrupting the reviewer.
            LOGGER.exception("Could not save labels to %s", self.store.labels_path)
            self.status_var.set("Could not save your decision — see the message.")
            messagebox.showerror(
                "Could not save",
                f"Your decision could not be written to:\n{self.store.labels_path}\n\n{exc}\n\n"
                "Check that the folder is writable and has free space; nothing has been recorded.",
            )
            return
        if record_undo:
            self.undo_stack.append({"before": before, "after": dict(changes)})
            self.redo_stack.clear()
        for path in changes:
            self._refresh_label_state(path)
        self.status_var.set(status)
        self._save_review_session()
        if retrain:
            self.update_algorithm()

    def label_focused_card(self, label: int) -> None:
        if self.focused_path is None:
            return
        self.set_label(self.focused_path, label)

    def mark_all_shown(self, label: int) -> None:
        """Label every image currently on screen. Confirmed first — it is a bulk edit."""
        if self.current_state is None:
            messagebox.showinfo("No results", "Run a scan first.")
            return
        # Only what is actually displayed, so filters make this a precise tool.
        paths = [str(sample["path"]) for sample in (self.displayed_samples or self.current_samples)]
        if not paths:
            messagebox.showinfo("Nothing shown", "There are no images on screen to label.")
            return
        verb = {1: "Accept", 0: "REJECT", 2: "Skip"}.get(int(label), "label")
        if not messagebox.askyesno(
            f"{verb} everything shown",
            f"{verb} all {len(paths)} images currently on screen?\n\nUse Ctrl+Z afterwards if that was not what you wanted.",
        ):
            return
        self._apply_label_batch(
            dict.fromkeys(paths, label),
            status=f"{verb}ed {len(paths)} shown images.",
            retrain=True,
        )

    def undo_last_label(self) -> None:
        if self.store is None or not self.undo_stack:
            return
        action = self.undo_stack.pop()
        before = action.get("before")
        if not isinstance(before, dict):
            return
        self.redo_stack.append(action)
        self._apply_label_batch(
            {str(path): value for path, value in before.items()},
            status="Undo label change.",
            retrain=True,
            record_undo=False,
        )

    def redo_last_label(self) -> None:
        if self.store is None or not self.redo_stack:
            return
        action = self.redo_stack.pop()
        after = action.get("after")
        if not isinstance(after, dict):
            return
        self.undo_stack.append(action)
        self._apply_label_batch(
            {str(path): value for path, value in after.items()},
            status="Redo label change.",
            retrain=True,
            record_undo=False,
        )

    def _refresh_label_state(self, path: str) -> None:
        card = self.cards.get(path)
        if card is not None:
            card.label_label.configure(text=f"Label: {self._label_display_text(path)}")
            card.details_label.configure(text=self._axis_details_text(path))
        if path == self.focused_path:
            self.preview_caption_var.set(self._preview_caption_text(path))

    def export_matches(self) -> None:
        if self.current_state is None:
            messagebox.showinfo("No results", "Run a scan first.")
            return
        matches = self._visible_matches()
        if not matches:
            messagebox.showinfo("No matches", "No images are above the current threshold.")
            return
        out_dir = filedialog.askdirectory(title="Choose export folder")
        if not out_dir:
            return
        out_path = Path(out_dir)
        csv_path = out_path / "bikini_matches.csv"
        transfer_root = out_path / "matches"
        plan = self._build_transfer_plan(matches, transfer_root)
        self._preview_transfer_dialog(
            title="Export matches",
            plan=plan,
            confirm_text="Export",
            on_confirm=lambda: self._execute_export(plan, csv_path, transfer_root),
            summary=f"CSV: {csv_path}",
        )

    def copy_matches_to_subfolder(self) -> None:
        if self.current_state is None or self.store is None:
            messagebox.showinfo("No results", "Run a scan first.")
            return
        matches = self._visible_matches()
        if not matches:
            messagebox.showinfo("No matches", "No images are above the current threshold.")
            return
        target_dir = self.store.folder / MATCHES_DIR_NAME
        plan = self._build_transfer_plan(matches, target_dir)
        self._preview_transfer_dialog(
            title="Copy matches",
            plan=plan,
            confirm_text="Copy" if not self.move_files_var.get() else "Move",
            on_confirm=lambda: self._execute_transfer(plan, target_dir),
            summary=f"Target: {target_dir}",
        )

    def export_html_report(self) -> None:
        if self.current_state is None:
            messagebox.showinfo("No results", "Run a scan first.")
            return
        samples = list(self.displayed_samples or self.current_samples)
        if not samples:
            messagebox.showinfo("No results", "Nothing to export.")
            return
        out_dir = filedialog.askdirectory(title="Choose report folder")
        if not out_dir:
            return
        out_path = Path(out_dir) / "bikini_report.html"
        scores = self._score_map()
        labels = self._label_map()
        axis_scores = {
            path: {
                axis_name: float(values[idx])
                for axis_name, values in self.current_state.axis_scores.items()
                if idx < len(values)
            }
            for idx, path in enumerate(self.current_state.paths)
        }
        build_html_report(out_path, samples, labels, scores, axis_scores=axis_scores, title="Bikini Scanner report")
        messagebox.showinfo("HTML report exported", f"Report written to {out_path}")

    def write_metadata_to_visible(self) -> None:
        if self.current_state is None:
            messagebox.showinfo("No results", "Run a scan first.")
            return
        paths = [str(sample["path"]) for sample in self.displayed_samples or self.current_samples]
        if not paths:
            messagebox.showinfo("No results", "Nothing selected.")
            return
        if not messagebox.askyesno("Write metadata", f"Write tags into {len(paths)} image files?"):
            return
        written = 0
        for path in paths:
            score = self._match_score_for_path(path)
            if write_image_metadata(path, "bikini", score=score):
                written += 1
        messagebox.showinfo("Metadata written", f"Updated {written}/{len(paths)} files.")

    def trash_visible_files(self) -> None:
        if self.current_state is None:
            messagebox.showinfo("No results", "Run a scan first.")
            return
        paths = [str(sample["path"]) for sample in self.displayed_samples or self.current_samples]
        if not paths:
            messagebox.showinfo("No results", "Nothing selected.")
            return
        if not messagebox.askyesno("Move to trash", f"Send {len(paths)} files to the recycle bin/trash?"):
            return
        outcome = trash_files(paths)
        if not outcome.available:
            messagebox.showinfo("Trash unavailable", f"Recycle-bin support is unavailable: {outcome.reason}")
            return
        # Report what actually happened, and refresh whenever anything moved: a partial
        # failure still changed the folder, so leaving the grid untouched would show
        # files that are already in the recycle bin.
        if outcome.failed_count:
            LOGGER.warning("Trashed %d of %d files", outcome.trashed_count, len(paths))
            first = "\n".join(f"{path}: {error}" for path, error in outcome.failures[:5])
            more = f"\n...and {outcome.failed_count - 5} more." if outcome.failed_count > 5 else ""
            messagebox.showwarning(
                "Trash partly complete",
                f"Moved {outcome.trashed_count} of {len(paths)} files to the recycle bin/trash.\n\n"
                f"{outcome.failed_count} could not be moved:\n{first}{more}",
            )
        else:
            messagebox.showinfo(
                "Trash complete", f"Moved {outcome.trashed_count} files to the recycle bin/trash."
            )
        if outcome.trashed_count:
            self._refresh_after_output_change(move=True)

    def show_log_viewer(self) -> None:
        path = configure_logging()
        dialog, outer = self._create_modal("Recent log", padding=10, geometry="900x600")
        ttk.Label(outer, text=f"Log file: {path}").pack(anchor="w", pady=(0, 6))
        text = Text(outer, wrap="none", state="normal")
        text.pack(side=LEFT, fill=BOTH, expand=True)
        scroll = ttk.Scrollbar(outer, orient="vertical", command=text.yview)
        scroll.pack(side=RIGHT, fill="y")
        text.configure(yscrollcommand=scroll.set)
        text.insert("1.0", read_log_tail())
        text.configure(state="disabled")
        buttons = ttk.Frame(dialog, padding=(10, 0, 10, 10))
        buttons.pack(fill="x")
        ttk.Button(buttons, text="Refresh", command=lambda: self._refresh_log_text(text)).pack(side=LEFT)
        ttk.Button(buttons, text="Open log folder", command=lambda: self.reveal_in_file_manager(str(path))).pack(
            side=LEFT, padx=8
        )
        ttk.Button(
            buttons, text="Close", command=dialog._safe_close  # type: ignore[attr-defined]
        ).pack(side=RIGHT)

    @staticmethod
    def _refresh_log_text(text: Text) -> None:
        text.configure(state="normal")
        text.delete("1.0", END)
        text.insert("1.0", read_log_tail())
        text.configure(state="disabled")

    def show_duplicate_groups(self) -> None:
        if self.store is None:
            messagebox.showinfo("Duplicate groups", "Run or resume a scan first.")
            return
        groups = self.store.duplicate_groups()
        dialog, outer = self._create_modal("Exact duplicate groups", padding=10, geometry="900x600")
        ttk.Label(outer, text=f"{len(groups)} duplicate groups").pack(anchor="w", pady=(0, 6))
        text = Text(outer, wrap="none", state="normal")
        text.pack(side=LEFT, fill=BOTH, expand=True)
        scroll = ttk.Scrollbar(outer, orient="vertical", command=text.yview)
        scroll.pack(side=RIGHT, fill="y")
        text.configure(yscrollcommand=scroll.set)
        for index, (content_hash, paths) in enumerate(sorted(groups.items()), start=1):
            text.insert("end", f"Group {index} ({len(paths)} files, {content_hash[:12]}...)\n")
            for path in paths:
                text.insert("end", f"  {path}\n")
            text.insert("end", "\n")
        text.configure(state="disabled")
        buttons = ttk.Frame(dialog, padding=(10, 0, 10, 10))
        buttons.pack(fill="x")
        ttk.Button(
            buttons, text="Keep first, trash rest", command=lambda: self._trash_duplicate_remainders(groups, dialog)
        ).pack(side=LEFT)
        ttk.Button(
            buttons, text="Close", command=dialog._safe_close  # type: ignore[attr-defined]
        ).pack(side=RIGHT)

    def _trash_duplicate_remainders(self, groups: dict[str, list[str]], dialog: Toplevel) -> None:
        duplicates = [path for paths in groups.values() for path in paths[1:]]
        if not duplicates:
            messagebox.showinfo("Duplicate groups", "No duplicate files need removal.", parent=dialog)
            return
        if not messagebox.askyesno(
            "Move duplicates to trash",
            f"Keep the first file in each group and move {len(duplicates)} duplicate files to the recycle bin/trash?",
            parent=dialog,
        ):
            return
        outcome = trash_files(duplicates)
        if not outcome.available:
            messagebox.showinfo(
                "Trash unavailable", f"Recycle-bin support is unavailable: {outcome.reason}", parent=dialog
            )
            return
        LOGGER.info("Moved %d of %d duplicate files to trash", outcome.trashed_count, len(duplicates))
        if outcome.failed_count:
            first = "\n".join(f"{path}: {error}" for path, error in outcome.failures[:5])
            more = f"\n...and {outcome.failed_count - 5} more." if outcome.failed_count > 5 else ""
            messagebox.showwarning(
                "Duplicates partly removed",
                f"Moved {outcome.trashed_count} of {len(duplicates)} duplicate files to the recycle bin/trash.\n\n"
                f"{outcome.failed_count} could not be moved:\n{first}{more}",
                parent=dialog,
            )
        else:
            messagebox.showinfo(
                "Duplicates removed",
                f"Moved {outcome.trashed_count} duplicate files to the recycle bin/trash.",
                parent=dialog,
            )
        dialog.destroy()

    def clear_cache(self) -> None:
        if self.store is None:
            messagebox.showinfo("No cache", "Run or select a folder first.")
            return
        size_bytes = self.store.cache_size_bytes()
        size_mb = size_bytes / (1024 * 1024)
        if not messagebox.askyesno(
            "Clear cache",
            f"Delete {size_mb:.1f} MB of cached data for this folder?\n\nThis removes embeddings, labels, metadata, face counts, and the saved classifier.",
        ):
            return
        self.store.clear_cache()
        self.current_state = None
        self.current_samples = []
        self.review_samples = []
        self.displayed_samples = []
        self.focused_path = None
        self.undo_stack.clear()
        self.redo_stack.clear()
        self.scorer = None
        self.thumbnail_cache.clear()
        self._clear_grid()
        self.status_var.set("Cache cleared. Run a new scan.")
        self._refresh_summary()
        self._update_preview()
        self._refresh_empty_state()

    def _score_map(self) -> dict[str, float]:
        if self.current_state is None:
            return {}
        return {
            path: float(score) for path, score in zip(self.current_state.paths, self.current_state.scores, strict=False)
        }

    def _label_map(self) -> dict[str, int]:
        return self.store.load_labels() if self.store is not None else {}

    def _output_options(self) -> OutputOptions:
        return OutputOptions(
            organization=self.output_organization_var.get().strip() or "flat",
            score_band_low=float(self.output_score_low_var.get()),
            score_band_high=float(self.output_score_high_var.get()),
            filename_template=self.output_template_var.get().strip() or "{stem}",
            duplicate_policy=self.output_duplicate_var.get().strip() or "rename",
        )

    def _build_transfer_plan(self, matches: list[str], destination_root: Path) -> list[PlannedTransfer]:
        if self.current_state is None:
            return []
        options = self._output_options()
        scores = self._score_map()
        labels = self._label_map()
        return build_transfer_plan(
            matches,
            destination_root,
            scores,
            labels,
            options,
            timestamp=self.current_state.scan_timestamp,
            move=self.move_files_var.get(),
        )

    def _preview_transfer_dialog(
        self,
        title: str,
        plan: list[PlannedTransfer],
        confirm_text: str,
        on_confirm,
        summary: str = "",
    ) -> None:
        dialog, outer = self._create_modal(title, geometry="900x600")
        header = ttk.Label(outer, text=summary)
        header.pack(side=TOP, anchor="w")
        counts = {
            "copy": sum(1 for item in plan if item.action == "copy"),
            "move": sum(1 for item in plan if item.action == "move"),
            "overwrite": sum(1 for item in plan if item.action == "overwrite"),
            "skip": sum(1 for item in plan if item.action == "skip"),
            "collisions": sum(1 for item in plan if item.collision),
        }
        ttk.Label(
            outer,
            text=f"{len(plan)} files | copy {counts['copy']} | move {counts['move']} | overwrite {counts['overwrite']} | skip {counts['skip']} | collisions {counts['collisions']}",
        ).pack(side=TOP, anchor="w", pady=(6, 8))
        text = Text(outer, height=24, wrap="none")
        text.pack(side=TOP, fill=BOTH, expand=True)
        scrollbar = ttk.Scrollbar(outer, orient="vertical", command=text.yview)
        scrollbar.place(relx=1.0, rely=0.16, relheight=0.7, anchor="ne")
        text.configure(yscrollcommand=scrollbar.set)
        for item in plan:
            text.insert(
                "end",
                f"{item.source} -> {item.destination} [{item.action}] {item.label} {item.score:.3f} {item.band} {item.reason}\n",
            )
        text.configure(state="disabled")
        button_row = ttk.Frame(outer)
        button_row.pack(side=TOP, fill=BOTH, pady=(10, 0))
        ttk.Button(
            button_row, text="Cancel", command=dialog._safe_close  # type: ignore[attr-defined]
        ).pack(side=RIGHT, padx=(8, 0))

        def proceed() -> None:
            dialog._safe_close()  # type: ignore[attr-defined]
            on_confirm()

        ttk.Button(button_row, text=confirm_text, command=proceed).pack(side=RIGHT)

    def _execute_transfer(self, plan: list, destination_root: Path) -> None:
        move = self.move_files_var.get()
        processed, skipped, retained, failed = execute_transfer_plan(plan, move=move)
        verb = "Moved" if move else "Copied"
        retained_text = f" {retained} source(s) remained in place." if retained else ""
        messagebox.showinfo(
            "Transfer complete",
            f"{verb} {processed} files to {destination_root} (skipped {skipped}).{retained_text}"
            f"{self._failure_text(plan, failed)}",
        )
        self._refresh_after_output_change(move=move)

    @staticmethod
    def _failure_text(plan: list, failed: int) -> str:
        """Name the files that could not be transferred instead of only counting them."""
        if not failed:
            return ""
        names = [f"{Path(item.source).name}: {item.error}" for item in plan if getattr(item, "error", "")]
        listed = "\n".join(f"  • {name}" for name in names[:8])
        more = f"\n  … and {len(names) - 8} more" if len(names) > 8 else ""
        return f"\n\n{failed} file(s) could not be transferred (see the log):\n{listed}{more}"

    def _execute_export(self, plan: list, csv_path: Path, transfer_root: Path) -> None:
        if self.current_state is None:
            return
        with csv_path.open("w", newline="", encoding="utf-8") as handle:
            writer = csv.writer(handle)
            writer.writerow(["path", "filename", "score", "label", "timestamp"])
            for item in plan:
                writer.writerow(
                    [
                        str(item.source),
                        item.source.name,
                        f"{item.score:.6f}",
                        item.label,
                        self.current_state.scan_timestamp,
                    ]
                )
        processed, skipped, retained, failed = execute_transfer_plan(plan, move=self.move_files_var.get())
        retained_text = f" {retained} source(s) remained in place." if retained else ""
        messagebox.showinfo(
            "Export complete",
            f"Wrote {csv_path} and {processed} files to {transfer_root} (skipped {skipped}).{retained_text}"
            f"{self._failure_text(plan, failed)}",
        )
        self._refresh_after_output_change(move=self.move_files_var.get())

    def _refresh_after_output_change(self, move: bool = False) -> None:
        if move and self.folder_var.get().strip():
            self.run_scan()
            return
        if self.current_state is not None:
            self._refresh_current_results()


def launch_gui(
    config: ScannerConfig | None = None,
    initial_folder: str = "",
    initial_threshold: float | None = None,
) -> None:
    # Configure logging before anything can fail. Without this, a packaged build that
    # cannot load its model shows a window and writes nothing anywhere, which is
    # indistinguishable from "still starting up".
    configure_logging()
    LOGGER.info("Starting Bikini Scanner %s (frozen=%s)", __version__, getattr(sys, "frozen", False))
    root = TkinterDnD.Tk() if TkinterDnD is not None else Tk()
    root.geometry("1100x800")
    if config is None:
        config = ScannerConfig()
    if initial_threshold is not None:
        config.threshold = float(initial_threshold)
    # The constructor binds initial_folder itself, before the model preload starts.
    BikiniScannerApp(root, config=config, initial_folder=initial_folder)
    root.mainloop()
