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
from datetime import datetime
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
    simpledialog,
    ttk,
)
from tkinter import font as tkfont
from typing import cast

import numpy as np
from PIL import Image, ImageTk

from . import cascade, vision_analysis
from .__version__ import __version__
from .backend_utils import ImageEmbeddingBackend
from .config import HIGH_ACCURACY_MODEL, ScannerConfig, filter_folder_override
from .config_profiles import (
    BUILTIN_PROFILES,
    delete_profile,
    profile_config,
    profile_names,
    save_profile,
    without_secrets,
)
from .global_store import GlobalLearningStore
from .image_formats import heif_supported, open_oriented, oriented_size
from .logging_setup import configure_logging, log_path, read_log_tail
from .output_ops import (
    OutputOptions,
    PlannedTransfer,
    TrashOutcome,
    build_html_report,
    build_transfer_plan,
    execute_transfer_plan,
    format_output_name,
    label_name,
    organization_parts,
    trash_files,
    write_image_metadata,
)
from .plugins import apply_plugins, plugins_dir
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
from .store import MATCHES_DIR_NAME, SUPPORTED_IMAGE_SUFFIXES, FolderStore, collect_image_paths
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

# How many cards one page of the results grid holds. Chosen so a page renders in well
# under a second on a slow disk; the pager below the grid walks the rest.
DEFAULT_PAGE_SIZE = 120
# Above this many images, confirm before starting: a folder this size is hours of work
# and the count is already in hand by the time we could ask.
LARGE_SCAN_WARNING = 20000

# Chrome on a card below its thumbnail: the action row, the padding and a little of
# the bucket heading above it. The grid floor is derived from this plus the thumbnail,
# because "enough room for one whole card" is the thing that actually matters — at
# 950x700 the preview took 45% of the height and the Accept/REJECT buttons on every
# card sat below the fold, so labelling from the grid meant scrolling for each photo.
CARD_CHROME_HEIGHT = 115
# Target width for one card when the column count is left on "auto". Widening the
# window used to make two cards bigger rather than fitting more in, so a 1500px window
# gave each card ~720px for ~550px of content and the rest was empty.
TARGET_CARD_WIDTH = 460
# ...and the preview never takes more than this share of the workspace.
MAX_PREVIEW_SHARE = 0.6

# Sort order for the review buckets, with the detected-view buckets after them. Built
# once: this used to be reconstructed inside the sort key, i.e. once per sample per
# sort, which on a folder of a few thousand is a few thousand identical dictionaries.
BUCKET_ORDER = {
    "Likely match": 0,
    "Likely false positive": 1,
    "Likely false negative": 2,
    "Uncertain": 3,
}
BUCKET_ORDER.update({name: 10 + position for position, name in enumerate(DETECTED_BUCKETS)})

# Label value to the word the filters and the UI match on.
_LABEL_WORDS: dict[int | None, str] = {1: "good", 0: "bad", 2: "skip"}
PAGE_SIZE_CHOICES = (60, 120, 240, 500, 1000)

# Labelling used to fire one full retrain per click, and a retrain refits the learned
# model and rescores every image in the folder. Measured: ~120 ms per click on 500
# images climbing to ~1 s as labels accumulate, and ~1.5-1.8 s per click on 5 000. A
# reviewer working at one photo a second was queueing work faster than the machine
# could clear it, so the queue was permanently behind.
#
# The decision itself is saved to disk the instant it is made — that part is not
# deferred and cannot be lost. Only the re-rank is batched, and a re-rank is an
# optimisation: it reorders what you have not looked at yet. Waiting out a short pause
# in labelling costs a little freshness and saves a retrain per click.
RETRAIN_IDLE_MS = 2500
# ...unless the reviewer is on a roll and never pauses, in which case fold the labels
# in every so often anyway so the ranking does not go stale over a long session.
RETRAIN_LABEL_BURST = 25

# The review-session snapshot is an atomic whole-file write of every sample, and it is
# triggered from the render, focus and label paths alike. Batching it the same way
# keeps a fast reviewer off the disk; labels.json is written synchronously regardless,
# so nothing a reviewer decided depends on this landing.
SESSION_SAVE_IDLE_MS = 1200


@dataclass(slots=True)
class FilterContext:
    """The filter settings, read once and reused for a whole pass.

    Every one of these is a Tk variable, and reading one is a round trip into Tcl.
    `_sample_visible` read seven of them per sample, so filtering a few thousand
    results meant tens of thousands of round trips to answer questions whose answers
    could not change part-way through the pass.
    """

    labels: dict[str, int]
    search: str
    label_mode: str
    match_mode: str
    hide_decided: bool
    threshold: float
    score_min: float | None
    score_max: float | None
    browse: bool


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
    # What was last drawn for this card. Comparing against these turns the focus pass
    # from "two Tk calls per card on screen" into "two Tk calls, total".
    focused: bool = False
    style: str = ""



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
        # Text mirror of the slider, so the value is both readable and typeable.
        self.threshold_text_var = StringVar(value=f"{float(self.config.threshold):.2f}")
        self.theme_var = StringVar(value=str(self.user_prefs.get("theme", "dark")))
        self.font_size_var = IntVar(value=int(self.user_prefs.get("font_size", 10)))
        # 0 means "fit to the window"; a saved non-zero value is an explicit choice and
        # is still honoured.
        self.columns_var = IntVar(value=int(self.user_prefs.get("columns", 0)))
        # 240, not 320: the grid is a contact sheet, and now that thumbnails fill their
        # slot rather than letterboxing, a smaller one reads just as well and fits half
        # again as many cards across. A saved preference still wins.
        self.thumbnail_size_var = IntVar(value=int(self.user_prefs.get("thumbnail_size", 240)))
        self.page_size_var = IntVar(value=int(self.user_prefs.get("page_size", DEFAULT_PAGE_SIZE)))
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
        # Deciding a photo takes it out of the grid, so Accept/REJECT visibly advances
        # instead of re-rendering the same page. Off, every view keeps showing what you
        # already judged, which is what made the queue look stuck.
        self.hide_decided_var = BooleanVar(value=bool(self.user_prefs.get("hide_decided", True)))
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
        # Bucket headings are reused across renders alongside the cards; the layout
        # signature says when card geometry changed and a real rebuild is unavoidable.
        self._bucket_headings: dict[str, ttk.Label] = {}
        # Card context menus are kept alive here; a Menu that only the card references
        # is garbage-collected out from under Tk and posts an empty popup.
        self._card_menus: list[Menu] = []
        self._grid_layout: tuple[int, int, int] | None = None
        # path -> row in current_state, rebuilt whenever the state object changes.
        self._path_index: dict[str, int] = {}
        self._path_index_state: ScoreState | None = None
        self.displayed_samples: list[dict[str, object]] = []
        # One page of `displayed_samples`. The grid builds a Tk frame plus a decoded
        # thumbnail per card, so rendering a few thousand detected files at once wedges
        # the main loop for minutes. Everything that acts on a selection still uses
        # `displayed_samples`; only rendering and keyboard focus use the page.
        self.page_samples: list[dict[str, object]] = []
        self.page_index = 0
        # Paths in the order they are currently shown. A retrain re-ranks everything;
        # this pins what is already on screen so the photo about to be judged does not
        # move under the cursor. Cleared whenever the reviewer asks for a new order.
        self._page_order: list[str] = []
        self.review_samples: list[dict[str, object]] = []
        self.photo_refs: list[ImageTk.PhotoImage] = []
        self.thumbnail_cache: OrderedDict[tuple[str, int], ImageTk.PhotoImage] = OrderedDict()
        self.preview_caption_var = StringVar(value="")
        self.preview_cache: OrderedDict[tuple[str, int, int], ImageTk.PhotoImage] = OrderedDict()
        self._preview_render_size: tuple[int, int] = (0, 0)
        self._preview_resize_after_id: str | None = None
        # How much of the window the active picture gets. The reviewer sets this by
        # dragging the sash between the picture and the grid; it is remembered here so
        # the next session opens on the split they chose rather than a fixed fraction.
        try:
            self._preview_share = float(self.user_prefs.get("preview_share", 0.45))
        except (TypeError, ValueError):
            self._preview_share = 0.45
        self._threshold_refresh_after_id: str | None = None
        # Collapsible chrome: advanced controls stay out of the way until asked for.
        self._panels: dict[str, ttk.Frame] = {}
        self._panel_open: dict[str, bool] = {"filters": False, "queue": False}
        self.current_state: ScoreState | None = None
        self.current_samples: list[dict[str, object]] = []
        self.view_mode = "review"
        self.similar_anchor_path: str | None = None
        # Where Find similar was launched from: (view, samples, page, focused path).
        self._view_return: tuple[str, list[dict[str, object]], int, str | None] | None = None
        self.focused_path: str | None = None
        self.undo_stack: list[dict[str, object]] = []
        self.redo_stack: list[dict[str, object]] = []
        # Names what Ctrl+Z would reverse, so a bulk Accept is distinguishable from a
        # single misclick before you press it.
        self.undo_hint_var = StringVar(value="")
        self.quality_history: deque[float] = deque(maxlen=6)
        self.scan_queue: list[str] = []
        self.queue_active = False
        self.queue_index = 0
        self.watch_enabled_var = BooleanVar(value=False)
        # Standing indicator that watch mode is on and what it last saw, so a rescan
        # that starts on its own is attributable rather than mysterious.
        self._watch_notice_var = StringVar(value="")
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
        # Set while a model load is running for a scan that is waiting on it.
        self._backend_loading = False
        self._scan_after_backend: str | None = None
        # Image count handed back from the pre-scan walk, consumed by run_scan.
        self._pending_scan_count: int | None = None
        # Set by Resume last scan so _scan_completed restores the saved view and page.
        self._resuming_review = False
        self._resume_target: tuple[str, int] = ("", 0)
        self._scan_active = False
        # Set when a retrain is asked for while one is already running, so labelling a
        # run of photos quickly produces one retrain at the end rather than a thread per
        # click, all racing each other over the same scorer and label store.
        self._retrain_pending = False
        # Labels waiting to be folded into the model, and the timer that will do it.
        # See RETRAIN_IDLE_MS: the decisions themselves are already on disk, this is
        # only the re-rank catching up with them.
        self._labels_since_retrain = 0
        self._retrain_after_id: str | None = None
        self._session_save_after_id: str | None = None
        self._scroll_after_id: str | None = None
        self._pending_scroll_path: str | None = None
        self._reflow_after_id: str | None = None
        self._sash_clamp_after_id: str | None = None
        # Focus mode: one photo, full screen, keyboard only.
        self._focus_window: Toplevel | None = None
        self._focus_image_label: ttk.Label | None = None
        self._focus_caption: ttk.Label | None = None
        self._focus_photo: ImageTk.PhotoImage | None = None
        self._focus_caption_var = StringVar(value="")
        self._focus_status_var = StringVar(value="")
        self._focus_resize_after_id: str | None = None
        # Images decoded ahead of the reviewer by a background worker. Keyed by
        # (path, kind, width, height); consumed once and then handed to Tk.
        self._decoded_cache: OrderedDict[tuple[str, str, int, int], Image.Image] = OrderedDict()
        self._decoding: set[tuple[str, str, int, int]] = set()
        self._prefetch_queue: deque[tuple[str, str, int, int]] = deque()
        self._decoded_lock = threading.Lock()
        self._prefetch_thread: threading.Thread | None = None
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
        # transient() alone only ties the dialog to its parent; where it opens is left
        # to the window manager, which on a multi-monitor setup can be a different
        # screen from the app. Centre it once it knows its own size.
        dialog.after(0, lambda: self._centre_on_parent(dialog))
        dialog.configure(bg=self._palette()["bg"])
        outer = ttk.Frame(dialog, padding=padding)
        outer.pack(fill=BOTH, expand=True)
        return dialog, outer

    def _centre_on_parent(self, dialog: Toplevel) -> None:
        """Place a dialog over the middle of the main window, clamped to the screen."""
        try:
            dialog.update_idletasks()
            width = dialog.winfo_width() or dialog.winfo_reqwidth()
            height = dialog.winfo_height() or dialog.winfo_reqheight()
            x = self.root.winfo_rootx() + (self.root.winfo_width() - width) // 2
            y = self.root.winfo_rooty() + (self.root.winfo_height() - height) // 3
            x = max(0, min(x, dialog.winfo_screenwidth() - width))
            y = max(0, min(y, dialog.winfo_screenheight() - height))
            dialog.geometry(f"+{x}+{y}")
        except Exception:  # noqa: BLE001
            return

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
        self._build_sidebar()
        self._build_filter_panel()
        self._build_queue_panel()
        self._build_view_switch()

        # The active picture and the results grid share the window through a sash the
        # reviewer drags. A fixed fraction of the window was never right for everyone:
        # judging a photo wants a big picture, scanning a folder wants a big grid.
        self.workspace = ttk.PanedWindow(self.root, orient="vertical")
        self.workspace.pack(side=TOP, fill=BOTH, expand=True)

        # Learning readout and undo hint: one quiet strip directly under the results
        # rather than a second row of the status bar competing with everything else.
        self.stats_strip = ttk.Frame(self.root, padding=(12, 2, 12, 4))
        self.stats_strip.pack(side=BOTTOM, fill="x")
        ttk.Label(self.stats_strip, textvariable=self.stats_var, style="Muted.TLabel").pack(side=LEFT)
        ttk.Label(self.stats_strip, textvariable=self.undo_hint_var, style="Muted.TLabel").pack(side=RIGHT)

        self.canvas = ttk.Frame(self.workspace)
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
        def _on_grid_configure(event) -> None:
            self.grid_canvas.itemconfigure(self.grid_window, width=event.width)
            # The grid's own width is what decides how many cards fit, and it changes
            # for reasons the root never sees — opening the left rail narrows it by
            # 330px without resizing the window at all.
            self._schedule_reflow()

        self.grid_canvas.bind("<Configure>", _on_grid_configure)

        # Active-picture preview; inserted above the grid as the first pane whenever a
        # card is focused. The image itself is packed with no fill so it stays
        # horizontally centred no matter how wide the window gets, and the caption and
        # buttons are pinned to the bottom so they stay put as the picture grows.
        self.preview_frame = ttk.Frame(self.workspace, padding=(10, 4, 10, 6))
        preview_buttons = ttk.Frame(self.preview_frame)
        preview_buttons.pack(side=BOTTOM, pady=(6, 0))
        ttk.Button(preview_buttons, text="Accept (A)", width=14, command=lambda: self.label_focused_card(1)).pack(
            side=LEFT, padx=(0, 8)
        )
        ttk.Button(preview_buttons, text="REJECT (D)", width=14, command=lambda: self.label_focused_card(0)).pack(
            side=LEFT
        )
        self.preview_caption_label = ttk.Label(
            self.preview_frame,
            textvariable=self.preview_caption_var,
            anchor="center",
            justify="center",
        )
        self.preview_caption_label.pack(side=BOTTOM, pady=(4, 0))
        self.preview_image_label = ttk.Label(self.preview_frame, anchor="center")
        self.preview_image_label.pack(side=TOP, expand=True)
        self.preview_image_label.bind(
            "<Double-Button-1>",
            lambda _event: self.view_image(self.focused_path) if self.focused_path else None,
        )
        self.workspace.add(self.canvas, weight=1)
        # A sash drag ends with a button release; that is when the new split is worth
        # writing down. Tracking it during the drag would rewrite the prefs file on
        # every pixel of motion.
        self.workspace.bind("<ButtonRelease-1>", self._remember_preview_sash, add="+")
        # A PanedWindow keeps its sash at a fixed pixel offset when the window shrinks,
        # so a split that left room for a card at 1500x950 starved the grid at 950x700
        # and the card buttons went back under the fold. Re-clamp whenever the
        # workspace changes size.
        self.workspace.bind("<Configure>", self._on_workspace_configure, add="+")
        # Re-render the picture at the height the sash gives it, debounced: dragging
        # the sash or the window frame fires a stream of these and every one of them
        # costs a full image resize.
        self.preview_frame.bind("<Configure>", self._on_preview_configure, add="+")
        # Resize the active picture with the window, debounced for the same reason.
        self.root.bind("<Configure>", self._on_root_configure, add="+")
        self._build_empty_state()
        self._sync_panel_buttons()
        self._refresh_empty_state()

    # --- chrome -------------------------------------------------------------
    def _build_command_bar(self) -> None:
        """One row for the things you actually do: pick a folder and scan it.

        This row used to carry ten buttons, eight of which duplicated a menu entry
        that was already there. What is left is the two controls with no equal in the
        menus — the folder you are working on, and starting the scan — plus the one
        panel and the one dialog people reach for constantly. Update rankings, the
        scan queue, the theme and the shortcut list all live in the menu bar above.
        """
        bar = ttk.Frame(self.root, padding=(12, 10, 12, 8), style="Toolbar.TFrame")
        bar.pack(side=TOP, fill="x")
        self.command_bar = bar

        # Right-hand side is packed first so it keeps its width when the path is long.
        settings_button = ttk.Button(bar, text="Settings", command=self.open_settings_dialog)
        settings_button.pack(side=RIGHT)
        self._tooltip(settings_button, "Sensitivity, model, prompts and everything else")
        self.filter_toggle = ttk.Button(bar, text="Filters & view", command=lambda: self._toggle_panel("filters"))
        self.filter_toggle.pack(side=RIGHT, padx=(0, 6))
        self._tooltip(self.filter_toggle, "Search, sort, score range, thumbnail size, hide decided")

        ttk.Label(bar, text="Folder", style="Muted.TLabel").pack(side=LEFT, padx=(0, 8))
        folder_entry = ttk.Entry(bar, textvariable=self.folder_var)
        folder_entry.pack(side=LEFT, fill="x", expand=True, padx=(0, 6))
        self._tooltip(folder_entry, "The folder to scan. You can also drag a folder onto this window.")
        choose_button = ttk.Button(bar, text="Choose…", command=self.choose_folder)
        choose_button.pack(side=LEFT, padx=(0, 12))

        self.run_button = ttk.Button(bar, text="Run scan", command=self.run_scan, style="Accent.TButton")
        self.run_button.pack(side=LEFT)
        self._tooltip(self.run_button, "Scan this folder for bikini, cleavage and midriff photos")
        self.stop_scan_button = ttk.Button(
            bar, text="Stop scan", command=self.cancel_scan, state="disabled", width=10
        )
        self.stop_scan_button.pack(side=LEFT, padx=(6, 12))
        ttk.Label(bar, textvariable=self.override_var, style="Muted.TLabel").pack(side=LEFT)

    def _build_sidebar(self) -> None:
        """Left rail holding the collapsible panels. Hidden until one is opened."""
        self.sidebar = ttk.Frame(self.root, padding=(10, 8, 10, 8), style="Toolbar.TFrame", width=330)
        self.sidebar.pack_propagate(False)

    def _build_filter_panel(self) -> None:
        """Filters and view controls, stacked for the left rail.

        These were one long horizontal strip across the top of the window. Vertical
        space is what the app is short of, so they run down a narrow column now and
        each control gets its own labelled line instead of competing for width.
        """
        panel = ttk.Frame(self.sidebar, style="Toolbar.TFrame")
        self._panels["filters"] = panel

        def section(title: str) -> None:
            ttk.Label(panel, text=title.upper(), style="BucketHeading.TLabel").pack(
                side=TOP, anchor="w", pady=(10, 4)
            )

        def field(label: str, widget_factory) -> None:
            row = ttk.Frame(panel, style="Toolbar.TFrame")
            row.pack(side=TOP, fill="x", pady=(0, 4))
            ttk.Label(row, text=label, style="Muted.TLabel", width=11, anchor="w").pack(side=LEFT)
            widget_factory(row).pack(side=LEFT, fill="x", expand=True)

        section("Find")
        field("Search", lambda parent: ttk.Entry(parent, textvariable=self.search_var))
        field(
            "Sort by",
            lambda parent: ttk.Combobox(
                parent, textvariable=self.sort_var, values=("score", "filename", "date"), state="readonly"
            ),
        )
        field(
            "Show",
            lambda parent: ttk.Combobox(
                parent, textvariable=self.match_filter_var,
                values=("all", "matched", "unmatched"), state="readonly",
            ),
        )
        field(
            "Labels",
            lambda parent: ttk.Combobox(
                parent, textvariable=self.label_filter_var,
                values=("all", "unlabeled", "labeled", "skipped"), state="readonly",
            ),
        )

        score_row = ttk.Frame(panel, style="Toolbar.TFrame")
        score_row.pack(side=TOP, fill="x", pady=(0, 4))
        ttk.Label(score_row, text="Score", style="Muted.TLabel", width=11, anchor="w").pack(side=LEFT)
        ttk.Entry(score_row, textvariable=self.score_min_var, width=6).pack(side=LEFT)
        # An en dash is the correct typography for a numeric range separator here.
        ttk.Label(score_row, text=" – ", style="Muted.TLabel").pack(side=LEFT)  # noqa: RUF001
        ttk.Entry(score_row, textvariable=self.score_max_var, width=6).pack(side=LEFT)

        hide_decided = ttk.Checkbutton(panel, text="Hide decided photos", variable=self.hide_decided_var)
        hide_decided.pack(side=TOP, anchor="w", pady=(6, 0))
        self._tooltip(
            hide_decided,
            "Take a photo out of the grid as soon as you Accept, REJECT or Skip it, so the "
            "queue moves on. Untick to keep looking at what you have already judged.",
        )
        ttk.Checkbutton(
            panel, text="Only explicit (NSFW)", variable=self.nsfw_only_var, command=self._toggle_nsfw_only
        ).pack(side=TOP, anchor="w", pady=(2, 0))
        ttk.Button(panel, text="Clear filters", command=self.clear_filters).pack(
            side=TOP, fill="x", pady=(8, 0)
        )

        section("Grid")
        columns_row = ttk.Frame(panel, style="Toolbar.TFrame")
        columns_row.pack(side=TOP, fill="x", pady=(0, 4))
        ttk.Label(columns_row, text="Columns", style="Muted.TLabel", width=11, anchor="w").pack(side=LEFT)
        ttk.Spinbox(columns_row, from_=0, to=8, textvariable=self.columns_var, width=5).pack(side=LEFT)
        ttk.Label(columns_row, text="0 = fit window", style="Muted.TLabel").pack(side=LEFT, padx=(6, 0))
        field(
            "Thumbnails",
            lambda parent: ttk.Spinbox(
                parent, from_=120, to=520, increment=20, textvariable=self.thumbnail_size_var, width=6
            ),
        )
        field(
            "Text size",
            lambda parent: ttk.Spinbox(parent, from_=8, to=16, textvariable=self.font_size_var, width=6),
        )
        ttk.Button(panel, text="Prompt tester", command=self.open_prompt_tester_dialog).pack(
            side=TOP, fill="x", pady=(10, 0)
        )

    def _build_queue_panel(self) -> None:
        panel = ttk.Frame(self.sidebar, style="Toolbar.TFrame")
        self._panels["queue"] = panel
        ttk.Label(panel, text="SCAN QUEUE", style="BucketHeading.TLabel").pack(
            side=TOP, anchor="w", pady=(10, 4)
        )
        self.queue_listbox = Listbox(panel, height=7)
        self.queue_listbox.pack(side=TOP, fill="x")

        # The queue was append-only: a mistyped folder meant clearing it and starting
        # again, and the scan order could not be changed once set.
        order_row = ttk.Frame(panel, style="Toolbar.TFrame")
        order_row.pack(side=TOP, fill="x", pady=(6, 0))
        for text, command, width in (
            ("Up", lambda: self.move_queue_item(-1), 5),
            ("Down", lambda: self.move_queue_item(1), 6),
            ("Remove", self.remove_queue_item, 9),
            ("Clear", self.clear_queue, 7),
        ):
            ttk.Button(order_row, text=text, width=width, command=command).pack(side=LEFT, padx=(0, 4))

        ttk.Button(panel, text="Add folder\u2026", command=self.add_folder_to_queue).pack(
            side=TOP, fill="x", pady=(8, 0)
        )
        run_row = ttk.Frame(panel, style="Toolbar.TFrame")
        run_row.pack(side=TOP, fill="x", pady=(4, 0))
        ttk.Button(run_row, text="Run queue", command=self.run_queue).pack(side=LEFT, fill="x", expand=True)
        ttk.Button(run_row, text="Stop", width=7, command=self.stop_queue).pack(side=LEFT, padx=(4, 0))

        ttk.Label(panel, text="WATCH", style="BucketHeading.TLabel").pack(side=TOP, anchor="w", pady=(14, 4))
        ttk.Checkbutton(
            panel,
            text="Watch this folder for\nnew photos",
            variable=self.watch_enabled_var,
            command=self._toggle_watch_mode,
        ).pack(side=TOP, anchor="w")

    def _build_view_switch(self) -> None:
        """Which view, how sensitive, and which page — one row instead of two.

        The sensitivity slider had a row to itself above this one. Both rows were
        about the same thing (what the grid below is showing), and between them they
        pushed the results down by a third of a toolbar for no gain.
        """
        row = ttk.Frame(self.root, padding=(12, 2, 12, 8))
        row.pack(side=TOP, fill="x")
        self.view_switch_row = row
        self.detected_button = ttk.Button(
            row, text="Detected files", command=self.show_detected_files, style="Accent.TButton"
        )
        self.detected_button.pack(side=LEFT)
        self._tooltip(self.detected_button, "Every photo found, grouped by what was detected")
        self.review_button = ttk.Button(row, text="Review queue", command=self.restore_review_view)
        self.review_button.pack(side=LEFT, padx=(6, 16))
        self._tooltip(self.review_button, "A curated shortlist to Accept or REJECT so the scanner learns")
        # Pager to the right before the slider claims the slack, so it keeps its width.
        self._build_pager(row)
        ttk.Label(row, text="Sensitivity", style="Muted.TLabel").pack(side=LEFT)
        slider = ttk.Scale(
            row,
            from_=0.0,
            to=1.0,
            orient="horizontal",
            variable=self.threshold_var,
            command=self._on_threshold_change,
        )
        slider.pack(side=LEFT, fill="x", expand=True, padx=(8, 6))
        self._tooltip(slider, "Left shows more photos (and more false alarms); right shows only the surest matches")
        # The most-used control in the app had no number attached: you could not see
        # that it sat at 0.35, could not set it exactly, and the summary reported how
        # many were "above threshold" without saying what the threshold was.
        threshold_entry = ttk.Entry(row, textvariable=self.threshold_text_var, width=6, justify="center")
        threshold_entry.pack(side=LEFT, padx=(0, 10))
        threshold_entry.bind("<Return>", self._commit_threshold_text)
        threshold_entry.bind("<FocusOut>", self._commit_threshold_text)
        self._tooltip(threshold_entry, "The exact sensitivity. Type a value between 0 and 1 and press Enter.")
        ttk.Label(row, textvariable=self.summary_var, style="Muted.TLabel").pack(side=LEFT, padx=(0, 12))

    def _build_pager(self, row: ttk.Frame) -> None:
        """Page controls for the results grid, right-aligned on the view-switch row."""
        pager = ttk.Frame(row)
        pager.pack(side=RIGHT)
        self.pager_frame = pager
        self.next_page_button = ttk.Button(pager, text="Next >", width=9, command=self.next_page)
        self.next_page_button.pack(side=RIGHT)
        self.prev_page_button = ttk.Button(pager, text="< Prev", width=9, command=self.previous_page)
        self.prev_page_button.pack(side=RIGHT, padx=(0, 4))
        self.page_status_var = StringVar(value="")
        ttk.Label(pager, textvariable=self.page_status_var, style="Muted.TLabel").pack(side=RIGHT, padx=(0, 8))
        size_box = ttk.Combobox(
            pager,
            width=6,
            state="readonly",
            values=[str(value) for value in PAGE_SIZE_CHOICES],
        )
        size_box.set(str(self._page_size()))
        size_box.pack(side=RIGHT, padx=(0, 8))
        size_box.bind("<<ComboboxSelected>>", self._on_page_size_selected)
        self.page_size_box = size_box
        ttk.Label(pager, text="Per page", style="Muted.TLabel").pack(side=RIGHT, padx=(0, 4))

    def _page_size(self) -> int:
        try:
            return max(20, min(2000, int(self.page_size_var.get())))
        except Exception:  # noqa: BLE001
            return DEFAULT_PAGE_SIZE

    def _on_page_size_selected(self, _event: object = None) -> None:
        try:
            self.page_size_var.set(int(self.page_size_box.get()))
        except Exception:  # noqa: BLE001
            return
        self.page_index = 0
        self._save_user_prefs()
        if self.current_state is not None:
            self._refresh_displayed_results(reset_page=True)

    def _page_count(self) -> int:
        size = self._page_size()
        return max(1, (len(self.displayed_samples) + size - 1) // size)

    def next_page(self) -> None:
        if self.page_index + 1 >= self._page_count():
            return
        self.page_index += 1
        # A page boundary is where a pending re-rank is allowed to take effect.
        self._page_order = []
        self._refresh_displayed_results(reset_page=False, keep_focus=False)

    def previous_page(self) -> None:
        if self.page_index <= 0:
            return
        self.page_index -= 1
        self._page_order = []
        self._refresh_displayed_results(reset_page=False, keep_focus=False)

    def _sync_pager(self) -> None:
        if not hasattr(self, "page_status_var"):
            return
        total = len(self.displayed_samples)
        pages = self._page_count()
        size = self._page_size()
        if total <= size:
            # One page holds everything; the controls would only be noise.
            self.page_status_var.set(f"{total} shown" if total else "")
            state = "disabled"
        else:
            first = self.page_index * size + 1
            last = min(total, first + size - 1)
            self.page_status_var.set(f"{first}-{last} of {total}  (page {self.page_index + 1}/{pages})")
            state = "normal"
        try:
            self.prev_page_button.configure(state="normal" if (state == "normal" and self.page_index > 0) else "disabled")
            self.next_page_button.configure(
                state="normal" if (state == "normal" and self.page_index + 1 < pages) else "disabled"
            )
        except Exception:  # noqa: BLE001
            pass

    def _sync_view_switch(self) -> None:
        if not hasattr(self, "detected_button"):
            return
        detected = self.view_mode == "detected"
        hints = {
            "detected": "Everything above the sensitivity threshold.",
            "review": "A shortlist chosen to teach the scanner fastest.",
            "similar": "Images similar to the one you picked.",
            "browse": "Not scanned — decisions here are saved for the next scan.",
        }
        hint = hints.get(self.view_mode, "")
        # Say it out loud: a photo vanishing on Accept is only obvious once you know
        # it is meant to, and the checkbox that governs it lives in a closed panel.
        if self.hide_decided_var.get():
            hint = f"{hint} Decided photos are hidden."
        try:
            self.detected_button.configure(style="Accent.TButton" if detected else "TButton")
            self.review_button.configure(style="TButton" if detected else "Accent.TButton")
            self.view_hint.configure(text=hint.strip())
        except Exception:  # noqa: BLE001
            pass

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

        # One line of transient status on the left, one cluster of standing indicators
        # on the right. There were two rows and up to nine independent labels here -
        # status, VLM, watch, loading, hardware, learning stats, undo hint, view hint,
        # plateau notice - and at 950px wide they simply ran into each other. The
        # learning readout moved onto the stats strip under the results, where it can
        # be read deliberately rather than glanced past.
        status_row = ttk.Frame(outer, style="Toolbar.TFrame")
        status_row.pack(side=TOP, fill="x")
        self._status_row = status_row
        ttk.Label(
            status_row, textvariable=self.status_var, style="Toolbar.TLabel", anchor="w"
        ).pack(side=LEFT, fill="x", expand=True)

        indicators = ttk.Frame(status_row, style="Toolbar.TFrame")
        indicators.pack(side=RIGHT)
        self.hardware_label = ttk.Label(indicators, textvariable=self.hardware_var, style="Muted.TLabel")
        if psutil is not None:
            self.hardware_label.pack(side=RIGHT, padx=(10, 0))
        ttk.Label(indicators, textvariable=self.notice_var, style="Muted.TLabel").pack(side=RIGHT, padx=(10, 0))
        self.watch_badge = ttk.Label(indicators, text="WATCHING", style="Accent.TLabel", padding=(5, 0))
        # VLM badge: visible only when VLM adjudication is enabled, so the user knows
        # the scan includes a second-opinion stage.
        self.vlm_badge = ttk.Label(indicators, text="VLM", style="Accent.TLabel", padding=(5, 0))
        self.vlm_badge.pack(side=RIGHT, padx=(6, 0))
        self._refresh_vlm_badge()
        ttk.Label(indicators, textvariable=self.loading_var, style="Toolbar.TLabel").pack(side=RIGHT, padx=(10, 0))
        self.view_hint = ttk.Label(status_row, text="", style="Muted.TLabel")
        self.view_hint.pack(side=RIGHT, padx=(16, 16))

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
            body = (
                f"{folder}\n\nPress Run scan to look through this folder. To judge the photos "
                "yourself without waiting for the model, use File > Browse folder without scanning."
            )
            action = ("Run scan", self.run_scan)
        elif not self.page_samples:
            hidden = self._hidden_decided_count()
            # Running out of photos because you judged them all is a result, not an
            # empty screen, and telling that reviewer to lower the sensitivity — which
            # is what this used to do — sends them looking for a problem that is not there.
            # "and you decided at least one" matters: a folder where every image was
            # filtered out by the gates also has nothing undecided left in it, and that
            # is not the same thing at all.
            finished = hidden > 0 or (
                self.store is not None and bool(self.store.load_labels()) and self._undecided_remaining() == 0
            )
            if finished and not self._filters_active():
                title = "Everything here has been decided"
                body = (
                    "Every photo this view can show carries an Accept, REJECT or Skip, so there "
                    "is nothing left to judge.\n\nScan another folder, drag the sensitivity "
                    "slider left to pull in the near misses, or look back over what you decided."
                )
                if self.hide_decided_var.get():
                    action = ("Show decided photos", lambda: self.hide_decided_var.set(False))
                else:
                    action = ("Show detected files", self.show_detected_files)
            elif self._filters_active():
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

    def _refresh_watch_badge(self) -> None:
        badge = getattr(self, "watch_badge", None)
        if badge is None:
            return
        try:
            if self.watch_enabled_var.get():
                badge.pack(side=RIGHT, padx=(6, 0))
                self._tooltip(badge, self._watch_notice_var.get() or "Watching this folder for new photos")
            else:
                badge.pack_forget()
        except Exception:  # noqa: BLE001
            return

    def _refresh_vlm_badge(self) -> None:
        """Show or hide the VLM badge in the status bar based on config."""
        if not hasattr(self, "vlm_badge"):
            return
        try:
            if self.config.vlm_enabled:
                # side=RIGHT to match how it is first packed in _build_status_bar;
                # re-packing it LEFT moved the badge across the indicator cluster every
                # time the setting was toggled.
                self.vlm_badge.pack(side=RIGHT, padx=(6, 0))
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

    def _output_scope(self) -> tuple[list[str], str]:
        """Every path the bulk actions operate on, and a sentence naming that set.

        Copy, export, metadata and trash all said "visible", which by now depends on
        the search box, the score range, the label filter, Hide decided AND paging.
        One helper so they cannot drift apart, and one description so the confirmation
        dialog states the scope instead of implying it.
        """
        samples = self.displayed_samples or self.current_samples
        paths = [str(sample["path"]) for sample in samples]
        pages = self._page_count()
        parts = [f"{len(paths)} image{'s' if len(paths) != 1 else ''}"]
        if pages > 1:
            parts.append(f"across all {pages} pages, not just the page on screen")
        conditions = []
        if self.hide_decided_var.get():
            conditions.append("photos you have already decided are excluded")
        if self._filters_active():
            conditions.append("the current search, score range and label filters are applied")
        description = ", ".join(parts)
        if conditions:
            description += " (" + "; ".join(conditions) + ")"
        return paths, description

    def _hidden_decided_count(self) -> int:
        """How many of the current results the 'Hide decided' rule is holding back."""
        if not self.hide_decided_var.get() or self.store is None or not self.current_samples:
            return 0
        if self.label_filter_var.get().strip() not in ("", "all", "unlabeled"):
            return 0
        labels = self.store.load_labels()
        return sum(1 for sample in self.current_samples if labels.get(str(sample["path"])) is not None)

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
        """Show or hide one of the side panels.

        These used to open as extra rows above the results. With both open that was
        four rows of chrome between the toolbar and the first photo, on the one axis
        the app is short of. They live in a left rail now, where the space they take
        is horizontal and the grid keeps the full window height.
        """
        panel = self._panels.get(name)
        if panel is None:
            return
        # Tracked state, not winfo_ismapped(): mapping is only settled after the event
        # loop runs, so two toggles in a row would read each other's stale answer.
        wanted = not self._panel_open.get(name, False)
        self._panel_open[name] = wanted
        for key in ("filters", "queue"):
            other = self._panels.get(key)
            if other is None:
                continue
            if self._panel_open.get(key):
                other.pack(side=TOP, fill="x", pady=(0, 10))
            else:
                other.pack_forget()
        if any(self._panel_open.values()):
            if not self.sidebar.winfo_ismapped():
                self.sidebar.pack(side=LEFT, fill="y", before=self.workspace)
        else:
            self.sidebar.pack_forget()
        self._sync_panel_buttons()

    def _sync_panel_buttons(self) -> None:
        button = getattr(self, "filter_toggle", None)
        if button is None:
            return
        label = "Filters & view"
        # Show a dot when filters are active so the user knows something is hidden.
        if self._filters_active():
            label = f"{label} ●"
        try:
            button.configure(text=f"{label} ▴" if self._panel_open.get("filters") else f"{label} ▾")
        except Exception:  # noqa: BLE001
            return

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
            "  n                     add or edit a note on the active photo\n"
            "  w                     why did this photo score what it did\n"
            "  f                     focus mode: one photo, full screen, no grid\n"
            "  Backspace             back out of Find similar\n"
            "  Ctrl+Z / Ctrl+Y       undo / redo a decision\n"
            "  double-click          open the full-size viewer\n\n"
            "Tips\n"
            "  • The big picture at the top is the active one; Accept and REJECT apply to it.\n"
            "  • Drag the divider under it to make that picture bigger or smaller. The split\n"
            "    is remembered between sessions.\n"
            "  • A photo leaves the grid the moment you decide it, so the queue counts down.\n"
            "    Untick 'Hide decided' in Filters & view to keep looking at what you judged.\n"
            "  • The status bar shows how many are decided and how many are left in the folder,\n"
            "    and names what Ctrl+Z would undo before you press it.\n"
            "  • Decisions are saved the moment you make them; the ranking catches up a\n"
            "    couple of seconds later, so a run of Accepts costs one re-rank, not twenty.\n"
            "  • Every Accept and REJECT trains the scanner, and now carries over to other folders.\n"
            "  • Watch 'learning:' in the status bar: it names the labels counted and how much\n"
            "    influence they currently have over the ranking.\n"
            "  • 'Detected files' lists everything found, grouped by what was detected.\n"
            "  • Long lists are paged: use Prev/Next at the right of the view buttons, or\n"
            "    raise 'Per page' to show more at once.\n"
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
                # Decision colours. Chosen to stay legible on the card background and
                # to differ in lightness as well as hue, so they survive colour-blind
                # vision and the word beside them still carries the meaning.
                "accepted": "#5fd08a",
                "rejected": "#ff8f8f",
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
            "accepted": "#1f7a45",
            "rejected": "#b3261e",
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
        # Tabbed dialogs. clam's default notebook draws a near-white tab strip, which
        # is the one surface in the app that ignored the dark palette entirely.
        style.configure("TNotebook", background=palette["panel"], bordercolor=palette["border"], tabmargins=(2, 4, 2, 0))
        style.configure(
            "TNotebook.Tab",
            background=palette["button_bg"],
            foreground=palette["fg"],
            bordercolor=palette["border"],
            lightcolor=palette["button_bg"],
            darkcolor=palette["button_bg"],
            padding=(14, 6),
        )
        style.map(
            "TNotebook.Tab",
            background=[("selected", palette["panel"]), ("active", palette["button_active"])],
            foreground=[("selected", palette["fg"])],
            lightcolor=[("selected", palette["panel"])],
            expand=[("selected", (1, 1, 1, 0))],
        )
        # Muted.TLabel sits on toolbars; forms sit on the panel colour, so a note inside
        # a dialog needs its own background or it draws a lighter rectangle.
        style.configure("FormMuted.TLabel", background=palette["panel"], foreground=palette["muted"])
        # Decision colours. These sit on cards, whose background is the panel colour in
        # both card styles, so one background works for focused and unfocused alike.
        base = int(self.font_size_var.get())
        for name, colour in (
            ("Accepted", palette["accepted"]),
            ("Rejected", palette["rejected"]),
            ("Skipped", palette["muted"]),
            ("Undecided", palette["muted"]),
        ):
            style.configure(
                f"{name}.TLabel",
                background=palette["panel"],
                foreground=colour,
                font=("TkDefaultFont", base, "bold" if name in ("Accepted", "Rejected") else "normal"),
            )
        style.configure(
            "BucketHeading.TLabel",
            background=palette["bg"],
            foreground=palette["accent"],
            font=("TkDefaultFont", max(8, base - 1), "bold"),
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
            self.hide_decided_var,
            self.score_min_var,
            self.score_max_var,
            self.update_url_var,
        ):
            try:
                variable.trace_add("write", lambda *_args: self._on_ui_pref_change())
            except Exception:  # noqa: BLE001
                continue

    def _on_ui_pref_change(self) -> None:
        self._sync_update_menu()
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
        # Below this the command bar clips its own buttons and the grid has no room
        # for a card. There was no floor at all, so the window could be dragged to a
        # size at which nothing in it worked.
        try:
            self.root.minsize(900, 620)
        except Exception:  # noqa: BLE001
            pass
        geometry = self.user_prefs.get("window_geometry")
        if isinstance(geometry, str) and geometry:
            try:
                self.root.geometry(geometry)
            except Exception:  # noqa: BLE001
                pass

    def _persistable_config(self) -> dict[str, object]:
        """The settings worth writing to the preferences file.

        Everything except the VLM API key. A key is a credential, the preferences file
        is plain JSON in the user data directory, and the documented setup — a local
        Ollama or llama.cpp server — needs no key at all. It stays in memory for the
        session; a remote endpoint has to have it re-entered next launch.
        """
        return without_secrets(self.global_config.to_dict())

    def _save_user_prefs(self) -> None:
        payload = {
            # Scanner settings, not just UI state. Without this nothing from Tools >
            # Settings survived a restart: the model, the age gate, deep-scan mode, the
            # VLM configuration and the prompts all reverted to their defaults on every
            # launch, and the only way to keep them was to export a file by hand.
            # global_config, not config: a folder override is stored with its folder and
            # must not be promoted into the settings every other folder inherits.
            "scanner_config": self._persistable_config(),
            "theme": self.theme_var.get().strip(),
            "font_size": int(self.font_size_var.get()),
            "columns": int(self.columns_var.get()),
            "thumbnail_size": int(self.thumbnail_size_var.get()),
            "thumbnail_cache_size": self._thumbnail_cache_limit(),
            "page_size": self._page_size(),
            "search": self.search_var.get(),
            "sort": self.sort_var.get(),
            "match_filter": self.match_filter_var.get(),
            "label_filter": self.label_filter_var.get(),
            "hide_decided": bool(self.hide_decided_var.get()),
            "score_min": self.score_min_var.get().strip(),
            "score_max": self.score_max_var.get().strip(),
            "preview_share": round(self._preview_height_share(), 3),
            "scan_images_per_second": self.user_prefs.get("scan_images_per_second", 0.0),
            "folder_history": self._folder_history(),
            "output_destinations": self.user_prefs.get("output_destinations", {}),
            "trashed_files": self.user_prefs.get("trashed_files", []),
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
        edit_menu = Menu(self.menu_bar_frame, tearoff=False)
        view_menu = Menu(self.menu_bar_frame, tearoff=False)
        tools_menu = Menu(self.menu_bar_frame, tearoff=False)
        help_menu = Menu(self.menu_bar_frame, tearoff=False)
        recent_menu = Menu(file_menu, tearoff=False)

        file_menu.add_command(label="Open folder...", command=self.choose_folder)
        file_menu.add_command(label="Browse folder without scanning...", command=self.browse_without_scanning)
        file_menu.add_command(label="Resume last scan", command=self.resume_last_scan)
        file_menu.add_cascade(label="Recent folders", menu=recent_menu)
        file_menu.add_separator()
        file_menu.add_command(label="Exit", command=self._on_close)

        # Labelled dynamically by _sync_history_controls, which is why the indices are
        # kept: entryconfigure needs them and they must not drift.
        edit_menu.add_command(label="Undo", command=self.undo_last_label, accelerator="Ctrl+Z", state="disabled")
        self._undo_menu_index = edit_menu.index("end")
        edit_menu.add_command(label="Redo", command=self.redo_last_label, accelerator="Ctrl+Y", state="disabled")
        self._redo_menu_index = edit_menu.index("end")
        edit_menu.add_separator()
        edit_menu.add_command(label="Accept everything shown", command=lambda: self.mark_all_shown(1))
        edit_menu.add_command(label="Reject everything shown", command=lambda: self.mark_all_shown(0))
        self.edit_menu = edit_menu

        view_menu.add_command(label="Detected files", command=self.show_detected_files)
        view_menu.add_command(label="Review queue", command=self.restore_review_view)
        view_menu.add_command(label="Focus mode (one photo, full screen)", command=self.toggle_focus_mode, accelerator="F")
        view_menu.add_command(label="Back", command=self.go_back, accelerator="Backspace")
        view_menu.add_separator()
        # The panel toggles used to be buttons on the command bar. Filters kept its
        # button because it is opened constantly; the queue panel is a once-a-session
        # thing and did not earn permanent space next to Run scan.
        view_menu.add_command(label="Filters & view panel", command=lambda: self._toggle_panel("filters"))
        view_menu.add_command(label="Scan queue panel", command=lambda: self._toggle_panel("queue"))
        view_menu.add_checkbutton(label="Hide decided photos", variable=self.hide_decided_var)
        view_menu.add_separator()
        # Radiobuttons rather than commands: the menu now shows which theme is active,
        # which is what the "Theme: Dark" button on the command bar was there to say.
        view_menu.add_radiobutton(label="Light theme", value="light", variable=self.theme_var)
        view_menu.add_radiobutton(label="Dark theme", value="dark", variable=self.theme_var)
        view_menu.add_radiobutton(label="Follow Windows", value="system", variable=self.theme_var)

        # Grouped rather than a flat list of twenty. It had reached the point where
        # "Trash visible files" sat two lines under "Add folder to queue", which is a
        # long way to fall from adding a folder to deleting photographs.
        export_menu = Menu(tools_menu, tearoff=False)
        queue_menu = Menu(tools_menu, tearoff=False)
        maintenance_menu = Menu(tools_menu, tearoff=False)
        inspect_menu = Menu(tools_menu, tearoff=False)

        tools_menu.add_command(label="Settings", command=self.open_settings_dialog)
        tools_menu.add_command(label="Settings profiles", command=self.open_profiles_dialog)
        tools_menu.add_command(label="Output options", command=self.open_output_options_dialog)
        tools_menu.add_separator()
        tools_menu.add_command(label="Update rankings", command=self.update_algorithm)
        tools_menu.add_cascade(label="Inspect", menu=inspect_menu)
        tools_menu.add_cascade(label="Export", menu=export_menu)
        tools_menu.add_cascade(label="Scan queue", menu=queue_menu)
        tools_menu.add_separator()
        tools_menu.add_cascade(label="Maintenance", menu=maintenance_menu)

        inspect_menu.add_command(label="Why this score?", command=self.explain_score, accelerator="W")
        inspect_menu.add_command(label="Age gate report", command=self.show_age_gate_report)
        inspect_menu.add_command(label="Prompt tester", command=self.open_prompt_tester_dialog)
        inspect_menu.add_separator()
        inspect_menu.add_command(label="Duplicate groups", command=self.show_duplicate_groups)
        inspect_menu.add_command(label="Files that could not be read", command=self.show_skipped_files)
        inspect_menu.add_command(label="Recently trashed", command=self.show_trashed_files)

        export_menu.add_command(label="Copy matches to subfolder", command=self.copy_matches_to_subfolder)
        export_menu.add_command(label="Export matches (CSV)", command=self.export_matches)
        export_menu.add_command(label="Export HTML report", command=self.export_html_report)
        export_menu.add_command(label="Write metadata tags", command=self.write_metadata_to_visible)
        export_menu.add_separator()
        export_menu.add_command(label="Import settings", command=self.import_settings)
        export_menu.add_command(label="Export settings", command=self.export_settings)

        queue_menu.add_command(label="Add folder to queue", command=self.add_folder_to_queue)
        queue_menu.add_command(label="Run queue", command=self.run_queue)
        queue_menu.add_command(label="Stop the whole queue", command=self.stop_queue)

        # Everything under here changes or removes files. One submenu, kept away from
        # the things people click every few minutes.
        maintenance_menu.add_command(label="Clear cached scan data", command=self.clear_cache)
        maintenance_menu.add_command(label="Delete decisions for this folder", command=self.delete_decisions)
        maintenance_menu.add_command(label="Reset cross-folder learning", command=self.reset_global_learning)
        maintenance_menu.add_separator()
        maintenance_menu.add_command(label="Trash visible files", command=self.trash_visible_files)

        help_menu.add_command(label="Guide", command=self.show_guide)
        help_menu.add_command(label="Keyboard shortcuts", command=self.show_shortcuts)
        help_menu.add_command(label="About", command=self.show_about)
        help_menu.add_command(label="Log viewer", command=self.show_log_viewer)
        help_menu.add_command(label="Check for updates", command=self.check_for_updates)
        self._update_menu_index = help_menu.index("end")
        self.help_menu = help_menu
        self._sync_update_menu()

        self.menus = [
            file_menu,
            edit_menu,
            view_menu,
            tools_menu,
            help_menu,
            recent_menu,
            export_menu,
            queue_menu,
            maintenance_menu,
            inspect_menu,
        ]
        for label, menu in (
            ("File", file_menu),
            ("Edit", edit_menu),
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

    def _folder_history(self) -> dict[str, dict[str, object]]:
        history = self.user_prefs.get("folder_history")
        return dict(history) if isinstance(history, dict) else {}

    def _record_folder_history(self, image_count: int, match_count: int) -> None:
        """Remember what a scan of this folder found, so Recent folders means something.

        Kept in the preferences file rather than read back off each folder: the recent
        list is rendered on every menu build and half those paths may be on drives that
        are slow or absent.
        """
        folder = self.folder_var.get().strip()
        if not folder:
            return
        labels = self.store.load_labels() if self.store is not None else {}
        history = self._folder_history()
        history[folder] = {
            "scanned": datetime.now().isoformat(timespec="seconds"),
            "images": int(image_count),
            "matches": int(match_count),
            "decided": len(labels),
        }
        # Bounded by the recent list it decorates, plus a little slack.
        if len(history) > 40:
            for key in sorted(history, key=lambda name: str(history[name].get("scanned", "")))[: len(history) - 40]:
                history.pop(key, None)
        self.user_prefs["folder_history"] = history
        self._save_user_prefs()

    def _folder_history_text(self, folder: str) -> str:
        record = self._folder_history().get(folder)
        if not isinstance(record, dict):
            return ""
        try:
            when = datetime.fromisoformat(str(record.get("scanned", "")))
        except ValueError:
            return ""
        days = (datetime.now() - when).days
        ago = "today" if days <= 0 else ("yesterday" if days == 1 else f"{days} days ago")
        def count(key: str) -> int:
            try:
                return int(cast(int, record.get(key, 0)))
            except (TypeError, ValueError):
                return 0

        return (
            f"{ago}: {count('images'):,} images, "
            f"{count('matches'):,} matches, {count('decided'):,} decided"
        )

    def _rebuild_recent_menu(self) -> None:
        if not hasattr(self, "recent_menu"):
            return
        self.recent_menu.delete(0, END)
        if not self.recent_folders:
            self.recent_menu.add_command(label="No recent folders", state="disabled")
            return
        for folder in self.recent_folders[:10]:
            # A bare path says nothing about whether that folder is done. The summary
            # comes from prefs, not from the folder, so an absent drive costs nothing.
            summary = self._folder_history_text(folder)
            label = f"{folder}    —    {summary}" if summary else folder
            self.recent_menu.add_command(
                label=label, command=lambda value=folder: self._open_recent_folder(value)  # type: ignore[misc]
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
        """Reopen the last folder and put the reviewer back where they stopped.

        This re-ran the scan and dropped you at the top of the detected list, which is
        not what "resume" means to someone who was halfway through a review queue on
        page 7. The scan still runs — a ScoreState has to exist — but it reads cached
        embeddings, and the view, page and active photo come back with it.
        """
        folder = self._load_last_folder()
        if not folder or not Path(folder).is_dir():
            messagebox.showinfo("Resume scan", "No existing last-scanned folder was found.")
            return
        self._resuming_review = True
        self.open_folder(folder, scan=True)

    def _session_payload(self) -> dict[str, object]:
        return {
            "focused_path": self.focused_path,
            "page_index": self.page_index,
            "view_mode": self.view_mode,
            "current_samples": self.current_samples,
            "review_samples": self.review_samples,
            "quality_history": list(self.quality_history),
        }

    def _save_review_session(self) -> None:
        """Queue a session snapshot; the write itself happens once things go quiet.

        This serialises every sample and writes it atomically — temp file, fsync,
        rename. It is called from the render path, the focus path and the label path,
        so a single Accept was paying for three full-file writes with an fsync each.
        The snapshot only exists so a crash does not lose your place; the decisions
        themselves live in labels.json and are still written synchronously.
        """
        if self.store is None:
            return
        if self._session_save_after_id is not None:
            try:
                self.root.after_cancel(self._session_save_after_id)
            except Exception:  # noqa: BLE001
                pass
            self._session_save_after_id = None
        self._session_save_after_id = self._after(SESSION_SAVE_IDLE_MS, self._flush_review_session)

    def _flush_review_session(self) -> None:
        """Write the queued session snapshot now."""
        self._session_save_after_id = None
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
        # Only consulted when the reviewer asked to resume; a normal scan should still
        # land on the detected list rather than silently reopening an old view.
        view_mode = payload.get("view_mode")
        page_index = payload.get("page_index")
        self._resume_target = (
            str(view_mode) if isinstance(view_mode, str) else "",
            int(page_index) if isinstance(page_index, int) else 0,
        )
        quality_history = payload.get("quality_history")
        if isinstance(quality_history, list):
            self.quality_history.clear()
            for value in quality_history:
                try:
                    self.quality_history.append(float(value))
                except Exception:  # noqa: BLE001
                    continue

    def _update_stats_panel(
        self, record_history: bool = False, labels: dict[str, int] | None = None
    ) -> None:
        if self.current_state is None:
            self.stats_var.set("")
            self.notice_var.set("")
            return
        # The caller usually has the label map already; dragging the sensitivity slider
        # calls through here on every tick, and a fresh copy per tick is wasted work.
        if labels is None:
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
        if self.page_samples and (not self.focused_path or self.focused_path not in self.cards):
            self.focused_path = str(self.page_samples[0]["path"])
        self._apply_focus_visuals()

    def _append_queue_item(self, folder: str) -> None:
        folder = str(Path(folder).expanduser().resolve())
        if folder in self.scan_queue:
            return
        self.scan_queue.append(folder)
        self.queue_listbox.insert(END, folder)

    def _selected_queue_index(self) -> int | None:
        try:
            selection = self.queue_listbox.curselection()
        except Exception:  # noqa: BLE001
            return None
        return int(selection[0]) if selection else None

    def _reload_queue_listbox(self, select: int | None = None) -> None:
        self.queue_listbox.delete(0, END)
        for folder in self.scan_queue:
            self.queue_listbox.insert(END, folder)
        if select is not None and 0 <= select < len(self.scan_queue):
            self.queue_listbox.selection_set(select)
            self.queue_listbox.see(select)

    def remove_queue_item(self) -> None:
        index = self._selected_queue_index()
        if index is None:
            self.status_var.set("Select a folder in the queue to remove it.")
            return
        if self.queue_active:
            messagebox.showinfo("Queue running", "Stop the queue before changing it.")
            return
        removed = self.scan_queue.pop(index)
        self._reload_queue_listbox(select=min(index, len(self.scan_queue) - 1))
        self.status_var.set(f"Removed {Path(removed).name} from the queue.")

    def move_queue_item(self, delta: int) -> None:
        index = self._selected_queue_index()
        if index is None:
            self.status_var.set("Select a folder in the queue to move it.")
            return
        if self.queue_active:
            messagebox.showinfo("Queue running", "Stop the queue before changing it.")
            return
        target = index + delta
        if not 0 <= target < len(self.scan_queue):
            return
        self.scan_queue[index], self.scan_queue[target] = self.scan_queue[target], self.scan_queue[index]
        self._reload_queue_listbox(select=target)

    def clear_queue(self) -> None:
        if self.queue_active:
            messagebox.showinfo("Queue running", "Stop the queue before changing it.")
            return
        if not self.scan_queue:
            return
        if not messagebox.askyesno("Clear queue", f"Remove all {len(self.scan_queue)} folders from the queue?"):
            return
        self.scan_queue.clear()
        self._reload_queue_listbox()
        self.status_var.set("Queue cleared.")

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
            self._watch_notice_var.set(f"Watching {Path(self.folder_var.get()).name or 'this folder'}")
            self._refresh_watch_badge()
            self.status_var.set("Watch mode enabled — this folder is checked for new photos periodically.")
        else:
            self._watch_notice_var.set("")
            self._refresh_watch_badge()
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
            added = len(set(current_snapshot) - set(self._watch_snapshot))
            removed = len(set(self._watch_snapshot) - set(current_snapshot))
            self._watch_snapshot = current_snapshot
            # Watch mode only helps if you are told. It used to rescan in silence, so
            # unless you happened to be looking at the status bar at that moment, new
            # photos simply appeared with no explanation of where they came from.
            change = ", ".join(
                part
                for part in (
                    f"{added} added" if added else "",
                    f"{removed} removed" if removed else "",
                )
                if part
            ) or "changed on disk"
            self._watch_notice_var.set(f"Watch: {change} — rescanning")
            self.status_var.set(f"Watch folder: {change}. Rescanning...")
            self.root.bell()
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
            self._retrain_after_id,
            self._session_save_after_id,
            self._scroll_after_id,
            self._reflow_after_id,
            self._sash_clamp_after_id,
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
        self._retrain_after_id = None
        self._session_save_after_id = None
        self._scroll_after_id = None
        self._reflow_after_id = None
        self._sash_clamp_after_id = None
        self._focus_resize_after_id = None
        if self._focus_window is not None:
            try:
                self._focus_window.destroy()
            except Exception:  # noqa: BLE001
                pass
            self._focus_window = None
        # Written, not queued: there is no "later" left after destroy().
        self._flush_review_session()
        self._save_user_prefs()
        self.root.destroy()

    def browse_without_scanning(self, folder: str = "") -> None:
        """List a folder's images and let them be judged, with no model involved.

        Every route into this app previously required a full scan first, which is a
        model download and minutes of embedding before a single photo can be looked
        at. Sometimes the folder is small, or already sorted, or you only want to
        record decisions to teach the scanner later — none of which needs a score.

        The decisions land in the same labels.json a scan would read, so a scan run
        afterwards starts already knowing what you decided here.
        """
        folder = folder or self.folder_var.get().strip()
        if not folder:
            folder = filedialog.askdirectory(title="Browse folder without scanning")
            if not folder:
                return
        self._set_folder(folder)
        try:
            paths = [str(path) for path in collect_image_paths(Path(folder))]
        except OSError as exc:
            messagebox.showerror("Could not read the folder", str(exc))
            return
        if not paths:
            messagebox.showinfo("No images", f"No supported images were found in:\n{folder}")
            return
        count = len(paths)
        # A state with no embeddings and no scores. Everything downstream keys off
        # `current_state`, so an honest empty one is what makes the rest of the app —
        # filters, paging, focus mode, undo — work here without special cases.
        self.current_state = ScoreState(
            paths=paths,
            embeddings=np.zeros((count, 0), dtype=np.float32),
            zero_shot_scores=np.zeros(count, dtype=np.float32),
            scores=np.zeros(count, dtype=np.float32),
            axis_scores={},
            face_counts=None,
            classifier_trained=False,
            classifier_label_count=0,
            excluded=np.zeros(count, dtype=bool),
        )
        self.scorer = None
        self.view_mode = "browse"
        self.similar_anchor_path = None
        self._view_return = None
        self.current_samples = [{"path": path, "score": 0.0, "bucket": "Not scanned"} for path in paths]
        self.review_samples = list(self.current_samples)
        self.focused_path = paths[0]
        self._page_order = []
        self._refresh_displayed_results()
        self.status_var.set(
            f"Browsing {count} images without scoring them. Accept/REJECT is recorded and will "
            "be used the next time you run a scan here."
        )

    def open_folder(self, folder: str, *, scan: bool = False) -> None:
        self._set_folder(folder)
        if scan:
            self.run_scan()

    def _set_folder(self, folder: str, scan: bool = False) -> None:
        folder = str(Path(folder).expanduser().resolve())
        # A retrain queued against the folder being left must not run against the one
        # being opened: it would rescore the new folder using the old one's state.
        self._cancel_retrain_timer()
        self._retrain_pending = False
        self._labels_since_retrain = 0
        # Flush before self.store is replaced below, or the outgoing folder's snapshot
        # gets written into the incoming folder's cache.
        self._flush_review_session()
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
        self.page_samples = []
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
        payload, refused = filter_folder_override(self.config.to_dict())
        self.store.save_config_override(payload)
        self.folder_override_active = True
        self.override_var.set("Folder override active")
        self.status_var.set("Saved settings override for this folder.")
        if refused:
            # These are dropped by the trust boundary, not by accident. Saying so is the
            # difference between a deliberate restriction and a setting that appeared to
            # save and then quietly came back as the default.
            messagebox.showinfo(
                "Folder override saved",
                f"{len(payload)} setting(s) were pinned to this folder.\n\n"
                "A folder override cannot carry settings that reach off this machine, run "
                "code, choose a model or affect the age gate, because the file lives inside "
                "the folder being scanned. These stayed on your global settings:\n\n"
                f"{', '.join(refused)}",
            )

    def _reset_folder_override(self) -> None:
        if self.store is None:
            return
        self.store.clear_config_override()
        self.folder_override_active = False
        self.config = ScannerConfig.from_mapping(self.global_config.to_dict())
        self.override_var.set("")
        self._sync_config_controls()
        self.status_var.set("Folder override reset to global settings.")

    def show_guide(self) -> None:
        messagebox.showinfo(
            "Bikini Scanner guide",
            "1. Choose a folder and run a scan.\n"
            "2. Review with Accept, REJECT, or Skip. Each decision takes that photo out of\n"
            "   the grid and teaches the scanner.\n"
            "3. Drag the divider under the big picture to give it more or less room.\n"
            "4. Filters & view holds search, sorting, thumbnail size and 'Hide decided'.\n"
            "5. Settings covers detection, model, prompts and everything advanced.\n"
            "6. No model yet? File > Browse folder without scanning lets you judge photos\n"
            "   straight away; a later scan starts from those decisions.\n"
            "7. The View, Tools and Help menus hold the rest: scan queue, exports,\n"
            "   themes and keyboard shortcuts.",
        )

    def show_about(self) -> None:
        messagebox.showinfo(
            "About Bikini Scanner",
            f"Bikini Scanner {__version__}\n"
            "Local CPU-first bikini-content scanner with optional active learning, review tools, and per-folder caches.\n"
            f"User data: {prefs_path().parent}\n"
            f"Log file: {log_path()}",
        )

    def _sync_update_menu(self) -> None:
        """Grey out the update check when no manifest URL is set.

        It ships empty, so out of the box this menu item could only ever answer "no URL
        is configured" — an action offered at full strength that cannot succeed.
        """
        menu = getattr(self, "help_menu", None)
        if menu is None:
            return
        configured = bool(self.update_url_var.get().strip())
        try:
            menu.entryconfigure(
                self._update_menu_index,
                label="Check for updates" if configured else "Check for updates (no URL set)",
                state="normal" if configured else "disabled",
            )
        except Exception:  # noqa: BLE001
            return

    def check_for_updates(self) -> None:
        url = self.update_url_var.get().strip()
        if not url:
            messagebox.showinfo(
                "Check for updates",
                "No update manifest URL is configured, so there is nothing to check against.\n\n"
                "Set one in Settings > Advanced if you host one. With it empty the app never "
                "contacts anything.",
            )
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

    def _sample_sort_key(self, sample: dict[str, object], sort_mode: str | None = None) -> tuple[object, ...]:
        """Sort key for one sample.

        `sort_mode` is passed in by the batch caller so the Tk variable is read once
        per sort rather than once per sample. More importantly, the modification time
        is only looked up when sorting by date: this used to stat() every file on
        every sort whatever the mode, which on a few thousand images — especially on a
        network share — was the bulk of the time a filter change took.
        """
        path = str(sample["path"])
        score = float(cast(float, sample.get("score", 0.0)))
        if sort_mode is None:
            sort_mode = self.sort_var.get().strip()
        # os.path.basename rather than Path(...).name: this runs once per sample per
        # sort and constructing a Path just to read its last component is twice the
        # cost of the string operation it wraps.
        if sort_mode == "filename":
            return (os.path.basename(path).lower(), -score, path)
        if sort_mode == "date":
            try:
                modified = os.stat(path).st_mtime
            except OSError:
                modified = 0.0
            return (-modified, -score, path)
        return (
            BUCKET_ORDER.get(str(sample.get("bucket", "Uncertain")), 20),
            -score,
            os.path.basename(path).lower(),
            path,
        )

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

    def _filter_context(self) -> FilterContext:
        """Read every filter setting once, for a whole pass over the samples."""
        score_min, score_max = self._score_range()
        return FilterContext(
            labels=self.store.load_labels() if self.store is not None else {},
            search=self.search_var.get().strip().lower(),
            label_mode=self.label_filter_var.get().strip(),
            match_mode=self.match_filter_var.get().strip(),
            hide_decided=bool(self.hide_decided_var.get()),
            threshold=float(self.threshold_var.get()),
            score_min=score_min,
            score_max=score_max,
            browse=self.view_mode == "browse",
        )

    def _sample_visible(self, sample: dict[str, object], context: FilterContext | None = None) -> bool:
        """Whether one sample passes the current filters.

        `context` is built once by the batch caller. Without it this reached into the
        store for the whole label map and into Tk for seven variables, per sample.
        """
        if self.current_state is None:
            return False
        if context is None:
            context = self._filter_context()
        path = str(sample["path"])
        label = _LABEL_WORDS.get(context.labels.get(path), "unlabeled")
        label_mode = context.label_mode
        if label_mode == "unlabeled" and label != "unlabeled":
            return False
        if label_mode == "labeled" and label == "unlabeled":
            return False
        if label_mode == "skipped" and label != "skip":
            return False
        # A photo you have decided leaves the grid. Asking for the labeled or skipped
        # ones by name overrides this: that request is explicit, and silently answering
        # it with an empty grid would be the same bug in the other direction.
        if context.hide_decided and label != "unlabeled" and label_mode in ("", "all", "unlabeled"):
            return False
        if context.browse:
            # Nothing here has been scored, so every score-based filter would hide the
            # lot. Only the label and search filters mean anything.
            return not context.search or context.search in f"{path} {label}".lower()
        bucket = str(sample.get("bucket", ""))
        if context.search and context.search not in f"{path} {label} {bucket}".lower():
            return False
        score = float(cast(float, sample.get("score", 0.0)))
        if context.match_mode != "all":
            matches = score >= context.threshold
            if context.match_mode == "matched" and not matches:
                return False
            if context.match_mode == "unmatched" and matches:
                return False
        if context.score_min is not None and score < context.score_min:
            return False
        if context.score_max is not None and score > context.score_max:
            return False
        return not (
            context.score_min is not None
            and context.score_max is not None
            and context.score_min > context.score_max
        )

    def _apply_display_filters(self, samples: list[dict[str, object]]) -> list[dict[str, object]]:
        # Every filter setting and the sort mode are read once for the whole pass. All
        # of them used to be fetched per sample, which is what made a filter change on
        # a large folder take the better part of a second.
        context = self._filter_context()
        sort_mode = self.sort_var.get().strip()
        filtered = [sample for sample in samples if self._sample_visible(sample, context)]
        return sorted(filtered, key=lambda sample: self._sample_sort_key(sample, sort_mode))

    def _refresh_current_results(self) -> None:
        if self.current_state is None:
            return
        self.current_samples = self._post_process_review_samples(self._current_samples_from_state())
        self.review_samples = list(self.current_samples)
        self.view_mode = "review"
        self.similar_anchor_path = None
        self._refresh_displayed_results()

    def _hold_page_order(self, page: list[dict[str, object]]) -> list[dict[str, object]]:
        """Keep photos already on this page where the reviewer last saw them.

        A retrain re-ranks everything, so without this the photo you were about to
        judge slides somewhere else the moment the model catches up — and the click
        you were lining up lands on something else. Positions are pinned for as long
        as you stay on the page; new arrivals fall in at the end, and turning the page
        or changing a filter adopts the new ranking in full.
        """
        if not self._page_order:
            return page
        rank = {path: position for position, path in enumerate(self._page_order)}
        seen = [sample for sample in page if str(sample["path"]) in rank]
        seen.sort(key=lambda sample: rank[str(sample["path"])])
        arrived = [sample for sample in page if str(sample["path"]) not in rank]
        return seen + arrived

    def _refresh_displayed_results(self, reset_page: bool = True, keep_focus: bool = True) -> None:
        self.displayed_samples = self._apply_display_filters(self.current_samples)
        if reset_page:
            # A new result set starts at the top; paging through one does not. It also
            # means the reviewer asked for a different order, so the pinned one goes.
            self.page_index = 0
            self._page_order = []
        pages = self._page_count()
        self.page_index = max(0, min(self.page_index, pages - 1))
        size = self._page_size()
        start = self.page_index * size
        self.page_samples = self._hold_page_order(self.displayed_samples[start : start + size])
        self._page_order = [str(sample["path"]) for sample in self.page_samples]
        if not keep_focus:
            # Turning a page moves the active picture onto that page, otherwise the
            # preview keeps showing a photo that is no longer in the grid.
            self.focused_path = str(self.page_samples[0]["path"]) if self.page_samples else None
        self._refresh_summary()
        self._render_samples()
        self._sync_pager()
        self._save_review_session()

    def explain_score(self, path: str | None = None) -> None:
        """Show how one photo's score was arrived at.

        Everything here was already computed and kept on `ScoreState`; it was just
        never put in front of anyone. Without it the only way to answer "why did this
        score 0.42?" was to change a prompt and rescan to see what moved, which is a
        slow way to learn what the scanner is doing.
        """
        path = path or self.focused_path
        state = self.current_state
        if not path or state is None:
            messagebox.showinfo("No photo selected", "Run a scan and pick a photo first.")
            return
        index = self._state_index(path)
        if index is None:
            messagebox.showinfo("Not in this scan", "That photo is not part of the current results.")
            return

        threshold = float(self.threshold_var.get())
        final = float(state.scores[index])
        zero_shot = float(state.zero_shot_scores[index]) if index < len(state.zero_shot_scores) else final
        lines: list[str] = []
        lines.append(f"{Path(path).name}")
        lines.append("")
        verdict = "above" if final >= threshold else "below"
        lines.append(f"Final score  {final:.3f}   ({verdict} the {threshold:.3f} sensitivity setting)")

        stage = state.cascade_stage[index] if index < len(state.cascade_stage) else ""
        reason = state.cascade_reason[index] if index < len(state.cascade_reason) else ""
        excluded = bool(state.excluded[index]) if state.excluded is not None and index < len(state.excluded) else False
        if excluded or reason:
            lines.append(f"Gate         {reason or stage}")
            if excluded:
                lines.append("             This image is excluded, so its score is forced to zero.")
        lines.append("")

        # Prompt evidence versus what the labels taught, and how the two were mixed.
        lines.append("How the score was made up")
        lines.append(f"  prompts / cascade      {zero_shot:.3f}")
        if state.classifier_trained and state.learning_summary:
            lines.append(f"  learned from labels    {state.learning_summary}")
            if abs(final - zero_shot) >= 0.0005:
                direction = "raised" if final > zero_shot else "lowered"
                lines.append(f"  your labels {direction} it by {abs(final - zero_shot):.3f}")
            else:
                lines.append("  your labels did not move this one")
        else:
            lines.append("  learned from labels    not trained yet (prompt score only)")
        lines.append("")

        lines.append("What each axis saw   (0.00 = nothing; these are what the gates compare)")
        readable = {
            "bikini": "bikini",
            "bikini_top": "bikini top",
            "bikini_bottom": "bikini bottom",
            "cleavage": "cleavage",
            "midriff": "midriff",
            "nsfw": "explicit",
        }
        for axis_name, label in readable.items():
            values = state.axis_scores.get(axis_name)
            if values is None or index >= len(values):
                continue
            raw = float(values[index])
            strength = float(cascade.evidence(np.asarray([raw]))[0])
            bar = "#" * int(round(strength * 20))
            lines.append(f"  {label:<16} {strength:.2f}  {bar}")
        lines.append("")

        lines.append("Subject")
        for axis_name, label in (("evidence_female", "reads as female"), ("evidence_adult", "reads as adult")):
            values = state.axis_scores.get(axis_name)
            if values is not None and index < len(values):
                lines.append(f"  {label:<16} {float(values[index]):.2f}")
        if state.face_counts is not None and index < len(state.face_counts):
            count = int(state.face_counts[index])
            if count >= 0:
                lines.append(f"  faces detected   {count}")
        region = state.detail_regions[index] if index < len(state.detail_regions) else ""
        lines.append(f"  best crop        {region or 'whole image only'}")
        if state.deep_scanned:
            lines.append("  (body-region crops were examined for this scan)")
        else:
            lines.append("  (whole frame only — turn Deep scan on to examine body crops)")

        if state.refine is not None and index < len(state.refine.scores):
            value = float(state.refine.scores[index])
            if np.isfinite(value):
                source = "vision-LLM" if state.refine.source == "vlm" else "high-accuracy model"
                lines.append("")
                lines.append(f"Second opinion   {source} scored this {value:.3f} and was blended in")

        note = self.note_for(path)
        if note:
            lines.append("")
            lines.append(f"Your note: {note}")

        dialog, outer = self._create_modal(
            "Why this score", padding=12, geometry="720x620", resizable=(True, True)
        )
        body = ttk.Frame(outer)
        body.pack(fill=BOTH, expand=True)
        text = Text(body, wrap="word", font=("TkFixedFont", 9))
        text.pack(side=LEFT, fill=BOTH, expand=True)
        scroll = ttk.Scrollbar(body, orient="vertical", command=text.yview)
        scroll.pack(side=RIGHT, fill="y")
        text.configure(yscrollcommand=scroll.set)
        palette = self._palette()
        text.configure(bg=palette["entry_bg"], fg=palette["fg"], insertbackground=palette["fg"], relief="solid")
        text.insert("1.0", "\n".join(lines))
        text.configure(state="disabled")
        row = self._modal_button_row(outer)
        ttk.Button(row, text="Close", command=dialog._safe_close).pack(side=RIGHT)  # type: ignore[attr-defined]
        def tune_prompts() -> None:
            dialog._safe_close()  # type: ignore[attr-defined]
            self.open_prompt_tester_dialog()

        ttk.Button(row, text="Tune the prompts", command=tune_prompts).pack(side=RIGHT, padx=(0, 8))

    def show_age_gate_report(self) -> None:
        """Summarise what the age gate did, so it can be checked without weakening it.

        The gate is on by default and deliberately errs toward excluding; flagged
        images are forced to zero and hidden everywhere, which is the right behaviour
        and also means the only evidence it is working is a count. This reports what it
        did without putting any excluded image back on screen.
        """
        state = self.current_state
        if state is None:
            messagebox.showinfo("Age gate", "Run a scan first.")
            return
        stages = list(state.cascade_stage)
        if not stages:
            messagebox.showinfo("Age gate", "This scan recorded no gate decisions.")
            return
        counts: dict[str, int] = {}
        for stage in stages:
            counts[stage] = counts.get(stage, 0) + 1
        minor_rows = [index for index, stage in enumerate(stages) if stage == cascade.STAGE_MINOR]
        adult = state.axis_scores.get("evidence_adult")
        child = state.axis_scores.get("evidence_child")
        lines = [
            f"Scanned {len(stages):,} images with the age gate "
            f"{'ON' if self.config.exclude_minors else 'OFF'} "
            f"(sensitivity {self.config.minor_threshold:.2f}; lower is stricter).",
            "",
            "Every image ended in one of these stages:",
        ]
        for stage, count in sorted(counts.items(), key=lambda item: -item[1]):
            label = cascade.STAGE_REASONS.get(stage) or ("scored normally" if stage == cascade.STAGE_SCORED else stage)
            share = 100.0 * count / len(stages)
            lines.append(f"  {count:>6,}  {share:5.1f}%   {label}")
        lines.append("")
        if minor_rows:
            lines.append(f"{len(minor_rows):,} image(s) were excluded by the age gate.")
            if child is not None and adult is not None:
                child_values = [float(child[i]) for i in minor_rows if i < len(child)]
                adult_values = [float(adult[i]) for i in minor_rows if i < len(adult)]
                if child_values:
                    lines.append(
                        f"  child evidence on those: min {min(child_values):.2f}, "
                        f"median {sorted(child_values)[len(child_values) // 2]:.2f}, max {max(child_values):.2f}"
                    )
                if adult_values:
                    lines.append(
                        f"  adult evidence on those: min {min(adult_values):.2f}, "
                        f"median {sorted(adult_values)[len(adult_values) // 2]:.2f}, max {max(adult_values):.2f}"
                    )
            lines.append("")
            lines.append(
                "Excluded images are not listed and cannot be shown: that is the point of the\n"
                "gate. If the count looks wrong for your material, the sensitivity setting in\n"
                "Settings > Detection is the control — lower excludes on less evidence."
            )
        else:
            lines.append("The age gate excluded nothing in this folder.")
        if not self._face_model_installed():
            lines.append("")
            lines.append(
                "Note: the face model is not installed, so the age check reads whole images\n"
                "rather than face crops. Installing it in Settings > Detection makes this\n"
                "markedly more reliable."
            )
        messagebox.showinfo("Age gate report", "\n".join(lines))

    def _face_model_installed(self) -> bool:
        return "off" not in self._face_model_status().lower()

    def _state_index(self, path: str) -> int | None:
        """Row for a path in the current state, via a map built once per scan.

        Both callers used a linear scan. `_axis_details_text` runs once per card, so a
        full page against a few thousand images was hundreds of thousands of string
        comparisons for a lookup that should be constant time.
        """
        state = self.current_state
        if state is None:
            return None
        if self._path_index_state is not state:
            self._path_index = {str(value): position for position, value in enumerate(state.paths)}
            self._path_index_state = state
        return self._path_index.get(path)

    def _axis_details_text(self, path: str) -> str:
        if self.current_state is None:
            return ""
        index = self._state_index(path)
        if index is None:
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
        index = self._state_index(path)
        if index is None or self.current_state is None:
            return 0.0
        return float(self.current_state.scores[index])

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

    def show_detected_files(self, reset_page: bool = True) -> None:
        if self.current_state is None:
            messagebox.showinfo("No results", "Run a scan first.")
            return
        samples = self._detected_samples()
        if not samples:
            messagebox.showinfo(
                "No detected files",
                "No files scored above the current threshold.\n\n"
                "Drag the Sensitivity slider left to include the near misses.",
            )
            return
        self._apply_detected_view(samples, reset_page=reset_page)

    def _apply_detected_view(self, samples: list[dict[str, object]], reset_page: bool) -> None:
        self.view_mode = "detected"
        self.similar_anchor_path = None
        self.current_samples = samples
        threshold = float(self.threshold_var.get())
        # Count what is on screen, not what was found: with decided photos hidden the
        # two differ, and a headline number the grid contradicts is worse than none.
        self._refresh_displayed_results(reset_page=reset_page)
        counts: dict[str, int] = {}
        for sample in self.displayed_samples:
            bucket = str(sample.get("bucket", ""))
            counts[bucket] = counts.get(bucket, 0) + 1
        breakdown = ", ".join(f"{name} {count}" for name, count in counts.items() if count)
        hidden = len(samples) - len(self.displayed_samples)
        self.status_var.set(
            f"{len(self.displayed_samples)} detected files at threshold {threshold:.3f}"
            + (f" — {breakdown}" if breakdown else "")
            + (f" ({hidden} already decided or filtered out)" if hidden > 0 else "")
            + ". Switch to 'Review queue' to teach the scanner."
        )

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
        # Remember where the reviewer was. Without this, "Find similar" was a one-way
        # door: the only ways out were the two view buttons, both of which rebuild
        # from scratch and lose the page and the active photo.
        self._view_return = (self.view_mode, list(self.current_samples), self.page_index, self.focused_path)
        self.view_mode = "similar"
        self.similar_anchor_path = anchor_path
        self.current_samples = [
            {"path": path, "score": similarity, "bucket": "Similar"} for path, similarity in ranking
        ]
        self.status_var.set(
            f"Showing images similar to {Path(anchor_path).name}. Press Backspace, or View > Back, to return."
        )
        self._refresh_displayed_results()
        self._refresh_summary()

    def go_back(self) -> None:
        """Return to the view Find similar was launched from, page and photo intact."""
        if self._view_return is None:
            self.status_var.set("Nothing to go back to.")
            return
        mode, samples, page_index, focused = self._view_return
        self._view_return = None
        self.view_mode = mode
        self.similar_anchor_path = None
        self.current_samples = list(samples)
        self.page_index = page_index
        self.focused_path = focused
        self._page_order = []
        self._refresh_displayed_results(reset_page=False)
        self.status_var.set("Back to where you were.")

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
        # Pan and zoom are tracked in source-image coordinates: (cx, cy) is the point of
        # the photo sitting under the middle of the canvas. A zoom of 0 means "not
        # decided yet" and resolves to fit-the-window on the first draw.
        state: dict[str, float] = {"zoom": 0.0, "cx": image.width / 2.0, "cy": image.height / 2.0}
        photo_ref: list[ImageTk.PhotoImage] = []
        pan_origin: dict[str, float] = {}
        MAX_ZOOM = 8.0

        def _canvas_size() -> tuple[int, int]:
            return max(int(canvas.winfo_width()), 1), max(int(canvas.winfo_height()), 1)

        def _fit_zoom(width: int, height: int) -> float:
            return min(width / image.width, height / image.height)

        def _clamped_zoom(value: float, width: int, height: int) -> float:
            # Never smaller than fitting the window, never past MAX_ZOOM. Zooming out
            # past the fit only adds empty space, and the ceiling is what stops a
            # 12 MP photo being asked for at 8x.
            return max(_fit_zoom(width, height), min(float(value), MAX_ZOOM))

        def redraw() -> None:
            width, height = _canvas_size()
            if state["zoom"] <= 0:
                state["zoom"] = _fit_zoom(width, height)
            zoom = _clamped_zoom(state["zoom"], width, height)
            state["zoom"] = zoom
            # The slice of the source that is actually on screen at this zoom.
            view_w = min(image.width, max(1, int(round(width / zoom))))
            view_h = min(image.height, max(1, int(round(height / zoom))))
            cx = min(max(float(state["cx"]), view_w / 2), image.width - view_w / 2)
            cy = min(max(float(state["cy"]), view_h / 2), image.height - view_h / 2)
            state["cx"], state["cy"] = cx, cy
            left = int(round(cx - view_w / 2))
            top = int(round(cy - view_h / 2))
            # Only the visible part is ever resampled, so the cost of a draw is bounded
            # by the size of the window rather than by the zoom. Resizing the whole
            # image instead meant 8x on a 12 MP photo allocated a ~750 megapixel
            # intermediate — gigabytes — and every mouse-move during a pan paid for
            # another full LANCZOS pass over the original.
            crop = image.crop((left, top, left + view_w, top + view_h))
            target = (
                max(1, min(width, int(round(view_w * zoom)))),
                max(1, min(height, int(round(view_h * zoom)))),
            )
            resample = Image.Resampling.LANCZOS if zoom < 1.0 else Image.Resampling.BILINEAR
            photo = ImageTk.PhotoImage(crop.resize(target, resample))
            photo_ref[:] = [photo]
            canvas.delete("image")
            canvas.create_image(width // 2, height // 2, anchor="center", image=photo, tags="image")
            canvas.tag_lower("image", overlay)
            canvas.coords(overlay, 10, height - 30)

        def _zoom_by(delta: float, pointer_x: float, pointer_y: float) -> str:
            width, height = _canvas_size()
            old = _clamped_zoom(state["zoom"] or _fit_zoom(width, height), width, height)
            new = _clamped_zoom(old * (1.1 if delta > 0 else 1 / 1.1), width, height)
            if new != old:
                # Keep whatever is under the pointer under the pointer.
                offset_x = pointer_x - width / 2
                offset_y = pointer_y - height / 2
                state["cx"] = float(state["cx"]) + offset_x * (1.0 / old - 1.0 / new)
                state["cy"] = float(state["cy"]) + offset_y * (1.0 / old - 1.0 / new)
                state["zoom"] = new
            redraw()
            return "break"

        def start_pan(event) -> None:
            pan_origin.update(
                {"x": float(event.x), "y": float(event.y), "cx": float(state["cx"]), "cy": float(state["cy"])}
            )

        def move_pan(event) -> None:
            if not pan_origin:
                return
            zoom = max(float(state["zoom"]), 1e-6)
            state["cx"] = pan_origin["cx"] - (float(event.x) - pan_origin["x"]) / zoom
            state["cy"] = pan_origin["cy"] - (float(event.y) - pan_origin["y"]) / zoom
            redraw()

        def close_viewer() -> None:
            image.close()
            try:
                viewer.grab_release()
            except Exception:  # noqa: BLE001
                pass
            viewer.destroy()

        canvas.bind("<MouseWheel>", lambda event: _zoom_by(getattr(event, "delta", 0), event.x, event.y))
        canvas.bind("<Button-4>", lambda event: _zoom_by(120, event.x, event.y))
        canvas.bind("<Button-5>", lambda event: _zoom_by(-120, event.x, event.y))
        canvas.bind("<ButtonPress-1>", start_pan)
        canvas.bind("<B1-Motion>", move_pan)
        viewer.protocol("WM_DELETE_WINDOW", close_viewer)
        viewer.bind("<Escape>", lambda _event: close_viewer())
        canvas.bind("<Configure>", lambda _event: redraw())
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

    def open_plugins_folder(self) -> None:
        """Open (creating if needed) the folder plugins are loaded from."""
        directory = plugins_dir()
        try:
            directory.mkdir(parents=True, exist_ok=True)
        except OSError as exc:
            messagebox.showerror("Plugins folder unavailable", f"{directory}\n\n{exc}")
            return
        self.reveal_in_file_manager(str(directory))

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
        # One 44-row form meant the window opened at whatever width the widest label
        # happened to need, could not be widened, and buried the two or three settings
        # anyone actually changes under thirty they never touch. Four short tabs, a
        # resizable window, and the rarely-touched numbers gathered under Advanced.
        dialog, outer = self._create_modal("Settings", resizable=(True, True))
        palette = self._palette()

        notebook = ttk.Notebook(outer)
        notebook.pack(side=TOP, fill=BOTH, expand=True)
        self.settings_notebook = notebook

        def add_tab(title: str) -> ttk.Frame:
            page = ttk.Frame(notebook, padding=(14, 12))
            notebook.add(page, text=title)
            # Column 1 carries the controls and takes the slack when the dialog widens.
            page.columnconfigure(1, weight=1)
            return page

        # Detection first: it holds the sensitivity and the age gate, which are the
        # settings a reviewer actually reaches for.
        detection_tab = add_tab("Detection")
        model_tab = add_tab("Model")
        prompts_tab = add_tab("Prompts")
        advanced_tab = add_tab("Advanced")

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
        enable_plugins_var = BooleanVar(value=self.config.enable_plugins)
        refine_var = BooleanVar(value=bool(self.config.refine_model))
        vlm_enabled_var = BooleanVar(value=self.config.vlm_enabled)
        vlm_base_url_var = StringVar(value=self.config.vlm_base_url)
        vlm_model_var = StringVar(value=self.config.vlm_model)
        vlm_concurrency_var = StringVar(value=str(self.config.vlm_concurrency))
        vlm_band_var = StringVar(value=str(self.config.vlm_band))
        vlm_max_images_var = StringVar(value=str(self.config.vlm_max_images))
        vlm_api_key_var = StringVar(value=self.config.vlm_api_key)
        vlm_timeout_var = StringVar(value=str(self.config.vlm_timeout))
        vlm_weight_var = StringVar(value=str(self.config.vlm_weight))
        refine_band_var = StringVar(value=str(self.config.refine_band))
        refine_max_images_var = StringVar(value=str(self.config.refine_max_images))
        refine_weight_var = StringVar(value=str(self.config.refine_weight))
        pipeline_var = StringVar(value=self.config.pipeline)

        def add_labeled_entry(
            parent: ttk.Frame, row: int, label: str, variable: StringVar, width: int = 40, tip: str = ""
        ) -> ttk.Entry:
            caption = ttk.Label(parent, text=label)
            caption.grid(row=row, column=0, sticky="w", pady=(0, 6), padx=(0, 12))
            entry = ttk.Entry(parent, textvariable=variable, width=width)
            # A four-character number stretched across the whole column reads as a
            # text field; only the wide ones take the slack when the dialog grows.
            entry.grid(row=row, column=1, sticky="ew" if width >= 24 else "w", pady=(0, 6))
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

        def add_inline_entry(
            parent: ttk.Frame, row: int, label: str, variable: StringVar, width: int = 10, tip: str = ""
        ) -> ttk.Entry:
            """A caption+entry pair for a row whose column 0 is already a checkbox.

            add_labeled_entry always puts its caption in column 0, so on the rows that
            pair a checkbox with a number the caption would land in the same cell as
            the checkbox and the two would be drawn on top of each other. Everything
            here lives in column 1.
            """
            holder = ttk.Frame(parent)
            holder.grid(row=row, column=1, sticky="w", padx=(14, 0), pady=(0, 6))
            caption = ttk.Label(holder, text=label)
            caption.pack(side=LEFT, padx=(0, 8))
            entry = ttk.Entry(holder, textvariable=variable, width=width)
            entry.pack(side=LEFT)
            if tip:
                self._tooltip(caption, tip)
                self._tooltip(entry, tip)
            return entry

        def add_combo(
            parent: ttk.Frame, row: int, label: str, variable: StringVar, values: tuple[str, ...], tip: str
        ) -> ttk.Combobox:
            caption = ttk.Label(parent, text=label)
            caption.grid(row=row, column=0, sticky="w", pady=(0, 6), padx=(0, 12))
            combo = ttk.Combobox(parent, textvariable=variable, values=values, state="readonly")
            combo.grid(row=row, column=1, sticky="ew", pady=(0, 6))
            self._tooltip(caption, tip)
            self._tooltip(combo, tip)
            return combo

        def add_section(parent: ttk.Frame, row: int, title: str, first: bool = False) -> None:
            """A bold header, with a rule above it once a tab has more than one group."""
            top = 0 if first else 10
            if not first:
                ttk.Separator(parent, orient="horizontal").grid(
                    row=row, column=0, columnspan=2, sticky="ew", pady=(top, 6)
                )
            ttk.Label(parent, text=title, font=("TkDefaultFont", 10, "bold")).grid(
                row=row + 1, column=0, columnspan=2, sticky="w", pady=(0, 8)
            )

        # --- Detection ------------------------------------------------------
        add_section(detection_tab, 0, "What counts as a match", first=True)
        threshold_entry = add_labeled_entry(
            detection_tab,
            2,
            "Sensitivity threshold",
            threshold_var,
            width=12,
            tip="The sensitivity a scan starts at, matching the slider on the main window. "
            "Lower shows more photos and more false alarms; higher shows only the surest "
            "matches. 0.35 suits the current scoring.",
        )
        nsfw_combo = add_combo(
            detection_tab,
            3,
            "Explicit images",
            nsfw_mode_var,
            ("include", "exclude", "only"),
            "What to do with explicit images. include keeps them alongside everything else, "
            "exclude drops them from results, only shows nothing else.",
        )
        add_labeled_entry(
            detection_tab,
            4,
            "Explicit threshold",
            nsfw_threshold_var,
            width=12,
            tip="How sure the scanner must be before treating an image as explicit, for the "
            "setting above. Lower catches more but misjudges more.",
        )

        add_section(detection_tab, 5, "Who is in the picture")
        add_check(
            detection_tab,
            7,
            0,
            "Exclude images that may show a minor",
            exclude_minors_var,
            "Drops anything that reads as showing a child. Flagged images are forced to a zero "
            "score and hidden from every view, so no threshold or filter can bring them back. "
            "Deliberately errs toward excluding.",
        )
        add_inline_entry(
            detection_tab,
            7,
            "Sensitivity (lower = stricter)",
            minor_threshold_var,
            tip="How much child-like evidence triggers the age exclusion to the left. LOWER IS "
            "STRICTER: 0.30 excludes on modest evidence, 0.60 waits for strong evidence and "
            "therefore excludes less. Age estimates are rough, which is why the default sits "
            "low. To switch the gate off entirely, untick the box rather than raising this.",
        )
        add_check(
            detection_tab,
            8,
            0,
            "Prefer female subjects",
            require_female_var,
            "Ranks images with a female subject higher. On its own this only re-orders results; "
            "it discards nothing unless you also raise the cut-off to the right.",
        )
        add_inline_entry(
            detection_tab,
            8,
            "Cut-off (0 = rank only)",
            female_threshold_var,
            tip="Leave at 0 to only re-order results. Above 0 it becomes a hard filter that "
            "discards images scoring below it. Kept at 0 by default because on real photos a "
            "hard cut-off silently binned a genuine match whose close-up crop gave the model "
            "nothing to judge sex from.",
        )
        add_check(
            detection_tab,
            9,
            0,
            "Require a person",
            require_person_var,
            "Discard images where no person is detected. Off by default for good reason: "
            "close-up body shots often score very low on 'is this a person', so this filter "
            "throws away real matches. Turn it on only if landscapes are cluttering results.",
        )
        add_inline_entry(
            detection_tab,
            9,
            "Confidence needed",
            person_threshold_var,
            tip="How sure the scanner must be that a person is present, used only when "
            "'Require a person' is ticked to the left.",
        )

        add_section(detection_tab, 10, "How closely images are examined")
        add_combo(
            detection_tab,
            12,
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
            detection_tab,
            13,
            0,
            "Count faces in every image",
            enable_face_detection_var,
            "Count faces in every image during the scan. Needs the face model installed below. "
            "The deep pass already detects faces for candidate images, so this mainly adds a "
            "face count to the details line.",
        )
        face_row = ttk.Frame(detection_tab)
        face_row.grid(row=14, column=0, columnspan=2, sticky="ew", pady=(6, 0))
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

        # --- Model ----------------------------------------------------------
        add_section(model_tab, 0, "Scoring model", first=True)
        backend_combo = add_combo(
            model_tab,
            2,
            "Engine",
            backend_var,
            ("clip-torch", "clip-onnx"),
            "Which engine runs the model. clip-torch is the normal choice. clip-onnx is "
            "experimental and needs the optional onnxruntime package installed. Changing "
            "this requires a new scan.",
        )
        add_labeled_entry(
            model_tab,
            3,
            "Model name",
            model_var,
            tip="The Hugging Face model used for scoring. openai/clip-vit-base-patch32 is the "
            "default and is downloaded once, then cached. openai/clip-vit-large-patch14 is "
            "noticeably more accurate but a ~1.7 GB download and several times slower per "
            "image. Changing this invalidates cached embeddings and needs a fresh scan.",
        )
        refine_box = add_check(
            model_tab,
            4,
            0,
            "High-accuracy re-check of borderline images",
            refine_var,
            "Re-scores only the images sitting closest to your sensitivity setting using a much "
            "larger model, then blends its opinion in. Those borderline cases are where "
            "mistakes live, so the accuracy is worth it there. Costs a one-time ~1.7 GB "
            "download and adds minutes to a scan; the rest of the images are untouched.",
        )
        refine_box.grid(columnspan=2)
        ttk.Label(
            model_tab,
            text=f"Uses {HIGH_ACCURACY_MODEL.split('/')[-1]} — a one-off ~1.7 GB download.",
            style="FormMuted.TLabel",
        ).grid(row=5, column=0, columnspan=2, sticky="w", padx=(22, 0), pady=(0, 6))
        # The checkbox above decides whether the re-check runs; these decide what it
        # costs. Exposing one without the others left no way to bound a slow scan.
        # Rows 6-8, immediately under that checkbox: they used to sit at 13-15, which
        # put them below the Hardware separator and read as hardware settings.
        add_labeled_entry(
            model_tab,
            6,
            "Re-check band around the threshold",
            refine_band_var,
            width=12,
            tip="Only images scoring within this distance of your sensitivity setting are "
            "re-checked. Wider catches more borderline cases and costs more time.",
        )
        add_labeled_entry(
            model_tab,
            7,
            "Re-check at most",
            refine_max_images_var,
            width=12,
            tip="Hard cap on how many images the high-accuracy model re-scores in one scan.",
        )
        add_labeled_entry(
            model_tab,
            8,
            "Re-check influence",
            refine_weight_var,
            width=12,
            tip="How much the larger model's opinion counts when it disagrees with the first "
            "pass, from 0 (ignored) to 1 (it decides). 0.65 by default.",
        )

        add_section(model_tab, 9, "Hardware")
        add_combo(
            model_tab,
            11,
            "Device",
            device_var,
            ("auto", "cpu", "cuda"),
            "Where the model runs. auto picks your NVIDIA GPU when one is usable and falls "
            "back to the CPU otherwise. Force cpu if a GPU driver misbehaves.",
        )
        add_combo(
            model_tab,
            12,
            "Precision",
            precision_var,
            ("auto", "fp32", "fp16"),
            "Numeric precision. fp16 roughly halves GPU memory and speeds scans up, but only "
            "applies on CUDA; on the CPU everything runs fp32 regardless. auto chooses for you.",
        )
        add_labeled_entry(
            model_tab,
            13,
            "Batch size",
            batch_var,
            width=12,
            tip="How many images are fed to the model at once. Larger is faster but uses more "
            "memory. 16 suits most machines; drop to 4-8 if scanning runs out of memory.",
        )
        add_check(
            model_tab,
            14,
            0,
            "Quantize CPU (int8)",
            quantize_cpu_var,
            "Compresses the model to 8-bit for CPU scanning. Faster and lighter on memory, at "
            "some cost in accuracy. Worth trying on a slow machine with a large folder.",
        )
        add_check(
            model_tab,
            15,
            0,
            "Load the model when the app starts",
            preload_backend_var,
            "Loads the model in the background as soon as the app opens, so your first scan "
            "starts immediately. Turn off if you want the app to open using less memory and "
            "do not mind waiting at the first scan instead.",
        )

        # --- Prompts --------------------------------------------------------
        prompts_tab.columnconfigure(0, weight=1)
        prompts_tab.rowconfigure(2, weight=1)
        prompts_tab.rowconfigure(5, weight=1)
        prompt_caption = ttk.Label(prompts_tab, text="Positive prompts — what you want found")
        prompt_caption.grid(row=0, column=0, columnspan=2, sticky="w")
        ttk.Label(
            prompts_tab,
            text="Primary scoring uses the canonical Bikini axis defaults.",
            style="FormMuted.TLabel",
        ).grid(row=1, column=0, columnspan=2, sticky="w", pady=(0, 4))
        positive_text = Text(prompts_tab, width=54, height=7, wrap="word")
        positive_text.grid(row=2, column=0, columnspan=2, sticky="nsew", pady=(0, 10))
        positive_text.insert("1.0", "\n".join(self.config.positive_prompts))
        positive_tip = (
            "One phrase per line describing what you WANT found. Each is compared against "
            "the image and the best match wins, so several differently worded lines beat "
            "one clever one. These feed the headline bikini axis; the cleavage, midriff, "
            "age and sex axes have their own built-in wording."
        )
        self._tooltip(prompt_caption, positive_tip)
        self._tooltip(positive_text, positive_tip)

        negative_caption = ttk.Label(prompts_tab, text="Negative prompts — what you do not want")
        negative_caption.grid(row=4, column=0, columnspan=2, sticky="w", pady=(0, 4))
        negative_text = Text(prompts_tab, width=54, height=7, wrap="word")
        negative_text.grid(row=5, column=0, columnspan=2, sticky="nsew")
        negative_text.insert("1.0", "\n".join(self.config.negative_prompts))
        negative_tip = (
            "One phrase per line for what you do NOT want. A score is positive evidence "
            "minus negative evidence, so these matter as much as the positives: list the "
            "things the scanner keeps confusing for a match, like 'a fully clothed person'."
        )
        self._tooltip(negative_caption, negative_tip)
        self._tooltip(negative_text, negative_tip)

        # --- Advanced -------------------------------------------------------
        add_section(advanced_tab, 0, "Learning", first=True)
        add_check(
            advanced_tab,
            2,
            0,
            "Learn across all folders",
            global_learning_var,
            "Pool your Accept/REJECT decisions so every folder benefits from all of them. Off "
            "means each folder learns from scratch. Each folder's own labels are saved either "
            "way; Tools > Reset cross-folder learning clears the shared pool.",
        )
        add_labeled_entry(
            advanced_tab,
            3,
            "Zero-shot scale",
            scale_var,
            width=12,
            tip="How sharply prompt scores separate. The model's raw agreement differences are "
            "tiny, so this multiplies them: too low and everything looks like a 50/50 guess, "
            "too high and every score slams to 0 or 1. 40 was chosen by measurement — leave it "
            "unless you are deliberately recalibrating.",
        )
        # Classifier weight and Zero-shot weight used to sit here. They only do
        # anything when pipeline="legacy", which nothing in the UI can select, so in
        # every configuration a user can reach they were two numbers that looked
        # adjustable and changed nothing. They remain in ScannerConfig for the legacy
        # pipeline and the CLI; they are simply no longer offered as settings.

        add_section(advanced_tab, 6, "Vision-LLM second opinion")
        add_check(
            advanced_tab,
            8,
            0,
            "Ask a local vision-LLM about borderline images",
            vlm_enabled_var,
            "Optional second opinion from a local Ollama or llama.cpp server. It only checks "
            "borderline images and uncertain age calls, in parallel, so the usual CLIP scan "
            "stays fast. The server must already be running.",
        )
        add_labeled_entry(
            advanced_tab,
            9,
            "Server URL",
            vlm_base_url_var,
            width=32,
            tip="OpenAI-compatible local endpoint, for example http://localhost:11434/v1. "
            "The stage is skipped if it cannot reach this address.",
        )
        add_labeled_entry(
            advanced_tab,
            10,
            "Model",
            vlm_model_var,
            width=32,
            tip="Model name served by Ollama or llama.cpp, for example qwen2.5vl:7b.",
        )
        add_labeled_entry(
            advanced_tab,
            11,
            "Requests at once",
            vlm_concurrency_var,
            width=12,
            tip="How many local requests run at once. Higher is faster only when your server "
            "has enough CPU/GPU memory; 4 is a sensible starting point.",
        )
        add_labeled_entry(
            advanced_tab,
            12,
            "Borderline band",
            vlm_band_var,
            width=12,
            tip="Only scores within this distance of the threshold are sent to the VLM, plus "
            "uncertain age calls. Wider is more accurate but costs more requests.",
        )
        add_labeled_entry(
            advanced_tab,
            13,
            "Maximum images per scan",
            vlm_max_images_var,
            width=12,
            tip="Hard cap on VLM requests per scan. When more images qualify than this, the "
            "ones closest to your sensitivity setting are sent first — those are where the "
            "second opinion is worth the most.",
        )
        # The server URL was configurable while the credential was not, so any endpoint
        # that needs authentication could be pointed at but never actually reached.
        add_labeled_entry(
            advanced_tab,
            22,
            "API key (optional)",
            vlm_api_key_var,
            width=32,
            tip="Sent as a bearer token to the endpoint above. Local Ollama and llama.cpp "
            "servers do not need one; leave it empty for those.",
        )
        add_labeled_entry(
            advanced_tab,
            23,
            "Request timeout (seconds)",
            vlm_timeout_var,
            width=12,
            tip="How long to wait for one adjudication before giving up on that image.",
        )
        add_labeled_entry(
            advanced_tab,
            24,
            "Influence",
            vlm_weight_var,
            width=12,
            tip="How much the vision-LLM's verdict counts against the first pass, from 0 "
            "(ignored) to 1 (it decides).",
        )
        add_section(advanced_tab, 25, "Scoring pipeline")
        add_combo(
            advanced_tab,
            27,
            "Pipeline",
            pipeline_var,
            ("cascade", "legacy"),
            "cascade is the current pipeline: staged gates, body-region crops, and a learned "
            "model whose influence is earned from its measured accuracy. legacy is the "
            "original single-pass blend, kept so old profiles still resolve. Legacy honours "
            "the two blend weights below; cascade decides the blend for itself.\n\n"
            "The age gate runs in both.",
        )
        add_labeled_entry(
            advanced_tab,
            28,
            "Classifier weight (legacy only)",
            classifier_weight_var,
            width=12,
            tip="Only used when the pipeline above is set to legacy.",
        )
        add_labeled_entry(
            advanced_tab,
            29,
            "Zero-shot weight (legacy only)",
            zero_shot_weight_var,
            width=12,
            tip="The other half of the legacy blend. Only used when the pipeline is legacy.",
        )

        add_section(advanced_tab, 30, "Extensions")
        plugins_box = add_check(
            advanced_tab,
            32,
            0,
            "Run result plugins from the plugins folder",
            enable_plugins_var,
            "Executes every .py file in your plugins folder after each scan, with full "
            "access to this machine. Only turn this on if you wrote the files yourself or "
            "trust whoever did. A plugin defines process_results(state, samples) and returns "
            "a reordered or filtered list.",
        )
        plugins_box.grid(columnspan=2)
        plugins_row = ttk.Frame(advanced_tab)
        plugins_row.grid(row=33, column=0, columnspan=2, sticky="ew", pady=(0, 6))
        ttk.Label(
            plugins_row,
            text="Code in this folder runs with your account's permissions.",
            style="FormMuted.TLabel",
        ).pack(side=LEFT, padx=(22, 0))
        ttk.Button(plugins_row, text="Open plugins folder", command=self.open_plugins_folder).pack(side=RIGHT)

        add_section(advanced_tab, 34, "Application")
        add_labeled_entry(
            advanced_tab,
            36,
            "Thumbnail cache entries",
            thumbnail_cache_var,
            width=12,
            tip="How many scaled thumbnails to keep in memory (32-2048). Higher makes scrolling "
            "and resizing smoother at the cost of RAM. Lower it if the app feels heavy.",
        )
        url_caption = ttk.Label(advanced_tab, text="Update manifest URL (optional)")
        url_caption.grid(row=37, column=0, sticky="w", pady=(0, 6), padx=(0, 12))
        update_url_entry = ttk.Entry(advanced_tab, textvariable=self.update_url_var, width=32)
        update_url_entry.grid(row=37, column=1, sticky="ew", pady=(0, 6))
        url_tip = (
            "Optional address of a JSON file listing the newest version, used by Help > Check "
            "for updates. Leave empty and the app never contacts anything for updates."
        )
        self._tooltip(url_caption, url_tip)
        self._tooltip(update_url_entry, url_tip)

        button_row = ttk.Frame(outer)
        button_row.pack(side=TOP, fill="x", pady=(12, 0))

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
                batch_size = int(batch_var.get().strip())
            except Exception:  # noqa: BLE001
                messagebox.showerror("Invalid settings", "Batch size must be a positive integer.", parent=dialog)
                return
            if batch_size < 1:
                messagebox.showerror("Invalid settings", "Batch size must be a positive integer.", parent=dialog)
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
                zero_shot_scale = float(scale_var.get().strip())
                classifier_weight = float(classifier_weight_var.get().strip())
                zero_shot_weight = float(zero_shot_weight_var.get().strip())
                threshold = float(threshold_var.get().strip())
                nsfw_threshold = float(nsfw_threshold_var.get().strip())
                person_threshold = float(person_threshold_var.get().strip())
                vlm_concurrency = int(vlm_concurrency_var.get().strip())
                vlm_band = float(vlm_band_var.get().strip())
                vlm_max_images = int(vlm_max_images_var.get().strip())
            except Exception:  # noqa: BLE001
                messagebox.showerror(
                    "Invalid settings",
                    "Scale, weights, thresholds, and person/NSFW values must be numeric.",
                    parent=dialog,
                )
                return
            if not 1 <= vlm_concurrency <= 64:
                messagebox.showerror("Invalid settings", "VLM concurrency must be from 1 to 64.", parent=dialog)
                return
            if not 0.0 <= vlm_band <= 1.0:
                messagebox.showerror("Invalid settings", "VLM band must be between 0 and 1.", parent=dialog)
                return
            if not 0 <= vlm_max_images <= 1_000_000:
                messagebox.showerror(
                    "Invalid settings",
                    "VLM maximum images must be from 0 to 1,000,000.",
                    parent=dialog,
                )
                return
            try:
                vlm_timeout = float(vlm_timeout_var.get().strip())
                vlm_weight = float(vlm_weight_var.get().strip())
                refine_band = float(refine_band_var.get().strip())
                refine_max_images = int(refine_max_images_var.get().strip())
                refine_weight = float(refine_weight_var.get().strip())
            except Exception:  # noqa: BLE001
                messagebox.showerror(
                    "Invalid settings", "The re-check and VLM timeout/influence values must be numeric.", parent=dialog
                )
                return
            if vlm_timeout <= 0:
                messagebox.showerror("Invalid settings", "VLM timeout must be above zero.", parent=dialog)
                return
            for name, value in (("VLM influence", vlm_weight), ("re-check influence", refine_weight),
                                ("re-check band", refine_band)):
                if not 0.0 <= value <= 1.0:
                    messagebox.showerror("Invalid settings", f"The {name} must be between 0 and 1.", parent=dialog)
                    return
            if refine_max_images < 0:
                messagebox.showerror("Invalid settings", "Re-check at most must be zero or more.", parent=dialog)
                return
            pipeline = pipeline_var.get().strip()
            if pipeline not in {"cascade", "legacy"}:
                messagebox.showerror("Invalid settings", "Pipeline must be cascade or legacy.", parent=dialog)
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
            if zero_shot_scale <= 0:
                messagebox.showerror("Invalid settings", "Zero-shot scale must be positive.", parent=dialog)
                return
            if classifier_weight < 0 or zero_shot_weight < 0:
                messagebox.showerror("Invalid settings", "Blend weights must be non-negative.", parent=dialog)
                return
            if not (0.0 <= threshold <= 1.0):
                messagebox.showerror("Invalid settings", "Threshold must be between 0 and 1.", parent=dialog)
                return
            if not (0.0 <= nsfw_threshold <= 1.0):
                messagebox.showerror("Invalid settings", "NSFW threshold must be between 0 and 1.", parent=dialog)
                return
            if not (0.0 <= person_threshold <= 1.0):
                messagebox.showerror("Invalid settings", "Person threshold must be between 0 and 1.", parent=dialog)
                return
            try:
                minor_threshold = float(minor_threshold_var.get().strip())
                female_threshold = float(female_threshold_var.get().strip())
            except Exception:  # noqa: BLE001
                messagebox.showerror("Invalid settings", "Minor and female cut-offs must be numeric.", parent=dialog)
                return
            if not (0.0 < minor_threshold <= 1.0):
                messagebox.showerror(
                    "Invalid settings",
                    "Minor sensitivity must be above 0 and at most 1. Lower is stricter; to turn the age gate "
                    "off entirely, clear the 'Exclude images that may show a minor' checkbox.",
                    parent=dialog,
                )
                return
            if not (0.0 <= female_threshold <= 1.0):
                messagebox.showerror("Invalid settings", "Female cut-off must be between 0 and 1.", parent=dialog)
                return
            deep_scan = deep_scan_var.get().strip()
            if deep_scan not in {"candidates", "always", "off"}:
                messagebox.showerror("Invalid settings", "Deep scan must be candidates, always, or off.", parent=dialog)
                return
            nsfw_mode = nsfw_mode_var.get().strip()
            if nsfw_mode not in {"include", "exclude", "only"}:
                messagebox.showerror("Invalid settings", "NSFW mode must be include, exclude, or only.", parent=dialog)
                return
            device = device_var.get().strip()
            if device not in {"auto", "cpu", "cuda"}:
                messagebox.showerror("Invalid settings", "Device must be auto, cpu, or cuda.", parent=dialog)
                return
            precision = precision_var.get().strip()
            if precision not in {"auto", "fp32", "fp16"}:
                messagebox.showerror("Invalid settings", "Precision must be auto, fp32, or fp16.", parent=dialog)
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
                or bool(enable_plugins_var.get()) != self.config.enable_plugins
            )
            # A deep-scan or refine change needs a fresh scan, not just a re-filter:
            # both change what gets embedded.
            rescan_needed = (
                pipeline != self.config.pipeline
                or refine_band != self.config.refine_band
                or refine_max_images != self.config.refine_max_images
                or deep_scan != self.config.deep_scan
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
            self.config.enable_plugins = bool(enable_plugins_var.get())
            self.config.refine_model = HIGH_ACCURACY_MODEL if bool(refine_var.get()) else ""
            self.config.vlm_enabled = bool(vlm_enabled_var.get())
            self.config.vlm_base_url = vlm_base_url
            self.config.vlm_model = vlm_model
            self.config.vlm_concurrency = vlm_concurrency
            self.config.vlm_band = vlm_band
            self.config.vlm_max_images = vlm_max_images
            self.config.vlm_api_key = vlm_api_key_var.get().strip()
            self.config.vlm_timeout = vlm_timeout
            self.config.vlm_weight = vlm_weight
            self.config.refine_band = refine_band
            self.config.refine_max_images = refine_max_images
            self.config.refine_weight = refine_weight
            self.config.pipeline = pipeline
            self.thumbnail_cache_size_var.set(thumbnail_cache_size)
            self._trim_thumbnail_cache()
            if not self.folder_override_active:
                self.global_config = ScannerConfig.from_mapping(self.config.to_dict())
            self._set_threshold(threshold)
            self.nsfw_only_var.set(self.config.nsfw_filter == "only")
            self._refresh_vlm_badge()
            self._refresh_summary()
            # Persist immediately rather than at shutdown: a crash between the two used
            # to lose everything that was just configured.
            self._save_user_prefs()

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

            if self.folder_override_active:
                # global_config was deliberately not updated above, so these values live
                # only as long as this folder is open. Saying so beats letting them
                # revert without explanation the next time a folder is chosen.
                self.status_var.set(
                    self.status_var.get()
                    + " This folder has an override, so these apply to it only — use "
                    "Save folder override to keep them."
                )
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
        for text_widget in (positive_text, negative_text):
            text_widget.configure(
                bg=palette["entry_bg"],
                fg=palette["fg"],
                insertbackground=palette["fg"],
                highlightbackground=palette["panel"],
                relief="solid",
            )
        for widget in (nsfw_combo, backend_combo):
            widget.configure(width=22)

        # Open at the size the widest tab actually needs, plus slack, and never past
        # the screen. The window stays resizable in both directions from there, which
        # is what the fixed-width version got wrong: a long label had nowhere to go.
        dialog.update_idletasks()
        wanted_width = max(600, dialog.winfo_reqwidth() + 32)
        wanted_height = max(460, dialog.winfo_reqheight() + 12)
        width = min(wanted_width, int(dialog.winfo_screenwidth() * 0.9))
        height = min(wanted_height, int(dialog.winfo_screenheight() * 0.85))
        dialog.geometry(f"{width}x{height}")
        dialog.minsize(min(600, width), min(420, height))
        threshold_entry.focus_set()

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

        # Say what the template actually produces. The token list told you the parts
        # but not the result, so the only way to find out was to run a transfer.
        preview_var = StringVar(value="")
        ttk.Label(form, textvariable=preview_var, style="FormMuted.TLabel", justify="left").grid(
            row=6, column=0, columnspan=2, sticky="w", pady=(0, 8)
        )

        def refresh_preview(*_args: object) -> None:
            sample = next(iter(self.displayed_samples or self.current_samples), None)
            source = Path(str(sample["path"])) if sample else Path("beach_photo.jpg")
            score = float(cast(float, sample.get("score", 0.0))) if sample else 0.62
            try:
                low = float(low_var.get())
                high = float(high_var.get())
            except (TypeError, ValueError):
                low, high = 0.35, 0.7
            try:
                name = format_output_name(
                    source,
                    score,
                    label_name(self._label_map().get(str(source))),
                    1,
                    None,
                    template_var.get(),
                )
                parts = organization_parts(
                    organization_var.get(), score, label_name(self._label_map().get(str(source))), low, high
                )
            except Exception as exc:  # noqa: BLE001
                preview_var.set(f"Example: template could not be applied — {exc}")
                return
            preview_var.set("Example: " + "/".join([*parts, name]))

        for variable in (template_var, organization_var, low_var, high_var):
            try:
                variable.trace_add("write", refresh_preview)
            except Exception:  # noqa: BLE001
                continue
        refresh_preview()

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
            # The VLM API key is left out: an exported settings file is meant to be
            # copied to another machine or sent to someone, and a bearer token has no
            # business travelling with it.
            payload = without_secrets(self.config.to_dict())
            Path(target).write_text(json.dumps(payload, indent=2, sort_keys=True), encoding="utf-8")
            self.status_var.set(f"Exported settings to {target}")
            if self.config.vlm_api_key:
                self.status_var.set(
                    f"Exported settings to {target} — the VLM API key was left out and must be re-entered."
                )
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
        # Settings files no longer carry the API key, so an imported one always has it
        # empty. Blanking a key the user has already entered would be a silent loss, so
        # the session's own key is kept unless the file explicitly supplies one.
        if not imported.vlm_api_key and self.config.vlm_api_key:
            imported.vlm_api_key = self.config.vlm_api_key
        self.config = imported
        self.global_config = ScannerConfig.from_mapping(imported.to_dict())
        self.folder_override_active = False
        self.override_var.set("")
        self.backend = None
        self.scorer = None
        self._sync_config_controls()
        self.status_var.set("Settings imported. Run a scan to apply them.")

    def _set_threshold(self, value: float) -> None:
        """Move the slider and the number beside it together.

        Tk only invokes a Scale's command for user interaction, so a programmatic
        `threshold_var.set()` left the entry showing the previous value — which is what
        happened after saving Settings, importing a file or applying a profile.
        """
        value = max(0.0, min(1.0, float(value)))
        self.threshold_var.set(value)
        self.threshold_text_var.set(f"{value:.2f}")

    def _sync_config_controls(self) -> None:
        """Point the on-screen controls at the config that is actually in force.

        The sensitivity slider is the value a scan is run with, so a config loaded from
        a file or a profile has to move it — otherwise an imported threshold of 0.7 is
        quietly overridden by whatever the slider was left on.
        """
        self._set_threshold(float(self.config.threshold))
        self.nsfw_only_var.set(self.config.nsfw_filter == "only")
        self._refresh_vlm_badge()
        self._refresh_summary()
        self._save_user_prefs()

    def open_profiles_dialog(self) -> None:
        dialog, outer = self._create_modal("Settings profiles")
        profile_var = StringVar(value=profile_names()[0] if profile_names() else "")
        combo = ttk.Combobox(outer, textvariable=profile_var, values=profile_names(), state="readonly", width=30)
        combo.pack(side=TOP, anchor="w", pady=(0, 10))
        ttk.Label(
            outer,
            text=(
                "A profile captures every setting, not just the threshold: detection gates, deep-scan\n"
                "mode, model choice and prompts. Applying one replaces all of them.\n\n"
                "Built-in profiles cannot be deleted. Apply one, adjust it, then Save as to keep your own."
            ),
            justify="left",
        ).pack(side=TOP, anchor="w", pady=(0, 10))
        summary = ttk.Label(outer, text="", style="FormMuted.TLabel", justify="left", wraplength=420)
        summary.pack(side=TOP, anchor="w", pady=(0, 10))

        def describe(*_args: object) -> None:
            selected = profile_config(profile_var.get())
            if selected is None:
                summary.configure(text="")
                return
            defaults = ScannerConfig().to_dict()
            changed = [
                f"{key} = {value}"
                for key, value in sorted(selected.to_dict().items())
                if defaults.get(key) != value and not isinstance(value, list)
            ]
            summary.configure(
                text=("Differs from the defaults in: " + ", ".join(changed)) if changed else "Matches the defaults."
            )

        combo.bind("<<ComboboxSelected>>", describe)
        describe()
        buttons = self._modal_button_row(outer, pady=(0, 0))

        def refresh() -> None:
            combo.configure(values=profile_names())

        def apply() -> None:
            selected = profile_config(profile_var.get())
            if selected is None:
                return
            # Profiles do not store the API key, so applying one must not wipe the one
            # the user entered this session.
            if not selected.vlm_api_key and self.config.vlm_api_key:
                selected.vlm_api_key = self.config.vlm_api_key
            self.config = selected
            self.global_config = ScannerConfig.from_mapping(selected.to_dict())
            self.backend = None
            self.scorer = None
            self._sync_config_controls()
            self.status_var.set(f"Applied profile {profile_var.get()}. Run a scan to apply it.")

        def save_as() -> None:
            # A profile is a name, not a file on disk. asksaveasfilename made people
            # pick a folder and an extension for something stored in the prefs file,
            # then quietly threw both away.
            name = simpledialog.askstring(
                "Save profile",
                "Name this profile:",
                parent=dialog,
                initialvalue=profile_var.get() if profile_var.get() not in BUILTIN_PROFILES else "",
            )
            if not name:
                return
            if name in profile_names() and not messagebox.askyesno(
                "Replace profile", f"A profile called {name} already exists. Replace it?", parent=dialog
            ):
                return
            try:
                save_profile(name.strip(), self.config)
                refresh()
                profile_var.set(name.strip())
                self.status_var.set(f"Saved the current settings as profile {name.strip()}.")
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
        # run_scan re-enters itself once for the model load and once for the image
        # count, so the order of these three steps decides how much work is repeated.
        #
        # Binding the folder comes first because _set_folder clears the backend — a
        # folder override can name a different model. Doing it after the model check
        # meant the model was loaded, discarded and loaded again; doing it after the
        # count meant the count was discarded on the way to the second load and the
        # whole folder tree was walked twice.
        #
        # The folder box is editable and callers may set it directly, so the store can
        # lag behind the path shown. Bind it here rather than assert later.
        if self.store is None or str(self.store.folder) != str(Path(folder).expanduser().resolve()):
            self._set_folder(folder)
        # Loading the model can mean a ~600 MB download. Doing that on the main thread
        # froze the whole window with no progress and no way out — which is exactly what
        # a first run looked like. Hand it to a worker and come back when it lands.
        if self.backend is None:
            self._load_backend_then_scan(folder)
            return
        # Pre-scan check: count supported images so we don't run a full scan on an
        # empty folder or one with only unsupported files (e.g. all .txt or .pdf).
        # Counting walks the whole tree, which on a network share or a deep folder is
        # seconds of frozen window, so it happens once, off the main thread.
        if self._pending_scan_count is None:
            self._count_images_then_scan(folder)
            return
        image_count = self._pending_scan_count
        self._pending_scan_count = None
        if image_count == 0:
            messagebox.showinfo(
                "No images found",
                f"No supported image files were found in this folder:\n{folder}\n\n"
                "Supported formats: JPG, PNG, BMP, GIF, WEBP, TIFF, HEIC.",
            )
            return
        if image_count >= LARGE_SCAN_WARNING and not self._confirm_large_scan(folder, image_count):
            return
        self._add_recent_folder(folder)
        LOGGER.info("Starting scan requested for %s", folder)
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

    def _count_images_then_scan(self, folder: str) -> None:
        """Walk the folder on a worker thread, then re-enter run_scan with the count."""
        self.run_button.configure(state="disabled")
        self.status_var.set("Looking through the folder…")
        target = Path(folder)

        def worker() -> None:
            try:
                # The same walk the scan itself does, so the two agree. A plain rglob
                # counted the app's own output back in: after "Copy matches to
                # subfolder", every copy under bikini_matches/ inflated the count that
                # drives the large-folder warning and its time estimate, and a folder
                # holding nothing but those copies looked like it had images to scan.
                count = len(collect_image_paths(target))
            except OSError:
                count = 0
            self._after(0, self._image_count_ready, count)

        threading.Thread(target=worker, name="scan-precount", daemon=True).start()

    def _image_count_ready(self, count: int) -> None:
        self.run_button.configure(state="normal")
        self._pending_scan_count = count
        self.run_scan()

    def _scan_rate_hint(self) -> float | None:
        """Images per second from the last completed scan, if one is on record."""
        rate = self.user_prefs.get("scan_images_per_second")
        if rate is None:
            return None
        try:
            value = float(rate)  # type: ignore[arg-type]
        except (TypeError, ValueError):
            return None
        return value if value > 0 else None

    def _confirm_large_scan(self, folder: str, image_count: int) -> bool:
        """Say how big this is before committing to it.

        The count is already in hand; spending it on a warning costs nothing and a
        folder of a quarter of a million photos is hours of work to start by accident.
        """
        rate = self._scan_rate_hint()
        estimate = ""
        if rate:
            seconds = image_count / rate
            if seconds >= 5400:
                estimate = f"\n\nAt your last scan's speed this would take roughly {seconds / 3600:.1f} hours."
            elif seconds >= 90:
                estimate = f"\n\nAt your last scan's speed this would take roughly {seconds / 60:.0f} minutes."
        return messagebox.askyesno(
            "Large folder",
            f"{folder}\n\nThis folder holds {image_count:,} images.{estimate}\n\n"
            "Scanning can be stopped at any point and the work done so far is cached, "
            "so a stopped scan is not wasted.\n\nStart the scan?",
        )

    def _load_backend_then_scan(self, folder: str) -> None:
        """Load the model in the background, then start the scan it was needed for."""
        if self._backend_loading:
            self.status_var.set("Still loading the scanning model — the scan starts when it is ready.")
            return
        self._backend_loading = True
        self._scan_after_backend = folder
        self.run_button.configure(state="disabled")
        # Stop is live during the load: the wait is long enough that being unable to
        # change your mind was its own bug.
        self.stop_scan_button.configure(state="normal")
        self.loading_var.set("Loading model...")
        self._show_progress(True)
        self._start_indeterminate_progress(
            "Loading the scanning model — the first run downloads it, which can take a few minutes…"
        )

        def worker() -> None:
            loaded = self._ensure_backend(show_errors=False)
            error: Exception | None = None
            if not loaded:
                try:
                    from .clip_backend import get_backend

                    get_backend(self.config)
                except Exception as exc:  # noqa: BLE001
                    error = exc
            self._after(0, self._backend_load_finished, loaded, error)

        threading.Thread(target=worker, name="backend-load", daemon=True).start()

    def _backend_load_finished(self, loaded: bool, error: Exception | None) -> None:
        self._backend_loading = False
        self.loading_var.set("")
        self.run_button.configure(state="normal")
        self.stop_scan_button.configure(state="disabled")
        self._reset_progress_bar()
        self._show_progress(False)
        pending = self._scan_after_backend
        self._scan_after_backend = None
        if not loaded:
            self.status_var.set("The scanning model could not be loaded.")
            if error is not None:
                messagebox.showerror("Could not load the scanning model", self._model_error_text(error))
            return
        self._refresh_hardware_status()
        if pending is None:
            # Cancelled while it loaded. The model is in memory for next time.
            self.status_var.set("Model loaded. Press Run scan when you are ready.")
            return
        self.run_scan()

    def cancel_scan(self) -> None:
        if self._backend_loading:
            # The load itself cannot be interrupted — it is inside the model library —
            # but the scan waiting behind it can be called off, which is what the
            # reviewer actually wants when they press Stop during a long download.
            self._scan_after_backend = None
            self.status_var.set("Scan cancelled. The model is still loading and will finish in the background.")
            self.stop_scan_button.configure(state="disabled")
            return
        if not self._scan_active or self._scan_cancel_event is None:
            return
        self._scan_cancel_event.set()
        self.status_var.set("Stopping scan after the current batch...")
        self.stop_scan_button.configure(state="disabled")

    def _schedule_retrain(self, labels_added: int = 1) -> None:
        """Fold new labels into the model once the reviewer pauses, not per click.

        Called instead of `update_algorithm` after every decision. The decision is
        already saved; this only queues the re-rank, and re-arming the timer on each
        new label means a run of twenty Accepts costs one retrain rather than twenty.

        Browse mode has no model and no embeddings on purpose, so there is nothing to
        re-rank there. Queueing one anyway loaded the model on the main thread — a
        ~600 MB download on a cold install — and then failed on the zero-width
        embeddings, turning the first Accept in a model-free browse into a "Scan
        failed" dialog. The decision is still written to labels.json either way, which
        is the whole point of browsing without scanning.
        """
        if self.view_mode == "browse" or self.current_state is None or self.current_state.embeddings.size == 0:
            return
        self._retrain_pending = True
        self._labels_since_retrain += max(1, int(labels_added))
        if self._labels_since_retrain >= RETRAIN_LABEL_BURST:
            self._flush_retrain()
            return
        self._cancel_retrain_timer()
        self._retrain_after_id = self._after(RETRAIN_IDLE_MS, self._flush_retrain)

    def _cancel_retrain_timer(self) -> None:
        if self._retrain_after_id is None:
            return
        try:
            self.root.after_cancel(self._retrain_after_id)
        except Exception:  # noqa: BLE001
            pass
        self._retrain_after_id = None

    def _flush_retrain(self) -> None:
        """Run the queued retrain now, if one is queued and nothing else is running."""
        self._retrain_after_id = None
        if not self._retrain_pending:
            return
        if self._scan_active or self.queue_active:
            # Leave the flag set: whatever is running now finishes by checking it.
            return
        self.update_algorithm()

    def update_algorithm(self) -> None:
        if self.store is None:
            messagebox.showinfo("Not ready", "Run a scan first.")
            return
        if self.current_state is None:
            messagebox.showinfo("Not ready", "Run a scan first.")
            return
        if self.view_mode == "browse" or self.current_state.embeddings.size == 0:
            # Browsing without scanning produces a state with no embeddings, so there is
            # nothing for a re-rank to work from. The decisions are already saved.
            detail = (
                "This folder was opened with File > Browse folder without scanning, so no "
                "image has been scored yet."
                if self.view_mode == "browse"
                else "There are no scored images in this folder."
            )
            messagebox.showinfo(
                "Nothing to re-rank",
                f"{detail}\n\nYour Accept/REJECT decisions are saved and will be used the "
                "next time you run a scan here.",
            )
            return
        # Asked for explicitly (Tools > Update rankings) or by the timer: either way
        # the queued labels are being folded in now, so stand the timer down.
        self._cancel_retrain_timer()
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
        # Cleared here rather than in the caller: every route into a running retrain
        # goes through this line, so the counters cannot be left claiming that labels
        # are still waiting when they are being folded in right now.
        self._retrain_pending = False
        self._labels_since_retrain = 0
        self.status_var.set("Updating algorithm...")
        self._reset_progress_bar()
        self._launch_background_scan(full_rescan=False)

    def _run_pending_retrain(self) -> None:
        """Start the retrain that was asked for while another pass was running."""
        if not self._retrain_pending or self._scan_active or self.queue_active:
            return
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
        self._cancel_retrain_timer()
        self._retrain_pending = False
        self._labels_since_retrain = 0
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
        self._cancel_retrain_timer()
        self._retrain_pending = False
        self._labels_since_retrain = 0
        self.status_var.set("Scan failed.")
        LOGGER.error("Scan failed: %s", exc, exc_info=(type(exc), exc, exc.__traceback__))
        # A failed folder used to leave queue_active set with nothing left to advance it.
        # That stalled the queue silently, blocked run_queue from restarting it, and —
        # because _flush_retrain returns early while a queue is active — stopped every
        # retrain for the rest of the session while labels piled up.
        queue_note = ""
        if self.queue_active:
            self.queue_index += 1
            if self.queue_index < len(self.scan_queue):
                remaining = len(self.scan_queue) - self.queue_index
                queue_note = f"\n\nSkipping this folder; {remaining} left in the queue."
                self._after(0, self._start_next_queue_item)
            else:
                self.queue_active = False
                queue_note = "\n\nThat was the last folder in the queue."
        # Map common failures to user-friendly text with a suggested action.
        friendly = self._friendly_error(exc)
        messagebox.showerror("Scan failed", f"{friendly}{queue_note}")

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
        previous_page = self.page_index
        # Kept so a retrain can report what the new labels actually changed. "It moved
        # 14 photos across the threshold" is the only honest proof that Accept/REJECT
        # reached the model; a bare "Scan complete" is not.
        previous_scores = (
            dict(zip(self.current_state.paths, self.current_state.scores, strict=False))
            if self.current_state is not None
            else {}
        )
        self.current_state = state
        processed_samples = self._post_process_review_samples(list(samples))
        self.review_samples = list(processed_samples)
        self.current_samples = list(processed_samples)
        self.view_mode = "review"
        self.similar_anchor_path = None
        if self._resuming_review:
            # Only Resume last scan wants the recorded view, page and active photo back.
            # Reading the snapshot on every completion also restored focused_path, so an
            # ordinary scan landed on whatever photo a previous session had left active.
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
                filtered_note += (
                    f"\n{skipped_count} file{'s' if skipped_count != 1 else ''} could not be read "
                    "(Tools > Files that could not be read lists them)."
                )
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
            self.status_var.set(self._retrain_report(state, previous_scores, threshold))
        # Labelling from page 7 of a long list should not throw the reviewer back to
        # page 1 every time the re-rank lands, so a retrain keeps the page it was on.
        keep_page = not full_rescan
        if keep_page:
            self.page_index = previous_page
        if self._resuming_review:
            # Resuming: honour the view and page the session recorded, not the default.
            self._resuming_review = False
            wanted_view, wanted_page = self._resume_target
            self.page_index = max(0, wanted_page)
            if wanted_view == "detected" and matches:
                self.show_detected_files(reset_page=False)
            else:
                self._refresh_displayed_results(reset_page=False)
            self._update_stats_panel(record_history=True)
            self._save_last_folder(self.folder_var.get().strip())
            self._watch_snapshot = self._collect_watch_snapshot()
            self._refresh_hardware_status()
            self.status_var.set(
                f"Resumed where you left off — page {self.page_index + 1}. {self._progress_note()}".strip()
            )
            return
        # A fresh scan lands on the full detected-files list; retrains keep whichever view was active.
        if matches and (full_rescan or previous_view == "detected"):
            self.show_detected_files(reset_page=not keep_page)
        else:
            self._refresh_displayed_results(reset_page=not keep_page)
        self._update_stats_panel(record_history=True)
        self._save_review_session()
        self._save_last_folder(self.folder_var.get().strip())
        # Must be built the same way the watch poll builds it. Deriving the baseline
        # from state.paths instead left out every file the scan skipped as unreadable,
        # so a folder with one corrupt image looked "changed" on every single poll and
        # rescanned itself forever.
        self._watch_snapshot = self._collect_watch_snapshot()
        if full_rescan:
            self._record_folder_history(len(state.paths), matches)
        if full_rescan and self._scan_start_monotonic is not None and state.paths:
            elapsed = max(time.monotonic() - self._scan_start_monotonic, 1e-6)
            self.user_prefs["scan_images_per_second"] = round(len(state.paths) / elapsed, 3)
            self._save_user_prefs()
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
        # This folder's own labels, not the pooled cross-folder total: the count beside
        # "images" has to match the Accepted/Rejected tally underneath it, or a reviewer
        # checking whether their decisions registered sees two different answers.
        labels = self.store.load_labels() if self.store is not None else {}
        # Short enough to share a row with the slider. The label count and whether the
        # classifier is running are both spelled out on the stats line at the bottom,
        # so repeating them here only cost the slider width.
        self.summary_var.set(f"{len(self.current_state.paths)} scanned · {matches} above threshold")
        self._update_stats_panel(record_history=False, labels=labels)

    def _commit_threshold_text(self, _event: object = None) -> str:
        """Apply a typed sensitivity, or put the real one back if it was nonsense."""
        raw = self.threshold_text_var.get().strip()
        try:
            value = float(raw)
        except ValueError:
            self.threshold_text_var.set(f"{float(self.threshold_var.get()):.2f}")
            return "break"
        value = max(0.0, min(1.0, value))
        if abs(value - float(self.threshold_var.get())) > 1e-9:
            self.threshold_var.set(value)
            self._on_threshold_change(str(value))
        self.threshold_text_var.set(f"{value:.2f}")
        return "break"

    def _on_threshold_change(self, _value: str) -> None:
        self.threshold_text_var.set(f"{float(self.threshold_var.get()):.2f}")
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
        # Dragging the slider past every match must not raise a modal; just empty the
        # grid and let the empty-state panel explain itself.
        self._apply_detected_view(self._detected_samples(), reset_page=True)

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
        self.root.bind_all("<KeyPress-w>", self._handle_explain_shortcut)
        self.root.bind_all("<KeyPress-n>", self._handle_note_shortcut)
        self.root.bind_all("<BackSpace>", self._handle_back_shortcut)
        self.root.bind_all("<KeyPress-f>", self._handle_focus_shortcut)
        self.root.bind_all("<Control-z>", self._handle_undo_shortcut)
        self.root.bind_all("<Control-y>", self._handle_redo_shortcut)

    # Widgets that consume a keystroke themselves. A blacklist of text widgets missed
    # Listbox, Spinbox and Scale, so selecting a folder in the scan queue and pressing
    # "d" rejected the active photo — a single letter, silently mislabelling a file.
    _KEY_CONSUMING_WIDGETS = frozenset(
        {"Entry", "TEntry", "Text", "TCombobox", "Listbox", "Spinbox", "TSpinbox", "TScale", "Scale"}
    )

    def _focus_is_text_input(self) -> bool:
        """True when the focused widget should get the keystroke instead of the grid."""
        widget = self.root.focus_get()
        if widget is None:
            return False
        try:
            widget_class = widget.winfo_class()
        except Exception:  # noqa: BLE001
            return True
        if widget_class in self._KEY_CONSUMING_WIDGETS:
            return True
        # A modal is open: its own bindings own the keyboard, whatever has focus in it.
        try:
            return widget.winfo_toplevel() is not self.root
        except Exception:  # noqa: BLE001
            return True

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

    def _handle_explain_shortcut(self, event) -> str:
        if self._focus_is_text_input():
            return ""
        self.explain_score()
        return "break"

    def _handle_note_shortcut(self, event) -> str:
        if self._focus_is_text_input():
            return ""
        self.edit_note()
        return "break"

    def _handle_back_shortcut(self, event) -> str:
        if self._focus_is_text_input():
            return ""
        self.go_back()
        return "break"

    def _handle_focus_shortcut(self, event) -> str:
        if self._focus_is_text_input():
            return ""
        self.toggle_focus_mode()
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
        paths = [str(sample["path"]) for sample in self.page_samples or self.current_samples]
        if not paths:
            return
        if self.focused_path not in paths:
            self.focused_path = paths[0]
        else:
            index = paths.index(self.focused_path)
            self.focused_path = paths[(index + delta) % len(paths)]
        self._apply_focus_visuals()

    def move_focus_grid(self, row_delta: int, column_delta: int) -> None:
        paths = [str(sample["path"]) for sample in self.page_samples or self.current_samples]
        if not paths:
            return
        # The grid's actual column count, not the raw preference: columns_var is 0 for
        # "fit to window", which is the default, and reading it directly made Up/Down
        # step by one card — identical to Left/Right — for anyone who had not pinned a
        # column count by hand.
        columns = max(1, self._grid_columns())
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
            if focused:
                style = "FocusedMatchCard.TFrame" if is_match else "FocusedCard.TFrame"
            else:
                style = "MatchCard.TFrame" if is_match else "Card.TFrame"
            # Every other card on the page is being redrawn to exactly what it already
            # shows. Moving the focus only ever changes two cards.
            if card.focused == focused and card.style == style:
                continue
            try:
                card.frame.configure(style=style)
            except Exception:  # noqa: BLE001
                pass
            card.name_label.configure(text=f"▶ {Path(path).name}" if focused else Path(path).name)
            card.focused = focused
            card.style = style
        if self.focused_path and self.focused_path in self.cards:
            self._scroll_card_into_view(self.focused_path)
        self._update_preview()
        self._save_review_session()

    # --- focus mode ---------------------------------------------------------
    def toggle_focus_mode(self) -> None:
        if self._focus_window is not None:
            self.exit_focus_mode()
        else:
            self.enter_focus_mode()

    def enter_focus_mode(self) -> None:
        """One photo, full screen, keyboard only.

        The grid is the right tool for surveying a folder and the wrong one for
        working through a queue: every decision costs a card teardown, a re-flow and
        a scroll, and the photo is a thumbnail beside its own metadata. Here the photo
        is the window, and a decision is a single keystroke that redraws one image.
        """
        if self._focus_window is not None:
            self._focus_window.lift()
            return
        if not self.page_samples:
            messagebox.showinfo("Nothing to review", "Run a scan first, or clear the filters hiding everything.")
            return
        if self.focused_path is None:
            self.focused_path = str(self.page_samples[0]["path"])
        palette = self._palette()
        window = Toplevel(self.root)
        window.title("Review")
        window.configure(bg=palette["bg"])
        window.protocol("WM_DELETE_WINDOW", self.exit_focus_mode)
        try:
            window.state("zoomed")
        except Exception:  # noqa: BLE001
            window.geometry(f"{self.root.winfo_width()}x{self.root.winfo_height()}+0+0")
        self._focus_window = window

        frame = ttk.Frame(window, padding=(16, 10))
        frame.pack(fill=BOTH, expand=True)
        bar = ttk.Frame(frame)
        bar.pack(side=BOTTOM, fill="x", pady=(10, 0))
        ttk.Label(bar, textvariable=self._focus_status_var, style="SurfaceMuted.TLabel").pack(side=LEFT)
        ttk.Label(
            bar,
            text="A accept   D reject   S skip   N note   W why   J/K move   Ctrl+Z undo   Esc close",
            style="SurfaceMuted.TLabel",
        ).pack(side=RIGHT)
        self._focus_caption = ttk.Label(frame, textvariable=self._focus_caption_var, anchor="center")
        self._focus_caption.pack(side=BOTTOM, pady=(8, 0))
        self._focus_image_label = ttk.Label(frame, anchor="center")
        self._focus_image_label.pack(side=TOP, fill=BOTH, expand=True)
        frame.configure(style="Surface.TFrame")
        self._focus_image_label.configure(style="SurfaceMuted.TLabel")

        # Bound on the window, not bind_all: the grid's own shortcuts stay untouched
        # and there is no ambiguity about which one is listening.
        for sequence, handler in (
            ("<KeyPress-a>", lambda _e: self._focus_decide(1)),
            ("<KeyPress-g>", lambda _e: self._focus_decide(1)),
            ("<KeyPress-d>", lambda _e: self._focus_decide(0)),
            ("<KeyPress-b>", lambda _e: self._focus_decide(0)),
            ("<KeyPress-s>", lambda _e: self._focus_decide(2)),
            ("<KeyPress-w>", lambda _e: self._focus_explain()),
            ("<KeyPress-n>", lambda _e: self._focus_note()),
            ("<KeyPress-j>", lambda _e: self._focus_step(1)),
            ("<KeyPress-k>", lambda _e: self._focus_step(-1)),
            ("<Right>", lambda _e: self._focus_step(1)),
            ("<Left>", lambda _e: self._focus_step(-1)),
            ("<Control-z>", lambda _e: self._focus_undo()),
            ("<Control-y>", lambda _e: self._focus_redo()),
            # Both must swallow the event: the same keys are bound with bind_all for
            # the grid, and without "break" closing the window would re-open it and
            # Ctrl+Z would undo twice.
            ("<Escape>", lambda _e: self._focus_close()),
            ("<KeyPress-f>", lambda _e: self._focus_close()),
        ):
            window.bind(sequence, handler)
        window.bind("<Configure>", self._on_focus_configure, add="+")
        window.focus_set()
        window.update_idletasks()
        self._refresh_focus_view()

    def exit_focus_mode(self) -> None:
        window = self._focus_window
        self._focus_window = None
        self._focus_image_label = None
        self._focus_photo = None
        if window is None:
            return
        try:
            window.destroy()
        except Exception:  # noqa: BLE001
            pass
        # The grid has been moving underneath all along; bring it to where the
        # reviewer left off rather than back to where they started.
        self._refresh_displayed_results(reset_page=False)

    def _on_focus_configure(self, event) -> None:
        if self._focus_window is None or getattr(event, "widget", None) is not self._focus_window:
            return
        if self._focus_resize_after_id is not None:
            try:
                self.root.after_cancel(self._focus_resize_after_id)
            except Exception:  # noqa: BLE001
                pass
        self._focus_resize_after_id = self._after(180, self._refresh_focus_view)

    def _focus_size(self) -> tuple[int, int]:
        window = self._focus_window
        if window is None:
            return (960, 720)
        try:
            width = max(320, int(window.winfo_width()) - 48)
            height = max(240, int(window.winfo_height()) - 110)
        except Exception:  # noqa: BLE001
            return (960, 720)
        return (width, height)

    def _refresh_focus_view(self) -> None:
        self._focus_resize_after_id = None
        window = self._focus_window
        label = self._focus_image_label
        if window is None or label is None:
            return
        path = self.focused_path
        if not path:
            self._focus_caption_var.set("Nothing left to review.")
            self._focus_status_var.set(self._progress_note())
            return
        width, height = self._focus_size()
        ready = self._take_decoded(path, "preview", width, height)
        try:
            image = ready if ready is not None else self._preview_letterbox(open_oriented(path), width, height)
            photo = ImageTk.PhotoImage(image)
        except Exception:  # noqa: BLE001
            photo = ImageTk.PhotoImage(Image.new("RGB", (width // 2, height), color=self._palette()["button_bg"]))
        self._focus_photo = photo
        label.configure(image=photo)
        self._focus_caption_var.set(self._preview_caption_text(path))
        self._focus_status_var.set(self._progress_note())
        self._prefetch_focus_neighbours(width, height)

    def _prefetch_focus_neighbours(self, width: int, height: int) -> None:
        """Decode the next few in the queue at focus-mode size, not grid size."""
        if self.focused_path is None:
            return
        order = [str(sample["path"]) for sample in self.page_samples]
        if self.focused_path not in order:
            return
        labels = self.store.load_labels() if self.store is not None else {}
        start = order.index(self.focused_path) + 1
        queued = 0
        for candidate in order[start:]:
            if labels.get(candidate) is not None:
                continue
            key = self._decoded_key(candidate, "preview", width, height)
            with self._decoded_lock:
                if key in self._decoded_cache or key in self._decoding:
                    continue
                self._decoding.add(key)
            self._prefetch_queue.append(key)
            queued += 1
            if queued >= 3:
                break
        self._start_prefetch_worker()

    def _focus_decide(self, label: int) -> str:
        if self.focused_path is None:
            return "break"
        path = self.focused_path
        successor = self._next_undecided_after(path)
        verb = {1: "Accepted", 0: "REJECTED", 2: "Skipped"}.get(int(label), "Labelled")
        self._apply_label_batch(
            {path: int(label)},
            status=f"{verb} {Path(path).name}",
            retrain=True,
        )
        if successor is not None:
            self.focused_path = successor
        else:
            self._advance_focus_after(path)
        self._refresh_focus_view()
        return "break"

    def _focus_step(self, delta: int) -> str:
        order = [str(sample["path"]) for sample in self.page_samples]
        if not order:
            return "break"
        position = order.index(self.focused_path) + delta if self.focused_path in order else 0
        self.focused_path = order[max(0, min(position, len(order) - 1))]
        self._refresh_focus_view()
        return "break"

    def _focus_explain(self) -> str:
        self.explain_score(self.focused_path)
        return "break"

    def _focus_note(self) -> str:
        self.edit_note(self.focused_path)
        self._refresh_focus_view()
        return "break"

    def _focus_close(self) -> str:
        self.exit_focus_mode()
        return "break"

    def _focus_undo(self) -> str:
        self.undo_last_label()
        self._refresh_focus_view()
        return "break"

    def _focus_redo(self) -> str:
        self.redo_last_label()
        self._refresh_focus_view()
        return "break"

    # --- image prefetch -----------------------------------------------------
    def _decoded_key(self, path: str, kind: str, width: int, height: int) -> tuple[str, str, int, int]:
        return (path, kind, int(width), int(height))

    def _letterbox_for(self, path: str, kind: str, width: int, height: int) -> Image.Image:
        """Decode and scale one image the way `kind` needs it."""
        source = open_oriented(path)
        if kind == "preview":
            return self._preview_letterbox(source, width, height)
        return self._thumbnail_fill(source, width, height)

    def _take_decoded(self, path: str, kind: str, width: int, height: int) -> Image.Image | None:
        """Hand back a prefetched image if the worker got to it first."""
        with self._decoded_lock:
            return self._decoded_cache.pop(self._decoded_key(path, kind, width, height), None)

    def _prefetch_upcoming(self, count: int = 3) -> None:
        """Decode the next few undecided photos before the reviewer asks for them.

        Deciding a photo costs two JPEG decodes of the *next* one — its card thumbnail
        and the enlarged preview — and on a 6 MP photo that was most of the half-second
        between pressing Accept and seeing the next picture. Doing it on a worker
        thread while the reviewer is still looking at the current photo makes the
        common case a cache hit. Only PIL work happens off the main thread; building
        the Tk image stays where Tk requires it.
        """
        if self._closing or not self.page_samples:
            return
        order = [str(sample["path"]) for sample in self.page_samples]
        start = 0
        if self.focused_path in order:
            start = order.index(self.focused_path) + 1
        labels = self.store.load_labels() if self.store is not None else {}
        preview_width, preview_height = self._preview_size()
        thumb_size = max(120, int(self.thumbnail_size_var.get()))
        queued = 0
        for candidate in order[start:]:
            if labels.get(candidate) is not None:
                continue
            for kind, width, height in (
                ("preview", preview_width, preview_height),
                ("thumb", thumb_size, thumb_size),
            ):
                key = self._decoded_key(candidate, kind, width, height)
                with self._decoded_lock:
                    if key in self._decoded_cache or key in self._decoding:
                        continue
                    self._decoding.add(key)
                self._prefetch_queue.append(key)
            queued += 1
            if queued >= count:
                break
        # Every decision also pulls one card onto the page from the next one to fill
        # the gap, and that card needs a thumbnail nobody has decoded yet. Warming the
        # items just past the page boundary is what makes the grid refill without a
        # visible stall.
        page_end = (self.page_index + 1) * self._page_size()
        for sample in self.displayed_samples[page_end : page_end + count]:
            key = self._decoded_key(str(sample["path"]), "thumb", thumb_size, thumb_size)
            with self._decoded_lock:
                if key in self._decoded_cache or key in self._decoding:
                    continue
                self._decoding.add(key)
            self._prefetch_queue.append(key)
        self._start_prefetch_worker()

    def _start_prefetch_worker(self) -> None:
        if not self._prefetch_queue:
            return
        worker = self._prefetch_thread
        if worker is not None and worker.is_alive():
            return
        self._prefetch_thread = threading.Thread(target=self._prefetch_worker, name="image-prefetch", daemon=True)
        self._prefetch_thread.start()

    def _prefetch_worker(self) -> None:
        while not self._closing:
            try:
                key = self._prefetch_queue.popleft()
            except IndexError:
                return
            path, kind, width, height = key
            try:
                image = self._letterbox_for(path, kind, width, height)
            except Exception:  # noqa: BLE001
                # A file that cannot be read is not a prefetch problem; the synchronous
                # path will hit the same error and draw its placeholder.
                image = None
            with self._decoded_lock:
                self._decoding.discard(key)
                if image is not None:
                    self._decoded_cache[key] = image
                    while len(self._decoded_cache) > 12:
                        self._decoded_cache.popitem(last=False)

    def _preview_caption_text(self, path: str) -> str:
        caption = (
            f"{Path(path).name}  —  score {self._match_score_for_path(path):.3f}  —  {self._label_display_text(path)}"
        )
        note = self.note_for(path)
        return f"{caption}\n{note}" if note else caption

    def _preview_chrome_height(self) -> int:
        """Vertical space inside the preview pane that is not the picture."""
        total = 20  # the frame's own top and bottom padding
        for name in ("preview_caption_label",):
            widget = getattr(self, name, None)
            if widget is not None:
                try:
                    total += max(int(widget.winfo_reqheight()), 0) + 4
                except Exception:  # noqa: BLE001
                    continue
        # The Accept/REJECT row, which is a button plus its padding.
        total += 40
        return total

    def _preview_height_share(self) -> float:
        """Fraction of the window the preview pane takes, as last left by the sash."""
        return max(0.2, min(self._preview_share, 0.8))

    def _min_grid_height(self) -> int:
        """Enough of the grid to show one whole card, buttons included."""
        return max(120, int(self.thumbnail_size_var.get())) + CARD_CHROME_HEIGHT

    def _preview_pane_shown(self) -> bool:
        try:
            return str(self.preview_frame) in self.workspace.panes()
        except Exception:  # noqa: BLE001
            return False

    def _show_preview_pane(self) -> None:
        if self._preview_pane_shown():
            return
        try:
            # weight 0: the grid absorbs the slack when the window is resized, so the
            # height the reviewer chose for the picture stays the height they chose.
            self.workspace.insert(0, self.preview_frame, weight=0)
        except Exception:  # noqa: BLE001
            return
        self._after(0, self._apply_preview_sash)

    def _hide_preview_pane(self) -> None:
        if not self._preview_pane_shown():
            return
        try:
            self.workspace.forget(self.preview_frame)
        except Exception:  # noqa: BLE001
            return

    def _apply_preview_sash(self) -> None:
        """Put the sash back where the reviewer last left it, within reason.

        The remembered share is honoured until it would leave the grid too short to
        act in. On a small window that is exactly what used to happen: the picture got
        its 45% and the buttons on every card fell off the bottom.
        """
        if not self._preview_pane_shown():
            return
        try:
            total = int(self.workspace.winfo_height())
            if total <= 1:
                return
            wanted = int(total * self._preview_height_share())
            ceiling = min(int(total * MAX_PREVIEW_SHARE), total - self._min_grid_height())
            # On a genuinely tiny window even the floor cannot be met; split evenly
            # rather than collapsing one side to nothing.
            ceiling = max(ceiling, int(total * 0.35))
            self.workspace.sashpos(0, max(180, min(wanted, max(180, ceiling))))
        except Exception:  # noqa: BLE001
            return

    def _on_workspace_configure(self, event) -> None:
        if getattr(event, "widget", None) is not self.workspace:
            return
        if self._sash_clamp_after_id is not None:
            try:
                self.root.after_cancel(self._sash_clamp_after_id)
            except Exception:  # noqa: BLE001
                pass
        self._sash_clamp_after_id = self._after(120, self._clamp_preview_sash)

    def _clamp_preview_sash(self) -> None:
        """Keep the split proportional as the window resizes, within the limits.

        A PanedWindow holds its sash at a fixed pixel offset, so shrinking the window
        starved the grid and re-growing it left the picture stuck small. What the
        reviewer chose by dragging is a *share*, so that is what is preserved; the
        floor that guarantees room for a whole card still wins over it.
        """
        self._sash_clamp_after_id = None
        if not self._preview_pane_shown():
            return
        try:
            total = int(self.workspace.winfo_height())
            if total <= 1:
                return
            position = int(self.workspace.sashpos(0))
            ceiling = min(int(total * MAX_PREVIEW_SHARE), total - self._min_grid_height())
            ceiling = max(ceiling, int(total * 0.35))
            wanted = max(150, min(int(total * self._preview_height_share()), max(150, ceiling)))
            if abs(wanted - position) > 2:
                self.workspace.sashpos(0, wanted)
                self._schedule_preview_resize()
        except Exception:  # noqa: BLE001
            return

    def _remember_preview_sash(self, _event: object = None) -> None:
        """Persist the split after a drag, so the next session opens the same way."""
        if not self._preview_pane_shown():
            return
        try:
            total = int(self.workspace.winfo_height())
            position = int(self.workspace.sashpos(0))
        except Exception:  # noqa: BLE001
            return
        if total <= 1 or position <= 0:
            return
        share = max(0.2, min(position / total, 0.8))
        if abs(share - self._preview_share) < 0.01:
            return
        self._preview_share = share
        self._save_user_prefs()

    def _preview_size(self) -> tuple[int, int]:
        """Size of the enlarged active picture.

        Once the pane is on screen its height is whatever the reviewer dragged the
        sash to, and the picture fills it. Before that — and whenever the pane has no
        usable height yet — it falls back to a share of the window.
        """
        try:
            root_width = int(self.root.winfo_width())
            root_height = int(self.root.winfo_height())
        except Exception:  # noqa: BLE001
            root_width, root_height = 0, 0
        if root_width <= 1 or root_height <= 1:
            root_width, root_height = 992, 1041
        height = 0
        try:
            if self.preview_frame.winfo_ismapped():
                height = int(self.preview_frame.winfo_height()) - self._preview_chrome_height()
        except Exception:  # noqa: BLE001
            height = 0
        if height > 60:
            # The pane's own height is the authority once it is on screen. Imposing a
            # floor above it rendered the picture taller than the space it had and the
            # bottom of the photo was simply clipped off.
            height = min(height, max(120, root_height - self._min_grid_height() - 60))
        else:
            # Not laid out yet: fall back to a share of the window.
            height = int(root_height * self._preview_height_share())
            height = max(180, min(height, max(180, root_height - self._min_grid_height() - 120)))
        height = max(80, height)
        # Full window width: a landscape photo is the common case, and capping the
        # width at 1.9x the height threw away most of the space the reviewer just
        # made by dragging the sash down.
        width = max(420, root_width - 60)
        return width, height

    def _on_root_configure(self, event) -> None:
        if getattr(event, "widget", None) is not self.root:
            return
        self._schedule_preview_resize()

    def _schedule_reflow(self) -> None:
        """Re-lay the grid when the window resize changes how many cards fit.

        Without this the column count was worked out once, at whatever size the window
        happened to be when the results first rendered, and never revisited — so
        "fit to window" fitted the window it started in.
        """
        if self._reflow_after_id is not None:
            try:
                self.root.after_cancel(self._reflow_after_id)
            except Exception:  # noqa: BLE001
                pass
        self._reflow_after_id = self._after(220, self._reflow_if_columns_changed)

    def _reflow_if_columns_changed(self) -> None:
        self._reflow_after_id = None
        if not self.page_samples:
            return
        columns = self._grid_columns()
        # _card_info_width tracks the width too, so compare the whole signature: a
        # resize that keeps the column count can still change how wide the text is.
        layout = (columns, self._card_info_width(columns), max(120, int(self.thumbnail_size_var.get())))
        if layout != self._grid_layout:
            self._render_samples()

    def _on_preview_configure(self, event) -> None:
        if getattr(event, "widget", None) is not self.preview_frame:
            return
        self._schedule_preview_resize()

    def _schedule_preview_resize(self) -> None:
        if self._preview_resize_after_id is not None:
            try:
                self.root.after_cancel(self._preview_resize_after_id)
            except Exception:  # noqa: BLE001
                pass
        self._preview_resize_after_id = self._after(180, self._resize_preview_if_needed)

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
            self._hide_preview_pane()
            return
        width, height = self._preview_size()
        self._preview_render_size = (width, height)
        cache_key = (path, width, height)
        photo = self.preview_cache.get(cache_key)
        if photo is None:
            try:
                ready = self._take_decoded(path, "preview", width, height)
                photo = ImageTk.PhotoImage(
                    ready if ready is not None else self._preview_letterbox(open_oriented(path), width, height)
                )
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
        self._show_preview_pane()
        # Warm the next few while the reviewer looks at this one.
        self._prefetch_upcoming()

    def _scroll_card_into_view(self, path: str) -> None:
        """Queue a scroll to this card, at most one per idle cycle.

        This used to force a synchronous layout pass with update_idletasks and then
        scroll the card to the top of the viewport whether or not it was already
        visible. Profiled at ~70 ms a call and two calls per Accept, it was the single
        most expensive thing about deciding a photo — and it yanked the grid around
        for no reason. Deferring lets Tk lay out once, on its own schedule.
        """
        self._pending_scroll_path = path
        if self._scroll_after_id is not None:
            return
        self._scroll_after_id = self._after(0, self._apply_pending_scroll)

    def _apply_pending_scroll(self) -> None:
        self._scroll_after_id = None
        path = self._pending_scroll_path
        self._pending_scroll_path = None
        card = self.cards.get(path) if path else None
        if card is None:
            return
        try:
            inner_height = max(int(self.grid_inner.winfo_height()), 1)
            view_height = max(int(self.grid_canvas.winfo_height()), 1)
            top = int(card.frame.winfo_y())
            bottom = top + int(card.frame.winfo_height())
            first = float(self.grid_canvas.yview()[0]) * inner_height
            # Already fully on screen: leave the reviewer's scroll position alone.
            if top >= first and bottom <= first + view_height:
                return
            self.grid_canvas.yview_moveto(max(min(top / inner_height, 1.0), 0.0))
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
        self._card_menus.clear()
        self._bucket_headings.clear()
        self._grid_layout = None

    def _grid_columns(self) -> int:
        """How many cards fit across, honouring an explicit choice when one is set.

        `columns` of 0 means auto: fit as many ~TARGET_CARD_WIDTH cards as the grid is
        wide. That is what makes a wide window show more photos instead of larger ones.
        """
        try:
            chosen = int(self.columns_var.get())
        except Exception:  # noqa: BLE001
            chosen = 0
        if chosen > 0:
            return max(1, min(chosen, 12))
        width = 0
        for widget in (self.grid_canvas, self.root):
            try:
                width = int(widget.winfo_width())
            except Exception:  # noqa: BLE001
                width = 0
            if width > 1:
                break
        if width <= 1:
            width = 1200
        thumb = max(120, int(self.thumbnail_size_var.get()))
        # A card is its thumbnail plus a readable text column plus padding.
        needed = max(TARGET_CARD_WIDTH, thumb + 220)
        return max(1, min(width // needed, 12))

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
        """Bring the grid in line with `page_samples`, reusing the cards already in it.

        This used to destroy every widget and build the page again from scratch. A
        card is about seventeen Tk widgets, so deciding one photo on a twenty-card
        page tore down and rebuilt three hundred and forty of them — profiled at over
        a second per Accept, which is most of what made the queue feel like it was
        grinding. Cards that are staying are re-gridded in place instead; only the
        ones that actually left are destroyed, and only the ones that actually
        arrived are built.
        """
        samples = self.page_samples if self.current_samples else []
        if not samples:
            self._clear_grid()
            self._update_preview()
            self._refresh_empty_state()
            self._sync_view_switch()
            return
        self._refresh_empty_state()
        self._sync_view_switch()
        columns = self._grid_columns()
        info_width = self._card_info_width(columns)
        thumb_size = max(120, int(self.thumbnail_size_var.get()))
        # Card geometry is baked in at build time, so when any of it changes there is
        # nothing to reuse and the whole page genuinely has to be rebuilt.
        layout = (columns, info_width, thumb_size)
        if layout != self._grid_layout:
            self._clear_grid()
            self._grid_layout = layout
        # Column config survives _clear_grid, so drop the settings for any column the
        # previous render used and this one does not — a stale weighted empty column
        # shifts every card sideways.
        for column in range(columns, 16):
            self.grid_inner.columnconfigure(column, weight=0, uniform="")
        for column in range(columns):
            self.grid_inner.columnconfigure(column, weight=1, uniform="cards")
        buckets: list[str] = []
        grouped: dict[str, list[dict[str, object]]] = {}
        for sample in samples:
            bucket = str(sample["bucket"])
            if bucket not in grouped:
                buckets.append(bucket)
                grouped[bucket] = []
            grouped[bucket].append(sample)
        # A bucket can straddle a page boundary, so the header has to say how many of
        # the bucket's total are on this page. A bare count would read as the whole
        # bucket and repeat itself unchanged on the next page.
        totals: dict[str, int] = {}
        for sample in self.displayed_samples:
            name = str(sample.get("bucket", ""))
            totals[name] = totals.get(name, 0) + 1

        wanted = {str(sample["path"]) for sample in samples}
        for path in [path for path in self.cards if path not in wanted]:
            departing = self.cards.pop(path)
            try:
                self.photo_refs.remove(departing.image_ref)
            except ValueError:
                pass
            departing.frame.destroy()
        for name in [name for name in self._bucket_headings if name not in grouped]:
            self._bucket_headings.pop(name).destroy()

        threshold = float(self.threshold_var.get())
        row = 0
        for bucket in buckets:
            items = grouped[bucket]
            total = totals.get(bucket, len(items))
            heading = f"{bucket} ({len(items)})" if total == len(items) else f"{bucket} ({len(items)} of {total})"
            label = self._bucket_headings.get(bucket)
            if label is None:
                # Bucket headings were the same weight as the card text below them, so
                # the boundaries between groups vanished while scrolling.
                label = ttk.Label(self.grid_inner, style="BucketHeading.TLabel")
                self._bucket_headings[bucket] = label
            label.configure(text=heading.upper())
            label.grid(row=row, column=0, columnspan=columns, sticky="w", padx=10, pady=(10, 4))
            row += 1
            for index, sample in enumerate(items):
                path = str(sample["path"])
                score = float(cast(float, sample["score"]))
                target_row = row + index // columns
                target_column = index % columns
                card = self.cards.get(path)
                if card is None:
                    self._render_card(
                        self.grid_inner,
                        path,
                        score,
                        target_row,
                        column=target_column,
                        columnspan=1,
                        info_width=info_width,
                    )
                    continue
                # Reused card: move it, and refresh the parts a retrain can change.
                card.score = score
                card.score_label.configure(text=f"Score: {score:.3f}")
                card.label_label.configure(
                    text=self._card_label_text(path), style=self._label_style(path)
                )
                card.details_label.configure(text=self._axis_details_text(path))
                focused = path == self.focused_path
                matched = score >= threshold
                if focused:
                    style = "FocusedMatchCard.TFrame" if matched else "FocusedCard.TFrame"
                else:
                    style = "MatchCard.TFrame" if matched else "Card.TFrame"
                card.frame.configure(style=style)
                card.frame.grid(
                    row=target_row, column=target_column, columnspan=1, sticky="nsew", padx=10, pady=6
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
                ready = self._take_decoded(path, "thumb", thumb_size, thumb_size)
                photo = ImageTk.PhotoImage(
                    ready if ready is not None else self._thumbnail_fill(open_oriented(path), thumb_size, thumb_size)
                )
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
        # A decision used to be one more line of grey text among four, so a page of
        # cards could not be read for state at a glance. It gets its own colour now.
        label_label = ttk.Label(
            info,
            text=self._card_label_text(path),
            wraplength=wrap_length,
            justify="left",
            style=self._label_style(path),
        )
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
            # Six more buttons under Accept/REJECT cost a whole row of height on every
            # card, which is most of what was squeezing the grid. They live on the
            # card's context menu now, and behind one "More" button for discoverability.
            ttk.Button(primary, text="Skip", width=8, command=lambda: self.set_label(path, 2)).pack(
                side=LEFT, padx=(6, 0)
            )
            more = ttk.Button(primary, text="More \u25be", width=8)
            more.pack(side=LEFT, padx=(6, 0))
            menu = self._card_menu(path)

            def _open_more(widget: ttk.Button = more, target: Menu = menu) -> None:
                self._post_menu(widget, target)

            more.configure(command=_open_more)
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

        card_menu = menu if show_actions else self._card_menu(path)

        def _on_right_click(event: object, target: Menu = card_menu) -> None:
            self.focus_path(path)
            try:
                target.tk_popup(int(getattr(event, "x_root", 0)), int(getattr(event, "y_root", 0)))
            finally:
                target.grab_release()

        for widget in (frame, image_label, right, info, name_label, score_label, label_label, details_label):
            widget.bind("<Button-1>", _on_click)
            widget.bind("<Double-Button-1>", _on_double_click)
            widget.bind("<Button-3>", _on_right_click)
        return photo

    def _card_menu(self, path: str) -> Menu:
        """The per-card actions that no longer need a button each."""
        menu = Menu(self.root, tearoff=False)
        menu.add_command(label="View full size", command=lambda: self.view_image(path))
        menu.add_command(label="Why this score?", command=lambda: self.explain_score(path))
        menu.add_command(label="Note...", command=lambda: self.edit_note(path))
        menu.add_separator()
        menu.add_command(label="Find similar", command=lambda: self.find_similar(path))
        menu.add_command(label="Reveal in file manager", command=lambda: self.reveal_in_file_manager(path))
        self._card_menus.append(menu)
        return menu

    @staticmethod
    def _post_menu(widget, menu: Menu) -> None:
        try:
            menu.tk_popup(widget.winfo_rootx(), widget.winfo_rooty() + widget.winfo_height())
        finally:
            menu.grab_release()

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
    def _thumbnail_fill(image: Image.Image, width: int, height: int) -> Image.Image:
        """Scale and centre-crop to exactly fill the slot.

        The grid used to letterbox, so a portrait photo in a square slot left about a
        third of its width empty and the cards beside it looked misaligned. The full
        frame is still shown untouched in the preview and the full-size viewer; this
        is only the contact sheet.
        """
        source = image.convert("RGB")
        width = max(1, int(width))
        height = max(1, int(height))
        scale = max(width / source.width, height / source.height)
        scaled = source.resize(
            (max(1, round(source.width * scale)), max(1, round(source.height * scale))),
            Image.Resampling.LANCZOS,
        )
        left = max(0, (scaled.width - width) // 2)
        top = max(0, (scaled.height - height) // 2)
        return scaled.crop((left, top, left + width, top + height))

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

    def _card_label_text(self, path: str) -> str:
        """Label plus note, which is what the card has room to show."""
        text = self._label_display_text(path)
        note = self.note_for(path)
        return f"{text} — {note}" if note else text

    def _label_style(self, path: str) -> str:
        return {
            "good": "Accepted.TLabel",
            "bad": "Rejected.TLabel",
            "skip": "Skipped.TLabel",
        }.get(self._label_text(path), "Undecided.TLabel")

    def note_for(self, path: str) -> str:
        if self.store is None:
            return ""
        return self.store.load_notes().get(path, "")

    def edit_note(self, path: str | None = None) -> None:
        """Record why a photo was judged the way it was.

        Accept/REJECT/Skip captures the decision and nothing else, so revisiting a
        folder months later leaves no way to reconstruct the reasoning — or to mark
        the one image that needs a second opinion.
        """
        path = path or self.focused_path
        if not path or self.store is None:
            messagebox.showinfo("No photo selected", "Pick a photo first.")
            return
        dialog, outer = self._create_modal("Note", padding=12, resizable=(True, True))
        ttk.Label(outer, text=Path(path).name).pack(side=TOP, anchor="w", pady=(0, 6))
        palette = self._palette()
        entry = Text(outer, width=52, height=6, wrap="word")
        entry.pack(side=TOP, fill=BOTH, expand=True)
        entry.configure(
            bg=palette["entry_bg"],
            fg=palette["fg"],
            insertbackground=palette["fg"],
            highlightbackground=palette["panel"],
            relief="solid",
        )
        entry.insert("1.0", self.note_for(path))
        entry.focus_set()
        row = self._modal_button_row(outer)

        def save() -> None:
            assert self.store is not None
            notes = self.store.load_notes()
            text = entry.get("1.0", "end").strip()
            if text:
                notes[path] = text
            else:
                notes.pop(path, None)
            try:
                self.store.save_notes(notes)
            except OSError as exc:
                messagebox.showerror("Could not save the note", str(exc), parent=dialog)
                return
            self._refresh_label_state(path)
            self.status_var.set(("Note saved for " if text else "Note cleared for ") + Path(path).name)
            dialog._safe_close()  # type: ignore[attr-defined]

        ttk.Button(row, text="Save", command=save).pack(side=RIGHT)
        ttk.Button(row, text="Cancel", command=dialog._safe_close).pack(  # type: ignore[attr-defined]
            side=RIGHT, padx=(0, 8)
        )
        dialog.bind("<Escape>", lambda _event: dialog._safe_close())  # type: ignore[attr-defined]

    def _label_display_text(self, path: str) -> str:
        return {
            "good": "Accepted",
            "bad": "REJECTED",
            "skip": "Skipped",
        }.get(self._label_text(path), "Unlabeled")

    def _label_text(self, path: str) -> str:
        if self.store is None:
            return "unlabeled"
        return _LABEL_WORDS.get(self.store.load_labels().get(path), "unlabeled")

    def set_label(self, path: str, label: int) -> None:
        if self.store is None:
            return
        # Deciding on the active picture moves to the next one. Without this the big
        # preview sits on the image you just judged and the click looks like it did
        # nothing. Both facts are read before the label lands: applying it takes this
        # photo out of the grid, and by then its position in the queue — which is the
        # only thing that makes "the next one" mean anything — is gone.
        was_focused = path == self.focused_path
        successor = self._next_undecided_after(path) if was_focused else None
        if successor is not None:
            # Moved before the label lands, not after, so the re-render below draws the
            # new active card once. Setting it afterwards meant every decision drew the
            # focus twice — the old card, then a correction — and each pass rebuilt the
            # enlarged preview image.
            self.focused_path = successor
        verb = {1: "Accepted", 0: "REJECTED", 2: "Skipped"}.get(int(label), "Labelled")
        self._apply_label_batch(
            {path: int(label)},
            status=f"{verb} {Path(path).name} — teaching the scanner...",
            retrain=True,
        )
        if not was_focused:
            return
        if successor is None or successor not in self.cards:
            # The successor moved off this page, or there was none: work it out again
            # against what is actually on screen now.
            self._advance_focus_after(path)

    def _next_undecided_after(self, path: str) -> str | None:
        """The next photo on this page, after `path`, that carries no decision yet.

        Wraps to the top of the page once, and never hands back `path` itself.
        """
        order = [str(sample["path"]) for sample in (self.page_samples or self.current_samples)]
        if len(order) < 2:
            return None
        try:
            index = order.index(path)
        except ValueError:
            index = -1
        labels = self.store.load_labels() if self.store is not None else {}
        # Everything after the current position, then wrap around to what came before.
        for candidate in order[index + 1 :] + order[: max(index, 0)]:
            if candidate != path and labels.get(candidate) is None:
                return candidate
        return None

    def _advance_focus_after(self, path: str) -> None:
        """Point the active picture at the next photo on this page you have not decided.

        The wrap-around here used to ignore labels, so once you reached the bottom of a
        batch the focus looped straight back onto photos you had already accepted or
        rejected — which is what made the queue feel like it never ended. Decided photos
        are skipped now, and when none are left the focus stays put and says so.
        """
        order = [str(sample["path"]) for sample in (self.page_samples or self.current_samples)]
        if len(order) < 2:
            return
        candidate = self._next_undecided_after(path)
        if candidate is not None:
            self.focused_path = candidate
            self._apply_focus_visuals()
            return
        # Nothing undecided left in front of the reviewer. Say so rather than silently
        # handing back a photo that was already judged.
        if self.page_index + 1 < self._page_count():
            self.next_page()
            self.status_var.set(
                f"Page finished — showing page {self.page_index + 1} of {self._page_count()}."
            )
            return
        remaining = self._undecided_remaining()
        if remaining:
            self.status_var.set(
                f"Every photo on screen is decided. {remaining} undecided left in this folder — "
                "the next batch arrives when the re-rank finishes."
            )
        else:
            self.status_var.set("Every photo in this folder has been decided. Nothing left to review.")

    def _undecided_remaining(self, labels: dict[str, int] | None = None) -> int:
        """How many scanned, visible photos still carry no Accept/REJECT/Skip.

        `labels` is accepted so a caller that already loaded them does not pay for a
        second copy of the whole label map on every keystroke.
        """
        if self.current_state is None or self.store is None:
            return 0
        if labels is None:
            labels = self.store.load_labels()
        mask = self._result_visibility_mask()
        return sum(
            1
            for path, include in zip(self.current_state.paths, mask, strict=False)
            if include and labels.get(str(path)) is None
        )

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
            self.undo_stack.append(
                {"before": before, "after": dict(changes), "description": self._describe_change(changes)}
            )
            self.redo_stack.clear()
        for path in changes:
            self._refresh_label_state(path)
        # Re-render now rather than waiting for the retrain that follows. The re-rank
        # runs on a background thread and can take seconds, and until it landed the
        # grid still showed every photo the reviewer had just decided — the click
        # looked like it had done nothing at all.
        if self.current_state is not None:
            self._refresh_displayed_results(reset_page=False)
        # Always say how much is actually left. Without this the queue refilling after
        # every retrain looks infinite, because nothing on screen ever counts down.
        self.status_var.set(f"{status} {self._progress_note()}".strip())
        self._sync_history_controls()
        self._save_review_session()
        if retrain:
            self._schedule_retrain(len(changes))

    def _progress_note(self) -> str:
        """Short 'how much is left' line for the status bar."""
        if self.current_state is None or self.store is None:
            return ""
        labels = self.store.load_labels()
        decided = sum(1 for value in labels.values() if value in (0, 1, 2))
        remaining = self._undecided_remaining(labels)
        on_screen = sum(
            1 for sample in self.page_samples if labels.get(str(sample["path"])) is None
        )
        note = f"[{decided} decided | {on_screen} left on this page | {remaining} left in folder]"
        # Say that the model is behind rather than letting it look like the decisions
        # went nowhere. They are saved either way; it is the ranking that is waiting.
        if self._retrain_pending:
            waiting = self._labels_since_retrain
            note += f" · {waiting} to fold into the ranking"
        return note

    def _retrain_report(
        self,
        state: ScoreState,
        previous_scores: dict[str, float],
        threshold: float,
    ) -> str:
        """Say what the labels just did, in terms the reviewer can check."""
        learning_text = state.learning_summary or "no model yet"
        moved = 0
        if previous_scores:
            for path, score in zip(state.paths, state.scores, strict=False):
                before = previous_scores.get(str(path))
                if before is None:
                    continue
                if (float(before) >= threshold) != (float(score) >= threshold):
                    moved += 1
        if state.classifier_label_count <= 0:
            return "Re-ranked. No Accept/REJECT decisions recorded yet, so the model is still zero-shot."
        effect = (
            f"{moved} photo{'s' if moved != 1 else ''} changed side of the threshold"
            if moved
            else "the ranking did not change"
        )
        return f"Re-ranked with your labels — {learning_text}; {effect}. {self._progress_note()}".strip()

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
        paths, scope = self._output_scope()
        if not paths:
            messagebox.showinfo("Nothing shown", "There are no images on screen to label.")
            return
        verb = {1: "Accept", 0: "REJECT", 2: "Skip"}.get(int(label), "label")
        if not messagebox.askyesno(
            f"{verb} everything shown",
            f"{verb} {scope}?\n\nUse Ctrl+Z afterwards if that was not what you wanted.",
        ):
            return
        self._apply_label_batch(
            dict.fromkeys(paths, label),
            status=f"{verb}ed {len(paths)} shown images.",
            retrain=True,
        )

    @staticmethod
    def _describe_change(changes: dict[str, int | None]) -> str:
        """Name a labelling action the way the reviewer would describe it.

        "Undo" on its own gives no way to tell a single misclick from a bulk Accept of
        four hundred photos, which are very different things to reverse blind.
        """
        if not changes:
            return "label change"
        verbs = {1: "Accept", 0: "REJECT", 2: "Skip", None: "clear"}
        values = set(changes.values())
        verb = verbs.get(next(iter(values)), "label") if len(values) == 1 else "label"
        if len(changes) == 1:
            return f"{verb} {Path(next(iter(changes))).name}"
        return f"{verb} {len(changes)} photos"

    def _undo_description(self) -> str:
        if not self.undo_stack:
            return ""
        return str(self.undo_stack[-1].get("description", "label change"))

    def _redo_description(self) -> str:
        if not self.redo_stack:
            return ""
        return str(self.redo_stack[-1].get("description", "label change"))

    def _sync_history_controls(self) -> None:
        """Keep the Edit menu and the status hint naming what Ctrl+Z would reverse."""
        menu = getattr(self, "edit_menu", None)
        if menu is None:
            return
        undo = self._undo_description()
        redo = self._redo_description()
        try:
            menu.entryconfigure(
                self._undo_menu_index,
                label=f"Undo {undo}" if undo else "Undo",
                state="normal" if undo else "disabled",
            )
            menu.entryconfigure(
                self._redo_menu_index,
                label=f"Redo {redo}" if redo else "Redo",
                state="normal" if redo else "disabled",
            )
        except Exception:  # noqa: BLE001
            pass
        self.undo_hint_var.set(f"Ctrl+Z undoes: {undo}" if undo else "")

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
            status=f"Undid {action.get('description', 'label change')}.",
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
            status=f"Redid {action.get('description', 'label change')}.",
            retrain=True,
            record_undo=False,
        )

    def _refresh_label_state(self, path: str) -> None:
        card = self.cards.get(path)
        if card is not None:
            card.label_label.configure(text=self._card_label_text(path), style=self._label_style(path))
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
        out_dir = self._ask_output_destination("Choose export folder", "export")
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
        build_html_report(
            out_path,
            samples,
            labels,
            scores,
            axis_scores=axis_scores,
            title="Bikini Scanner report",
            match_threshold=float(self.threshold_var.get()),
        )
        messagebox.showinfo("HTML report exported", f"Report written to {out_path}")

    def write_metadata_to_visible(self) -> None:
        if self.current_state is None:
            messagebox.showinfo("No results", "Run a scan first.")
            return
        paths, scope = self._output_scope()
        if not paths:
            messagebox.showinfo("No results", "Nothing selected.")
            return
        if not messagebox.askyesno("Write metadata", f"Write bikini keyword tags into {scope}?"):
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
        paths, scope = self._output_scope()
        if not paths:
            messagebox.showinfo("No results", "Nothing selected.")
            return
        if not messagebox.askyesno("Move to trash", f"Send {scope} to the recycle bin?"):
            return
        outcome = trash_files(paths)
        if not outcome.available:
            messagebox.showinfo("Trash unavailable", f"Recycle-bin support is unavailable: {outcome.reason}")
            return
        self._record_trashed(outcome)
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
        """The log, with a level filter and a search box.

        It was a raw tail, and it is where several answers still only exist — scan
        failures, plugin errors, per-file skip reasons. Scrolling a few thousand lines
        looking for the word ERROR is not a way to find them.
        """
        path = configure_logging()
        dialog, outer = self._create_modal(
            "Recent log", padding=10, geometry="960x620", resizable=(True, True)
        )
        ttk.Label(outer, text=f"Log file: {path}").pack(anchor="w", pady=(0, 6))
        controls = ttk.Frame(outer)
        controls.pack(fill="x", pady=(0, 6))
        level_var = StringVar(value="all")
        search_var = StringVar(value="")
        count_var = StringVar(value="")
        ttk.Label(controls, text="Level").pack(side=LEFT)
        ttk.Combobox(
            controls,
            textvariable=level_var,
            values=("all", "warnings and errors", "errors only"),
            state="readonly",
            width=20,
        ).pack(side=LEFT, padx=(4, 12))
        ttk.Label(controls, text="Containing").pack(side=LEFT)
        ttk.Entry(controls, textvariable=search_var, width=32).pack(side=LEFT, padx=(4, 12))
        ttk.Label(controls, textvariable=count_var, style="FormMuted.TLabel").pack(side=LEFT)

        body = ttk.Frame(outer)
        body.pack(fill=BOTH, expand=True)
        text = Text(body, wrap="none", state="normal")
        text.pack(side=LEFT, fill=BOTH, expand=True)
        scroll = ttk.Scrollbar(body, orient="vertical", command=text.yview)
        scroll.pack(side=RIGHT, fill="y")
        text.configure(yscrollcommand=scroll.set)

        def apply_filter(*_args: object) -> None:
            wanted = level_var.get()
            needle = search_var.get().strip().lower()
            keep = []
            total = 0
            for line in read_log_tail().splitlines():
                total += 1
                if wanted == "errors only" and " ERROR" not in line and " CRITICAL" not in line:
                    continue
                if wanted == "warnings and errors" and not any(
                    token in line for token in (" WARNING", " ERROR", " CRITICAL")
                ):
                    continue
                if needle and needle not in line.lower():
                    continue
                keep.append(line)
            text.configure(state="normal")
            text.delete("1.0", END)
            text.insert("1.0", "\n".join(keep))
            text.configure(state="disabled")
            text.see(END)
            count_var.set(f"{len(keep)} of {total} lines")

        for variable in (level_var, search_var):
            try:
                variable.trace_add("write", apply_filter)
            except Exception:  # noqa: BLE001
                continue
        apply_filter()

        buttons = ttk.Frame(dialog, padding=(10, 0, 10, 10))
        buttons.pack(fill="x")
        ttk.Button(buttons, text="Refresh", command=apply_filter).pack(side=LEFT)
        ttk.Button(buttons, text="Open log folder", command=lambda: self.reveal_in_file_manager(str(path))).pack(
            side=LEFT, padx=8
        )

        def copy_shown() -> None:
            self.root.clipboard_clear()
            self.root.clipboard_append(text.get("1.0", END))
            self.status_var.set("Copied the filtered log lines to the clipboard.")

        ttk.Button(buttons, text="Copy shown", command=copy_shown).pack(side=LEFT)
        ttk.Button(
            buttons, text="Close", command=dialog._safe_close  # type: ignore[attr-defined]
        ).pack(side=RIGHT)

    @staticmethod
    def _refresh_log_text(text: Text) -> None:
        text.configure(state="normal")
        text.delete("1.0", END)
        text.insert("1.0", read_log_tail())
        text.configure(state="disabled")

    def _record_trashed(self, outcome: TrashOutcome) -> None:
        """Keep a note of what went to the recycle bin, and when.

        The files are recoverable — that is what the recycle bin is for — but the app
        forgot them the instant they left the grid, so "which photos did I just bin?"
        had no answer at all. This does not undo anything; it makes the list findable.
        """
        moved = [str(path) for path in outcome.trashed]
        if not moved:
            return
        history = self.user_prefs.get("trashed_files")
        history = list(history) if isinstance(history, list) else []
        stamp = datetime.now().isoformat(timespec="seconds")
        history.extend({"path": path, "when": stamp} for path in moved)
        self.user_prefs["trashed_files"] = history[-500:]
        self._save_user_prefs()

    def show_trashed_files(self) -> None:
        """List what this app has sent to the recycle bin, most recent first."""
        history = self.user_prefs.get("trashed_files")
        history = list(history) if isinstance(history, list) else []
        if not history:
            messagebox.showinfo("Recently trashed", "This app has not sent any files to the recycle bin.")
            return
        dialog, outer = self._create_modal(
            "Recently trashed", padding=10, geometry="820x480", resizable=(True, True)
        )
        ttk.Label(
            outer,
            text=(
                f"{len(history)} file(s) sent to the recycle bin by this app.\n"
                "They are still there: restore them from the recycle bin itself, which is the\n"
                "only place that can put a file back where it came from."
            ),
            justify="left",
        ).pack(anchor="w", pady=(0, 8))
        body = ttk.Frame(outer)
        body.pack(fill=BOTH, expand=True)
        text = Text(body, wrap="none")
        text.pack(side=LEFT, fill=BOTH, expand=True)
        scroll = ttk.Scrollbar(body, orient="vertical", command=text.yview)
        scroll.pack(side=RIGHT, fill="y")
        text.configure(yscrollcommand=scroll.set)
        palette = self._palette()
        text.configure(bg=palette["entry_bg"], fg=palette["fg"], insertbackground=palette["fg"], relief="solid")
        for row in reversed(history):
            if isinstance(row, dict):
                text.insert("end", f"{row.get('when', '')}   {row.get('path', '')}\n")
        text.configure(state="disabled")
        buttons = self._modal_button_row(outer)

        def copy_list() -> None:
            self.root.clipboard_clear()
            self.root.clipboard_append(
                "\n".join(str(row.get("path", "")) for row in history if isinstance(row, dict))
            )
            self.status_var.set(f"Copied {len(history)} trashed path(s) to the clipboard.")

        def forget() -> None:
            if not messagebox.askyesno(
                "Clear the list",
                "Forget this list? The files themselves stay in the recycle bin.",
                parent=dialog,
            ):
                return
            self.user_prefs["trashed_files"] = []
            self._save_user_prefs()
            dialog._safe_close()  # type: ignore[attr-defined]

        ttk.Button(buttons, text="Close", command=dialog._safe_close).pack(side=RIGHT)  # type: ignore[attr-defined]
        ttk.Button(buttons, text="Copy list", command=copy_list).pack(side=RIGHT, padx=(0, 8))
        ttk.Button(buttons, text="Clear the list", command=forget).pack(side=LEFT)

    def show_skipped_files(self) -> None:
        """List the files the last scan could not read, with the reason for each.

        The count was reported ("3 files could not be read") and the detail went only
        to the log, so the one question that count raises — which ones? — could not be
        answered from inside the app.
        """
        if self.store is None or not self.store.metadata_path.exists():
            messagebox.showinfo("Skipped files", "Run a scan first.")
            return
        try:
            metadata = json.loads(self.store.metadata_path.read_text(encoding="utf-8"))
            skipped = list(metadata.get("skipped", []))
        except (OSError, ValueError) as exc:
            messagebox.showerror("Skipped files", f"Could not read the scan metadata:\n{exc}")
            return
        if not skipped:
            messagebox.showinfo("Skipped files", "Every image in this folder was read successfully.")
            return
        dialog, outer = self._create_modal(
            "Files that could not be read", padding=10, geometry="900x520", resizable=(True, True)
        )
        ttk.Label(
            outer,
            text=f"{len(skipped)} file(s) were skipped by the last scan of this folder.",
        ).pack(anchor="w", pady=(0, 6))
        # HEIC needs an optional package. Without it every .heic in the folder lands
        # here looking like a corrupt file, when one install would read them all.
        heic = [row for row in skipped if str(row.get("filename", "")).lower().endswith((".heic", ".heif"))]
        if heic and not heif_supported():
            ttk.Label(
                outer,
                text=(
                    f"{len(heic)} of these are HEIC/HEIF photos and support for that format is not "
                    "installed.\nInstalling the optional 'pillow-heif' package and rescanning would "
                    "read them."
                ),
                style="Accent.TLabel",
                justify="left",
                padding=(6, 4),
            ).pack(anchor="w", pady=(0, 8))
        body = ttk.Frame(outer)
        body.pack(fill=BOTH, expand=True)
        text = Text(body, wrap="none")
        text.pack(side=LEFT, fill=BOTH, expand=True)
        scroll = ttk.Scrollbar(body, orient="vertical", command=text.yview)
        scroll.pack(side=RIGHT, fill="y")
        text.configure(yscrollcommand=scroll.set)
        palette = self._palette()
        text.configure(bg=palette["entry_bg"], fg=palette["fg"], insertbackground=palette["fg"], relief="solid")
        for row in skipped:
            text.insert("end", f"{row.get('path', row.get('filename', '?'))}\n    {row.get('error', 'unknown error')}\n")
        text.configure(state="disabled")
        row = self._modal_button_row(outer)

        def copy_list() -> None:
            self.root.clipboard_clear()
            self.root.clipboard_append(
                "\n".join(f"{item.get('path', '')}\t{item.get('error', '')}" for item in skipped)
            )
            self.status_var.set(f"Copied {len(skipped)} skipped file path(s) to the clipboard.")

        ttk.Button(row, text="Close", command=dialog._safe_close).pack(side=RIGHT)  # type: ignore[attr-defined]
        ttk.Button(row, text="Copy list", command=copy_list).pack(side=RIGHT, padx=(0, 8))

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
        # Copies of one photo deserve one decision, not one per copy. Trashing was the
        # only bulk action here, which meant the obvious middle ground - "these are all
        # the same photo, judge them together" - had to be done by hand.
        ttk.Button(
            buttons, text="Accept every group", command=lambda: self._label_duplicate_groups(groups, dialog, 1)
        ).pack(side=LEFT, padx=(8, 0))
        ttk.Button(
            buttons, text="REJECT every group", command=lambda: self._label_duplicate_groups(groups, dialog, 0)
        ).pack(side=LEFT, padx=(6, 0))
        ttk.Button(
            buttons, text="Close", command=dialog._safe_close  # type: ignore[attr-defined]
        ).pack(side=RIGHT)

    def _label_duplicate_groups(self, groups: dict[str, list[str]], dialog: Toplevel, label: int) -> None:
        """Apply one decision to every copy in every duplicate group."""
        paths = [path for members in groups.values() for path in members]
        if not paths:
            messagebox.showinfo("Duplicate groups", "There are no duplicates to label.", parent=dialog)
            return
        verb = {1: "Accept", 0: "REJECT"}.get(int(label), "label")
        if not messagebox.askyesno(
            f"{verb} every duplicate",
            f"{verb} all {len(paths)} files across {len(groups)} duplicate groups?\n\n"
            "Ctrl+Z afterwards reverses the whole batch.",
            parent=dialog,
        ):
            return
        self._apply_label_batch(
            dict.fromkeys(paths, label),
            status=f"{verb}ed {len(paths)} duplicate files.",
            retrain=True,
        )
        dialog._safe_close()  # type: ignore[attr-defined]

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

    def delete_decisions(self) -> None:
        """Remove this folder's labels and notes, on purpose and by that name."""
        if self.store is None:
            messagebox.showinfo("No folder", "Run or select a folder first.")
            return
        labels = self.store.load_labels()
        notes = self.store.load_notes()
        if not labels and not notes:
            messagebox.showinfo("Nothing to delete", "There are no decisions or notes recorded for this folder.")
            return
        parts = []
        if labels:
            parts.append(f"{len(labels)} Accept/REJECT/Skip decision{'s' if len(labels) != 1 else ''}")
        if notes:
            parts.append(f"{len(notes)} note{'s' if len(notes) != 1 else ''}")
        if not messagebox.askyesno(
            "Delete decisions",
            f"Permanently delete {' and '.join(parts)} for this folder?\n\n"
            "This cannot be undone and cannot be rebuilt by rescanning — it is the work "
            "you did by hand.\n\n"
            "Cached embeddings are left alone; use Clear cached scan data for those.",
            default="no",
            icon="warning",
        ):
            return
        self.store.delete_decisions()
        self.undo_stack.clear()
        self.redo_stack.clear()
        self._sync_history_controls()
        if self.current_state is not None:
            self._refresh_active_view()
        self._refresh_summary()
        self.status_var.set("Deleted this folder's decisions and notes.")

    def clear_cache(self) -> None:
        if self.store is None:
            messagebox.showinfo("No cache", "Run or select a folder first.")
            return
        size_bytes = self.store.cache_size_bytes()
        size_mb = size_bytes / (1024 * 1024)
        labels = self.store.load_labels()
        notes = self.store.load_notes()
        kept = []
        if labels:
            kept.append(f"{len(labels)} decision{'s' if len(labels) != 1 else ''}")
        if notes:
            kept.append(f"{len(notes)} note{'s' if len(notes) != 1 else ''}")
        if self.store.config_override_path.exists():
            kept.append("this folder's saved settings")
        kept_text = (
            "\n\nKept: " + ", ".join(kept) + ". Use Tools > Delete decisions for this folder to remove those."
            if kept
            else ""
        )
        if not messagebox.askyesno(
            "Clear cached scan data",
            f"Delete {size_mb:.1f} MB of cached scan data for this folder?\n\n"
            "This removes embeddings, region scores, face counts, scan metadata and the "
            "trained classifier — all of which are rebuilt by the next scan."
            f"{kept_text}",
        ):
            return
        self.store.clear_cache(keep_decisions=True)
        self.current_state = None
        self.current_samples = []
        self.review_samples = []
        self.displayed_samples = []
        self.page_samples = []
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

    def _note_map(self) -> dict[str, str]:
        return self.store.load_notes() if self.store is not None else {}

    def _ask_output_destination(self, title: str, kind: str) -> str:
        """Ask where output goes, starting from wherever it went last time.

        Every export re-asked for a destination with no memory of the previous answer,
        so repeating yesterday's export meant navigating the same tree again.
        """
        remembered = self.user_prefs.get("output_destinations")
        remembered = dict(remembered) if isinstance(remembered, dict) else {}
        previous = str(remembered.get(kind, "") or "")
        chosen = filedialog.askdirectory(
            title=title, initialdir=previous if previous and Path(previous).is_dir() else None
        )
        if not chosen:
            return ""
        remembered[kind] = chosen
        self.user_prefs["output_destinations"] = remembered
        self._save_user_prefs()
        return chosen

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
        notes = self._note_map()
        with csv_path.open("w", newline="", encoding="utf-8") as handle:
            writer = csv.writer(handle)
            # The note travels with the row: an export that records the decision but
            # not the reason is only half of what the reviewer wrote down.
            writer.writerow(["path", "filename", "score", "label", "note", "timestamp"])
            for item in plan:
                writer.writerow(
                    [
                        str(item.source),
                        item.source.name,
                        f"{item.score:.6f}",
                        item.label,
                        notes.get(str(item.source), ""),
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
        # Start from the settings the user last saved rather than the built-in
        # defaults; from_mapping(None) yields the defaults when nothing is stored.
        config = ScannerConfig.from_mapping(load_user_prefs().get("scanner_config"))
    if initial_threshold is not None:
        config.threshold = float(initial_threshold)
    # The constructor binds initial_folder itself, before the model preload starts.
    BikiniScannerApp(root, config=config, initial_folder=initial_folder)
    root.mainloop()
