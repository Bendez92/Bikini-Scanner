# bikini_scanner/

Two hot paths in this package are held together by optimisations that are easy to
undo by accident. Re-read before changing `gui.py`, `scorer.py`, or the sampling code.

## Review Loop Performance

Labelling is the hot path, and three separate things used to make it slow. All three
are load-bearing; re-check them before changing anything in this area.

- **Retrains are batched, not per click.** `_schedule_retrain` waits `RETRAIN_IDLE_MS`
  after the last decision (or `RETRAIN_LABEL_BURST` labels). A retrain refits the model
  and rescores the folder; per click that was 1.5-1.8 s on 5 000 images. The label
  itself is written to `labels.json` synchronously and never deferred.
- **The grid updates incrementally.** `_render_samples` diffs against `self.cards` and
  re-grids survivors. Rebuilding the page cost ~340 Tk widgets per decision. A change
  to card geometry (columns, thumbnail size, info width) invalidates `_grid_layout` and
  forces a full rebuild, which is correct - just do not make that the common path.
- **`bucketed_sampling` is vectorised.** The diversity selection is a running minimum
  distance updated with one matrix-vector product per pick. The scalar version was
  1.07 s of a 1.47 s retrain at 5 000 images. It reproduces the scalar tie-breaking
  exactly (`np.lexsort`, stable, earliest-ranked wins); if you touch it, diff the
  selected paths against the old implementation rather than trusting the timings.

Image decoding is prefetched on a worker thread (`_prefetch_upcoming`); only PIL work
happens off the main thread, and `ImageTk.PhotoImage` construction stays on it.

At 80 000 photos the cost that mattered was not any one pass but how often the
labelling path ran them. Measured per Accept, main thread, before and after:

- **A decision in the all-results view does not re-filter or re-sort.** The bands are
  cut off the score and a decided photo keeps its place and fades, so the displayed
  list and its order provably cannot change; only the card is repainted. Doing the
  pass anyway was ~400 ms per click at 80 000 photos.
- **`_triage_page_plan` is cached against `_display_generation`.** One refresh asks for
  it three times (`_page_count`, twice via `_sync_pager`, plus `_page_slice`), and each
  build groups the whole folder.
- **The summary and stats line are debounced** (`SUMMARY_IDLE_MS`). Both are O(the whole
  folder) — the stats line walks every label working out whether the scanner agreed —
  and neither can move by more than one between two clicks. Inline, they were ~65 ms of
  a 113 ms Accept once 40 000 photos had been judged, and they grew with every label.
  The status line, which is the per-click feedback anyone actually reads, is still
  written synchronously.

Net: ~400 ms -> ~45 ms per decision, and flat in the number of labels rather than
growing with it.

A filter/sort pass over the results is the other hot path, and it was ~937 ms for 5 000
samples until four things were hoisted out of the per-sample loop. All four matter:

- `_filter_context()` reads the label map and the seven Tk filter variables **once** per
  pass. `_sample_visible` used to fetch each of them per sample; reading a Tk variable is
  a round trip into Tcl, and `store.load_labels()` returns a defensive copy of the whole
  map.
- `_sample_sort_key` only calls `os.stat` when the sort mode is actually `date`. It used
  to stat every file on every sort regardless, which on a network share dominated
  everything else.
- `BUCKET_ORDER` is module-level; it was rebuilt inside the sort key, once per sample.
- `os.path.basename`, not `Path(...).name` — twice as fast, and it runs per sample.

`_state_index()` caches path -> row for the current `ScoreState`. `_axis_details_text`
runs once per card and `_match_score_for_path` once per focus change; both were linear
scans over every path in the scan.

Net: 937 ms -> 27 ms at 5 000 samples.

Two more things must stay off the main thread, both of which used to freeze the window
with no progress and no way out: loading the model (`_load_backend_then_scan`, a ~600 MB
download on a cold install) and the pre-scan image count (`_count_images_then_scan`,
a full tree walk). `run_scan` re-enters itself once each has landed.

## Window Layout

Vertical space is the scarce resource; several things exist only to defend it.

- **The grid always keeps room for one whole card.** `_min_grid_height()` is the
  thumbnail size plus `CARD_CHROME_HEIGHT`, and `_apply_preview_sash` /
  `_clamp_preview_sash` respect it. A flat floor is not enough: at 950x700 with a
  240px thumbnail the Accept/REJECT row fell below the fold.
- **The sash is a share, not a pixel offset.** A PanedWindow holds its offset on
  resize, which starved the grid when the window shrank and left the picture stuck
  small when it grew again. `_clamp_preview_sash` re-derives it from
  `_preview_height_share()` on every workspace `<Configure>`.
- **`_preview_size` trusts the pane once it is mapped.** Imposing a floor above the
  pane height rendered the photo taller than its space and clipped the bottom off.
- **Columns follow the grid width** (`_grid_columns`, `columns_var == 0` means auto).
  The reflow is bound to `grid_canvas` `<Configure>`, *not* the root: opening the left
  rail narrows the grid by 330px without resizing the window at all, and a root-bound
  reflow missed that entirely.
- **Filters and the scan queue live in the left rail** (`_build_sidebar`), not as extra
  rows above the results. With both open as rows that was four bands of chrome between
  the toolbar and the first photo.
- Grid thumbnails use `_thumbnail_fill` (scale-and-crop). The preview and the full-size
  viewer keep `_preview_letterbox` — cropping is right for a contact sheet and wrong
  for judging a photo.


## A Modal In A Test Blocks Forever

`messagebox` waits for someone to press OK, and in a test nobody ever does, so an
unstubbed dialog does not fail the run — it stops it. One test that drove a full scan
took the suite from 61 s to 1 490 s, and its own runtime swung between 21 s and 877 s
depending on what else was on screen.

Stubbing each call site by hand only ever covered the ones already found, so
`tests/test_functional.py` replaces `gui.messagebox` **once, for the whole file**, with
`_NonBlockingDialogs` (the module-level `DIALOGS`). Every dialog is answered rather than
waited on, and every call is recorded.

- Questions return `DIALOGS.answer`, **False** by default — the safe reply to "delete
  these?". A test that needs yes wraps the call in `with DIALOGS.answering(True):`.
- `with DIALOGS.recording() as dialogs:` captures just the dialogs raised in the block,
  which is how a test asserts one *was not* shown.
- Anything that must not depend on the machine, such as whether send2trash exists, gets
  its own helper (`_bin_available`).

`GuiReviewQueue.test_no_dialog_can_stall_the_suite` guards the net itself: it drives
several dialogs with nothing stubbed locally and fails if the call does not return
promptly. Do not go back to per-test stubbing.

## The Scanner Decides The Easy Ones

The point of the app is that a person does not look at 80 000 photos. Bands, bursts and
bulk actions all reduce the count; `autodecide.py` is the part that removes the majority
outright, by finding the score above which the model has demonstrably been right and the
score below which it has demonstrably been wrong, and deciding everything past them.

Three things make this safe, and all three are load-bearing:

- **The accuracy is measured out-of-fold.** `_out_of_fold_scores` re-scores the labelled
  photos with `cross_val_scores`, so every number comes from a model that had not seen
  that photo. Scoring a photo with the model it helped fit gives a flattering answer,
  and here that answer decides whether tens of thousands of images get labelled unseen.
- **The score calibrated on must be the score applied.** `state.scores` is
  `blend(zero_shot, learned, weight)`, so the out-of-fold figure is blended the same way
  with the same weight before any cut is taken from it. Calibrating on one scale and
  cutting on another silently produces confident nonsense — measured at 36% error while
  reporting 9%.
- **A cut is judged on `wilson_lower_bound`, not the raw fraction.** Twenty right out of
  twenty is evidence of about 83% accuracy, not 100%. Using the observed rate let a cut
  resting on a couple of hundred labels promise nobody would ever be wrong: measured, it
  claimed 0 mistakes and made 356. The bound also makes `expected_mistakes` pessimistic,
  which is the right direction for a number someone is deciding on.

A measurement overtaken by a retrain is re-run against the state that replaced it
(`_out_of_fold_ready`, bounded by `_OOF_MAX_RETRIES`). Handing the stale answer back
instead made the dialog recompute, find no measurement and nothing in flight, and
tell a reviewer with thousands of labels to go and judge forty photos.

`MIN_SUPPORT` is the floor under any cut, and refusing is a normal outcome — at 400
labels a 99.9% target is simply not supportable and the dialog says so rather than
approximating it. The dialog previews counts and expected mistakes before anything is
written, and the whole batch is one `_apply_label_batch`, so Ctrl+Z takes it all back.

## Deciding More Than One Photo At Once

80 000 photos cannot be reviewed one at a time, so there are three widths of decision
and they must stay distinguishable:

- **One card.** `set_label`.
- **A selection.** `selected_paths`, built by `click_card` with the file-manager rules:
  plain click collapses, Ctrl-click toggles, Shift-click takes the range from
  `_selection_anchor` in *page order*. `_decision_targets` is the single place that
  turns "what was clicked" into "what gets decided", and it follows the same rule a
  file manager does: a decision aimed inside the selection takes the selection, aimed
  outside it takes only that card. No confirmation — the set is on screen and outlined.
  `_prune_selection` runs on every page rebuild, because a selection that outlived its
  page would decide photos the reviewer can no longer see.
- **Everything shown.** `mark_all_shown`, which reaches across pages and therefore *does*
  confirm, and the band actions in `_band_actions`.

Selection is drawn as a **border**, never a background: each card line paints its own
background, so a third card colour would need a third copy of every label style.
`_frame_style` is still the one rule, and selection outranks focus and fading in it.

## Bursts

`FolderStore.duplicate_groups` is byte-identical files. `duplicates.near_duplicate_groups`
is the same *shot* — eight frames of one pose, no two files alike, all wanting one
verdict. Exact comparison is 3.2 billion pairs at 80 000 photos, so it is random-projection
LSH over the embeddings: `BANDS` signatures of `BITS` bits each, only same-signature
vectors compared properly, survivors merged with a union-find (so a long burst joins up
even when its first and last frames never pair directly). `MAX_BUCKET` caps a degenerate
bucket rather than letting it reintroduce the quadratic blow-up.

`DEFAULT_THRESHOLD` is deliberately strict (0.97): a false group applies a verdict to a
photo nobody looked at. Measured at 80 000 photos it takes ~25 s, so `ensure_near_duplicates`
runs it on a worker thread and nothing waits on it — decisions made before it lands
simply do not expand.

The expansion itself (`_expand_to_duplicates`) is **off by default** and behind a named
checkbox, because one click deciding eight photos has to be something the reviewer
turned on. The card says `1 of N near-identical` either way: it is the reason to switch
it on, and once on it is the only warning that one keystroke is about to decide a burst.

## Four Views, and What a Card Says

The results grid answers four different questions and `view_mode` names which one:

- **`triage`** — every photo the gates allow, in three confidence bands. This is what a
  finished scan lands on and the only view that shows the whole folder.

- **`detected`** — everything above the sensitivity threshold, grouped by what was seen.
- **`review`** — the curated shortlist of *undecided* photos, built by `bucketed_sampling`.
- **`decided`** — everything carrying an Accept/REJECT/Skip, grouped by whether the
  decision agreed with the scanner.

The third exists because the first two between them could not show a decided photo the
scanner missed: the queue holds only undecided photos and the detected list only what
scored above the threshold, so a false negative appeared in no view at all and "what did
it get wrong?" had no answer on screen. `decided` is therefore the one view that
**ignores `hide_decided`** — hiding decided photos there would empty it by definition
(`FilterContext.decided`, `_sample_visible`, `_hidden_decided_count`).

The three `triage` bands are cut off the threshold by `triage_band`: at or above it is
**Detected**, within `TRIAGE_MARGIN` below it is **Possible**, and the rest is
**Probable reject**. They sub-divide the scanner's own answer rather than competing with
it, so everything in Detected reads DETECTED on its card and nothing else does — a band
and a card can never contradict each other.

Two rules keep that view usable and are easy to undo:

- **A page is split across the bands, not sliced off a sorted list.** `_band_quotas`
  gives each band a share of the page and lets a band too small to spend its share hand
  the remainder back. Cutting one band-sorted list into pages put 300 detected photos on
  the first fifteen pages and nothing else, which is exactly what the separate views
  already did.
- **A decided photo stays where it is and fades.** `triage` is the only view that does
  this, and with `hide_decided` deliberately ignored (`FilterContext.keep_decided`).
  Rows that empty as you work move the next photo under the cursor mid-click. Because a
  ttk label paints its own background, every card line exists twice — once per card
  background — and `_paint_card_state` picks the pair with a `Dim` prefix. `_frame_style`
  is the single rule for the frame itself, shared with `_apply_focus_visuals`; when those
  two disagreed, moving the focus repainted a faded card as an undecided one.

A band action reads `_band_paths`, i.e. the whole band across every page. The
destructive one passes `deletable_only=True`, which drops photos labelled Accept or
Skip: this view keeps decided photos on screen, so without it "Reject all & delete"
overwrote an Accept and binned the file. It also checks `trash_available()` *before*
writing any label, because those labels are permanent and retained — discovering
afterwards that nothing can be deleted leaves the irreversible half done alone.

Each band carries the one bulk action that belongs to it (`_band_actions`), because
doing any of them a card at a time is the work the bands exist to avoid: **Detected**
exports, **Probable reject** rejects the lot and bins the files, and **Possible** gets
no bulk action at all — it is the band meant to be judged by hand, and a stray click
there would decide photos the reviewer had not looked at. Band actions operate on
`_band_paths`, i.e. the whole band across every page, not the page on screen.

"Reject all & delete" writes the labels first, marks them in the global store via
`retain`, and only then bins the files. That order is load-bearing twice over: the
retrain that folds the labels in runs off cached embeddings a moment later, when the
photos are already gone, and `training_set` would otherwise drop those labels for having
no file — quietly undoing the training the action is named for. Deletion is
`trash_files`, i.e. the recycle bin, never an unlink.

The `decided` view does **not** fade: every photo in it is decided by definition, so
dimming them all would grey out the whole screen.

`decision_outcome` is the single definition of true/false positive/negative, and
`VERDICTS` maps each outcome to its card text, style and plain-English gloss. Both
depend on the threshold, so dragging the sensitivity slider re-buckets this view rather
than only re-listing it (`_after_threshold_settles`), and a retrain rebuilds it because
a re-rank can move a decided photo across the threshold.

A card states each fact **once**, and it is worth keeping it that way:

- `DETECTED · 0.840` — what the scanner said, styled by which side of the threshold.
- `REJECTED` — what you said.
- `✘ FALSE POSITIVE` — whether those two agree. The scanner's mistakes are uppercase and
  coloured; its correct calls stay quiet, so a page can be skimmed for the things worth
  a second look.

The term is **named** on the card and **glossed** only in the preview caption
(`_verdict_detail`), which has a full row to itself. The card line used to read
"✘ FALSE POSITIVE — detected, but you rejected it", which restated the two lines above
it on every card in the grid and was the line that wrapped.
