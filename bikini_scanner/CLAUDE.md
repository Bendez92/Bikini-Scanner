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

