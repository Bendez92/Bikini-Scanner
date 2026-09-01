# Agent Notes for Bikini Scanner

This file captures project-specific knowledge useful for continuing work on the
Bikini Scanner Tkinter desktop app.

## Repository

- **Path**: `C:\Users\Ben\Desktop\bikini-scanner`
- **Python**: 3.12 (virtual environment at `.venv`)
- **Shell**: PowerShell 7 (`C:\Program Files\PowerShell\7\pwsh.exe`)

## Verification

Run these before committing to `main`:

```powershell
.venv\Scripts\python.exe tests\test_functional.py
.venv\Scripts\python.exe -m ruff check bikini_scanner tests scripts
.venv\Scripts\mypy bikini_scanner
.venv\Scripts\python.exe scripts\score_baseline.py --compare tests\baseline_scores.json
```

Expected outcomes:

- 146 functional tests pass.
- Ruff reports `All checks passed!`.
- Mypy reports `Success: no issues found in ... source files`.
- Baseline reports `Baseline matches ...`.

Tests default to `tests.fake_backend.FakeBackend` unless
`BIKINI_SCANNER_REAL_BACKEND=1` is set. The fake backend is deterministic and
avoids downloading model weights.

## Module Map

| Module | Responsibility |
|--------|----------------|
| `bikini_scanner/gui.py` | Tkinter UI and controller. |
| `bikini_scanner/scorer.py` | Scan orchestration, cascade scoring, learning, VLM/refine passes. |
| `bikini_scanner/cascade.py` | Region aggregation, gates, final score/stage computation. |
| `bikini_scanner/clip_backend.py` | PyTorch CLIP backend registry. |
| `bikini_scanner/onnx_backend.py` | ONNX CLIP backend. |
| `bikini_scanner/backend_utils.py` | Image decoding, hashing, batch iteration, backend protocol. |
| `bikini_scanner/image_formats.py` | HEIF registration and EXIF-aware loading. |
| `bikini_scanner/regions.py` | Face-anchored and fallback region planning. |
| `bikini_scanner/store.py` | Per-folder cache access (SQLite + legacy migration). |
| `bikini_scanner/sqlite_cache.py` | SQLite embedding / image-record / face-count / region cache. |
| `bikini_scanner/global_store.py` | Cross-folder active-learning storage. |
| `bikini_scanner/learning.py` | Active-learning orchestration. |
| `bikini_scanner/linear_model.py` | Logistic regression, Platt calibration, AUC/splitting helpers. |
| `bikini_scanner/vlm_backend.py` | OpenAI-compatible local VLM client. |
| `bikini_scanner/output_ops.py` | Copy, move, export and report operations. |
| `bikini_scanner/config.py` | Settings and validation. |
| `bikini_scanner/safe_io.py` | Atomic writes, fsync, quarantine. |
| `bikini_scanner/plugins.py` | Optional post-scan result hooks (off unless enabled). |
| `tests/fake_backend.py` | Deterministic in-memory CLIP-like backend for tests. |
| `tests/test_functional.py` | End-to-end functional suite. |
| `scripts/score_baseline.py` | Golden score comparison harness. |

## SQLite Cache Architecture

`FolderStore` persists per-folder derived data in `.bikini_scanner_cache/cache.db`:

- `embeddings` — keyed by content hash, stores embedding blobs (serialized via
  `numpy.save(..., allow_pickle=False)`).
- `image_records` — path, content hash, mtime nanoseconds, file size.
- `face_counts` — content hash to detected face count.
- `region_embeddings` — content hash, namespace, region key, embedding blob.

Legacy `embeddings.npz`, `embeddings_index.json`, `region_embeddings.npz`, and
`face_counts.json` are migrated once and then removed.

Decode-version changes invalidate derived caches so EXIF orientation fixes are
reflected in cached embeddings.

## Test Pragmas

The test suite sets `BIKINI_SCANNER_TEST_SQLITE_PRAGMAS=1`, which causes the
SQLite cache to use `journal_mode=MEMORY` and `synchronous=OFF`. This keeps
file-backed caches (so cross-instance cache reuse still works) while reducing
disk I/O overhead during tests.

## Trust Boundaries

Two inputs are attacker-influenced whenever a downloaded or shared folder is scanned,
and both are constrained on purpose:

- **`.bikini_scanner_cache/config_override.json`** lives *inside the scanned folder*.
  Only `FOLDER_OVERRIDE_ALLOWED_KEYS` (`config.py`) may come from one; anything that
  could reach off this machine (`vlm_*`), run code (`enable_plugins`), fetch a remote
  model (`backend`, `model_name`, `refine_model`), or weaken the age gate (`pipeline`,
  `exclude_minors`, the minor thresholds, `axis_prompts`) is refused and reported to
  the user. Route every new override read through `filter_folder_override`.
- **`classifier.pkl`**, per-folder and global, is loaded with `RestrictedUnpickler`
  (`store.py`). Never call bare `pickle.load` on either.

The age gate is not a property of the cascade pipeline: `pipeline="legacy"` runs it too.
`exclude_minors` must mean the same thing in both pipelines.

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

## Two Kinds of Folder State

`.bikini_scanner_cache/` holds two categories of file and they must not be treated
alike:

- **Recomputable** — `cache.db`, embeddings, region scores, face counts, scan metadata,
  `classifier.pkl`. A rescan rebuilds all of it.
- **Irreplaceable** — `labels.json`, `notes.json`, `config_override.json`. These are
  hand-made and nothing can rebuild them.

`FolderStore.clear_cache()` defaults to `keep_decisions=True` and preserves the second
group; `delete_decisions()` removes labels and notes only, and is a separate,
separately-worded menu action. `clear_cache()` used to rmtree the whole directory,
which meant an action named for the recomputable half silently destroyed hours of
review. Anything new that lands in the cache directory has to be classified into one of
these two groups.

## Important Conventions

- Do not commit the untracked `.devin/` or `.vscode/` directories; they are
  local workspace configuration.
- Do not add floating dependency versions; prefer packages published for at
  least 7 days.
- Avoid silently swallowing exceptions; log and surface errors.
- Run the verification commands above and review `git diff` before committing.
