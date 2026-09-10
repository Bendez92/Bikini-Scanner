# Bikini Scanner

Project knowledge for the Bikini Scanner Tkinter desktop app. Guidance scoped to
the application package lives in `bikini_scanner/CLAUDE.md` and loads when you
work in that directory.

## Verification

Run these before committing to `main`:

```powershell
.venv\Scripts\python.exe tests\test_functional.py
.venv\Scripts\python.exe -m ruff check bikini_scanner tests scripts
.venv\Scripts\mypy bikini_scanner
.venv\Scripts\python.exe scripts\score_baseline.py --compare tests\baseline_scores.json
```

Expected outcomes:

- 261 functional tests pass (1 skipped without a display).
- Ruff reports `All checks passed!`.
- Mypy reports `Success: no issues found in ... source files`.
- Baseline reports `Baseline matches ...`.

Tests default to `tests.fake_backend.FakeBackend` unless
`BIKINI_SCANNER_REAL_BACKEND=1` is set. The fake backend is deterministic and
avoids downloading model weights.

## Cache Invalidation

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

## "Age Unknown" Is Not "Adult"

The per-subject age gate reads an age from a subject's **face crop**, and a subject can
exist without one: `plan_regions` drops any crop that clamps below `_MIN_CROP_PX`, so a
face under about 28px yields waist and torso crops and no face crop at all. Such a
subject is `present` but can never be `minor`.

`all_minor` is therefore taken over `readable = present & has_face`, not over `present`.
Counting an unaged subject on the not-a-minor side let a photo whose only readable
subject was a child surface on the unaged subject's body crops, scoring exactly as if a
confirmed adult were in frame. And when *no* subject has a readable face
(`has_readable_subjects` is False), `evaluate` leaves the whole-frame gate in charge
rather than handing the decision to a per-subject verdict with nothing behind it —
excluding those outright would bin every distant subject in a folder.

## The Cache Is Read From Two Threads

`SQLiteCache` shares one connection, and the scan worker writes to it while the main
thread reads: Tools > Duplicate groups and Tools > Clear cached scan data are both
reachable during a scan.

So reads go through `_fetchall`/`_fetchone`, which hold the lock across execute **and**
fetch. `_execute` released it as soon as the statement was issued and handed back a
cursor, so the caller's `.fetchall()` ran unguarded — that returned truncated blobs
(`np.load` raising "No data left in file") and silently dropped writes. The lock is an
`RLock` so `clear()` can hold it across purge, close, unlink and schema rebuild; doing
the rebuild outside it left a window where a reader saw a database with no tables.

`GlobalLearningStore._load_index` hands back the live cache dict, so anything iterating
it must snapshot first — `record()` mutates it in place.

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

## Never Rewrite the User's Originals

`write_image_metadata` (`output_ops.py`) is the only code that writes back to a file the
user pointed us at, and it must never re-encode one. It used to decode the image, apply
its EXIF orientation and save it again at Pillow's default JPEG quality — so an action
called "write keyword tags" silently recompressed every original it touched, dropped the
ICC profile, and baked the rotation into the raster.

- **JPEG** is edited as a byte stream: the header segments are parsed, the existing APP1
  EXIF/XMP segments are replaced, and everything from the start-of-scan marker onwards is
  copied across untouched. `Image.open(...).getexif()` reads the old tags without
  decoding the raster.
- **PNG** may be re-saved, because PNG is lossless. Existing text chunks are carried
  across rather than replaced, and orientation is not applied.
- **TIFF and WebP** are only tagged when the optional `pyexiv2` is installed. Without it
  the file is left exactly as it is and the function returns False. Refusing is correct;
  re-encoding a WebP to attach a keyword would degrade the photo it was asked to
  annotate.

`tests/test_functional.py::OutputOperations` asserts the pixels are byte-identical after
tagging. Anything new that writes to a scanned file belongs under the same rule.

## Where Settings Live

`ScannerConfig` is persisted in the user preferences file under the `scanner_config` key
(`user_prefs.py` owns the file; `gui._save_user_prefs` writes the key and both
`run.main` and `gui.launch_gui` read it back). Without that round trip nothing in
Tools > Settings survived a restart.

It is `global_config` that is saved, never `config`: `config` carries the current
folder's override merged in, and a per-folder value must not be promoted into the
settings every other folder inherits. A folder override is stored with its folder and is
still filtered through `filter_folder_override` in both directions.

## Credentials Never Reach a File

`config_profiles.SECRET_KEYS` names the settings that are credentials rather than
configuration (`vlm_api_key` today). `without_secrets()` blanks them, and every path
that writes a config to disk goes through it: `_persistable_config` (prefs.json),
`save_profile` (profiles.json) and `export_settings` (a file the user chooses). All
three are things people copy between machines, sync, or send to someone.

The key therefore lives in memory for the session only. The complement matters too:
importing a settings file or applying a profile must **not** blank a key the user has
already entered, so both keep the session's key when the incoming value is empty.

## Bounded Model Cache

`backend_utils.remember_bounded` caps the loaded-backend caches at
`MAX_CACHED_BACKENDS` (2 — the scan model and the refine model). A CLIP model is 600 MB
and the high-accuracy one is 1.7 GB; unbounded, changing model in Settings and running
one refine pass left three of them resident and unreachable for the life of the
process. Eviction only drops the cache's reference — a backend still in use survives
through the caller's.

## A Live Store Outlives a Settings Change

The GUI builds a new `FolderStore` in `_set_folder`, i.e. only when the *folder*
changes. Changing the model in Settings clears the backend and the scorer and keeps the
store, so `ensure_embedding_namespace` runs against a store whose SQLite connection is
already open, mid-scan.

That means `SQLiteCache.clear()` has to leave a **usable, empty** database behind, not a
deleted file — it rebuilds the schema at the end for exactly this reason, and
`_discard_derived_caches` must not delete `cache.db` again afterwards. Getting this wrong
raised "no such table: image_records" on the next query and left the folder's cache
broken for the rest of the session. `tests/LiveStoreModelSwitch` covers it; note that a
test which constructs a fresh `FolderStore` before each call does **not** exercise this
path, because `__post_init__` recreates the schema.

## Important Conventions

- Do not commit the untracked `.devin/` or `.vscode/` directories; they are
  local workspace configuration.
- Do not add floating dependency versions; prefer packages published for at
  least 7 days.
- Avoid silently swallowing exceptions; log and surface errors.
- Run the verification commands above and review `git diff` before committing.
