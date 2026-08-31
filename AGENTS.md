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

- 83 functional tests pass.
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

## Important Conventions

- Do not commit the untracked `.devin/` or `.vscode/` directories; they are
  local workspace configuration.
- Do not add floating dependency versions; prefer packages published for at
  least 7 days.
- Avoid silently swallowing exceptions; log and surface errors.
- Run the verification commands above and review `git diff` before committing.
