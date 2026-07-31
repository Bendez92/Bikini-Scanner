# Feature status

Audit of the requested feature list and what the program now does.

## Performance
- **Multiprocessing/Threading (I/O ↔ inference)** — ✅ Images are decoded on a
  thread pool; the *next* batch decodes while the model runs the current batch,
  keeping the CPU busy instead of idling on disk reads.
- **Batch Size Tuning** — ✅ Batching was always used; batch size is now
  configurable via the in-app Settings dialog (`batch_size`, default 16).
- **Model Export (ONNX / TensorRT)** — ✅ ONNX (opt-in). An ONNX backend
  (`clip-onnx`) + export script are included; on our tests ONNX matched the
  PyTorch scores to machine precision. **TensorRT is NVIDIA-GPU-only and does
  not apply to a CPU machine, so it is intentionally not included.** See
  "Enabling ONNX" below.
- **Error Handling (Validation)** — ✅ Each file load is guarded; a corrupt or
  unreadable image is logged and skipped (and recorded in the metadata audit
  trail) without stopping the scan.

## UX & Workflow
- **Dynamic Confidence Thresholding** — ✅ Threshold lives in the in-app Settings dialog and
  `--threshold`, and is still adjustable live via the GUI slider. Other tunables
  (prompts, batch size, blend weights, scale, device, precision, quantization,
  preload) are also in the config.
- **UI Preferences** — ✅ Theme (light/dark/system), font size, grid columns,
  thumbnail size, bounded thumbnail-cache size, and recent folders persist in a
  user-level prefs file outside the repo.
- **Watch-folder efficiency** — ✅ Watch mode compares a lightweight
  path/mtime/size snapshot and only launches a rescan when image files change;
  it does not launch a scan on every polling tick.
- **Large HTML reports** — ✅ Small reports remain self-contained with embedded
  thumbnails. Reports above the embedding limit automatically use a sibling
  assets directory with relative thumbnail references to bound HTML size and
  memory usage.
- **Checkpointing / Resume** — ✅ Embeddings are cached and now flushed
  incrementally during a scan, so a killed/paused run resumes from where it left
  off instead of re-embedding everything.
- **Scan cancellation** — ✅ GUI scans expose a cooperative **Stop scan**
  control, and headless scans honor `Ctrl-C`; completed batches are flushed
  before stopping.
- **Cache Management** — ✅ A Clear cache control wipes the current folder's
  embedding/label/metadata/classifier cache on demand.
- **Auto-Categorization / Move** — ✅ "Copy matches to subfolder" copies matches
  into `<folder>/bikini_matches`; "Export matches" writes a CSV + copies files.
  A **Move instead of copy** checkbox switches either action to move (with a
  confirmation, since moving is destructive).
- **Output Organization** — ✅ Output can be grouped by score band and/or
  label, with filename templates, duplicate policies, a dry-run preview, HTML
  contact-sheet export, metadata tagging, and recycle-bin trash support.
- **Headless output parity** — ✅ `--html-report`, `--dry-run`, and
  `--write-metadata` expose the corresponding GUI output operations to
  automation.
- **Reliability / Recovery** — ✅ Rotating user-level logs, an in-app log
  viewer, incremental large-folder cache flushing, explicit resume-last-scan,
  and exact duplicate groups with keep-one/trash-the-rest support.
- **Configurability** — ✅ In-app settings profiles, explicit settings
  import/export, per-folder configuration overrides, a headless CSV/JSON scan
  mode, and optional post-processing plugins discovered from the user data
  directory; plugins post-process the review sample list before rendering and
  headless export.
- **Distribution / Polish** — ✅ Versioned About/title, guarded application
  icon assets, standard and ONNX-only PyInstaller specs, Windows installer
  script, opt-in update checking, and optional Authenticode signing hooks.

## Feature Expansion
- **Staged cascade pipeline** — ✅ Scanning runs as gates rather than one verdict:
  people → female subject → age exclusion → bikini/cleavage/midriff detail
  (`cascade.py`). Each gate both scales the score and can exclude outright. Age
  exclusion is forced to zero and dropped from every view, so no threshold or
  filter setting can surface a flagged image.
- **Region (deep) scan** — ✅ Candidate images are re-scored on face and body-region
  crops (`regions.py`) instead of a single shrunken frame, which is what makes
  cleavage and midriff detectable at all. Crops are cached per content hash, model,
  and geometry version, so re-scans skip the work.
- **Crops vote by position and by confidence** — ✅ A crop only contributes to axes its
  position supports (the lower band cannot evidence cleavage; the upper band cannot
  evidence a bare midriff), and unanchored fallback bands get only half the distance
  they claim above the full frame. Taking a plain max over four guessed bands was the
  main source of false positives with no face model installed: on a real folder the
  files above the threshold fell from 12 to 7 with every accepted image retained.
- **Multi-Axis Scoring** — ✅ bikini, bikini top, bikini bottom, midriff, cleavage,
  NSFW, person, female, child, and adult axes, each a prompt ensemble. Axis scores
  are sigmoids centred on 0.5; gates and the UI compare *evidence*, `(s-0.5)*2`.
- **NSFW / Person Filters** — ✅ The review buckets and match/copy/export set respect the in-app NSFW filter mode plus the optional person gate.
- **Face Detection** — ✅ YuNet (`cv2.FaceDetectorYN`) when the model is installed
  from Settings; region planning falls back to fixed body bands without it.
  (OpenCV 5 removed `CascadeClassifier` and ships no Haar XML, so the previous
  Haar-based counting silently detected nothing.)
- **Learning that compounds** — ✅ Accept/REJECT trains on whole-image + best-crop
  embeddings plus axis scores. A prototype model works from the first label, a
  cross-validated logistic model takes over as evidence accumulates, and its share
  of the final score is set by its own measured AUC rather than a fixed weight
  (`learning.py`). Labels pool across folders in the user data directory
  (`global_store.py`), and the review queue surfaces the images where the learned
  model and the prompts most disagree.
- **Output Metadata (JSON/CSV)** — ✅ Every scan writes
  `.bikini_scanner_cache/scan_metadata.json` with filename, path, final score,
  zero-shot score, matched flag, and ISO timestamp per image (plus a list of
  skipped/unreadable files). The export CSV now includes path, filename, score,
  and timestamp.
- **Photo keyword metadata** — ✅ JPEG files receive EXIF-compatible keywords
  plus XMP `dc:subject` and `lr:hierarchicalSubject`; PNG keeps text chunks.
  TIFF uses optional `pyexiv2` XMP support when available and otherwise falls
  back to Pillow metadata.
- **Modular Model Loader (Strategy pattern)** — ✅ Backends implement a common
  interface and are selected by `config.backend` through a factory, so a new
  model can be added as one class + registry entry without touching the scanner
  or GUI. `clip-torch` (default) and `clip-onnx` ship today.
- **Hash-Based De-duplication** — ✅ Each file is content-hashed; identical files
  (even with different names) share a single embedding and are embedded only
  once, and re-scans reuse cached embeddings by content hash.
- **Human-in-the-Loop (Review UI)** — ✅ The whole app: review borderline
  buckets (uncertain, likely false positives/negatives, model disagreements), mark
  Accept/REJECT, and the model retrains and re-ranks immediately.
- **Gallery / Viewer UX** — ✅ The review area is a responsive grid with a
  full-size viewer, path/label/bucket search, score-range filtering, sort,
  filter, and reveal/open actions.
- **Hardware Monitoring** — ✅ Live CPU%, system RAM%, and this process's memory
  (RSS) in the status bar (via `psutil`). GPU temp/VRAM don't apply on a
  CPU-only target. Auto-throttling/pausing is intentionally **not** implemented —
  forcibly killing a scan risks data loss and the OS already manages thermals.
- **Format Normalization** — ✅ WebP/TIFF/BMP/GIF already worked; **HEIC/HEIF**
  is now supported via `pillow-heif`, and all images are converted to RGB before
  inference. **RAW** (CR2/NEF/etc.) needs a heavy extra library and is out of
  scope.
- **Process Notification** — ✅ When a full scan finishes, the app rings the
  system bell, updates the status to "Scan complete — N images, M matches", and
  shows a summary dialog. The per-click retrain stays silent.
- **Scan progress** — ✅ A determinate bar plus a live count line above it
  ("Reading images — 1,204 / 4,096 files · 12.4/s · ETA 03:53"). Every phase of a
  scan reports — the embedding pass, the body-region pass, the optional second
  opinion, and scoring — and each owns a fixed slice of the bar, so the percentage
  never walks backwards when a later phase discovers more work. Model loading and
  retrains, whose size is not known in advance, show an indeterminate bar instead of
  a made-up percentage.

## Enabling ONNX (optional, faster CPU inference)
1. `pip install -r requirements-onnx.txt`
2. `python -m scripts.export_onnx` (creates `models/clip_vision.onnx` and
   `models/clip_text.onnx`)
3. Open **Settings** in the app and set the backend to `clip-onnx`.

If ONNX files or `onnxruntime` are missing when `clip-onnx` is selected, the app
shows a clear error and you can switch back to the default `clip-torch`.

The app also includes Help > Guide for a built-in workflow overview, plus
optional folder drag-and-drop if `tkinterdnd2` is installed.
