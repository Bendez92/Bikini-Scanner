# Technical analysis & optimizations

This document explains how the scanner works, how it uses the hardware, and the
performance work done on it.

## How it works (pipeline)

1. **Collect** — `store.collect_image_paths` walks the chosen folder recursively
   and keeps supported image types.
2. **Embed** — `clip_backend.ClipBackend` runs each image through CLIP
   (`openai/clip-vit-base-patch32`) to get a 512-dim L2-normalized vector. Text
   prompts (positive/negative) are embedded once per session.
3. **Cache** — embeddings are persisted per-folder in
   `.bikini_scanner_cache/embeddings.npz`, keyed by a hash of
   `path + mtime + size`, so re-scans skip re-embedding unchanged files.
4. **Score** — `scorer.BikiniScorer`:
   - *Zero-shot*: cosine sim to positive vs negative prompts → sigmoid.
   - *Active learning*: once ≥6 labels exist with both classes, a
     `LogisticRegression` is trained on the CLIP vectors; final score =
     `0.7*classifier + 0.3*zero_shot`.
5. **Sample** — `bucketed_sampling` shows only the borderline images:
   *Uncertain* (closest to threshold), *Likely false positive*, *Likely false
   negative*.
6. **GUI** — Tkinter renders cards; labeling triggers an automatic retrain.

## How it uses the hardware

- **CPU-only inference.** No GPU is required; CLIP runs on CPU via PyTorch/MKL.
  Throughput is bound by matrix multiplies (ViT attention/MLP) and by image
  decode+resize (PIL).
- **Threads.** PyTorch uses intra-op threads for the matmuls; PIL releases the
  GIL during decode, so image decoding can run on worker threads in parallel
  with inference.
- **Disk.** The embedding cache is the only significant disk I/O; labels are a
  small JSON file.
- **Memory.** Embeddings are tiny (200 images ≈ 0.4 MB of float32), so the
  working set is dominated by the model weights (~600 MB) and decoded image
  batches.

## Optimizations applied

### Inference / CPU
- **Use all cores**: `torch.set_num_threads(os.cpu_count())` (+ interop threads)
  set at model load.
- **`torch.inference_mode()`** instead of `no_grad()` — fully disables autograd
  bookkeeping.
- **Parallel decode pipeline**: images are decoded on a `ThreadPoolExecutor`,
  and the *next* batch is decoded while the current batch runs through the
  model — decode latency is hidden behind inference instead of serialized before
  it.
- **Removed a redundant decode**: new images used to be opened twice (once for
  `Image.verify()`, once to embed). Now each image is decoded exactly once.

### Active-learning loop (the per-click hot path)
- **Fast in-memory rescore**: labeling used to re-glob the folder, re-read the
  cache, and rewrite the whole embedding archive on *every* Good/Bad click. Now,
  when the image set is unchanged, it reuses the in-memory embeddings and only
  reloads labels, retrains the classifier, and re-samples — no folder walk, no
  image decode, no embedding disk I/O.
- **Race guard**: a refresh generation counter drops stale background results if
  you click several times quickly.

### Caching subsystem
- **In-memory label cache** in `FolderStore` (labels were re-read from disk once
  per card on every render).
- **Write-only-when-changed** embeddings, and **uncompressed `np.savez`**
  instead of `savez_compressed` (compression cost dominated for these tiny
  arrays). Old compressed caches still load.

### GUI
- **Thumbnail cache**: decoded 400×400 `PhotoImage`s are cached by path and
  reused across re-renders (a re-render happens on every label click), instead
  of re-decoding from disk each time. Cleared when the folder changes.

### Numerical
## Packaged build size

Measured on the v1.2.0 build (`dist/BikiniScannerApp`, then the Inno Setup output):

| | bundle | installer |
|---|---|---|
| before | 882.8 MB | 214.6 MB |
| after | ~634 MB | ~162 MB |

What was removed, and why it was safe:

- **scikit-learn + scipy (~96 MB of binaries)** — replaced by `linear_model.py`, verified
  against the originals on real labels before the swap (rank correlation 0.99+).
- **`torch/include` (37 MB) and every `*.lib` (39 MB)** — C++ headers and static link
  libraries, used only when compiling against torch.
- **OpenCV's ffmpeg DLL (29 MB) and libx265 (21 MB)** — video I/O and HEIC *encoding*;
  this app only ever decodes.

What was tried and reverted, because the packaged app broke while the source tree
stayed fine:

- **excluding `unittest`** — `import torch` imports it, so the window opened and the
  model silently never loaded.
- **excluding `hf_xet`** — `huggingface_hub` imports it directly.
- **dropping transformers' loose `.py` files** — PyInstaller's hook collects them as
  data *instead of* putting them in the archive, so they are the package itself.

Still large and not reducible without losing features: torch (364 MB) and OpenCV
(82 MB, one monolithic `.pyd`; removing it would remove face-anchored regions).

- Zero-shot sigmoid uses `linear_model.sigmoid`, a branch-on-sign implementation that
  cannot overflow. scikit-learn and scipy were dropped entirely (~96 MB of binaries in
  the packaged build) in favour of `linear_model.py`, which implements the logistic
  regression, standardisation, Platt calibration, stratified folds, and ROC AUC this
  app actually used — verified against the originals before the swap
  (values unchanged).

## Benchmark (200 synthetic images, CPU)

| Path | Before | After | Improvement |
|------|--------|-------|-------------|
| Cold scan (embed all 200) | 3.185 s | 2.552 s | ~20% faster |
| Warm rescore (per label click) | 0.061 s | 0.044 s | ~28% faster |

Score distribution (min/median/max) was identical before and after, so these are
pure speedups with no change to results. Gains scale with folder size and core
count; the parallel-decode and in-memory-rescore wins grow with larger folders.

## Deliberately not done (risk/complexity vs. benefit)
- **int8 dynamic quantization** of the model: can speed up CPU Linear layers but
  can shift scores; left out to keep results identical. Easy to add behind a
  flag if you want to trade a little accuracy for speed.
- **`torch.compile`**: adds a large first-run compile cost and packaging
  complexity for modest CPU gains.
- **GPU support**: not applicable to the current CPU-only target.
