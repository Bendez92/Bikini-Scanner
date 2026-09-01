# Bikini Scanner

Local desktop app for scanning a folder of images for bikini-related content:

- bikini top
- bikini bottom
- bare midriff / exposed stomach

The app only works on a folder you choose locally. It does not scrape, download, or fetch images from the network.

## Features

- Tkinter GUI with dark and light themes, collapsible advanced controls, and an
  enlarged active picture you can Accept or REJECT from the keyboard
- Staged scan pipeline: people → female subject → age exclusion → bikini / cleavage /
  midriff detail, with each stage both ranking and able to exclude
- Face and body-region crops for candidate images, so small subjects are scored at a
  useful resolution instead of a shrunken whole frame
- CLIP zero-shot scoring with `openai/clip-vit-base-patch32`, plus an opt-in
  high-accuracy re-check of borderline images with `openai/clip-vit-large-patch14`
- Learning that starts at the first label and pools across folders, weighted by its
  own measured cross-validated AUC
- Cached embeddings and region crops so rescans are fast
- Review buckets for uncertain images, likely false positives/negatives, and the
  images where the learned model most disagrees with the prompts
- In-app Settings for prompts, backend, device/precision, thresholds, gates and filters
- HEIC/HEIF support when `pillow-heif` is installed

### Age exclusion

Images that read as showing a minor are forced to a zero score and dropped from every
view, so no threshold or filter setting can surface them. The gate errs toward
excluding: it fires on absolute child evidence, on a child reading that outweighs the
adult reading, and on missing adult evidence for anything that would otherwise be a
match. It can be tuned or switched off in Settings.

## Requirements

- Python 3.10 or newer (developed and packaged on 3.12)
- `python3-tk` from the system package manager

Install Tkinter on Ubuntu:

```bash
sudo apt-get install -y python3-tk
```

## Install

```bash
python3 -m venv .venv
source .venv/bin/activate
pip install --upgrade pip
pip install -r requirements.txt
```

The first run downloads CLIP weights from Hugging Face.
Optional extras:

- `requirements-onnx.txt` for the experimental ONNX backend
- `pillow-heif` enables HEIC/HEIF image support
- `psutil` enables live CPU/RAM usage in the GUI
- `pyexiv2` is optional; when installed, TIFF keyword writes use XMP metadata

## Run

```bash
python -m bikini_scanner
```

You can also view the command-line help:

```bash
python -m bikini_scanner --help
```

## How a scan works

1. Every image is embedded once with CLIP and scored on the whole frame.
2. Images that could contain a person go through the **deep pass**: face and body-region
   crops are embedded and scored separately, so a bikini top is judged at a useful size.
3. Each crop only votes on the axes its position can support, and by how much. A crop
   anchored to a detected face sits where the geometry says it does and votes in full;
   the fallback bands used when no face model is installed are guesses, so the bottom
   band cannot claim "cleavage" at all and any band gets only half the distance it
   claims above the full-frame score (`UNANCHORED_CROP_SHARE` in `cascade.py`).
4. The **cascade** (`cascade.py`) turns those scores into gates — people, female subject,
   age exclusion — and then combines the bikini / cleavage / midriff / top / bottom axes.
   Axis scores are sigmoids centred on 0.5, so the gates compare *evidence*, `(s-0.5)*2`.
5. The learned model blends in on top, weighted by its own cross-validated AUC.

The scan reports progress per phase (`Reading images`, `Checking body regions`,
`Second opinion`, `Scoring`), each phase owning a fixed slice of the bar so the
percentage only moves forwards. Pass a one-argument `progress_callback` to
`scan_and_score_folder` to receive a `ScanProgress`; the older two- and four-argument
signatures still work.

## How the learning loop works

1. Accept/REJECT decisions are saved per folder and pooled in your user data directory.
2. From the very first label, a prototype model (accepted centroid vs rejected centroid)
   starts re-ranking; a cross-validated logistic model takes over as evidence builds.
3. Training features are the whole-image embedding, the best body-crop embedding, and
   the axis scores — so a label teaches the model about the swimwear, not the beach.
4. The review queue prioritises the images where the learned model and the prompts
   disagree, because those labels resolve the most uncertainty.
5. **Retrain** re-ranks immediately; a rescore never re-embeds anything.

Labels whose files no longer exist are pruned automatically, and
**Tools → Reset cross-folder learning** clears the pool without touching per-folder labels.

## Optional local vision-LLM adjudication

CLIP remains the fast first pass. An optional local vision-language model can then
judge only the small band of images near the threshold, plus uncertain age calls.
Requests run in parallel, include the full frame and the best body crop, and are
cached by image hash, model, and prompt version. A cheap skin-colour estimate only
prioritises images within the eligible band; it never excludes an image.

The stage is off by default and needs an OpenAI-compatible local server. Ollama is
the simplest option:

```bash
ollama pull qwen2.5vl:7b
ollama serve
```

Enable **Use local vision-LLM adjudication** in Settings, leave the server URL at
`http://localhost:11434/v1`, and choose the model served by Ollama. The same API
works with `llama-server` from llama.cpp. This improves borderline accuracy at the
cost of extra local inference; an unreachable server is detected once and the
ordinary CLIP result is retained.

## Cache and labels

Each scanned folder stores:

- an embedding cache under `.bikini_scanner_cache/`
- cached region crops, keyed by content hash, model, and crop-geometry version
- label persistence as JSON so your Accept/REJECT choices survive restarts
- a saved classifier, metadata log, and face-count cache when available

Use **Clear cache** in the app to wipe the current folder's cache and force a
fresh rebuild on the next scan.

User-level UI preferences are stored outside the repo at:
`~/.config/bikini-scanner/prefs.json` on Linux, `~/Library/Application Support/bikini-scanner/prefs.json`
on macOS, or `%APPDATA%\bikini-scanner\prefs.json` on Windows.

Reliability logs are stored outside the repo at
`~/.local/state/bikini-scanner/bikini_scanner.log` on Linux (with equivalent
user-level locations on macOS and Windows). The Help menu includes a recent
log viewer. The Tools menu also provides exact duplicate groups and a
keep-one/trash-the-rest action.

Settings profiles are managed entirely in the in-app Settings profiles
dialog and are stored under the same user configuration directory as the UI
preferences. Settings can also be explicitly imported/exported through the
GUI file chooser; no settings file is loaded automatically.

Headless example:

```bash
python run_app.py --headless --folder /path/to/images \
  --format json --output /tmp/results.json --profile Strict
```

To copy matched files without opening the GUI:

```bash
python run_app.py --headless --folder /path/to/images \
  --format csv --output /tmp/results.csv --organization score_band --copy
```

Headless output extras are opt-in:

```bash
python run_app.py --headless --folder /path/to/images \
  --format json --output /tmp/results.json \
  --html-report /tmp/report.html --write-metadata
```

Preview a copy plan without changing files:

```bash
python run_app.py --headless --folder /path/to/images \
  --output /tmp/results.json --dry-run
```

Headless scans flush their embedding cache incrementally. Press `Ctrl-C` to
stop after the current cooperative batch; the next scan resumes from cached
embeddings. GUI scans expose the same behavior through **Stop scan**.

For JPEG files, metadata tagging writes the existing EXIF keyword-compatible
field plus XMP `dc:subject` and Lightroom
`lr:hierarchicalSubject`. PNG files retain their PNG text metadata fallback.
TIFF files use `pyexiv2` XMP fields when that optional package is already
installed, otherwise they retain the Pillow EXIF fallback.

Optional plugins live under the user configuration directory's `plugins/`
folder (Linux: `~/.config/bikini-scanner/plugins/`). Each `.py` module may
define `process_results(state, samples)`, returning a replacement sample list
or `None`. The returned list is what the review grid renders and what the
headless CSV/JSON export writes; plugin exceptions are logged and skipped.

The application version is `1.0.0`. The Windows build is produced by
`make_installer.ps1` and is unsigned by default. To opt into Authenticode
signing, provide the user's own certificate and password through:

```text
BIKINI_SIGN_PFX=C:\path\to\certificate.pfx
BIKINI_SIGN_PASS=certificate-password
```

When both are set, the app executable and the installer are signed.

The smaller ONNX-only build is available through `build_onnx.bat` and
`bikini_scanner_onnx.spec`. It excludes torch and transformers, requires the
exported ONNX graphs from `scripts/export_onnx.py` in `models/`, and does not
download the Torch CLIP model.

Update checks are opt-in. Set an update manifest URL in the in-app Settings
dialog, then use Help > Check for updates. The default URL is empty, no
network request is made by default, and updates are never downloaded or
installed automatically.

## Tests

```bash
PYTHONPATH=. python tests/test_functional.py
PYTHONPATH=. python -m pytest
```

The end-to-end suite covers the scan pipeline and caching, the cascade gates
(including that age-excluded images are forced to zero and stay invisible), the
learning loop and its cross-folder store, the numpy learning primitives, review
sampling, export/report/metadata output, configuration round-tripping, and
robustness against unreadable images, empty folders, and corrupt caches.

The suite redirects every user-state path into a temporary directory first, so
running it never touches your real preferences, labels, or learning memory. Pass a
class name to run one group, e.g. `python tests/test_functional.py NumericPrimitives`.

## Optional ONNX backend

The default backend is `clip-torch`. If you want to export and use the optional
ONNX path:

1. Install the extra dependencies from `requirements-onnx.txt`.
2. Export the towers:

   ```bash
   python -m scripts.export_onnx
   ```

3. Open **Settings** in the app and set the backend to `clip-onnx`.

The exported graphs live under `models/` and the ONNX backend still uses the
same CLIP preprocessing. RAW formats such as CR2/NEF are still out of scope.

The Settings dialog also exposes:

- device selection: auto / cpu / cuda
- precision: auto / fp32 / fp16
- CPU int8 quantization toggle
- backend preload on startup

The app also has a built-in guide under Help, recent folders in File, optional
folder drag-and-drop when `tkinterdnd2` is installed, and UI prefs for theme,
font size, grid columns, thumbnail size, and the bounded thumbnail-cache size.
Watch mode keeps a lightweight path/mtime/size snapshot and rescans only when
an image is added, removed, or changed. Large HTML reports automatically write
thumbnail assets beside the report instead of embedding every image in the HTML
(the default embedding limit is 512 thumbnails); small reports remain
self-contained.

## Installing on Windows

Run `installer-output\BikiniScannerSetup.exe` and follow the wizard. It installs
to Program Files, adds a Start menu entry (desktop shortcut optional), and
registers a normal entry under Add or Remove Programs. Python does not need to
be installed on the target machine.

The installer is unsigned unless built with a certificate, so SmartScreen shows
a "Windows protected your PC" warning on first run. Choose **More info** >
**Run anyway**.

Uninstalling removes the program files only. Scan caches live beside the scanned
images and preferences live in `%APPDATA%\bikini-scanner`, so both survive an
uninstall and can be deleted by hand if wanted.

## Building the Windows installer

Prerequisites:

- Windows 10 or later, 64-bit
- Python from [python.org](https://www.python.org/downloads/)
- Internet access for the first launch, so Hugging Face can download the CLIP model weights

From a PowerShell prompt in the project directory:

```powershell
.\make_installer.ps1
```

Or double-click `make_installer.bat` from File Explorer.

The script:

- creates or reuses a local `.venv`
- installs runtime and build dependencies
- runs PyInstaller with `bikini_scanner.spec` (onedir)
- installs Inno Setup through winget if it is missing
- compiles `installer.iss`

Expected output:

- `installer-output\BikiniScannerSetup.exe`

Useful flags when iterating:

```powershell
.\make_installer.ps1 -SkipDeps    # .venv already provisioned
.\make_installer.ps1 -SkipBuild   # recompile the installer only
```

Notes:

- the build is large, roughly 930 MB on disk, because it bundles PyTorch and related binaries
- the bundle is onedir rather than onefile, so the app starts immediately instead of unpacking to `%TEMP%` on every launch
- the CLIP model weights are not bundled; the first launch downloads them from Hugging Face, roughly 600 MB
