# NOTE: The CLIP model weights are not bundled. Transformers downloads them on first launch
# into the user's Hugging Face cache, so the packaged EXE needs internet the first time it runs.
#
# This is a onedir build: COLLECT emits dist/BikiniScannerApp/, which installer.iss packages.
# Onefile would re-extract the payload to %TEMP% on every launch.
#
# Size discipline (the bundle is dominated by torch, so everything else has to earn its place):
#   * Pure-python packages are NOT collect_all'd. PyInstaller compiles imported modules into
#     the embedded archive already; collect_all additionally copies the .py sources into
#     _internal, so every module ends up in the build twice. That duplication alone was
#     ~58 MB (transformers 44.6 MB + sklearn 13.1 MB of .py files).
#   * PRUNE_SUFFIXES/PRUNE_PARTS drop files that can never be used at runtime: C++ headers,
#     static link libraries, and codec DLLs for video/HEIC encoding this app never does.
#   * scikit-learn and scipy are gone entirely - bikini_scanner.linear_model implements the
#     handful of primitives that were used, in numpy.

from pathlib import Path

from PyInstaller.utils.hooks import collect_all, collect_data_files, collect_submodules

block_cipher = None
ROOT = Path(SPECPATH).resolve()

# Packages with real data files or compiled extensions still need collecting.
pillow_datas, pillow_bins, pillow_hidden = collect_all("PIL")
pillow_heif_datas, pillow_heif_bins, pillow_heif_hidden = collect_all("pillow_heif")
psutil_datas, psutil_bins, psutil_hidden = collect_all("psutil")
safetensors_datas, safetensors_bins, safetensors_hidden = collect_all("safetensors")
tokenizers_datas = collect_data_files("tokenizers")
torch_datas, torch_bins, torch_hidden = collect_all("torch")
try:
    tkinterdnd2_datas, tkinterdnd2_bins, tkinterdnd2_hidden = collect_all("tkinterdnd2")
except Exception:
    tkinterdnd2_datas, tkinterdnd2_bins, tkinterdnd2_hidden = [], [], []
try:
    send2trash_datas, send2trash_bins, send2trash_hidden = collect_all("send2trash")
except Exception:
    send2trash_datas, send2trash_bins, send2trash_hidden = [], [], []

# transformers is pure python: let the bundled hook + these hidden imports pull the modules
# into the archive instead of copying 2,500 source files next to the exe.
hiddenimports = sorted(
    {
        *pillow_hidden,
        *pillow_heif_hidden,
        *psutil_hidden,
        *safetensors_hidden,
        *tkinterdnd2_hidden,
        *send2trash_hidden,
        *torch_hidden,
        *collect_submodules("transformers.models.clip"),
        "transformers",
        "transformers.models.auto",
        "transformers.models.auto.image_processing_auto",
        "transformers.models.auto.processing_auto",
        "transformers.models.auto.tokenization_auto",
    }
)

datas = [
    (str(ROOT / "assets"), "assets"),
    *pillow_datas,
    *pillow_heif_datas,
    *psutil_datas,
    *safetensors_datas,
    *tokenizers_datas,
    *tkinterdnd2_datas,
    *send2trash_datas,
    *torch_datas,
]
# Conditionally bundle exported ONNX graphs so the clip-onnx backend works in a
# packaged build without a manual copy. Only included when the models have been
# exported (python -m scripts.export_onnx); absent models don't break the build.
_onnx_model_dir = ROOT / "models"
if (_onnx_model_dir / "clip_vision.onnx").is_file() and (_onnx_model_dir / "clip_text.onnx").is_file():
    datas.append((str(_onnx_model_dir), "models"))
binaries = [
    *pillow_bins,
    *pillow_heif_bins,
    *psutil_bins,
    *safetensors_bins,
    *tkinterdnd2_bins,
    *send2trash_bins,
    *torch_bins,
]

# Never imported by this app or its dependencies.
#
# Do NOT add "unittest", "test" or "pydoc_data" here: torch imports unittest during
# `import torch`, so excluding it produces a build whose window opens but whose model
# never loads. Do not exclude "hf_xet" either - huggingface_hub imports it directly.
# Both were tried, and both broke the packaged app while leaving the source tree fine.
# Kept deliberately short: only packages this app genuinely dropped. Excluding things
# that simply are not installed buys nothing and risks upsetting libraries that probe
# for optional modules with find_spec (torch._dynamo does exactly that for onnx).
EXCLUDES = [
    "scipy",
    "sklearn",
    "joblib",
    "threadpoolctl",
    "tkinter.test",
]

# Link-time payloads and the video codec. .lib files exist only to compile against
# torch, and the OpenCV ffmpeg DLL decodes video, which this app never opens.
#
# libx265 is NOT pruned despite being a 21 MB encoder this app never invokes: libheif
# links it statically, so removing it makes _pillow_heif fail to load and takes all
# HEIC/HEIF support with it. That was tried, and the packaged app logged
# "Unable to register HEIF opener: DLL load failed" on every start.
PRUNE_SUFFIXES = (".lib", ".a", ".h", ".hpp", ".cuh", ".pdb")
PRUNE_PARTS = (
    "torch/include/",
    "torch\\include\\",
    "torch/test/",
    "torch\\test\\",
    "opencv_videoio_ffmpeg",
)


def _keep(entry) -> bool:
    destination = str(entry[0]).replace("\\", "/")
    lowered = destination.lower()
    if lowered.endswith(PRUNE_SUFFIXES):
        return False
    return not any(part.replace("\\", "/") in lowered for part in PRUNE_PARTS)


datas = [entry for entry in datas if _keep(entry)]
binaries = [entry for entry in binaries if _keep(entry)]

a = Analysis(
    [str(ROOT / "run_app.py")],
    pathex=[str(ROOT)],
    binaries=binaries,
    datas=datas,
    hiddenimports=hiddenimports,
    hookspath=[],
    hooksconfig={},
    runtime_hooks=[],
    excludes=EXCLUDES,
    noarchive=False,
)

# Analysis adds its own discoveries, so filter once more after it has run.
a.datas = [entry for entry in a.datas if _keep(entry)]
a.binaries = [entry for entry in a.binaries if _keep(entry)]

pyz = PYZ(a.pure, a.zipped_data, cipher=block_cipher)

exe = EXE(
    pyz,
    a.scripts,
    [],
    exclude_binaries=True,
    name="BikiniScanner",
    icon=str(ROOT / "assets" / "bikini_scanner.ico"),
    console=False,
    debug=False,
    strip=False,
    upx=False,
    disable_windowed_traceback=False,
    argv_emulation=False,
    target_arch=None,
    codesign_identity=None,
    entitlements_file=None,
)

coll = COLLECT(
    exe,
    a.binaries,
    a.zipfiles,
    a.datas,
    strip=False,
    upx=False,
    upx_exclude=[],
    name="BikiniScannerApp",
)
