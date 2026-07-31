---
name: testing-bikini-scanner-gui
description: How to run and end-to-end test the Bikini Scanner Tkinter desktop app, including the optional local vision-LLM (VLM) adjudication stage, without downloading a real multi-GB VLM.
---

# Testing the Bikini Scanner GUI

## Running the app
- `cd <repo> && PYTHONPATH=. python -m bikini_scanner` on a real X display (`DISPLAY=:0`). `python3-tk` is required.
- First launch downloads `openai/clip-vit-base-patch32` from Hugging Face (~1 min, no token needed) and opens an
  extra "Bikini Scanner guide" window — close it before recording, then maximize with
  `wmctrl -r "Bikini Scanner 1.3.0" -b add,maximized_vert,maximized_horz`.
- Unit suite: `xvfb-run -a env PYTHONPATH=. python tests/test_functional.py`.

## Where the logs are (important)
Almost nothing useful goes to stderr. Warnings/errors go to the rotating app log at
`~/.local/state/bikini-scanner/bikini_scanner.log` (see `logging_setup.py`). Always grep BOTH that file and the
process stdout/stderr for `Traceback` / `WARNING` when asserting "no errors".

## UI map (v1.3.x)
- Top bar: Folder entry (typing a path directly is faster and more reliable than the "Choose…" file dialog),
  "Run scan", "Stop" (enabled only during a scan), "Retrain", "Settings", "Theme".
- Settings dialog → "Detection pipeline" section holds the VLM controls: enable checkbox, VLM server URL, VLM model,
  VLM concurrency, VLM borderline band, VLM maximum images. Save shows
  "Settings saved. Run a new scan to apply them." A VLM settings change forces a full rescan, not a rescore.
- Status bar (bottom-left) is the assertion surface: "Scan complete — N images, M matches",
  "Stopping scan after the current batch...", "Scan stopped. Cached work is available for the next scan."

## Tk gotchas when driving with xdotool/computer-use
- Ctrl+A in a Tk `Entry` means "go to line start", NOT select-all. To clear a field use triple-click then
  `BackSpace`. Getting this wrong silently leaves text behind and can make a validation test falsely pass.
- After dismissing a modal, avoid an extra click at the same coordinates: it lands on the result grid's
  Accept/REJECT buttons underneath and silently labels images (changes later scores via the learned classifier).

## Testing the VLM stage without a real model
Stand up a tiny OpenAI-compatible fake server instead of pulling a multi-GB VLM. It needs:
- `GET /v1/models` → `{"data":[...]}` (used by `VLMClient.probe()`; if this fails the whole stage is skipped),
- `POST /v1/chat/completions` → `{"choices":[{"message":{"content":"{\"bikini\":0.9,...}"}}]}` with all ten axes
  (`person, female, child, adult, bikini, bikini_top, bikini_bottom, midriff, cleavage, nsfw`).
Log every request so you can assert on request counts; drive failure modes from a mode file the handler reads per
request (`normal` / `garbage` non-JSON / `slow` sleep) so you never have to restart the server mid-run.
`tests/test_functional.py::VLMAdjudication` has a minimal reference handler.

Useful tricks:
- Set **VLM borderline band = 1.0** in Settings so every image is in-band; otherwise a synthetic-image folder may
  produce zero VLM candidates and the test proves nothing.
- Generate innocuous synthetic images with PIL (solid colours, gradients, ellipses). Never source real photos.
- Strong differential assertion: the same folder scanned with VLM off vs on should change the match count
  (e.g. 0 matches → 10 matches) because the fake verdicts feed `cascade.evaluate`. A "scan completed" alone is weak.
- Verdict cache lives at `<folder>/.bikini_scanner_cache/vlm_verdicts.json`, keyed `<sha1>|<model>|vlm-json-v1`.
  Delete `.bikini_scanner_cache` to force cache misses between scenarios, or use a separate image folder per case.
- Cancellation is bounded by in-flight HTTP requests (ThreadPoolExecutor futures already dispatched must return), so
  after pressing Stop expect a wait of up to the per-request latency / `vlm_timeout` (default 60 s) before the
  status flips to "Scan stopped."

## Devin Secrets Needed
None. The app is fully local; Hugging Face downloads work unauthenticated (an `HF_TOKEN` would only raise rate limits).
