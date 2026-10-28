# SafeLens

Minimal review export of a Manifest V3 Chrome extension and local APIs for image observation and safe placeholder generation. The extension is observe-only here; weights are not included.

## Run

Install Python dependencies with `python -m pip install -r requirements/runtime.txt`. Install the browser runtime with `npm ci`, then run:

```powershell
powershell -ExecutionPolicy Bypass -File scripts/prepare_extension_vendor.ps1
python -m uvicorn generation_service.app:app --host 127.0.0.1 --port 8787
```

Load `extension/dist` as an unpacked Chrome extension. The optional Roboflow service requires `ROBOFLOW_API_KEY`, network access, and sends selected images to Roboflow:

```powershell
$env:ROBOFLOW_API_KEY = Read-Host "Roboflow API key"
powershell -ExecutionPolicy Bypass -File scripts/start_restricted_detector_service.ps1
```

The optional CUDA inpainting dependency snapshot is `requirements/generation-cu126.lock.txt`.

## Files and limits

Runtime instructions are above. Generated images and reports are written to `outputs/` and `reports/generation_service/`. This export contains no evaluation data or benchmark results, so it makes no performance claim. Required ONNX model files and vendor assets are absent; the extension will not classify images until compatible weights are supplied.

The extension requests access to all web pages. Local APIs bind to loopback and accept only configured local origins. Input images are restricted to `uploads/`; see `safelens_runtime_config.py`.

No project license is supplied. Do not redistribute until the owner selects one and checks the exact model terms. See [third-party notices](THIRD_PARTY_NOTICES.md). Dependencies: [ONNX Runtime Web](https://github.com/microsoft/onnxruntime), [Roboflow Inference SDK](https://inference.roboflow.com/quickstart/roboflow_ecosystem/), and optional [YOLOv8](https://docs.ultralytics.com/models/yolov8/).

## Parameters, metrics and response time

Code defaults, not validated recommendations: images up to 25 MiB and 40 megapixels; PNG, JPG/JPEG, WebP, and BMP. `/v1/generate-placeholder` defaults to 512×512 (`width`/`height` range 128–1024), style `soft_abstract`, and seed 42; it draws a local placeholder and is not model inference. The restricted detector defaults to the cigarette/vape model, hit confidence 0.35, and registry cleanup threshold 0.50; alcohol detection is experimental. Research inpainting defaults to dry-run (`execute=false`), 15 steps, `max_size=512`, box expansion 0.12, and mask blur radius 3.0. Category states and thresholds are in `config/restricted_category_registry.json`.

Timing fields are per-request server-side measurements, not fixed response guarantees or end-to-end browser times. `/v1/generate-placeholder` returns `latency_ms` for prompt/cache/placeholder work. `/v1/restricted/detect-local-file` returns `latency_seconds` for client creation and the Roboflow inference call; a single retry may add a 0.8-second wait. Executed `/v1/inpaint-local-research` can return detector, inpainting, and total latency plus peak VRAM; its default dry-run does not run the model.

This export contains no detector-quality benchmark metrics or measured end-to-end response times. Full service runtime and model inference were not run; performance figures are **NOT MEASURED**.
