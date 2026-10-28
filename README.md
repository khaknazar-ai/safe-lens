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
