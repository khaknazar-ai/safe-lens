from __future__ import annotations

import mimetypes
import random
import threading
import time
from pathlib import Path

from fastapi import FastAPI, HTTPException, Request
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import FileResponse
from PIL import Image, ImageDraw

from generation_service.cache import get_output_path, make_cache_key
from generation_service.inpainting_backend import run_detector_guided_inpainting
from safelens_runtime_config import (
    ALLOWED_BROWSER_ORIGINS,
    resolve_input_file,
    resolve_report_file,
    safe_output_stem,
)
from generation_service.prompt_templates import build_negative_prompt, build_safe_placeholder_prompt
from generation_service.schemas import (
    GenerationMetadataRequest,
    GenerationResponse,
    InpaintResearchRequest,
    InpaintResearchResponse,
)


ROOT = Path(__file__).resolve().parents[1]

IMAGE_SCOPES = {
    "generation": ROOT / "outputs" / "generation",
    "generation_service": ROOT / "outputs" / "generation_service",
}

ALLOWED_IMAGE_SUFFIXES = {".png", ".jpg", ".jpeg", ".webp"}

INPAINTING_LOCK = threading.Lock()

app = FastAPI(
    title="SafeLens Local Generation Service",
    version="0.3.0",
    description="Local generation API boundary for SafeLens replacement images and research inpainting.",
)


app.add_middleware(
    CORSMiddleware,
    allow_origins=list(ALLOWED_BROWSER_ORIGINS),
    allow_credentials=False,
    allow_methods=["*"],
    allow_headers=["*"],
    max_age=600,
)


@app.get("/health")
def health() -> dict[str, str]:
    return {
        "status": "ok",
        "service": "safelens-generation-service",
        "mode": "local",
    }


def validate_image_filename(filename: str) -> str:
    candidate = Path(filename)

    if candidate.name != filename:
        raise HTTPException(status_code=400, detail="Nested paths are not allowed.")

    suffix = candidate.suffix.lower()
    if suffix not in ALLOWED_IMAGE_SUFFIXES:
        raise HTTPException(status_code=400, detail=f"Unsupported image suffix: {suffix}")

    return filename


def resolve_served_image_path(image_scope: str, filename: str) -> Path:
    if image_scope not in IMAGE_SCOPES:
        raise HTTPException(status_code=404, detail=f"Unknown image scope: {image_scope}")

    safe_filename = validate_image_filename(filename)
    base_dir = IMAGE_SCOPES[image_scope].resolve()
    image_path = (base_dir / safe_filename).resolve()

    if base_dir not in image_path.parents and image_path != base_dir:
        raise HTTPException(status_code=400, detail="Resolved path escaped allowed image directory.")

    if not image_path.exists():
        raise HTTPException(status_code=404, detail=f"Image not found: {image_scope}/{safe_filename}")

    return image_path


@app.get("/v1/images/{image_scope}/{filename}", name="serve_image")
def serve_image(image_scope: str, filename: str) -> FileResponse:
    image_path = resolve_served_image_path(image_scope, filename)
    media_type = mimetypes.guess_type(str(image_path))[0] or "application/octet-stream"

    return FileResponse(
        path=image_path,
        media_type=media_type,
        filename=image_path.name,
    )


def build_image_url(request: Request, image_scope: str, image_path: str | Path | None) -> str | None:
    if not image_path:
        return None

    filename = Path(image_path).name
    return str(request.url_for("serve_image", image_scope=image_scope, filename=filename))


def create_safe_placeholder_image(request: GenerationMetadataRequest, output_path: Path) -> None:
    rng = random.Random(request.seed)

    image = Image.new("RGB", (request.width, request.height), (245, 247, 250))
    draw = ImageDraw.Draw(image)

    for _ in range(18):
        x1 = rng.randint(-request.width // 4, request.width)
        y1 = rng.randint(-request.height // 4, request.height)
        x2 = x1 + rng.randint(request.width // 4, request.width)
        y2 = y1 + rng.randint(request.height // 8, request.height // 2)

        shade = rng.randint(205, 245)
        draw.ellipse(
            [x1, y1, x2, y2],
            fill=(shade, min(255, shade + 5), min(255, shade + 10)),
        )

    margin = max(16, min(request.width, request.height) // 14)
    card = [
        margin,
        request.height // 3,
        request.width - margin,
        request.height - request.height // 4,
    ]

    draw.rounded_rectangle(card, radius=24, fill=(255, 255, 255), outline=(210, 220, 235), width=2)

    title = "SafeLens"
    subtitle = f"{request.category} / {request.severity}"
    note = "local safe replacement"

    draw.text((card[0] + 24, card[1] + 24), title, fill=(35, 45, 65))
    draw.text((card[0] + 24, card[1] + 56), subtitle[:42], fill=(80, 95, 120))
    draw.text((card[0] + 24, card[1] + 88), note, fill=(110, 125, 145))

    output_path.parent.mkdir(parents=True, exist_ok=True)
    image.save(output_path)


@app.post("/v1/generate-placeholder", response_model=GenerationResponse)
def generate_placeholder(payload: GenerationMetadataRequest, request: Request) -> GenerationResponse:
    started_at = time.perf_counter()

    if payload.unsafe_pixels_used:
        raise HTTPException(
            status_code=400,
            detail="Production-safe placeholder generation must not receive unsafe pixels.",
        )

    request_payload = payload.model_dump()
    prompt = build_safe_placeholder_prompt(payload)
    negative_prompt = build_negative_prompt()

    cache_key = make_cache_key(
        {
            "endpoint": "generate-placeholder",
            "request": request_payload,
            "prompt": prompt,
            "negative_prompt": negative_prompt,
        }
    )

    output_path = get_output_path(cache_key)

    if not output_path.exists():
        create_safe_placeholder_image(payload, output_path)

    latency_ms = round((time.perf_counter() - started_at) * 1000)

    return GenerationResponse(
        status="completed",
        backend="safe_placeholder",
        cache_key=cache_key,
        image_path=str(output_path),
        image_url=build_image_url(request, "generation_service", output_path) or "",
        prompt=prompt,
        negative_prompt=negative_prompt,
        latency_ms=latency_ms,
        unsafe_pixels_used=False,
        note="Safe placeholder backend validates the production-safe API contract.",
    )


@app.post("/v1/inpaint-local-research", response_model=InpaintResearchResponse)
def inpaint_local_research(payload: InpaintResearchRequest, request: Request) -> InpaintResearchResponse:
    try:
        input_path = resolve_input_file(payload.input_image)
        if payload.report_path:
            resolve_report_file(payload.report_path)
    except FileNotFoundError as exc:
        raise HTTPException(status_code=404, detail=str(exc)) from exc
    except ValueError as exc:
        raise HTTPException(status_code=400, detail=str(exc)) from exc

    if not payload.execute:
        return InpaintResearchResponse(
            status="dry_run_validated",
            backend="local_inpainting_research",
            execute=False,
            target_label=payload.target_label,
            input_image=str(input_path),
            note="Dry-run validates the research API contract without loading the diffusion model.",
        )

    acquired = INPAINTING_LOCK.acquire(blocking=False)

    if not acquired:
        raise HTTPException(
            status_code=409,
            detail="Local inpainting research backend is busy. Try again after the current job completes.",
        )

    try:
        try:
            report = run_detector_guided_inpainting(payload)
        except Exception as exc:
            raise HTTPException(
                status_code=500,
                detail=str(exc),
            ) from exc
    finally:
        INPAINTING_LOCK.release()

    generation = report.get("generation", {})
    detector = report.get("detector", {})

    output_image = generation.get("output_image")
    mask_image = generation.get("mask_image")
    debug_grid = generation.get("debug_grid")

    return InpaintResearchResponse(
        status="completed",
        backend="local_inpainting_research",
        execute=True,
        target_label=payload.target_label,
        input_image=str(input_path),
        note="Executed detector-guided local inpainting through the generation service.",
        report_path=report.get("_service_report_path"),
        output_image=output_image,
        mask_image=mask_image,
        debug_grid=debug_grid,
        output_url=build_image_url(request, "generation", output_image),
        mask_url=build_image_url(request, "generation", mask_image),
        debug_grid_url=build_image_url(request, "generation", debug_grid),
        detector_latency_seconds=detector.get("latency_seconds"),
        inpainting_latency_seconds=generation.get("inpainting_latency_seconds"),
        total_latency_seconds=generation.get("total_latency_seconds"),
        peak_vram_mb=generation.get("peak_vram_mb"),
        target_detection=detector.get("target_detection"),
        unsafe_pixels_used_for_inpainting_conditioning=bool(
            report.get("unsafe_pixels_used_for_inpainting_conditioning", True)
        ),
        production_default=bool(report.get("production_default", False)),
        local_only=bool(report.get("local_only", True)),
    )


# ---------------------------------------------------------------------------
# SafeLens bbox-guided local cleanup endpoint
# ---------------------------------------------------------------------------
# This endpoint is intentionally small and explicit:
# - it accepts a detector-provided bbox;
# - it does not re-detect the object;
# - it modifies only the bbox region plus a configurable margin;
# - it writes a report for evaluation.
#
# It is used for medium-risk localized restricted objects such as cigarette/vape.

from pathlib import Path as _SafeLensPath
from typing import Any as _SafeLensAny

import cv2 as _safelens_cv2
import numpy as _safelens_np
from pydantic import BaseModel as _SafeLensBaseModel, Field as _SafeLensField


_SAFELENS_ROOT = _SafeLensPath(__file__).resolve().parents[1]
_SAFELENS_GENERATION_DIR = _SAFELENS_ROOT / "outputs" / "generation"


class SafeLensBboxInpaintRequest(_SafeLensBaseModel):
    input_image: str = _SafeLensField(..., description="Image path under the configured uploads directory.")
    box_xyxy: list[float] = _SafeLensField(..., min_length=4, max_length=4)
    target_label: str = "restricted_object"
    output_stem: str = "restricted_object_bbox_cleanup"
    report_path: str = "reports/generation_service/restricted_object_bbox_cleanup_report.json"
    box_expand_ratio: float = _SafeLensField(default=0.18, ge=0.0, le=0.5)
    inpaint_radius: float = _SafeLensField(default=5.0, ge=1.0, le=10.0)
    method: str = "telea"


def _safelens_clip_bbox(box_xyxy: list[float], width: int, height: int, expand_ratio: float) -> tuple[int, int, int, int]:
    x1, y1, x2, y2 = [float(v) for v in box_xyxy]

    if not all(_safelens_np.isfinite(value) for value in (x1, y1, x2, y2, expand_ratio)):
        raise HTTPException(status_code=400, detail="Bounding box values must be finite numbers.")

    if x2 < x1:
        x1, x2 = x2, x1

    if y2 < y1:
        y1, y2 = y2, y1

    box_w = max(1.0, x2 - x1)
    box_h = max(1.0, y2 - y1)

    expand_x = box_w * float(expand_ratio)
    expand_y = box_h * float(expand_ratio)

    x1 = max(0, int(round(x1 - expand_x)))
    y1 = max(0, int(round(y1 - expand_y)))
    x2 = min(width - 1, int(round(x2 + expand_x)))
    y2 = min(height - 1, int(round(y2 + expand_y)))

    if x2 <= x1:
        x2 = min(width - 1, x1 + 1)

    if y2 <= y1:
        y2 = min(height - 1, y1 + 1)

    return x1, y1, x2, y2


@app.post("/v1/inpaint-local-bbox")
def safelens_inpaint_local_bbox(
    request: SafeLensBboxInpaintRequest,
    http_request: Request,
) -> dict[str, _SafeLensAny]:
    import time as _safelens_time

    started = _safelens_time.perf_counter()

    try:
        input_path = resolve_input_file(request.input_image)
        report_path = resolve_report_file(request.report_path)
    except FileNotFoundError as exc:
        raise HTTPException(status_code=404, detail=str(exc)) from exc
    except ValueError as exc:
        raise HTTPException(status_code=400, detail=str(exc)) from exc

    image = _safelens_cv2.imread(str(input_path), _safelens_cv2.IMREAD_COLOR)

    if image is None:
        raise ValueError(f"OpenCV could not read input image: {input_path}")

    height, width = image.shape[:2]
    x1, y1, x2, y2 = _safelens_clip_bbox(
        request.box_xyxy,
        width=width,
        height=height,
        expand_ratio=request.box_expand_ratio,
    )

    mask = _safelens_np.zeros((height, width), dtype=_safelens_np.uint8)
    mask[y1:y2, x1:x2] = 255

    kernel_size = max(3, int(round(min(width, height) * 0.01)))
    if kernel_size % 2 == 0:
        kernel_size += 1

    mask = _safelens_cv2.GaussianBlur(mask, (kernel_size, kernel_size), 0)
    mask = (mask > 8).astype(_safelens_np.uint8) * 255

    method = str(request.method).lower().strip()
    if method not in {"ns", "telea"}:
        raise HTTPException(status_code=400, detail="method must be 'ns' or 'telea'.")

    if method == "ns":
        inpaint_method = _safelens_cv2.INPAINT_NS
    else:
        inpaint_method = _safelens_cv2.INPAINT_TELEA

    cleaned = _safelens_cv2.inpaint(
        image,
        mask,
        float(request.inpaint_radius),
        inpaint_method,
    )

    _SAFELENS_GENERATION_DIR.mkdir(parents=True, exist_ok=True)

    safe_stem = safe_output_stem(request.output_stem, default="restricted_object_bbox_cleanup")

    output_path = _SAFELENS_GENERATION_DIR / f"{safe_stem}_output.png"
    mask_path = _SAFELENS_GENERATION_DIR / f"{safe_stem}_mask.png"
    debug_path = _SAFELENS_GENERATION_DIR / f"{safe_stem}_debug_grid.png"

    _safelens_cv2.imwrite(str(output_path), cleaned)
    _safelens_cv2.imwrite(str(mask_path), mask)

    debug_grid = _safelens_np.concatenate(
        [
            image,
            _safelens_cv2.cvtColor(mask, _safelens_cv2.COLOR_GRAY2BGR),
            cleaned,
        ],
        axis=1,
    )
    _safelens_cv2.imwrite(str(debug_path), debug_grid)

    report_path.parent.mkdir(parents=True, exist_ok=True)

    latency_ms = int(round((_safelens_time.perf_counter() - started) * 1000))

    output_url = str(
        http_request.url_for(
            "serve_image",
            image_scope="generation",
            filename=output_path.name,
        )
    )

    report = {
        "status": "completed",
        "backend": "bbox_guided_local_cleanup",
        "input_image": str(input_path),
        "target_label": request.target_label,
        "original_box_xyxy": [float(v) for v in request.box_xyxy],
        "expanded_box_xyxy": [x1, y1, x2, y2],
        "box_expand_ratio": request.box_expand_ratio,
        "inpaint_radius": request.inpaint_radius,
        "method": method,
        "output_path": str(output_path),
        "mask_path": str(mask_path),
        "debug_grid_path": str(debug_path),
        "output_url": output_url,
        "latency_ms": latency_ms,
        "integrity_note": "BBox came from the dedicated restricted-object detector. The endpoint does not re-detect the object.",
    }

    import json as _safelens_json

    report_path.write_text(
        _safelens_json.dumps(report, indent=2, ensure_ascii=False) + "\n",
        encoding="utf-8",
    )

    return {
        "status": "completed",
        "backend": "bbox_guided_local_cleanup",
        "image_url": output_url,
        "output_url": output_url,
        "report_path": str(report_path),
        "latency_ms": latency_ms,
        "target_label": request.target_label,
        "box_xyxy": [x1, y1, x2, y2],
    }
