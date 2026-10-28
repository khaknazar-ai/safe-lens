from __future__ import annotations

import argparse
import json
import math
import sys
import time
import traceback
from pathlib import Path
from typing import Any

ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from safelens_runtime_config import resolve_input_file, resolve_report_file, safe_output_stem

import numpy as np
import onnxruntime as ort
import torch
from PIL import Image, ImageDraw, ImageFilter

from diffusers import AutoPipelineForInpainting

DEFAULT_MODEL_ID = "runwayml/stable-diffusion-inpainting"

DEFAULT_INPUT_IMAGE = ROOT / "uploads" / "input.png"
DETECTOR_MODEL_PATH = ROOT / "models" / "detectors" / "yolov8n_coco_640.onnx"

GEN_DIR = ROOT / "outputs" / "generation"
DEFAULT_REPORT_PATH = ROOT / "reports" / "generation_service" / "detector_guided_inpainting_research.json"

DETECTOR_INPUT_SIZE = 640
DETECTOR_CONF_THRESHOLD = 0.25
DETECTOR_IOU_THRESHOLD = 0.45

COCO_NAMES = [
    "person", "bicycle", "car", "motorcycle", "airplane", "bus", "train", "truck", "boat", "traffic light",
    "fire hydrant", "stop sign", "parking meter", "bench", "bird", "cat", "dog", "horse", "sheep", "cow",
    "elephant", "bear", "zebra", "giraffe", "backpack", "umbrella", "handbag", "tie", "suitcase", "frisbee",
    "skis", "snowboard", "sports ball", "kite", "baseball bat", "baseball glove", "skateboard", "surfboard",
    "tennis racket", "bottle", "wine glass", "cup", "fork", "knife", "spoon", "bowl", "banana", "apple",
    "sandwich", "orange", "broccoli", "carrot", "hot dog", "pizza", "donut", "cake", "chair", "couch",
    "potted plant", "bed", "dining table", "toilet", "tv", "laptop", "mouse", "remote", "keyboard",
    "cell phone", "microwave", "oven", "toaster", "sink", "refrigerator", "book", "clock", "vase",
    "scissors", "teddy bear", "hair drier", "toothbrush"
]


def write_report(report: dict[str, Any]) -> None:
    report_path = resolve_report_file(report.get("_report_path", DEFAULT_REPORT_PATH))
    report_path.parent.mkdir(parents=True, exist_ok=True)

    clean_report = {key: value for key, value in report.items() if key != "_report_path"}
    report_path.write_text(json.dumps(clean_report, indent=2), encoding="utf-8")

    print(json.dumps(clean_report, indent=2))


def resize_for_diffusion(image: Image.Image, max_size: int = 512) -> Image.Image:
    image = image.convert("RGB")
    w, h = image.size

    scale = min(max_size / w, max_size / h)
    new_w = int(w * scale)
    new_h = int(h * scale)

    new_w = max(64, (new_w // 8) * 8)
    new_h = max(64, (new_h // 8) * 8)

    return image.resize((new_w, new_h), Image.Resampling.LANCZOS)


def letterbox_image(image: Image.Image, size: int = DETECTOR_INPUT_SIZE) -> tuple[np.ndarray, dict[str, float]]:
    image = image.convert("RGB")
    original_w, original_h = image.size

    scale = min(size / original_w, size / original_h)
    resized_w = int(round(original_w * scale))
    resized_h = int(round(original_h * scale))

    pad_x = int(round((size - resized_w) / 2))
    pad_y = int(round((size - resized_h) / 2))

    resized = image.resize((resized_w, resized_h), Image.Resampling.BILINEAR)

    canvas = Image.new("RGB", (size, size), (114, 114, 114))
    canvas.paste(resized, (pad_x, pad_y))

    arr = np.asarray(canvas).astype(np.float32) / 255.0
    tensor = np.transpose(arr, (2, 0, 1))[None, :, :, :]

    meta = {
        "original_width": float(original_w),
        "original_height": float(original_h),
        "scale": float(scale),
        "pad_x": float(pad_x),
        "pad_y": float(pad_y),
    }

    return tensor, meta


def xywh_to_xyxy(cx: float, cy: float, width: float, height: float) -> list[float]:
    return [
        cx - width / 2,
        cy - height / 2,
        cx + width / 2,
        cy + height / 2,
    ]


def unletterbox_box(box: list[float], meta: dict[str, float]) -> list[float]:
    x1, y1, x2, y2 = box

    x1 = (x1 - meta["pad_x"]) / meta["scale"]
    y1 = (y1 - meta["pad_y"]) / meta["scale"]
    x2 = (x2 - meta["pad_x"]) / meta["scale"]
    y2 = (y2 - meta["pad_y"]) / meta["scale"]

    x1 = max(0.0, min(meta["original_width"], x1))
    y1 = max(0.0, min(meta["original_height"], y1))
    x2 = max(0.0, min(meta["original_width"], x2))
    y2 = max(0.0, min(meta["original_height"], y2))

    return [round(x1, 2), round(y1, 2), round(x2, 2), round(y2, 2)]


def box_iou(a: list[float], b: list[float]) -> float:
    x1 = max(a[0], b[0])
    y1 = max(a[1], b[1])
    x2 = min(a[2], b[2])
    y2 = min(a[3], b[3])

    inter_w = max(0.0, x2 - x1)
    inter_h = max(0.0, y2 - y1)
    intersection = inter_w * inter_h

    area_a = max(0.0, a[2] - a[0]) * max(0.0, a[3] - a[1])
    area_b = max(0.0, b[2] - b[0]) * max(0.0, b[3] - b[1])

    union = area_a + area_b - intersection
    return 0.0 if union <= 0 else intersection / union


def nms(detections: list[dict[str, Any]], iou_threshold: float) -> list[dict[str, Any]]:
    sorted_items = sorted(detections, key=lambda item: item["confidence"], reverse=True)
    selected: list[dict[str, Any]] = []

    while sorted_items:
        current = sorted_items.pop(0)
        selected.append(current)

        kept = []
        for item in sorted_items:
            same_class = item["class_id"] == current["class_id"]
            overlap = box_iou(item["box_xyxy"], current["box_xyxy"]) > iou_threshold

            if same_class and overlap:
                continue

            kept.append(item)

        sorted_items = kept

    return selected


def run_yolo_detection(image: Image.Image) -> list[dict[str, Any]]:
    if not DETECTOR_MODEL_PATH.exists():
        raise FileNotFoundError(f"Detector model not found: {DETECTOR_MODEL_PATH}")

    tensor, meta = letterbox_image(image)

    session = ort.InferenceSession(
        str(DETECTOR_MODEL_PATH),
        providers=["CPUExecutionProvider"],
    )

    input_name = session.get_inputs()[0].name
    output_name = session.get_outputs()[0].name

    output = session.run([output_name], {input_name: tensor})[0]

    if output.ndim != 3:
        raise RuntimeError(f"Unexpected YOLO output ndim: {output.shape}")

    raw = output[0]

    if raw.shape[0] == 84:
        data = raw
    elif raw.shape[1] == 84:
        data = raw.T
    else:
        raise RuntimeError(f"Unexpected YOLO output shape: {output.shape}")

    num_channels, num_candidates = data.shape

    if num_channels != 84:
        raise RuntimeError(f"Expected 84 YOLO channels, got: {num_channels}")

    candidates: list[dict[str, Any]] = []

    for i in range(num_candidates):
        cx = float(data[0, i])
        cy = float(data[1, i])
        width = float(data[2, i])
        height = float(data[3, i])

        scores = data[4:, i]
        class_id = int(np.argmax(scores))
        confidence = float(scores[class_id])

        if confidence < DETECTOR_CONF_THRESHOLD:
            continue

        box = unletterbox_box(xywh_to_xyxy(cx, cy, width, height), meta)
        box_w = box[2] - box[0]
        box_h = box[3] - box[1]

        if box_w <= 1 or box_h <= 1:
            continue

        label = COCO_NAMES[class_id] if class_id < len(COCO_NAMES) else f"class_{class_id}"

        candidates.append(
            {
                "label": label,
                "class_id": class_id,
                "confidence": round(confidence, 4),
                "box_xyxy": box,
            }
        )

    return nms(candidates, DETECTOR_IOU_THRESHOLD)


def select_target_detection(detections: list[dict[str, Any]], target_label: str) -> dict[str, Any]:
    matches = [item for item in detections if item["label"] == target_label]

    if not matches:
        labels = [f"{item['label']}:{item['confidence']}" for item in detections[:10]]
        raise RuntimeError(f"No detection found for target_label={target_label}. Top detections: {labels}")

    return sorted(matches, key=lambda item: item["confidence"], reverse=True)[0]


def scale_box_to_resized_image(
    box_xyxy: list[float],
    original_size: tuple[int, int],
    resized_size: tuple[int, int],
) -> list[float]:
    original_w, original_h = original_size
    resized_w, resized_h = resized_size

    scale_x = resized_w / original_w
    scale_y = resized_h / original_h

    return [
        box_xyxy[0] * scale_x,
        box_xyxy[1] * scale_y,
        box_xyxy[2] * scale_x,
        box_xyxy[3] * scale_y,
    ]


def expand_box(box: list[float], image_size: tuple[int, int], ratio: float) -> list[int]:
    w, h = image_size
    x1, y1, x2, y2 = box

    box_w = x2 - x1
    box_h = y2 - y1

    x1 -= box_w * ratio
    x2 += box_w * ratio
    y1 -= box_h * ratio
    y2 += box_h * ratio

    return [
        int(max(0, math.floor(x1))),
        int(max(0, math.floor(y1))),
        int(min(w, math.ceil(x2))),
        int(min(h, math.ceil(y2))),
    ]


def create_mask_from_detection(
    detection: dict[str, Any],
    original_size: tuple[int, int],
    resized_size: tuple[int, int],
    expand_ratio: float,
    blur_radius: float,
) -> tuple[Image.Image, list[int]]:
    mask = Image.new("L", resized_size, 0)
    draw = ImageDraw.Draw(mask)

    scaled_box = scale_box_to_resized_image(
        detection["box_xyxy"],
        original_size=original_size,
        resized_size=resized_size,
    )

    expanded = expand_box(scaled_box, image_size=resized_size, ratio=expand_ratio)

    draw.rounded_rectangle(expanded, radius=18, fill=255)

    if blur_radius > 0:
        mask = mask.filter(ImageFilter.GaussianBlur(radius=blur_radius))

    return mask, expanded


def save_debug_grid(
    input_image: Image.Image,
    mask_image: Image.Image,
    output_image: Image.Image,
    path: Path,
) -> None:
    w, h = input_image.size
    grid = Image.new("RGB", (w * 3, h), "white")

    mask_rgb = Image.merge("RGB", [mask_image, mask_image, mask_image])

    grid.paste(input_image, (0, 0))
    grid.paste(mask_rgb, (w, 0))
    grid.paste(output_image.resize((w, h)), (w * 2, 0))

    grid.save(path)


def load_inpainting_pipeline(model_id: str, dtype: torch.dtype):
    attempts = [
        {
            "name": "fp16_safetensors",
            "kwargs": {
                "torch_dtype": dtype,
                "use_safetensors": True,
                "variant": "fp16",
            },
        },
        {
            "name": "fp16_bin",
            "kwargs": {
                "torch_dtype": dtype,
                "use_safetensors": False,
                "variant": "fp16",
            },
        },
        {
            "name": "default_bin",
            "kwargs": {
                "torch_dtype": dtype,
                "use_safetensors": False,
            },
        },
    ]

    errors = []

    for attempt in attempts:
        try:
            print(f"Trying pipeline load mode: {attempt['name']}")
            pipe = AutoPipelineForInpainting.from_pretrained(model_id, **attempt["kwargs"])
            return pipe, attempt["name"], errors
        except Exception as exc:
            error_text = f"{type(exc).__name__}: {exc}"
            print(f"Load attempt failed: {attempt['name']} -> {error_text}")
            errors.append(
                {
                    "attempt": attempt["name"],
                    "error": error_text,
                }
            )

    raise RuntimeError(f"All pipeline load attempts failed: {errors}")


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--model_id", default=DEFAULT_MODEL_ID)
    parser.add_argument("--input_image", default=str(DEFAULT_INPUT_IMAGE))
    parser.add_argument("--report_path", default=str(DEFAULT_REPORT_PATH))
    parser.add_argument("--milestone", default="research")
    parser.add_argument("--mode", default="detector_guided_local_inpainting_research")
    parser.add_argument("--output_stem", default=None)
    parser.add_argument("--target_label", default="bus")
    parser.add_argument("--steps", type=int, default=15)
    parser.add_argument("--guidance_scale", type=float, default=7.5)
    parser.add_argument("--seed", type=int, default=123)
    parser.add_argument("--max_size", type=int, default=512)
    parser.add_argument("--box_expand_ratio", type=float, default=0.08)
    parser.add_argument("--mask_blur_radius", type=float, default=3.0)
    parser.add_argument("--prompt", default="empty clean city street background, realistic building facade, natural road surface, no vehicle")
    parser.add_argument("--negative_prompt", default="bus, car, vehicle, people, text, logo, watermark, distorted, low quality")
    args = parser.parse_args()

    report_path = resolve_report_file(args.report_path)
    input_image_path = resolve_input_file(args.input_image)
    args.report_path = str(report_path)
    args.output_stem = safe_output_stem(
        args.output_stem or f"detector_guided_{input_image_path.stem}_{args.target_label}",
        default="detector_guided_inpainting",
    )

    GEN_DIR.mkdir(parents=True, exist_ok=True)
    report_path.parent.mkdir(parents=True, exist_ok=True)

    started_at = time.perf_counter()

    try:
        original_image = Image.open(input_image_path).convert("RGB")
        original_size = original_image.size

        detector_started_at = time.perf_counter()
        detections = run_yolo_detection(original_image)
        detector_latency_s = time.perf_counter() - detector_started_at

        target_detection = select_target_detection(detections, args.target_label)

        resized_image = resize_for_diffusion(original_image, max_size=args.max_size)
        resized_size = resized_image.size

        mask_image, expanded_mask_box = create_mask_from_detection(
            target_detection,
            original_size=original_size,
            resized_size=resized_size,
            expand_ratio=args.box_expand_ratio,
            blur_radius=args.mask_blur_radius,
        )

        output_stem = args.output_stem

        input_path = GEN_DIR / f"{output_stem}_input.png"
        mask_path = GEN_DIR / f"{output_stem}_mask.png"
        output_path = GEN_DIR / f"{output_stem}_output.png"
        grid_path = GEN_DIR / f"{output_stem}_debug_grid.png"

        resized_image.save(input_path)
        mask_image.save(mask_path)

        device = "cuda" if torch.cuda.is_available() else "cpu"
        dtype = torch.float16 if device == "cuda" else torch.float32

        pipe, load_mode, load_errors = load_inpainting_pipeline(args.model_id, dtype)

        if device == "cuda":
            pipe = pipe.to("cuda")
            pipe.enable_attention_slicing()
            torch.cuda.reset_peak_memory_stats()
        else:
            pipe.enable_attention_slicing()

        generator = torch.Generator(device=device).manual_seed(args.seed)

        inference_started_at = time.perf_counter()

        result = pipe(
            prompt=args.prompt,
            negative_prompt=args.negative_prompt,
            image=resized_image,
            mask_image=mask_image,
            num_inference_steps=args.steps,
            guidance_scale=args.guidance_scale,
            generator=generator,
        ).images[0]

        inpainting_latency_s = time.perf_counter() - inference_started_at
        total_latency_s = time.perf_counter() - started_at

        result.save(output_path)
        save_debug_grid(resized_image, mask_image, result, grid_path)

        peak_vram_mb = None
        if device == "cuda":
            peak_vram_mb = round(torch.cuda.max_memory_allocated() / 1024 / 1024, 2)

        report = {
            "milestone": args.milestone,
            "status": "completed",
            "mode": args.mode,
            "input_image_source": str(input_image_path),
            "model_id": args.model_id,
            "pipeline_load_mode": load_mode,
            "pipeline_load_fallback_errors": load_errors,
            "device": device,
            "dtype": str(dtype),
            "detector": {
                "model_path": str(DETECTOR_MODEL_PATH),
                "provider": "CPUExecutionProvider",
                "latency_seconds": round(detector_latency_s, 3),
                "target_label": args.target_label,
                "target_detection": target_detection,
                "top_detections": detections[:10],
            },
            "mask": {
                "source": "yolo_detection_box",
                "box_expand_ratio": args.box_expand_ratio,
                "mask_blur_radius": args.mask_blur_radius,
                "expanded_mask_box_resized_xyxy": expanded_mask_box,
            },
            "generation": {
                "input_size": list(resized_size),
                "input_image": str(input_path),
                "mask_image": str(mask_path),
                "output_image": str(output_path),
                "debug_grid": str(grid_path),
                "prompt": args.prompt,
                "negative_prompt": args.negative_prompt,
                "steps": args.steps,
                "guidance_scale": args.guidance_scale,
                "seed": args.seed,
                "inpainting_latency_seconds": round(inpainting_latency_s, 3),
                "total_latency_seconds": round(total_latency_s, 3),
                "peak_vram_mb": peak_vram_mb,
            },
            "integrity_note": "Research-only detector-guided inpainting on a safe demo image. This branch conditions on original image pixels and is not the default production replacement path.",
            "unsafe_pixels_used_for_inpainting_conditioning": True,
            "production_default": False,
            "local_only": True,
        }

        report["_report_path"] = str(args.report_path)
        write_report(report)

    except Exception as exc:
        report = {
            "milestone": args.milestone,
            "status": "failed",
            "mode": args.mode,
            "error_type": type(exc).__name__,
            "error": str(exc),
            "traceback": traceback.format_exc(),
            "total_latency_seconds": round(time.perf_counter() - started_at, 3),
            "note": "Failure report is intentional so research runs are reproducible instead of silently failing.",
        }
        report["_report_path"] = str(args.report_path)
        write_report(report)
        raise


if __name__ == "__main__":
    main()
