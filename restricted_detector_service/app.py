from __future__ import annotations

import os
import time
from pathlib import Path
from typing import Any

from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from inference_sdk import InferenceHTTPClient

from restricted_detector_service.registry import (
    get_registry_target_classes,
    map_predictions_to_registry_hits,
    summarize_registry,
)
from safelens_runtime_config import ALLOWED_BROWSER_ORIGINS, resolve_input_file
from pydantic import BaseModel, Field


DEFAULT_MODEL_ID = "cigarette-vape-detection-xrnvm/1"
DEFAULT_TARGET_CLASSES = ["cigarette", "vape"]
DEFAULT_HIT_CONFIDENCE = 0.35

app = FastAPI(
    title="SafeLens Restricted Detector Service",
    version="0.1.0",
)


app.add_middleware(
    CORSMiddleware,
    allow_origins=list(ALLOWED_BROWSER_ORIGINS),
    allow_credentials=False,
    allow_methods=["*"],
    allow_headers=["*"],
)


class DetectLocalFileRequest(BaseModel):
    image_path: str = Field(..., description="Image path under the configured uploads directory.")
    model_id: str = DEFAULT_MODEL_ID
    target_classes: list[str] = Field(default_factory=lambda: list(DEFAULT_TARGET_CLASSES))
    hit_confidence: float = DEFAULT_HIT_CONFIDENCE


def get_api_key() -> str:
    api_key = os.environ.get("ROBOFLOW_API_KEY", "").strip()

    if not api_key:
        raise HTTPException(
            status_code=500,
            detail="ROBOFLOW_API_KEY is not configured in the local service environment.",
        )

    return api_key


def resolve_upload_image_path(image_path: str) -> Path:
    try:
        return resolve_input_file(image_path)
    except FileNotFoundError as exc:
        raise HTTPException(status_code=404, detail=str(exc)) from exc
    except ValueError as exc:
        raise HTTPException(status_code=400, detail=str(exc)) from exc


def normalize_prediction(prediction: dict[str, Any]) -> dict[str, Any]:
    x = float(prediction.get("x", 0.0))
    y = float(prediction.get("y", 0.0))
    width = float(prediction.get("width", 0.0))
    height = float(prediction.get("height", 0.0))

    x1 = x - width / 2.0
    y1 = y - height / 2.0
    x2 = x + width / 2.0
    y2 = y + height / 2.0

    label = str(prediction.get("class", prediction.get("class_name", "unknown")))
    confidence = float(prediction.get("confidence", 0.0))

    return {
        "label": label,
        "confidence": confidence,
        "box_xyxy": [x1, y1, x2, y2],
        "raw": prediction,
    }



def infer_with_retry(client: InferenceHTTPClient, image_path: Path, model_id: str, retries: int = 1) -> dict[str, Any]:
    last_error: Exception | None = None

    for attempt in range(retries + 1):
        try:
            return client.infer(str(image_path), model_id=model_id)
        except Exception as exc:
            last_error = exc
            if attempt >= retries:
                break
            time.sleep(0.8)

    raise last_error if last_error else RuntimeError("Unknown restricted detector inference error")


def build_client() -> InferenceHTTPClient:
    return InferenceHTTPClient(
        api_url="https://serverless.roboflow.com",
        api_key=get_api_key(),
    )


@app.get("/health")
def health() -> dict[str, Any]:
    registry_summary = summarize_registry()

    return {
        "status": "ok",
        "service": "safelens-restricted-detector-service",
        "default_model_id": DEFAULT_MODEL_ID,
        "target_classes": DEFAULT_TARGET_CLASSES,
        "registry": registry_summary,
        "mode": "local_backend_wrapper",
    }


@app.get("/v1/restricted/registry")
def restricted_registry() -> dict[str, Any]:
    return {
        "status": "completed",
        "registry": summarize_registry(),
        "enabled_target_classes": get_registry_target_classes(),
    }


@app.post("/v1/restricted/detect-local-file")
def detect_local_file(request: DetectLocalFileRequest) -> dict[str, Any]:
    image_path = resolve_upload_image_path(request.image_path)
    target_classes = {item.lower() for item in request.target_classes}

    started = time.perf_counter()

    client = build_client()
    raw = infer_with_retry(client, image_path=image_path, model_id=request.model_id, retries=1)

    latency_seconds = time.perf_counter() - started

    predictions = [
        normalize_prediction(item)
        for item in raw.get("predictions", [])
    ]

    registry_hits = map_predictions_to_registry_hits(
        predictions,
        min_confidence=request.hit_confidence,
    )

    target_hits = [
        item
        for item in predictions
        if item["label"].lower() in target_classes
        and item["confidence"] >= request.hit_confidence
    ]

    return {
        "status": "completed",
        "service": "safelens-restricted-detector-service",
        "model_id": request.model_id,
        "risk_level": "medium",
        "default_action": "localized_object_cleanup",
        "requires_bbox": True,
        "image_path": image_path.name,
        "latency_seconds": latency_seconds,
        "prediction_count": len(predictions),
        "target_hit_count": len(target_hits),
        "category_hit_count": len(registry_hits),
        "bbox_present": len(predictions) > 0,
        "target_bbox_present": len(target_hits) > 0,
        "category_bbox_present": len(registry_hits) > 0,
        "predictions": predictions,
        "target_hits": target_hits,
        "category_hits": registry_hits,
        "registry": summarize_registry(),
        "policy_note": "Medium-risk restricted-object detections are eligible for localized cleanup only when bbox is present.",
    }
