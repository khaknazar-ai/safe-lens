from __future__ import annotations

import json
from functools import lru_cache
from pathlib import Path
from typing import Any


ROOT = Path(__file__).resolve().parents[1]
REGISTRY_PATH = ROOT / "config" / "restricted_category_registry.json"


@lru_cache(maxsize=1)
def load_registry() -> dict[str, Any]:
    return json.loads(REGISTRY_PATH.read_text(encoding="utf-8-sig"))


def get_enabled_categories(registry: dict[str, Any] | None = None) -> dict[str, Any]:
    registry = registry or load_registry()
    categories = registry.get("categories", {})

    return {
        name: category
        for name, category in categories.items()
        if bool(category.get("enabled", False))
    }


def get_registry_target_classes(registry: dict[str, Any] | None = None) -> list[str]:
    labels: list[str] = []

    for category in get_enabled_categories(registry).values():
        labels.extend(str(label) for label in category.get("labels", []))

    return sorted(set(labels))


def summarize_registry(registry: dict[str, Any] | None = None) -> dict[str, Any]:
    registry = registry or load_registry()
    categories = registry.get("categories", {})
    branches = registry.get("model_branches", {})

    enabled_categories = get_enabled_categories(registry)

    return {
        "version": registry.get("version"),
        "category_count": len(categories),
        "enabled_category_count": len(enabled_categories),
        "enabled_categories": sorted(enabled_categories.keys()),
        "enabled_target_classes": get_registry_target_classes(registry),
        "model_branch_count": len(branches),
        "enabled_model_branches": sorted([
            name
            for name, branch in branches.items()
            if bool(branch.get("enabled", False))
        ]),
    }


def map_predictions_to_registry_hits(
    predictions: list[dict[str, Any]],
    min_confidence: float | None = None,
    registry: dict[str, Any] | None = None,
) -> list[dict[str, Any]]:
    registry = registry or load_registry()
    hits: list[dict[str, Any]] = []

    for category_name, category in get_enabled_categories(registry).items():
        labels = {str(label).lower() for label in category.get("labels", [])}
        thresholds = category.get("thresholds", {})
        category_hit_confidence = float(thresholds.get("hit_confidence", 0.0))

        if min_confidence is not None:
            category_hit_confidence = max(category_hit_confidence, float(min_confidence))

        for prediction in predictions:
            label = str(prediction.get("label", "")).lower()
            confidence = float(prediction.get("confidence", 0.0))

            if label not in labels:
                continue

            if confidence < category_hit_confidence:
                continue

            hits.append({
                "category": category_name,
                "label": prediction.get("label"),
                "confidence": confidence,
                "box_xyxy": prediction.get("box_xyxy"),
                "risk_level": category.get("risk_level"),
                "default_action": category.get("default_action"),
                "requires_bbox": bool(category.get("requires_bbox", True)),
                "cleanup_allowed": bool(category.get("cleanup_allowed", False)),
                "replacement_mode": category.get("replacement_mode"),
                "model_branch": category.get("model_branch"),
                "threshold_used": category_hit_confidence,
            })

    hits.sort(key=lambda item: float(item.get("confidence", 0.0)), reverse=True)
    return hits


def choose_primary_registry_hit(hits: list[dict[str, Any]]) -> dict[str, Any] | None:
    if not hits:
        return None

    return max(hits, key=lambda item: float(item.get("confidence", 0.0)))
