from __future__ import annotations

from generation_service.schemas import GenerationMetadataRequest


def build_safe_placeholder_prompt(request: GenerationMetadataRequest) -> str:
    return (
        "family-friendly abstract safety replacement image, "
        f"category hint: {request.category}, "
        f"severity: {request.severity}, "
        f"style: {request.style}, "
        "calm visual design, no unsafe content, no readable text"
    )


def build_negative_prompt() -> str:
    return (
        "nudity, gore, blood, weapon, drug, cigarette, vape, alcohol, gambling, "
        "explicit content, violence, logo, watermark, readable text, distorted"
    )
