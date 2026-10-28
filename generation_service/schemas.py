from __future__ import annotations

from typing import Any, Literal

from pydantic import BaseModel, Field


PolicyAction = Literal["allow", "warn", "replace", "unknown"]
Severity = Literal["none", "low", "medium", "high", "unknown"]
GenerationBackend = Literal["safe_placeholder", "local_inpainting_research"]


class GenerationMetadataRequest(BaseModel):
    recommended_action: PolicyAction = Field(..., description="Policy recommendation from SafeLens policy engine.")
    category: str = Field(..., description="Policy category such as possible_alcohol_container or explicit_nsfw.")
    severity: Severity = Field(..., description="Policy severity.")
    replacement_type: str = Field(default="safe_generated_placeholder")
    safe_prompt_family: str = Field(default="family_friendly_abstract_placeholder")

    width: int = Field(default=512, ge=128, le=1024)
    height: int = Field(default=512, ge=128, le=1024)
    style: str = Field(default="soft_abstract")
    seed: int = Field(default=42, ge=0)

    unsafe_pixels_used: bool = Field(
        default=False,
        description="Production-safe endpoint must keep this false.",
    )


class GenerationResponse(BaseModel):
    status: str
    backend: GenerationBackend
    cache_key: str
    image_path: str
    image_url: str
    prompt: str
    negative_prompt: str
    latency_ms: int
    unsafe_pixels_used: bool
    note: str


class InpaintResearchRequest(BaseModel):
    input_image: str = Field(..., description="Local image path only.")
    target_label: str = Field(..., description="YOLO target label to inpaint, e.g. bottle.")
    prompt: str = Field(default="clean realistic background, object removed")
    negative_prompt: str = Field(default="object, text, logo, watermark, distorted, low quality")

    steps: int = Field(default=15, ge=1, le=50)
    max_size: int = Field(default=512, ge=256, le=768)
    box_expand_ratio: float = Field(default=0.12, ge=0.0, le=0.5)
    mask_blur_radius: float = Field(default=3.0, ge=0.0, le=20.0)

    execute: bool = Field(
        default=False,
        description="Dry-run by default. True executes local detector-guided inpainting.",
    )

    output_stem: str | None = Field(
        default=None,
        description="Optional deterministic output stem for research runs.",
    )

    report_path: str | None = Field(
        default=None,
        description="Optional local report path for research runs.",
    )


class InpaintResearchResponse(BaseModel):
    status: str
    backend: GenerationBackend
    execute: bool
    target_label: str
    input_image: str
    note: str

    report_path: str | None = None
    output_image: str | None = None
    mask_image: str | None = None
    debug_grid: str | None = None

    output_url: str | None = None
    mask_url: str | None = None
    debug_grid_url: str | None = None

    detector_latency_seconds: float | None = None
    inpainting_latency_seconds: float | None = None
    total_latency_seconds: float | None = None
    peak_vram_mb: float | None = None

    target_detection: dict[str, Any] | None = None
    unsafe_pixels_used_for_inpainting_conditioning: bool = True
    production_default: bool = False
    local_only: bool = True
