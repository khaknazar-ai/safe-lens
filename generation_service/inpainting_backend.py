from __future__ import annotations

import json
import subprocess
import sys
from pathlib import Path
from typing import Any

from generation_service.schemas import InpaintResearchRequest
from safelens_runtime_config import resolve_input_file, resolve_report_file, safe_output_stem

ROOT = Path(__file__).resolve().parents[1]
RUNNER = ROOT / "scripts" / "generation" / "run_detector_guided_inpainting_research.py"


def make_default_output_stem(request: InpaintResearchRequest) -> str:
    input_stem = Path(request.input_image).stem
    return safe_output_stem(
        f"service_{input_stem}_{request.target_label}",
        default="service_inpainting",
    )


def make_default_report_path(request: InpaintResearchRequest) -> Path:
    output_stem = safe_output_stem(
        request.output_stem or make_default_output_stem(request),
        default="service_inpainting",
    )
    return resolve_report_file(f"{output_stem}_inpainting_report.json")


def run_detector_guided_inpainting(request: InpaintResearchRequest) -> dict[str, Any]:
    if not RUNNER.is_file():
        raise FileNotFoundError("Inpainting runner is missing from this export.")

    input_path = resolve_input_file(request.input_image)
    output_stem = safe_output_stem(
        request.output_stem or make_default_output_stem(request),
        default="service_inpainting",
    )
    report_path = (
        resolve_report_file(request.report_path)
        if request.report_path
        else make_default_report_path(request)
    )
    report_path.parent.mkdir(parents=True, exist_ok=True)

    command = [
        sys.executable,
        str(RUNNER),
        "--milestone",
        "research",
        "--mode",
        "generation_service_detector_guided_inpainting",
        "--input_image",
        str(input_path),
        "--report_path",
        str(report_path),
        "--output_stem",
        output_stem,
        "--target_label",
        request.target_label,
        "--steps",
        str(request.steps),
        "--max_size",
        str(request.max_size),
        "--box_expand_ratio",
        str(request.box_expand_ratio),
        "--mask_blur_radius",
        str(request.mask_blur_radius),
        "--prompt",
        request.prompt,
        "--negative_prompt",
        request.negative_prompt,
    ]

    completed = subprocess.run(
        command,
        cwd=str(ROOT),
        text=True,
        capture_output=True,
        check=False,
    )

    if completed.returncode != 0:
        raise RuntimeError(
            "Detector-guided inpainting subprocess failed.\n"
            f"Return code: {completed.returncode}\n"
            f"STDOUT:\n{completed.stdout[-4000:]}\n"
            f"STDERR:\n{completed.stderr[-4000:]}"
        )

    if not report_path.is_file():
        raise FileNotFoundError("Inpainting report was not created.")

    report = json.loads(report_path.read_text(encoding="utf-8"))
    report["_service_report_path"] = str(report_path)
    report["_subprocess_stdout_tail"] = completed.stdout[-4000:]
    report["_subprocess_stderr_tail"] = completed.stderr[-4000:]
    return report
