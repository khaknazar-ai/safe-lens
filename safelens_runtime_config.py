from __future__ import annotations

import os
import re
from pathlib import Path
from urllib.parse import urlsplit

ROOT = Path(__file__).resolve().parent
UPLOAD_ROOT = Path(os.environ.get("SAFELENS_INPUT_DIR", ROOT / "uploads")).expanduser().resolve()
REPORT_ROOT = (ROOT / "reports" / "generation_service").resolve()
ALLOWED_IMAGE_SUFFIXES = {".png", ".jpg", ".jpeg", ".webp", ".bmp"}
MAX_IMAGE_BYTES = 25 * 1024 * 1024
MAX_IMAGE_PIXELS = 40_000_000

_DEFAULT_BROWSER_ORIGINS = (
    "http://localhost:8765",
    "http://127.0.0.1:8765",
)


def _validate_origin(origin: str) -> str:
    parsed = urlsplit(origin)
    if parsed.scheme == "chrome-extension":
        valid = bool(re.fullmatch(r"[a-p]{32}", parsed.netloc))
    elif parsed.scheme == "http":
        valid = parsed.hostname in {"localhost", "127.0.0.1"} and parsed.port == 8765
    else:
        valid = False
    if not valid or parsed.username or parsed.password or parsed.path or parsed.query or parsed.fragment:
        raise ValueError(
            "Allowed origins must be exact local demo origins or a specific Chrome extension ID."
        )
    return origin.rstrip("/")


def _load_browser_origins() -> tuple[str, ...]:
    raw = os.environ.get("SAFELENS_ALLOWED_ORIGINS")
    values = _DEFAULT_BROWSER_ORIGINS if raw is None else tuple(
        value.strip() for value in raw.split(",") if value.strip()
    )
    checked = tuple(dict.fromkeys(_validate_origin(value) for value in values))
    if not checked:
        raise ValueError("SAFELENS_ALLOWED_ORIGINS must contain at least one exact origin.")
    return checked


ALLOWED_BROWSER_ORIGINS = _load_browser_origins()


def _is_within(candidate: Path, parent: Path) -> bool:
    return candidate == parent or parent in candidate.parents


def resolve_input_file(value: str | Path) -> Path:
    raw = Path(value).expanduser()
    if raw.is_absolute():
        candidate = raw.resolve()
    elif raw.parts and raw.parts[0].lower() == "uploads":
        candidate = (UPLOAD_ROOT / Path(*raw.parts[1:])).resolve()
    else:
        candidate = (UPLOAD_ROOT / raw).resolve()

    if not _is_within(candidate, UPLOAD_ROOT):
        raise ValueError("Input image must be inside the configured uploads directory.")
    if not candidate.is_file():
        raise FileNotFoundError("Input image was not found in the configured uploads directory.")
    if candidate.suffix.lower() not in ALLOWED_IMAGE_SUFFIXES:
        raise ValueError(f"Unsupported image type: {candidate.suffix or '(no extension)'}")
    if candidate.stat().st_size > MAX_IMAGE_BYTES:
        raise ValueError(f"Input image exceeds the {MAX_IMAGE_BYTES // (1024 * 1024)} MiB limit.")

    try:
        from PIL import Image

        with Image.open(candidate) as image:
            width, height = image.size
            if width <= 0 or height <= 0 or width * height > MAX_IMAGE_PIXELS:
                raise ValueError("Input image dimensions exceed the configured pixel limit.")
            image.verify()
    except ValueError:
        raise
    except Exception as exc:
        raise ValueError("Input file is not a readable image.") from exc

    return candidate


def resolve_report_file(value: str | Path) -> Path:
    raw = Path(value).expanduser()
    if raw.is_absolute():
        candidate = raw.resolve()
    elif raw.parts and tuple(part.lower() for part in raw.parts[:2]) == ("reports", "generation_service"):
        candidate = (ROOT / raw).resolve()
    else:
        candidate = (REPORT_ROOT / raw).resolve()

    if not _is_within(candidate, REPORT_ROOT):
        raise ValueError("Report path must stay inside reports/generation_service.")
    if candidate.suffix.lower() != ".json":
        raise ValueError("Report path must use the .json suffix.")
    return candidate


def safe_output_stem(value: str | None, default: str = "safelens_output") -> str:
    candidate = value or default
    safe = re.sub(r"[^A-Za-z0-9_-]+", "_", candidate).strip("_-")[:64]
    return safe or default
