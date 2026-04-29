"""Project metadata and repository-level constants."""

from __future__ import annotations

from pathlib import Path

PROJECT_NAME = "VisionCloak"
PROJECT_SLUG = "visioncloak"
PROJECT_TAGLINE = (
    "Adversarial image cloaking for modern vision-language systems with "
    "multi-signal, CPU-first optimization."
)
PHASE_LABEL = "VisionCloak"
PHASE_STATUS = "In Progress"
PHASE_SUMMARY = (
    "Multi-surrogate semantic cloaking, patch-token divergence, frequency shaping, "
    "color-histogram divergence, SSIM-aware optimization, and JPEG resilience."
)

PROJECT_ROOT = Path(__file__).resolve().parents[1]
TORCH_CACHE_DIR = PROJECT_ROOT / ".torch-cache"
SPACE_URL_TEMPLATE = "https://huggingface.co/spaces/{repo_id}"
DEFAULT_SPACE_SLUG = "visioncloak"

DEFAULT_SURROGATES: tuple[str, ...] = ("clip_l14", "siglip", "dinov2")

PINNED_RUNTIME_DEPENDENCIES: tuple[tuple[str, str], ...] = (
    ("torch", "2.2.2"),
    ("torchvision", "0.17.2"),
    ("facenet-pytorch", "2.6.0"),
    ("Pillow", "10.2.0"),
    ("numpy", "1.26.4"),
    ("scikit-image", "0.24.0"),
    ("gradio", "6.12.0"),
    ("huggingface-hub", "0.36.2"),
    ("transformers", "4.41.2"),
    ("pytorch-msssim", "1.0.0"),
    ("timm", "1.0.0"),
)

SPACE_UPLOAD_ALLOW_PATTERNS: tuple[str, ...] = (
    "app.py",
    "eval.py",
    "ablation.py",
    "README.md",
    "requirements.txt",
    "visioncloak/**",
    "uacloak/**",
)


def requirements_lines() -> list[str]:
    """Return the direct runtime dependencies for Hugging Face Spaces."""

    return [f"{name}=={version}" for name, version in PINNED_RUNTIME_DEPENDENCIES]
