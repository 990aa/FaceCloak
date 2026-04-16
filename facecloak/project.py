"""Project metadata and repository-level constants."""

from __future__ import annotations

from pathlib import Path

PROJECT_NAME = "FaceCloak"
PROJECT_SLUG = "facecloak"
PROJECT_TAGLINE = "Adversarial Pixel Poisoning for Biometric Privacy Preservation"
PHASE_LABEL = "FaceCloak"
PHASE_STATUS = "Ready"
PHASE_SUMMARY = (
    "Face/general routing, dual-backbone similarity (FaceNet + CLIP), "
    "PGD-based universal cloaking, post-cloak verification, and oracle benchmarking are implemented."
)

PROJECT_ROOT = Path(__file__).resolve().parents[1]
TORCH_CACHE_DIR = PROJECT_ROOT / ".torch-cache"
SPACE_URL_TEMPLATE = "https://huggingface.co/spaces/{repo_id}"

PINNED_RUNTIME_DEPENDENCIES: tuple[tuple[str, str], ...] = (
    ("torch", "2.2.2"),
    ("torchvision", "0.17.2"),
    ("facenet-pytorch", "2.6.0"),
    ("Pillow", "10.2.0"),
    ("numpy", "1.26.4"),
    ("scikit-image", "0.22.0"),
    ("gradio", "6.12.0"),
    ("huggingface-hub", "0.36.2"),
    ("transformers", "4.41.2"),
)

SPACE_UPLOAD_ALLOW_PATTERNS: tuple[str, ...] = (
    "app.py",
    "eval.py",
    "ablation.py",
    "README.md",
    "requirements.txt",
    "pyproject.toml",
    "uv.lock",
    "facecloak/**",
    "benchmarks/**",
    "scripts/**",
    "tests/**",
)


def requirements_lines() -> list[str]:
    """Return the direct runtime dependencies for Hugging Face Spaces."""

    return [f"{name}=={version}" for name, version in PINNED_RUNTIME_DEPENDENCIES]
