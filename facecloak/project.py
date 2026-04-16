"""Project metadata and repository-level constants."""

from __future__ import annotations

from pathlib import Path

PROJECT_NAME = "FaceCloak"
PROJECT_SLUG = "facecloak"
PROJECT_TAGLINE = "Adversarial Pixel Poisoning for Biometric Privacy Preservation"
PHASE_LABEL = "Phases 2-3"
PHASE_STATUS = "Complete"
PHASE_SUMMARY = (
    "Face detection, embedding extraction, cosine similarity scoring, "
    "and PGD-based face cloaking are now implemented."
)

PROJECT_ROOT = Path(__file__).resolve().parents[1]
TORCH_CACHE_DIR = PROJECT_ROOT / ".torch-cache"
SPACE_URL_TEMPLATE = "https://huggingface.co/spaces/{repo_id}"

PINNED_RUNTIME_DEPENDENCIES: tuple[tuple[str, str], ...] = (
    ("torch", "2.2.2"),
    ("torchvision", "0.17.2"),
    ("facenet-pytorch", "2.6.0"),
    ("pillow", "10.2.0"),
    ("numpy", "1.26.4"),
    ("gradio", "6.12.0"),
    ("huggingface-hub", "1.10.2"),
)

SPACE_UPLOAD_ALLOW_PATTERNS: tuple[str, ...] = (
    "app.py",
    "main.py",
    "README.md",
    "requirements.txt",
    "pyproject.toml",
    "uv.lock",
    "facecloak/**",
    "scripts/**",
    "tests/**",
)


def requirements_lines() -> list[str]:
    """Return the direct runtime dependencies for Hugging Face Spaces."""

    return [f"{name}=={version}" for name, version in PINNED_RUNTIME_DEPENDENCIES]
