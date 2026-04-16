"""Project-level metadata used across the Phase 1 scaffold."""

from __future__ import annotations

PROJECT_NAME = "FaceCloak"
PROJECT_TAGLINE = "Adversarial Pixel Poisoning for Biometric Privacy Preservation"
PHASE_LABEL = "Phase 1"
PHASE_STATUS = "Complete"
PHASE_SUMMARY = (
    "Environment setup, dependency locking, Hugging Face Space entrypoints, "
    "and runtime diagnostics are in place."
)

PINNED_RUNTIME_DEPENDENCIES: tuple[tuple[str, str], ...] = (
    ("torch", "2.2.2"),
    ("torchvision", "0.17.2"),
    ("facenet-pytorch", "2.6.0"),
    ("pillow", "10.2.0"),
    ("numpy", "1.26.4"),
    ("gradio", "6.12.0"),
)


def requirements_lines() -> list[str]:
    """Return the direct runtime dependencies for Hugging Face Spaces."""

    return [f"{name}=={version}" for name, version in PINNED_RUNTIME_DEPENDENCIES]
