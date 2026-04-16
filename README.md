# FaceCloak

FaceCloak is a fully local biometric privacy project that will use white-box adversarial optimization to push a face image away from its original embedding while keeping the image visually unchanged for a human viewer.

Phase 1 is complete in this repository. The environment is reproducible with `uv`, the Hugging Face Space entrypoint is in place, the core CPU dependencies are pinned, and the app exposes a runtime diagnostics panel that confirms Torch is working correctly on CPU.

## Phase 1 Deliverables

- `uv`-managed Python 3.12 workflow
- Pinned direct runtime dependencies in [`requirements.txt`](requirements.txt)
- Locked local dependency graph in [`uv.lock`](uv.lock)
- Hugging Face Space-compatible [`app.py`](app.py)
- Local launcher in [`main.py`](main.py)
- Initial `facecloak` package with runtime diagnostics
- Pytest coverage for the scaffold and environment checks

## Current Scope

This phase intentionally stops before model loading, face detection, embedding extraction, or PGD optimization. Those will land in later phases. Right now the goal is a stable, testable base that can run locally and deploy cleanly to a CPU Hugging Face Space.

## Quickstart

These commands keep Python and the `uv` cache inside the repository so the setup stays self-contained.

```powershell
$env:UV_CACHE_DIR = "$PWD/.uv-cache"
$env:UV_PYTHON_INSTALL_DIR = "$PWD/.uv-python"
uv python install 3.12
uv sync
uv run python main.py
```

## Run Tests

```powershell
$env:UV_CACHE_DIR = "$PWD/.uv-cache"
$env:UV_PYTHON_INSTALL_DIR = "$PWD/.uv-python"
uv run pytest
```

## Hugging Face Space Notes

Hugging Face Spaces still require a root-level `app.py` and `requirements.txt`. This repository now includes both:

- `app.py` exposes the Gradio demo for Spaces
- `requirements.txt` contains only the direct runtime dependencies, pinned to the versions validated locally
- `uv.lock` remains the source of truth for fully reproducible local development

## Verified in Phase 1

- `torch` imports successfully
- `torch.cuda.is_available()` is `False`, which matches the intended CPU deployment target
- Basic tensor math works
- The Gradio app builds successfully
- The repository has automated tests for the environment scaffold
