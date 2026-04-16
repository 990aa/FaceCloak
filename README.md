---
title: FaceCloak
emoji: 🛡️
colorFrom: blue
colorTo: yellow
sdk: gradio
sdk_version: 6.12.0
python_version: "3.12"
app_file: app.py
suggested_hardware: cpu-basic
---

# FaceCloak

FaceCloak is a fully local biometric privacy project that uses white-box adversarial optimization to push a face image away from its original embedding while keeping the image visually unchanged for a human viewer.

Phases 2 and 3 are now complete in this repository. The app can detect and align a face with MTCNN, extract a 512-dimensional embedding with `InceptionResnetV1`, compare embeddings with cosine similarity, and run a white-box PGD attack that pushes the cloaked face away from its original identity in embedding space.

## Phase 2 and 3 Deliverables

- Cached MTCNN face detection and alignment on CPU
- Frozen `InceptionResnetV1(pretrained="vggface2")` embedding extraction
- Cosine similarity scoring for same-person vs different-person checks
- L-infinity PGD cloaking over the aligned face tensor
- Reverse conversion from the standardized tensor back to a downloadable PIL image
- A Gradio app with comparison and cloaking workflows
- A programmatic Hugging Face Space deployment helper that reads `FACECLOAK_HF_TOKEN` from `.env`
- Expanded test coverage, including real-image integration checks

## Quickstart

These commands keep Python and the `uv` cache inside the repository so the setup stays self-contained.

```powershell
$env:UV_CACHE_DIR = "$PWD/.uv-cache"
$env:UV_PYTHON_INSTALL_DIR = "$PWD/.uv-python"
uv python install 3.12
uv sync
uv run python main.py
```

The first model-backed run will download the pretrained FaceNet weights into `.torch-cache/`.

## App Workflow

The Gradio app now exposes two tabs:

- `Cloak Face`: upload one portrait, align the primary face, run PGD, and inspect the cloaked face plus perturbation preview
- `Compare Faces`: upload two portraits and measure cosine similarity between their face embeddings

The cloaking engine works on the aligned face crop returned by MTCNN. The tensors entering FaceNet are in the facenet-pytorch standardized range produced by `fixed_image_standardization`, which is why the default perturbation budget is specified on the `[-1, 1]`-style scale instead of in raw `0-255` pixels.

## PGD Objective

The PGD loop uses an ascent step of the form `delta += alpha * sign(grad)`. Because the step is ascent, the objective is defined as:

```text
objective = -cosine_similarity(original_embedding, cloaked_embedding) - lambda * ||delta||²
```

Maximizing that objective lowers cosine similarity while gently discouraging unnecessarily large perturbations. This is mathematically equivalent to minimizing cosine similarity with a descent update.

## Run Tests

```powershell
$env:UV_CACHE_DIR = "$PWD/.uv-cache"
$env:UV_PYTHON_INSTALL_DIR = "$PWD/.uv-python"
uv run pytest
```

The suite includes real-image integration tests using public-domain portrait photos in `tests/fixtures/faces/`.

## Hugging Face Space Notes

Hugging Face Spaces still require a root-level `app.py` and `requirements.txt`. This repository now includes both:

- `app.py` exposes the Gradio demo for Spaces
- `requirements.txt` contains only the direct runtime dependencies, pinned to the versions validated locally
- `uv.lock` remains the source of truth for fully reproducible local development

You can create or update the Space programmatically with the token stored in `.env`:

```powershell
$env:UV_CACHE_DIR = "$PWD/.uv-cache"
$env:UV_PYTHON_INSTALL_DIR = "$PWD/.uv-python"
uv run python scripts/create_or_update_space.py
```

The script will:

- read `FACECLOAK_HF_TOKEN` from the environment or `.env`
- determine the Hugging Face username with `whoami`
- create `username/facecloak` as a Gradio Space on `cpu-basic` hardware if it does not already exist
- upload the repository contents needed to build the Space

## Benchmarks Verified Locally

Using the public-domain portrait fixtures bundled for integration tests:

- same-person pairs scored above `0.8`
- different-person pairs scored below `0.3`
- the default cloaking hyperparameters (`epsilon=0.03`, `num_steps=30`) drove original-vs-cloaked similarity sharply downward on CPU

## Verified in Phase 1

- `torch` imports successfully
- `torch.cuda.is_available()` is `False`, which matches the intended CPU deployment target
- Basic tensor math works
- The Gradio app builds successfully
- The repository has automated tests for the environment scaffold
