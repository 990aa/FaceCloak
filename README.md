---
title: Universal Adversarial Cloak
emoji: "🛡️"
colorFrom: indigo
colorTo: blue
sdk: gradio
sdk_version: 6.12.0
python_version: "3.11"
app_file: app.py
pinned: false
---

# Universal Adversarial Cloak

Universal adversarial pixel cloaking against modern visual recognition systems with explicit imperceptibility controls and oracle validation.

[Live Demo on Hugging Face Spaces](https://huggingface.co/spaces/a-01a/universal-adversarial-cloak)

## What This Project Does

This project perturbs input images with constrained PGD so that independent recognition models fail to match them while human-visible quality remains high.

- Face route: MTCNN face detection + FaceNet/CLIP-aligned attack objective.
- General-image route: CLIP-space semantic cloaking for scenes, products, and documents.
- Hard constraints: bounded perturbation norms with SSIM and PSNR tracking.
- Independent oracle validation: ArcFace for face identity and CLIP ViT-L/14 for general similarity.

## Benchmarking and Ablations

The repository includes two complementary evaluation tracks:

- Benchmarking suite (`uacloak/benchmarking.py`): fixed defaults, per-sample metrics, runtime stage breakdown, FGSM baseline comparison, and post-processing robustness checks.
- Ablation suite (`uacloak/ablation.py`): epsilon, steps, loss design, norm/budget sensitivity, and surrogate-to-oracle transfer matrix.

All generated artifacts are written under `results/` by default.

## Quick Start

Dependencies are managed with `uv`.

```bash
# Install runtime and dev dependencies
uv sync --group dev

# Launch the local Gradio app
uv run app.py
```

## Reproducible Evaluation Workflow

```bash
# 1) Refresh general-domain fixtures and manifest
uv run python scripts/download_general_images.py

# 2) Run fixed-condition benchmark suite
uv run python -m uacloak.benchmarking \
	--manifest benchmarks/benchmarking_manifest.csv \
	--output-csv results/benchmark_metrics.csv \
	--output-summary results/benchmark_summary.md \
	--output-json results/benchmark_summary.json

# 3) Run ablations
uv run python -m uacloak.ablation \
	--manifest benchmarks/ablation_sample_manifest.csv \
	--output-dir results/ablations \
	--allow-small-set --skip-convnext

# 4) Build benchmark visuals + report markdown for the UI tab
uv run python scripts/generate_report.py \
	--manifest benchmarks/benchmarking_manifest.csv \
	--csv results/benchmark_metrics.csv \
	--json results/benchmark_summary.json \
	--output-dir results
```

The Benchmark Results tab in the app reads these generated files:

- `results/pca.png`
- `results/scatter.png`
- `results/grid.png`
- `results/benchmark_report.md`

## Notebook Generation

Generate the technical walkthrough notebook from local fixtures:

```bash
uv run python scripts/generate_notebook.py
```

This writes `universal_cloaking_demo.ipynb` and includes all images from:

- `tests/fixtures/faces`
- `tests/fixtures/general`

## Deployment

Set a Hugging Face token in environment variable `UACLOAK_HF_TOKEN` (or `.env`) and run:

```bash
uv run python scripts/create_or_update_space.py a-01a/universal-adversarial-cloak
```

## Privacy

The pipeline is local-first. No external telemetry or hosted inference calls are required for normal operation; model weights are downloaded and executed locally.
