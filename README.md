---
title: VisionCloak v1.0
emoji: "🛡️"
colorFrom: indigo
colorTo: blue
sdk: gradio
sdk_version: 6.12.0
python_version: "3.11"
app_file: app.py
pinned: false
---

# VisionCloak v1.0

Adversarial image cloaking for modern vision-language systems, evolved from the original UACloak baseline into a multi-signal engine designed to disrupt SOTA multimodal vision pipelines on CPU.

[Live Demo on Hugging Face Spaces](https://huggingface.co/spaces/a-01a/facecloak)

## Status

The repository currently ships the original `uacloak/` implementation and evaluation flow. This README now documents the VisionCloak v1.0 direction and target architecture so the codebase, roadmap, and future refactor all point at the same end state.

## Why UACloak Needed To Evolve

The current UACloak attack optimizes against FaceNet and CLIP ViT-B/32 with L-infinity PGD. That remains useful for classic recognition APIs, but it is no longer enough for state-of-the-art vision LLMs such as GPT-5, Gemini 3.x, Claude Opus, Grok 4, and Qwen-class multimodal systems.

These systems do not rely on a small CLIP-B/32-style encoder alone. They typically use:

- Much larger vision backbones such as ViT-L/14, ViT-H, ViT-G, SigLIP, and InternViT-scale encoders.
- Patch-level tokenization where 14x14 or 16x16 image regions become semantically meaningful tokens.
- Multi-scale processing where thumbnails and higher-resolution tiles are consumed together.
- JPEG-aware ingestion that can wash out weak, purely high-frequency perturbations.
- Holistic scene reasoning over layout, colors, object relations, and regional semantics.

That means the attack surface is no longer just pixel space against one surrogate. VisionCloak v1.0 targets the full pipeline: semantic embeddings, patch tokens, frequency structure, color statistics, and perceptual quality constraints together.

## VisionCloak v1.0 Architecture

### Phase 1: Package and Engine Redesign

The planned package rename is:

- `uacloak/` -> `visioncloak/`

The target module layout is:

- `visioncloak/models.py` — expanded surrogate registry for large CPU-capable encoders.
- `visioncloak/engine.py` — core optimization loop replacing the current `cloaking.py`.
- `visioncloak/transforms.py` — preprocessing, resize, tiling, and compression-resilience transforms.
- `visioncloak/losses.py` — composable differentiable loss units.
- `visioncloak/pipeline.py` — image routing and execution flow, updated for general images rather than face-centric assumptions.
- `visioncloak/interface.py` — updated Gradio interface and outputs.
- `visioncloak/evaluation.py` — updated oracle and transferability evaluation suite.
- `visioncloak/project.py`, `visioncloak/errors.py` — minor metadata and exception updates.

### Surrogate Ensemble Strategy

The main weakness of UACloak is its narrow surrogate set. VisionCloak v1.0 expands this into a CPU-friendly ensemble:

Primary surrogates:

- `openai/clip-vit-large-patch14`
- `laion/CLIP-ViT-H-14-laion2B-s32B-b79K`
- `google/siglip-so400m-patch14-384`
- `facebook/dinov2-large`

Secondary diversity surrogates:

- `openai/clip-vit-base-patch16`
- `microsoft/swin-base-patch4-window7-224`

Design constraints:

- All surrogates load in eval mode.
- All parameters stay frozen.
- Only the perturbation delta receives gradients.
- CPU-only execution uses `torch.float32`.
- Models load lazily with `functools.lru_cache`.
- Users can choose subsets with a `--surrogates` CLI flag.
- The default full attack ensemble is `clip_l14 + siglip + dinov2`.

### Phase 2: Multi-Signal Loss Redesign

VisionCloak v1.0 combines several differentiable objectives instead of relying on one embedding loss:

1. Embedding divergence across every surrogate encoder.
2. Patch-token divergence for ViT-style models so local scene regions drift semantically, not just the global pooled representation.
3. Frequency-domain shaping with differentiable 2D DCT to push perturbation energy toward low-to-mid frequencies and away from fragile high-frequency noise.
4. Soft color-histogram divergence in both RGB and HSV space so color naming and palette cues become less reliable.
5. SSIM preservation as a soft perceptual constraint, with a default threshold of `0.92`.
6. L2 regularization on the perturbation to prevent unnecessary energy growth.

The composed objective is:

`L_total = sum_i w_i * L_embed,i + w_patch * L_patch + w_dct * L_dct + w_hist * L_hist + lambda_ssim * L_ssim + lambda_l2 * ||delta||^2`

Default tuning targets:

- `w_embed = 1.0` per surrogate
- `w_patch = 0.5`
- `w_dct = 0.3`
- `w_hist = 0.4`
- `lambda_ssim = 5.0`
- `lambda_l2 = 0.01`

## What Makes VisionCloak Different

- It is designed for multimodal LLM transfer, not just classic face or CLIP retrieval attack settings.
- It attacks local patch semantics instead of only final global embeddings.
- It explicitly accounts for JPEG and resize fragility by shaping perturbation frequency content.
- It treats color as a first-class semantic signal rather than a side effect.
- It preserves human-visible fidelity with an explicit SSIM floor instead of relying only on norm bounds.
- It remains CPU-first so the attack can run in constrained environments without requiring a GPU fleet.

## Current Repository Workflow

Until the package rename and engine migration land, the current commands still use `uacloak`:

```bash
# Install runtime and dev dependencies
uv sync --group dev

# Launch the local Gradio app
uv run app.py
```

Reproducible evaluation remains:

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

## Documentation

- VisionCloak v1.0 architecture overview: [docs/visioncloak_v1_overview.md](docs/visioncloak_v1_overview.md)

## Notebook Generation

Generate the technical walkthrough notebook from local fixtures:

```bash
uv run python scripts/generate_notebook.py
```

This writes `universal_cloaking_demo.ipynb`.

## Limitations

- The current checked-in implementation is still the UACloak baseline, so the VisionCloak v1.0 redesign described here is a specification and migration target, not yet the shipped engine.
- CPU-only execution of large surrogate ensembles will require careful batching, lazy loading, and subset selection to stay memory-safe.
- Transfer to closed production vision stacks should improve with the broader surrogate set, but no single open ensemble can guarantee perfect coverage.
- Stronger multi-signal perturbations still need careful tuning to preserve visual invisibility under resize and recompression.

## Deployment

Set a Hugging Face token in environment variable `UACLOAK_HF_TOKEN` (or `.env`) and run:

```bash
uv run python scripts/create_or_update_space.py a-01a/facecloak
```

## Privacy

The pipeline is local-first. No hosted inference calls are required for normal operation; model weights are downloaded and executed locally.
