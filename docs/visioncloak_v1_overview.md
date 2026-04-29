# VisionCloak v1.0 Overview

## Purpose

VisionCloak v1.0 is the next-generation architecture for adversarial image cloaking in this repository. It evolves the original UACloak design from a FaceNet-plus-CLIP surrogate attack into a broader engine aimed at confusing the visual front ends of modern multimodal LLMs.

This document is a design reference for the migration from `uacloak/` to `visioncloak/`.

## Problem Statement

The current UACloak implementation is effective in a narrower setting:

- Face identity attacks via FaceNet.
- General semantic attacks via CLIP ViT-B/32.
- L-infinity PGD in pixel space.

That setup does not match how top-tier vision LLMs process images today. Modern multimodal systems often rely on:

- Larger encoders such as ViT-L/14, ViT-H, ViT-G, SigLIP, and InternViT-scale models.
- Patch-level token streams instead of a single pooled embedding.
- Multi-resolution pipelines using both thumbnails and tiled crops.
- JPEG and resize transforms before model ingestion.
- Holistic scene reasoning over colors, layout, relations, and object-level regions.

The redesign therefore treats adversarial cloaking as a multi-signal optimization problem instead of a single-surrogate embedding attack.

## Phase 1: Architecture Redesign

### Package Rename

Target rename:

- `uacloak/` -> `visioncloak/`

### Target Module Layout

- `visioncloak/models.py`
  Surrogate registry, model loading, caching, and feature extraction helpers.

- `visioncloak/engine.py`
  Main optimization loop replacing the current `cloaking.py`.

- `visioncloak/transforms.py`
  Resize, normalization, tiling, JPEG-aware transforms, and pre/post-processing utilities.

- `visioncloak/losses.py`
  Differentiable loss components with configurable weights.

- `visioncloak/pipeline.py`
  End-to-end orchestration for image routing and attack execution.

- `visioncloak/interface.py`
  Gradio UI updated for general image cloaking and richer attack diagnostics.

- `visioncloak/evaluation.py`
  Oracle evaluation and transfer benchmarking for the new ensemble.

- `visioncloak/project.py`, `visioncloak/errors.py`
  Minor metadata and error-layer updates.

## Surrogate Registry

### Primary Surrogates

These provide the core semantic attack surface and broad vendor coverage:

- `openai/clip-vit-large-patch14`
  ViT-L/14, widely representative of large CLIP-style image encoders used across multimodal stacks.

- `laion/CLIP-ViT-H-14-laion2B-s32B-b79K`
  Public ViT-H/14 CLIP-scale model for stronger global and regional semantic pressure.

- `google/siglip-so400m-patch14-384`
  SigLIP-family surrogate aligned with Gemini-adjacent image encoding strategies.

- `facebook/dinov2-large`
  Self-supervised large ViT model that helps capture structure and texture cues beyond CLIP-style alignment.

### Secondary Diversity Surrogates

- `openai/clip-vit-base-patch16`
  Adds patch-grid diversity relative to B/32 and L/14.

- `microsoft/swin-base-patch4-window7-224`
  Adds a windowed transformer with different locality biases from plain ViTs.

### CPU Loading Rules

- Use `torch.float32` throughout.
- Run models in eval mode.
- Freeze all parameters.
- Compute forwards under `torch.no_grad()` where gradients are not needed through parameters.
- Only the perturbation tensor should require gradients.
- Load models lazily via `functools.lru_cache`.
- Expose `--surrogates` so users can select subsets such as `clip_l14,siglip,dinov2`.

### Default Ensemble

Recommended default:

- `clip_l14`
- `siglip`
- `dinov2`

This gives broad cross-vendor coverage while remaining more realistic for CPU-only runs than always forcing the largest full ensemble.

## Phase 2: Loss Function Redesign

### 1. Embedding Divergence Loss

For each surrogate `M_i` with embedding function `f_i`:

`L_embed,i = -CosineSim(f_i(x_orig), f_i(x_adv))`

This generalizes the current CLIP-style objective to every encoder in the ensemble.

### 2. Patch-Level Embedding Divergence

Global pooled embeddings are not enough for vision LLM transfer. ViT-based surrogates should expose patch-token representations from a late transformer block so the attack also degrades local semantic consistency.

Implementation direction:

- Hook a late hidden-state output before the final projection head.
- Extract the full patch-token matrix of shape `[N_patches, D]`.
- Compute mean cosine divergence across aligned patch positions.

Why this matters:

- Vision LLMs attend to local regions when producing grounded descriptions such as object positions, clothing colors, or interactions between entities.
- A CLS-only attack can still leave region-level semantics largely intact.

### 3. Frequency-Domain Loss

Standard PGD perturbations tend to over-concentrate energy in brittle high-frequency noise. VisionCloak should bias perturbation structure toward low-to-mid frequencies using a differentiable 2D DCT.

Implementation direction:

- Compute a 2D DCT on each perturbation channel.
- Penalize excessive energy in very high-frequency coefficients.
- Encourage mid-band energy distribution rather than pure pixel-noise behavior.

Rationale:

- JPEG and resize pipelines often erase weak high-frequency artifacts.
- Low-to-mid-frequency structure is more likely to survive real ingestion pipelines and affect scene understanding.

### 4. Color Histogram Divergence

Color is a strong cue for captioning and semantic matching. VisionCloak should explicitly attack color statistics with differentiable soft histograms.

Implementation direction:

- Compute soft histograms in RGB and HSV space with Gaussian kernel binning.
- Treat each histogram as a probability distribution.
- Maximize divergence between original and adversarial distributions.

Why HSV matters:

- Hue is closely tied to how vision LLMs verbalize colors.
- A perturbation can preserve natural appearance while still shifting color naming confidence.

### 5. SSIM Preservation Constraint

Perceptual fidelity remains mandatory. VisionCloak should enforce:

`SSIM(x_orig, x_adv) >= tau`

with default:

- `tau = 0.92`

Soft penalty form:

`L_ssim = max(0, tau - SSIM(x_orig, x_adv)) * lambda_ssim`

Recommended implementation:

- Use `pytorch-msssim` for differentiable CPU-friendly SSIM.

### 6. L2 Regularization

Include a small perturbation-energy regularizer:

`lambda_l2 * ||delta||^2`

This keeps optimization from growing unnecessary energy when other losses are already effective.

## Total Objective

`L_total = sum_i w_i * L_embed,i + w_patch * L_patch + w_dct * L_dct + w_hist * L_hist + lambda_ssim * L_ssim + lambda_l2 * ||delta||^2`

Recommended default weights:

- `w_embed = 1.0` per surrogate
- `w_patch = 0.5`
- `w_dct = 0.3`
- `w_hist = 0.4`
- `lambda_ssim = 5.0`
- `lambda_l2 = 0.01`

## Expected Benefits

- Better transfer to closed multimodal systems by covering more encoder families.
- Better local-region disruption via patch-token objectives.
- Better robustness to recompression and resizing through frequency shaping.
- Better disruption of color-sensitive captioning and scene description.
- Stronger perceptual safeguards through explicit SSIM-aware optimization.

## Migration Notes

This document describes the target design, not the current shipped implementation. At the time of writing:

- The repository still uses the `uacloak/` package.
- Existing evaluation and UI commands still reference `uacloak`.
- Deployment still uses `UACLOAK_HF_TOKEN`.

The recommended rollout is:

1. Introduce the new `visioncloak/` package alongside the current baseline.
2. Port model loading into the new surrogate registry.
3. Split the current attack loop into `engine.py`, `losses.py`, and `transforms.py`.
4. Update evaluation and UI after the new engine is stable.
5. Retire or alias the legacy `uacloak/` paths only after parity and migration docs are complete.
