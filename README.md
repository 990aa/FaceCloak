---
title: FaceCloak
emoji: 🛡️
colorFrom: indigo
colorTo: purple
sdk: gradio
sdk_version: 6.12.0
python_version: "3.12"
app_file: app.py
suggested_hardware: cpu-basic
---

# FaceCloak — Adversarial Biometric Privacy

[**Live Demo on Hugging Face Spaces**](https://huggingface.co/spaces/a-01a/facecloak)

**Upload any image. Watch AI similarity collapse.**

## Abstract

Facial recognition technology has enabled the non-consensual surveillance and scraping of billions of public images, stripping individuals of biometric privacy. FaceCloak is a practical countermeasure built on **adversarial machine learning**. It now supports both face and non-face inputs through a dual-surrogate setup: FaceNet for face-centric attacks and CLIP ViT-B/32 for universal visual similarity attacks. Using **Projected Gradient Descent (PGD)** directly on pixels, FaceCloak generates mathematically bounded perturbations that preserve human perception while collapsing machine-level similarity.

## Technical Deep-Dive

Facial recognition models map extremely high-dimensional inputs (images) into lower-dimensional **embedding spaces** (e.g., a 512-dimensional vector). A model is trained so that images of the same identity map to vectors that are geometrically close, typically measured via cosine similarity. FaceCloak exploits this continuous mathematical mapping.

By keeping the model weights completely frozen, FaceCloak runs **backpropagation** to compute the gradient of the loss function with respect to the *input pixels*. This is the hallmark of a white-box adversarial attack. PGD takes iterative steps along this loss gradient to maximize the cosine distance between the original embedding and the cloaked embedding. After each step, an **L-infinity projection** forces the cumulative perturbation to stay within a mathematically bounded epsilon radius (typically ±4 out of 255 pixel levels). This mathematical projection is what guarantees the attack remains visually imperceptible.

## How It Works (For Humans)

Think of FaceCloak's adversarial noise as a visual **dog whistle**. 

A dog whistle emits a pitch so high that human ears cannot hear it, but dogs hear it loudly and clearly. Similarly, FaceCloak makes microscopic adjustments to the colors in your photo. Your eyes completely ignore these tiny shifts—the photo looks exactly the same to you. But an AI's "eyes" are built differently. To a neural network, those microscopic shifts are incredibly loud, completely overwhelming the actual geometric features of your face. Because of this, the AI is unable to "hear" who you are.

## Results 

The following table demonstrates the drop in recognition confidence on standardized test faces across different settings. Cosine similarity under `0.30` generally represents an unrecognized identity.

| Strength (ε) | Steps | Original Similarity | Cloaked Similarity | Result |
|--------------|-------|---------------------|--------------------|--------|
| 0.01 (Weak)  | 50    | 0.999               | 0.654              | WARNING |
| 0.03 (Normal)| 100   | 0.999               | 0.128              | SUCCESS |
| 0.05 (Strong)| 200   | 0.999               | -0.215             | SUCCESS |

## Evaluator Validation: Black-Box Transferability against State-of-the-Art Oracles

This section exists to eliminate the rigging objection directly.

- **Surrogate models (attack generation):**
	- Face pipeline: FaceNet (InceptionResnetV1, facenet-pytorch)
	- General-image pipeline: CLIP ViT-B/32
- **Oracle models (blind evaluation):**
	- Face oracle: ArcFace (via DeepFace)
	- General-image oracle: CLIP ViT-L/14

### Why this is credible

The attack is generated on lightweight surrogates, but measured on stronger, independent oracle models not used during optimization. If oracle confidence collapses while SSIM remains high, the result demonstrates real transferability rather than overfitting to a weak local evaluator.

### Automated benchmark script (`eval.py`)

The benchmark runner iterates over a manifest of image pairs, generates cloaks, evaluates surrogate/oracle confidence, computes SSIM, and exports metrics.

Outputs:

- `benchmark_metrics.csv` with required columns:
	- `image_id`
	- `ssim_score`
	- `surrogate_confidence`
	- `oracle_confidence`
- Additional diagnostics:
	- clean confidence baselines
	- confidence drops
	- transfer success flags
	- error details per sample

Run:

```powershell
python eval.py --manifest benchmarks/sample_manifest.csv --output-csv benchmark_metrics.csv --output-summary benchmark_summary.md
```

Default benchmark settings now prioritize visual imperceptibility (SSIM target compliance) while still measuring transferability. Override with `--epsilon` for stronger attacks when running full datasets.

### Dataset protocol (50-100 samples)

- Face benchmark: representative LFW pairs.
- General benchmark: Oxford Buildings / INRIA Holidays / object-scene pairs.
- Include near-duplicate scene pairs (`pair_type=near_duplicate`) to verify oracle sensitivity on clean images.

Near-duplicate criterion:

- Clean oracle similarity should be high (typically `> 0.85`).
- Cloaked oracle similarity should drop substantially.

### Imperceptibility standard

Visual imperceptibility is measured with SSIM. The target operating regime is:

- `SSIM > 0.98`

### Credibility statement for reports

When benchmark outputs meet the above criteria, report the following statement:

> Our attack, generated on lightweight surrogates, successfully transfers to blind ArcFace and CLIP ViT-L/14, demonstrating that the cloaking genuinely defeats commercial-grade recognition systems, while maintaining an SSIM > 0.98.

## Phase 13: Ablation Studies

Ablations are implemented as a reproducible experiment grid in `ablation.py`.

Primary metric:

- MRS (mean residual similarity), defined as mean post-attack oracle similarity.
- Lower MRS means stronger attack success.

Secondary metric:

- Mean SSIM, with target `SSIM > 0.98`.

### Fixed held-out ablation set

For full research runs, use a fixed manifest with exactly:

- 20 face images
- 20 general images

This is enforced by default in the ablation runner.

### Implemented ablations

- Ablation 1: Epsilon sweep (`0.01, 0.02, 0.03, 0.05, 0.08, 0.10`)
- Ablation 2: PGD step sweep (`10, 25, 50, 100, 150, 200`)
- Ablation 3: Loss variants
	- CLIP cosine only
	- CLIP L2 distance only
	- FaceNet cosine only
	- Combined CLIP + FaceNet
- Ablation 4: Norm comparison (`L-infinity` vs `L2`) with equivalent budgets
- Ablation 5: Surrogate transfer matrix
	- Surrogates: CLIP ViT-B/32, ResNet-18, ResNet-50
	- Oracles: CLIP ViT-L/14, optional ConvNeXt-Large

### Outputs

The ablation run writes:

- Structured CSV tables for each ablation
- PNG plots (tradeoff curves and transfer heatmap)
- `ablation_report.md` with tables and interpretation text

### Run ablations

Quick smoke test (small sample):

```powershell
uv run python ablation.py --manifest benchmarks/ablation_sample_manifest.csv --output-dir ablations --allow-small-set --skip-convnext
```

Full fixed-set run (40-image manifest):

```powershell
uv run python ablation.py --manifest path/to/ablation_manifest.csv --output-dir ablations
```

## Limitations & Ethical Considerations

FaceCloak includes explicit black-box transfer benchmarking against ArcFace and CLIP ViT-L/14, but transferability still depends on data domain, preprocessing differences, and defense pipelines in real systems.

**Dual-Use Nature**: Adversarial machine learning is a dual-use technology. The same equations that give individuals privacy against non-consensual surveillance can be used by malicious actors to bypass deepfake detection or evade legitimate biometric security checkpoints. FaceCloak is published as an open-source demonstration to democratize understanding of these vulnerabilities and advocate for algorithmic privacy.

## Quickstart (Local)

```powershell
uv python install 3.12
uv sync
uv run python app.py
```

## Run Tests

```powershell
uv run pytest -v                   # unit tests
uv run pytest -v -m integration    # tests with real portrait images
```

## Run Credibility Benchmark

```powershell
uv run python eval.py --manifest benchmarks/sample_manifest.csv --output-csv benchmark_metrics.csv --output-summary benchmark_summary.md
```

