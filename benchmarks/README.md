# Benchmark Dataset Guide

This folder contains manifest files used by `eval.py` and `uacloak/benchmarking.py`.

It also contains ablation manifests used by `ablation.py`.

## Why this exists

The benchmark is designed to address the rigging objection directly:

- Cloaks are generated on lightweight surrogate models.
- Evaluation is performed on stronger, independent oracle models.
- Visual imperceptibility is verified with SSIM.

## Recommended curation protocol

1. Face benchmark (50-100 pairs):
   - Build pairs from Labeled Faces in the Wild (LFW).
   - Include same-identity and different-identity pairs.
2. General-image benchmark (50-100 pairs):
   - Use Oxford Buildings, INRIA Holidays, or your own object/scene pairs.
   - Mark true near-duplicate pairs using `pair_type=near_duplicate`.

## Manifest format

Required columns:

- `image_id`
- `modality` (`face` or `general`)
- `image_path`
- `reference_path`

Optional columns:

- `pair_type` (`standard` or `near_duplicate`)

## Ablation manifest format

Required columns:

- `image_id`
- `modality` (`face` or `general`)
- `image_path`

For full runs, use a fixed held-out set with exactly 40 rows:

- 20 face rows
- 20 general rows

The ablation runner enforces this by default. Use `--allow-small-set` only for quick smoke tests.

## Example

See `sample_manifest.csv` for a working minimal example that references local test fixtures.

For ablations, see `ablation_sample_manifest.csv`.

## Run

```powershell
python eval.py --manifest benchmarks/sample_manifest.csv --output-csv results/benchmark_metrics.csv --output-summary results/benchmark_summary.md
```

For a full benchmark, point `--manifest` to your 50-100 image-pair manifest.

For fixed-condition attack benchmarks (PGD + FGSM + runtime + robustness):

```powershell
python -m uacloak.benchmarking --manifest benchmarks/benchmarking_manifest.csv --output-csv results/benchmark_metrics.csv --output-summary results/benchmark_summary.md --output-json results/benchmark_summary.json
```

Run ablations:

```powershell
python ablation.py --manifest benchmarks/ablation_sample_manifest.csv --output-dir results/ablations --allow-small-set --skip-convnext
```
