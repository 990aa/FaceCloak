## Benchmark Summary

### Fixed Benchmark Conditions

| Setting | Value |
| --- | --- |
| epsilon | 0.0300 |
| num_steps | 100 |
| alpha_fraction | 0.1000 |
| l2_lambda | 0.0100 |
| norm | linf |
| face_loss | combined_clip_plus_facenet |
| general_loss | clip_only |
| success_threshold | 0.3000 |

### Run Health

| Metric | Value |
| --- | --- |
| Rows | 2 |
| Valid Rows | 2 |
| Failed Rows | 0 |

### Face Benchmark: Attack Success Rate

| Metric | Value |
| --- | --- |
| Count | 1 |
| Success Rate (similarity < 0.30) | 0.0000 |
| Mean MRS | 0.8864 |
| MRS Std | 0.0000 |
| MRS P10 | 0.8864 |
| MRS P90 | 0.8864 |

### General Benchmark: Attack Success by Category

| Category | Count | Success Rate | Mean MRS | Mean Confidence Drop |
| --- | --- | --- | --- | --- |
| scene | 1 | 0.0000 | 0.8750 | 0.0838 |

### Perceptual Quality

| Metric | Value |
| --- | --- |
| Mean SSIM | 0.9339 |
| Min SSIM | 0.9175 |
| SSIM Pass Rate (>= 0.98) | 0.0000 |
| Mean PSNR (dB) | 36.2833 |
| Min PSNR (dB) | 34.7368 |
| PSNR Pass Rate (>= 35 dB) | 0.5000 |

### Robustness to Post-processing

| Condition | Mean Oracle Similarity |
| --- | --- |
| PGD (no post-processing) | 0.8807 |
| JPEG quality 90 | 0.8750 |
| Resize 50% then restore | 0.8786 |
| Gaussian blur sigma 0.5 | 0.8812 |

### Runtime Performance

| Stage | Mean Time (s) |
| --- | --- |
| Preprocess | 0.1400 |
| Detection | 0.8047 |
| Initial Embedding | 2.6821 |
| PGD Attack | 104.7456 |
| Verification | 22.5182 |
| Output Generation | 0.1078 |
| Total | 130.9983 |
| P90 Total | 167.6899 |
| Max Total | 176.8628 |
| Under 45s Rate | 0.0000 |

### Baseline Comparison: FGSM vs PGD

| Metric | Value |
| --- | --- |
| PGD Mean MRS | 0.8807 |
| FGSM Mean MRS | 0.8618 |
| PGD Success Rate | 0.0000 |
| FGSM Success Rate | 0.0000 |
| MRS Gap (FGSM - PGD) | -0.0188 |

### Notes and Limitations

- Mean SSIM is below the 0.98 imperceptibility target. Consider reducing epsilon and rerunning all benchmarks.
- Mean pipeline runtime exceeds the 45-second CPU target. Profile attack and oracle verification stages for optimization.
