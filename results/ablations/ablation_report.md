# Ablation Study Report

## Primary Metric

Primary metric is MRS (mean residual similarity), defined as average post-attack oracle similarity. Lower is better. Secondary metric is mean SSIM, with a quality target of SSIM > 0.98.

## Ablation 1: Epsilon

| Setting | MRS Face | MRS General | Mean SSIM | Runtime (s) |
| --- | --- | --- | --- | --- |
| 0.01 | 0.9619 | 0.9313 | 0.9878 | 829.95 |
| 0.03 | 0.8893 | 0.8935 | 0.9349 | 2326.17 |
| 0.05 | 0.7334 | 0.8169 | 0.8562 | 1167.25 |

Best epsilon under SSIM constraint was 0.01, with mean SSIM 0.9878.
This indicates the practical operating point is near the tradeoff knee where MRS is reduced without violating the imperceptibility target.

## Ablation 2: PGD Steps

| Setting | MRS Face | MRS General | Mean SSIM | Runtime (s) |
| --- | --- | --- | --- | --- |
| 25 | 0.9233 | 0.8793 | 0.9556 | 233.60 |
| 50 | 0.9139 | 0.8922 | 0.9429 | 359.67 |
| 100 | 0.8893 | 0.8935 | 0.9349 | 535.30 |

No step setting met the SSIM constraint on this run.

## Ablation 3: Loss Function

| Loss Variant | MRS Face ArcFace | MRS Face CLIP-L/14 | MRS General CLIP-L/14 | Mean SSIM | Runtime (s) |
| --- | --- | --- | --- | --- | --- |
| clip_cosine_only | 0.9930 | 0.9649 | 0.8935 | 0.9520 | 502.04 |
| clip_l2_only | 1.0000 | 1.0000 | 0.8910 | 0.9586 | 581.03 |
| facenet_cosine_only | 0.8942 | 0.9240 | n/a | 0.9488 | 234.12 |
| combined_clip_facenet | 0.8893 | 0.8715 | 0.8935 | 0.9349 | 598.52 |

Combined CLIP plus FaceNet loss is expected to balance transferability across oracle families, while single-space losses specialize to their own geometry.

## Ablation 4: Norm Type

| Norm | Epsilon | Equivalent L2 Radius | MRS Face | MRS General | Mean SSIM | Runtime (s) |
| --- | --- | --- | --- | --- | --- | --- |
| linf | 0.01 | 3.880 | 0.9619 | 0.9313 | 0.9878 | 642.86 |
| linf | 0.03 | 11.639 | 0.8893 | 0.8935 | 0.9349 | 659.98 |
| l2 | 0.01 | 3.880 | 0.9855 | 0.9668 | 0.9983 | 785.36 |
| l2 | 0.03 | 11.639 | 0.9886 | 0.9775 | 0.9983 | 4934.21 |

At equivalent budgets, L2 projection can produce smoother perturbation structure while L-infinity offers stricter per-pixel bounds; the table shows which retains SSIM > 0.98 most reliably.

## Ablation 5: Surrogate Choice Transfer Matrix

| Surrogate | Oracle | MRS General |
| --- | --- | --- |
| clip_vit_b32 | clip_vit_l14 | 0.8935 |
| resnet18 | clip_vit_l14 | 0.9231 |
| resnet50 | clip_vit_l14 | 0.9308 |

Cross-architecture transferability is evaluated by comparing surrogate-trained perturbations against CLIP-L/14 and ConvNeXt-Large oracle scoring.

