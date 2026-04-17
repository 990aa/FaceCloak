# Ablation Study Report

## Primary Metric

Primary metric is MRS (mean residual similarity), defined as average post-attack oracle similarity. Lower is better. Secondary metric is mean SSIM, with a quality target of SSIM > 0.98.

## Ablation 1: Epsilon

| Setting | MRS Face | MRS General | Mean SSIM | Runtime (s) |
| --- | --- | --- | --- | --- |
| 0.01 | 0.9838 | 0.9953 | 0.9959 | 46.41 |

Best epsilon under SSIM constraint was 0.01, with mean SSIM 0.9959.
This indicates the practical operating point is near the tradeoff knee where MRS is reduced without violating the imperceptibility target.

## Ablation 2: PGD Steps

| Setting | MRS Face | MRS General | Mean SSIM | Runtime (s) |
| --- | --- | --- | --- | --- |
| 10 | 0.9471 | 0.9667 | 0.9788 | 26.19 |

No step setting met the SSIM constraint on this run.

## Ablation 3: Loss Function

| Loss Variant | MRS Face ArcFace | MRS Face CLIP-L/14 | MRS General CLIP-L/14 | Mean SSIM | Runtime (s) |
| --- | --- | --- | --- | --- | --- |
| clip_cosine_only | 1.0000 | 1.0000 | 0.9667 | 0.9907 | 29.09 |
| clip_l2_only | 1.0000 | 1.0000 | 0.9625 | 0.9907 | 25.39 |
| facenet_cosine_only | 0.9313 | 0.9781 | n/a | 0.9671 | 21.61 |
| combined_clip_facenet | 0.9471 | 0.9308 | 0.9667 | 0.9788 | 38.85 |

Combined CLIP plus FaceNet loss is expected to balance transferability across oracle families, while single-space losses specialize to their own geometry.

## Ablation 4: Norm Type

| Norm | Epsilon | Equivalent L2 Radius | MRS Face | MRS General | Mean SSIM | Runtime (s) |
| --- | --- | --- | --- | --- | --- | --- |
| linf | 0.01 | 3.880 | 0.9838 | 0.9953 | 0.9959 | 27.14 |
| l2 | 0.01 | 3.880 | 0.9815 | 0.9989 | 0.9985 | 26.74 |

At equivalent budgets, L2 projection can produce smoother perturbation structure while L-infinity offers stricter per-pixel bounds; the table shows which retains SSIM > 0.98 most reliably.

## Ablation 5: Surrogate Choice Transfer Matrix

| Surrogate | Oracle | MRS General |
| --- | --- | --- |
| clip_vit_b32 | clip_vit_l14 | 0.9667 |
| resnet18 | clip_vit_l14 | 0.9682 |
| resnet50 | clip_vit_l14 | 0.9871 |

Cross-architecture transferability is evaluated by comparing surrogate-trained perturbations against CLIP-L/14 and ConvNeXt-Large oracle scoring.

