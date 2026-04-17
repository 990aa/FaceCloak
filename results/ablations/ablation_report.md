# Ablation Study Report

## Primary Metric

Primary metric is MRS (mean residual similarity), defined as average post-attack oracle similarity. Lower is better. Secondary metric is mean SSIM, with a quality target of SSIM > 0.98.

## Ablation 1: Epsilon

| Setting | MRS Face | MRS General | Mean SSIM | Runtime (s) |
| --- | --- | --- | --- | --- |
| 0.03 | 0.9544 | 0.9132 | 0.9531 | 156.35 |

No epsilon setting met the SSIM constraint on this run.

## Ablation 2: PGD Steps

| Setting | MRS Face | MRS General | Mean SSIM | Runtime (s) |
| --- | --- | --- | --- | --- |
| 25 | 0.9544 | 0.9132 | 0.9531 | 213.33 |

No step setting met the SSIM constraint on this run.

## Ablation 3: Loss Function

| Loss Variant | MRS Face ArcFace | MRS Face CLIP-L/14 | MRS General CLIP-L/14 | Mean SSIM | Runtime (s) |
| --- | --- | --- | --- | --- | --- |
| clip_cosine_only | 1.0000 | 1.0000 | 0.9132 | 0.9714 | 137.35 |
| clip_l2_only | 1.0000 | 1.0000 | 0.9181 | 0.9720 | 173.03 |
| facenet_cosine_only | 0.9122 | 0.9492 | n/a | 0.9555 | 198.78 |
| combined_clip_facenet | 0.9544 | 0.8474 | 0.9132 | 0.9531 | 442.51 |

Combined CLIP plus FaceNet loss is expected to balance transferability across oracle families, while single-space losses specialize to their own geometry.

## Ablation 4: Norm Type

| Norm | Epsilon | Equivalent L2 Radius | MRS Face | MRS General | Mean SSIM | Runtime (s) |
| --- | --- | --- | --- | --- | --- | --- |
| linf | 0.03 | 11.639 | 0.9544 | 0.9132 | 0.9531 | 290.55 |
| l2 | 0.03 | 11.639 | 0.9793 | 0.9664 | 0.9980 | 252.70 |

At equivalent budgets, L2 projection can produce smoother perturbation structure while L-infinity offers stricter per-pixel bounds; the table shows which retains SSIM > 0.98 most reliably.

## Ablation 5: Surrogate Choice Transfer Matrix

| Surrogate | Oracle | MRS General |
| --- | --- | --- |
| clip_vit_b32 | clip_vit_l14 | 0.9132 |
| resnet18 | clip_vit_l14 | 0.9490 |
| resnet50 | clip_vit_l14 | 0.9642 |

Cross-architecture transferability is evaluated by comparing surrogate-trained perturbations against CLIP-L/14 and ConvNeXt-Large oracle scoring.

