"""Gradio interface for VisionCloak."""

from __future__ import annotations

import json
from pathlib import Path
import tempfile
from typing import Any

import gradio as gr
import numpy as np
from PIL import Image
import torch

from visioncloak.engine import CloakHyperparameters, cloak_general_image
from visioncloak.evaluation import evaluate_cloak_pair
from visioncloak.errors import VisionCloakError
from visioncloak.models import SURROGATE_SPECS, parse_surrogate_names
from visioncloak.pipeline import classify_image_type, ensure_rgb
from visioncloak.project import PROJECT_NAME
from visioncloak.transforms import (
    pil_to_unit_batch,
    save_dual_format_outputs,
    simulate_jpeg,
    unit_batch_to_pil,
)

APP_CSS = """
@import url('https://fonts.googleapis.com/css2?family=IBM+Plex+Sans:wght@300;400;500;600;700&display=swap');

:root {
    --vc-bg: #f4f0e8;
    --vc-surface: #fffdf8;
    --vc-text: #1c1917;
    --vc-muted: #57534e;
    --vc-border: #d6d3d1;
    --vc-accent: #9a3412;
}

body, .gradio-container {
    font-family: 'IBM Plex Sans', system-ui, sans-serif !important;
    background:
        radial-gradient(circle at top left, rgba(249, 115, 22, 0.10), transparent 30%),
        linear-gradient(180deg, #f7f2e8 0%, #efe7d7 100%) !important;
    color: var(--vc-text) !important;
}

#hero {
    background: linear-gradient(135deg, #7c2d12 0%, #9a3412 45%, #c2410c 100%);
    border-radius: 24px;
    padding: 2.25rem 2rem;
    box-shadow: 0 22px 50px rgba(124, 45, 18, 0.22);
    margin-bottom: 1.25rem;
}

#hero, #hero * {
    color: #fff7ed !important;
}

.panel {
    background: rgba(255, 253, 248, 0.92) !important;
    border: 1px solid var(--vc-border) !important;
    border-radius: 18px !important;
    box-shadow: 0 8px 24px rgba(28, 25, 23, 0.06) !important;
    padding: 1rem 1.25rem !important;
}

#cloak-btn {
    background: linear-gradient(135deg, var(--vc-accent), #ea580c) !important;
    color: white !important;
    border: none !important;
    border-radius: 14px !important;
    font-weight: 700 !important;
    padding: 0.9rem 1.4rem !important;
}
"""

APP_THEME = gr.themes.Base(
    primary_hue="orange",
    secondary_hue="amber",
    neutral_hue="stone",
    font=["IBM Plex Sans", "system-ui", "sans-serif"],
    font_mono=["Consolas", "Courier New", "monospace"],
)


def _load_image_from_path(path: str, field_name: str) -> Image.Image:
    image_path = Path(path)
    if not image_path.exists():
        raise VisionCloakError(
            f"{field_name} could not be loaded because the file does not exist."
        )
    with Image.open(image_path) as pil_image:
        return pil_image.convert("RGB")


def _coerce_image_input(image: Any, field_name: str) -> Image.Image:
    if image is None:
        raise VisionCloakError(f"{field_name} is missing.")
    if isinstance(image, Image.Image):
        return image.convert("RGB")
    if isinstance(image, str):
        return _load_image_from_path(image, field_name)
    if isinstance(image, dict):
        for key in ("path", "name"):
            candidate = image.get(key)
            if isinstance(candidate, str) and candidate:
                return _load_image_from_path(candidate, field_name)
        raise VisionCloakError(f"{field_name} payload is missing a usable file path.")
    if isinstance(image, np.ndarray):
        array = np.asarray(image)
        if array.ndim == 2:
            return Image.fromarray(np.clip(array, 0, 255).astype(np.uint8), mode="L").convert("RGB")
        if array.ndim == 3 and array.shape[2] in (3, 4):
            mode = "RGB" if array.shape[2] == 3 else "RGBA"
            return Image.fromarray(np.clip(array, 0, 255).astype(np.uint8), mode=mode).convert("RGB")
        raise VisionCloakError(f"{field_name} has unsupported array shape: {array.shape}.")
    raise VisionCloakError(f"{field_name} has unsupported type: {type(image).__name__}.")


def _surrogate_choices() -> list[str]:
    return list(SURROGATE_SPECS)


def _surrogate_descriptions_markdown() -> str:
    descriptions = {
        "clip_l14": "CLIP ViT-L/14 — primary large CLIP-style oracle target",
        "clip_h14": "CLIP ViT-H/14 — larger CLIP diversity surrogate",
        "siglip": "SigLIP So400M — Gemini-style image encoder family",
        "dinov2": "DINOv2 Large — structure and texture pressure",
        "clip_b16": "CLIP ViT-B/16 — patch-grid diversity",
        "swin": "Swin Base — shifted-window local structure surrogate",
    }
    lines = ["### Surrogate Coverage"]
    for name in _surrogate_choices():
        lines.append(f"- `{name}`: {descriptions.get(name, SURROGATE_SPECS[name].model_id)}")
    return "\n".join(lines)


def _format_surrogate_table(similarities: dict[str, float] | None) -> str:
    if not similarities:
        return "No surrogate scores available."
    lines = ["| Surrogate | Similarity | Drop |", "|---|---:|---:|"]
    for name, similarity in sorted(similarities.items()):
        lines.append(f"| `{name}` | `{similarity:.4f}` | `{1.0 - similarity:.4f}` |")
    return "\n".join(lines)


def _format_summary_markdown(
    evaluation,
    route_label: str,
    similarities: dict[str, float] | None,
) -> str:
    lines = [
        f"**Routing:** {route_label}",
        f"**SSIM:** `{evaluation.ssim_score:.4f}`",
        f"**Mean Oracle Cosine:** `{evaluation.mean_cosine_similarity:.4f}`",
        f"**Success:** `{'yes' if evaluation.success else 'no'}`",
        "",
        "### Oracle Breakdown",
    ]
    for name, score in evaluation.per_oracle.items():
        lines.append(f"- `{name}`: `{score:.4f}`")
    lines.extend(["", "### Surrogate Breakdown", _format_surrogate_table(similarities)])
    return "\n".join(lines)


def generate_cloak(
    image,
    surrogate_models,
    epsilon,
    num_steps,
    alpha_fraction,
    jpeg_augment,
    multi_resolution,
):
    try:
        image = _coerce_image_input(image, "Input image")
    except VisionCloakError as exc:
        raise gr.Error(
            "No image provided. Please upload a photo before clicking Generate Cloak."
        ) from exc

    image = ensure_rgb(image)
    route = classify_image_type(image)
    surrogates = parse_surrogate_names(",".join(surrogate_models))
    params = CloakHyperparameters(
        surrogates=tuple(surrogates),
        epsilon=float(epsilon),
        num_steps=int(num_steps),
        alpha_fraction=float(alpha_fraction),
        jpeg_augment=bool(jpeg_augment),
        multi_resolution=bool(multi_resolution),
    )

    progress_lines = [
        route.display_label,
        f"Selected surrogates: {', '.join(surrogates)}",
        "",
    ]

    class _ProgressAccumulator:
        def __init__(self) -> None:
            self.lines: list[str] = []

        def __call__(self, step: int, total: int, sim: float) -> None:
            self.lines.append(
                f"Step {step:>4d} of {total}: mean surrogate similarity -> {sim:.4f}"
            )

    accumulator = _ProgressAccumulator()
    result = cloak_general_image(
        image,
        parameters=params,
        progress_callback=accumulator,
    )

    chunk_size = max(1, len(accumulator.lines) // 20)
    status_so_far = "\n".join(progress_lines) + "\n"
    for start in range(0, len(accumulator.lines), chunk_size):
        status_so_far += "\n".join(accumulator.lines[start : start + chunk_size]) + "\n"
        yield (None, None, None, None, status_so_far, None, None, None)

    evaluation = evaluate_cloak_pair(image, result.cloaked_image)
    summary_markdown = _format_summary_markdown(
        evaluation,
        route.display_label,
        result.surrogate_similarities,
    )

    temp_dir = Path(tempfile.mkdtemp(prefix="visioncloak_"))
    png_path = temp_dir / "cloaked.png"
    jpeg_path = temp_dir / "cloaked_q95.jpg"
    summary_path = temp_dir / "summary.json"
    save_dual_format_outputs(result.cloaked_image, png_path=png_path, jpeg_path=jpeg_path)
    summary_path.write_text(
        json.dumps(
            {
                "success": evaluation.success,
                "ssim_score": evaluation.ssim_score,
                "mean_oracle_cosine": evaluation.mean_cosine_similarity,
                "per_oracle": evaluation.per_oracle,
                "surrogate_similarities": result.surrogate_similarities,
                "postprocess_metadata": result.postprocess_metadata,
            },
            indent=2,
            sort_keys=True,
        ),
        encoding="utf-8",
    )

    status_final = (
        status_so_far
        + f"\nDone. Mean oracle cosine: {evaluation.mean_cosine_similarity:.4f}"
        + f"\nSSIM: {evaluation.ssim_score:.4f}"
        + f"\nBinary success: {'yes' if evaluation.success else 'no'}"
    )

    yield (
        result.original_image,
        result.cloaked_image,
        result.amplified_diff,
        summary_markdown,
        status_final,
        str(png_path),
        str(jpeg_path),
        str(summary_path),
    )


def run_compression_test(image):
    try:
        cloaked = _coerce_image_input(image, "Cloaked image")
    except VisionCloakError as exc:
        raise gr.Error(str(exc)) from exc

    qualities = (75, 85, 95)
    original_batch = pil_to_unit_batch(cloaked).to(torch.float32)
    rows = ["| Quality | Self Similarity |", "|---|---:|"]
    preview_images: list[Image.Image] = []
    for quality in qualities:
        recompressed_batch = simulate_jpeg(original_batch, quality)
        recompressed_image = unit_batch_to_pil(recompressed_batch)
        preview_images.append(recompressed_image)
        evaluation = evaluate_cloak_pair(cloaked, recompressed_image)
        rows.append(f"| `Q{quality}` | `{evaluation.mean_cosine_similarity:.4f}` |")

    return (
        "\n".join(
            [
                "### Compression Stability",
                "Mean oracle cosine here compares the uploaded cloaked image against its recompressed variants.",
                "",
                *rows,
            ]
        ),
        preview_images[0],
        preview_images[1],
        preview_images[2],
    )


def build_demo() -> gr.Blocks:
    with gr.Blocks(title=PROJECT_NAME, theme=APP_THEME, css=APP_CSS) as demo:
        gr.Markdown(
            """
# VisionCloak
## Adversarial cloaking for modern vision-language systems

Patch-aware, JPEG-aware, multi-surrogate optimization designed to reduce transfer to large multimodal vision stacks while preserving human-visible image quality.
""",
            elem_id="hero",
        )

        gr.Markdown(
            """
### Scope

The app uses one unified cloaking flow for faces, objects, scenes, and documents. Face detection is optional and only augments the surrogate set internally when a strong face crop is available.
""",
            elem_classes=["panel"],
        )

        with gr.Tab("Generate Cloak"):
            with gr.Row():
                with gr.Column(scale=1):
                    input_image = gr.Image(
                        label="Input Image",
                        type="pil",
                        sources=["upload", "webcam", "clipboard"],
                    )
                with gr.Column(scale=1):
                    surrogate_models = gr.CheckboxGroup(
                        choices=_surrogate_choices(),
                        value=parse_surrogate_names(None),
                        label="Surrogate Models",
                        info="Choose the surrogate ensemble used during optimization.",
                    )
                    gr.Markdown(_surrogate_descriptions_markdown())
                    epsilon = gr.Slider(0.01, 0.10, value=0.05, step=0.005, label="Perturbation Strength")
                    num_steps = gr.Slider(20, 300, value=150, step=10, label="Optimization Steps")
                    alpha_fraction = gr.Slider(
                        0.05,
                        0.50,
                        value=0.25,
                        step=0.05,
                        label="Initial Step Size Fraction",
                    )
                    jpeg_augment = gr.Checkbox(
                        value=True,
                        label="JPEG Augmentation",
                        info="Simulate JPEG during optimization for stronger real-world transfer.",
                    )
                    multi_resolution = gr.Checkbox(
                        value=True,
                        label="Multi-Resolution Loss",
                        info="Add a downsampled loss term to reduce resolution-specific perturbations.",
                    )

            cloak_btn = gr.Button("Generate Cloak", variant="primary", elem_id="cloak-btn")
            status_box = gr.Textbox(
                label="Live Progress",
                lines=8,
                interactive=False,
                placeholder="Progress will appear here after you start the attack.",
            )

            with gr.Row():
                orig_img = gr.Image(label="Original Image", type="pil", interactive=False)
                cloaked_img = gr.Image(label="Cloaked Image", type="pil", interactive=False)

            diff_img = gr.Image(
                label="Amplified Perturbation View",
                type="pil",
                interactive=False,
            )
            summary_md = gr.Markdown(elem_classes=["panel"])
            png_download = gr.DownloadButton(label="Download PNG", variant="secondary")
            jpeg_download = gr.DownloadButton(label="Download JPEG (Q95)", variant="secondary")
            summary_download = gr.DownloadButton(label="Download JSON Summary", variant="secondary")

            cloak_btn.click(
                fn=generate_cloak,
                inputs=[
                    input_image,
                    surrogate_models,
                    epsilon,
                    num_steps,
                    alpha_fraction,
                    jpeg_augment,
                    multi_resolution,
                ],
                outputs=[
                    orig_img,
                    cloaked_img,
                    diff_img,
                    summary_md,
                    status_box,
                    png_download,
                    jpeg_download,
                    summary_download,
                ],
            )

        with gr.Tab("Compression Test"):
            gr.Markdown(
                "Upload a cloaked image to see how stable its oracle-space representation is after JPEG re-encoding at Q75, Q85, and Q95.",
                elem_classes=["panel"],
            )
            compression_input = gr.Image(label="Cloaked Image", type="pil")
            compression_button = gr.Button("Run Compression Test", variant="primary")
            compression_report = gr.Markdown(elem_classes=["panel"])
            with gr.Row():
                q75_preview = gr.Image(label="JPEG Q75", type="pil", interactive=False)
                q85_preview = gr.Image(label="JPEG Q85", type="pil", interactive=False)
                q95_preview = gr.Image(label="JPEG Q95", type="pil", interactive=False)
            compression_button.click(
                fn=run_compression_test,
                inputs=[compression_input],
                outputs=[compression_report, q75_preview, q85_preview, q95_preview],
            )

    return demo


demo = build_demo()
