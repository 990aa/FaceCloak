"""Gradio interface for the Phase 2 and 3 FaceCloak workflow."""

from __future__ import annotations

import gradio as gr

from facecloak.cloaking import CloakHyperparameters, cloak_face_tensor
from facecloak.environment import render_runtime_markdown
from facecloak.errors import FaceCloakError
from facecloak.pipeline import (
    cosine_similarity,
    detect_primary_face,
    extract_embedding_numpy,
)
from facecloak.project import (
    PHASE_LABEL,
    PHASE_STATUS,
    PHASE_SUMMARY,
    PROJECT_NAME,
    PROJECT_TAGLINE,
)

APP_CSS = """
.gradio-container {
    background:
        radial-gradient(circle at top left, rgba(180, 214, 255, 0.35), transparent 32%),
        radial-gradient(circle at top right, rgba(255, 212, 170, 0.30), transparent 28%),
        linear-gradient(135deg, #f6f2eb 0%, #eef4fb 100%);
}
#hero {
    padding-top: 0.75rem;
}
.panel {
    border: 1px solid rgba(32, 52, 84, 0.10);
    border-radius: 18px;
    background: rgba(255, 255, 255, 0.82);
    padding: 0.75rem 1rem;
    box-shadow: 0 18px 45px rgba(50, 70, 105, 0.08);
}
"""

APP_THEME = gr.themes.Base(
    primary_hue="amber",
    secondary_hue="blue",
    neutral_hue="slate",
    font=["Georgia", "Palatino Linotype", "serif"],
    font_mono=["Consolas", "Courier New", "monospace"],
)


def hero_markdown() -> str:
    return "\n".join(
        [
            f"# {PROJECT_NAME}",
            f"## {PROJECT_TAGLINE}",
            "",
            "A self-contained, locally-run tool for biometric privacy preservation.",
            "",
            f"**{PHASE_LABEL} status:** {PHASE_STATUS}",
            "",
            PHASE_SUMMARY,
        ]
    )


def roadmap_markdown() -> str:
    return "\n".join(
        [
            "### Current Workflow",
            "1. Detect and align the primary face with MTCNN",
            "2. Extract a 512-dimensional embedding with InceptionResnetV1",
            "3. Score cosine similarity for verification",
            "4. Run L-infinity PGD directly on the aligned face tensor",
            "5. Return the cloaked aligned face and a perturbation preview",
        ]
    )


def _format_detection_probability(probability: float | None) -> str:
    return "unknown" if probability is None else f"{probability:.4f}"


def compare_faces(image_a, image_b):
    try:
        if image_a is None or image_b is None:
            raise FaceCloakError("Please provide both images before running a similarity check.")

        detected_a = detect_primary_face(image_a)
        detected_b = detect_primary_face(image_b)
        embedding_a = extract_embedding_numpy(detected_a.tensor)
        embedding_b = extract_embedding_numpy(detected_b.tensor)
        similarity = cosine_similarity(embedding_a, embedding_b)
        summary = "\n".join(
            [
                "### Similarity Result",
                f"- Cosine Similarity: `{similarity:.4f}`",
                f"- Image A Detection Confidence: `{_format_detection_probability(detected_a.probability)}`",
                f"- Image B Detection Confidence: `{_format_detection_probability(detected_b.probability)}`",
                "- Interpretation: values near `1.0` indicate the same person; values near `0` or below indicate unrelated identities.",
            ]
        )
        return detected_a.image, detected_b.image, similarity, summary
    except FaceCloakError as exc:
        raise gr.Error(str(exc)) from exc


def cloak_face(image, epsilon, num_steps, l2_lambda):
    try:
        if image is None:
            raise FaceCloakError("Please upload an image before running the cloaking attack.")

        detected = detect_primary_face(image)
        result = cloak_face_tensor(
            detected.tensor,
            parameters=CloakHyperparameters(
                epsilon=float(epsilon),
                num_steps=int(num_steps),
                l2_lambda=float(l2_lambda),
            ),
        )
        summary = "\n".join(
            [
                "### Cloaking Result",
                f"- Detection Confidence: `{_format_detection_probability(detected.probability)}`",
                f"- Original Similarity: `{result.original_similarity:.4f}`",
                f"- Final Similarity: `{result.final_similarity:.4f}`",
                f"- Similarity Drop: `{result.similarity_drop:.4f}`",
                f"- Epsilon (L-inf): `{result.parameters.epsilon:.4f}`",
                f"- Alpha: `{result.parameters.alpha:.4f}`",
                f"- Steps: `{result.parameters.num_steps}`",
                f"- Delta L-inf: `{result.delta_l_inf:.4f}`",
                f"- Delta RMS: `{result.delta_rms:.4f}`",
            ]
        )
        return (
            result.original_face_image,
            result.cloaked_face_image,
            result.perturbation_preview,
            result.final_similarity,
            summary,
        )
    except FaceCloakError as exc:
        raise gr.Error(str(exc)) from exc


def build_demo() -> gr.Blocks:
    with gr.Blocks(title=PROJECT_NAME) as demo:
        gr.Markdown(hero_markdown(), elem_id="hero")

        with gr.Tab("Cloak Face"):
            with gr.Row():
                input_image = gr.Image(label="Input Portrait", type="pil")
                aligned_face = gr.Image(label="Aligned Face", type="pil")
                cloaked_face = gr.Image(label="Cloaked Face", type="pil")

            with gr.Row():
                perturbation = gr.Image(label="Perturbation Preview", type="pil")
                with gr.Column():
                    epsilon = gr.Slider(
                        minimum=0.01,
                        maximum=0.05,
                        value=0.03,
                        step=0.005,
                        label="Epsilon (L-inf budget)",
                    )
                    num_steps = gr.Slider(
                        minimum=10,
                        maximum=60,
                        value=30,
                        step=5,
                        label="PGD Steps",
                    )
                    l2_lambda = gr.Slider(
                        minimum=0.0,
                        maximum=0.05,
                        value=0.01,
                        step=0.005,
                        label="L2 Regularization Weight",
                    )
                    cloak_button = gr.Button("Run Cloaking Attack", variant="primary")
                    final_similarity = gr.Number(
                        label="Final Original-vs-Cloaked Similarity",
                        precision=4,
                    )
                    cloak_summary = gr.Markdown(elem_classes=["panel"])

            cloak_button.click(
                fn=cloak_face,
                inputs=[input_image, epsilon, num_steps, l2_lambda],
                outputs=[
                    aligned_face,
                    cloaked_face,
                    perturbation,
                    final_similarity,
                    cloak_summary,
                ],
            )

        with gr.Tab("Compare Faces"):
            with gr.Row():
                compare_image_a = gr.Image(label="Image A", type="pil")
                compare_image_b = gr.Image(label="Image B", type="pil")
            with gr.Row():
                aligned_a = gr.Image(label="Aligned Face A", type="pil")
                aligned_b = gr.Image(label="Aligned Face B", type="pil")
            compare_button = gr.Button("Compare Embeddings", variant="secondary")
            pair_similarity = gr.Number(label="Cosine Similarity", precision=4)
            pair_summary = gr.Markdown(elem_classes=["panel"])
            compare_button.click(
                fn=compare_faces,
                inputs=[compare_image_a, compare_image_b],
                outputs=[aligned_a, aligned_b, pair_similarity, pair_summary],
            )

        with gr.Tab("Diagnostics"):
            diagnostics = gr.Markdown(
                value=render_runtime_markdown(),
                elem_classes=["panel"],
            )
            refresh_button = gr.Button("Re-run environment check", variant="secondary")
            refresh_button.click(fn=render_runtime_markdown, outputs=diagnostics)

        with gr.Accordion("Pipeline Notes", open=False):
            gr.Markdown(roadmap_markdown(), elem_classes=["panel"])

    return demo


demo = build_demo()
