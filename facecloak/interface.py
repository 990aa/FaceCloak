"""Gradio interface for the Phase 1 FaceCloak scaffold."""

from __future__ import annotations

import gradio as gr

from facecloak.environment import render_runtime_markdown
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
    background: rgba(255, 255, 255, 0.78);
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
            "### Planned Pipeline",
            "1. Face detection and preprocessing",
            "2. Embedding extraction with a frozen face recognition model",
            "3. Projected Gradient Descent to optimize pixel-space perturbations",
            "4. Similarity verification against the original embedding",
            "5. Export and comparison workflow inside Gradio",
        ]
    )


def build_demo() -> gr.Blocks:
    with gr.Blocks(title=PROJECT_NAME) as demo:
        gr.Markdown(hero_markdown(), elem_id="hero")

        with gr.Row():
            diagnostics = gr.Markdown(
                value=render_runtime_markdown(),
                elem_classes=["panel"],
            )

        refresh_button = gr.Button("Re-run environment check", variant="primary")
        refresh_button.click(fn=render_runtime_markdown, outputs=diagnostics)

        with gr.Accordion("What lands in later phases", open=False):
            gr.Markdown(roadmap_markdown(), elem_classes=["panel"])

    return demo


demo = build_demo()
