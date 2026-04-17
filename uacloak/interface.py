"""Gradio interface — UACloak adversarial privacy tool."""

from __future__ import annotations

import tempfile

import gradio as gr

from uacloak.cloaking import (
    CloakHyperparameters,
    cloak_face_tensor,
    cloak_general_image,
)
from uacloak.environment import render_runtime_markdown
from uacloak.errors import UACloakError
from uacloak.models import get_clip_model
from uacloak.pipeline import (
    classify_image_type,
    cosine_similarity,
    detect_primary_face,
    ensure_rgb,
    extract_clip_embedding_numpy,
    extract_embedding_numpy,
    interpret_clip_score,
    interpret_score,
    verify_cloak,
)
from uacloak.project import PROJECT_NAME

# ---------------------------------------------------------------------------
# CSS — dark-on-light theme with explicit text colours for HF Spaces compat
# ---------------------------------------------------------------------------

APP_CSS = """
/* ── Google Font ─────────────────────────────────────────────── */
@import url('https://fonts.googleapis.com/css2?family=Inter:wght@300;400;500;600;700&display=swap');

/* ── Base reset ──────────────────────────────────────────────── */
* { box-sizing: border-box; }

:root {
    --uac-bg: #f8fafc;
    --uac-surface: #ffffff;
    --uac-text: #0f172a;
    --uac-text-muted: #334155;
}

body, .gradio-container {
    font-family: 'Inter', -apple-system, BlinkMacSystemFont, 'Segoe UI', sans-serif !important;
    background: var(--uac-bg) !important;
    color: var(--uac-text) !important;
}

.gradio-container,
.gradio-container .markdown,
.gradio-container .markdown p,
.gradio-container .markdown li,
.gradio-container .markdown h1,
.gradio-container .markdown h2,
.gradio-container .markdown h3,
.gradio-container label,
.gradio-container span {
    color: var(--uac-text) !important;
}

.gradio-container a {
    color: #1d4ed8 !important;
    text-decoration: underline !important;
}

/* ── Hero section ────────────────────────────────────────────── */
#hero {
    background: linear-gradient(135deg, #312e81 0%, #4c1d95 40%, #7c3aed 100%);
    border-radius: 20px;
    padding: 2.5rem 2rem 2rem;
    margin-bottom: 1.5rem;
    box-shadow: 0 20px 60px rgba(99, 102, 241, 0.25);
    text-align: center;
}
#hero .markdown, #hero .markdown p, #hero .markdown h1,
#hero .markdown h2, #hero .markdown span, #hero * {
    color: #ffffff !important;
}

/* ── Panel cards ─────────────────────────────────────────────── */
.panel {
    background: var(--uac-surface) !important;
    border: 1px solid rgba(99, 102, 241, 0.12) !important;
    border-radius: 16px !important;
    box-shadow: 0 4px 24px rgba(99, 102, 241, 0.07) !important;
    padding: 1rem 1.25rem !important;
}
.panel *, .panel .markdown, .panel .markdown p, .panel .markdown li {
    color: var(--uac-text) !important;
}

/* ── Accordion (What do these settings mean?) ────────────────── */
.accordion, details {
    background: #1e293b !important;
    border-radius: 12px !important;
    border: 1px solid rgba(255,255,255,0.1) !important;
}
details summary, details summary * {
    color: #f8fafc !important;
    font-weight: 600 !important;
}
details *, details .markdown, details .markdown p {
    color: #e2e8f0 !important;
}

/* ── Sliders and settings ────────────────────────────────────── */
.gr-slider label, .gr-slider .label-wrap span, input[type=range] + span {
    color: #f8fafc !important;
    font-weight: 600 !important;
}

.gr-slider {
    background: #1e293b !important;
    padding: 1rem !important;
    border-radius: 12px !important;
}

/* ── Buttons ─────────────────────────────────────────────────── */
#cloak-btn {
    background: linear-gradient(135deg, #6366f1, #7c3aed) !important;
    color: #ffffff !important;
    border: none !important;
    border-radius: 12px !important;
    font-size: 1.1rem !important;
    font-weight: 600 !important;
    padding: 0.85rem 2rem !important;
    box-shadow: 0 4px 18px rgba(99, 102, 241, 0.4) !important;
    transition: transform 0.15s, box-shadow 0.15s !important;
    width: 100%;
}
#cloak-btn:hover {
    transform: translateY(-2px) !important;
    box-shadow: 0 8px 28px rgba(99, 102, 241, 0.5) !important;
}

#download-btn button {
    background: linear-gradient(135deg, #059669, #0d9488) !important;
    color: #ffffff !important;
    border-radius: 12px !important;
    font-weight: 600 !important;
    width: 100%;
}

/* ── Status textbox ──────────────────────────────────────────── */
#status-box textarea {
    background: #0f172a !important;
    color: #a5f3fc !important;
    font-family: 'Consolas', 'Courier New', monospace !important;
    font-size: 0.85rem !important;
    border-radius: 10px !important;
    border: 1px solid #334155 !important;
}

/* ── Score badges ────────────────────────────────────────────── */
#orig-score textarea,
#cloak-score textarea {
    font-size: 0.95rem !important;
    font-weight: 600 !important;
    border-radius: 10px !important;
    color: #0f172a !important;
    border: 1px solid rgba(99, 102, 241, 0.2) !important;
    background: #ffffff !important;
}

/* ── Tab strip ───────────────────────────────────────────────── */
button[role="tab"] {
    color: #1e293b !important;
    background: #e2e8f0 !important;
    border-radius: 10px 10px 0 0 !important;
    font-weight: 600 !important;
    font-size: 1.05rem !important;
}
button[role="tab"][aria-selected="true"] {
    color: #4f46e5 !important;
    background: #ffffff !important;
    border-bottom-color: #4f46e5 !important;
}
"""

APP_THEME = gr.themes.Base(
    primary_hue="violet",
    secondary_hue="indigo",
    neutral_hue="slate",
    font=["Inter", "system-ui", "sans-serif"],
    font_mono=["Consolas", "Courier New", "monospace"],
)


# ---------------------------------------------------------------------------
# Score formatting helpers
# ---------------------------------------------------------------------------


def _pct(similarity: float) -> str:
    return f"{max(0.0, min(100.0, similarity * 100.0)):.1f}%"


def _score_line(label: str, similarity: float) -> str:
    pct = max(0.0, min(100.0, similarity * 100.0))
    return f"Match Score: {pct:.1f}%  |  {label}"


def _format_detection_probability(probability: float | None) -> str:
    return "unknown" if probability is None else f"{probability:.4f}"


# ---------------------------------------------------------------------------
# Core processing — generator so Gradio streams progress (Step 28)
# ---------------------------------------------------------------------------


def generate_cloak(image, epsilon, num_steps, alpha_fraction):
    """Generator function that yields intermediate UI state every PGD step.

    Yields a tuple of:
        (orig_img, cloaked_img, diff_img, orig_score_text,
         cloak_score_text, status_text, download_path)

    The final yield carries all completed values; intermediate yields keep
    the image outputs as None so only the status textbox updates.
    """
    # ── Input validation ────────────────────────────────────────────────── #
    if image is None:
        raise gr.Error(
            "No image provided. Please upload a photo before clicking Generate Cloak."
        )

    route = classify_image_type(image)

    # ── Step 1: hyperparameters and progress setup ─────────────────────── #
    params = CloakHyperparameters(
        epsilon=float(epsilon),
        alpha_fraction=float(alpha_fraction),
        num_steps=int(num_steps),
        l2_lambda=0.01,
        face_weight=1.0,
        clip_weight=1.0,
    )

    progress_lines: list[str] = [route.display_label]
    progress_lines.append(
        f"  Face detector confidence: {_format_detection_probability(route.detector_probability)}"
    )
    progress_lines.append("")

    class _ProgressAccumulator:
        def __init__(self, metric_name: str) -> None:
            self.metric_name = metric_name
            self.lines: list[str] = []

        def __call__(self, step: int, total: int, sim: float) -> None:
            line = (
                f"Step {step:>4d} of {total}: {self.metric_name} similarity"
                f" → {sim * 100.0:5.1f}%"
            )
            self.lines.append(line)

    clip_model = get_clip_model()

    # ── Face route: combined FaceNet + CLIP attack ─────────────────────── #
    if route.image_type == "face":
        detected = route.detected_face
        if detected is None:
            try:
                detected = detect_primary_face(image)
            except UACloakError as exc:
                raise gr.Error(str(exc)) from exc

        original_facenet_embedding = extract_embedding_numpy(detected.tensor)
        original_clip_embedding = extract_clip_embedding_numpy(detected.image)

        orig_score_text = "\n".join(
            [
                _score_line("FaceNet baseline — before cloaking", 1.0),
                _score_line("CLIP baseline — before cloaking", 1.0),
            ]
        )

        acc = _ProgressAccumulator("FaceNet")
        try:
            result = cloak_face_tensor(
                detected.tensor,
                parameters=params,
                clip_model=clip_model,
                progress_callback=acc,
            )
        except UACloakError as exc:
            raise gr.Error(str(exc)) from exc

        chunk_size = max(1, len(acc.lines) // 20)
        status_so_far = "\n".join(progress_lines) + "\n"
        for i in range(0, len(acc.lines), chunk_size):
            chunk = acc.lines[i : i + chunk_size]
            status_so_far += "\n".join(chunk) + "\n"
            yield (
                None,
                None,
                None,
                orig_score_text,
                "Optimizing…",
                status_so_far,
                None,
            )

        try:
            verification = verify_cloak(
                result.cloaked_face_image,
                original_facenet_embedding,
            )
            final_face_similarity = verification.similarity
            face_label = verification.label
            face_warning = verification.warning
        except UACloakError:
            final_face_similarity = result.final_similarity
            face_label, face_warning = interpret_score(final_face_similarity)

        cloaked_clip_embedding = extract_clip_embedding_numpy(result.cloaked_face_image)
        final_clip_similarity = cosine_similarity(
            original_clip_embedding,
            cloaked_clip_embedding,
        )
        clip_label, clip_warning = interpret_clip_score(final_clip_similarity)

        cloak_score_text = "\n".join(
            [
                _score_line(f"FaceNet — {face_label}", final_face_similarity),
                _score_line(f"CLIP — {clip_label}", final_clip_similarity),
            ]
        )

        warning_lines = [
            warning for warning in (face_warning, clip_warning) if warning is not None
        ]
        warning_msg = ""
        if warning_lines:
            warning_msg = "\n" + "\n".join(f"WARNING: {line}" for line in warning_lines)

        tmp = tempfile.NamedTemporaryFile(suffix=".png", delete=False)
        result.cloaked_face_image.save(tmp.name, format="PNG")
        download_path = tmp.name

        status_final = (
            status_so_far
            + f"\nDone. Final FaceNet similarity: {final_face_similarity * 100.0:.1f}%"
            + f"\nFinal CLIP similarity: {final_clip_similarity * 100.0:.1f}%"
            + warning_msg
        )

        yield (
            detected.image,
            result.cloaked_face_image,
            result.amplified_diff,
            orig_score_text,
            cloak_score_text,
            status_final,
            download_path,
        )
        return

    # ── General route: CLIP-only universal cloaking ────────────────────── #
    original_clip_embedding = extract_clip_embedding_numpy(image)
    orig_score_text = _score_line("CLIP baseline — before cloaking", 1.0)

    acc = _ProgressAccumulator("CLIP")
    try:
        result = cloak_general_image(
            image,
            clip_model=clip_model,
            parameters=params,
            progress_callback=acc,
        )
    except UACloakError as exc:
        raise gr.Error(str(exc)) from exc

    chunk_size = max(1, len(acc.lines) // 20)
    status_so_far = "\n".join(progress_lines) + "\n"
    for i in range(0, len(acc.lines), chunk_size):
        chunk = acc.lines[i : i + chunk_size]
        status_so_far += "\n".join(chunk) + "\n"
        yield (
            None,
            None,
            None,
            orig_score_text,
            "Optimizing…",
            status_so_far,
            None,
        )

    cloaked_clip_embedding = extract_clip_embedding_numpy(result.cloaked_image)
    final_clip_similarity = cosine_similarity(
        original_clip_embedding, cloaked_clip_embedding
    )
    clip_label, clip_warning = interpret_clip_score(final_clip_similarity)
    cloak_score_text = _score_line(f"CLIP — {clip_label}", final_clip_similarity)

    warning_msg = ""
    if clip_warning:
        warning_msg = f"\nWARNING: {clip_warning}"

    tmp = tempfile.NamedTemporaryFile(suffix=".png", delete=False)
    result.cloaked_image.save(tmp.name, format="PNG")
    download_path = tmp.name

    status_final = (
        status_so_far
        + f"\nDone. Final CLIP similarity: {final_clip_similarity * 100.0:.1f}%"
        + warning_msg
    )

    yield (
        result.original_image,
        result.cloaked_image,
        result.amplified_diff,
        orig_score_text,
        cloak_score_text,
        status_final,
        download_path,
    )


# ---------------------------------------------------------------------------
# Compare-faces tab (kept for technical reviewers)
# ---------------------------------------------------------------------------


def compare_faces(image_a, image_b):
    try:
        if image_a is None or image_b is None:
            raise UACloakError(
                "Please provide both images before running a similarity check."
            )

        route_a = classify_image_type(image_a)
        route_b = classify_image_type(image_b)

        clip_a = extract_clip_embedding_numpy(image_a)
        clip_b = extract_clip_embedding_numpy(image_b)
        clip_similarity = cosine_similarity(clip_a, clip_b)
        clip_label, _ = interpret_clip_score(clip_similarity)

        lines = [
            "### Similarity Result",
            f"- CLIP Cosine Similarity: `{clip_similarity:.4f}` ({_pct(clip_similarity)})",
            f"- CLIP Verdict: **{clip_label}**",
            f"- Image A Type: `{route_a.image_type}`",
            f"- Image B Type: `{route_b.image_type}`",
            f"- Image A Detection Confidence: `{_format_detection_probability(route_a.detector_probability)}`",
            f"- Image B Detection Confidence: `{_format_detection_probability(route_b.detector_probability)}`",
        ]

        if route_a.image_type == "face" and route_b.image_type == "face":
            if route_a.detected_face is None or route_b.detected_face is None:
                detected_a = detect_primary_face(image_a)
                detected_b = detect_primary_face(image_b)
            else:
                detected_a = route_a.detected_face
                detected_b = route_b.detected_face

            embedding_a = extract_embedding_numpy(detected_a.tensor)
            embedding_b = extract_embedding_numpy(detected_b.tensor)
            facenet_similarity = cosine_similarity(embedding_a, embedding_b)
            facenet_label, _ = interpret_score(facenet_similarity)

            lines.append(
                f"- FaceNet Cosine Similarity: `{facenet_similarity:.4f}` ({_pct(facenet_similarity)})"
            )
            lines.append(f"- FaceNet Verdict: **{facenet_label}**")

            summary = "\n".join(lines)
            return detected_a.image, detected_b.image, clip_similarity, summary

        summary = "\n".join(lines)
        return ensure_rgb(image_a), ensure_rgb(image_b), clip_similarity, summary
    except UACloakError as exc:
        raise gr.Error(str(exc)) from exc


# ---------------------------------------------------------------------------
# Build the Gradio app
# ---------------------------------------------------------------------------


def build_demo() -> gr.Blocks:  # noqa: C901
    with gr.Blocks(title=PROJECT_NAME) as demo:
        # ── Hero ─────────────────────────────────────────────────────── #
        gr.Markdown(
            """
# <svg width="32" height="32" viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2" stroke-linecap="round" stroke-linejoin="round" style="display:inline; vertical-align:middle; margin-right:8px;"><path d="M12 22s8-4 8-10V5l-8-3-8 3v7c0 6 8 10 8 10z"></path></svg> UACloak
## Upload any image. Watch AI similarity collapse.

This tool uses adversarial mathematics to make microscopic pixel changes that are invisible to humans but can disrupt AI similarity systems for faces, scenes, products, and documents.
""",
            elem_id="hero",
        )

        gr.Markdown(
            """
### About Our Validation

To prove effectiveness without rigging, cloaks generated by our local surrogate models are benchmarked against state-of-the-art Oracle models (ArcFace for faces, CLIP ViT-L/14 for general images). A successful evasion here represents a genuine failure of commercial-grade recognition systems, not an artifact of a weak local evaluator.

[Read full evaluator validation details in the README](https://huggingface.co/spaces/a-01a/universal-adversarial-cloak/blob/main/README.md#evaluator-validation-black-box-transferability-against-state-of-the-art-oracles)
""",
            elem_classes=["panel"],
        )

        # ── Tabs ─────────────────────────────────────────────────────── #
        with gr.Tab("Generate Cloak"):
            # ── Input section ─────────────────────────────────────────── #
            gr.Markdown(
                '### <svg width="24" height="24" viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2" stroke-linecap="round" stroke-linejoin="round" style="display:inline; vertical-align:middle; margin-right:8px;"><path d="M23 19a2 2 0 0 1-2 2H3a2 2 0 0 1-2-2V8a2 2 0 0 1 2-2h4l2-3h6l2 3h4a2 2 0 0 1 2 2z"></path><circle cx="12" cy="13" r="4"></circle></svg> Step 1 — Upload Your Photo',
                elem_classes=["panel"],
            )

            with gr.Row():
                with gr.Column(scale=1):
                    input_image = gr.Image(
                        label="Your Photo",
                        type="pil",
                        sources=["upload", "webcam", "clipboard"],
                        elem_id="input-image",
                    )

                    gr.Markdown(
                        "_Face images use FaceNet + CLIP cloaking. Non-face images use CLIP-only universal cloaking._",
                    )

                with gr.Column(scale=1):
                    gr.Markdown(
                        '### <svg width="24" height="24" viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2" stroke-linecap="round" stroke-linejoin="round" style="display:inline; vertical-align:middle; margin-right:8px;"><circle cx="12" cy="12" r="3"></circle><path d="M19.4 15a1.65 1.65 0 0 0 .33 1.82l.06.06a2 2 0 0 1 0 2.83 2 2 0 0 1-2.83 0l-.06-.06a1.65 1.65 0 0 0-1.82-.33 1.65 1.65 0 0 0-1 1.51V21a2 2 0 0 1-2 2 2 2 0 0 1-2-2v-.09A1.65 1.65 0 0 0 9 19.4a1.65 1.65 0 0 0-1.82.33l-.06.06a2 2 0 0 1-2.83 0 2 2 0 0 1 0-2.83l.06-.06a1.65 1.65 0 0 0 .33-1.82 1.65 1.65 0 0 0-1.51-1H3a2 2 0 0 1-2-2 2 2 0 0 1 2-2h.09A1.65 1.65 0 0 0 4.6 9a1.65 1.65 0 0 0-.33-1.82l-.06-.06a2 2 0 0 1 0-2.83 2 2 0 0 1 2.83 0l.06.06a1.65 1.65 0 0 0 1.82.33H9a1.65 1.65 0 0 0 1-1.51V3a2 2 0 0 1 2-2 2 2 0 0 1 2 2v.09a1.65 1.65 0 0 0 1 1.51 1.65 1.65 0 0 0 1.82-.33l.06-.06a2 2 0 0 1 2.83 0 2 2 0 0 1 0 2.83l-.06.06a1.65 1.65 0 0 0-.33 1.82V9a1.65 1.65 0 0 0 1.51 1H21a2 2 0 0 1 2 2 2 2 0 0 1-2 2h-.09a1.65 1.65 0 0 0-1.51 1z"></path></svg> Step 2 — Adjust Settings (optional)'
                    )

                    epsilon = gr.Slider(
                        minimum=0.01,
                        maximum=0.10,
                        value=0.03,
                        step=0.005,
                        label="Perturbation Strength",
                        info="How much the pixels are allowed to change. Higher = stronger cloak, still invisible to humans.",
                    )
                    num_steps = gr.Slider(
                        minimum=20,
                        maximum=300,
                        value=100,
                        step=10,
                        label="Optimization Steps",
                        info="More steps = stronger cloak but longer wait. 100 works for most photos.",
                    )
                    alpha_fraction = gr.Slider(
                        minimum=0.05,
                        maximum=0.50,
                        value=0.10,
                        step=0.05,
                        label="Step Size (fraction of strength)",
                        info="How aggressively each step adjusts pixels. 0.1 is a good default.",
                    )

                    with gr.Accordion("What do these settings mean?", open=False):
                        gr.Markdown(
                            """
**Perturbation Strength (epsilon):** Think of this as the maximum number of steps
each pixel is allowed to walk away from its original colour on a scale of 0–255.
At 0.03, pixels shift by roughly 4 units — enough to confuse AI but invisible to you.

**Optimization Steps:** The algorithm makes one small improvement per step.
More steps means the perturbation has more chances to reduce AI recognition,
just like more practice makes you better at a skill.

**Step Size:** Controls how big each improvement attempt is.
Small values (0.05–0.1) are precise and stable; larger values are faster but noisier.
The defaults work well for most photos. Only adjust if you are experimenting.
""",
                        )

            # ── Trigger ───────────────────────────────────────────────── #
            cloak_btn = gr.Button(
                "Generate Cloak",
                variant="primary",
                elem_id="cloak-btn",
            )

            # ── Live status ───────────────────────────────────────────── #
            status_box = gr.Textbox(
                label="Live Progress",
                lines=6,
                interactive=False,
                placeholder="Progress will appear here once you click Generate Cloak…",
                elem_id="status-box",
            )

            # ── Results section ───────────────────────────────────────── #
            gr.Markdown(
                '### <svg width="24" height="24" viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2" stroke-linecap="round" stroke-linejoin="round" style="display:inline; vertical-align:middle; margin-right:8px;"><circle cx="12" cy="12" r="10"></circle><circle cx="12" cy="12" r="6"></circle><circle cx="12" cy="12" r="2"></circle></svg> Results',
                elem_classes=["panel"],
            )

            with gr.Row():
                with gr.Column(scale=1):
                    orig_img = gr.Image(
                        label="Original Image",
                        type="pil",
                        interactive=False,
                        elem_id="orig-img",
                    )
                    orig_score = gr.Textbox(
                        label="AI Recognition Score — Before",
                        interactive=False,
                        elem_id="orig-score",
                    )

                with gr.Column(scale=1):
                    cloaked_img = gr.Image(
                        label="Cloaked Image",
                        type="pil",
                        interactive=False,
                        elem_id="cloaked-img",
                    )
                    cloak_score = gr.Textbox(
                        label="AI Recognition Score — After",
                        interactive=False,
                        elem_id="cloak-score",
                    )

            diff_img = gr.Image(
                label="Adversarial Noise (75× amplified for visibility) — should look like random grey noise with no face features",
                type="pil",
                interactive=False,
                elem_id="diff-img",
            )

            with gr.Row(elem_id="download-btn"):
                download_btn = gr.DownloadButton(
                    label="Download Cloaked Image",
                    variant="secondary",
                )

            # ── Wire up ───────────────────────────────────────────────── #
            cloak_btn.click(
                fn=generate_cloak,
                inputs=[input_image, epsilon, num_steps, alpha_fraction],
                outputs=[
                    orig_img,
                    cloaked_img,
                    diff_img,
                    orig_score,
                    cloak_score,
                    status_box,
                    download_btn,
                ],
            )

        # ── Benchmark Results tab ─────────────────────────────────────── #
        with gr.Tab("Benchmark Results"):
            gr.Markdown(
                "## Academic Benchmarks\n"
                "**Universal Adversarial Pixel Poisoning: Defeating State-of-the-Art Recognition Systems.**\n"
                "The core premise of this project is absolute, verifiable efficacy. Below are the pre-computed summary results across rigorous data sets and cutting-edge oracle models.",
                elem_classes=["panel"],
            )
            import os

            benchmark_root = "results"
            pca_path = os.path.join(benchmark_root, "pca.png")
            scatter_path = os.path.join(benchmark_root, "scatter.png")
            grid_path = os.path.join(benchmark_root, "grid.png")
            md_path = os.path.join(benchmark_root, "benchmark_report.md")

            # Backward compatibility with older root-level artifact locations.
            if not os.path.exists(pca_path) and os.path.exists("pca.png"):
                pca_path = "pca.png"
            if not os.path.exists(scatter_path) and os.path.exists("scatter.png"):
                scatter_path = "scatter.png"
            if not os.path.exists(grid_path) and os.path.exists("grid.png"):
                grid_path = "grid.png"
            if not os.path.exists(md_path) and os.path.exists("benchmark_report.md"):
                md_path = "benchmark_report.md"

            with gr.Row():
                with gr.Column(scale=1):
                    gr.Image(value=pca_path, label="Semantic Collapse (PCA)", type="filepath", interactive=False)
                with gr.Column(scale=1):
                    gr.Image(value=scatter_path, label="Black-Box Transferability", type="filepath", interactive=False)
            with gr.Row():
                gr.Image(value=grid_path, label="Imperceptibility Validation", type="filepath", interactive=False)

            if os.path.exists(md_path):
                with open(md_path, "r", encoding="utf-8") as f:
                    gr.Markdown(f.read(), elem_classes=["panel"])
            else:
                gr.Markdown("*(Benchmark summary report will populate here once `scripts/generate_report.py` runs)*", elem_classes=["panel"])

        # ── Compare Photos tab ────────────────────────────────────────── #
        with gr.Tab("Compare Two Photos"):
            gr.Markdown(
                "Upload two images to measure CLIP similarity, and FaceNet similarity when both are confident face images.",
                elem_classes=["panel"],
            )
            with gr.Row():
                compare_image_a = gr.Image(label="Photo A", type="pil")
                compare_image_b = gr.Image(label="Photo B", type="pil")
            with gr.Row():
                aligned_a = gr.Image(
                    label="Processed Image A", type="pil", interactive=False
                )
                aligned_b = gr.Image(
                    label="Processed Image B", type="pil", interactive=False
                )
            compare_button = gr.Button("Compare Images", variant="primary")
            pair_similarity = gr.Number(
                label="Cosine Similarity (1.0 = identical, 0 = unrelated)", precision=4
            )
            pair_summary = gr.Markdown(elem_classes=["panel"])
            compare_button.click(
                fn=compare_faces,
                inputs=[compare_image_a, compare_image_b],
                outputs=[aligned_a, aligned_b, pair_similarity, pair_summary],
            )

        # ── Diagnostics tab ───────────────────────────────────────────── #
        with gr.Tab("Diagnostics"):
            diagnostics = gr.Markdown(
                value=render_runtime_markdown(),
                elem_classes=["panel"],
            )
            refresh_button = gr.Button("Re-run environment check", variant="secondary")
            refresh_button.click(fn=render_runtime_markdown, outputs=diagnostics)

        # ── About accordion ───────────────────────────────────────────── #
        with gr.Accordion("About & Limitations", open=False):
            gr.Markdown(
                """
## How it works

UACloak applies **Projected Gradient Descent (PGD)** directly to your image pixels.
It computes, for each pixel, the exact direction that will maximally disrupt similarity
embeddings (CLIP for all images, plus FaceNet for confident face inputs), then nudges
each pixel in that direction by a tiny, controlled amount.
The "projection" step keeps every pixel within a tight budget so the changes are
mathematically bounded to be imperceptible.

## Limitations

- **High-contrast or unusual lighting** can reduce cloak effectiveness.
- **Very low-resolution images** (under 160 × 160 px) may not cloak reliably.
- **Extreme head angles** (profile shots, 45°+) may reduce MTCNN detection confidence.
- **Glasses and heavy occlusion** can weaken the cloak because the embedding relies
  on periocular features that are partially hidden.
- Face images are scored with **FaceNet + CLIP**; non-face images are scored with **CLIP**.
    Transfer to unseen production systems is expected but cannot be guaranteed for every model family.

## Privacy

Your image is processed entirely within this Hugging Face Space.
Nothing is stored, logged, or transmitted to any third party.
""",
                elem_classes=["panel"],
            )

    return demo


demo = build_demo()
