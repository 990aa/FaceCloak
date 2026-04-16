"""Gradio interface — FaceCloak adversarial privacy tool."""

from __future__ import annotations

import tempfile

import gradio as gr

from facecloak.cloaking import CloakHyperparameters, cloak_face_tensor
from facecloak.environment import render_runtime_markdown
from facecloak.errors import FaceCloakError
from facecloak.pipeline import (
    cosine_similarity,
    detect_primary_face,
    extract_embedding_numpy,
    interpret_score,
    verify_cloak,
)
from facecloak.project import PROJECT_NAME

# ---------------------------------------------------------------------------
# CSS — dark-on-light theme with explicit text colours for HF Spaces compat
# ---------------------------------------------------------------------------

APP_CSS = """
/* ── Google Font ─────────────────────────────────────────────── */
@import url('https://fonts.googleapis.com/css2?family=Inter:wght@300;400;500;600;700&display=swap');

/* ── Base reset ──────────────────────────────────────────────── */
* { box-sizing: border-box; }

body, .gradio-container {
    font-family: 'Inter', -apple-system, BlinkMacSystemFont, 'Segoe UI', sans-serif !important;
    color: #0f172a !important;
    background: #f8fafc !important;
}

/* ── Background gradient ──────────────────────────────────────── */
.gradio-container {
    background:
        radial-gradient(ellipse 80% 60% at 10% 0%, rgba(99, 102, 241, 0.08) 0%, transparent 55%),
        radial-gradient(ellipse 60% 50% at 90% 5%, rgba(244, 114, 182, 0.07) 0%, transparent 50%),
        linear-gradient(160deg, #f0f4ff 0%, #faf5ff 45%, #fff7f0 100%) !important;
    min-height: 100vh;
}

/* ── Force dark text everywhere ──────────────────────────────── */
label, .label-wrap, .block > label,
span, p, h1, h2, h3, h4, h5, h6,
.markdown, .markdown p, .markdown li,
.markdown h1, .markdown h2, .markdown h3,
.svelte-1ed2p3z, .svelte-1gfkn6j,
textarea, input, .gr-textbox textarea,
.gr-number input, .gr-slider label {
    color: #0f172a !important;
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
    background: rgba(255, 255, 255, 0.85) !important;
    border: 1px solid rgba(99, 102, 241, 0.12) !important;
    border-radius: 16px !important;
    box-shadow: 0 4px 24px rgba(99, 102, 241, 0.07) !important;
    padding: 1rem 1.25rem !important;
    color: #0f172a !important;
}
.panel *, .panel .markdown, .panel .markdown p, .panel .markdown li {
    color: #0f172a !important;
}

/* ── Accordion ───────────────────────────────────────────────── */
.accordion, details, details summary {
    background: rgba(248, 250, 252, 0.9) !important;
    color: #334155 !important;
    border-radius: 12px !important;
}
details *, details .markdown, details .markdown p {
    color: #334155 !important;
}

/* ── Sliders ─────────────────────────────────────────────────── */
.gr-slider label, input[type=range] + span,
.gr-slider .label-wrap span {
    color: #334155 !important;
    font-weight: 500;
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
    background: rgba(255,255,255,0.9) !important;
}

/* ── Images ──────────────────────────────────────────────────── */
.image-container img {
    border-radius: 12px !important;
    box-shadow: 0 4px 20px rgba(0,0,0,0.1) !important;
}

/* ── Tab strip ───────────────────────────────────────────────── */
.tab-nav button {
    color: #475569 !important;
    font-weight: 500 !important;
}
.tab-nav button.selected {
    color: #4f46e5 !important;
    border-bottom-color: #4f46e5 !important;
    font-weight: 600 !important;
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

    # ── Step 1: detect face ─────────────────────────────────────────────── #
    try:
        detected = detect_primary_face(image)
    except FaceCloakError as exc:
        raise gr.Error(str(exc)) from exc

    orig_embedding = extract_embedding_numpy(detected.tensor)

    # Compute baseline similarity to itself (should be ~1.0 / 100%)
    orig_sim = cosine_similarity(orig_embedding, orig_embedding)
    orig_label, _ = interpret_score(orig_sim)
    orig_score_text = _score_line("Original — before cloaking", orig_sim)

    # Accumulate progress lines for the status box
    progress_lines: list[str] = [
        "Face detected. Starting adversarial optimization…",
        f"  Detection confidence: {_format_detection_probability(detected.probability)}",
        "",
    ]

    # ── Step 2-N: PGD with per-step progress updates ─────────────────────  #
    params = CloakHyperparameters(
        epsilon=float(epsilon),
        alpha_fraction=float(alpha_fraction),
        num_steps=int(num_steps),
        l2_lambda=0.01,
    )

    # We'll store the result here once the loop finishes
    cloak_result_holder: list = []

    def _on_progress(step: int, total: int, sim: float) -> None:
        pct = sim * 100.0
        line = f"Step {step:>4d} of {total}: Match score → {pct:5.1f}%"
        progress_lines.append(line)

    # We can't yield inside the callback, so we run cloaking synchronously
    # and yield a progress snapshot every N steps via an intermediate
    # generator approach using a thread or simply run and batch-yield.
    #
    # For Gradio's streaming generator pattern we use a list that
    # the callback appends to, then drain it between yields.
    #
    # Practical approach: run the full loop collecting all histories,
    # then replay the progress lines as a stream on the final yield.
    # For a true real-time feel we split into chunks.

    REPORT_EVERY = max(1, int(num_steps) // 20)  # yield ~20 intermediate updates

    class _ProgressAccumulator:
        def __init__(self) -> None:
            self.lines: list[str] = []
            self.pending: list[str] = []

        def __call__(self, step: int, total: int, sim: float) -> None:
            line = f"Step {step:>4d} of {total}: Match score → {sim * 100.0:5.1f}%"
            self.lines.append(line)
            self.pending.append(line)

    acc = _ProgressAccumulator()

    try:
        result = cloak_face_tensor(
            detected.tensor,
            parameters=params,
            progress_callback=acc,
        )
    except FaceCloakError as exc:
        raise gr.Error(str(exc)) from exc

    # ── Re-play progress as a streaming generator ─────────────────────── #
    header_lines = progress_lines.copy()
    all_lines = header_lines + acc.lines

    # Stream intermediate updates in chunks
    chunk_size = max(1, len(acc.lines) // 20)
    status_so_far = "\n".join(header_lines) + "\n"
    for i in range(0, len(acc.lines), chunk_size):
        chunk = acc.lines[i : i + chunk_size]
        status_so_far += "\n".join(chunk) + "\n"
        yield (
            None,  # orig_img – not yet
            None,  # cloaked_img – not yet
            None,  # diff_img – not yet
            orig_score_text,
            "Optimizing…",
            status_so_far,
            None,  # download path – not yet
        )

    # ── Post-cloak verification (Step 21) ─────────────────────────────── #
    try:
        verification = verify_cloak(
            result.cloaked_face_image,
            orig_embedding,
        )
        cloak_score_text = _score_line(verification.label, verification.similarity)
    except FaceCloakError:
        # If re-detection fails on the cloaked image, use the PGD final sim
        cloak_sim = result.final_similarity
        cloak_label, _ = interpret_score(cloak_sim)
        cloak_score_text = _score_line(cloak_label, cloak_sim)
        verification = None

    # ── Step 33: partial-failure warning ─────────────────────────────── #
    final_sim = verification.similarity if verification else result.final_similarity
    warning_msg = ""
    if final_sim > 0.5:
        warning_msg = (
            "\n⚠️  Partial cloak achieved. "
            "Try increasing the number of steps or the epsilon value."
        )

    # ── Save cloaked image for download ───────────────────────────────── #
    tmp = tempfile.NamedTemporaryFile(suffix=".png", delete=False)
    result.cloaked_face_image.save(tmp.name, format="PNG")
    download_path = tmp.name

    status_final = (
        status_so_far
        + f"\nDone. Final match score: {final_sim * 100.0:.1f}%"
        + warning_msg
    )

    # ── Final yield with all outputs filled in ────────────────────────── #
    yield (
        detected.image,  # original aligned face
        result.cloaked_face_image,
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
            raise FaceCloakError(
                "Please provide both images before running a similarity check."
            )

        detected_a = detect_primary_face(image_a)
        detected_b = detect_primary_face(image_b)
        embedding_a = extract_embedding_numpy(detected_a.tensor)
        embedding_b = extract_embedding_numpy(detected_b.tensor)
        similarity = cosine_similarity(embedding_a, embedding_b)
        label, _ = interpret_score(similarity)
        summary = "\n".join(
            [
                "### Similarity Result",
                f"- Cosine Similarity: `{similarity:.4f}` ({_pct(similarity)})",
                f"- Verdict: **{label}**",
                f"- Image A Detection Confidence: `{_format_detection_probability(detected_a.probability)}`",
                f"- Image B Detection Confidence: `{_format_detection_probability(detected_b.probability)}`",
            ]
        )
        return detected_a.image, detected_b.image, similarity, summary
    except FaceCloakError as exc:
        raise gr.Error(str(exc)) from exc


# ---------------------------------------------------------------------------
# Build the Gradio app
# ---------------------------------------------------------------------------


def build_demo() -> gr.Blocks:  # noqa: C901
    with gr.Blocks(title=PROJECT_NAME) as demo:
        # ── Hero ─────────────────────────────────────────────────────── #
        gr.Markdown(
            """
# 🛡️ FaceCloak
## Upload your photo. Watch AI become blind to your face.

This tool uses adversarial mathematics to make microscopic pixel changes that are invisible to humans but completely scramble AI facial recognition systems.
""",
            elem_id="hero",
        )

        # ── Tabs ─────────────────────────────────────────────────────── #
        with gr.Tab("Generate Cloak"):
            # ── Input section ─────────────────────────────────────────── #
            gr.Markdown("### 📸 Step 1 — Upload Your Photo", elem_classes=["panel"])

            with gr.Row():
                with gr.Column(scale=1):
                    input_image = gr.Image(
                        label="Your Photo",
                        type="pil",
                        sources=["upload", "webcam", "clipboard"],
                        elem_id="input-image",
                    )

                    gr.Markdown(
                        "_If multiple faces are present, the largest face will be cloaked._",
                    )

                with gr.Column(scale=1):
                    gr.Markdown("### ⚙️ Step 2 — Adjust Settings (optional)")

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

                    with gr.Accordion("🔬 What do these settings mean?", open=False):
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
            gr.Markdown("### 🎯 Results", elem_classes=["panel"])

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
                    label="⬇️  Download Cloaked Image",
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

        # ── Compare Faces tab ─────────────────────────────────────────── #
        with gr.Tab("Compare Two Photos"):
            gr.Markdown(
                "Upload two photos to measure how similar they look to an AI facial recognition system.",
                elem_classes=["panel"],
            )
            with gr.Row():
                compare_image_a = gr.Image(label="Photo A", type="pil")
                compare_image_b = gr.Image(label="Photo B", type="pil")
            with gr.Row():
                aligned_a = gr.Image(
                    label="Detected Face A", type="pil", interactive=False
                )
                aligned_b = gr.Image(
                    label="Detected Face B", type="pil", interactive=False
                )
            compare_button = gr.Button("Compare Faces", variant="primary")
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

FaceCloak applies **Projected Gradient Descent (PGD)** directly to your image pixels.
It computes, for each pixel, the exact direction that will maximally confuse a FaceNet
recognition model, then nudges each pixel in that direction by a tiny, controlled amount.
The "projection" step keeps every pixel within a tight budget so the changes are
mathematically bounded to be imperceptible.

## Limitations

- **High-contrast or unusual lighting** can reduce cloak effectiveness.
- **Very low-resolution images** (under 160 × 160 px) may not cloak reliably.
- **Extreme head angles** (profile shots, 45°+) may reduce MTCNN detection confidence.
- **Glasses and heavy occlusion** can weaken the cloak because the embedding relies
  on periocular features that are partially hidden.
- This tool targets **FaceNet (VGGFace2)**. Other recognition models may behave differently.

## Privacy

Your image is processed entirely within this Hugging Face Space.
Nothing is stored, logged, or transmitted to any third party.
""",
                elem_classes=["panel"],
            )

    return demo


demo = build_demo()
