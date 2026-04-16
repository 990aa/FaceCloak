import gradio as gr
from PIL import Image

from facecloak.interface import build_demo, hero_markdown, roadmap_markdown
from facecloak.project import PROJECT_NAME, PROJECT_TAGLINE


def test_hero_markdown_contains_project_identity() -> None:
    text = hero_markdown()

    assert PROJECT_NAME in text
    assert PROJECT_TAGLINE in text
    assert "Phases 2-3 status" in text


def test_roadmap_markdown_mentions_current_pipeline() -> None:
    text = roadmap_markdown()

    assert "Detect and align the primary face with MTCNN" in text
    assert "Extract a 512-dimensional embedding" in text
    assert "Run L-infinity PGD" in text


def test_build_demo_returns_gradio_blocks_with_project_title() -> None:
    demo = build_demo()

    assert isinstance(demo, gr.Blocks)
    assert demo.title == PROJECT_NAME


def test_compare_faces_requires_both_images() -> None:
    from facecloak.interface import compare_faces

    blank = Image.new("RGB", (16, 16), "white")
    try:
        compare_faces(blank, None)
    except gr.Error as exc:
        assert "Please provide both images" in str(exc)
    else:
        raise AssertionError("compare_faces should reject missing inputs.")
