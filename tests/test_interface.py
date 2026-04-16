import gradio as gr

from facecloak.interface import build_demo, hero_markdown, roadmap_markdown
from facecloak.project import PROJECT_NAME, PROJECT_TAGLINE


def test_hero_markdown_contains_project_identity() -> None:
    text = hero_markdown()

    assert PROJECT_NAME in text
    assert PROJECT_TAGLINE in text
    assert "Phase 1 status" in text


def test_roadmap_markdown_mentions_future_pipeline() -> None:
    text = roadmap_markdown()

    assert "Face detection and preprocessing" in text
    assert "Projected Gradient Descent" in text
    assert "Similarity verification" in text


def test_build_demo_returns_gradio_blocks_with_project_title() -> None:
    demo = build_demo()

    assert isinstance(demo, gr.Blocks)
    assert demo.title == PROJECT_NAME
