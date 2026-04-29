"""Tests for the Gradio interface."""

from __future__ import annotations

import gradio as gr

from visioncloak.interface import build_demo, generate_cloak
from visioncloak.project import PROJECT_NAME


def test_build_demo_returns_gradio_blocks_with_project_title() -> None:
    demo = build_demo()
    assert isinstance(demo, gr.Blocks)
    assert demo.title == PROJECT_NAME


def test_generate_cloak_raises_gracefully_when_image_is_none() -> None:
    gen = generate_cloak(None, ["clip_l14", "siglip", "dinov2"], 0.05, 20, 0.25, True, True)
    try:
        next(gen)
        raise AssertionError("Expected gr.Error to be raised")
    except gr.Error as exc:
        assert "No image provided" in str(exc)


def test_interface_source_mentions_unified_flow_and_compression_tab() -> None:
    import inspect
    from visioncloak import interface

    source = inspect.getsource(interface)
    assert "Compression Test" in source
    assert "Surrogate Models" in source
    assert "one unified cloaking flow" in source

