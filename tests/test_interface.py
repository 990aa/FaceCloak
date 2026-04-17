"""Tests for the Gradio interface."""

from __future__ import annotations

import gradio as gr
from PIL import Image

from uacloak.interface import build_demo
from uacloak.project import PROJECT_NAME


def test_build_demo_returns_gradio_blocks_with_project_title() -> None:
    demo = build_demo()

    assert isinstance(demo, gr.Blocks)
    assert demo.title == PROJECT_NAME


def test_generate_cloak_raises_gracefully_when_image_is_none() -> None:
    """generate_cloak must raise gr.Error for missing input (Step 30)."""
    from uacloak.interface import generate_cloak

    gen = generate_cloak(None, 0.03, 20, 0.1)
    try:
        next(gen)
        raise AssertionError("Expected gr.Error to be raised")
    except gr.Error as exc:
        assert "No image provided" in str(exc)


def test_compare_faces_requires_both_images() -> None:
    from uacloak.interface import compare_faces

    blank = Image.new("RGB", (16, 16), "white")
    try:
        compare_faces(blank, None)
    except gr.Error as exc:
        assert "Please provide both images" in str(exc)
    else:
        raise AssertionError("compare_faces should reject missing inputs.")


def test_interface_has_no_phase_references() -> None:
    """The interface markdown copy must not mention phase numbers."""
    from uacloak.interface import build_demo

    demo = build_demo()
    # The demo object doesn't expose raw text easily; check the source module
    import inspect
    from uacloak import interface

    source = inspect.getsource(interface)
    # Should not contain "Phase 2", "Phase 3" etc. in user-facing strings
    import re

    matches = re.findall(r"\bPhase [0-9]\b", source)
    # Allow zero matches
    assert len(matches) == 0, f"Found phase number references in interface: {matches}"


def test_interface_mentions_oracle_validation_copy() -> None:
    import inspect
    from uacloak import interface

    source = inspect.getsource(interface)
    assert "About Our Validation" in source
    assert "ArcFace" in source
    assert "CLIP ViT-L/14" in source
