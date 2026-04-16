"""Tests for environment diagnostics."""

from facecloak.environment import collect_runtime_report, render_runtime_markdown


def test_collect_runtime_report_returns_expected_values() -> None:
    report = collect_runtime_report()

    assert report.status == "ready"
    assert report.tensor_sanity == (1, 2, 3)
    assert report.device == ("cuda" if report.cuda_available else "cpu")
    assert report.torch_version.startswith("2.2.2")
    assert report.facenet_pytorch_version == "2.6.0"
    assert report.gradio_version == "6.12.0"
    assert report.huggingface_hub_version == "1.10.2"
    assert isinstance(report.transformers_version, str)
    assert len(report.transformers_version) > 0


def test_collect_runtime_report_has_no_phase_field() -> None:
    """RuntimeReport must not contain a phase field after the refactor."""
    report = collect_runtime_report()
    assert not hasattr(report, "phase")


def test_render_runtime_markdown_contains_environment_summary() -> None:
    markdown = render_runtime_markdown()

    assert "Runtime Diagnostics" in markdown
    assert "CUDA Available" in markdown
    assert "Torch Cache" in markdown
    assert "transformers" in markdown
    # Must NOT mention phase numbers
    assert "Phase 2" not in markdown
    assert "Phase 3" not in markdown
