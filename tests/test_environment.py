from facecloak.environment import collect_runtime_report, render_runtime_markdown
from facecloak.project import PHASE_LABEL


def test_collect_runtime_report_returns_expected_values() -> None:
    report = collect_runtime_report()

    assert report.phase == PHASE_LABEL
    assert report.status == "ready"
    assert report.tensor_sanity == (1, 2, 3)
    assert report.device == ("cuda" if report.cuda_available else "cpu")
    assert report.torch_version.startswith("2.2.2")
    assert report.facenet_pytorch_version == "2.6.0"
    assert report.gradio_version == "6.12.0"


def test_render_runtime_markdown_contains_environment_summary() -> None:
    markdown = render_runtime_markdown()

    assert "Runtime Diagnostics" in markdown
    assert "CUDA Available" in markdown
    assert "Tensor Sanity" in markdown
    assert "Environment setup, dependency locking" in markdown
