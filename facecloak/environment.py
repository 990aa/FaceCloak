"""Runtime diagnostics for the local and Hugging Face Space environments."""

from __future__ import annotations

from dataclasses import dataclass
from importlib.metadata import PackageNotFoundError, version
import platform

from facecloak.project import TORCH_CACHE_DIR


@dataclass(frozen=True, slots=True)
class RuntimeReport:
    python_version: str
    operating_system: str
    torch_version: str
    torchvision_version: str
    facenet_pytorch_version: str
    gradio_version: str
    huggingface_hub_version: str
    numpy_version: str
    pillow_version: str
    cuda_available: bool
    device: str
    tensor_sanity: tuple[int, ...]
    torch_cache_dir: str
    status: str
    notes: str


def _installed_version(distribution_name: str) -> str:
    try:
        return version(distribution_name)
    except PackageNotFoundError:
        return "not installed"


def collect_runtime_report() -> RuntimeReport:
    import torch

    tensor_sanity = tuple((torch.arange(3) + 1).tolist())
    cuda_available = bool(torch.cuda.is_available())
    device = "cuda" if cuda_available else "cpu"
    status = "ready" if tensor_sanity == (1, 2, 3) else "check failed"

    return RuntimeReport(
        python_version=platform.python_version(),
        operating_system=platform.platform(),
        torch_version=torch.__version__,
        torchvision_version=_installed_version("torchvision"),
        facenet_pytorch_version=_installed_version("facenet-pytorch"),
        gradio_version=_installed_version("gradio"),
        huggingface_hub_version=_installed_version("huggingface-hub"),
        numpy_version=_installed_version("numpy"),
        pillow_version=_installed_version("pillow"),
        cuda_available=cuda_available,
        device=device,
        tensor_sanity=tensor_sanity,
        torch_cache_dir=str(TORCH_CACHE_DIR),
        status=status,
        notes="CPU execution is the intended deployment target for the Hugging Face Space.",
    )


def format_runtime_markdown(report: RuntimeReport) -> str:
    cuda_label = "Yes" if report.cuda_available else "No"
    tensor_label = ", ".join(str(value) for value in report.tensor_sanity)

    return "\n".join(
        [
            "### Runtime Diagnostics",
            f"- Status: {report.status}",
            f"- Python: `{report.python_version}`",
            f"- Platform: `{report.operating_system}`",
            f"- Torch: `{report.torch_version}`",
            f"- Torchvision: `{report.torchvision_version}`",
            f"- facenet-pytorch: `{report.facenet_pytorch_version}`",
            f"- NumPy: `{report.numpy_version}`",
            f"- Pillow: `{report.pillow_version}`",
            f"- Gradio: `{report.gradio_version}`",
            f"- huggingface-hub: `{report.huggingface_hub_version}`",
            f"- CUDA Available: `{cuda_label}`",
            f"- Active Device: `{report.device}`",
            f"- Tensor Sanity: `[{tensor_label}]`",
            f"- Torch Cache: `{report.torch_cache_dir}`",
            f"- Notes: {report.notes}",
        ]
    )


def render_runtime_markdown() -> str:
    return format_runtime_markdown(collect_runtime_report())
