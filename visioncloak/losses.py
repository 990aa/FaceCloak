"""Composable differentiable losses for VisionCloak."""

from __future__ import annotations

from functools import lru_cache
import math

import torch
import torch.nn.functional as F

from visioncloak.transforms import dct_2d, rgb_to_hsv


def embedding_attack_term(
    original_embedding: torch.Tensor, adversarial_embedding: torch.Tensor
) -> torch.Tensor:
    cosine = F.cosine_similarity(original_embedding, adversarial_embedding, dim=1)
    return -cosine.mean()


def patch_attack_term(
    original_tokens: torch.Tensor | None, adversarial_tokens: torch.Tensor | None
) -> torch.Tensor:
    if original_tokens is None or adversarial_tokens is None:
        reference = original_tokens if original_tokens is not None else adversarial_tokens
        device = torch.device("cpu") if reference is None else reference.device
        dtype = torch.float32 if reference is None else reference.dtype
        return torch.zeros((), device=device, dtype=dtype)

    min_tokens = min(original_tokens.shape[1], adversarial_tokens.shape[1])
    if min_tokens == 0:
        return torch.zeros(
            (), device=original_tokens.device, dtype=original_tokens.dtype
        )

    original = original_tokens[:, :min_tokens]
    adversarial = adversarial_tokens[:, :min_tokens]
    cosine = F.cosine_similarity(original, adversarial, dim=2)
    return -cosine.mean()


def frequency_attack_term(delta: torch.Tensor) -> torch.Tensor:
    coefficients = dct_2d(delta)
    _, _, height, width = coefficients.shape

    y = torch.linspace(0.0, 1.0, height, device=delta.device, dtype=delta.dtype)
    x = torch.linspace(0.0, 1.0, width, device=delta.device, dtype=delta.dtype)
    grid_y, grid_x = torch.meshgrid(y, x, indexing="ij")
    radius = torch.sqrt(grid_x.square() + grid_y.square()) / math.sqrt(2.0)

    low_mask = (radius <= 0.20).to(delta.dtype)
    mid_mask = ((radius > 0.20) & (radius <= 0.60)).to(delta.dtype)
    high_mask = (radius > 0.60).to(delta.dtype)

    energy = coefficients.square()
    total = energy.mean() + 1e-8
    mid_energy = (energy * mid_mask).mean() / total
    high_energy = (energy * high_mask).mean() / total
    low_energy = (energy * low_mask).mean() / total
    return mid_energy + 0.25 * low_energy - high_energy


def _soft_histogram_channel(
    channel: torch.Tensor,
    *,
    num_bins: int,
    bandwidth: float,
    circular: bool = False,
) -> torch.Tensor:
    bins = torch.linspace(
        0.0,
        1.0,
        num_bins,
        device=channel.device,
        dtype=channel.dtype,
    ).view(1, 1, num_bins)
    values = channel.reshape(channel.shape[0], -1, 1)
    delta = (values - bins).abs()
    if circular:
        delta = torch.minimum(delta, 1.0 - delta)
    weights = torch.exp(-(delta.square()) / (2.0 * bandwidth * bandwidth))
    hist = weights.mean(dim=1)
    return hist / (hist.sum(dim=1, keepdim=True) + 1e-8)


def _soft_histogram(
    batch: torch.Tensor,
    *,
    num_bins: int = 16,
    bandwidth: float = 0.05,
    circular_hue: bool = False,
) -> torch.Tensor:
    channels = []
    for channel_index in range(batch.shape[1]):
        channels.append(
            _soft_histogram_channel(
                batch[:, channel_index],
                num_bins=num_bins,
                bandwidth=bandwidth,
                circular=circular_hue and channel_index == 0,
            )
        )
    return torch.stack(channels, dim=1)


def histogram_divergence_attack_term(
    original_batch: torch.Tensor,
    adversarial_batch: torch.Tensor,
    *,
    num_bins: int = 16,
    bandwidth: float = 0.05,
) -> torch.Tensor:
    original_rgb = _soft_histogram(
        original_batch,
        num_bins=num_bins,
        bandwidth=bandwidth,
    )
    adversarial_rgb = _soft_histogram(
        adversarial_batch,
        num_bins=num_bins,
        bandwidth=bandwidth,
    )
    original_hsv = _soft_histogram(
        rgb_to_hsv(original_batch),
        num_bins=num_bins,
        bandwidth=bandwidth,
        circular_hue=True,
    )
    adversarial_hsv = _soft_histogram(
        rgb_to_hsv(adversarial_batch),
        num_bins=num_bins,
        bandwidth=bandwidth,
        circular_hue=True,
    )

    rgb_kl = (
        original_rgb * ((original_rgb + 1e-8).log() - (adversarial_rgb + 1e-8).log())
    ).sum(dim=2).mean()
    hsv_kl = (
        original_hsv * ((original_hsv + 1e-8).log() - (adversarial_hsv + 1e-8).log())
    ).sum(dim=2).mean()
    return 0.5 * (rgb_kl + hsv_kl)


@lru_cache(maxsize=32)
def _gaussian_kernel(kernel_size: int, sigma: float) -> torch.Tensor:
    coords = torch.arange(kernel_size, dtype=torch.float32) - kernel_size // 2
    kernel = torch.exp(-(coords.square()) / (2 * sigma * sigma))
    kernel = kernel / kernel.sum()
    window = torch.outer(kernel, kernel)
    return window / window.sum()


def _manual_ssim(original: torch.Tensor, adversarial: torch.Tensor) -> torch.Tensor:
    channel = original.shape[1]
    window = _gaussian_kernel(11, 1.5).to(
        device=original.device,
        dtype=original.dtype,
    )
    window = window.view(1, 1, 11, 11).expand(channel, 1, 11, 11)

    mu_x = F.conv2d(original, window, padding=5, groups=channel)
    mu_y = F.conv2d(adversarial, window, padding=5, groups=channel)

    mu_x_sq = mu_x.square()
    mu_y_sq = mu_y.square()
    mu_xy = mu_x * mu_y

    sigma_x_sq = F.conv2d(original * original, window, padding=5, groups=channel) - mu_x_sq
    sigma_y_sq = F.conv2d(adversarial * adversarial, window, padding=5, groups=channel) - mu_y_sq
    sigma_xy = F.conv2d(original * adversarial, window, padding=5, groups=channel) - mu_xy

    c1 = 0.01 ** 2
    c2 = 0.03 ** 2
    numerator = (2.0 * mu_xy + c1) * (2.0 * sigma_xy + c2)
    denominator = (mu_x_sq + mu_y_sq + c1) * (sigma_x_sq + sigma_y_sq + c2)
    return (numerator / (denominator + 1e-8)).mean()


def ssim_value(original: torch.Tensor, adversarial: torch.Tensor) -> torch.Tensor:
    try:
        from pytorch_msssim import ssim

        return ssim(original, adversarial, data_range=1.0, size_average=True)
    except Exception:
        return _manual_ssim(original, adversarial)


def ssim_penalty(
    original: torch.Tensor,
    adversarial: torch.Tensor,
    *,
    threshold: float,
) -> torch.Tensor:
    return torch.relu(torch.tensor(threshold, device=original.device, dtype=original.dtype) - ssim_value(original, adversarial))

