"""Fifth MAG experiment: residual multiscale encoder + v4 transition head.

Only encoder depth/width and bottleneck capacity change compared to v4.
The detector still compares [center, before, after, after-before] and emits
one scalar per 600-second native MAG window. No test-event information enters
the network training, and no physical measurements are generated here.

For one temporal convolution per residual block (kernel size 5) and
dilations 1,2,4,8,16,32,64, plus the initial kernel-7 projection:
theoretical receptive field = 7 + 4 * (1+2+4+8+16+32+64) = 515 samples.
In actual seconds this depends on native MAG cadence.
"""
from __future__ import annotations

import torch
from torch import nn

from src.detection.mag_transition_autoencoder import MagTransitionAutoencoder


class ResidualDilatedBlock(nn.Module):
    """One dilated temporal convolution with a length-preserving skip."""

    def __init__(self, channels: int, dilation: int):
        super().__init__()
        if channels <= 0 or dilation <= 0:
            raise ValueError("Channels and dilation must be positive.")
        self.temporal = nn.Conv1d(
            channels, channels, kernel_size=5,
            dilation=dilation, padding=2 * dilation
        )
        self.activation = nn.GELU()
        self.channel_mix = nn.Conv1d(channels, channels, kernel_size=1)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return x + self.channel_mix(self.activation(self.temporal(x)))


class MagResidualTransitionAutoencoder(MagTransitionAutoencoder):
    """Residual encoder (hidden=32, latent=8) and the original transition head.

    The decoder, temporal pooling method, target labels and loss formulas
    remain unchanged in design compared to the fourth MAG experiment.
    """

    DILATIONS = (1, 2, 4, 8, 16, 32, 64)

    def __init__(
        self,
        hidden: int = 32,
        latent_channels: int = 8,
        pooling_sigma_s: float = 90.0,
        side_offset_s: float = 120.0,
        center_exclusion_s: float = 15.0,
    ) -> None:
        super().__init__(
            hidden=hidden,
            latent_channels=latent_channels,
            pooling_sigma_s=pooling_sigma_s,
            side_offset_s=side_offset_s,
            center_exclusion_s=center_exclusion_s,
        )
        layers: list[nn.Module] = [
            nn.Conv1d(4, hidden, kernel_size=7, padding=3),
            nn.GELU(),
        ]
        layers.extend(
            ResidualDilatedBlock(hidden, dilation=d)
            for d in self.DILATIONS
        )
        layers.append(nn.Conv1d(hidden, latent_channels, kernel_size=1))
        self.encoder = nn.Sequential(*layers)

    @property
    def receptive_field_samples(self) -> int:
        """Theoretical receptive field when all native samples are contiguous."""
        return 7 + 4 * sum(self.DILATIONS)
