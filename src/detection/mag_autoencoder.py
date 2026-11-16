"""First mask-aware convolutional MAG autoencoder with supervised MPB head.

Input: MAG (batch, 4, native_time); output: latent, reconstructed MAG, score.
The score is temporal proximity to a catalog annotation, not a calibrated MPB
probability. No reconstruction-error anomaly detection is performed.
"""
from __future__ import annotations

import torch
from torch import nn


class MagSupervisedAutoencoder(nn.Module):
    """A deliberately compact, temporal-resolution-preserving baseline."""

    def __init__(self, hidden: int = 16, latent_channels: int = 2,
                 pooling_sigma_s: float = 90.0):
        super().__init__()
        if min(hidden, latent_channels) < 1 or pooling_sigma_s <= 0:
            raise ValueError("Channels and pooling_sigma_s must be positive.")
        self.pooling_sigma_s = float(pooling_sigma_s)

        encoder = [nn.Conv1d(4, hidden, kernel_size=7, padding=3), nn.GELU()]
        for dilation in (2, 4, 8, 16, 32):
            encoder.extend([
                nn.Conv1d(hidden, hidden, kernel_size=5, dilation=dilation,
                          padding=2 * dilation),
                nn.GELU(),
            ])
        encoder.append(nn.Conv1d(hidden, latent_channels, kernel_size=1))
        self.encoder = nn.Sequential(*encoder)

        self.decoder = nn.Sequential(
            nn.Conv1d(latent_channels, hidden, kernel_size=5, padding=2),
            nn.GELU(),
            nn.Conv1d(hidden, 4, kernel_size=7, padding=3),
        )
        self.detector = nn.Sequential(
            nn.Linear(2 * latent_channels, hidden),
            nn.GELU(),
            nn.Linear(hidden, 1),
            nn.Sigmoid(),
        )

    def forward(self, x: torch.Tensor, time_mask: torch.Tensor,
                rel_time_s: torch.Tensor):
        if x.ndim != 3 or x.shape[1] != 4:
            raise ValueError("MAG input must have shape (B,4,T).")
        expected = (x.shape[0], x.shape[-1])
        if time_mask.shape != expected or rel_time_s.shape != expected:
            raise ValueError("time_mask and rel_time_s must have shape (B,T).")
        if time_mask.dtype != torch.bool:
            raise ValueError("time_mask must be boolean.")
        if not bool(time_mask.any(dim=1).all()):
            raise ValueError("Each window needs at least one valid time sample.")

        mask = time_mask[:, None, :].to(x.dtype)
        z = self.encoder(x * mask) * mask
        reconstructed = self.decoder(z) * mask

        # Nearest native measurement to the physical window center t_c=0.
        distance = rel_time_s.abs().masked_fill(~time_mask, float("inf"))
        indices = distance.argmin(dim=1)
        center = z.gather(
            2, indices[:, None, None].expand(-1, z.shape[1], 1)
        ).squeeze(-1)

        # Gaussian local pooling in REAL seconds, excluding padded samples.
        weights = torch.exp(-0.5 * (rel_time_s / self.pooling_sigma_s).square())
        weights = weights * time_mask.to(weights.dtype)
        pooled = (z * weights[:, None, :]).sum(dim=-1) / (
            weights.sum(dim=-1).clamp_min(1e-9)[:, None]
        )
        score = self.detector(torch.cat([center, pooled], dim=1))
        return z, reconstructed, score
