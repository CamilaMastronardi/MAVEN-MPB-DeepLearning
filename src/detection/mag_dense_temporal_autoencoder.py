"""Sixth MAG experiment: residual encoder with dense temporal MPB scores.

Inputs are 600-s MAG windows at native timestamps; outputs are:
  z: (B,8,T), reconstruction: (B,4,T), temporal score: (B,T).

Unlike the fifth experiment's one scalar per window, each real epoch has a
supervised score. The detector compares masked latent averages BEFORE and
AFTER EACH epoch with the latent field at that epoch. All time intervals
use actual relative seconds, not fixed sample positions or orbital labels.

The score is proximity to catalog MPB annotations; NOT a calibrated
probability or proof that a physical discontinuity is a boundary.
"""
from __future__ import annotations

import torch
from torch import nn
from torch.nn import functional as F

from src.detection.mag_residual_transition_autoencoder import (
    MagResidualTransitionAutoencoder,
)


class MagDenseTemporalAutoencoder(MagResidualTransitionAutoencoder):
    """Keep v5 residual encoder/decoder; substitute a dense temporal head."""

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
        # A bounded temporal support centered +/-120 s, excluding the
        # +/-15-s region nearest the sample under classification.
        half_side_width_s = pooling_sigma_s
        self.before_start_s = -side_offset_s - half_side_width_s
        self.before_end_s = -center_exclusion_s
        self.after_start_s = center_exclusion_s
        self.after_end_s = side_offset_s + half_side_width_s
        if not (self.before_start_s < self.before_end_s < 0
                <= self.after_start_s < self.after_end_s):
            raise ValueError("Invalid asymmetric temporal pooling supports.")
        self.detector = nn.Sequential(
            nn.Conv1d(4 * latent_channels, hidden, kernel_size=1),
            nn.GELU(),
            nn.Conv1d(hidden, 1, kernel_size=1),
            nn.Sigmoid(),
        )

    def temporal_transition_features(
        self, z: torch.Tensor, time_mask: torch.Tensor,
        rel_time_s: torch.Tensor,
    ) -> torch.Tensor:
        """Compute [local, before, after, after-before] at every native epoch.

        Sorted masked relative seconds and prefix sums give O(B*C*T) masked
        pooling instead of quadratic temporal attention. Search positions
        are nondifferentiable time metadata; latent sums remain differentiable.
        """
        if z.ndim != 3:
            raise ValueError("Expected latent tensor (B,C,T).")
        b, c, t = z.shape
        if (time_mask.shape != (b, t)
                or rel_time_s.shape != (b, t)
                or time_mask.dtype != torch.bool
                or not bool(time_mask.any(dim=1).all())):
            raise ValueError("Invalid dense temporal MAG time/mask shape.")
        mask = time_mask[:, None, :].to(z.dtype)
        z = z * mask
        sorted_times = rel_time_s.masked_fill(
            ~time_mask, float("inf")
        ).contiguous()
        cumulative = F.pad(z.cumsum(dim=-1), (1, 0))
        cumulative_counts = F.pad(
            time_mask.long().cumsum(dim=-1), (1, 0)
        )

        def local_mean(low: float, high: float):
            low_index = torch.searchsorted(
                sorted_times, (rel_time_s + low).contiguous(), right=False
            )
            high_index = torch.searchsorted(
                sorted_times, (rel_time_s + high).contiguous(), right=True
            )
            start = low_index[:, None, :].expand(-1, c, -1)
            end = high_index[:, None, :].expand(-1, c, -1)
            count = (
                cumulative_counts.gather(1, high_index)
                - cumulative_counts.gather(1, low_index)
            )
            pooled = (
                cumulative.gather(2, end) - cumulative.gather(2, start)
            ) / count.clamp_min(1)[:, None, :]
            return torch.where(count[:, None, :] > 0, pooled, z)

        before = local_mean(self.before_start_s, self.before_end_s)
        after = local_mean(self.after_start_s, self.after_end_s)
        return torch.cat(
            (z, before, after, after - before), dim=1
        ) * mask

    def forward(
        self, x: torch.Tensor, time_mask: torch.Tensor,
        rel_time_s: torch.Tensor,
    ):
        if x.ndim != 3 or x.shape[1] != 4:
            raise ValueError("MAG input must have shape (B,4,T).")
        if (time_mask.shape != (x.shape[0], x.shape[-1])
                or rel_time_s.shape != time_mask.shape
                or time_mask.dtype != torch.bool
                or not bool(time_mask.any(dim=1).all())):
            raise ValueError("Invalid dense temporal MAG input mask/times.")
        mask = time_mask[:, None, :].to(x.dtype)
        z = self.encoder(x * mask) * mask
        reconstructed = self.decoder(z) * mask
        features = self.temporal_transition_features(
            z, time_mask, rel_time_s
        )
        score = self.detector(features).squeeze(1)
        return z, reconstructed, score * time_mask.to(score.dtype)
