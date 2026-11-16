"""MAG autoencoder with an explicitly asymmetric before/after MPB detector.

Only the detector changes relative to MagSupervisedAutoencoder v1:
the encoder, latent size, decoder, loss and one-score-per-600s-window
interface are preserved. No deeper encoder or per-time-step output yet.

Input: (batch,4,native_time), output: latent, reconstruction, score.
The score is NOT a calibrated physical MPB probability.
"""
from __future__ import annotations

import torch
from torch import nn

from src.detection.mag_autoencoder import MagSupervisedAutoencoder


class MagTransitionAutoencoder(MagSupervisedAutoencoder):
    """Ablation: temporal BEFORE / CENTER / AFTER / AFTER-minus-BEFORE features.

    The before/after summaries use masked Gaussian weights located at
    -side_offset_s and +side_offset_s in actual seconds from the center;
    samples close to center are excluded from the two side summaries.

    Missing side samples (unusual with QC-accepted complete windows) fall
    back to the center representation, avoiding NaNs and fabricated data.
    """

    def __init__(
        self,
        hidden: int = 16,
        latent_channels: int = 2,
        pooling_sigma_s: float = 90.0,
        side_offset_s: float = 120.0,
        center_exclusion_s: float = 15.0,
    ) -> None:
        if not (0 <= center_exclusion_s < side_offset_s):
            raise ValueError("Require 0 <= center exclusion < side offset.")
        super().__init__(
            hidden=hidden,
            latent_channels=latent_channels,
            pooling_sigma_s=pooling_sigma_s,
        )
        self.side_offset_s = float(side_offset_s)
        self.center_exclusion_s = float(center_exclusion_s)
        # Four latent summaries, compared to two in the baseline detector.
        self.detector = nn.Sequential(
            nn.Linear(4 * latent_channels, hidden),
            nn.GELU(),
            nn.Linear(hidden, 1),
            nn.Sigmoid(),
        )

    def transition_features(
        self,
        z: torch.Tensor,
        time_mask: torch.Tensor,
        rel_time_s: torch.Tensor,
    ) -> torch.Tensor:
        """Return [center, before, after, after-before], mask-aware.

        The temporal supports and Gaussian weights use real physical seconds,
        never array indices. This remains compatible with native MAG cadence.
        """
        if z.ndim != 3:
            raise ValueError("Latent sequence must have shape (B,C,T).")
        expected = (z.shape[0], z.shape[-1])
        if (time_mask.shape != expected or rel_time_s.shape != expected
                or time_mask.dtype != torch.bool
                or not bool(time_mask.any(dim=1).all())):
            raise ValueError("Expected valid (B,T) mask and relative times.")

        # Exact same nearest-valid-center definition as the baseline.
        distance = rel_time_s.abs().masked_fill(~time_mask, float("inf"))
        nearest = distance.argmin(dim=1)
        center = z.gather(
            2, nearest[:, None, None].expand(-1, z.shape[1], 1)
        ).squeeze(-1)

        before_mask = time_mask & (rel_time_s < -self.center_exclusion_s)
        after_mask = time_mask & (rel_time_s > self.center_exclusion_s)
        before_weights = torch.exp(
            -0.5 * ((rel_time_s + self.side_offset_s) / self.pooling_sigma_s).square()
        ) * before_mask.to(z.dtype)
        after_weights = torch.exp(
            -0.5 * ((rel_time_s - self.side_offset_s) / self.pooling_sigma_s).square()
        ) * after_mask.to(z.dtype)

        def weighted_side(weights: torch.Tensor) -> torch.Tensor:
            weight_sum = weights.sum(dim=-1, keepdim=True)
            pooled = (z * weights[:, None, :]).sum(dim=-1) / (
                weight_sum.clamp_min(1e-9)
            )
            return torch.where(weight_sum > 0, pooled, center)

        before = weighted_side(before_weights)
        after = weighted_side(after_weights)
        delta = after - before
        return torch.cat((center, before, after, delta), dim=1)

    def forward(
        self,
        x: torch.Tensor,
        time_mask: torch.Tensor,
        rel_time_s: torch.Tensor,
    ):
        if x.ndim != 3 or x.shape[1] != 4:
            raise ValueError("MAG input must have shape (B,4,T).")
        expected = (x.shape[0], x.shape[-1])
        if (time_mask.shape != expected or rel_time_s.shape != expected
                or time_mask.dtype != torch.bool
                or not bool(time_mask.any(dim=1).all())):
            raise ValueError("Expected valid (B,T) mask and relative times.")

        mask = time_mask[:, None, :].to(x.dtype)
        z = self.encoder(x * mask) * mask
        reconstructed = self.decoder(z) * mask
        features = self.transition_features(z, time_mask, rel_time_s)
        score = self.detector(features)
        return z, reconstructed, score
