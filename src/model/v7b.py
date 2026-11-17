"""Seventh model: MAG+SWEA dual-encoder autoencoders with physical region heads.

The MPB is modeled as a change of plasma regime, not a Gaussian regression
target. Each 600-s candidate window is independently encoded at native MAG
and SWEA cadences. Outputs:
 - transition_logit (B,) = whether this candidate is a catalog MPB;
 - region_logits (B,2) = upstream/downstream plasma region:
   0 magnetosheath, 1 induced magnetosphere;
 - two instrument-specific reconstructions for auxiliary representation
   learning. Neither absolute timestamps nor source catalog metadata enter.
"""
from __future__ import annotations

import torch
from torch import nn


class ResidualTimeBlock(nn.Module):
    def __init__(self, channels, dilation):
        super().__init__()
        self.f = nn.Sequential(
            nn.Conv1d(
                channels, channels, kernel_size=5,
                padding=2 * dilation, dilation=dilation
            ),
            nn.GELU(),
            nn.Conv1d(channels, channels, kernel_size=1),
        )

    def forward(self, x):
        return x + self.f(x)


class SensorEncoderDecoder(nn.Module):
    def __init__(self, features, hidden, latent, dilations):
        super().__init__()
        self.encoder = nn.Sequential(
            nn.Conv1d(features, hidden, 5, padding=2),
            nn.GELU(),
            *(ResidualTimeBlock(hidden, d) for d in dilations),
            nn.Conv1d(hidden, latent, 1),
        )
        self.decoder = nn.Sequential(
            nn.Conv1d(latent, hidden, 5, padding=2),
            nn.GELU(),
            nn.Conv1d(hidden, features, 5, padding=2),
        )

    def forward(self, x, time_mask):
        if (x.ndim != 3 or time_mask.shape !=
                (x.shape[0], x.shape[-1])):
            raise ValueError("Bad sensor tensor or native time mask.")
        m = time_mask[:, None, :].to(x.dtype)
        z = self.encoder(x * m) * m
        return z, self.decoder(z) * m


def masked_region_pool(z, time_mask, relative_time_s, *, side, gap_s):
    if (relative_time_s.shape != time_mask.shape or
            time_mask.dtype != torch.bool):
        raise ValueError("Sensor timestamps and masks disagree.")
    if side == "left":
        allowed = time_mask & (relative_time_s < -gap_s)
    elif side == "right":
        allowed = time_mask & (relative_time_s > gap_s)
    else:
        raise ValueError("Unknown candidate side.")
    if not bool(allowed.any(dim=1).all()):
        raise ValueError("Both physical sides need measured samples.")
    w = allowed[:, None, :].to(z.dtype)
    return (z * w).sum(-1) / w.sum(-1).clamp_min(1.0)


class MagSweaRegionAutoencoder(nn.Module):
    """Classification of region transition; reconstruction is auxiliary only."""

    def __init__(
        self, n_swea_energies=12, mag_hidden=24, swea_hidden=24,
        latent=16, region_gap_s=60.0, dropout=0.15
    ):
        super().__init__()
        if n_swea_energies < 2 or latent < 2 or region_gap_s <= 0:
            raise ValueError("Invalid model input dimensions or side gap.")
        self.region_gap_s = float(region_gap_s)
        self.mag = SensorEncoderDecoder(
            4, mag_hidden, latent, (1, 2, 4, 8)
        )
        self.swea = SensorEncoderDecoder(
            n_swea_energies, swea_hidden, latent, (1, 2)
        )
        dim = 2 * latent
        self.region_head = nn.Sequential(
            nn.Linear(dim, 32), nn.GELU(), nn.Dropout(dropout),
            nn.Linear(32, 1)
        )
        self.transition_head = nn.Sequential(
            nn.Linear(4 * dim, 64),
            nn.GELU(), nn.Dropout(dropout),
            nn.Linear(64, 1)
        )

    def forward(
        self, mag, mag_mask, mag_relative_s,
        swea, swea_mask, swea_relative_s
    ):
        zm, rm = self.mag(mag, mag_mask)
        zs, rs = self.swea(swea, swea_mask)
        left_mag = masked_region_pool(
            zm, mag_mask, mag_relative_s,
            side="left", gap_s=self.region_gap_s
        )
        right_mag = masked_region_pool(
            zm, mag_mask, mag_relative_s,
            side="right", gap_s=self.region_gap_s
        )
        left_swea = masked_region_pool(
            zs, swea_mask, swea_relative_s,
            side="left", gap_s=self.region_gap_s
        )
        right_swea = masked_region_pool(
            zs, swea_mask, swea_relative_s,
            side="right", gap_s=self.region_gap_s
        )
        left = torch.cat((left_mag, left_swea), dim=1)
        right = torch.cat((right_mag, right_swea), dim=1)
        region_logits = torch.stack(
            (self.region_head(left).squeeze(-1),
             self.region_head(right).squeeze(-1)),
            dim=1
        )
        transition_features = torch.cat(
            (left, right, right - left, (right - left).abs()), dim=1
        )
        transition_logit = self.transition_head(
            transition_features
        ).squeeze(-1)
        return {
            "transition_logit": transition_logit,
            "region_logits": region_logits,
            "mag_reconstruction": rm,
            "swea_reconstruction": rs,
        }


def masked_reconstruction_loss(reconstructed, original, mask):
    if reconstructed.shape != original.shape or mask.shape != (
        original.shape[0], original.shape[-1]
    ):
        raise ValueError("Reconstruction and temporal mask shapes disagree.")
    w = mask[:, None, :].to(original.dtype)
    return ((reconstructed - original).square() * w).sum() / (
        mask.sum() * original.shape[1]
    ).clamp_min(1)
