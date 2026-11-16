"""Dense MAG annotation targets, masked loss and physical-time score fusion.

A dense model emits one score per actual native MAG sample, not one scalar
per 600-s sliding window. Targets are Gaussian proximity to the nearest
catalog label represented by each window's saved center-to-label offset.
This is a supervised weak label, NOT a calibrated physical probability.

At evaluation, overlapping 600-s windows are averaged at their ORIGINAL
UTC sample timestamps, then the maximum is searched only within +/-30 min
of the catalog crossing (same conditioned search region as v1-v5).
"""
from __future__ import annotations

import numpy as np
import torch


def dense_gaussian_targets(batch, sigma_seconds: float) -> torch.Tensor:
    """Target (B,T) on actual epochs, not on artificial 1-s grid."""
    if sigma_seconds <= 0:
        raise ValueError("sigma_seconds must be positive.")
    relative = batch.rel_time_s
    if batch.time_mask.shape != relative.shape:
        raise ValueError("Invalid batch masks for dense target.")
    offsets = batch.label_offset_s.to(
        device=relative.device, dtype=relative.dtype
    )[:, None]
    physical_time_from_label = relative + offsets
    y = torch.exp(-0.5 * (physical_time_from_label / sigma_seconds).square())
    return y * batch.time_mask.to(y.dtype)


def dense_weighted_mse(
    scores: torch.Tensor,
    batch,
    *,
    sigma_seconds: float,
    near_label_weight: float,
) -> torch.Tensor:
    """Mask native padding and normalize weighted Gaussian target MSE."""
    if near_label_weight < 1:
        raise ValueError("near_label_weight must be >=1.")
    if scores.ndim != 2 or scores.shape != batch.time_mask.shape:
        raise ValueError("Dense MPB scores must be shaped (B,T).")
    if not bool(batch.time_mask.any()):
        raise ValueError("No valid dense target samples.")
    target = dense_gaussian_targets(batch, sigma_seconds)
    valid = batch.time_mask.to(scores.dtype)
    weights = (1 + (near_label_weight - 1) * target) * valid
    return ((scores - target).square() * weights).sum() / weights.sum()


def dense_loss_sums(scores, batch, *, sigma_seconds, near_label_weight):
    """Independent numerator/denominator for exact monitor/outer VAL losses."""
    target = dense_gaussian_targets(batch, sigma_seconds)
    valid = batch.time_mask.to(scores.dtype)
    weights = (1 + (near_label_weight - 1) * target) * valid
    return (
        float(((scores - target).square() * weights).sum().item()),
        float(weights.sum().item()),
    )


class DenseNativeScoreFusion:
    """Average overlapping scores on real epoch timestamps, without resampling."""

    def __init__(self, windows, event_time_s: float, *, radius_s=1800.):
        if not windows or radius_s <= 0:
            raise ValueError("Dense fusion requires windows and a time radius.")
        self.event_time_s = float(event_time_s)
        self.radius_s = float(radius_s)
        full = np.concatenate([
            w.center_s + w.relative_times_s for w in windows
        ])
        if not np.isfinite(full).all():
            raise ValueError("Nonfinite time coordinate in dense windows.")
        grid = np.unique(full)
        self.times_s = grid[
            (grid >= event_time_s - radius_s)
            & (grid <= event_time_s + radius_s)
        ]
        if not len(self.times_s):
            raise ValueError("No valid MAG epochs inside dense evaluation range.")
        self.score_sum = np.zeros(len(self.times_s), dtype=np.float64)
        self.coverage = np.zeros(len(self.times_s), dtype=np.int32)

    def add_window(self, window, dense_scores):
        times = np.asarray(
            window.center_s + window.relative_times_s, dtype=np.float64
        )
        values = np.asarray(dense_scores, dtype=np.float64)
        if values.shape != times.shape or not np.isfinite(values).all():
            raise ValueError("Invalid native-time density scores.")
        chosen = (
            (times >= self.event_time_s - self.radius_s)
            & (times <= self.event_time_s + self.radius_s)
        )
        if not chosen.any():
            return
        timestamps = times[chosen]
        indexes = np.searchsorted(self.times_s, timestamps)
        valid = indexes < len(self.times_s)
        if not np.all(valid):
            raise ValueError("Dense fusion timestamp is missing from grid.")
        if not np.allclose(self.times_s[indexes], timestamps, atol=1e-4, rtol=0.):
            raise ValueError("Misaligned native MAG timestamp in density fusion.")
        np.add.at(self.score_sum, indexes, values[chosen])
        np.add.at(self.coverage, indexes, 1)

    def finish(self):
        """Physical epoch grid, averaged scores, peak offset and peak score."""
        valid = self.coverage > 0
        if not valid.any():
            raise ValueError("No native MAG samples received scores.")
        grid = self.times_s[valid]
        score = self.score_sum[valid] / self.coverage[valid]
        peak = int(np.argmax(score))
        return {
            "times_s": grid,
            "scores": score,
            "coverage": self.coverage[valid],
            "peak_offset_s": float(grid[peak] - self.event_time_s),
            "peak_score": float(score[peak]),
        }
