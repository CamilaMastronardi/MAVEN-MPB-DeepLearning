#!/usr/bin/env python3
"""Per-fold train-only preprocessing and masked native-time minibatches.

Offline utilities: NO BigQuery queries, file I/O or science-data persistence.
Raw event blocks have shape (native_time, channels). MAG uses 4 channels.
SWEA uses the 30–100 eV channels only, after angular reduction.

Important:
  - Accumulate normalization on UNIQUE TRAINING measurement epochs, not
    duplicated overlapping sliding windows; repeated blocks across nearby
    labels must be merged/deduplicated by the eventual data loader.
  - MAG: identity feature transform.
  - SWEA: log10(1 + max(flux,0)), preserving the chosen energy axis.
  - Standardize each channel with training-only mean and population std.
  - No normalization parameter is fit on held-out validation/test data.
  - Pad native variable lengths within a batch and carry the time mask.
"""
from __future__ import annotations

from dataclasses import dataclass
from importlib import import_module
from pathlib import Path
from typing import Literal, Sequence
import sys

import numpy as np
import torch

ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

Window = import_module("scripts.3_windowing.3a_windowing").Window

Instrument = Literal["MAG", "SWEA"]


def feature_transform(x: np.ndarray, instrument: Instrument) -> np.ndarray:
    """Transform 2D time×channel input WITHOUT standardizing it."""
    data = np.asarray(x, dtype=np.float64)
    if data.ndim != 2 or data.shape[1] < 1:
        raise ValueError("Expected a time-by-channel matrix with at least one feature.")
    if not np.isfinite(data).all():
        raise ValueError("Measurements contain nonfinite values.")
    if instrument == "MAG":
        if data.shape[1] != 4:
            raise ValueError("MAG must contain exactly Bx, By, Bz, |B| (4 channels).")
        return data
    if instrument == "SWEA":
        if np.any(data < 0):
            raise ValueError("Selected SWEA fluxes must be nonnegative.")
        return np.log10(1.0 + data)
    raise ValueError(f"Unknown instrument: {instrument}")


class FoldChannelNormalizer:
    """Streaming Welford standardizer fitted on training blocks ONLY.

    A separate instance is required for each held-out fold and instrument.
    Only unique raw epochs from eligible training months should be passed.
    """

    def __init__(self, instrument: Instrument, held_out_fold: int) -> None:
        if instrument not in ("MAG", "SWEA"):
            raise ValueError("Instrument must be MAG or SWEA.")
        if held_out_fold not in (1, 2, 3, 4):
            raise ValueError("held_out_fold must be 1,2,3,4.")
        self.instrument = instrument
        self.held_out_fold = held_out_fold
        self.count = 0
        self.n_features: int | None = None
        self.mean: np.ndarray | None = None
        self.m2: np.ndarray | None = None
        self.scale: np.ndarray | None = None
        self.is_frozen = False

    def update(self, raw_block: np.ndarray, *, source_fold: int) -> None:
        """Accumulate a contiguous training block, before generating windows."""
        if self.is_frozen:
            raise RuntimeError("Normalizer is frozen; cannot update statistics.")
        if source_fold == self.held_out_fold or source_fold not in (1, 2, 3, 4):
            raise ValueError("Refusing to fit on held-out/invalid CV fold.")
        x = feature_transform(raw_block, self.instrument)
        if x.shape[0] == 0:
            return
        if self.n_features is None:
            self.n_features = x.shape[1]
            self.mean = np.zeros(self.n_features, dtype=np.float64)
            self.m2 = np.zeros(self.n_features, dtype=np.float64)
        if x.shape[1] != self.n_features:
            raise ValueError("Channel mismatch between training blocks.")
        block_count = x.shape[0]
        block_mean = np.mean(x, axis=0)
        block_m2 = np.square(x - block_mean).sum(axis=0)
        delta = block_mean - self.mean
        total = self.count + block_count
        self.mean += delta * (block_count / total)
        self.m2 += block_m2 + delta**2 * self.count * block_count / total
        self.count = total

    def freeze(self) -> None:
        if self.count < 2 or self.m2 is None:
            raise ValueError("Fit at least two training epochs before freezing.")
        variance = self.m2 / self.count
        # Constant training channels are centered but not divided by zero.
        self.scale = np.where(variance > 1e-12, np.sqrt(variance), 1.0)
        self.is_frozen = True

    def transform(self, block: np.ndarray) -> np.ndarray:
        if not self.is_frozen or self.mean is None or self.scale is None:
            raise RuntimeError("Freeze TRAIN-only statistics before transforming.")
        data = feature_transform(block, self.instrument)
        if data.shape[1] != self.n_features:
            raise ValueError("Unexpected number of channels.")
        return ((data - self.mean) / self.scale).astype(np.float32)


@dataclass(frozen=True)
class NativeBatch:
    x: torch.Tensor  # MAG: (B, C, T); SWEA: (B, 1, E, T)
    time_mask: torch.Tensor  # (B,T) boolean; False for padded time indices
    rel_time_s: torch.Tensor  # (B,T) actual center-relative seconds, padded 0
    y: torch.Tensor  # (B,1) Gaussian temporal-proximity targets
    center_s: torch.Tensor  # (B,)
    label_offset_s: torch.Tensor  # (B,)


def collate_native_windows(
    windows: Sequence[Window],
    *,
    normalizer: FoldChannelNormalizer,
) -> NativeBatch:
    """Pad variable-length native windows AFTER train-fitted normalization.

    Padded normalized values are zero. A reconstruction loss MUST mask
    these values; no model should treat them as genuine measurements.
    """
    if not windows:
        raise ValueError("Cannot collate an empty window batch.")
    if not normalizer.is_frozen:
        raise ValueError("Train-only normalizer must be frozen before collating.")

    n_features = normalizer.n_features
    max_t = max(len(w.data) for w in windows)
    batch_size = len(windows)
    x = np.zeros((batch_size, n_features, max_t), dtype=np.float32)
    time_mask = np.zeros((batch_size, max_t), dtype=np.bool_)
    relative_time = np.zeros((batch_size, max_t), dtype=np.float32)
    labels = np.empty((batch_size, 1), dtype=np.float32)
    centers = np.empty(batch_size, dtype=np.float64)
    offsets = np.empty(batch_size, dtype=np.float32)

    for i, w in enumerate(windows):
        transformed = normalizer.transform(w.data)
        t = len(transformed)
        if w.relative_times_s.shape != (t,):
            raise ValueError("Window has inconsistent timestamps and measurements.")
        x[i, :, :t] = transformed.T
        time_mask[i, :t] = True
        relative_time[i, :t] = np.asarray(w.relative_times_s, dtype=np.float32)
        labels[i, 0] = w.detection_score
        centers[i] = w.center_s
        offsets[i] = w.offset_from_nearest_mpb_s

    features = torch.from_numpy(x)
    if normalizer.instrument == "SWEA":
        features = features.unsqueeze(1)
    return NativeBatch(
        x=features,
        time_mask=torch.from_numpy(time_mask),
        rel_time_s=torch.from_numpy(relative_time),
        y=torch.from_numpy(labels),
        center_s=torch.from_numpy(centers),
        label_offset_s=torch.from_numpy(offsets),
    )


def masked_reconstruction_mse(
    reconstructed: torch.Tensor,
    batch: NativeBatch,
) -> torch.Tensor:
    """Mean squared error only on actual (unpadded) samples, not padding."""
    if reconstructed.shape != batch.x.shape:
        raise ValueError("Reconstruction has wrong shape.")
    if batch.x.ndim == 3:  # MAG (B,C,T)
        mask = batch.time_mask[:, None, :]
    elif batch.x.ndim == 4:  # SWEA (B,1,E,T)
        mask = batch.time_mask[:, None, None, :]
    else:
        raise ValueError("Unsupported input tensor rank.")
    expanded = mask.expand_as(batch.x)
    count = expanded.sum()
    if count.item() == 0:
        raise ValueError("No valid time samples in reconstruction batch.")
    return (torch.square(reconstructed - batch.x) * expanded).sum() / count



def masked_mag_modulus_mse(
    reconstructed: torch.Tensor,
    batch: NativeBatch,
    normalizer: FoldChannelNormalizer,
) -> torch.Tensor:
    """Penalize both MAG magnitude reconstruction and physical consistency.

    MAG has independently standardized [Bx, By, Bz, |B|] input/output channels.
    Recover physical nT using the FROZEN, TRAIN-only per-fold normalizer before
    computing the norm of reconstructed vector components.

    The two equally weighted terms compare:
      1. decoded |B| channel vs observed |B| (nT);
      2. norm of decoded Bx, By, Bz vs observed |B| (nT).
    The mean is over observed samples ONLY (masked padding excluded).
    Divide by training-fold magnitude variance to make it dimensionless and
    comparable in scale to normalized reconstruction MSE.

    This adds a differentiable physical constraint without changing the
    architecture, and does not assume that a decoder channel itself equals
    the Euclidean norm unless learned. No calibration/MPB probability implied.
    """
    if (normalizer.instrument != "MAG" or not normalizer.is_frozen
            or normalizer.mean is None or normalizer.scale is None):
        raise ValueError("Require a frozen MAG normalizer.")
    if (reconstructed.shape != batch.x.shape
            or reconstructed.ndim != 3 or reconstructed.shape[1] != 4
            or batch.time_mask.shape != (batch.x.shape[0], batch.x.shape[2])):
        raise ValueError("Expected MAG (batch,4,time) and matching time mask.")
    if not bool(batch.time_mask.any()):
        raise ValueError("Cannot calculate magnitude loss without valid epochs.")

    mu = torch.as_tensor(
        normalizer.mean, dtype=reconstructed.dtype, device=reconstructed.device
    ).reshape(1, 4, 1)
    scale = torch.as_tensor(
        normalizer.scale, dtype=reconstructed.dtype, device=reconstructed.device
    ).reshape(1, 4, 1)
    if not bool(torch.isfinite(scale).all()) or not bool((scale > 0).all()):
        raise ValueError("Invalid MAG normalization scale.")

    reconstructed_nt = reconstructed * scale + mu
    observed_nt = batch.x * scale + mu
    observed_magnitude = observed_nt[:, 3, :]
    reconstructed_magnitude_channel = reconstructed_nt[:, 3, :]
    reconstructed_magnitude_vector = torch.linalg.vector_norm(
        reconstructed_nt[:, :3, :], dim=1
    )
    valid = batch.time_mask.to(reconstructed.dtype)
    magnitude_variance = scale[0, 3, 0].square()
    # Both differences are in nT², divided by (fold TRAIN std |B|)².
    channel_error = (reconstructed_magnitude_channel - observed_magnitude).square()
    vector_error = (reconstructed_magnitude_vector - observed_magnitude).square()
    return ((channel_error + vector_error) * valid).sum() / (
        2.0 * valid.sum() * magnitude_variance
    )



def weighted_proximity_mse(
    predicted: torch.Tensor,
    batch: NativeBatch,
    *,
    near_label_weight: float = 10.0,
) -> torch.Tensor:
    """Optional loss emphasizing crossing-near targets, NOT physical negatives.

    Multipliers range from 1 on far-from-known-label scores to
    near_label_weight near the annotation. Tune using training months only.
    """
    if predicted.shape != batch.y.shape or near_label_weight < 1:
        raise ValueError("Expected predictions (B,1) and weight >= 1.")
    w = 1.0 + (near_label_weight - 1.0) * batch.y
    return (w * torch.square(predicted - batch.y)).sum() / w.sum()
