"""Native-cadence MAG/SWEA candidate clips and fold-isolated normalization.

No temporal interpolation: MAG and SWEA each use their own physical sample
times, padded only inside a minibatch with an explicit boolean mask. SWEA
spectral log-energy interpolation happens upstream and is source-bound.
"""
from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import torch


@dataclass
class DualClip:
    mag: np.ndarray
    swea: np.ndarray
    mag_time_s: np.ndarray
    swea_time_s: np.ndarray


def extract_clip(
    mag_block, swea_block, center_s, *, half_seconds=300.
):
    """Return None for missing/invalid/incomplete/gapped native windows."""
    series = []
    for block, kind in ((mag_block, "MAG"), (swea_block, "SWEA")):
        times, values = block
        if len(times) < 4:
            return None
        left = int(np.searchsorted(times, center_s-half_seconds, "left"))
        right = int(np.searchsorted(times, center_s+half_seconds, "right"))
        t = times[left:right]
        x = values[left:right]
        if len(t) < (540 if kind == "MAG" else 12):
            return None
        dt = np.diff(t)
        if (np.any(dt <= 0) or
                not np.isfinite(x).all() or
                not np.isfinite(t).all()):
            return None
        cadence = float(np.median(dt))
        if kind == "MAG":
            edge, gap = 2.0, 2.5
        else:
            edge, gap = 50., 3.
        if (t[0] - (center_s-half_seconds) > edge or
                (center_s+half_seconds) - t[-1] > edge or
                dt.max() > gap * cadence):
            return None
        if (not np.any(t < center_s - 60.) or
                not np.any(t > center_s + 60.)):
            return None
        series.append((np.ascontiguousarray(x, dtype=np.float32),
                       np.ascontiguousarray(t-center_s, dtype=np.float32)))
    return DualClip(
        mag=series[0][0], swea=series[1][0],
        mag_time_s=series[0][1], swea_time_s=series[1][1],
    )


def fit_unique_epochs(blocks, times, *, instrument):
    """Fit only on fit-TRAIN events, avoiding duplicate overlapping epochs."""
    chunks = []
    for event in times:
        t, x = blocks[event][instrument]
        if len(t):
            chunks.append((t, x))
    if not chunks:
        raise ValueError(f"No valid TRAIN {instrument} data.")
    # Vectorized de-duplication avoids millions of Python objects for MAG.
    timestamps = np.concatenate([t for t, _ in chunks]).astype(np.float64)
    values = np.concatenate([x for _, x in chunks]).astype(np.float64)
    order = np.argsort(timestamps, kind="stable")
    unique = np.ones(len(order), dtype=bool)
    unique[1:] = np.diff(timestamps[order]) >= .001
    x = values[order[unique]]
    if len(x) < 2 or not np.isfinite(x).all():
        raise ValueError(f"Insufficient finite {instrument} TRAIN epochs.")
    mean = x.mean(axis=0)
    variance = np.maximum(np.mean((x - mean).square(), axis=0), 0.)
    scale = np.where(variance > 1e-12, np.sqrt(variance), 1.)
    return {"mean": mean, "scale": scale, "n_epochs": len(x)}


def collate_dual(batch_items, normalizers, device="cpu"):
    """batch_items: list[(DualClip, transition_label, side_labels)]"""
    if not batch_items:
        raise ValueError("Cannot collate an empty dual-instrument batch.")
    size = len(batch_items)
    result = {}
    for instrument, field, timefield in (
        ("MAG", "mag", "mag_time_s"),
        ("SWEA", "swea", "swea_time_s")
    ):
        maxlen = max(len(getattr(item[0], field)) for item in batch_items)
        features = 4 if instrument == "MAG" else 12
        x = np.zeros((size, features, maxlen), dtype=np.float32)
        mask = np.zeros((size, maxlen), dtype=bool)
        rel = np.zeros((size, maxlen), dtype=np.float32)
        mu = normalizers[instrument]["mean"]
        sigma = normalizers[instrument]["scale"]
        for i, (clip, _, _) in enumerate(batch_items):
            values = getattr(clip, field)
            t = len(values)
            if values.shape != (t, features):
                raise ValueError(f"Unexpected {instrument} clip dimensions.")
            x[i, :, :t] = ((values-mu)/sigma).T.astype(np.float32)
            mask[i, :t] = True
            rel[i, :t] = getattr(clip, timefield)
        key = instrument.lower()
        result[key] = torch.tensor(x, device=device)
        result[key+"_mask"] = torch.tensor(mask, device=device)
        result[key+"_time"] = torch.tensor(rel, device=device)
    result["transition"] = torch.tensor(
        [float(item[1]) for item in batch_items],
        device=device, dtype=torch.float32
    )
    result["regions"] = torch.tensor(
        [item[2] for item in batch_items],
        device=device, dtype=torch.float32
    )
    return result
