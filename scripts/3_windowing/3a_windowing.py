#!/usr/bin/env python3
"""Pure, native-cadence sliding-window helpers (no BigQuery dependencies).

Each window is 600 physical seconds, centered at a native measurement
timestamp. Windows slide by exactly one native time sample. Invalid,
incomplete, or severely gapped windows are skipped, never interpolated.

Inputs are time-sorted samples for ONE event neighborhood and instrument.
The returned arrays are temporary views into the prefetched RAM block.
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import Iterator

import numpy as np


@dataclass(frozen=True)
class Window:
    center_s: float
    data: np.ndarray  # (time_samples, feature_channels), native cadence
    relative_times_s: np.ndarray
    detection_score: float
    offset_from_nearest_mpb_s: float


@dataclass
class WindowStats:
    candidates: int = 0
    accepted: int = 0
    incomplete: int = 0
    large_gap: int = 0
    invalid_data: int = 0


def nearest_label_offset_s(center_s: float, sorted_labels_s: np.ndarray) -> float:
    """Signed center-to-closest-labeled-MPB offset, in seconds."""
    labels = np.asarray(sorted_labels_s, dtype=np.float64)
    if labels.ndim != 1 or labels.size == 0:
        raise ValueError("At least one MPB timestamp is required.")
    j = int(np.searchsorted(labels, center_s))
    candidates = []
    if j > 0:
        candidates.append(center_s - float(labels[j - 1]))
    if j < len(labels):
        candidates.append(center_s - float(labels[j]))
    return min(candidates, key=abs)


def iter_native_windows(
    timestamps_s: np.ndarray,
    values: np.ndarray,
    *,
    center_label_s: float,
    split_labels_s: np.ndarray,
    window_seconds: float = 600.0,
    center_range_seconds: float = 1800.0,
    sigma_seconds: float = 60.0,
    max_gap_cadences: float = 2.5,
    edge_tolerance_cadences: float = 1.75,
    stats: WindowStats | None = None,
) -> Iterator[Window]:
    """Yield one native-time-centered window per eligible sample.

    For each center in +/-center_range_seconds of center_label_s, take all
    available samples whose timestamps are in [center - window/2,
    center + window/2]. Values are not resampled or normalized.

    y = exp[-0.5 * (dt_nearest_MPB / sigma_seconds)^2] measures temporal
    proximity to a known annotation. It is NOT a calibrated probability,
    and far-from-label examples are not established physical negatives.

    The caller should feed one event block at a time, ensuring that all
    blocks/windows are assigned to the appropriate CV fold.
    """
    times = np.asarray(timestamps_s, dtype=np.float64)
    x = np.asarray(values, dtype=np.float64)
    labels = np.sort(np.asarray(split_labels_s, dtype=np.float64))
    if x.ndim != 2 or times.ndim != 1 or times.size != len(x):
        raise ValueError("Expected timestamps (N,) and measurements (N,C).")
    if labels.ndim != 1 or not labels.size or not np.isfinite(labels).all():
        raise ValueError("split_labels_s must contain finite MPB times.")
    if (
        window_seconds <= 0 or center_range_seconds < 0 or sigma_seconds <= 0
        or max_gap_cadences <= 1 or edge_tolerance_cadences <= 0
    ):
        raise ValueError("Invalid window, target sigma, or cadence tolerances.")
    if not np.isfinite(times).all():
        raise ValueError("Invalid instrument timestamps.")
    if len(times) < 3:
        return
    diffs = np.diff(times)
    if np.any(diffs <= 0):
        raise ValueError("Timestamps must be strictly increasing, without duplicates.")
    reference_cadence_s = float(np.median(diffs))
    if reference_cadence_s <= 0:
        raise ValueError("Invalid native cadence.")
    if stats is None:
        stats = WindowStats()

    half = window_seconds / 2
    left = int(np.searchsorted(times, center_label_s-center_range_seconds, side="left"))
    right = int(np.searchsorted(times, center_label_s+center_range_seconds, side="right"))
    for idx in range(left, right):
        stats.candidates += 1
        center = float(times[idx])
        start = center - half
        end = center + half
        lo = int(np.searchsorted(times, start, side="left"))
        hi = int(np.searchsorted(times, end, side="right"))
        if hi - lo < 3:
            stats.incomplete += 1
            continue
        block_time = times[lo:hi]
        if (
            block_time[0] - start > edge_tolerance_cadences * reference_cadence_s
            or end - block_time[-1] > edge_tolerance_cadences * reference_cadence_s
        ):
            stats.incomplete += 1
            continue
        if np.max(np.diff(block_time)) > max_gap_cadences * reference_cadence_s:
            stats.large_gap += 1
            continue
        data = x[lo:hi]  # a view; no huge window array is materialized
        if not np.isfinite(data).all():
            stats.invalid_data += 1
            continue
        offset = nearest_label_offset_s(center, labels)
        score = float(np.exp(-0.5 * (offset / sigma_seconds) ** 2))
        stats.accepted += 1
        yield Window(
            center_s=center,
            data=data,
            relative_times_s=block_time-center,
            detection_score=score,
            offset_from_nearest_mpb_s=offset,
        )
