"""Post-processing for a tentative MAG+SWEA MPB scan detector.

A scan is evaluated without selecting its maximum by reference to the target
MPB. Detection threshold and temporal non-maximum suppression are fixed on
inner-month monitor scans before any outer-fold evaluation.

IMPORTANT: event-centered 70-min blocks contain a known MPB. Their negative
background is only weakly verified, not an independent sample of full orbits.
"""
from __future__ import annotations

import numpy as np


def independent_peaks(times_s, scores, *, threshold, min_separation_s=300.):
    """Threshold + greedy temporal non-maximum suppression (no target times).

    Return chronological (center_s, score) tuples. Higher-score candidates
    win suppression even when scores form a broad plateau.
    """
    times = np.asarray(times_s, dtype=float)
    values = np.asarray(scores, dtype=float)
    if (times.ndim != 1 or values.shape != times.shape or
            not np.isfinite(times).all() or
            not np.isfinite(values).all() or
            not 0 <= threshold <= 1 or min_separation_s <= 0):
        raise ValueError("Invalid scan timestamps, scores or peak parameters.")
    if len(times) > 1 and np.any(np.diff(times) <= 0):
        raise ValueError("Scan candidate timestamps must increase.")
    eligible = np.flatnonzero(values >= threshold)
    order = eligible[np.argsort(-values[eligible], kind="stable")]
    chosen = []
    for index in order:
        if all(abs(times[index] - times[j]) >= min_separation_s
               for j in chosen):
            chosen.append(int(index))
    return [(float(times[j]), float(values[j])) for j in sorted(chosen)]


def _near_catalog(time_s, catalog_s, radius_s):
    a = np.asarray(catalog_s, dtype=float)
    return bool(len(a) and np.min(np.abs(a-time_s)) <= radius_s)


def evaluate_scans(
    scans, all_mpb_s, all_bs_s, *, threshold, tolerance_s=120.,
    exclusion_s=390., min_separation_s=300.,
):
    """Evaluate one detection per catalog event, plus weak background alarms.

    'scans': sequence of dicts with event_id, target_s, times_s, scores.
    An unmatched peak near ANY catalog MPB/BS is UNVERIFIED, not forced
    to a false positive. Reported precision is therefore an optimistic
    proxy limited to weakly labeled event neighborhoods, NOT full-orbit PPV.
    """
    if not 0 < tolerance_s < exclusion_s:
        raise ValueError("Require 0 < target tolerance < exclusion radius.")
    records = []
    tp = fp = ignored = 0
    absolute_errors = []
    for scan in scans:
        target = float(scan["target_s"])
        peaks = independent_peaks(
            scan["times_s"], scan["scores"], threshold=threshold,
            min_separation_s=min_separation_s,
        )
        nearby = [(t, score) for t, score in peaks
                  if abs(t-target) <= tolerance_s]
        chosen = max(nearby, key=lambda row: row[1]) if nearby else None
        if chosen is not None:
            tp += 1
            absolute_errors.append(abs(chosen[0]-target))
        false_here, unknown_here = 0, 0
        for time_s, score in peaks:
            if chosen is not None and time_s == chosen[0]:
                continue
            if (_near_catalog(time_s, all_mpb_s, exclusion_s) or
                    _near_catalog(time_s, all_bs_s, exclusion_s)):
                unknown_here += 1
            else:
                false_here += 1
        fp += false_here
        ignored += unknown_here
        records.append({
            "mpb_timestamp_utc": scan["event_id"],
            "detected": int(chosen is not None),
            "predicted_offset_s": (
                chosen[0] - target if chosen is not None else ""
            ),
            "absolute_error_s": (
                abs(chosen[0]-target) if chosen is not None else ""
            ),
            "peak_score": chosen[1] if chosen is not None else "",
            "n_independent_peaks": len(peaks),
            "n_weak_background_false_peaks": false_here,
            "n_unverified_peaks": unknown_here,
            "n_scored_windows": len(scan["scores"]),
        })
    fn = len(scans) - tp
    precision = tp / (tp+fp) if tp+fp else 0.
    recall = tp / len(scans) if scans else 0.
    return {
        "n_target_events": len(scans),
        "target_tp": tp,
        "target_fn": fn,
        "weak_background_fp": fp,
        "unverified_peaks": ignored,
        "weak_detection_precision": precision,
        "target_recall": recall,
        "weak_detection_f1": (
            2*precision*recall/(precision+recall)
            if precision+recall else 0.
        ),
        "weak_fp_per_70min_block": fp/max(1, len(scans)),
        "median_detected_abs_error_s": (
            float(np.median(absolute_errors)) if absolute_errors else None
        ),
        "fraction_targets_detected_within_60s": (
            sum(err <= 60. for err in absolute_errors)/max(1,len(scans))
        ),
        "warning": (
            "Positive-centered blocks; unknown peaks near catalog MPB/BS "
            "excluded from FP. Precision is a weak proxy, not orbit-wide "
            "precision or false-alarm rate."
        ),
    }, records


def calibrate_scan_threshold(scans, all_mpb_s, all_bs_s):
    """Select one fold-specific threshold using ONLY inner TRAIN-month scans."""
    if not scans:
        raise ValueError("No monitor scans to calibrate.")
    best_key, best = None, None
    for threshold in np.linspace(.05, .95, 19):
        result, _ = evaluate_scans(
            scans, all_mpb_s, all_bs_s, threshold=float(threshold)
        )
        key = (
            result["weak_detection_f1"],
            -result["weak_background_fp"],
            threshold,
        )
        if best_key is None or key > best_key:
            best_key, best = key, (float(threshold), result)
    return best
