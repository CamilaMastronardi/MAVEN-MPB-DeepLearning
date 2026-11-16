"""Conservative, auditable weak MPB/BS supervision for seventh classifier.

Catalog MPB and associated BS timestamps determine an APPROXIMATE physical
orientation. One cannot infer MPB thickness from the annotation. Intervals
near MPB, BS and any other annotated MPB remain unknown, not negatives.
The model receives none of the event labels/timestamps as input features.
"""
from __future__ import annotations

from dataclasses import dataclass

import numpy as np


SHEATH = 0
INDUCED = 1


@dataclass(frozen=True)
class Candidate:
    event_time_s: float
    center_s: float
    transition: int | None
    left_region: int | None
    right_region: int | None
    kind: str


def region_orientation(event):
    """Use chronological BS/MPB association, not MAG-derived answers."""
    mpb = event["time"].timestamp()
    bs = event["bs_time"].timestamp()
    if not np.isfinite(bs) or abs(bs - mpb) < 300:
        return None
    # Entering Mars: BS -> sheath -> MPB -> induced.
    # Leaving Mars: induced -> MPB -> sheath -> BS.
    if bs < mpb:
        return SHEATH, INDUCED
    return INDUCED, SHEATH


def nearest_other_distance(center, labels, own_time):
    others = np.asarray(labels, dtype=float)
    others = others[np.abs(others - own_time) > .01]
    if not len(others):
        return float("inf")
    return float(np.min(np.abs(others - center)))


def classify_candidate(
    event, candidate_s, *, all_mpb_s, all_bs_s,
    half_window_s=300.0, positive_radius_s=35.0,
    stable_radius_s=600.0, exclusion_s=90.0
):
    """Return positive/weak negative plus region labels or unknown candidate.

    Positive MPB candidates must be isolated from BS/other MPB by the
    complete input support, and have reliable before/after regions.
    Negative candidates must lie wholly in one inferred stable region,
    far from ALL catalog MPB/BS times; this does not guarantee true
    physical negatives if the catalog is incomplete.
    """
    t = event["time"].timestamp()
    bs = event["bs_time"].timestamp()
    offset = candidate_s - t
    sides = region_orientation(event)
    if sides is None:
        return Candidate(t, candidate_s, None, None, None, "unknown_bs")
    margin = half_window_s + exclusion_s
    if abs(candidate_s - bs) <= margin:
        return Candidate(t, candidate_s, None, None, None, "near_bs")
    other_dist = nearest_other_distance(candidate_s, all_mpb_s, t)
    if other_dist <= margin:
        return Candidate(t, candidate_s, None, None, None, "other_mpb")
    bs_array = np.asarray(all_bs_s, dtype=float)
    if len(bs_array):
        other_bs = bs_array[np.abs(bs_array - bs) > .01]
        if len(other_bs) and np.min(np.abs(other_bs - candidate_s)) <= margin:
            return Candidate(t, candidate_s, None, None, None, "other_bs")

    if abs(offset) <= positive_radius_s:
        # The full left/right supports must fall on their intended sides
        # of the associated BS; MPB-centered input crosses the MPB by design.
        if abs(bs - t) <= half_window_s + positive_radius_s + exclusion_s:
            return Candidate(t, candidate_s, None, None, None, "short_sheath")
        return Candidate(t, candidate_s, 1, sides[0], sides[1], "positive")

    if abs(offset) < stable_radius_s:
        return Candidate(t, candidate_s, None, None, None, "mpb_ambiguous")
    if abs(offset) <= margin:
        return Candidate(t, candidate_s, None, None, None, "near_mpb")

    # Candidate support is entirely one side of the annotated MPB.
    same_region = sides[0] if offset < 0 else sides[1]
    # The sheath should be between BS and MPB, not beyond the BS into
    # solar wind. Requiring BS exclusion also protects its support.
    if same_region == SHEATH:
        if not (min(bs, t) + margin < candidate_s
                < max(bs, t) - margin):
            return Candidate(t, candidate_s, None, None, None, "outside_sheath")
    return Candidate(
        t, candidate_s, 0, same_region, same_region, "stable_control"
    )


def candidate_training_grid(event, all_mpb_s, all_bs_s):
    """Three positives, up to eight hard controls with physical safeguards."""
    center = event["time"].timestamp()
    offsets = (-30., 0., 30., -1500., -1200., -900., -600.,
               600., 900., 1200., 1500.)
    candidates = [
        classify_candidate(
            event, center + d,
            all_mpb_s=all_mpb_s, all_bs_s=all_bs_s
        )
        for d in offsets
    ]
    return [c for c in candidates if c.transition is not None]


def candidate_evaluation_grid(event, all_mpb_s, all_bs_s, *, step_s=32.):
    """Physical search over +/-30 min, with unknown candidates retained."""
    if step_s <= 0:
        raise ValueError("Evaluation stride must be positive.")
    center = event["time"].timestamp()
    offsets = np.arange(-1800., 1800.01, step_s)
    return [
        classify_candidate(
            event, center + float(d),
            all_mpb_s=all_mpb_s, all_bs_s=all_bs_s
        ) for d in offsets
    ]
