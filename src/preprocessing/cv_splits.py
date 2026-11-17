"""Separacion FIT/MONITOR por meses con soportes fisicos disjuntos.

Funciones extraidas SIN CAMBIAR las reglas ni la semilla usada en V7b.
"""
from __future__ import annotations
import random

SUPPORT_MINUTES = 35.0

def check_disjoint_supports(train, val):
    """Reject train/val overlap in the +/-35 min physical measurement supports."""
    times = sorted(r["time"].timestamp() for r in train)
    held = sorted(r["time"].timestamp() for r in val)
    if not times or not held:
        raise ValueError("Each fold needs both training and validation events.")
    j = 0
    for t in held:
        while j + 1 < len(times) and times[j + 1] <= t:
            j += 1
        for k in (j, j + 1):
            if k < len(times) and abs(times[k] - t) <= 2 * SUPPORT_MINUTES * 60:
                raise ValueError("Train/val physical data intervals overlap.")

def split_inner_monitor_months(train, fold, seed):
    """Keep OUTER validation independent of early stopping.

    Hold out ~10% of training months for early-stop monitoring. Neighboring
    TRAIN event supports that overlap an inner-monitor support are removed
    from gradient training (never moved to outer validation).
    V7b ajusta su normalizacion solo sobre FIT, excluyendo MONITOR
    y el fold externo de evaluacion.
    """
    from bisect import bisect_left
    months = sorted({r["month_block"] for r in train})
    if len(months) < 2:
        raise ValueError("Need >=2 different TRAIN month blocks for inner holdout.")
    rng = random.Random(seed + fold * 191)
    rng.shuffle(months)
    count = max(1, min(len(months)-1, round(0.10 * len(months))))
    held = set(months[:count])
    inner = [r for r in train if r["month_block"] in held]
    fit = [r for r in train if r["month_block"] not in held]
    if not inner or not fit:
        raise ValueError("Inner-month split produced an empty group.")

    held_times = sorted(r["time"].timestamp() for r in inner)
    safe_fit = []
    excluded_edges = 0
    for row in fit:
        t = row["time"].timestamp()
        j = bisect_left(held_times, t)
        neighbors = held_times[max(0, j-1):j+1]
        if any(abs(t - v) <= 2 * SUPPORT_MINUTES * 60 for v in neighbors):
            excluded_edges += 1
        else:
            safe_fit.append(row)
    check_disjoint_supports(safe_fit, inner)
    if not safe_fit:
        raise ValueError("No disjoint TRAIN events after inner-month holdout.")
    return safe_fit, inner, excluded_edges

