"""Descriptores de la forma del score temporal de una MPB (post-inferencia).

No entrena modelos, no utiliza BigQuery ni interpreta sigmoid como
probabilidad calibrada. Los huecos temporales separan segmentos: nunca se
interpolan picos ni anchuras a través de datos ausentes.
"""
from __future__ import annotations

import numpy as np
from scipy.signal import find_peaks, peak_widths


def score_shape_metrics(
    offsets_s,
    scores,
    *,
    expected_stride_s=32.0,
    competitor_distance_s=300.0,
    min_secondary_prominence=0.05,
    gap_factor=1.5,
):
    """Caracteriza el máximo elegido por argmax, igual que evaluate_fold.

    Parámetros
    ----------
    offsets_s : array
        Centros candidatos en segundos respecto a la MPB catalogada.
    scores : array
        Salida sigmoid del clasificador de transición, sin calibración.
    expected_stride_s : float
        Paso de búsqueda nominal. Gaps mayores a gap_factor * paso
        impiden medir una anchura que cruce ese intervalo.
    competitor_distance_s : float
        Separación mínima respecto del máximo para definir una alternativa.
    min_secondary_prominence : float
        Prominencia mínima (escala de score) de un máximo local secundario.

    Retorna
    -------
    dict | None
        None si no hay centros finitos; NaN en descriptores no medibles.
        La anchura se calcula a la mitad de la prominencia (no a score=0.5).
        El segundo candidato es el de score más alto a >= distancia mínima;
        no se exige que sea un máximo local, para incluir extremos de gaps.
    """
    if expected_stride_s <= 0 or competitor_distance_s <= 0:
        raise ValueError("expected_stride_s y competitor_distance_s deben ser positivos.")
    if min_secondary_prominence < 0 or gap_factor <= 1:
        raise ValueError("Prominencia >= 0 y gap_factor > 1 son obligatorios.")

    x = np.asarray(offsets_s, dtype=float).reshape(-1)
    y = np.asarray(scores, dtype=float).reshape(-1)
    if x.size != y.size:
        raise ValueError("offsets_s y scores deben tener la misma longitud.")
    valid = np.isfinite(x) & np.isfinite(y)
    x, y = x[valid], y[valid]
    if not x.size:
        return None

    order = np.argsort(x, kind="stable")
    x, y = x[order], y[order]
    if np.any(np.diff(x) <= 0):
        raise ValueError("Los tiempos de los candidatos deben ser únicos.")

    best = int(np.argmax(y))  # primer máximo, como la evaluación V7b
    center = float(x[best])
    split = np.flatnonzero(np.diff(x) > gap_factor * expected_stride_s) + 1
    starts = np.r_[0, split]
    ends = np.r_[split, len(x)]

    prominence = float("nan")
    width_s = float("nan")
    left_s = float("nan")
    right_s = float("nan")
    best_edge = False
    secondary_peaks = 0

    for a, b in zip(starts, ends):
        xx, yy = x[a:b], y[a:b]
        locs, properties = find_peaks(
            yy, prominence=min_secondary_prominence
        ) if len(yy) >= 3 else (np.asarray([], dtype=int), {})
        for loc in locs:
            global_idx = a + int(loc)
            if (global_idx != best
                    and abs(x[global_idx] - center) >= competitor_distance_s):
                secondary_peaks += 1

        if a <= best < b:
            local_best = best - a
            best_edge = local_best == 0 or local_best == len(xx) - 1
            # La primera muestra de una meseta puede no ser el centro del pico
            # reconocido por scipy: no se asigna anchura ficticia.
            matches = np.flatnonzero(locs == local_best)
            if matches.size and not best_edge:
                prominence = float(properties["prominences"][int(matches[0])])
                _, _, left_ips, right_ips = peak_widths(
                    yy, [local_best], rel_height=0.5
                )
                grid = np.arange(len(xx), dtype=float)
                left_s = float(np.interp(left_ips[0], grid, xx))
                right_s = float(np.interp(right_ips[0], grid, xx))
                width_s = right_s - left_s

    remote = np.flatnonzero(np.abs(x - center) >= competitor_distance_s)
    remote_idx = int(remote[np.argmax(y[remote])]) if len(remote) else None
    competitor_score = float(y[remote_idx]) if remote_idx is not None else float("nan")

    return {
        "peak_offset_s": center,
        "absolute_error_s": abs(center),
        "peak_score": float(y[best]),
        "peak_prominence": prominence,
        "width_half_prominence_s": width_s,
        "width_left_s": left_s,
        "width_right_s": right_s,
        "competitor_offset_s": float(x[remote_idx]) if remote_idx is not None else float("nan"),
        "competitor_score": competitor_score,
        "peak_vs_competitor": float(y[best] - competitor_score)
        if remote_idx is not None else float("nan"),
        "n_competing_peaks": int(secondary_peaks),
        "peak_at_segment_edge": bool(best_edge),
        "n_segments": int(len(starts)),
        "n_centers": int(len(x)),
    }
