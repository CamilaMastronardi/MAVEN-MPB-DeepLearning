#!/usr/bin/env python3
"""Evaluate MAG tanh checkpoints using ONE 70-minute overlap-added classification.

This does NOT retrain models, access BigQuery, or overwrite the old pilot.
The model receives only 600-second MAG windows; their classification curves
are combined on original MAG timestamps before a single global tanh fit.

Run from the repo root after the 5o pilot has produced checkpoints:
  python scripts/5_training/5p_eval_mag_global_tanh.py --folds 1 --pilot-val 12
"""
from __future__ import annotations

import argparse
import csv
import importlib.util
import json
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from scipy.optimize import least_squares
import torch

ROOT = Path(__file__).resolve().parents[2]
PREVIOUS = ROOT / "scripts/5_training/5o_train_mag_tanh_zero.py"
VARIANTS = ("fixed", "adaptive", "consistent")


def load_previous():
    spec = importlib.util.spec_from_file_location("mag_tanh_5o", PREVIOUS)
    if spec is None or spec.loader is None:
        raise FileNotFoundError(f"Cannot import previous MAG experiment: {PREVIOUS}")
    import sys
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


def window_centers(stride_s: int) -> np.ndarray:
    centers = np.arange(-1800., 1800.01, stride_s)
    if centers[-1] < 1800:
        centers = np.append(centers, 1800.)
    return np.unique(centers)


def predict_global_curve(old, model, block, normalizer, anchor_s: float,
                         *, stride_s: int) -> dict:
    """Overlap-add native-timestamp predictions; no temporal interpolation.

    The anchor is a known catalog MPB defining a CONDITIONAL ±30-minute
    search area, not a network input and not an offset supplied to a fit.
    """
    times, B = block
    roi = (times >= anchor_s - 2100.) & (times <= anchor_s + 2100.)
    original_indices = np.flatnonzero(roi)
    if len(original_indices) < 3000:
        raise ValueError("Incomplete source support for 70-minute evaluation")
    idx0, idx1 = int(original_indices[0]), int(original_indices[-1]) + 1
    global_time = times[idx0:idx1] - anchor_s
    B_roi = B[idx0:idx1]
    accumulator = np.zeros(len(global_time), dtype=np.float64)
    total_weight = np.zeros(len(global_time), dtype=np.float64)
    used = 0
    for offset in window_centers(stride_s):
        center = anchor_s + float(offset)
        clip = old.extract_window(block, center)
        if clip is None:
            continue
        rel_time, X = clip
        # clip uses the exact time-selection convention from the cached MAG block.
        left = int(np.searchsorted(times, center - 300., side="left"))
        right = int(np.searchsorted(times, center + 300., side="right"))
        if right-left != len(rel_time):
            raise RuntimeError("MAG clip indices disagree with physical timestamps")
        # The CNN has zero-padded edges. Down-weight edges, but allow coverage.
        weight = 0.05 + 0.95 * np.maximum(0., 1. - np.abs(rel_time.astype(float))/300.)
        curve, _ = old.predict_clip(model, clip, normalizer)
        a, b = left - idx0, right - idx0
        if a < 0 or b > len(global_time):
            raise RuntimeError("A 600-second clip exceeds the 70-minute support")
        accumulator[a:b] += weight * curve
        total_weight[a:b] += weight
        used += 1
    valid = total_weight > 0
    signal = np.full(len(global_time), np.nan, dtype=np.float64)
    signal[valid] = accumulator[valid] / total_weight[valid]
    return {"time_s": global_time, "class_curve": signal,
            "B_nt": B_roi, "covered": valid, "coverage_fraction": float(valid.mean()),
            "n_windows": used}


def fit_one_global_tanh(t, y, valid) -> dict:
    """Fit exactly one amplitude*tanh((t-zero)/tau), not one per clip.

    A fitted zero is accepted only if measured model output crosses zero
    and its full-interval tails have opposite signs. No catalog zero prior
    enters the optimizer's initialization.
    """
    m = valid & np.isfinite(y) & (t >= -2090.) & (t <= 2090.)
    x = np.asarray(t[m], dtype=float)
    f = np.asarray(y[m], dtype=float)
    if len(x) < 2000:
        return {"ok": False, "reason": "insufficient_covered_samples"}
    left = f[x < -1500.]
    right = f[x > 1500.]
    if len(left) < 150 or len(right) < 150:
        return {"ok": False, "reason": "insufficient_tails"}
    left_level = float(np.median(left))
    right_level = float(np.median(right))
    contrast = abs(right_level-left_level)
    if left_level*right_level >= 0 or contrast < 0.35:
        return {"ok": False, "reason": "no_global_sign_transition",
                "tail_contrast": contrast}
    # Find observed sign-changing transitions in prediction, not a catalog-based seed.
    smooth = np.convolve(f, np.ones(31)/31, mode="same")
    changes = np.flatnonzero(smooth[:-1]*smooth[1:] <= 0)
    if not len(changes):
        return {"ok": False, "reason": "no_observed_zero_crossing",
                "tail_contrast": contrast}
    guesses = x[changes]
    # Prefer a crossing in a region with sustained before/after contrast.
    # Do not use distance to the catalog label for selecting it.
    slopes = np.abs(smooth[np.minimum(changes+30, len(smooth)-1)] -
                    smooth[np.maximum(changes-30, 0)])
    seed_center = float(guesses[int(np.argmax(slopes))])
    if abs(seed_center) > 1800.:
        return {"ok": False, "reason": "observed_zero_outside_search"}
    direction = np.sign(right_level-left_level)
    amplitude_seed = float(np.clip(direction*contrast/2., -1.4, 1.4))
    initial = [amplitude_seed, seed_center, 120.]
    try:
        fit = least_squares(
            lambda p: p[0]*np.tanh((x-p[1])/p[2])-f,
            x0=initial, bounds=([-1.5, -1800., 8.], [1.5, 1800., 900.]),
            loss="soft_l1", f_scale=0.2, max_nfev=100,
        )
    except (ValueError, FloatingPointError) as exc:
        return {"ok": False, "reason": f"fit_failed:{exc}"}
    amplitude, zero, tau = (float(z) for z in fit.x)
    mse = float(np.mean((amplitude*np.tanh((x-zero)/tau)-f)**2))
    if (not fit.success or abs(amplitude) < 0.2 or
            abs(zero) >= 1799. or abs(tau-900.) < 1.e-3):
        return {"ok": False, "reason": "degenerate_global_fit",
                "fit_mse": mse, "tail_contrast": contrast}
    return {"ok": True, "pred_offset_s": zero, "fit_tau_s": tau,
            "amplitude": amplitude, "fit_mse": mse, "tail_contrast": contrast,
            "fit_curve": amplitude*np.tanh((t-zero)/tau)}


def write_csv(filename, rows):
    if not rows:
        return
    columns = sorted({k for r in rows for k in r})
    filename.parent.mkdir(parents=True, exist_ok=True)
    with filename.open("w", encoding="utf-8", newline="") as h:
        writer = csv.DictWriter(h, fieldnames=columns)
        writer.writeheader()
        writer.writerows(rows)


def plot_one(output, event, curves):
    fig, (ax, bx) = plt.subplots(2, 1, figsize=(13, 7), sharex=True)
    for variant, (signal, fit) in curves.items():
        x = signal["time_s"]
        ax.plot(x, signal["class_curve"], alpha=0.67,
                label=f"{variant}: clasificación combinada")
        if fit["ok"]:
            ax.plot(x, fit["fit_curve"], ls="--", lw=1.3,
                    label=f"{variant}: tanh global")
            ax.axvline(fit["pred_offset_s"], ls=":", lw=1.1,
                       label=f"{variant}: MPB estimada")
    first = next(iter(curves.values()))[0]
    bx.plot(first["time_s"], first["B_nt"][:, 3], color="gray", alpha=0.8,
            label="|B| observado")
    for a in (ax, bx):
        a.axvline(0, color="black", ls="--", label="MPB catalogada (solo referencia)")
        a.legend(loc="best", fontsize=8)
    ax.set_ylabel("Clasificación reconstruida [-1,1]")
    bx.set_ylabel("|B| medido (nT)")
    bx.set_xlabel("Tiempo respecto a MPB catalogada (s)")
    ax.set_title(f"Predicción global a partir de ventanas de 600 s | {event['mpb_timestamp_utc']}")
    fig.tight_layout()
    fig.savefig(output, dpi=160)
    plt.close(fig)


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--folds", choices=["all", "1", "2", "3", "4"], default="1")
    p.add_argument("--variants", choices=["all", *VARIANTS], default="all")
    p.add_argument("--pilot-val", type=int, default=12)
    p.add_argument("--scan-stride", type=int, default=64)
    p.add_argument("--plot-events", type=int, default=3)
    p.add_argument("--torch-threads", type=int, default=2)
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--checkpoints-dir", type=Path, default=ROOT/"results/metrics/mag_tanh_zero_pilot")
    p.add_argument("--output-dir", type=Path, default=ROOT/"results/metrics/mag_tanh_global_eval_pilot")
    args = p.parse_args()
    if args.scan_stride < 1 or args.pilot_val < 0 or args.plot_events < 0 or args.torch_threads < 1:
        p.error("Invalid positive scan stride, plot count or PyTorch threads")
    torch.set_num_threads(args.torch_threads)
    old = load_previous()
    catalog = old.read_catalog(ROOT / "data/splits/cv4_monthly_manifest.csv")
    cache_files = old.locate_mag_cache(ROOT / "data/cache/mag_events", catalog)
    variants = VARIANTS if args.variants == "all" else (args.variants,)
    folds = [1,2,3,4] if args.folds == "all" else [int(args.folds)]
    args.output_dir.mkdir(parents=True, exist_ok=True)
    summaries = []
    for fold in folds:
        val = [r for r in catalog if r["fold"] == fold and r["time"] in cache_files]
        val = old.select_even(val, args.pilot_val)
        if not val:
            print(f"Fold {fold}: no cached MAG validation events", flush=True)
            continue
        root = args.output_dir / f"fold_{fold:02d}"
        root.mkdir(parents=True, exist_ok=True)
        all_rows, figures = [], {}
        for variant in variants:
            checkpoint = args.checkpoints_dir/f"fold_{fold:02d}"/f"fold_{fold:02d}_{variant}_best.pt"
            if not checkpoint.is_file():
                raise FileNotFoundError(f"Need old trained checkpoint: {checkpoint}")
            ck = torch.load(checkpoint, map_location="cpu", weights_only=False)
            if ck.get("variant") != variant or ck.get("fold") != fold:
                raise RuntimeError(f"Unexpected model identity in checkpoint {checkpoint}")
            model = old.MagTanhAE()
            model.load_state_dict(ck["state_dict"])
            model.eval()
            norm = ck["normalizer"]
            rows = []
            for j, event in enumerate(val):
                block = old.load_block(cache_files[event["time"]], event["time"])
                signal = predict_global_curve(old, model, block, norm, event["time"],
                                              stride_s=args.scan_stride)
                fit = fit_one_global_tanh(signal["time_s"], signal["class_curve"],
                                          signal["covered"])
                item = {"variant": variant, "fold": fold,
                        "event_utc": event["mpb_timestamp_utc"],
                        "evaluated": fit["ok"], "failure_reason": "" if fit["ok"] else fit["reason"],
                        "n_windows": signal["n_windows"], "coverage_fraction": signal["coverage_fraction"],
                        "pred_offset_s": fit.get("pred_offset_s"),
                        "abs_error_s": abs(fit["pred_offset_s"]) if fit["ok"] else None,
                        "tau_fit_s": fit.get("fit_tau_s"), "fit_mse": fit.get("fit_mse"),
                        "tail_contrast": fit.get("tail_contrast")}
                rows.append(item)
                print(f"Fold {fold} {variant} {j+1}/{len(val)}: "
                      f"{'err='+str(round(item['abs_error_s'],1))+'s' if fit['ok'] else fit['reason']}",
                      flush=True)
                if j < args.plot_events:
                    figures.setdefault(j, {"event": event})[variant] = (signal, fit)
            all_rows.extend(rows)
            write_csv(root/f"fold_{fold:02d}_{variant}_global_events.csv", rows)
            errors = [r["abs_error_s"] for r in rows if r["evaluated"]]
            summary = {"fold": fold, "variant": variant, "count": len(rows),
                       "evaluable": len(errors), "evaluable_fraction": len(errors)/len(rows),
                       "median_abs_error_s": float(np.median(errors)) if errors else None,
                       "fraction_within_60s_all": sum(e<=60 for e in errors)/len(rows),
                       "fraction_within_120s_all": sum(e<=120 for e in errors)/len(rows),
                       "fraction_within_60s_evaluable": (sum(e<=60 for e in errors)/len(errors)
                                                          if errors else None),
                       "fraction_within_120s_evaluable": (sum(e<=120 for e in errors)/len(errors)
                                                          if errors else None)}
            summaries.append(summary)
            print(summary, flush=True)
        for j, group in figures.items():
            plot_one(root/f"fold_{fold:02d}_global_{j+1:02d}.png", group["event"],
                     {k:v for k,v in group.items() if k!="event"})
        # No model-selection decisions are made from these held-out results.
        del all_rows
    write_csv(args.output_dir/"summary.csv", summaries)
    (args.output_dir/"summary.json").write_text(json.dumps({"args":vars(args),
         "results":summaries}, indent=2, default=str)+"\n", encoding="utf-8")
    print(f"Global 70-min inference results: {args.output_dir}", flush=True)


if __name__ == "__main__":
    main()