#!/usr/bin/env python3
"""Post-run seventh MAG+SWEA CV figures, using saved JSON/CSV only.

No BigQuery, no model execution. Paired comparisons match shared event
timestamps and fold, explicitly excluding v5-only SWEA-missing events.
Candidate F1 uses WEAK labels and is not orbital detection performance.
"""
from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np


def load_events(base):
    result = {}
    for fold in range(1, 5):
        path = base/f"fold_{fold:02d}_event_metrics.csv"
        if not path.is_file():
            continue
        with path.open(newline="", encoding="utf-8") as f:
            for row in csv.DictReader(f):
                if row.get("absolute_error_s", "") == "":
                    continue
                key = (fold, row["mpb_timestamp_utc"])
                if key in result:
                    raise ValueError(f"Duplicate MPB timestamp {key}.")
                result[key] = float(row["absolute_error_s"])
    return result


def compare_shared(fifth, seventh, out):
    shared = sorted(set(fifth) & set(seventh))
    if not shared:
        raise ValueError("No paired MAG-only and MAG+SWEA events available.")
    x = np.asarray([fifth[k] for k in shared])
    y = np.asarray([seventh[k] for k in shared])
    report = {
        "paired_evaluable_events": len(shared),
        "fifth_only_events": len(set(fifth) - set(seventh)),
        "seventh_only_events": len(set(seventh) - set(fifth)),
        "fifth_paired_median_error_s": float(np.median(x)),
        "seventh_paired_median_error_s": float(np.median(y)),
        "fifth_paired_within_60s": int(np.sum(x <= 60.)),
        "seventh_paired_within_60s": int(np.sum(y <= 60.)),
        "individual_improvements": int(np.sum(y < x-.002)),
        "individual_worsenings": int(np.sum(y > x+.002)),
        "individual_unchanged": int(np.sum(np.abs(y-x) <= .002)),
        "note": (
            "Only shared, evaluable catalog events. Both peaks conditioned "
            "on a labeled MPB within a +/-30-minute search."
        ),
    }
    fig, axs = plt.subplots(1, 2, figsize=(11.5, 4.8),
                            layout="constrained")
    max_min = max(30., float(np.max(np.r_[x, y])/60.))
    axs[0].scatter(x/60., y/60., s=10, alpha=.5)
    axs[0].plot([0, max_min], [0, max_min], "k--", lw=1)
    axs[0].set(xlabel="V5 MAG-only: error absoluto (min)",
               ylabel="V7 MAG+SWEA: error absoluto (min)",
               title=f"Comparación de {len(shared)} MPB comunes",
               xlim=(0, max_min), ylim=(0, max_min))
    thresholds = np.linspace(0, max_min, 500)
    for z, label in ((x, "V5 MAG"), (y, "V7 MAG+SWEA")):
        axs[1].plot(
            thresholds,
            np.searchsorted(np.sort(z/60.), thresholds, side="right")
            *100/len(z), label=label
        )
    axs[1].set(
        xlabel="Tolerancia de localización (min)",
        ylabel="MPB dentro de tolerancia (%)",
        title="Fracción acumulada, mismos eventos",
        ylim=(0, 101)
    )
    axs[1].legend()
    for ax in axs:
        ax.grid(alpha=.2)
    fig.savefig(out, dpi=190)
    plt.close(fig)
    return report


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument(
        "--results-dir", type=Path,
        default=Path("results/metrics/mag_swea_regions_v1")
    )
    p.add_argument(
        "--compare-v5", type=Path,
        default=Path("results/metrics/mag_cv_residual_transition_v1")
    )
    args = p.parse_args()
    directory = args.results_dir
    summary_path = directory/"cv_summary.json"
    if not summary_path.exists():
        raise FileNotFoundError(summary_path)
    summary = json.loads(summary_path.read_text())
    if sorted(summary.get("completed_folds", [])) != [1, 2, 3, 4]:
        raise ValueError("All four folds must be complete before CV figures.")
    rows = sorted(
        summary["fold_results"], key=lambda row: int(row["fold"])
    )
    figures = directory/"figures"
    figures.mkdir(parents=True, exist_ok=True)

    fig, axs = plt.subplots(1, 3, figsize=(15, 4.5),
                            layout="constrained")
    folds = [r["fold"] for r in rows]
    axs[0].bar(folds, [r["median_abs_peak_error_s"]/60. for r in rows])
    axs[0].set(title="MPB: error mediano", xlabel="Fold",
               ylabel="Minutos")
    axs[1].bar(folds, [100*r["fraction_within_60s"] for r in rows])
    axs[1].set(title="Localización dentro de 60 s", xlabel="Fold",
               ylabel="Eventos (%)")
    axs[2].bar(
        folds, [r["candidate_weak_f1"] for r in rows]
    )
    axs[2].set(title="F1 candidatos con etiquetas débiles", xlabel="Fold",
               ylabel="F1", ylim=(0, 1))
    for ax in axs:
        ax.set_xticks(folds)
        ax.grid(axis="y", alpha=.2)
    fig.suptitle(
        "MAVEN MAG+SWEA V7 — 4-fold CV | F1 no es detección orbital",
        fontsize=12
    )
    fig.savefig(figures/"mag_swea_cv_overview.png", dpi=190)
    plt.close(fig)

    fig, axes = plt.subplots(3, 4, figsize=(15, 9),
                             layout="constrained")
    losses = [
        ("train_det", "val_det", "Transición MPB (BCE)"),
        ("train_region", "val_region", "Región plasmática (BCE)"),
        ("train_total", "val_total", "Pérdida total"),
    ]
    for column, fold in enumerate(range(1, 5)):
        record = json.loads(
            (directory/f"fold_{fold:02d}_metrics.json").read_text()
        )
        history = record["train_history"]
        epochs = [h["epoch"] for h in history]
        for line, (tr, va, title) in enumerate(losses):
            ax = axes[line, column]
            ax.plot(epochs, [h[tr] for h in history], label="TRAIN")
            ax.plot(epochs, [h[va] for h in history], label="Monitor")
            ax.axvline(record["best_epoch"], ls="--", color="gray")
            ax.set_title(f"Fold {fold}: {title}", fontsize=9)
            ax.grid(alpha=.2)
        axes[2, column].set_xlabel("Época")
    axes[0, 0].legend()
    fig.suptitle(
        "TRAIN vs monitor interno (meses TRAIN, no validación externa)"
    )
    fig.savefig(figures/"mag_swea_cv_losses.png", dpi=190)
    plt.close(fig)

    events = load_events(directory)
    errors = np.array(list(events.values()), dtype=float)
    fig, axs = plt.subplots(1, 2, figsize=(11, 4.3),
                            layout="constrained")
    axs[0].hist(errors/60, bins=np.arange(0, 32, 1))
    axs[0].set(xlabel="Error absoluto (min)", ylabel="Eventos",
               title="Distribución de errores")
    thresholds = np.linspace(0, 30, 301)
    axs[1].plot(thresholds, 100*np.searchsorted(
        np.sort(errors/60.), thresholds, side="right"
    )/len(errors))
    axs[1].set(xlabel="Tolerancia (min)",
               ylabel="MPB localizadas (%)",
               title="Fracción acumulada")
    for ax in axs:
        ax.grid(alpha=.2)
    fig.savefig(figures/"mag_swea_cv_errors.png", dpi=190)
    plt.close(fig)

    print(f"V7 evaluable MPBs: {len(errors)}")
    print(f"V7 global median error: {np.median(errors):.2f} s")
    print(f"V7 within 60 s: {sum(errors <= 60)}/{len(errors)}")
    if args.compare_v5.exists():
        earlier = load_events(args.compare_v5)
        if earlier:
            report = compare_shared(
                earlier, events, figures/"mag_swea_v5_vs_v7_shared.png"
            )
            (figures/"mag_swea_v5_vs_v7_shared.json").write_text(
                json.dumps(report, indent=2), encoding="utf-8"
            )
            print("Shared-event comparison:", report)
    print(f"Saved all figures to {figures.resolve()}")


if __name__ == "__main__":
    main()
