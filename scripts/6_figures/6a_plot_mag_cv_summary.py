#!/usr/bin/env python3
"""Figure for any full 4-fold MAG training, using saved CV metrics only.

Reads cv_summary.json, not BigQuery. Does not hard-code measured results.
Default input:  results/metrics/mag_cv_600windows_det10/cv_summary.json
Default output: results/metrics/mag_cv_600windows_det10/figures/mag_cv_second_training.png
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import matplotlib

matplotlib.use("Agg")  # Render safely from WSL/tmux without a graphical display.
import matplotlib.pyplot as plt


DEFAULT_RESULTS = Path("results/metrics/mag_cv_600windows_det10")


def load_summary(path: Path):
    if not path.is_file():
        raise FileNotFoundError(f"No se encontró el resumen: {path}")
    data = json.loads(path.read_text(encoding="utf-8"))
    if data.get("instrument") != "MAG":
        raise ValueError("El resumen no corresponde al instrumento MAG.")
    if data.get("pilot_subset"):
        raise ValueError("Esta figura se reserva al entrenamiento completo, no a un piloto.")

    rows = data.get("fold_results")
    if not isinstance(rows, list) or not rows:
        raise ValueError("No hay métricas por fold en cv_summary.json.")
    folds = [int(r["fold"]) for r in rows]
    if len(set(folds)) != len(folds):
        raise ValueError("Hay folds duplicados.")
    if sorted(folds) != [1, 2, 3, 4]:
        raise ValueError("Se esperan los cuatro folds 1, 2, 3 y 4.")

    ordered = sorted(rows, key=lambda r: int(r["fold"]))
    for row in ordered:
        n = int(row["val_evaluable_events"])
        total = int(row["val_total_events"])
        median = float(row["median_abs_peak_error_s"])
        frac = float(row["fraction_within_60s"])
        if not (0 < n <= total and median >= 0 and 0 <= frac <= 1):
            raise ValueError(f"Métricas fuera de rango en fold {row['fold']}.")
    return ordered


def make_figure(rows, output_path: Path, *, experiment_label: str):
    folds = [int(r["fold"]) for r in rows]
    medians_min = [float(r["median_abs_peak_error_s"]) / 60. for r in rows]
    successes_pct = [100. * float(r["fraction_within_60s"]) for r in rows]
    counts = [int(r["val_evaluable_events"]) for r in rows]

    # The fraction is computed from the number of successes in each fold.
    n_total = sum(counts)
    success_total = sum(round(float(r["fraction_within_60s"]) * n)
                        for r, n in zip(rows, counts))
    success_pct = 100. * success_total / n_total

    fig, axes = plt.subplots(1, 2, figsize=(11.5, 5.1), layout="constrained")
    x = range(len(folds))

    bars1 = axes[0].bar(x, medians_min, width=0.65)
    axes[0].set_title("(a) Error de localización")
    axes[0].set_ylabel("Error temporal absoluto mediano (min)")
    axes[0].set_xticks(list(x), [str(f) for f in folds])
    axes[0].set_xlabel("Fold de validación")
    axes[0].set_ylim(0, max(medians_min) * 1.24)
    axes[0].grid(axis="y", alpha=0.25)
    axes[0].set_axisbelow(True)
    for bar, median, n in zip(bars1, medians_min, counts):
        axes[0].annotate(
            f"{median:.1f} min\nN={n}",
            (bar.get_x() + bar.get_width() / 2, bar.get_height()),
            ha="center", va="bottom", xytext=(0, 4),
            textcoords="offset points", fontsize=9
        )

    bars2 = axes[1].bar(x, successes_pct, width=0.65)
    axes[1].set_title("(b) Localización dentro de 60 s")
    axes[1].set_ylabel("Cruces con error absoluto ≤ 60 s (%)")
    axes[1].set_xlabel("Fold de validación")
    axes[1].set_xticks(list(x), [str(f) for f in folds])
    axes[1].set_ylim(0, max(successes_pct) * 1.30)
    axes[1].grid(axis="y", alpha=0.25)
    axes[1].set_axisbelow(True)
    for bar, success in zip(bars2, successes_pct):
        axes[1].annotate(
            f"{success:.1f}%",
            (bar.get_x() + bar.get_width() / 2, bar.get_height()),
            ha="center", va="bottom", xytext=(0, 4),
            textcoords="offset points", fontsize=9
        )

    fig.suptitle(f"MAVEN MAG — {experiment_label}, validación cruzada (4 folds)",
                 fontsize=14, fontweight="bold")
    fig.supxlabel(
        f"Total: {n_total} MPB evaluadas  |  "
        f"Error ≤ 60 s: {success_total}/{n_total} ({success_pct:.1f}%)\n"
        "Métricas de localización condicionadas a cruces etiquetados; "
        "no representan precisión o recall.",
        fontsize=9
    )

    output_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(output_path, dpi=220, bbox_inches="tight")
    plt.close(fig)
    return output_path


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--summary", type=Path, default=DEFAULT_RESULTS / "cv_summary.json",
                        help="Ruta a cv_summary.json del entrenamiento MAG completo.")
    parser.add_argument("--output", type=Path, default=None,
                        help="Ruta PNG de salida. Por defecto junto al resumen elegido.")
    args = parser.parse_args()

    rows = load_summary(args.summary)
    summary = json.loads(args.summary.read_text(encoding="utf-8"))
    config = summary.get("training_configuration", {})
    is_second = (
        float(config.get("lambda_detection", 1.)) == 10.
        and int(config.get("train_targets_per_event", 24)) == 600
    )
    name = "segundo entrenamiento" if is_second else "primer entrenamiento"
    if args.output is None:
        filename = ("mag_cv_second_training.png" if is_second
                    else "mag_cv_first_training.png")
        args.output = args.summary.parent / "figures" / filename
    saved = make_figure(rows, args.output, experiment_label=name)
    print(f"Figura guardada en: {saved.resolve()}")


if __name__ == "__main__":
    main()
