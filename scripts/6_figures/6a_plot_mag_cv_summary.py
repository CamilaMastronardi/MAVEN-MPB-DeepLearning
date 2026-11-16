#!/usr/bin/env python3
"""Figure for any full 4-fold MAG training, using saved CV metrics only.

Reads cv_summary.json, fold_XX_metrics.json and fold_XX_event_metrics.csv.
Uses saved metrics only: NEVER queries BigQuery, trains, or changes checkpoints.
By default analyzes the THIRD MAG experiment with physical |B| regularization.
Produces a fold summary, training/inner-monitor loss curves and error diagnostics.
"""
from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path

import numpy as np
import matplotlib

matplotlib.use("Agg")  # Render safely from WSL/tmux without a graphical display.
import matplotlib.pyplot as plt


DEFAULT_RESULTS = Path("results/metrics/mag_cv_600windows_det10_mag1_pat20_ep1000")


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



def read_fold_histories(results_dir: Path):
    """Read only actual saved training histories; outer VAL was not used per epoch."""
    histories = {}
    for fold in range(1, 5):
        path = results_dir / f"fold_{fold:02d}_metrics.json"
        if not path.is_file():
            raise FileNotFoundError(f"Faltan métricas del fold {fold}: {path}")
        record = json.loads(path.read_text(encoding="utf-8"))
        if int(record.get("fold", -1)) != fold:
            raise ValueError(f"Fold ID mismatch in {path}.")
        history = record.get("train_history", [])
        if not history:
            raise ValueError(f"No saved training history for fold {fold}.")
        histories[fold] = (history, int(record["best_epoch"]))
    return histories


def plot_training_histories(histories, output_path: Path):
    """Three losses x four folds: gradient TRAIN vs inner-month monitor.

    Raw L_det and L_mag are plotted, WITHOUT multiplying by their weights.
    The total gradient objective is L_rec + lambda_det*L_det + lambda_mag*L_mag.
    """
    metrics = [
        ("train_rec", "val_rec", "Reconstrucción (MSE)"),
        ("train_det", "val_det", "Detección MPB (MSE ponderado)"),
        ("train_mag", "val_mag", "Módulo magnético (MSE físico escalado)"),
    ]
    fig, axes = plt.subplots(3, 4, figsize=(16.5, 9.5), sharex="col",
                             layout="constrained")
    for col, fold in enumerate(range(1, 5)):
        history, best_epoch = histories[fold]
        epochs = np.asarray([int(h["epoch"]) for h in history])
        for row, (train_key, monitor_key, metric_label) in enumerate(metrics):
            ax = axes[row, col]
            has_metric = (
                all(train_key in h and monitor_key in h for h in history)
            )
            if has_metric:
                train = np.asarray([float(h[train_key]) for h in history])
                inner = np.asarray([float(h[monitor_key]) for h in history])
                ax.plot(epochs, train, label="TRAIN", lw=1.3)
                ax.plot(epochs, inner, label="Monitor interno", lw=1.4)
                ax.axvline(best_epoch, ls="--", lw=1.1, color="gray",
                           label="Mejor época")
                ax.grid(alpha=0.25)
            else:
                ax.text(0.5, 0.5, "Métrica no registrada",
                        ha="center", va="center", transform=ax.transAxes)
            if row == 0:
                ax.set_title(f"Fold {fold} — mejor época: {best_epoch}")
            if col == 0:
                ax.set_ylabel(metric_label)
            if row == 2:
                ax.set_xlabel("Época")
    handles, labels = axes[0, 0].get_legend_handles_labels()
    if handles:
        fig.legend(handles, labels, loc="lower center", ncol=3,
                   bbox_to_anchor=(0.5, -0.01), frameon=False)
    fig.suptitle("MAG — evolución de las pérdidas por fold",
                 fontsize=15, fontweight="bold")
    fig.text(
        0.5, 0.01,
        "Monitor interno: meses separados de TRAIN para early stopping. "
        "No es la validación externa de CV. Pérdidas sin multiplicar por λ.",
        ha="center", fontsize=9
    )
    output_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(output_path, dpi=200, bbox_inches="tight")
    plt.close(fig)
    return output_path


def read_event_errors(results_dir: Path):
    """Exact error list across held-out folds (not a mean of fold medians)."""
    errors = {}
    for fold in range(1, 5):
        path = results_dir / f"fold_{fold:02d}_event_metrics.csv"
        if not path.is_file():
            raise FileNotFoundError(f"Faltan errores individuales: {path}")
        values = []
        with path.open(newline="", encoding="utf-8") as handle:
            for item in csv.DictReader(handle):
                if item.get("absolute_error_s") in ("", None):
                    continue
                error = float(item["absolute_error_s"])
                if not np.isfinite(error) or error < 0:
                    raise ValueError(f"Invalid error in {path}.")
                values.append(error)
        if not values:
            raise ValueError(f"Fold {fold} has no evaluable event errors.")
        errors[fold] = np.asarray(values, dtype=np.float64)
    return errors


def plot_event_error_distribution(errors, output_path: Path):
    """Histogram and cumulative localization accuracy (full held-out events)."""
    total = np.concatenate([errors[fold] for fold in range(1, 5)])
    median_s = float(np.median(total))
    accurate = int(np.count_nonzero(total <= 60.))
    right = float(max(1800., np.ceil(total.max() / 60.) * 60.))
    bins = np.arange(0., right + 60.01, 60.)

    fig, axes = plt.subplots(1, 2, figsize=(12.7, 4.9),
                             layout="constrained")
    axes[0].hist(total / 60., bins=bins / 60., edgecolor="white",
                 linewidth=0.4)
    axes[0].axvline(median_s / 60., linestyle="--", color="black",
                    label=f"Mediana global: {median_s/60.:.1f} min")
    axes[0].axvline(1., linestyle=":", color="gray",
                    label="Umbral: 1 min")
    axes[0].set_xlabel("Error absoluto de localización (min)")
    axes[0].set_ylabel("Cruces MPB")
    axes[0].set_title("(a) Distribución de errores")
    axes[0].legend(fontsize=8)
    axes[0].grid(axis="y", alpha=0.2)

    thresholds = np.linspace(0., right, 600)
    for fold in range(1, 5):
        values = errors[fold]
        cdf = np.searchsorted(np.sort(values), thresholds, side="right") / len(values)
        axes[1].plot(thresholds / 60., 100 * cdf, lw=1.2,
                     label=f"Fold {fold}")
    global_cdf = np.searchsorted(np.sort(total), thresholds, side="right") / len(total)
    axes[1].plot(thresholds / 60., 100 * global_cdf,
                 linewidth=2.4, color="black", label="Todos los folds")
    axes[1].set_xlabel("Tolerancia de error (min)")
    axes[1].set_ylabel("Cruces localizados dentro del umbral (%)")
    axes[1].set_ylim(0, 101)
    axes[1].set_xlim(0, right / 60.)
    axes[1].set_title("(b) Fracción acumulada")
    axes[1].legend(fontsize=8)
    axes[1].grid(alpha=0.2)
    fig.suptitle(
        f"MAG — distribución de error en VAL externa: "
        f"{accurate}/{len(total)} ({100*accurate/len(total):.1f}%) ≤ 60 s",
        fontsize=13, fontweight="bold"
    )
    output_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(output_path, dpi=200, bbox_inches="tight")
    plt.close(fig)
    return output_path, median_s, accurate, len(total)



def plot_fold_comparison(current, previous, path: Path, *,
                         current_label="Tercer entrenamiento",
                         previous_label=previous_label):
    """Same-fold, same-sample-resolution CV comparison of MAG experiments."""
    if [int(r["fold"]) for r in current] != [int(r["fold"]) for r in previous]:
        raise ValueError("Comparison requires identical fold ordering.")
    for a, b in zip(current, previous):
        if int(a["val_evaluable_events"]) != int(b["val_evaluable_events"]):
            raise ValueError("Comparison requires equal evaluated event counts.")

    folds = [int(row["fold"]) for row in current]
    x = np.arange(len(folds), dtype=float)
    fig, axes = plt.subplots(1, 2, figsize=(12.5, 5.),
                             layout="constrained")
    metrics = [
        ("median_abs_peak_error_s", 1 / 60., "Error mediano (min)"),
        ("fraction_within_60s", 100., "Cruces con error ≤60 s (%)"),
    ]
    for ax, (key, scale, label) in zip(axes, metrics):
        old = [float(row[key]) * scale for row in previous]
        new = [float(row[key]) * scale for row in current]
        bars_old = ax.bar(x - 0.19, old, width=0.38,
                          label="Segundo entrenamiento")
        bars_new = ax.bar(x + 0.19, new, width=0.38,
                          label=current_label)
        ax.set_xticks(x, [f"Fold {fold}" for fold in folds])
        ax.set_ylabel(label)
        ax.set_ylim(0, max(old + new) * 1.20)
        ax.grid(axis="y", alpha=0.22)
        ax.set_axisbelow(True)
        for bar, value in zip(bars_old, old):
            ax.annotate(f"{value:.1f}", (bar.get_x() + bar.get_width()/2,
                                        bar.get_height()),
                        ha="center", va="bottom", fontsize=8,
                        xytext=(0, 4), textcoords="offset points")
        for bar, value in zip(bars_new, new):
            ax.annotate(f"{value:.1f}", (bar.get_x() + bar.get_width()/2,
                                        bar.get_height()),
                        ha="center", va="bottom", fontsize=8,
                        xytext=(0, 4), textcoords="offset points")
    axes[0].legend(loc="upper right", fontsize=8)
    fig.suptitle(f"MAG — {previous_label} vs {current_label}",
                 fontsize=14, fontweight="bold")
    fig.supxlabel(
        "Comparación con igual resolución de centros y mismos folds. "
        "Las diferencias pueden deberse a múltiples cambios de entrenamiento.",
        fontsize=9
    )
    path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(path, dpi=200, bbox_inches="tight")
    plt.close(fig)
    return path



def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--summary", type=Path, default=DEFAULT_RESULTS / "cv_summary.json",
                        help="Ruta a cv_summary.json del entrenamiento MAG completo.")
    parser.add_argument("--output", type=Path, default=None,
                        help="Ruta PNG de salida para la figura resumen.")
    parser.add_argument("--summary-only", action="store_true",
                        help="Generar solo figura resumen, sin diagnósticos extra.")
    parser.add_argument("--compare-to", type=Path, default=None,
                        help="cv_summary.json anterior para comparación por fold.")
    parser.add_argument("--no-comparison", action="store_true",
                        help="Omitir la comparación automática con el segundo entrenamiento.")
    args = parser.parse_args()

    rows = load_summary(args.summary)
    summary = json.loads(args.summary.read_text(encoding="utf-8"))
    config = summary.get("training_configuration", {})
    if config.get("architecture") == "transition":
        name = "cuarto entrenamiento: detector de transiciones"
        filename = "mag_cv_transition_training.png"
    elif float(config.get("lambda_magnitude", 0.)) > 0.:
        name = "tercer entrenamiento: regularización del módulo"
        filename = "mag_cv_third_training.png"
    elif (float(config.get("lambda_detection", 1.)) == 10.
          and int(config.get("train_targets_per_event", 24)) == 600):
        name = "segundo entrenamiento"
        filename = "mag_cv_second_training.png"
    else:
        name = "primer entrenamiento"
        filename = "mag_cv_first_training.png"
    if args.output is None:
        args.output = args.summary.parent / "figures" / filename
    saved = make_figure(rows, args.output, experiment_label=name)
    print(f"Resumen de folds: {saved.resolve()}")
    if not args.summary_only:
        figures_dir = args.summary.parent / "figures"
        histories = read_fold_histories(args.summary.parent)
        print("\nTERCER ENTRENAMIENTO MAG — métricas por fold"
              if args.summary.parent == DEFAULT_RESULTS
              else "\nEntrenamiento MAG — métricas por fold")
        print("Fold | Eventos | Mejor época | Error mediano (min) | Error ≤60 s (%)")
        for row in rows:
            fold = int(row["fold"])
            _, best_epoch = histories[fold]
            print(
                f"{fold:4d} | {int(row['val_evaluable_events']):7d} | "
                f"{best_epoch:11d} | "
                f"{float(row['median_abs_peak_error_s'])/60.:18.2f} | "
                f"{float(row['fraction_within_60s'])*100:14.2f}"
            )
        losses = plot_training_histories(
            histories, figures_dir / "mag_cv_training_losses.png"
        )
        errors = read_event_errors(args.summary.parent)
        expected_events = sum(int(row["val_evaluable_events"]) for row in rows)
        if sum(len(v) for v in errors.values()) != expected_events:
            raise ValueError(
                "Inconsistent CV summary and event CSV counts; refusing plots."
            )
        distribution, median_s, n_accurate, n_total = plot_event_error_distribution(
            errors, figures_dir / "mag_cv_error_distribution.png"
        )
        print(f"Curvas de pérdidas: {losses.resolve()}")
        print(f"Distribución de errores: {distribution.resolve()}")
        print(f"Mediana global EXACTA: {median_s:.2f} s "
              f"({median_s/60.:.2f} min).")
        print(f"Error <= 60 s: {n_accurate}/{n_total} "
              f"({n_accurate/n_total:.2%}).")

        prior = args.compare_to
        if prior is None and not args.no_comparison and args.summary.parent == DEFAULT_RESULTS:
            prior = Path("results/metrics/mag_cv_600windows_det10/cv_summary.json")
        if prior is not None:
            if not prior.is_file():
                if args.compare_to is not None:
                    raise FileNotFoundError(f"Comparison summary not found: {prior}")
                print(f"Comparación omitida: no existe {prior}.")
            else:
                earlier = load_summary(prior)
                earlier_config = json.loads(
                    prior.read_text(encoding="utf-8")
                ).get("training_configuration", {})
                centers_current = int(config.get("val_centers_per_event", 121))
                centers_previous = int(
                    earlier_config.get("val_centers_per_event", 121)
                )
                if centers_current != centers_previous:
                    raise ValueError(
                        "Comparison requires identical validation-center counts; "
                        f"found {centers_current} vs {centers_previous}."
                    )
                previous_name = ("Segundo entrenamiento"
                                 if prior.parent.name == "mag_cv_600windows_det10"
                                 else prior.parent.name)
                compared = plot_fold_comparison(
                    rows, earlier, figures_dir / "mag_cv_second_vs_third.png",
                    current_label=name,
                    previous_label=previous_name,
                )
                print(f"Comparación 2º vs 3º: {compared.resolve()}")


if __name__ == "__main__":
    main()
