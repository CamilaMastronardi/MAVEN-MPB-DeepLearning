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
                         previous_label="Segundo entrenamiento",
                         dense_output=False):
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
                          label=previous_label)
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
        (
            "Mismas MPB y 600 ventanas por evento. "
            "V6 fusiona puntajes en tiempos MAG nativos; v5 usa centros."
        ) if dense_output else (
            "Mismos folds y resolución de centros. "
            "Las diferencias pueden incluir cambios de entrenamiento."
        ),
        fontsize=9
    )
    path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(path, dpi=200, bbox_inches="tight")
    plt.close(fig)
    return path




def load_localization_events(results_dir: Path):
    """Read signed and absolute peak errors, keyed by exact fold/MPB timestamp."""
    records = {}
    for fold in range(1, 5):
        path = results_dir / f"fold_{fold:02d}_event_metrics.csv"
        with path.open(newline="", encoding="utf-8") as handle:
            reader = csv.DictReader(handle)
            required = {
                "mpb_timestamp_utc", "source_group", "n_windows",
                "peak_offset_s", "absolute_error_s"
            }
            if not required.issubset(set(reader.fieldnames or ())):
                raise ValueError(f"Missing event columns in {path}.")
            for row in reader:
                key = (fold, row["mpb_timestamp_utc"])
                if key in records:
                    raise ValueError(f"Duplicate validation event: {key}.")
                n_windows = int(row["n_windows"])
                if n_windows < 1:
                    raise ValueError(
                        f"Unevaluable event in comparison: {key}; "
                        "use a matched full validation catalog."
                    )
                absolute = float(row["absolute_error_s"])
                signed = float(row["peak_offset_s"])
                if not (np.isfinite(absolute) and np.isfinite(signed)
                        and absolute >= 0 and
                        abs(absolute - abs(signed)) < 0.002):
                    raise ValueError(f"Invalid or inconsistent peak error: {key}.")
                records[key] = {
                    "fold": fold,
                    "mpb_timestamp_utc": key[1],
                    "source_group": row["source_group"],
                    "absolute_error_s": absolute,
                    "peak_offset_s": signed,
                    "n_windows": n_windows,
                }
    return records


def paired_localization_statistics(previous, current):
    """Match identical held-out MPBs, never compare independent populations."""
    if set(previous) != set(current):
        gone = set(previous) - set(current)
        added = set(current) - set(previous)
        raise ValueError(
            f"Unmatched validation MPBs: {len(gone)} only previous, "
            f"{len(added)} only current."
        )
    if not previous:
        raise ValueError("Empty CV event comparison.")
    keys = sorted(previous)
    for key in keys:
        if previous[key]["source_group"] != current[key]["source_group"]:
            raise ValueError(f"Source group changed for event {key}.")
        if previous[key]["n_windows"] != current[key]["n_windows"]:
            raise ValueError(f"Different evaluation-center count for event {key}.")

    before = np.array(
        [previous[key]["absolute_error_s"] for key in keys], dtype=float
    )
    after = np.array(
        [current[key]["absolute_error_s"] for key in keys], dtype=float
    )
    offset_before = np.array(
        [previous[key]["peak_offset_s"] for key in keys], dtype=float
    )
    offset_after = np.array(
        [current[key]["peak_offset_s"] for key in keys], dtype=float
    )
    delta = after - before  # Negative = improved localization.
    improved = int(np.count_nonzero(delta < -0.002))
    worsened = int(np.count_nonzero(delta > 0.002))
    unchanged = len(delta) - improved - worsened
    previous_only = int(np.count_nonzero((before <= 60) & (after > 60)))
    current_only = int(np.count_nonzero((before > 60) & (after <= 60)))
    both_accurate = int(np.count_nonzero((before <= 60) & (after <= 60)))
    results = {
        "n_paired_events": len(keys),
        "previous_median_error_s": float(np.median(before)),
        "current_median_error_s": float(np.median(after)),
        "median_paired_error_change_s": float(np.median(delta)),
        "mean_paired_error_change_s": float(np.mean(delta)),
        "improved_events": improved,
        "worsened_events": worsened,
        "unchanged_events": unchanged,
        "previous_within_60s": int(np.count_nonzero(before <= 60)),
        "current_within_60s": int(np.count_nonzero(after <= 60)),
        "previous_only_within_60s": previous_only,
        "current_only_within_60s": current_only,
        "both_within_60s": both_accurate,
        "previous_peaks_near_interval_edges_25to30min": int(
            np.count_nonzero(np.abs(offset_before) >= 1500)
        ),
        "current_peaks_near_interval_edges_25to30min": int(
            np.count_nonzero(np.abs(offset_after) >= 1500)
        ),
    }
    for limit in (120, 300, 600):
        results[f"previous_within_{limit}s"] = int(np.count_nonzero(before <= limit))
        results[f"current_within_{limit}s"] = int(np.count_nonzero(after <= limit))
    paired_rows = [
        {
            "fold": key[0],
            "mpb_timestamp_utc": key[1],
            "source_group": previous[key]["source_group"],
            "previous_abs_error_s": float(before[i]),
            "current_abs_error_s": float(after[i]),
            "change_current_minus_previous_s": float(delta[i]),
            "previous_peak_offset_s": float(offset_before[i]),
            "current_peak_offset_s": float(offset_after[i]),
        }
        for i, key in enumerate(keys)
    ]
    return results, paired_rows


def plot_paired_localization(paired_rows, stats, out_path: Path):
    """Paired improvement, accuracy and edge-selection diagnostics."""
    prior = np.array([x["previous_abs_error_s"] for x in paired_rows]) / 60
    newer = np.array([x["current_abs_error_s"] for x in paired_rows]) / 60
    changes = newer - prior
    prev_offset = np.array(
        [x["previous_peak_offset_s"] for x in paired_rows]
    ) / 60
    new_offset = np.array(
        [x["current_peak_offset_s"] for x in paired_rows]
    ) / 60
    fig, axs = plt.subplots(2, 2, figsize=(12.4, 10.1),
                            layout="constrained")
    max_error = max(30., float(np.max(np.concatenate([prior, newer])))) + .5
    axs[0, 0].scatter(prior, newer, s=9, alpha=.5)
    axs[0, 0].plot([0, max_error], [0, max_error],
                   color="black", linestyle="--", lw=1.1)
    axs[0, 0].set_xlim(0, max_error)
    axs[0, 0].set_ylim(0, max_error)
    axs[0, 0].set_xlabel("Error absoluto anterior (min)")
    axs[0, 0].set_ylabel("Error absoluto actual (min)")
    axs[0, 0].set_title("(a) Cada MPB comparada consigo misma")
    axs[0, 0].text(.02, .98, "Bajo diagonal = mejora",
                   va="top", transform=axs[0, 0].transAxes, fontsize=9)

    bins = np.linspace(-max_error, max_error, 51)
    axs[0, 1].hist(changes, bins=bins, edgecolor="white")
    axs[0, 1].axvline(0, linestyle="--", color="black")
    axs[0, 1].set_xlabel("Cambio del error (actual - anterior, min)")
    axs[0, 1].set_ylabel("Cantidad de MPB")
    axs[0, 1].set_title(
        "(b) Mejora/empeoramiento: "
        f"{stats['improved_events']}/{stats['worsened_events']}"
    )

    thresholds = np.arange(0., max_error + .1, .25)
    for values, label in ((prior, "Anterior"), (newer, "Actual")):
        cum = np.searchsorted(np.sort(values), thresholds, side="right")
        axs[1, 0].plot(
            thresholds, 100 * cum / len(values), linewidth=1.8, label=label
        )
    axs[1, 0].set_xlim(0, max_error)
    axs[1, 0].set_ylim(0, 101)
    axs[1, 0].set_xlabel("Tolerancia de localización (min)")
    axs[1, 0].set_ylabel("MPB dentro del umbral (%)")
    axs[1, 0].set_title("(c) Fracción acumulada, mismos eventos")
    axs[1, 0].legend()

    offset_bins = np.arange(-30., 30.01, 2.)
    axs[1, 1].hist(
        prev_offset, bins=offset_bins, histtype="step", linewidth=1.7,
        label="Anterior"
    )
    axs[1, 1].hist(
        new_offset, bins=offset_bins, histtype="step", linewidth=1.7,
        label="Actual"
    )
    axs[1, 1].axvline(0, color="gray", linestyle="--")
    axs[1, 1].set_xlabel("Máximo predicho respecto a MPB catalogada (min)")
    axs[1, 1].set_ylabel("Cantidad de MPB")
    axs[1, 1].set_title("(d) Sesgo temporal y máximos en extremos")
    axs[1, 1].legend()

    for ax in axs.ravel():
        ax.grid(alpha=.2)
    fig.suptitle(
        f"MAG — comparación emparejada de {stats['n_paired_events']} cruces "
        "MPB en validación externa",
        fontsize=14, fontweight="bold"
    )
    out_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_path, dpi=200, bbox_inches="tight")
    plt.close(fig)
    return out_path


def save_paired_localization(
    paired_rows, stats, figures_dir, *, fifth=False, sixth=False
):
    if fifth and sixth:
        raise ValueError("Paired comparison cannot be both fifth and sixth.")
    basename = (
        "mag_cv_paired_fifth_vs_sixth" if sixth else
        "mag_cv_paired_fourth_vs_fifth" if fifth else
        "mag_cv_paired_comparison"
    )
    output = figures_dir / basename
    image_path = plot_paired_localization(
        paired_rows, stats, output.with_suffix(".png")
    )
    output.with_suffix(".json").write_text(
        json.dumps(stats, indent=2, sort_keys=True), encoding="utf-8"
    )
    with output.with_suffix(".csv").open(
        "w", newline="", encoding="utf-8"
    ) as handle:
        writer = csv.DictWriter(handle, fieldnames=list(paired_rows[0]))
        writer.writeheader()
        writer.writerows(paired_rows)
    return image_path, output.with_suffix(".json")



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
    if config.get("architecture") == "dense-temporal":
        name = "sexto entrenamiento: detector temporal denso"
        filename = "mag_cv_dense_temporal_training.png"
    elif config.get("architecture") == "residual-transition":
        name = "quinto entrenamiento: encoder residual y detector de transiciones"
        filename = "mag_cv_residual_transition_training.png"
    elif config.get("architecture") == "transition":
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
                previous_name = (
                    "Segundo entrenamiento"
                    if prior.parent.name == "mag_cv_600windows_det10"
                    else "Cuarto entrenamiento"
                    if prior.parent.name == "mag_cv_transition_v1"
                    else "Quinto entrenamiento"
                    if prior.parent.name == "mag_cv_residual_transition_v1"
                    else prior.parent.name
                )
                comparison_filename = (
                    "mag_cv_fifth_vs_sixth.png"
                    if config.get("architecture") == "dense-temporal"
                    else "mag_cv_fourth_vs_fifth.png"
                    if config.get("architecture") == "residual-transition"
                    else "mag_cv_second_vs_third.png"
                )
                compared = plot_fold_comparison(
                    rows, earlier, figures_dir / comparison_filename,
                    current_label=name,
                    previous_label=previous_name,
                    dense_output=(config.get("architecture") == "dense-temporal"),
                )
                print(f"Comparación {previous_name} vs {name}: "
                      f"{compared.resolve()}")

                # Paired analysis: identical MPBs evaluated by both models.
                earlier_events = load_localization_events(prior.parent)
                current_events = load_localization_events(args.summary.parent)
                stats, pairs = paired_localization_statistics(
                    earlier_events, current_events
                )
                paired_figure, paired_json = save_paired_localization(
                    pairs, stats, figures_dir,
                    fifth=(config.get("architecture") == "residual-transition"),
                    sixth=(config.get("architecture") == "dense-temporal"),
                )
                print(f"Comparación emparejada: {paired_figure.resolve()}")
                print(f"Métricas emparejadas: {paired_json.resolve()}")
                print(f"MPB mejoradas/empeoradas/iguales: "
                      f"{stats['improved_events']}/"
                      f"{stats['worsened_events']}/"
                      f"{stats['unchanged_events']}")
                print(f"MPB ≤60 s anterior/actual: "
                      f"{stats['previous_within_60s']}/"
                      f"{stats['current_within_60s']}")
                print(f"Ganadas/perdidas en ≤60 s: "
                      f"{stats['current_only_within_60s']}/"
                      f"{stats['previous_only_within_60s']}")
                print(f"Máximos a ≥25 min de la etiqueta, anterior/actual: "
                      f"{stats['previous_peaks_near_interval_edges_25to30min']}/"
                      f"{stats['current_peaks_near_interval_edges_25to30min']}")


if __name__ == "__main__":
    main()
