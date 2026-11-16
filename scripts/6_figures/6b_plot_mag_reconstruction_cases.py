#!/usr/bin/env python3
"""Plot two REAL MAVEN MAG 70-minute events: well and poorly localized MPB.

A trained MAG model accepts 10-minute windows, NOT 70-minute inputs. This
script uses the SAME number of spaced 10-minute windows as the saved outer-CV
event evaluation (600 for the second, third and fourth trainings; 121 for
the first). Each window has (1) reconstructed Bx/By/Bz/|B| and (2) a
supervised MPB proximity score. Overlapping decoder outputs are averaged
at original measurement timestamps to visualize the full 70-minute block.

Selects representative good (error <=60 s) and bad (error >=600 s) examples
from saved per-event CV metrics, without retraining. Reads scientific MAG
measurements only for the TWO selected events; BigQuery dry-run by default.
A scalar score in [0,1] is NOT a calibrated MPB probability.
"""
from __future__ import annotations

import argparse
import csv
import json
import sys
from datetime import timedelta
from importlib import import_module
from pathlib import Path

import matplotlib
matplotlib.use("Agg")  # Safe on WSL without a GUI.
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D
from matplotlib.patches import Patch
import numpy as np
import torch
from google.cloud import bigquery

ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

bq = import_module("scripts.1_bigquery.1c_bigquery_event_pilot")
pilot = import_module("scripts.3_windowing.3b_bigquery_window_pilot")
windowing = import_module("scripts.3_windowing.3a_windowing")
batching = import_module("scripts.4_preprocessing.4a_native_batches")
trainer = import_module("scripts.5_training.5a_train_mag_cv")
from src.detection.mag_autoencoder import MagSupervisedAutoencoder
from src.detection.mag_transition_autoencoder import MagTransitionAutoencoder

DEFAULT_RESULTS = Path("results/metrics/mag_cv_600windows_det10_mag1_pat20_ep1000")
DEFAULT_NORMALIZERS = Path("results/metrics/fold_normalization")
DEFAULT_MANIFEST = Path("data/splits/cv4_monthly_manifest.csv")
GIB = 1024**3
HALF_WINDOW_S = 300.0


def args_parser():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--results-dir", type=Path, default=DEFAULT_RESULTS)
    p.add_argument("--normalizers-dir", type=Path, default=DEFAULT_NORMALIZERS)
    p.add_argument("--manifest", type=Path, default=DEFAULT_MANIFEST)
    p.add_argument("--output", type=Path, default=None)
    p.add_argument("--fold", choices=("all", "1", "2", "3", "4"), default="1",
                   help="Choose success/failure from SAME held-out model; all allows different folds.")
    p.add_argument("--good-max-seconds", type=float, default=60.0)
    p.add_argument("--bad-min-seconds", type=float, default=600.0)
    p.add_argument("--batch-size", type=int, default=16,
                   help="Must match batch size used for original outer-CV predictions.")
    p.add_argument("--val-centers-per-event", type=int, default=None,
                   help="Number of validation centers; defaults to value saved in CV summary.")
    p.add_argument("--sigma-seconds", type=float, default=None,
                   help="Gaussian sigma in seconds; defaults to value saved in CV summary.")
    p.add_argument("--max-query-gib", type=float, default=1.)
    p.add_argument("--max-total-gib", type=float, default=2.)
    p.add_argument("--max-returned-rows", type=int, default=25000)
    p.add_argument("--torch-threads", type=int, default=2)
    p.add_argument("--execute", action="store_true",
                   help="Actually fetch the two 70-minute MAG series and save figure.")
    a = p.parse_args()
    if (a.good_max_seconds < 0 or a.bad_min_seconds <= a.good_max_seconds
        or min(a.batch_size, a.max_query_gib,
               a.max_total_gib, a.max_returned_rows, a.torch_threads) <= 0
        or (a.val_centers_per_event is not None and a.val_centers_per_event <= 0)
        or (a.sigma_seconds is not None and a.sigma_seconds <= 0)):
        p.error("Invalid thresholds, batch size, or BigQuery query limits.")
    if a.output is None:
        a.output = a.results_dir / "figures/mag_cv_reconstruction_good_vs_bad.png"
    return a


def load_evaluation_config(results_dir: Path):
    """Recover exact held-out scoring settings from the trained experiment.

    First run has no training_configuration and used 121 centers, sigma=60 s.
    The new run records 600 centers and sigma=60 s.
    """
    path = results_dir / "cv_summary.json"
    summary = json.loads(path.read_text(encoding="utf-8"))
    if (summary.get("instrument") != "MAG"
            or summary.get("pilot_subset") is not False
            or sorted(summary.get("completed_folds", [])) != [1, 2, 3, 4]):
        raise ValueError("Require completed, full four-fold MAG CV results.")
    config = summary.get("training_configuration", {})
    centers = int(config.get("val_centers_per_event", 121))
    sigma = float(config.get("sigma_seconds", 60.))
    if centers <= 0 or sigma <= 0:
        raise ValueError("Invalid window evaluation settings in saved summary.")
    return centers, sigma


def read_event_scores(results_dir, fold_filter="all"):
    """Only CSV metrics of the actual full four-fold experiment."""
    summary_file = results_dir / "cv_summary.json"
    if not summary_file.is_file():
        raise FileNotFoundError(f"No CV summary at {summary_file}")
    summary = json.loads(summary_file.read_text(encoding="utf-8"))
    if (summary.get("instrument") != "MAG"
            or summary.get("pilot_subset") is not False
            or sorted(summary.get("completed_folds", [])) != [1, 2, 3, 4]):
        raise ValueError("Require completed, full four-fold MAG CV results.")

    folds = range(1, 5) if fold_filter == "all" else [int(fold_filter)]
    records = []
    for fold in folds:
        path = results_dir / f"fold_{fold:02d}_event_metrics.csv"
        with path.open(newline="", encoding="utf-8") as handle:
            for record in csv.DictReader(handle):
                if not record["absolute_error_s"] or int(record["n_windows"]) < 1:
                    continue
                err = float(record["absolute_error_s"])
                if not np.isfinite(err) or err < 0:
                    continue
                records.append({
                    "fold": fold,
                    "time": bq.to_utc(record["mpb_timestamp_utc"]),
                    "error_s": err,
                    "saved_offset_s": float(record["peak_offset_s"]),
                    "source_group": record["source_group"],
                })
    if not records:
        raise ValueError("No evaluable events in saved event metrics.")
    return records


def select_cases(records, good_threshold, bad_threshold):
    """Choose near-median cases within good/bad groups, not extreme outliers."""
    good = [r for r in records if r["error_s"] <= good_threshold]
    bad = [r for r in records if r["error_s"] >= bad_threshold]
    if not good or not bad:
        raise ValueError(
            f"Need a saved MPB event <= {good_threshold:g} s and another >= "
            f"{bad_threshold:g} s. Try --fold all or change thresholds."
        )

    def representative(rows):
        median = float(np.median([r["error_s"] for r in rows]))
        return min(rows, key=lambda r: (
            abs(r["error_s"] - median), r["fold"], r["time"]
        ))

    return representative(good), representative(bad)


def load_model_and_normalizer(fold, args, manifest_rows):
    norm_path = (args.normalizers_dir
                 / f"fold_{fold:02d}_mag_normalizer.json")
    normalizer, _ = trainer.fold_normalizer(
        norm_path, args.manifest, manifest_rows, fold
    )
    model_path = args.results_dir / f"fold_{fold:02d}_final.pt"
    if not model_path.is_file():
        raise FileNotFoundError(f"No final fold checkpoint: {model_path}")
    # This is a trusted checkpoint from the user's own run, not an untrusted
    # downloaded .pt. weights_only avoids arbitrary pickle objects.
    state = torch.load(model_path, map_location="cpu", weights_only=True)
    if state.get("fold") != fold:
        raise ValueError("Checkpoint fold disagrees with the annotated event.")
    config = state.get("config", {})
    architecture = config.get("architecture", "baseline")
    shared = {
        "hidden": int(config.get("hidden", 16)),
        "latent_channels": int(config.get("latent_channels", 2)),
        "pooling_sigma_s": float(config.get("pooling_sigma_s", 90.0)),
    }
    if architecture == "baseline":
        model = MagSupervisedAutoencoder(**shared)
    elif architecture == "transition":
        model = MagTransitionAutoencoder(
            **shared,
            side_offset_s=float(config["transition_side_offset_s"]),
            center_exclusion_s=float(config["transition_center_exclusion_s"]),
        )
    else:
        raise ValueError(f"Unknown saved MAG detector architecture: {architecture}")
    model.load_state_dict(state["model"], strict=True)
    model.eval()
    return model, normalizer, state.get("best_epoch")


def build_validation_windows(times, values, event, manifest_rows, args):
    """Identical native-window generation and spacing to 5a validation."""
    labels = np.asarray([
        r["time"].timestamp() for r in manifest_rows
        if r["fold"] == event["fold"] and r["mag_available"] == "1"
    ], dtype=np.float64)
    if not len(labels):
        raise ValueError("No validation-fold labels.")
    candidates = list(windowing.iter_native_windows(
        times, values,
        center_label_s=event["time"].timestamp(),
        split_labels_s=labels,
        window_seconds=600.,
        center_range_seconds=1800.,
        sigma_seconds=args.sigma_seconds
    ))
    chosen = trainer.select_spaced(candidates, args.val_centers_per_event)
    if not chosen:
        raise ValueError(f"No valid 10-min MAG windows near {event['time']}.")
    return chosen


@torch.inference_mode()
def reconstruct_block(model, normalizer, times, windows, batch_size):
    """Use original validation centers; average decoded overlapping windows.

    The predictions and individual 10-min reconstructions are on native
    measurement timestamps. No interpolation, temporal re-gridding, or
    altered sampling cadence. Missing/gapped regions remain NaN.
    """
    sums = np.zeros((len(times), 4), dtype=np.float64)
    coverage = np.zeros(len(times), dtype=np.int32)
    scores = []
    centers = []
    targets = []

    for begin in range(0, len(windows), batch_size):
        group = windows[begin:begin + batch_size]
        batch = batching.collate_native_windows(group, normalizer=normalizer)
        _, decoded, outputs = model(
            batch.x, batch.time_mask, batch.rel_time_s
        )
        prediction = outputs.squeeze(-1).cpu().numpy()
        reconstructed = decoded.cpu().numpy()

        for j, w in enumerate(group):
            center = float(w.center_s)
            lo = int(np.searchsorted(times, center - HALF_WINDOW_S, side="left"))
            hi = int(np.searchsorted(times, center + HALF_WINDOW_S, side="right"))
            n = hi - lo
            if n != len(w.data):
                raise RuntimeError("Inconsistent raw event/window sample alignment.")
            if not np.allclose(times[lo:hi] - center, w.relative_times_s,
                               atol=1e-4, rtol=0):
                raise RuntimeError("Timestamp alignment mismatch.")

            # Decoder is in standardized coordinates. Restore physical nT.
            raw_nt = (
                reconstructed[j, :, :n].T.astype(np.float64)
                * normalizer.scale + normalizer.mean
            )
            sums[lo:hi] += raw_nt
            coverage[lo:hi] += 1
            centers.append(center)
            scores.append(float(prediction[j]))
            targets.append(float(w.detection_score))

    blended = np.full_like(sums, np.nan)
    covered = coverage > 0
    blended[covered] = sums[covered] / coverage[covered, None]
    return {
        "reconstruction_nt": blended,
        "coverage": coverage,
        "centers_s": np.asarray(centers),
        "scores": np.asarray(scores),
        "target": np.asarray(targets),
    }


def verify_saved_prediction(case, inferred_peak_s, tolerance_s=2.):
    """Avoid labeling a changed inference result as a saved good/bad example."""
    predicted_offset = inferred_peak_s - case["time"].timestamp()
    discrepancy = abs(predicted_offset - case["saved_offset_s"])
    if discrepancy > tolerance_s:
        raise RuntimeError(
            f"Inferred peak differs from stored CV peak by {discrepancy:.1f} s "
            f"for fold {case['fold']}. Check model, batch size and val centers. "
            "Do not plot a misleading example."
        )
    return predicted_offset


def plot_cases(cases, out):
    """Two columns (good/bad); four physical channels and score below."""
    fig, axes = plt.subplots(
        5, 2, figsize=(18, 15.5), sharex="col",
        gridspec_kw={"height_ratios": [1, 1, 1, 1, 1.3]}
    )
    fig.subplots_adjust(left=0.07, right=0.98, top=0.86,
                        bottom=0.105, wspace=0.22, hspace=0.13)
    channels = [r"$B_x$", r"$B_y$", r"$B_z$", r"$|B|$"]
    colors = {
        "real": "#30343B", "recon": "#346FA4",
        "true": "#2A875D", "peak": "#D16D36",
        "target": "#868B95", "score": "#2C5592",
        "vector_norm": "#8D4F9D"
    }

    for column, (kind, event, times, values, output, peak, best_epoch) in enumerate(cases):
        t0 = event["time"].timestamp()
        time_min = (times - t0) / 60.
        score_time_min = (output["centers_s"] - t0) / 60.
        peak_min = (peak - t0) / 60.
        error_s = abs(peak - t0)
        for row in range(5):
            ax = axes[row, column]
            # Highlight the two physical 10-minute windows, not a measured
            # MPB duration or thickness. Both are explicitly center windows.
            ax.axvspan(-5, 5, color=colors["true"], alpha=0.11)
            ax.axvspan(peak_min - 5, peak_min + 5,
                       color=colors["peak"], alpha=0.10)
            ax.axvline(0, color=colors["true"], lw=1.6)
            ax.axvline(peak_min, color=colors["peak"], lw=1.6,
                       linestyle="--")
            ax.set_xlim(-35, 35)
            ax.grid(alpha=0.20)

            if row < 4:
                ax.plot(time_min, values[:, row], color=colors["real"],
                        linewidth=0.7, label="MAG observado", zorder=2)
                ax.plot(time_min, output["reconstruction_nt"][:, row],
                        color=colors["recon"], linewidth=0.85,
                        label="Reconstrucción CNN", zorder=3)
                if row == 3:
                    # This is the magnitude of the *blended* reconstructed
                    # vector field, not an invented fifth measurement.
                    reconstructed_xyz_norm = np.linalg.norm(
                        output["reconstruction_nt"][:, :3], axis=1
                    )
                    ax.plot(
                        time_min, reconstructed_xyz_norm,
                        color=colors["vector_norm"], linestyle="--",
                        linewidth=1.0,
                        label="Módulo de Bxyz reconstruido", zorder=4
                    )
                ax.set_ylabel(channels[row] + " (nT)")
            else:
                ax.plot(score_time_min, output["target"], color=colors["target"],
                        linestyle=":", linewidth=1.25,
                        label="Objetivo gaussiano", zorder=2)
                ax.plot(score_time_min, output["scores"], color=colors["score"],
                        linewidth=1.45, marker=".", markersize=2.5,
                        label="Puntaje del detector", zorder=3)
                ax.set_ylabel("Puntaje MPB")
                ax.set_ylim(-0.04, 1.04)
                ax.set_xlabel("Minutos respecto al cruce MPB etiquetado")

        title = "MPB bien localizada" if kind == "good" else "MPB mal localizada"
        when = event["time"].strftime("%Y-%m-%d %H:%M:%S UTC")
        axes[0, column].set_title(
            f"{title}  |  Fold {event['fold']}  |  error {error_s:.1f} s\n"
            f"{when}  |  mejor época {best_epoch}", fontsize=11
        )

    handles = [
        Line2D([], [], color=colors["real"], lw=1.3, label="MAG medido"),
        Line2D([], [], color=colors["recon"], lw=1.5, label="Reconstrucción (nT)"),
        Line2D([], [], color=colors["vector_norm"], ls="--", lw=1.5,
               label="Módulo de componentes reconstruidas"),
        Patch(facecolor=colors["true"], alpha=0.30,
              label="Ventana ±5 min alrededor de MPB catalogada"),
        Patch(facecolor=colors["peak"], alpha=0.30,
              label="Ventana ±5 min alrededor del máximo predicho"),
        Line2D([], [], color=colors["score"], lw=1.5,
               label="Puntaje MPB no calibrado"),
        Line2D([], [], color=colors["target"], ls=":", lw=1.5,
               label="Objetivo gaussiano de entrenamiento"),
    ]
    fig.suptitle(
        "MAVEN MAG — señal real, reconstrucción y localización de la MPB",
        fontsize=15, fontweight="bold", y=0.99
    )
    fig.legend(handles=handles, loc="upper center",
               bbox_to_anchor=(0.5, 0.955), ncol=4, frameon=False, fontsize=8.5)
    fig.text(
        0.5, 0.025,
        "Cada serie abarca 70 min. Reconstrucción: promedio de ventanas "
        "solapadas de 10 min; las bandas no representan el espesor de la MPB.\n"
        "El puntaje entre 0 y 1 indica cercanía a una MPB etiquetada; "
        "no es una probabilidad calibrada.",
        ha="center", fontsize=9
    )
    out.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out, dpi=200, bbox_inches="tight")
    plt.close(fig)


def main():
    args = args_parser()
    saved_centers, saved_sigma = load_evaluation_config(args.results_dir)
    # A different set of evaluation centers could create a different peak.
    # We must reconstruct the exact scoring used to label each good/bad case.
    if args.val_centers_per_event not in (None, saved_centers):
        raise ValueError(
            f"Saved experiment used {saved_centers} centers per event. "
            "Refusing a non-equivalent evaluation."
        )
    if args.sigma_seconds not in (None, saved_sigma):
        raise ValueError(
            f"Saved experiment used sigma={saved_sigma:g} seconds. "
            "Refusing a non-equivalent label for the plot."
        )
    args.val_centers_per_event = saved_centers
    args.sigma_seconds = saved_sigma
    torch.set_num_threads(args.torch_threads)
    print(f"Using saved CV evaluation: {saved_centers} window centers/event, "
          f"Gaussian sigma={saved_sigma:g} s.", flush=True)
    manifest_rows = pilot.read_manifest(args.manifest)
    scores = read_event_scores(args.results_dir, args.fold)
    good, bad = select_cases(scores, args.good_max_seconds, args.bad_min_seconds)
    selected = [("good", good), ("bad", bad)]
    time_lookup = {(r["fold"], r["time"]): r for r in manifest_rows}
    for kind, case in selected:
        key = (case["fold"], case["time"])
        if key not in time_lookup or time_lookup[key]["mag_available"] != "1":
            raise ValueError("Saved CV event does not match the current manifest.")
        print(f"{kind.upper()}: fold={case['fold']}, "
              f"event={case['time'].isoformat()}, "
              f"stored error={case['error_s']:.1f} s.", flush=True)

    client = bigquery.Client(project=bq.PROJECT)
    bq.validate_schema(client)
    sql = bq.MAG_SQL.format(mag=bq.MAG)
    plans, total = [], 0
    for kind, case in selected:
        start = case["time"] - timedelta(minutes=35)
        end = case["time"] + timedelta(minutes=35)
        dry = client.query(
            sql, job_config=bq.query_config(
                start, end, dry_run=True, max_bytes=1
            )
        )
        estimate = int(dry.total_bytes_processed or 0)
        total += estimate
        if estimate > args.max_query_gib * GIB:
            raise RuntimeError(f"{kind} event query exceeds max-query-gib.")
        plans.append((kind, case, start, end, estimate))
    print(f"BigQuery DRY RUN: total for two 70-min events = "
          f"{total/GIB:.5f} GiB; cap {args.max_total_gib:.3f} GiB.",
          flush=True)
    if total > args.max_total_gib * GIB:
        raise RuntimeError("Query estimates exceed the total cap; no data fetched.")
    if not args.execute:
        print("Dry-run only. Add --execute to fetch two REAL MAG events and plot.")
        return

    results = []
    for kind, case, start, end, _ in plans:
        job = client.query(
            sql, job_config=bq.query_config(
                start, end, dry_run=False,
                max_bytes=int(args.max_query_gib * GIB)
            )
        )
        rows = []
        for row in job.result(page_size=250):
            rows.append(row)
            if len(rows) > args.max_returned_rows:
                raise MemoryError("BigQuery row cap exceeded.")
        times, values, _ = pilot.read_mag(rows)
        del rows
        if len(times) < 3 or np.any(np.diff(times) <= 0):
            raise ValueError("Insufficient/duplicate MAG timestamps.")
        model, normalizer, best_epoch = load_model_and_normalizer(
            case["fold"], args, manifest_rows
        )
        windows = build_validation_windows(
            times, values, case, manifest_rows, args
        )
        inferred = reconstruct_block(
            model, normalizer, times, windows, args.batch_size
        )
        best_index = int(np.argmax(inferred["scores"]))
        peak = float(inferred["centers_s"][best_index])
        offset_s = verify_saved_prediction(case, peak)
        results.append((kind, case, times, values, inferred, peak, best_epoch))
        print(f"{kind.upper()}: evaluated {len(windows)} 10-min windows, "
              f"peak offset {offset_s:.2f} s; "
              f"reconstruction coverage "
              f"{np.mean(inferred['coverage'] > 0):.1%} of epochs.",
              flush=True)

    plot_cases(results, args.output)

    def magnitude_diagnostics(observed, reconstructed):
        """Errors in physical nT from actual covered measurement epochs only."""
        valid = np.isfinite(reconstructed).all(axis=1)
        if not np.any(valid):
            return {
                "rmse_decoded_magnitude_nt": None,
                "rmse_vector_magnitude_nt": None,
                "rmse_magnitude_consistency_nt": None,
            }
        mag_channel = reconstructed[valid, 3]
        vector_magnitude = np.linalg.norm(reconstructed[valid, :3], axis=1)
        real_mag = observed[valid, 3]
        return {
            "rmse_decoded_magnitude_nt":
                float(np.sqrt(np.mean((mag_channel - real_mag) ** 2))),
            "rmse_vector_magnitude_nt":
                float(np.sqrt(np.mean((vector_magnitude - real_mag) ** 2))),
            "rmse_magnitude_consistency_nt":
                float(np.sqrt(np.mean((vector_magnitude - mag_channel) ** 2))),
        }

    metadata = {
        "figure": str(args.output),
        "experiment": str(args.results_dir),
        "val_centers_per_event": args.val_centers_per_event,
        "sigma_seconds": args.sigma_seconds,
        "cases": [{
            "kind": kind, "fold": case["fold"],
            "label_utc": case["time"].isoformat(),
            "saved_abs_error_s": case["error_s"],
            "recomputed_abs_error_s": abs(peak-case["time"].timestamp()),
            "best_epoch": best_epoch,
            "n_scored_windows": len(output["scores"]),
            "n_real_epochs": len(times),
            "magnitude_rmse": magnitude_diagnostics(
                values, output["reconstruction_nt"]
            ),
        } for kind, case, times, values, output, peak, best_epoch in results],
        "notes": [
            "MAG measurements are not saved; only PNG and this metadata.",
            "Reconstruction is average of overlapping 600-s CNN decodings.",
            "The label and prediction bands show centered 600-s windows.",
            "The supervised score is NOT a calibrated MPB probability.",
            "Magnitude metrics are computed on 70-min reconstruction averages.",
            "Vector-derived |B| is computed from the reconstructed xyz components.",
        ],
    }
    sidecar = args.output.with_suffix(".json")
    sidecar.write_text(json.dumps(metadata, indent=2), encoding="utf-8")
    print(f"Figure saved: {args.output.resolve()}", flush=True)
    for item in metadata["cases"]:
        print(
            f"{item['kind'].upper()} fold {item['fold']} "
            f"magnitude RMSE (nT): {item['magnitude_rmse']}",
            flush=True
        )
    print(f"Selection metadata: {sidecar.resolve()}", flush=True)


if __name__ == "__main__":
    main()
