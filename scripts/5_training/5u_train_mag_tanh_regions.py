#!/usr/bin/env python3
"""MAG-only region classifier: 8 Conv1D layers on 10-minute windows.

The input is [Bx, By, Bz, |B|] at the native MAG cadence, in 600-s
windows. The ONLY model output is a 600-s segment of a signed global tanh:
    -1 = magnetosheath proxy; +1 = planetary/induced-region proxy;
     0 = boundary transition (also low-confidence classification).
A SINGLE reference tanh is defined per 70-minute catalog event and sliced
at each window's native timestamps; it is NOT re-centered per window.

At evaluation, overlapping 600-s model outputs are combined into one 70-min
classification series. A SINGLE tanh is then fit to that combined series;
its zero is a conditional MPB time estimate, where fit quality permits.
No measurement-derived plasma composition is claimed from MAG alone.

This script reuses existing integrity-checked local MAG caching, monthly CV,
physical boundary exclusions and global overlap-add evaluation routines.
No BigQuery requests, MAG reconstruction, SWEA or model weight overwrites.

Example:
 python scripts/5_training/5u_train_mag_tanh_regions.py --folds 1 \\
   --pilot-train 48 --pilot-monitor 12 --pilot-val 12 \\
   --epochs 60 --patience 20
"""
from __future__ import annotations

import argparse
import csv
import hashlib
import importlib.util
import json
import random
import sys
from datetime import datetime, timezone
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import torch
from torch import nn

ROOT = Path(__file__).resolve().parents[2]
DATA_HELPER = ROOT / "scripts/5_training/5s_train_mag_tanh_8conv_16ch.py"
GLOBAL_HELPER = ROOT / "scripts/5_training/5t_eval_mag_global_tanh_16ch.py"
ARCHITECTURE = "MAG-region-8conv-16ch-stride16-global-tanh-v1"


def load_module(path: Path, name: str):
    if not path.is_file():
        raise FileNotFoundError(f"Falta el archivo requerido: {path}")
    spec = importlib.util.spec_from_file_location(name, path)
    if spec is None or spec.loader is None:
        raise ImportError(f"No se pudo importar {path}")
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


class MagRegionCNN(nn.Module):
    """8 convolutions and a temporal bottleneck of 16 x ceil(T/16).

    Four stride-2 layers: 600 samples -> 38 latent time positions.
    No global pooling, magnetic decoder, catalog times or BS information.
    Relative time features are window-local coordinates, not MPB labels.
    """
    LATENT_CHANNELS = 16
    ENCODER_CONV_LAYERS = 8
    TEMPORAL_REDUCTION = 16

    def __init__(self):
        super().__init__()
        sizes = (9, 32, 32, 32, 32, 32, 24, 24, 16)
        dilations = (1, 1, 2, 1, 4, 1, 2, 1)
        self.encoder = nn.ModuleList([
            nn.Conv1d(sizes[i], sizes[i + 1], kernel_size=5,
                      stride=2 if i in (1, 3, 5, 7) else 1,
                      padding=2 * dilations[i], dilation=dilations[i])
            for i in range(self.ENCODER_CONV_LAYERS)
        ])
        self.activation = nn.GELU()
        self.decoder = nn.Sequential(
            nn.Conv1d(16 + 5, 32, 5, padding=2), nn.GELU(),
            nn.Conv1d(32, 32, 5, padding=2), nn.GELU(),
            nn.Conv1d(32, 16, 5, padding=2), nn.GELU(),
            nn.Conv1d(16, 1, 1), nn.Tanh()
        )

    @staticmethod
    def time_features(t):
        z = t[:, None, :] / 300.
        return torch.cat((z, torch.sin(torch.pi * z), torch.cos(torch.pi * z),
                          torch.sin(2 * torch.pi * z),
                          torch.cos(2 * torch.pi * z)), dim=1)

    def forward(self, x, mask, t):
        if (x.ndim != 3 or x.shape[1] != 4 or
                mask.shape != (x.shape[0], x.shape[-1]) or
                t.shape != mask.shape):
            raise ValueError("Expected x:Bx4xT, mask:BxT, t:BxT")
        coord = self.time_features(t)
        z = torch.cat((x, coord), dim=1) * mask[:, None, :]
        for i, layer in enumerate(self.encoder):
            z = layer(z)
            if i < len(self.encoder) - 1:
                z = self.activation(z)
        z = nn.functional.interpolate(z, size=x.shape[-1],
                                      mode="linear", align_corners=False)
        z = torch.cat((z, coord), dim=1)
        return self.decoder(z).squeeze(1) * mask


def predict_clip(model, clip, norm):
    """Interface for existing overlap-add evaluator. No MAG reconstruction."""
    t, measured = clip
    x = ((measured - norm["mean"]) / norm["scale"]).T.copy()
    x = torch.from_numpy(np.asarray(x[None], dtype=np.float32))
    tt = torch.from_numpy(np.asarray(t[None], dtype=np.float32))
    mask = torch.ones_like(tt, dtype=torch.bool)
    model.eval()
    with torch.inference_mode():
        y = model(x, mask, tt)[0].cpu().numpy()
    return y, None


def train_one_epoch(model, pools, normalizer, optimizer, args, epoch, *, monitor):
    rng = random.Random(args.seed + 37 * epoch)
    examples = []
    for positives, negatives in pools.values():
        if monitor:
            examples.extend(positives[:args.positive_windows])
            examples.extend(negatives[:args.control_windows])
        else:
            examples.extend(rng.sample(positives,
                         min(len(positives), args.positive_windows)))
            examples.extend(rng.sample(negatives,
                         min(len(negatives), args.control_windows)))
    if not examples:
        raise ValueError("No hay ventanas fisicamente validas")
    if not monitor:
        rng.shuffle(examples)
    model.train(not monitor)
    weighted_total, basic_total, count = 0., 0., 0
    for start in range(0, len(examples), args.batch_size):
        batch = examples[start:start + args.batch_size]
        data = DATA.collate(batch, normalizer)
        with torch.set_grad_enabled(not monitor):
            predicted = model(data["x"], data["mask"], data["t"])
            target = data["y"]
            # Emphasize the neighborhood of the zero WITHOUT making a
            # new target per window. Plateau regions remain supervised.
            weights = 1. + args.transition_weight * (1. - target.square()) * (
                data["positive"][:, None]
            )
            valid = data["mask"]
            mse = ((predicted - target).square() * valid).sum() / valid.sum().clamp_min(1)
            loss = (weights * (predicted - target).square() * valid).sum() / valid.sum().clamp_min(1)
            if not monitor:
                optimizer.zero_grad(set_to_none=True)
                loss.backward()
                nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.)
                optimizer.step()
        weighted_total += float(loss.detach()) * len(batch)
        basic_total += float(mse.detach()) * len(batch)
        count += len(batch)
    return {"weighted_mse": weighted_total / count,
            "mse": basic_total / count, "windows": count}


def write_csv(path: Path, rows):
    if not rows:
        return
    path.parent.mkdir(parents=True, exist_ok=True)
    columns = sorted({key for item in rows for key in item})
    with path.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=columns)
        writer.writeheader()
        writer.writerows(rows)


def physically_justified_mask(event, times, all_mpb, all_bs):
    """Mask ambiguous region labels beyond BS/other catalog boundaries.

    The target is a weak proxy for regional identity, not measured plasma
    composition. Unknown intervals MUST NOT contribute to region MSE.
    """
    mpb, bs = event["time"], event["bs_time"]
    mask = (times > mpb - 1800.) & (times < mpb + 1800.)
    # The sheath side terminates at the associated bow shock. Beyond it
    # the solar wind is not to be relabeled as magnetosheath.
    if bs < mpb:
        mask &= times > bs + 300.
    else:
        mask &= times < bs - 300.
    for other in all_mpb:
        if abs(other - mpb) > .01 and abs(other - mpb) <= 2400.:
            mask &= np.abs(times - other) > 300.
    for other in all_bs:
        if abs(other - bs) > .01 and abs(other - mpb) <= 2400.:
            mask &= np.abs(times - other) > 300.
    return mask


def classify_levels(y, coverage, threshold):
    labels = np.full(len(y), "unknown", dtype="<U20")
    labels[coverage & (y >= threshold)] = "planetary_proxy"
    labels[coverage & (y <= -threshold)] = "magnetosheath"
    labels[coverage & (np.abs(y) < threshold)] = "transition_uncertain"
    return labels


def plot_event(path, event, signal, target, prediction, fit,
               *, threshold=.5, zoom=False):
    time = signal["time_s"]
    fig, axes = plt.subplots(3, 1, figsize=(13, 9), sharex=True)
    axes[0].plot(time, target, color="black", ls="--", lw=1.8,
                 label="Tanh objetivo global (referencia)")
    axes[0].plot(time, prediction, lw=1.2,
                 label="Clasificacion global reconstruida")
    if fit["ok"]:
        axes[0].plot(time, fit["fit_curve"], lw=1.1, ls=":",
                     label="Tanh ajustada a la reconstruccion")
        axes[0].axvline(fit["pred_offset_s"], color="tab:orange", ls=":",
                        label="MPB predicha")
    axes[0].axvline(0., color="grey", ls="--", alpha=.7,
                    label="MPB catalogada (referencia)")
    axes[0].axhline(0., color="grey", lw=.6)
    axes[0].set_ylabel("Clase continua [-1, 1]")
    axes[0].legend(fontsize=8, ncol=2)
    axes[1].plot(time, (prediction + 1.) / 2., label="Score plasma planetario")
    axes[1].plot(time, (1. - prediction) / 2., label="Score magnetovaina")
    axes[1].axhline((1. + threshold) / 2., color="gray", ls=":", lw=.8)
    axes[1].set_ylim(-.05, 1.05)
    axes[1].set_ylabel("Scores (no calibrados)")
    axes[1].legend(fontsize=8)
    axes[2].plot(time, signal["B_nt"][:, 3], color="black", lw=.9,
                 label="|B| observado")
    axes[2].axvline(0., color="gray", ls="--", alpha=.7)
    axes[2].set_ylabel("|B| (nT)")
    axes[2].set_xlabel("Tiempo relativo a la MPB catalogada (s)")
    axes[2].legend(fontsize=8)
    axes[0].set_title(f"8 Conv1D, MAG 600s -> tanh 600s | {event['mpb_timestamp_utc']}")
    if zoom:
        axes[2].set_xlim(-300., 300.)
    fig.tight_layout()
    path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(path, dpi=160)
    plt.close(fig)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--folds", choices=["1", "2", "3", "4", "all"], default="1")
    parser.add_argument("--tau-mode", choices=["adaptive", "fixed"], default="adaptive")
    parser.add_argument("--pilot-train", type=int, default=48)
    parser.add_argument("--pilot-monitor", type=int, default=12)
    parser.add_argument("--pilot-val", type=int, default=12)
    parser.add_argument("--epochs", type=int, default=60)
    parser.add_argument("--min-epochs", type=int, default=10)
    parser.add_argument("--patience", type=int, default=20)
    parser.add_argument("--batch-size", type=int, default=8)
    parser.add_argument("--positive-windows", type=int, default=5)
    parser.add_argument("--control-windows", type=int, default=6)
    parser.add_argument("--transition-weight", type=float, default=3.)
    parser.add_argument("--learning-rate", type=float, default=3.e-4)
    parser.add_argument("--scan-stride", type=int, default=64)
    parser.add_argument("--region-threshold", type=float, default=.5)
    parser.add_argument("--plot-events", type=int, default=6)
    parser.add_argument("--torch-threads", type=int, default=2)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--mag-cache", type=Path,
                        default=ROOT / "data/cache/mag_events")
    parser.add_argument("--manifest", type=Path,
                        default=ROOT / "data/splits/cv4_monthly_manifest.csv")
    parser.add_argument("--output-dir", type=Path, default=None)
    args = parser.parse_args()
    if (min(args.epochs, args.min_epochs, args.patience, args.batch_size,
            args.positive_windows, args.control_windows, args.scan_stride,
            args.torch_threads) < 1 or args.min_epochs > args.epochs or
            min(args.pilot_train, args.pilot_monitor, args.pilot_val,
                args.plot_events) < 0 or args.transition_weight < 0 or
            args.learning_rate <= 0 or not 0. < args.region_threshold < 1.):
        parser.error("Argumentos fuera de rango")

    torch.set_num_threads(args.torch_threads)
    torch.manual_seed(args.seed)
    np.random.seed(args.seed)
    random.seed(args.seed)
    global DATA
    DATA = load_module(DATA_HELPER, "mag_tanh_5s_data")
    global GLOBAL
    GLOBAL = load_module(GLOBAL_HELPER, "mag_tanh_5t_global")

    catalog = DATA.read_catalog(args.manifest)
    caches = DATA.locate_mag_cache(args.mag_cache, catalog)
    if not caches:
        raise FileNotFoundError(f"No hay datos MAG en cache: {args.mag_cache}")
    all_mpb = np.asarray([r["time"] for r in catalog])
    all_bs = np.asarray([r["bs_time"] for r in catalog])
    pilot = any((args.pilot_train, args.pilot_monitor, args.pilot_val))
    output = args.output_dir or (ROOT / "results/metrics" /
                ("mag_tanh_regions_8conv16_pilot" if pilot else "mag_tanh_regions_8conv16"))
    output.mkdir(parents=True, exist_ok=True)
    manifest_hash = hashlib.sha256(args.manifest.read_bytes()).hexdigest()
    folds = (1, 2, 3, 4) if args.folds == "all" else (int(args.folds),)
    results = []

    for fold in folds:
        train = [r for r in catalog if r["fold"] != fold and r["time"] in caches]
        val = [r for r in catalog if r["fold"] == fold and r["time"] in caches]
        fit, monitor = DATA.split_fit_monitor(train, fold, args.seed)
        split = {
            "fit": DATA.select_even(fit, args.pilot_train),
            "monitor": DATA.select_even(monitor, args.pilot_monitor),
            "val": DATA.select_even(val, args.pilot_val),
        }
        if not all(split.values()):
            print(f"Fold {fold}: sin datos suficientes para alguna particion")
            continue
        print(f"Fold {fold}: FIT={len(split['fit'])} MON={len(split['monitor'])} "
              f"VAL={len(split['val'])}", flush=True)
        fold_out = output / f"fold_{fold:02d}"
        fold_out.mkdir(parents=True, exist_ok=True)
        blocks = {r["time"]: DATA.load_block(caches[r["time"]], r["time"])
                  for subset in split.values() for r in subset}
        norm = DATA.norm_fit({r["time"]: blocks[r["time"]] for r in split["fit"]})
        train_pools = DATA.make_pools(split["fit"], blocks, all_mpb, all_bs,
                                      args.tau_mode)
        monitor_pools = DATA.make_pools(split["monitor"], blocks, all_mpb, all_bs,
                                        args.tau_mode)
        if not train_pools or not monitor_pools:
            raise RuntimeError(f"Fold {fold}: sin ventanas supervisadas validas")
        print(f"Ventanas por evento: max {args.positive_windows} positivas y "
              f"{args.control_windows} controles | "
              f"eventos validos {len(train_pools)} FIT, {len(monitor_pools)} MON",
              flush=True)
        torch.manual_seed(args.seed + fold)
        model = MagRegionCNN()
        optimizer = torch.optim.AdamW(
            model.parameters(), lr=args.learning_rate, weight_decay=1e-3)
        history = []
        best, best_epoch, stale = float("inf"), 0, 0
        checkpoint = fold_out / f"fold_{fold:02d}_regions_best.pt"
        for epoch in range(1, args.epochs + 1):
            tr = train_one_epoch(model, train_pools, norm, optimizer, args,
                                 epoch, monitor=False)
            mon = train_one_epoch(model, monitor_pools, norm, None, args,
                                  epoch, monitor=True)
            history.append({"epoch": epoch, "train": tr, "monitor": mon})
            if mon["weighted_mse"] < best - 1.e-4:
                best, best_epoch, stale = mon["weighted_mse"], epoch, 0
                torch.save({
                    "state_dict": model.state_dict(), "normalizer": norm,
                    "architecture": ARCHITECTURE,
                    "fold": fold, "best_epoch": epoch, "tau_mode": args.tau_mode,
                    "manifest_sha256": manifest_hash,
                    "transition_weight": args.transition_weight,
                    "region_threshold": args.region_threshold,
                }, checkpoint)
            else:
                stale += 1
            print(f"Fold {fold} ep {epoch:3d}: "
                  f"train={tr['weighted_mse']:.4f} "
                  f"monitor={mon['weighted_mse']:.4f} "
                  f"best={best_epoch} stale={stale}", flush=True)
            if epoch >= args.min_epochs and stale >= args.patience:
                break
        (fold_out / f"fold_{fold:02d}_history.json").write_text(
            json.dumps(history, indent=2) + "\n", encoding="utf-8")
        ck = torch.load(checkpoint, map_location="cpu", weights_only=False)
        model.load_state_dict(ck["state_dict"])
        model.eval()
        # Reuse the tested 70-min overlap-add and single global tanh fitter.
        interface = type("WindowInterface", (), {
            "extract_window": staticmethod(DATA.extract_window),
            "predict_clip": staticmethod(predict_clip)})
        evaluations = []
        for i, event in enumerate(split["val"]):
            block = blocks[event["time"]]
            signal = GLOBAL.predict_global_curve(
                interface, model, block, norm, event["time"],
                stride_s=args.scan_stride)
            t = signal["time_s"]
            y = signal["class_curve"]
            covered = signal["covered"] & np.isfinite(y)
            tau, reliable = DATA.estimate_adaptive_tau(block, event["time"])
            if args.tau_mode == "fixed":
                tau = 60.
            direction = 1. if event["bs_time"] < event["time"] else -1.
            target = direction * np.tanh(t / tau)
            justified = physically_justified_mask(
                event, t + event["time"], all_mpb, all_bs)
            reference_valid = justified & covered
            pred_labels = classify_levels(y, covered, args.region_threshold)
            measured_label = classify_levels(target, reference_valid,
                                             args.region_threshold)
            evaluated_mask = reference_valid
            reference_mse = (float(np.mean((y[evaluated_mask] -
                                            target[evaluated_mask]) ** 2))
                             if evaluated_mask.any() else None)
            stable_mask = evaluated_mask & (np.abs(target) >= args.region_threshold)
            stable_accuracy = (float(np.mean(pred_labels[stable_mask] ==
                                              measured_label[stable_mask]))
                               if stable_mask.any() else None)
            fit = GLOBAL.fit_one_global_tanh(t, y, signal["covered"])
            record = {
                "event_utc": event["mpb_timestamp_utc"],
                "fold": fold, "target_tau_s": tau,
                "tau_reliable": reliable,
                "samples": len(t), "n_windows": signal["n_windows"],
                "coverage_fraction": signal["coverage_fraction"],
                "reference_covered_samples": int(evaluated_mask.sum()),
                "reference_classification_mse": reference_mse,
                "stable_region_accuracy": stable_accuracy,
                "evaluated": bool(fit["ok"]),
                "reason": "" if fit["ok"] else fit["reason"],
                "pred_offset_s": fit.get("pred_offset_s"),
                "abs_error_s": abs(fit["pred_offset_s"]) if fit["ok"] else None,
                "tau_fit_s": fit.get("fit_tau_s"),
            }
            evaluations.append(record)
            fmt = (f"error={record['abs_error_s']:.1f} s"
                   if fit["ok"] else fit["reason"])
            print(f"Fold {fold} evento {i+1}/{len(split['val'])}: {fmt} "
                  f"| MSE regiones={reference_mse}", flush=True)
            sample_rows = [
                {"relative_time_s": round(float(t[j]), 3),
                 "classification": float(y[j]) if covered[j] else "",
                 "region": str(pred_labels[j]),
                 "planetary_score": float((y[j] + 1) / 2) if covered[j] else "",
                 "magnetosheath_score": float((1 - y[j]) / 2) if covered[j] else "",
                 "reference_target": float(target[j]),
                 "reference_valid": bool(reference_valid[j]),
                 "B_total_nt": float(signal["B_nt"][j, 3])}
                for j in range(len(t))
            ]
            write_csv(fold_out / f"fold_{fold:02d}_event_{i+1:02d}_regions.csv",
                      sample_rows)
            if i < args.plot_events:
                plot_event(fold_out / f"fold_{fold:02d}_event_{i+1:02d}_global.png",
                           event, signal, target, y, fit,
                           threshold=args.region_threshold)
                plot_event(fold_out / f"fold_{fold:02d}_event_{i+1:02d}_zoom.png",
                           event, signal, target, y, fit,
                           threshold=args.region_threshold, zoom=True)
        write_csv(fold_out / f"fold_{fold:02d}_events.csv", evaluations)
        err = [x["abs_error_s"] for x in evaluations if x["evaluated"]]
        region_mse = [x["reference_classification_mse"] for x in evaluations
                      if x["reference_classification_mse"] is not None]
        stable = [x["stable_region_accuracy"] for x in evaluations
                  if x["stable_region_accuracy"] is not None]
        report = {
            "fold": fold, "architecture": ARCHITECTURE,
            "tau_mode": args.tau_mode, "best_epoch": best_epoch,
            "events": len(evaluations), "evaluable": len(err),
            "median_abs_error_s": float(np.median(err)) if err else None,
            "within_60s_over_all": sum(x <= 60 for x in err) / len(evaluations),
            "within_120s_over_all": sum(x <= 120 for x in err) / len(evaluations),
            "mean_reference_classification_mse": (
                float(np.mean(region_mse)) if region_mse else None),
            "mean_stable_region_accuracy": float(np.mean(stable)) if stable else None,
            "train_events": len(train_pools),
            "monitor_events": len(monitor_pools),
            "note": "region labels inferred from MPB/BS catalog, not measured composition",
        }
        results.append(report)
        print(f"Fold {fold} resumen: {report}", flush=True)
        del blocks
    write_csv(output / "summary.csv", results)
    (output / "summary.json").write_text(json.dumps({
        "configuration": vars(args), "results": results,
    }, indent=2, default=str) + "\n", encoding="utf-8")
    print(f"Resultados: {output}", flush=True)


if __name__ == "__main__":
    main()