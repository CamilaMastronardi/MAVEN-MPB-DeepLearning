#!/usr/bin/env python3
"""Grafica la tanh objetivo y la clasificacion reconstruida para MAG.

Usa checkpoints existentes de 5o_train_mag_tanh_zero.py y datos MAG
originales en cache. No reentrena, no consulta BigQuery.

La figura muestra una ventana de 600 s centrada en la MPB catalogada:
es el FRAGMENTO CENTRAL de la unica tanh objetivo del evento, no una
nueva tanh calculada por ventana.

Ejemplo:
  python scripts/5_training/plot_tanh_reconstruction.py --fold 1 --event-index 0
"""
from __future__ import annotations

import argparse
import importlib.util
import sys
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import torch

ROOT = Path(__file__).resolve().parents[2]
TRAIN_SCRIPT = ROOT / "scripts/5_training/5o_train_mag_tanh_zero.py"
VARIANTS = ("fixed", "adaptive", "consistent")


def load_training_module():
    spec = importlib.util.spec_from_file_location("mag_tanh_train_plot", TRAIN_SCRIPT)
    if spec is None or spec.loader is None:
        raise FileNotFoundError(f"No se pudo importar {TRAIN_SCRIPT}")
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--fold", type=int, choices=(1, 2, 3, 4), default=1)
    parser.add_argument("--event-index", type=int, default=0,
                        help="Indice (desde cero) de eventos del fold seleccionado")
    parser.add_argument("--pilot-val", type=int, default=12,
                        help="Mismo subconjunto de validacion del piloto")
    parser.add_argument("--torch-threads", type=int, default=2)
    parser.add_argument(
        "--output-dir", type=Path,
        default=ROOT / "results/figures/mag_tanh_reconstruction"
    )
    parser.add_argument(
        "--checkpoints-dir", type=Path,
        default=ROOT / "results/metrics/mag_tanh_zero_pilot"
    )
    args = parser.parse_args()
    if args.event_index < 0 or args.pilot_val < 0 or args.torch_threads < 1:
        parser.error("event-index >= 0, pilot-val >= 0 y torch-threads >= 1")
    torch.set_num_threads(args.torch_threads)

    old = load_training_module()
    catalog = old.read_catalog(ROOT / "data/splits/cv4_monthly_manifest.csv")
    cached = old.locate_mag_cache(ROOT / "data/cache/mag_events", catalog)
    val = [row for row in catalog
           if row["fold"] == args.fold and row["time"] in cached]
    val = old.select_even(val, args.pilot_val)

    if not val:
        raise RuntimeError(f"No hay cruces MAG en el fold {args.fold}")
    if args.event_index >= len(val):
        raise IndexError(
            f"event-index={args.event_index}: el maximo es {len(val) - 1}"
        )

    row = val[args.event_index]
    event_time = row["time"]
    block = old.load_block(cached[event_time], event_time)
    clip = old.extract_window(block, event_time)
    if clip is None:
        raise RuntimeError(
            "El evento no tiene una ventana central MAG valida de 600 segundos"
        )

    t, observed_mag = clip
    # La orientacion y tau son EXACTAMENTE los usados al construir
    # las etiquetas positivas durante el entrenamiento de 5o.
    direction = 1. if row["bs_time"] < event_time else -1.
    adaptive_tau, reliable = old.estimate_adaptive_tau(block, event_time)
    targets = {
        "fixed": direction * np.tanh(t / 60.),
        "adaptive": direction * np.tanh(t / adaptive_tau),
        "consistent": direction * np.tanh(t / adaptive_tau),
    }

    colors = {
        "fixed": "tab:blue",
        "adaptive": "tab:orange",
        "consistent": "tab:green",
    }
    curves = {}
    reconstructed_mag = {}
    for variant in VARIANTS:
        checkpoint = (
            args.checkpoints_dir /
            f"fold_{args.fold:02d}" /
            f"fold_{args.fold:02d}_{variant}_best.pt"
        )
        if not checkpoint.is_file():
            raise FileNotFoundError(f"No existe el checkpoint {checkpoint}")
        ck = torch.load(checkpoint, map_location="cpu", weights_only=False)
        if ck.get("variant") != variant or ck.get("fold") != args.fold:
            raise ValueError(f"Checkpoint incompatible: {checkpoint}")
        model = old.MagTanhAE()
        model.load_state_dict(ck["state_dict"])
        model.eval()
        norm = ck["normalizer"]
        class_curve, mag_normalized = old.predict_clip(model, clip, norm)
        curves[variant] = class_curve
        reconstructed_mag[variant] = (
            mag_normalized * norm["scale"] + norm["mean"]
        )

    fig, axes = plt.subplots(2, 1, figsize=(12, 8), sharex=True)

    ax = axes[0]
    ax.plot(t, observed_mag[:, 3], color="black", lw=1.5,
            label="|B| observado")
    for variant in VARIANTS:
        ax.plot(t, reconstructed_mag[variant][:, 3],
                color=colors[variant], lw=1.2, alpha=.85,
                label=f"|B| reconstruido ({variant})")
    ax.axvline(0, color="gray", ls="--", lw=1, label="MPB catalogada")
    ax.set_ylabel("|B| (nT)")
    ax.legend(fontsize=8)
    ax.set_title(
        f"Fold {args.fold} | Evento {args.event_index} | "
        f"{row['mpb_timestamp_utc']}\n"
        f"Tanh objetivo global (fragmento central de 600 s) | "
        f"tau adaptativo={adaptive_tau:.1f} s | fiable={reliable}"
    )

    ax = axes[1]
    ax.plot(t, targets["fixed"], color="black", ls="--", lw=2,
            label="Tanh objetivo fixed (tau=60 s)")
    ax.plot(t, targets["adaptive"], color="purple", ls="-.", lw=2,
            label=f"Tanh objetivo adaptive/consistent (tau={adaptive_tau:.1f} s)")
    for variant in VARIANTS:
        ax.plot(t, curves[variant], color=colors[variant], lw=1.4,
                label=f"Clasificacion reconstruida ({variant})")
    ax.axhline(0, color="gray", lw=.8)
    ax.axvline(0, color="gray", ls="--", lw=1)
    ax.set_xlabel("Tiempo respecto de la MPB catalogada (s)")
    ax.set_ylabel("Clasificacion [-1, 1]")
    ax.set_ylim(-1.15, 1.15)
    ax.legend(fontsize=8)

    fig.tight_layout()
    args.output_dir.mkdir(parents=True, exist_ok=True)
    output = (args.output_dir /
              f"fold_{args.fold:02d}_event_{args.event_index:02d}_tanh_reconstruction.png")
    fig.savefig(output, dpi=180)
    plt.close(fig)
    print(f"Figura guardada: {output}")


if __name__ == "__main__":
    main()
