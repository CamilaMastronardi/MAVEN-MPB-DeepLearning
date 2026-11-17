#!/usr/bin/env python3
"""Diagnóstico MAG-only de la anchura de una etiqueta tanh en cruces MPB.

Lee EXCLUSIVAMENTE los NPZ existentes en data/cache/mag_events/.
No accede a BigQuery, no entrena un modelo ni modifica checkpoints.
Ejemplo (desde la raíz del repositorio):
    python scripts/analisis_mag_tanh.py
    python scripts/analisis_mag_tanh.py --scan-events 100 --plot-events 8
"""
from __future__ import annotations

import argparse
import csv
import json
from datetime import datetime, timezone
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from scipy.signal import savgol_filter


def utc_seconds(value: str) -> float:
    return datetime.fromisoformat(value.replace("Z", "+00:00")).timestamp()


def locate_root(folder: Path) -> Path:
    for path in (folder, *folder.parents):
        if (path / "data/splits/cv4_monthly_manifest.csv").exists():
            return path
    raise FileNotFoundError(
        "No encuentro data/splits/cv4_monthly_manifest.csv. "
        "Ejecutá este script desde la raíz del repositorio."
    )


def catalog(path: Path) -> dict[float, dict]:
    with path.open(newline="", encoding="utf-8") as handle:
        rows = list(csv.DictReader(handle))
    return {utc_seconds(row["mpb_timestamp_utc"]): row for row in rows}


def cached_files(directory: Path, manifest: dict) -> list[tuple[float, Path]]:
    found = {}
    for file in sorted(directory.glob("mag_*.npz")):
        try:
            with np.load(file, allow_pickle=False) as archive:
                meta = json.loads(str(archive["metadata"].item()))
            event = utc_seconds(meta["spec"]["event_utc"])
        except (KeyError, ValueError, OSError, json.JSONDecodeError) as exc:
            print(f"[AVISO] Caché ilegible {file.name}: {exc}")
            continue
        if event in manifest:
            found[event] = file
    return sorted(found.items())


def measure(event_time: float, file: Path, row: dict,
            tau_min: float, tau_max: float) -> dict | None:
    if not row.get("bs_timestamp_utc"):
        return None
    bs = utc_seconds(row["bs_timestamp_utc"])
    if not 300 <= abs(bs - event_time) <= 7200:
        return None

    with np.load(file, allow_pickle=False) as archive:
        times = archive["times"].copy()
        values = archive["values"].copy()
    if times.ndim != 1 or values.shape != (len(times), 4):
        return None
    mask = (times >= event_time - 300) & (times <= event_time + 300)
    t = times[mask] - event_time
    B = values[mask]
    if len(t) < 540 or not np.isfinite(B).all() or not np.isfinite(t).all():
        return None
    dt = np.diff(t)
    if (t[0] > -298 or t[-1] < 298 or
            np.any(dt <= 0) or np.max(dt) > 2.5):
        return None

    bmag = B[:, 3]
    smooth = savgol_filter(bmag, 31, 2)
    before = (t >= -150) & (t <= -60)
    after = (t >= 60) & (t <= 150)
    middle = (t >= -45) & (t <= 45)
    if min(before.sum(), after.sum(), middle.sum()) < 15:
        return None
    jump = float(np.median(smooth[after]) - np.median(smooth[before]))
    noise = float(1.4826 * np.median(np.abs(bmag - smooth)))
    derivative = np.gradient(smooth, t)
    slope = float(np.percentile(np.abs(derivative[middle]), 85))
    reliable = abs(jump) > max(1.0, 3 * noise) and slope > 0.01
    raw_tau = abs(jump) / (2 * slope) if reliable else float("nan")
    tau = float(np.clip(raw_tau, tau_min, tau_max)) if reliable else 60.0

    # BS anterior a MPB: 0 (magnetovaina) -> 1 (magnetosfera inducida).
    # BS posterior a MPB: transición inversa.
    direction = 1 if bs < event_time else -1
    target = 0.5 * (1 + direction * np.tanh(t / tau))
    mismatch = float(np.median(np.abs(bmag - np.linalg.norm(B[:, :3], axis=1))))
    return {
        "event_utc": datetime.fromtimestamp(event_time, timezone.utc).isoformat(),
        "fold": int(row["fold"]),
        "group": row["source_group"],
        "tau_s": tau,
        "tau_unclipped_s": raw_tau,
        "jump_nt": jump,
        "slope_nt_per_s": slope,
        "noise_nt": noise,
        "reliable_slope": reliable,
        "label_direction": direction,
        "median_magnitude_mismatch_nt": mismatch,
        "n_mag_samples": len(t),
        "t": t, "bmag": bmag, "smooth": smooth, "target": target,
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--root", type=Path, default=Path.cwd(),
                        help="Raíz del repositorio (por defecto: directorio actual)")
    parser.add_argument("--scan-events", type=int, default=60,
                        help="Máximo de eventos distribuidos a revisar")
    parser.add_argument("--plot-events", type=int, default=6,
                        help="Máximo de eventos válidos para la figura")
    parser.add_argument("--tau-min", type=float, default=10.)
    parser.add_argument("--tau-max", type=float, default=120.)
    parser.add_argument("--output-dir", type=Path, default=None)
    args = parser.parse_args()
    if (args.scan_events < 1 or args.plot_events < 1 or
            not 0 < args.tau_min < 60 < args.tau_max):
        parser.error("Parámetros inválidos: scan/plot > 0 y tau-min < 60 < tau-max")

    root = locate_root(args.root.resolve())
    manifest = catalog(root / "data/splits/cv4_monthly_manifest.csv")
    cache = root / "data/cache/mag_events"
    files = cached_files(cache, manifest)
    print(f"Catálogo: {len(manifest)} eventos | MAG en caché: {len(files)}")
    if not files:
        parser.error(f"No hay NPZ MAG compatibles en {cache}")

    ids = np.unique(np.linspace(0, len(files) - 1,
                                min(args.scan_events, len(files)), dtype=int))
    analyzed = []
    for i in ids:
        event, file = files[int(i)]
        result = measure(event, file, manifest[event], args.tau_min, args.tau_max)
        if result is not None:
            analyzed.append(result)
    if not analyzed:
        parser.error("Ninguna de las ventanas seleccionadas superó los controles de calidad")

    out = (args.output_dir or (root / "results/figures/mag_tanh_pilot")).resolve()
    out.mkdir(parents=True, exist_ok=True)
    columns = ["event_utc", "fold", "group", "tau_s", "tau_unclipped_s",
               "jump_nt", "slope_nt_per_s", "noise_nt", "reliable_slope",
               "label_direction", "median_magnitude_mismatch_nt", "n_mag_samples"]
    with (out / "mag_tanh_event_summary.csv").open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=columns)
        writer.writeheader()
        writer.writerows({k: r[k] for k in columns} for r in analyzed)

    selected_ids = np.unique(np.linspace(0, len(analyzed) - 1,
                        min(args.plot_events, len(analyzed)), dtype=int))
    selected = [analyzed[int(i)] for i in selected_ids]
    fig, axes = plt.subplots(len(selected), 1, figsize=(12, 3.8 * len(selected)),
                             squeeze=False)
    for ax, result in zip(axes[:, 0], selected):
        t = result["t"]
        ax.plot(t, result["bmag"], label="|B| observado", alpha=0.5, linewidth=0.9)
        ax.plot(t, result["smooth"], label="|B| suavizado", linewidth=1.5)
        ax.axvspan(-150, -60, color="gray", alpha=0.10)
        ax.axvspan(60, 150, color="gray", alpha=0.10)
        ax.axvline(0, color="gray", ls="--", label="MPB catalogada")
        ax.set_ylabel("|B| (nT)")
        ax.set_xlim(-300, 300)
        ax2 = ax.twinx()
        ax2.plot(t, result["target"], color="black", ls=":", lw=1.8,
                 label="Tanh objetivo")
        ax2.set_ylim(-0.05, 1.05)
        ax2.set_ylabel("Etiqueta débil de región")
        ax.set_title(f'{result["event_utc"]} | fold {result["fold"]} | '
                     f'Δ|B|={result["jump_nt"]:.1f} nT | '
                     f'τ={result["tau_s"]:.1f} s | pendiente fiable={result["reliable_slope"]}')
        ax.set_xlabel("Tiempo respecto de MPB (s)")
        ax.legend(loc="upper left", fontsize=8)
    fig.tight_layout()
    output = out / "mag_tanh_ejemplos.png"
    fig.savefig(output, dpi=180)
    plt.close(fig)

    n_reliable = sum(r["reliable_slope"] for r in analyzed)
    print(f"Eventos analizados: {len(analyzed)} / {len(ids)}")
    print(f"Pendiente fiable: {n_reliable}/{len(analyzed)}")
    print(f"Tau mediana: {np.median([r['tau_s'] for r in analyzed]):.1f} s")
    print(f"Resultados: {out}")
    print(f"- {output.name}")
    print("- mag_tanh_event_summary.csv")
    print("Este es un diagnóstico de etiquetas, no un entrenamiento ni una evaluación predictiva.")


if __name__ == "__main__":
    main()
