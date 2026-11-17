#!/usr/bin/env python3
"""MAG-only ablation: reconstruct field, learn tanh region transition, locate MPB at fitted zero.

Input: the verified MAG NPZ cache, catalog and month folds (no BigQuery).
Each candidate window is 600 s of native MAG samples. No catalog date, BS,
MPB time, or tau is passed to the network during inference. The fitted zero
is computed from predicted *classification curve*, never magnetic amplitude.

Three comparable variants, all predicting classification to enable localization:
  fixed       : field reconstruction + tanh target with fixed tau=60 s
  adaptive    : field reconstruction + tau from signed mean d|B|/dt in +/-30 s
  consistent  : adaptive + magnetic norm consistency of decoder output

Pilot example:
  python scripts/5_training/5o_train_mag_tanh_zero.py --folds 1 \\
    --pilot-train 48 --pilot-monitor 12 --pilot-val 12 --epochs 20

Full experiment: --folds all --epochs 100 --patience 20.
This is conditional localization within +/-30 min of a catalog MPB, not an
orbital autonomous detection benchmark. No performance is assumed in advance.
"""
from __future__ import annotations

import argparse
import csv
import json
import random
import sys
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from scipy.optimize import least_squares
from scipy.signal import savgol_filter
import torch
from torch import nn

ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from src.data.mag_event_cache import array_hash  # Cache source integrity check
from src.data.mpb_weak_regions import classify_candidate

VARIANTS = ("fixed", "adaptive", "consistent")
POS_OFFSETS = (-240, -180, -120, -60, 0, 60, 120, 180, 240)
NEG_OFFSETS = (-1500, -1200, -900, -600, 600, 900, 1200, 1500)


def utc_seconds(raw: str) -> float:
    return datetime.fromisoformat(raw.replace("Z", "+00:00")).timestamp()


def read_catalog(path: Path) -> list[dict]:
    with path.open(newline="", encoding="utf-8") as f:
        rows = list(csv.DictReader(f))
    if not rows:
        raise ValueError("Catálogo vacío")
    for row in rows:
        row["time"] = utc_seconds(row["mpb_timestamp_utc"])
        row["bs_time"] = utc_seconds(row["bs_timestamp_utc"])
        row["fold"] = int(row["fold"])
    assert len({r["time"] for r in rows}) == len(rows)
    return sorted(rows, key=lambda r: r["time"])


def locate_mag_cache(root: Path, rows: list[dict]) -> dict[float, Path]:
    wanted = {r["time"] for r in rows if r["mag_available"] == "1"}
    mapping = {}
    for path in sorted(root.glob("mag_*.npz")):
        with np.load(path, allow_pickle=False) as archive:
            raw = json.loads(str(archive["metadata"].item()))
        date = utc_seconds(raw["spec"]["event_utc"])
        if date in wanted:
            mapping[date] = path
    return mapping


def load_block(path: Path, expected_time: float) -> tuple[np.ndarray, np.ndarray]:
    with np.load(path, allow_pickle=False) as archive:
        times, field = archive["times"].copy(), archive["values"].copy()
        metadata = json.loads(str(archive["metadata"].item()))
    if abs(utc_seconds(metadata["spec"]["event_utc"]) - expected_time) > 0.001:
        raise ValueError("MAG event cache metadata mismatch")
    if (times.dtype != np.float64 or field.dtype != np.float64 or
            field.shape != (len(times), 4) or not np.isfinite(times).all() or
            not np.isfinite(field).all() or np.any(np.diff(times) <= 0)):
        raise ValueError("MAG cache invalid")
    if (int(metadata["n_rows"]) != len(times) or
            metadata["array_sha256"] != array_hash(times, field)):
        raise ValueError("MAG cache SHA256 mismatch")
    return times, field


def select_even(rows: list[dict], maximum: int) -> list[dict]:
    if maximum <= 0 or len(rows) <= maximum:
        return rows
    positions = np.unique(np.linspace(0, len(rows) - 1, maximum).round().astype(int))
    return [rows[int(i)] for i in positions]


def split_fit_monitor(rows: list[dict], fold: int, seed: int) -> tuple[list[dict], list[dict]]:
    months = sorted({r["month_block"] for r in rows})
    if len(months) < 2:
        raise ValueError("Se requieren >=2 meses TRAIN para monitor independiente")
    random.Random(seed + 191 * fold).shuffle(months)
    monitor_months = set(months[:max(1, min(len(months)-1, round(.1*len(months))))])
    mon = [r for r in rows if r["month_block"] in monitor_months]
    fit = [r for r in rows if r["month_block"] not in monitor_months]
    # Prevent data overlap within +/-35 minutes even across UTC month edges.
    mon_times = np.array([r["time"] for r in mon])
    fit = [r for r in fit if np.min(np.abs(mon_times - r["time"])) > 4200]
    if not fit or not mon:
        raise ValueError("Empty FIT/MONITOR")
    return fit, mon


def extract_window(block: tuple[np.ndarray, np.ndarray], center: float):
    times, values = block
    first = np.searchsorted(times, center - 300, side="left")
    # Include the final sample if on the right edge, without modifying cadence.
    right = np.searchsorted(times, center + 300, side="right")
    t, x = times[first:right] - center, values[first:right]
    if len(t) < 540 or len(t) > 640 or not np.isfinite(x).all():
        return None
    dt = np.diff(t)
    if (np.any(dt <= 0) or np.max(dt) > 2.5 or
            t[0] > -298 or t[-1] < 298):
        return None
    return np.ascontiguousarray(t, dtype=np.float32), np.ascontiguousarray(x, dtype=np.float32)


def estimate_adaptive_tau(block: tuple[np.ndarray, np.ndarray], mpb: float,
                          *, derivative_window_s: float = 60.) -> tuple[float, bool]:
    clip = extract_window(block, mpb)
    if clip is None:
        return 60., False
    t, B = clip
    b = B[:, 3].astype(float)
    smooth = savgol_filter(b, 31, 2)
    before = (t >= -150) & (t <= -60)
    after = (t >= 60) & (t <= 150)
    middle = (t >= -derivative_window_s / 2) & (t <= derivative_window_s / 2)
    if min(before.sum(), after.sum(), middle.sum()) < 15:
        return 60., False
    jump = float(np.median(smooth[after]) - np.median(smooth[before]))
    noise = float(1.4826 * np.median(np.abs(b - smooth)))
    d = float(np.mean(np.gradient(smooth, t)[middle]))  # Signed mean, not max!
    reliable = (abs(jump) > max(1., 3*noise) and abs(d) > .01
                and np.sign(jump) == np.sign(d))
    return (float(np.clip(abs(jump)/(2*abs(d)), 10., 120.)), True) if reliable else (60., False)


@dataclass
class Example:
    t: np.ndarray              # seconds relative to candidate center
    x: np.ndarray              # measured MAG nT; shape T x 4
    y: np.ndarray              # signed tanh target [-1, 1] or constant control
    positive: bool
    event_time: float
    center: float


def build_event_examples(row: dict, block, all_mpb: np.ndarray, all_bs: np.ndarray,
                         variant: str) -> tuple[list[Example], list[Example]]:
    mpb = row["time"]
    tau, _ = estimate_adaptive_tau(block, mpb)
    if variant == "fixed":
        tau = 60.
    # Weak orientation derived from BS order; NEVER model input.
    direction = 1. if row["bs_time"] < mpb else -1.
    positives, negatives = [], []
    # The existing V7b physical-label utility expects aware datetime objects.
    # Keep float Unix times for numerical windows but adapt this one API.
    physical_event = {
        **row,
        "time": datetime.fromtimestamp(mpb, timezone.utc),
        "bs_time": datetime.fromtimestamp(row["bs_time"], timezone.utc),
    }
    for offset in (*POS_OFFSETS, *NEG_OFFSETS):
        center = mpb + offset
        candidate = classify_candidate(physical_event, center, all_mpb_s=all_mpb, all_bs_s=all_bs)
        positive = offset in POS_OFFSETS
        if positive:
            if candidate.kind not in ("positive", "mpb_ambiguous"):
                continue
        else:
            if candidate.transition != 0 or candidate.left_region is None:
                continue
        window = extract_window(block, center)
        if window is None:
            continue
        t, values = window
        if positive:
            target = direction * np.tanh((t + offset) / tau)
        else:
            target = np.full_like(t, -1. if candidate.left_region == 0 else 1.)
        e = Example(t, values, target.astype(np.float32), positive, mpb, center)
        (positives if positive else negatives).append(e)
    return positives, negatives


def norm_fit(blocks: dict[float, tuple[np.ndarray, np.ndarray]]) -> dict:
    ts = np.concatenate([pair[0] for pair in blocks.values()])
    x = np.concatenate([pair[1] for pair in blocks.values()])
    order = np.argsort(ts, kind="stable")
    unique = np.ones(len(order), bool)
    unique[1:] = np.diff(ts[order]) >= .001
    x = x[order[unique]]
    mu, scale = x.mean(0), x.std(0)
    # Do not divide by an almost-zero standard deviation for a quiet channel.
    scale = np.where(scale > 1.e-3, scale, 1.)
    return {"mean": mu.astype(np.float32), "scale": scale.astype(np.float32)}


def collate(samples: list[Example], normalizer: dict, device="cpu") -> dict:
    length = max(len(item.t) for item in samples)
    b = len(samples)
    x = np.zeros((b, 4, length), dtype=np.float32)
    t = np.zeros((b, length), dtype=np.float32)
    y = np.zeros((b, length), dtype=np.float32)
    mask = np.zeros((b, length), dtype=np.bool_)
    positive = np.empty((b,), dtype=np.bool_)
    for i, ex in enumerate(samples):
        n = len(ex.t)
        x[i, :, :n] = ((ex.x - normalizer["mean"]) / normalizer["scale"]).T
        y[i, :n], t[i, :n], mask[i, :n] = ex.y, ex.t, True
        positive[i] = ex.positive
    return {
        "x": torch.as_tensor(x, device=device),
        "y": torch.as_tensor(y, device=device),
        "t": torch.as_tensor(t, device=device),
        "mask": torch.as_tensor(mask, device=device),
        "positive": torch.as_tensor(positive, device=device),
    }


class ResidualBlock(nn.Module):
    def __init__(self, width: int, dilation: int):
        super().__init__()
        self.conv = nn.Conv1d(width, width, kernel_size=5,
                              padding=2*dilation, dilation=dilation)
        self.point = nn.Conv1d(width, width, kernel_size=1)
        self.act = nn.GELU()

    def forward(self, x):
        return x + self.point(self.act(self.conv(x)))


class MagTanhAE(nn.Module):
    """V7b-style residual encoder with separate magnetic and classification decoders."""
    def __init__(self):
        super().__init__()
        self.encoder = nn.Sequential(
            nn.Conv1d(4, 32, 5, padding=2), nn.GELU(),
            *[ResidualBlock(32, d) for d in (1, 2, 4, 8)],
            nn.Conv1d(32, 16, 1), nn.GELU(),
        )
        self.field_decoder = nn.Sequential(
            nn.Conv1d(16, 32, 5, padding=2), nn.GELU(),
            nn.Conv1d(32, 4, 5, padding=2),
        )
        self.class_decoder = nn.Sequential(
            nn.Conv1d(16, 32, 5, padding=2), nn.GELU(),
            nn.Conv1d(32, 1, 5, padding=2), nn.Tanh(),
        )

    def forward(self, x, mask):
        z = self.encoder(x * mask[:, None, :])
        return (self.field_decoder(z),
                self.class_decoder(z).squeeze(1))


def masked_mean(value: torch.Tensor, mask: torch.Tensor) -> torch.Tensor:
    return (value*mask).sum()/mask.sum().clamp_min(1.)


def losses(pred_field: torch.Tensor, pred_class: torch.Tensor, batch: dict,
           norm: dict, consistency: bool, cls_weight: float,
           norm_weight: float) -> tuple[torch.Tensor, dict]:
    mask = batch["mask"]
    recon = masked_mean((pred_field-batch["x"]).square().mean(1), mask)
    # Extra focus on the transition: y≈0, rather than saturating y≈+/-1.
    weights = 1. + 2.*(1. - batch["y"].square()) * batch["positive"][:, None]
    classification = masked_mean(weights*(pred_class-batch["y"]).square(), mask)
    if consistency:
        mu = torch.as_tensor(norm["mean"], device=pred_field.device)
        scale = torch.as_tensor(norm["scale"], device=pred_field.device)
        p = pred_field*scale[None, :, None] + mu[None, :, None]
        calc = torch.linalg.vector_norm(p[:, :3], dim=1)
        delta = (p[:, 3] - calc)/scale[3].clamp_min(1.)
        normloss = masked_mean(delta.square(), mask)
    else:
        normloss = recon.new_zeros(())
    total = recon + cls_weight*classification + norm_weight*normloss
    return total, {"reconstruction": float(recon.detach()),
                   "classification": float(classification.detach()),
                   "consistency": float(normloss.detach()),
                   "total": float(total.detach())}


def train_epoch(model, optimizer, pools, norm, variant, args, epoch, monitor=False):
    randomizer = random.Random(args.seed + epoch)
    all_examples = []
    for positive, negative in pools.values():
        if monitor:
            all_examples.extend(positive[:3])
            all_examples.extend(negative[:2])
        else:
            all_examples.extend(randomizer.sample(positive, min(3, len(positive))))
            all_examples.extend(randomizer.sample(negative, min(2, len(negative))))
    if not monitor:
        randomizer.shuffle(all_examples)
    if not all_examples:
        raise ValueError("No eligible training/monitor examples")
    model.train(not monitor)
    total, size = {}, 0
    for start in range(0, len(all_examples), args.batch_size):
        selection = all_examples[start:start+args.batch_size]
        b = collate(selection, norm)
        with torch.set_grad_enabled(not monitor):
            pred, class_curve = model(b["x"], b["mask"])
            loss, metrics = losses(pred, class_curve, b, norm,
                                   variant == "consistent", args.class_weight,
                                   args.consistency_weight)
            if not monitor:
                optimizer.zero_grad(set_to_none=True)
                loss.backward()
                nn.utils.clip_grad_norm_(model.parameters(), 1.)
                optimizer.step()
        for key, value in metrics.items():
            total[key] = total.get(key, 0.) + value*len(selection)
        size += len(selection)
    return {key: value/size for key, value in total.items()}


def tanh_model(t, amplitude, center, tau):
    return amplitude * np.tanh((t-center)/tau)


def fit_classification_zero(t: np.ndarray, y: np.ndarray):
    """Fit a tanh to reconstructed CLASSIFICATION; zero crossing = center.

    Zero is y=0 because classification head has [-1, +1] output. For a
    [0,1] sigmoid parameterization the same MPB would be at level 0.5.
    """
    if len(t) < 30 or not np.isfinite(y).all():
        return None
    t = t.astype(np.float64)
    y = y.astype(np.float64)
    direction = float(np.sign(y[-1]-y[0])) or 1.
    initial = [float(np.clip((y[-1]-y[0])/2., -1., 1.)), 0., 60.]
    if abs(initial[0]) < 1.e-3:
        initial[0] = direction*.01
    try:
        result = least_squares(
            lambda pars: tanh_model(t, *pars)-y,
            x0=initial,
            bounds=([-1.5, -240., 8.], [1.5, 240., 180.]),
            max_nfev=65, method="trf",
        )
    except ValueError:
        return None
    amplitude, center, tau = (float(z) for z in result.x)
    mse = float(np.mean(result.fun**2))
    contrast = abs(float(np.mean(y[t > 180]) - np.mean(y[t < -180])))
    # We require a significant transition, a non-boundary location and
    # an actual zero-crossing within the measured support.
    if (not result.success or abs(amplitude) < .25 or contrast < .25 or
            abs(center) >= 239.9 or abs(center) >= max(abs(t[0]), abs(t[-1]))):
        return None
    return {"zero_s": center, "tau_fit_s": tau, "amplitude": amplitude,
            "mse_fit": mse, "contrast": contrast,
            "score": abs(amplitude)*contrast/(0.1+mse),
            "tanh": tanh_model(t, amplitude, center, tau)}


def predict_clip(model, clip, norm):
    t, x = clip
    placeholder = Example(t, x, np.zeros_like(t), False, 0., 0.)
    b = collate([placeholder], norm)
    with torch.no_grad():
        pred, curve = model(b["x"], b["mask"])
    return curve[0, :len(t)].cpu().numpy(), pred[0, :, :len(t)].cpu().numpy().T


def search_event(model, row, block, norm, scan_stride: int, max_fit: int):
    """Catalog defines +/-30min search vicinity; no true MPB to model inputs."""
    offsets = np.arange(-1800., 1800.01, scan_stride)
    candidates = []
    model.eval()
    for off in offsets:
        clip = extract_window(block, row["time"]+float(off))
        if clip is None:
            continue
        y, field = predict_clip(model, clip, norm)
        t = clip[0]
        contrast = abs(float(np.mean(y[t > 180]) - np.mean(y[t < -180])))
        if contrast > .15:
            candidates.append((contrast, float(off), clip, y, field))
    candidates.sort(key=lambda item: item[0], reverse=True)
    best = None
    for _, off, clip, y, field in candidates[:max_fit]:
        fit = fit_classification_zero(clip[0], y)
        if fit is None:
            continue
        t_pred = row["time"]+off+fit["zero_s"]
        if best is None or fit["score"] > best["score"]:
            best = {**fit, "pred_s": t_pred, "center_offset_s": off,
                    "clip": clip, "curve": y, "reconstructed_field": field}
    return best


def save_centered_diagnostic(output: Path, row: dict, traces: dict):
    """RETROSPECTIVE only: center uses catalog time for illustration, not inference."""
    fig, axes = plt.subplots(2, 1, figsize=(11, 7), sharex=True)
    for variant, (t, B, curve, field, norm) in traces.items():
        axes[0].plot(t, curve, alpha=.85, label=f"{variant}: clasificación")
        reconstructed = field*norm["scale"]+norm["mean"]
        axes[1].plot(t, reconstructed[:, 3], label=f"{variant}: |B| reconstr.", alpha=.7)
        if variant == next(iter(traces)):
            axes[1].plot(t, B[:, 3], color="k", alpha=.5, label="|B| observado")
    for axis in axes:
        axis.axvline(0., color="k", ls="--", label="MPB catalogada")
        axis.legend(fontsize=8)
        axis.set_xlabel("Tiempo respecto de la MPB catalogada (s)")
    axes[0].set_ylabel("Salida de clasificación [-1, +1]")
    axes[1].set_ylabel("|B| (nT)")
    axes[0].set_title("Diagnóstico RETROSPECTIVO centrado en catálogo (NO inferencia)")
    fig.tight_layout()
    fig.savefig(output, dpi=160)
    plt.close(fig)


def serialize_result(result, row, variant, fold):
    if result is None:
        return {"variant": variant, "fold": fold,
                "event_utc": row["mpb_timestamp_utc"], "evaluated": False}
    return {"variant": variant, "fold": fold,
            "event_utc": row["mpb_timestamp_utc"], "evaluated": True,
            "pred_utc": datetime.fromtimestamp(result["pred_s"], timezone.utc).isoformat(),
            "error_s": result["pred_s"]-row["time"],
            "abs_error_s": abs(result["pred_s"]-row["time"]),
            "fit_tau_s": result["tau_fit_s"],
            "fit_amplitude": result["amplitude"],
            "fit_mse": result["mse_fit"], "candidate_score": result["score"],
            "selected_center_offset_s": result["center_offset_s"]}


def write_csv(path, rows):
    if not rows:
        return
    fields = sorted({k for row in rows for k in row})
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields)
        writer.writeheader()
        writer.writerows(rows)


def selected_rows(catalog_rows, fold, available, args):
    val = [r for r in catalog_rows if r["fold"] == fold and r["time"] in available]
    train = [r for r in catalog_rows if r["fold"] != fold and r["time"] in available]
    fit, mon = split_fit_monitor(train, fold, args.seed)
    return {
        "fit": select_even(fit, args.pilot_train),
        "monitor": select_even(mon, args.pilot_monitor),
        "val": select_even(val, args.pilot_val),
    }


def make_pools(rows, blocks, all_mpb, all_bs, variant):
    result = {}
    for row in rows:
        pos, neg = build_event_examples(row, blocks[row["time"]], all_mpb, all_bs, variant)
        if pos and neg:
            result[row["time"]] = (pos, neg)
    return result


def train_variant(args, fold, variant, splits, blocks, all_mpb, all_bs, out):
    fit_blocks = {r["time"]: blocks[r["time"]] for r in splits["fit"]}
    norm = norm_fit(fit_blocks)
    train_pools = make_pools(splits["fit"], blocks, all_mpb, all_bs, variant)
    mon_pools = make_pools(splits["monitor"], blocks, all_mpb, all_bs, variant)
    if not train_pools or not mon_pools:
        raise RuntimeError(f"No safe training/monitor examples for fold {fold}")
    torch.manual_seed(args.seed+fold)
    model = MagTanhAE()
    opt = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=1e-3)
    best_loss, best_epoch, stale = float("inf"), 0, 0
    history = []
    best_path = out/f"fold_{fold:02d}_{variant}_best.pt"
    for epoch in range(1, args.epochs+1):
        tr = train_epoch(model, opt, train_pools, norm, variant, args, epoch)
        mon = train_epoch(model, None, mon_pools, norm, variant, args, epoch, monitor=True)
        history.append({"epoch": epoch, "train": tr, "monitor": mon})
        if mon["total"] < best_loss - 1.e-4:
            best_loss, best_epoch, stale = mon["total"], epoch, 0
            torch.save({"state_dict": model.state_dict(), "variant": variant,
                        "fold": fold, "epoch": epoch,
                        "normalizer": norm}, best_path)
        else:
            stale += 1
        print(f"Fold {fold} {variant:10s} ep {epoch:3d}: "
              f"train={tr['total']:.4f} monitor={mon['total']:.4f} "
              f"best={best_epoch} stale={stale}", flush=True)
        if epoch >= args.min_epochs and stale >= args.patience:
            break
    checkpoint = torch.load(best_path, map_location="cpu", weights_only=False)
    model.load_state_dict(checkpoint["state_dict"])
    model.eval()
    history_path = out/f"fold_{fold:02d}_{variant}_history.json"
    history_path.write_text(json.dumps(history, indent=2)+"\n", encoding="utf-8")
    return model, norm, {"best_epoch": best_epoch,
                         "train_pools": len(train_pools),
                         "monitor_pools": len(mon_pools)}


def parse_args():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--manifest", type=Path, default=ROOT/"data/splits/cv4_monthly_manifest.csv")
    p.add_argument("--mag-cache", type=Path, default=ROOT/"data/cache/mag_events")
    p.add_argument("--output-dir", type=Path, default=None)
    p.add_argument("--folds", default="1", choices=["all", "1", "2", "3", "4"])
    p.add_argument("--variants", default="all", choices=["all", *VARIANTS])
    p.add_argument("--pilot-train", type=int, default=48)
    p.add_argument("--pilot-monitor", type=int, default=12)
    p.add_argument("--pilot-val", type=int, default=12)
    p.add_argument("--epochs", type=int, default=20)
    p.add_argument("--min-epochs", type=int, default=4)
    p.add_argument("--patience", type=int, default=6)
    p.add_argument("--lr", type=float, default=3.e-4)
    p.add_argument("--class-weight", type=float, default=1.0)
    p.add_argument("--consistency-weight", type=float, default=.2)
    p.add_argument("--batch-size", type=int, default=16)
    p.add_argument("--scan-stride", type=int, default=64)
    p.add_argument("--max-fits", type=int, default=8)
    p.add_argument("--plot-events", type=int, default=3)
    p.add_argument("--torch-threads", type=int, default=2)
    p.add_argument("--seed", type=int, default=42)
    args = p.parse_args()
    if (min(args.epochs, args.min_epochs, args.patience, args.batch_size,
            args.scan_stride, args.max_fits, args.torch_threads) < 1 or
            args.min_epochs > args.epochs or args.pilot_train < 0 or
            args.pilot_monitor < 0 or args.pilot_val < 0 or
            args.lr <= 0 or args.class_weight <= 0 or args.consistency_weight < 0 or
            args.plot_events < 0):
        p.error("Invalid training/evaluation parameters")
    return args


def main():
    args = parse_args()
    torch.set_num_threads(args.torch_threads)
    np.random.seed(args.seed)
    random.seed(args.seed)
    catalog_rows = read_catalog(args.manifest)
    cached = locate_mag_cache(args.mag_cache, catalog_rows)
    print(f"MAG cache: {len(cached)} / {len(catalog_rows)} catalog events", flush=True)
    if not cached:
        raise FileNotFoundError(f"No MAG cache at {args.mag_cache}")
    chosen_folds = [1, 2, 3, 4] if args.folds == "all" else [int(args.folds)]
    variants = VARIANTS if args.variants == "all" else (args.variants,)
    all_mpb = np.array([r["time"] for r in catalog_rows])
    all_bs = np.array([r["bs_time"] for r in catalog_rows])
    pilot = any((args.pilot_train, args.pilot_monitor, args.pilot_val))
    output = (args.output_dir or ROOT/"results/metrics"/
              ("mag_tanh_zero_pilot" if pilot else "mag_tanh_zero"))
    output.mkdir(parents=True, exist_ok=True)
    summary = []
    for fold in chosen_folds:
        splits = selected_rows(catalog_rows, fold, cached, args)
        if not all(splits.values()):
            print(f"Fold {fold}: no usable cached events for split", flush=True)
            continue
        print(f"Fold {fold}: FIT={len(splits['fit'])} MON={len(splits['monitor'])} "
              f"VAL={len(splits['val'])}", flush=True)
        blocks = {r["time"]: load_block(cached[r["time"]], r["time"])
                  for subset in splits.values() for r in subset}
        fold_out = output/f"fold_{fold:02d}"
        fold_out.mkdir(parents=True, exist_ok=True)
        all_pred_rows = []
        plot_results = {}
        centered_diagnostics = {}
        for variant in variants:
            model, norm, details = train_variant(
                args, fold, variant, splits, blocks, all_mpb, all_bs, fold_out
            )
            pred_rows = []
            for n, row in enumerate(splits["val"]):
                result = search_event(model, row, blocks[row["time"]], norm,
                                      args.scan_stride, args.max_fits)
                pred_rows.append(serialize_result(result, row, variant, fold))
                if n < args.plot_events:
                    plot_results.setdefault(row["time"], {"row": row})[variant] = (result, norm)
                    centered_clip = extract_window(blocks[row["time"]], row["time"])
                    if centered_clip is not None:
                        centered_y, centered_field = predict_clip(model, centered_clip, norm)
                        centered_diagnostics.setdefault(row["time"], {"row": row})[variant] = (
                            centered_clip[0], centered_clip[1], centered_y, centered_field, norm
                        )
                print(f"Fold {fold} {variant} VAL {n+1}/{len(splits['val'])}: "
                      f"error={pred_rows[-1].get('abs_error_s')}", flush=True)
            write_csv(fold_out/f"fold_{fold:02d}_{variant}_events.csv", pred_rows)
            all_pred_rows += pred_rows
            errors = [r["abs_error_s"] for r in pred_rows if r["evaluated"]]
            report = {"fold": fold, "variant": variant,
                      "val_count": len(pred_rows), "evaluable": len(errors),
                      "evaluable_fraction": len(errors)/len(pred_rows),
                      "fraction_within_60s_over_all": sum(e <= 60 for e in errors)/len(pred_rows),
                      "fraction_within_120s_over_all": sum(e <= 120 for e in errors)/len(pred_rows),
                      "median_abs_error_s": float(np.median(errors)) if errors else None,
                      "fraction_within_60s": sum(e <= 60 for e in errors)/len(errors) if errors else None,
                      "fraction_within_120s": sum(e <= 120 for e in errors)/len(errors) if errors else None,
                      **details}
            summary.append(report)
            print(f"{variant} fold {fold} summary: {report}", flush=True)
        # Fair paired comparison: only events localized by ALL variants.
        if len(variants) > 1:
            successful = None
            for variant in variants:
                ids = {r["event_utc"] for r in all_pred_rows if
                       r["variant"] == variant and r["evaluated"]}
                successful = ids if successful is None else successful & ids
            paired = [r for r in all_pred_rows if r["evaluated"] and
                      r["event_utc"] in successful]
            write_csv(fold_out/f"fold_{fold:02d}_paired_events.csv", paired)
        for j, (event, stored) in enumerate(centered_diagnostics.items()):
            traces = {key: value for key, value in stored.items() if key != "row"}
            if traces:
                save_centered_diagnostic(
                    fold_out/f"fold_{fold:02d}_diagnostic_centered_{j+1:02d}.png",
                    stored["row"], traces
                )
        for j, (event, stored) in enumerate(plot_results.items()):
            # Each variant has its own fold-normalizer; only use respective norm.
            fig, axes = plt.subplots(2, 1, figsize=(11, 7), sharex=True)
            for variant in variants:
                if variant not in stored:
                    continue
                res, norm = stored[variant]
                if res is None:
                    continue
                t, x = res["clip"]
                ts = t + res["center_offset_s"]
                axes[0].plot(ts, res["curve"], alpha=.5, label=f"{variant} clasificación")
                axes[0].plot(ts, res["tanh"], ls="--", label=f"{variant} tanh")
                axes[0].axvline(res["pred_s"]-event, ls=":", label=f"{variant} MPB")
                field = res["reconstructed_field"]*norm["scale"]+norm["mean"]
                axes[1].plot(ts, field[:, 3], label=f"{variant} |B| reconstruido", alpha=.8)
                if variant == variants[0]:
                    axes[1].plot(ts, x[:, 3], color="k", alpha=.3, label="|B| observado")
            axes[0].axvline(0, color="k", ls="--", label="MPB catálogo")
            axes[1].axvline(0, color="k", ls="--")
            axes[0].set_ylabel("Clase continua [-1,1]")
            axes[1].set_ylabel("|B| (nT)")
            for ax in axes:
                ax.legend(fontsize=8)
                ax.set_xlabel("Tiempo respecto de MPB catálogo (s)")
            fig.tight_layout()
            fig.savefig(fold_out/f"fold_{fold:02d}_example_{j+1:02d}.png", dpi=160)
            plt.close(fig)
        del blocks
    (output/"summary.json").write_text(json.dumps({"configuration": vars(args),
        "results": summary}, default=str, indent=2)+"\n", encoding="utf-8")
    write_csv(output/"summary.csv", summary)
    print(f"Resultados: {output}", flush=True)


if __name__ == "__main__":
    main()