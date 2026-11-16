#!/usr/bin/env python3
"""Train the REAL-MAVEN supervised MAG autoencoder on 4 held-out-month CV folds.

No synthetic inputs or cached measurement files. The program preflights bounded
BigQuery MAG event queries, fetches each distinct selected event ONCE into RAM,
fits only model weights (never train/val normalization), runs fixed-epoch CV,
and saves only model checkpoints plus small event-level metrics.

Dry run is the default; --execute must be explicitly supplied. A small
--max-train-events / --max-val-events run is a PILOT, NOT full cross-validation.
"""
from __future__ import annotations

import argparse
import csv
import hashlib
import json
import os
import random
import sys
from collections import defaultdict
from datetime import timedelta
from importlib import import_module
from pathlib import Path

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
fitter = import_module("scripts.4_preprocessing.4c_fit_fold_normalizers")
from src.detection.mag_autoencoder import MagSupervisedAutoencoder

DEFAULT_MANIFEST = Path("data/splits/cv4_monthly_manifest.csv")
DEFAULT_NORMALIZERS = Path("results/metrics/fold_normalization")
DEFAULT_OUTPUT = Path("results/metrics/mag_cv")
WINDOW_S = 600.0
CENTER_RANGE_S = 1800.0
SUPPORT_MINUTES = 35.0
GIB = 1024 ** 3


def parser():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--manifest", type=Path, default=DEFAULT_MANIFEST)
    p.add_argument("--normalizers-dir", type=Path, default=DEFAULT_NORMALIZERS)
    p.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT)
    p.add_argument("--folds", choices=("all", "1", "2", "3", "4"), default="all")
    p.add_argument("--epochs", type=int, default=100)
    p.add_argument("--early-stop-patience", type=int, default=10)
    p.add_argument("--early-stop-min-delta", type=float, default=1e-4)
    p.add_argument("--monitor-centers-per-event", type=int, default=21,
                   help="Number of fixed validation centers per event to check per epoch.")
    p.add_argument("--batch-size", type=int, default=16)
    p.add_argument("--train-windows-per-event", type=int, default=24)
    p.add_argument("--val-centers-per-event", type=int, default=121)
    p.add_argument("--max-train-events", type=int, default=0,
                   help="0 = ALL training events; >0 is a subset pilot.")
    p.add_argument("--max-val-events", type=int, default=0,
                   help="0 = ALL validation events; >0 is a subset pilot.")
    p.add_argument("--sigma-seconds", type=float, default=60.)
    p.add_argument("--near-radius-seconds", type=float, default=120.)
    p.add_argument("--near-label-weight", type=float, default=10.)
    p.add_argument("--lambda-detection", type=float, default=1.)
    p.add_argument("--learning-rate", type=float, default=1e-3)
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--torch-threads", type=int, default=2)
    p.add_argument("--max-query-gib", type=float, default=1.)
    p.add_argument("--max-total-gib", type=float, default=20.,
                   help="Hard bound on sum of dry-run estimated bytes in this run.")
    p.add_argument("--max-returned-rows", type=int, default=25000)
    p.add_argument("--device", choices=("auto", "cpu", "cuda"), default="auto")
    p.add_argument("--resume", action="store_true",
                   help="Resume from previous completed epoch; reloads MAVEN into RAM.")
    p.add_argument("--execute", action="store_true",
                   help="Actually read MAVEN and train. Default is estimate only.")
    a = p.parse_args()
    positive = [
        a.epochs, a.early_stop_patience, a.monitor_centers_per_event,
        a.batch_size, a.train_windows_per_event,
        a.val_centers_per_event, a.sigma_seconds, a.near_radius_seconds,
        a.near_label_weight, a.learning_rate, a.max_query_gib,
        a.max_total_gib, a.max_returned_rows, a.torch_threads
    ]
    if any(v <= 0 for v in positive) or a.near_label_weight < 1:
        p.error("All sizes, time scales, rates, and byte limits must be positive.")
    if (a.max_train_events < 0 or a.max_val_events < 0
            or a.lambda_detection < 0 or a.early_stop_min_delta < 0):
        p.error("Invalid subset counts or detection-loss multiplier.")
    if a.train_windows_per_event < 2:
        p.error("--train-windows-per-event must be >= 2.")
    return a


def choose_events(rows, fold, train, cap, seed):
    eligible = [
        r for r in rows if r["mag_available"] == "1" and
        (r["fold"] != fold if train else r["fold"] == fold)
    ]
    if cap == 0 or cap >= len(eligible):
        return eligible

    # Equal-round-robin over catalog source groups, with deterministic shuffling.
    rng = random.Random(seed + fold * 1009 + (0 if train else 500000))
    groups = defaultdict(list)
    for row in eligible:
        groups[row["source_group"]].append(row)
    for members in groups.values():
        rng.shuffle(members)
    names = sorted(groups)
    selected = []
    while len(selected) < cap:
        changed = False
        for name in names:
            if groups[name] and len(selected) < cap:
                selected.append(groups[name].pop())
                changed = True
        if not changed:
            break
    return sorted(selected, key=lambda row: row["time"])


def check_disjoint_supports(train, val):
    """Reject train/val overlap in the +/-35 min physical measurement supports."""
    times = sorted(r["time"].timestamp() for r in train)
    held = sorted(r["time"].timestamp() for r in val)
    if not times or not held:
        raise ValueError("Each fold needs both training and validation events.")
    j = 0
    for t in held:
        while j + 1 < len(times) and times[j + 1] <= t:
            j += 1
        for k in (j, j + 1):
            if k < len(times) and abs(times[k] - t) <= 2 * SUPPORT_MINUTES * 60:
                raise ValueError("Train/val physical data intervals overlap.")


def fold_normalizer(path, manifest, rows, fold):
    if not path.is_file():
        raise FileNotFoundError(f"Missing COMPLETE normalizer: {path}")
    data = json.loads(path.read_text(encoding="utf-8"))
    if (data.get("schema") != 1 or data.get("instrument") != "MAG"
            or data.get("held_out_fold") != fold
            or int(data.get("n_unique_train_epochs", 0)) < 2):
        raise ValueError(f"Wrong/incomplete MAG normalizer for fold {fold}.")
    intervals, _, _ = fitter.plan_training_days(rows, fold, "MAG")
    expected = fitter.planned_signature(manifest, fold, "MAG", intervals)
    if data.get("signature") != expected:
        raise ValueError("Normalizer and CV manifest disagree; refusing data leakage.")
    mean = np.asarray(data["mean"], dtype=np.float64)
    scale = np.asarray(data["scale"], dtype=np.float64)
    if (mean.shape != (4,) or scale.shape != (4,)
            or not np.isfinite(mean).all()
            or not np.isfinite(scale).all() or np.any(scale <= 0)):
        raise ValueError("MAG normalizer requires 4 finite means and positive scales.")

    normalizer = batching.FoldChannelNormalizer("MAG", held_out_fold=fold)
    normalizer.mean = mean
    normalizer.scale = scale
    normalizer.n_features = 4
    normalizer.count = int(data["n_unique_train_epochs"])
    normalizer.is_frozen = True
    return normalizer, hashlib.sha256(path.read_bytes()).hexdigest()


def signature(args, fold, train, val, normalizer_sha):
    cfg = {
        "schema": 2, "architecture": "MAG-CNN-latent2-v1-earlystop",
        "fold": fold, "normalizer_sha256": normalizer_sha,
        "manifest_sha256": hashlib.sha256(args.manifest.read_bytes()).hexdigest(),
        "train_times": [r["time"].isoformat() for r in train],
        "val_times": [r["time"].isoformat() for r in val],
        "epochs": args.epochs, "batch_size": args.batch_size,
        "train_windows_per_event": args.train_windows_per_event,
        "val_centers_per_event": args.val_centers_per_event,
        "sigma_seconds": args.sigma_seconds,
        "near_radius_seconds": args.near_radius_seconds,
        "near_label_weight": args.near_label_weight,
        "lambda_detection": args.lambda_detection,
        "learning_rate": args.learning_rate, "seed": args.seed,
        "early_stop_patience": args.early_stop_patience,
        "early_stop_min_delta": args.early_stop_min_delta,
        "monitor_centers_per_event": args.monitor_centers_per_event,
    }
    return hashlib.sha256(json.dumps(cfg, sort_keys=True).encode()).hexdigest()


def event_plan(row):
    t0 = row["time"]
    start = t0 - timedelta(minutes=SUPPORT_MINUTES)
    end = t0 + timedelta(minutes=SUPPORT_MINUTES)
    sql = bq.MAG_SQL.format(mag=bq.MAG)
    return (row, "MAG", sql, start, end)


def preflight(client, selected, args):
    """Dry-run EVERY DISTINCT event once, before any billable query starts."""
    unique = {r["time"]: r for fold in selected.values()
              for split in fold.values() for r in split}
    plans = []
    total = 0
    for i, row in enumerate(sorted(unique.values(), key=lambda r: r["time"]), 1):
        event, instrument, sql, start, end = event_plan(row)
        dry = client.query(
            sql, job_config=bq.query_config(
                start, end, dry_run=True, max_bytes=1
            )
        )
        estimate = int(dry.total_bytes_processed or 0)
        if estimate > args.max_query_gib * GIB:
            raise RuntimeError(
                f"Query for {row['time']} exceeds --max-query-gib={args.max_query_gib}."
            )
        total += estimate
        plans.append((event, instrument, sql, start, end, estimate))
        if i % 25 == 0 or i == len(unique):
            print(f"PRE-FLIGHT {i}/{len(unique)} unique events: "
                  f"{total/GIB:.3f} GiB cumulative", flush=True)
    print(f"TOTAL ESTIMATED BigQuery bytes: {total/GIB:.3f} GiB "
          f"(cap {args.max_total_gib:.3f} GiB).", flush=True)
    if total > args.max_total_gib * GIB:
        raise RuntimeError("Cumulative estimated BigQuery bytes exceed the limit. "
                           "No measurement queries were executed.")
    return plans, total


def fetch_ram(client, plans, args):
    """Fetch each event ONCE; science arrays never leave this process."""
    blocks = {}
    for i, plan in enumerate(plans, 1):
        row = plan[0]
        result = client.query(
            plan[2], job_config=bq.query_config(
                plan[3], plan[4], dry_run=False,
                max_bytes=int(args.max_query_gib * GIB)
            )
        )
        rows = []
        for record in result.result(page_size=250):
            rows.append(record)
            if len(rows) > args.max_returned_rows:
                raise MemoryError("Event row cap exceeded. No data saved.")
        times, values, _ = pilot.read_mag(rows)
        del rows
        if len(times) > 1 and np.any(np.diff(times) <= 0):
            raise ValueError(f"Repeated or reversed MAG times for {row['time']}.")
        blocks[row["time"]] = (times, values)
        if i % 25 == 0 or i == len(plans):
            print(f"FETCHED {i}/{len(plans)} event blocks into RAM.",
                  flush=True)
    return blocks


def select_spaced(windows, n):
    if n <= 0 or not windows:
        return []
    count = min(n, len(windows))
    indices = np.linspace(0, len(windows) - 1, count).round().astype(int)
    return [windows[int(i)] for i in sorted(set(indices))]


def event_windows(row, blocks, labels, args, *, train):
    times, values = blocks[row["time"]]
    if len(times) < 3:
        return []
    candidate = list(windowing.iter_native_windows(
        times, values,
        center_label_s=row["time"].timestamp(),
        split_labels_s=labels,
        window_seconds=WINDOW_S,
        center_range_seconds=CENTER_RANGE_S,
        sigma_seconds=args.sigma_seconds,
    ))
    if not candidate:
        return []
    if not train:
        return select_spaced(candidate, args.val_centers_per_event)

    near = [w for w in candidate
            if abs(w.offset_from_nearest_mpb_s) <= args.near_radius_seconds]
    far = [w for w in candidate
           if abs(w.offset_from_nearest_mpb_s) > args.near_radius_seconds]
    n_near = args.train_windows_per_event // 2
    chosen = select_spaced(near, n_near)
    chosen += select_spaced(far, args.train_windows_per_event - len(chosen))
    if len(chosen) < args.train_windows_per_event:
        used = {w.center_s for w in chosen}
        remaining = [w for w in candidate if w.center_s not in used]
        chosen += select_spaced(remaining, args.train_windows_per_event - len(chosen))
    return sorted(chosen, key=lambda w: w.center_s)


def make_batch(windows, normalizer, device):
    batch = batching.collate_native_windows(windows, normalizer=normalizer)
    return batching.NativeBatch(
        x=batch.x.to(device),
        time_mask=batch.time_mask.to(device),
        rel_time_s=batch.rel_time_s.to(device),
        y=batch.y.to(device),
        center_s=batch.center_s,  # timestamps stay on CPU
        label_offset_s=batch.label_offset_s,
    )


def training_epoch(model, optimizer, windows, normalizer, args, device, fold, epoch):
    rng = random.Random(args.seed + fold * 10000 + epoch)
    order = list(range(len(windows)))
    rng.shuffle(order)
    model.train()
    total_rec = total_det = total_loss = steps = 0.
    for pos in range(0, len(order), args.batch_size):
        group = [windows[i] for i in order[pos:pos + args.batch_size]]
        batch = make_batch(group, normalizer, device)
        optimizer.zero_grad(set_to_none=True)
        _, recon, pred = model(batch.x, batch.time_mask, batch.rel_time_s)
        rec = batching.masked_reconstruction_mse(recon, batch)
        det = batching.weighted_proximity_mse(
            pred, batch, near_label_weight=args.near_label_weight
        )
        loss = rec + args.lambda_detection * det
        if not bool(torch.isfinite(loss)):
            raise FloatingPointError("Nonfinite loss; aborting fold.")
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
        optimizer.step()
        total_rec += float(rec.detach())
        total_det += float(det.detach())
        total_loss += float(loss.detach())
        steps += 1
    if not steps:
        raise RuntimeError("No valid training windows.")
    return {"train_loss": total_loss/steps,
            "train_rec": total_rec/steps,
            "train_det": total_det/steps,
            "train_batches": int(steps)}


@torch.no_grad()
def validate_fold(model, val_windows, normalizer, args, device):
    """Localization diagnostics on known MPB events only; NOT precision/F1."""
    model.eval()
    event_rows = []
    weighted_error = weighted_total = 0.
    for row, windows in val_windows:
        if not windows:
            event_rows.append({
                "mpb_timestamp_utc": row["time"].isoformat(),
                "source_group": row["source_group"], "n_windows": 0,
                "peak_offset_s": "", "absolute_error_s": "",
                "peak_score": "", "target_mse_weighted": ""
            })
            continue
        centers, predictions = [], []
        event_werr = event_wtotal = 0.
        for pos in range(0, len(windows), args.batch_size):
            pack = windows[pos:pos+args.batch_size]
            batch = make_batch(pack, normalizer, device)
            _, _, scores = model(batch.x, batch.time_mask, batch.rel_time_s)
            target = batch.y
            weight = 1 + (args.near_label_weight-1)*target
            err = (weight*(scores-target).square()).sum().item()
            wtot = weight.sum().item()
            event_werr += err
            event_wtotal += wtot
            centers.extend(w.center_s for w in pack)
            predictions.extend(float(x) for x in scores.flatten().cpu())
        peak_index = int(np.argmax(predictions))
        offset = centers[peak_index]-row["time"].timestamp()
        event_rows.append({
            "mpb_timestamp_utc": row["time"].isoformat(),
            "source_group": row["source_group"], "n_windows": len(windows),
            "peak_offset_s": float(offset),
            "absolute_error_s": float(abs(offset)),
            "peak_score": predictions[peak_index],
            "target_mse_weighted": event_werr/event_wtotal
        })
        weighted_error += event_werr
        weighted_total += event_wtotal
    good = [r for r in event_rows if r["n_windows"] > 0]
    errors = np.asarray([r["absolute_error_s"] for r in good], dtype=float)
    return {
        "val_total_events": len(event_rows),
        "val_evaluable_events": len(good),
        "val_missing_events": len(event_rows)-len(good),
        "val_weighted_proximity_mse": weighted_error/weighted_total if weighted_total else None,
        "median_abs_peak_error_s": float(np.median(errors)) if len(errors) else None,
        "p90_abs_peak_error_s": float(np.percentile(errors, 90)) if len(errors) else None,
        "fraction_within_60s": float(np.mean(errors <= 60)) if len(errors) else None,
        "fraction_within_120s": float(np.mean(errors <= 120)) if len(errors) else None,
        "note": "Only known labeled MPB events; peak localization, not precision/F1."
    }, event_rows



def monitoring_windows(val_windows, per_event):
    """Fixed, small per-event subset for validation every epoch.

    Every available held-out event is used: this is not an event subset.
    Including both near-label and distant windows avoids evaluating a model
    only on one regime. Full event localization uses all val centers later.
    """
    result = []
    near_count = (per_event + 1) // 2
    for _, windows in val_windows:
        if not windows:
            continue
        near = [w for w in windows if abs(w.offset_from_nearest_mpb_s) <= 180.]
        far = [w for w in windows if abs(w.offset_from_nearest_mpb_s) > 180.]
        chosen = select_spaced(near, near_count)
        chosen.extend(select_spaced(far, per_event - len(chosen)))
        if len(chosen) < per_event:
            used = {w.center_s for w in chosen}
            extra = [w for w in windows if w.center_s not in used]
            chosen.extend(select_spaced(extra, per_event - len(chosen)))
        result.extend(chosen)
    return result


@torch.no_grad()
def evaluate_monitor(model, windows, normalizer, args, device):
    """Fixed validation score used for early stopping (not a test set)."""
    model.eval()
    det_numerator = det_denominator = 0.0
    rec_numerator = rec_denominator = 0.0
    for i in range(0, len(windows), args.batch_size):
        batch = make_batch(windows[i:i+args.batch_size], normalizer, device)
        _, reconstruction, scores = model(
            batch.x, batch.time_mask, batch.rel_time_s
        )
        weights = 1 + (args.near_label_weight-1) * batch.y
        det_numerator += float((weights * (scores - batch.y).square()).sum())
        det_denominator += float(weights.sum())
        # Reconstruction only on observed (unpadded) samples.
        mask = batch.time_mask[:, None, :].to(batch.x.dtype)
        rec_numerator += float(((reconstruction - batch.x).square() * mask).sum())
        rec_denominator += float(mask.sum()) * batch.x.shape[1]
    if det_denominator == 0 or rec_denominator == 0:
        raise ValueError("No evaluable held-out validation windows.")
    return {
        "val_det": det_numerator/det_denominator,
        "val_rec": rec_numerator/rec_denominator,
    }


def save_json(path, object_value):
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    with temporary.open("w", encoding="utf-8") as handle:
        json.dump(object_value, handle, indent=2, sort_keys=True)
    os.replace(temporary, path)


def save_checkpoint(path, object_value):
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(".tmp")
    torch.save(object_value, temporary)
    os.replace(temporary, path)


def save_events_csv(path, rows):
    path.parent.mkdir(parents=True, exist_ok=True)
    fields = ["mpb_timestamp_utc", "source_group", "n_windows",
              "peak_offset_s", "absolute_error_s", "peak_score",
              "target_mse_weighted"]
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fields)
        writer.writeheader()
        writer.writerows(rows)


def train_fold(fold, selected, blocks, normalizer, run_sig, args, device,
               all_manifest_rows):
    train = selected["train"]
    val = selected["val"]
    # Even in a small-event pilot, labels include ALL annotated events from
    # the correct split, so an omitted nearby MPB is not treated as unknown.
    training_labels = np.asarray([
        r["time"].timestamp() for r in all_manifest_rows
        if r["fold"] != fold and r["mag_available"] == "1"
    ], dtype=float)
    validation_labels = np.asarray([
        r["time"].timestamp() for r in all_manifest_rows
        if r["fold"] == fold and r["mag_available"] == "1"
    ], dtype=float)
    print(f"\nFOLD {fold}: prepare {len(train)} TRAIN / {len(val)} VAL events.",
          flush=True)

    training_windows = []
    invalid_train = 0
    for row in train:
        ws = event_windows(row, blocks, training_labels, args, train=True)
        if not ws:
            invalid_train += 1
        training_windows.extend(ws)
    if len(training_windows) < args.batch_size:
        raise RuntimeError("Too few accepted training windows.")

    val_windows = [
        (row, event_windows(row, blocks, validation_labels, args, train=False))
        for row in val
    ]
    print(f"FOLD {fold}: {len(training_windows)} training windows; "
          f"{sum(len(ws) for _, ws in val_windows)} evaluation windows; "
          f"{invalid_train} unusable train events.", flush=True)

    # Make fold initialization independent of execution/resume order.
    torch.manual_seed(args.seed + fold)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(args.seed + fold)
    model = MagSupervisedAutoencoder().to(device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=args.learning_rate)
    checkpoint_path = args.output_dir / f"fold_{fold:02d}_last.pt"
    best_path = args.output_dir / f"fold_{fold:02d}_best.pt"
    final_path = args.output_dir / f"fold_{fold:02d}_final.pt"
    metrics_path = args.output_dir / f"fold_{fold:02d}_metrics.json"
    event_path = args.output_dir / f"fold_{fold:02d}_event_metrics.csv"
    start_epoch = 0
    history = []
    best_val = float("inf")
    best_epoch = 0
    stale_epochs = 0
    monitor_windows = monitoring_windows(val_windows, args.monitor_centers_per_event)
    print(f"FOLD {fold}: early stopping monitors {len(monitor_windows)} "
          f"fixed held-out windows from ALL evaluable VAL events.", flush=True)
    if not monitor_windows:
        raise ValueError("No validation windows available to monitor.")

    if checkpoint_path.exists():
        if not args.resume:
            raise RuntimeError(
                f"{checkpoint_path} already exists. Use --resume or a new output-dir."
            )
        state = torch.load(checkpoint_path, map_location=device, weights_only=False)
        if state["signature"] != run_sig:
            raise RuntimeError("Checkpoint signature mismatch. Use another output-dir.")
        model.load_state_dict(state["model"])
        optimizer.load_state_dict(state["optimizer"])
        start_epoch = int(state["epoch"])
        history = list(state["history"])
        best_val = float(state["best_val"])
        best_epoch = int(state["best_epoch"])
        stale_epochs = int(state["stale_epochs"])
        if not best_path.exists():
            raise RuntimeError("Best checkpoint is missing; refuse unsafe resume.")
        print(f"FOLD {fold}: resumed after epoch {start_epoch}, "
              f"best epoch {best_epoch}.", flush=True)

    for epoch in range(start_epoch+1, args.epochs+1):
        stats = training_epoch(model, optimizer, training_windows, normalizer,
                               args, device, fold, epoch)
        val_stats = evaluate_monitor(
            model, monitor_windows, normalizer, args, device
        )
        candidate = val_stats["val_det"]
        improved = (best_epoch == 0 or
                    candidate < best_val - args.early_stop_min_delta)
        if improved:
            best_val = candidate
            best_epoch = epoch
            stale_epochs = 0
            save_checkpoint(best_path, {
                "schema": 2, "fold": fold, "signature": run_sig,
                "epoch": epoch, "best_val_det": best_val,
                "model": model.state_dict()
            })
        else:
            stale_epochs += 1

        history.append({"epoch": epoch, **stats, **val_stats,
                        "best_epoch_so_far": best_epoch})
        save_checkpoint(checkpoint_path, {
            "schema": 2, "fold": fold, "signature": run_sig,
            "epoch": epoch, "model": model.state_dict(),
            "optimizer": optimizer.state_dict(), "history": history,
            "best_val": best_val, "best_epoch": best_epoch,
            "stale_epochs": stale_epochs
        })
        print(f"FOLD {fold} EPOCH {epoch}/{args.epochs}: "
              f"train_total={stats['train_loss']:.5f} "
              f"train_det={stats['train_det']:.5f} "
              f"val_det={val_stats['val_det']:.5f} "
              f"val_rec={val_stats['val_rec']:.5f} "
              f"best_epoch={best_epoch} stale={stale_epochs}",
              flush=True)
        if stale_epochs >= args.early_stop_patience:
            print(f"FOLD {fold}: early stopping after {epoch} epochs "
                  f"(best epoch {best_epoch}).", flush=True)
            break

    if not best_path.exists():
        raise RuntimeError("Training ended without a best-model checkpoint.")
    best_state = torch.load(best_path, map_location=device, weights_only=False)
    if best_state["signature"] != run_sig:
        raise RuntimeError("Best checkpoint signature mismatch.")
    model.load_state_dict(best_state["model"])
    print(f"FOLD {fold}: evaluating FULL validation at best epoch "
          f"{best_epoch}, not last weights.", flush=True)

    metrics, event_rows = validate_fold(model, val_windows, normalizer, args, device)
    result = {
        "schema": 1, "fold": fold, "signature": run_sig,
        "epochs_max": args.epochs, "epochs_completed": len(history),
        "best_epoch": best_epoch, "best_monitor_val_det": best_val,
        "early_stopped": stale_epochs >= args.early_stop_patience,
        "train_events_selected": len(train),
        "train_events_unusable": invalid_train,
        "train_windows": len(training_windows),
        "val_events_selected": len(val), "train_history": history,
        "pilot_subset": bool(args.max_train_events or args.max_val_events),
        **metrics,
    }
    save_checkpoint(final_path, {
        "schema": 2, "fold": fold, "signature": run_sig,
        "best_epoch": best_epoch, "model": model.state_dict(), "config": {
            "sigma_seconds": args.sigma_seconds, "hidden": 16,
            "latent_channels": 2, "pooling_sigma_s": 90.0,
        }
    })
    save_events_csv(event_path, event_rows)
    save_json(metrics_path, result)
    print(f"FOLD {fold}: median error={metrics['median_abs_peak_error_s']} s; "
          f"evaluable={metrics['val_evaluable_events']}/{len(val)}.",
          flush=True)
    return result


def main():
    args = parser()
    torch.set_num_threads(args.torch_threads)
    torch.manual_seed(args.seed)
    random.seed(args.seed)
    np.random.seed(args.seed)
    if args.device == "cuda" and not torch.cuda.is_available():
        raise SystemExit("CUDA explicitly requested but unavailable.")
    device = torch.device(
        "cuda" if args.device == "cuda" or
        (args.device == "auto" and torch.cuda.is_available()) else "cpu"
    )

    rows = pilot.read_manifest(args.manifest)
    requested = (1, 2, 3, 4) if args.folds == "all" else (int(args.folds),)
    selected = {}
    normalizers, signatures = {}, {}
    for fold in requested:
        train = choose_events(rows, fold, True, args.max_train_events, args.seed)
        val = choose_events(rows, fold, False, args.max_val_events, args.seed)
        check_disjoint_supports(train, val)
        normalizer, normalizer_sha = fold_normalizer(
            args.normalizers_dir / f"fold_{fold:02d}_mag_normalizer.json",
            args.manifest, rows, fold
        )
        selected[fold] = {"train": train, "val": val}
        normalizers[fold] = normalizer
        signatures[fold] = signature(args, fold, train, val, normalizer_sha)
        print(f"FOLD {fold}: {len(train)} TRAIN / {len(val)} VAL, "
              f"normalizer={normalizer.count} unique TRAIN epochs.", flush=True)

    if args.max_train_events or args.max_val_events:
        print("IMPORTANT: SUBSET PILOT; not full 569-event cross-validation.",
              flush=True)
    print(f"Device: {device}; epochs: {args.epochs}; "
          "no MAVEN measurements will be written to disk.", flush=True)

    remaining = {}
    results = {}
    for fold in requested:
        metrics_file = args.output_dir / f"fold_{fold:02d}_metrics.json"
        if (args.resume and metrics_file.exists() and
                (args.output_dir / f"fold_{fold:02d}_final.pt").exists()):
            saved = json.loads(metrics_file.read_text(encoding="utf-8"))
            if saved.get("signature") != signatures[fold]:
                raise RuntimeError("Existing fold metrics disagree with this run.")
            print(f"FOLD {fold} already complete; skipping.", flush=True)
            results[fold] = saved
        else:
            remaining[fold] = selected[fold]

    if not remaining:
        print("All requested folds already completed.")
        return

    client = bigquery.Client(project=bq.PROJECT)
    bq.validate_schema(client)
    plans, _ = preflight(client, remaining, args)
    if not args.execute:
        print("DRY RUN ONLY. Supply --execute to fetch MAVEN and train.")
        return

    blocks = fetch_ram(client, plans, args)
    try:
        for fold in remaining:
            results[fold] = train_fold(
                fold, selected[fold], blocks, normalizers[fold],
                signatures[fold], args, device, rows
            )
            # Existing data are held in RAM throughout the run so each
            # distinct selected event is fetched from BigQuery exactly once.
    finally:
        blocks.clear()

    summary = {
        "schema": 1,
        "instrument": "MAG",
        "requested_folds": list(requested),
        "completed_folds": sorted(results),
        "pilot_subset": bool(args.max_train_events or args.max_val_events),
        "fold_results": [{
            "fold": f,
            "median_abs_peak_error_s": results[f].get("median_abs_peak_error_s"),
            "fraction_within_60s": results[f].get("fraction_within_60s"),
            "val_evaluable_events": results[f].get("val_evaluable_events"),
            "val_total_events": results[f].get("val_total_events")
        } for f in sorted(results)],
        "caution": "Only known labeled events; not precision/recall/F1 or calibration."
    }
    save_json(args.output_dir / "cv_summary.json", summary)
    print("DONE: completed all requested folds. CV summary saved.", flush=True)


if __name__ == "__main__":
    main()
