#!/usr/bin/env python3
"""Seventh experiment: real MAVEN MAG+SWEA MPB plasma-regime classifier.

Two native-cadence sensor encoders, two auxiliary autoencoders, a region
classification head and a binary MPB transition head. Weak labels come
from chronologically paired catalog MPB/BS annotations, with unknown
intervals explicitly excluded. Distinct outer-month folds and inner-month
checkpoint monitors; no outer-validation model selection.

No measurements are invented. Previously cached MAG is reused, missing
SWEA spectra are queried with bounded BigQuery dry-runs, then cached.
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
previous = import_module("scripts.5_training.5a_train_mag_cv")

from src.data.mag_event_cache import MagEventCache
from src.data.swea_event_cache import (
    SweaEventCache, ENERGY_EV, energy_features
)
from src.data.mpb_weak_regions import (
    candidate_training_grid, candidate_evaluation_grid
)
from src.data.mpb_dual_windows import (
    extract_clip, fit_unique_epochs, collate_dual
)
from src.detection.mag_swea_region_model import (
    MagSweaRegionAutoencoder, masked_reconstruction_loss
)


OUT = Path("results/metrics/mag_swea_regions_v1")
MANIFEST = Path("data/splits/cv4_monthly_manifest.csv")
MAG_CACHE = Path("data/cache/mag_events")
SWEA_CACHE = Path("data/cache/swea_events")
WINDOW_SUPPORT_MIN = 35
GIB = 1024 ** 3
SWEA_LOW_EV = 25.
SWEA_HIGH_EV = 330.


def args_parser():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--manifest", type=Path, default=MANIFEST)
    p.add_argument("--output-dir", type=Path, default=OUT)
    p.add_argument("--mag-cache-dir", type=Path, default=MAG_CACHE)
    p.add_argument("--swea-cache-dir", type=Path, default=SWEA_CACHE)
    p.add_argument("--refresh-swea-cache", action="store_true")
    p.add_argument("--folds", choices=("all", "1", "2", "3", "4"),
                   default="all")
    p.add_argument("--pilot-events-per-fold", type=int, default=0,
                   help="0 uses all joint MAG/SWEA catalog events.")
    p.add_argument("--max-epochs", type=int, default=100)
    p.add_argument("--min-epochs", type=int, default=10)
    p.add_argument("--patience", type=int, default=8)
    p.add_argument("--batch-size", type=int, default=16)
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--learning-rate", type=float, default=0.0003)
    p.add_argument("--reconstruction-weight", type=float, default=0.15)
    p.add_argument("--region-weight", type=float, default=0.5)
    p.add_argument("--evaluation-stride-s", type=float, default=32.)
    p.add_argument("--torch-threads", type=int, default=2)
    p.add_argument("--max-query-gib", type=float, default=2.)
    p.add_argument("--max-total-gib", type=float, default=150.)
    p.add_argument("--max-mag-rows", type=int, default=25000)
    p.add_argument("--max-swea-rows", type=int, default=2000)
    p.add_argument("--resume", action="store_true")
    p.add_argument("--execute", action="store_true",
                   help="Actual BigQuery measurement access and training.")
    args = p.parse_args()
    positive = (
        args.max_epochs, args.min_epochs, args.patience, args.batch_size,
        args.learning_rate, args.evaluation_stride_s, args.torch_threads,
        args.max_query_gib, args.max_total_gib,
        args.max_mag_rows, args.max_swea_rows
    )
    if (min(positive) <= 0 or args.min_epochs > args.max_epochs or
            args.pilot_events_per_fold < 0 or
            args.reconstruction_weight < 0 or args.region_weight < 0):
        p.error("Invalid positive parameters or epoch limits.")
    return args


def chosen_events(rows, fold, *, pilot_count):
    selected = [
        r for r in rows if r["fold"] == fold and
        r["mag_available"] == "1" and r["swea_available"] == "1"
    ]
    if pilot_count and pilot_count < len(selected):
        # Spacing across months is more useful than taking contiguous days.
        ids = np.linspace(0, len(selected)-1, pilot_count).round().astype(int)
        selected = [selected[int(i)] for i in np.unique(ids)]
    return selected


def sha256_file(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def query_config(start, end, *, dry_run, max_bytes):
    config = bq.query_config(
        start, end, dry_run=dry_run, max_bytes=max_bytes
    )
    # Existing 1c pilot defaults to 30--100 eV. V7 requests 25--330 eV
    # from source so the 30--300 eV FIXED grid can be interpolated within
    # genuine measured spectral channels, without extrapolation.
    rest = [
        p for p in config.query_parameters
        if p.name not in ("energy_min_ev", "energy_max_ev")
    ]
    config.query_parameters = rest + [
        bigquery.ScalarQueryParameter(
            "energy_min_ev", "FLOAT64", SWEA_LOW_EV
        ),
        bigquery.ScalarQueryParameter(
            "energy_max_ev", "FLOAT64", SWEA_HIGH_EV
        ),
    ]
    return config


def event_bounds(row):
    t = row["time"]
    return t-timedelta(minutes=WINDOW_SUPPORT_MIN), (
        t+timedelta(minutes=WINDOW_SUPPORT_MIN)
    )


def query_sources():
    return {
        "MAG": bq.MAG_SQL.format(mag=bq.MAG),
        "SWEA": bq.SWEA_SQL.format(swea=bq.SWEA, axes=bq.AXES)
    }


def load_event_blocks(rows, args):
    """Cache first; dry-run all misses BEFORE ANY billable query."""
    sqls = query_sources()
    caches = {
        "MAG": MagEventCache(
            args.mag_cache_dir, project=bq.PROJECT,
            table=bq.MAG, sql=sqls["MAG"]
        ),
        "SWEA": SweaEventCache(
            args.swea_cache_dir, project=bq.PROJECT,
            table=bq.SWEA+" + "+bq.AXES, sql=sqls["SWEA"]
        )
    }
    events = {r["time"]: r for r in rows}
    blocks = {t: {} for t in events}
    pending = []
    hits = defaultdict(int)
    for row in sorted(events.values(), key=lambda r: r["time"]):
        start, end = event_bounds(row)
        for sensor in ("MAG", "SWEA"):
            hit = None
            if sensor != "SWEA" or not args.refresh_swea_cache:
                hit = caches[sensor].load(
                    row["time"], start, end,
                    max_rows=(args.max_mag_rows if sensor == "MAG"
                              else args.max_swea_rows)
                )
            if hit is None:
                pending.append((row, sensor, start, end))
            else:
                blocks[row["time"]][sensor] = hit
                hits[sensor] += 1
    print(
        f"Cache: MAG {hits['MAG']}/{len(events)}; "
        f"SWEA {hits['SWEA']}/{len(events)}; "
        f"missing source intervals={len(pending)}.", flush=True
    )
    client = None
    plans, total = [], 0
    if pending:
        client = bigquery.Client(project=bq.PROJECT)
        bq.validate_schema(client)
        for i, (row, sensor, start, end) in enumerate(pending, 1):
            sql = sqls[sensor]
            dry = client.query(
                sql, job_config=query_config(
                    start, end, dry_run=True, max_bytes=1
                )
            )
            estimate = int(dry.total_bytes_processed or 0)
            if estimate > args.max_query_gib * GIB:
                raise RuntimeError(
                    f"{sensor} {row['time']}: dry-run exceeds per-query "
                    f"{args.max_query_gib:g} GiB cap."
                )
            total += estimate
            if total > args.max_total_gib * GIB:
                raise RuntimeError(
                    f"BigQuery estimated {total/GIB:.3f} GiB, exceeds "
                    f"{args.max_total_gib:g} GiB cumulative cap."
                )
            plans.append((row, sensor, start, end, sql, estimate))
            if i % 50 == 0 or i == len(pending):
                print(
                    f"SOURCE DRY-RUN {i}/{len(pending)}, "
                    f"estimated total {total/GIB:.3f} GiB.", flush=True
                )
    else:
        print("Complete MAG/SWEA cache: no BigQuery client or queries.", flush=True)
    if not args.execute:
        print("DRY RUN: no measurements fetched, no training.")
        return None

    for i, (row, sensor, start, end, sql, _) in enumerate(plans, 1):
        job = client.query(
            sql, job_config=query_config(
                start, end, dry_run=False,
                max_bytes=int(args.max_query_gib * GIB)
            )
        )
        result = []
        cap = args.max_mag_rows if sensor == "MAG" else args.max_swea_rows
        for record in job.result(page_size=250):
            result.append(record)
            if len(result) > cap:
                raise MemoryError(f"{sensor} BigQuery row cap exceeded.")
        if sensor == "MAG":
            ts, values, _ = pilot.read_mag(result)
        else:
            ts, values = energy_features(result)
        del result
        caches[sensor].save(
            row["time"], start, end, ts, values
        )
        blocks[row["time"]][sensor] = (ts, values)
        if i % 50 == 0 or i == len(plans):
            print(f"FETCHED {i}/{len(plans)} sensor blocks.", flush=True)
    for event_time, data in blocks.items():
        if set(data) != {"MAG", "SWEA"}:
            raise RuntimeError(f"Missing source data for {event_time}.")
    return blocks


def safe_json(path, obj):
    path.parent.mkdir(parents=True, exist_ok=True)
    temp = path.with_suffix(path.suffix+".tmp")
    temp.write_text(json.dumps(obj, indent=2, sort_keys=True),
                    encoding="utf-8")
    os.replace(temp, path)


def safe_checkpoint(path, data):
    path.parent.mkdir(parents=True, exist_ok=True)
    temp = path.with_suffix(path.suffix+".tmp")
    torch.save(data, temp)
    os.replace(temp, path)


def write_csv(path, rows):
    path.parent.mkdir(parents=True, exist_ok=True)
    if not rows:
        raise ValueError("Cannot write empty event CV results.")
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)


def make_signature(args, fold, fit, monitor, val):
    params = {
        "model": "v7-dual-native-regime-transitions-v1",
        "manifest_sha": sha256_file(args.manifest),
        "fold": fold,
        "fit": [r["time"].isoformat() for r in fit],
        "monitor": [r["time"].isoformat() for r in monitor],
        "val": [r["time"].isoformat() for r in val],
        "spectral_energy_ev": ENERGY_EV.tolist(),
        "max_epochs": args.max_epochs,
        "min_epochs": args.min_epochs,
        "patience": args.patience, "batch_size": args.batch_size,
        "lr": args.learning_rate, "seed": args.seed,
        "region_weight": args.region_weight,
        "reconstruction_weight": args.reconstruction_weight,
        "evaluation_stride_s": args.evaluation_stride_s,
    }
    return hashlib.sha256(json.dumps(params, sort_keys=True).encode()).hexdigest()


def gather_candidates(
    events, blocks, all_mpb_s, all_bs_s, *, evaluate=False,
    stride_s=32.
):
    """Build only verified measured dual clips; retain explicit skip counts."""
    records = {}
    skipped = defaultdict(int)
    for row in events:
        t = row["time"]
        mag, swea = blocks[t]["MAG"], blocks[t]["SWEA"]
        grid = (
            candidate_evaluation_grid(
                row, all_mpb_s, all_bs_s, step_s=stride_s
            ) if evaluate else candidate_training_grid(
                row, all_mpb_s, all_bs_s
            )
        )
        prepared = []
        for candidate in grid:
            clip = extract_clip(mag, swea, candidate.center_s)
            if clip is None:
                skipped["no_joint_coverage"] += 1
                continue
            if not evaluate and candidate.transition is None:
                skipped[candidate.kind] += 1
                continue
            prepared.append((candidate, clip))
        if prepared:
            records[t] = prepared
        else:
            skipped["no_candidates_event"] += 1
    return records, dict(skipped)


def classification_losses(model, batch, args):
    out = model(
        batch["mag"], batch["mag_mask"], batch["mag_time"],
        batch["swea"], batch["swea_mask"], batch["swea_time"]
    )
    transition = torch.nn.functional.binary_cross_entropy_with_logits(
        out["transition_logit"], batch["transition"]
    )
    region = torch.nn.functional.binary_cross_entropy_with_logits(
        out["region_logits"], batch["regions"]
    )
    mr = masked_reconstruction_loss(
        out["mag_reconstruction"], batch["mag"], batch["mag_mask"]
    )
    sr = masked_reconstruction_loss(
        out["swea_reconstruction"], batch["swea"], batch["swea_mask"]
    )
    total = (transition + args.region_weight*region +
             args.reconstruction_weight*(mr+sr))
    return out, total, {
        "det": float(transition.detach()),
        "region": float(region.detach()),
        "mag_rec": float(mr.detach()),
        "swea_rec": float(sr.detach()),
        "total": float(total.detach())
    }


def entries_from_candidates(candidates, *, allow_unknown=False):
    output = []
    for event_time, pair in candidates.items():
        for candidate, clip in pair:
            if candidate.transition is None and not allow_unknown:
                continue
            y = (candidate.transition if candidate.transition is not None
                 else 0)
            lr = (
                (candidate.left_region, candidate.right_region)
                if candidate.left_region is not None else (0, 0)
            )
            output.append((event_time, candidate, clip, y, lr))
    return output


def select_epoch_examples(fit_candidates, epoch, seed):
    """Up to 2 positives+2 controls/event, avoiding massive negative skew."""
    rng = random.Random(seed+epoch)
    examples = []
    for event, pairs in fit_candidates.items():
        positive = [(event, c, x, c.transition,
                     (c.left_region, c.right_region))
                    for c, x in pairs if c.transition == 1]
        negative = [(event, c, x, c.transition,
                     (c.left_region, c.right_region))
                    for c, x in pairs if c.transition == 0]
        if not positive or not negative:
            continue
        examples.extend(rng.sample(positive, min(2, len(positive))))
        examples.extend(rng.sample(negative, min(2, len(negative))))
    rng.shuffle(examples)
    return examples


def batch_for_examples(examples, norm, device):
    return collate_dual(
        [(item[2], item[3], item[4]) for item in examples],
        norm, device
    )


def train_epoch(model, optimizer, candidates, norm, args, device, epoch, fold):
    selected = select_epoch_examples(
        candidates, epoch, seed=args.seed + fold*10000
    )
    if not selected:
        raise ValueError("No usable positive/control candidates in TRAIN.")
    model.train()
    totals = defaultdict(float)
    steps = 0
    for start in range(0, len(selected), args.batch_size):
        batch = batch_for_examples(
            selected[start:start+args.batch_size], norm, device
        )
        optimizer.zero_grad(set_to_none=True)
        _, loss, log = classification_losses(model, batch, args)
        if not bool(torch.isfinite(loss)):
            raise FloatingPointError("Nonfinite seventh loss.")
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        optimizer.step()
        steps += 1
        for k, v in log.items():
            totals[k] += v
    return {k: totals[k]/steps for k in totals} | {
        "n_examples": len(selected)
    }


@torch.no_grad()
def monitor_epoch(model, candidates, norm, args, device):
    entries = entries_from_candidates(candidates)
    if not entries:
        raise ValueError("No usable inner-monitor candidates.")
    model.eval()
    acc = defaultdict(float)
    detections, truths = [], []
    total = 0
    for start in range(0, len(entries), args.batch_size):
        group = entries[start:start+args.batch_size]
        batch = batch_for_examples(group, norm, device)
        output, _, losses = classification_losses(model, batch, args)
        scores = torch.sigmoid(
            output["transition_logit"]
        ).cpu().numpy()
        detections.extend(map(float, scores))
        truths.extend(item[3] for item in group)
        total += len(group)
        for key, val in losses.items():
            acc[key] += val * len(group)
    metrics = {key: val/total for key, val in acc.items()}
    # Model-selection objective is entirely on inner TRAIN-month monitor.
    metrics["select_loss"] = metrics["det"] + args.region_weight*metrics["region"]
    return metrics, np.asarray(detections), np.asarray(truths)


def monitor_threshold(scores, truths):
    """Choose binary candidate threshold using ONLY inner-month data."""
    truth = np.asarray(truths, dtype=bool)
    best, threshold = -1., 0.5
    for value in np.arange(.1, .901, .025):
        selected = scores >= value
        tp = int(np.count_nonzero(selected & truth))
        fp = int(np.count_nonzero(selected & ~truth))
        fn = int(np.count_nonzero(~selected & truth))
        f1 = (2*tp / (2*tp+fp+fn)) if (2*tp+fp+fn) else 0.
        if f1 > best + 1e-9:
            best, threshold = f1, float(value)
    return threshold, best


@torch.no_grad()
def evaluate_fold(model, val_candidates, norm, args, device, threshold, val):
    """Outer fold only: peak localization and weak-label candidate metrics."""
    model.eval()
    records = []
    tp = fp = fn = tn = 0
    region_correct = region_total = 0
    for row in val:
        event_time = row["time"]
        prepared = val_candidates.get(event_time, [])
        scores, offsets, known = [], [], []
        for start in range(0, len(prepared), args.batch_size):
            group = prepared[start:start+args.batch_size]
            batch_items = [
                (clip,
                 candidate.transition if candidate.transition is not None else 0,
                 (candidate.left_region, candidate.right_region)
                 if candidate.left_region is not None else (0, 0))
                for candidate, clip in group
            ]
            batch = collate_dual(batch_items, norm, device)
            output = model(
                batch["mag"], batch["mag_mask"], batch["mag_time"],
                batch["swea"], batch["swea_mask"], batch["swea_time"]
            )
            probabilities = torch.sigmoid(
                output["transition_logit"]
            ).cpu().numpy()
            region_predictions = (
                torch.sigmoid(output["region_logits"]).cpu().numpy() >= .5
            )
            for j, (candidate, _) in enumerate(group):
                score = float(probabilities[j])
                scores.append(score)
                offsets.append(candidate.center_s-event_time.timestamp())
                if candidate.transition is not None:
                    pred = score >= threshold
                    true = candidate.transition == 1
                    tp += int(pred and true)
                    fp += int(pred and not true)
                    fn += int(not pred and true)
                    tn += int(not pred and not true)
                    target = np.array([
                        candidate.left_region, candidate.right_region
                    ], dtype=bool)
                    region_correct += int(np.sum(
                        region_predictions[j] == target
                    ))
                    region_total += 2
        if scores:
            j = int(np.argmax(scores))
            offset = float(offsets[j])
            error = abs(offset)
            peak = scores[j]
        else:
            offset, error, peak = "", "", ""
        records.append({
            "mpb_timestamp_utc": event_time.isoformat(),
            "source_group": row["source_group"],
            "n_windows": len(prepared),
            "peak_offset_s": offset,
            "absolute_error_s": error,
            "peak_score": peak,
            "bs_timestamp_utc": row["bs_timestamp_utc"],
            "weak_candidate_threshold": threshold,
        })
    valid = [float(row["absolute_error_s"]) for row in records
             if row["absolute_error_s"] != ""]
    if not valid:
        raise RuntimeError("No outer validation events have joint sensor clips.")
    errors = np.asarray(valid)
    return {
        "val_total_events": len(val),
        "val_evaluable_events": len(valid),
        "median_abs_peak_error_s": float(np.median(errors)),
        "p90_abs_peak_error_s": float(np.percentile(errors, 90)),
        "fraction_within_60s": float(np.mean(errors <= 60.)),
        "fraction_within_120s": float(np.mean(errors <= 120.)),
        "fraction_within_300s": float(np.mean(errors <= 300.)),
        "candidate_weak_tp": tp, "candidate_weak_fp": fp,
        "candidate_weak_fn": fn, "candidate_weak_tn": tn,
        "candidate_weak_precision": tp/max(1, tp+fp),
        "candidate_weak_recall": tp/max(1, tp+fn),
        "candidate_weak_f1": 2*tp/max(1, 2*tp+fp+fn),
        "weak_region_accuracy": region_correct/max(1, region_total),
        "note": (
            "Localization is CONDITIONAL on a labeled MPB in +/-30 min. "
            "Candidate F1 uses weak catalog labels, not full-orbit precision."
        )
    }, records


def fold_training(fold, selected, blocks, args, all_rows, device):
    train, val = selected["train"], selected["val"]
    fit, monitor, removed = previous.split_inner_monitor_months(
        train, fold, args.seed
    )
    normalizers = {
        instrument: fit_unique_epochs(
            blocks, [r["time"] for r in fit],
            instrument=instrument
        ) for instrument in ("MAG", "SWEA")
    }
    all_mpb_s = np.asarray(
        [r["time"].timestamp() for r in all_rows], dtype=np.float64
    )
    all_bs_s = np.asarray(
        [r["bs_time"].timestamp() for r in all_rows], dtype=np.float64
    )
    fit_pool, fit_skip = gather_candidates(
        fit, blocks, all_mpb_s, all_bs_s
    )
    monitor_pool, monitor_skip = gather_candidates(
        monitor, blocks, all_mpb_s, all_bs_s
    )
    val_pool, val_skip = gather_candidates(
        val, blocks, all_mpb_s, all_bs_s,
        evaluate=True, stride_s=args.evaluation_stride_s
    )
    signature = make_signature(args, fold, fit, monitor, val)
    print(
        f"FOLD {fold}: fit={len(fit)} monitor={len(monitor)} "
        f"outer VAL={len(val)}; usable candidate pools "
        f"{len(fit_pool)}/{len(monitor_pool)}/{len(val_pool)}; "
        f"excluded overlapping TRAIN={removed}.", flush=True
    )
    if not fit_pool or not monitor_pool or not val_pool:
        raise ValueError("Missing usable data pools for fold.")
    torch.manual_seed(args.seed + fold)
    model = MagSweaRegionAutoencoder().to(device)
    optimizer = torch.optim.AdamW(
        model.parameters(), lr=args.learning_rate, weight_decay=1e-3
    )
    folder = args.output_dir
    last_file = folder/f"fold_{fold:02d}_last.pt"
    best_file = folder/f"fold_{fold:02d}_best.pt"
    start_epoch, best_epoch, stale = 0, 0, 0
    best_loss = float("inf")
    history = []
    if last_file.exists():
        if not args.resume:
            raise RuntimeError(
                f"{last_file} exists; use --resume or a different output-dir."
            )
        state = torch.load(
            last_file, map_location=device, weights_only=False
        )
        if state["signature"] != signature:
            raise RuntimeError("Seventh checkpoint signature mismatch.")
        model.load_state_dict(state["model"])
        optimizer.load_state_dict(state["optimizer"])
        start_epoch = int(state["epoch"])
        best_epoch = int(state["best_epoch"])
        best_loss = float(state["best_loss"])
        stale = int(state["stale"])
        history = state["history"]
        print(f"FOLD {fold}: resumed from epoch {start_epoch}.", flush=True)

    for epoch in range(start_epoch+1, args.max_epochs+1):
        if epoch >= args.min_epochs and stale >= args.patience:
            break
        tr = train_epoch(
            model, optimizer, fit_pool, normalizers, args, device,
            epoch, fold
        )
        mon, _, _ = monitor_epoch(
            model, monitor_pool, normalizers, args, device
        )
        if epoch >= args.min_epochs:
            if mon["select_loss"] < best_loss-1e-4:
                best_loss = mon["select_loss"]
                best_epoch = epoch
                stale = 0
                safe_checkpoint(best_file, {
                    "signature": signature, "epoch": epoch,
                    "model": model.state_dict(),
                })
            else:
                stale += 1
        history.append({
            "epoch": epoch, "train_det": tr["det"],
            "train_region": tr["region"],
            "train_mag_rec": tr["mag_rec"],
            "train_swea_rec": tr["swea_rec"],
            "train_total": tr["total"],
            "train_n": tr["n_examples"],
            "val_det": mon["det"],
            "val_region": mon["region"],
            "val_mag_rec": mon["mag_rec"],
            "val_swea_rec": mon["swea_rec"],
            "val_total": mon["total"],
            "val_select": mon["select_loss"],
            "best_epoch": best_epoch, "stale": stale,
        })
        safe_checkpoint(last_file, {
            "signature": signature, "epoch": epoch,
            "model": model.state_dict(),
            "optimizer": optimizer.state_dict(),
            "history": history,
            "best_epoch": best_epoch,
            "best_loss": best_loss, "stale": stale,
        })
        print(
            f"FOLD {fold} EPOCH {epoch}: train_det={tr['det']:.4f} "
            f"train_region={tr['region']:.4f} "
            f"val_det={mon['det']:.4f} "
            f"val_region={mon['region']:.4f} "
            f"val_select={mon['select_loss']:.4f} "
            f"best={best_epoch} stale={stale}", flush=True
        )
        if epoch >= args.min_epochs and stale >= args.patience:
            print(
                f"FOLD {fold}: early stopped at {epoch}; best {best_epoch}.",
                flush=True
            )
            break
    if not best_file.exists():
        raise RuntimeError("No usable v7 best checkpoint.")
    best_state = torch.load(
        best_file, map_location=device, weights_only=False
    )
    if best_state["signature"] != signature:
        raise RuntimeError("Wrong v7 best checkpoint.")
    model.load_state_dict(best_state["model"])
    monitor_metrics, monitor_scores, monitor_truths = monitor_epoch(
        model, monitor_pool, normalizers, args, device
    )
    threshold, monitor_f1 = monitor_threshold(
        monitor_scores, monitor_truths
    )
    metrics, event_rows = evaluate_fold(
        model, val_pool, normalizers, args, device, threshold, val
    )
    result = {
        "schema": 1, "architecture": "dual-mag-swea-weak-regions-v1",
        "fold": fold, "signature": signature,
        "best_epoch": best_epoch, "epochs_completed": len(history),
        "monitor_best_loss": best_loss,
        "monitor_threshold": threshold,
        "monitor_weak_candidate_f1": monitor_f1,
        "train_events": len(fit), "inner_monitor_events": len(monitor),
        "excluded_train_support_overlaps": removed,
        "train_candidate_events": len(fit_pool),
        "monitor_candidate_events": len(monitor_pool),
        "train_skip_counts": fit_skip,
        "monitor_skip_counts": monitor_skip,
        "val_skip_counts": val_skip,
        "normalizers": {
            key: {"mean": value["mean"].tolist(),
                  "scale": value["scale"].tolist(),
                  "n_epochs": value["n_epochs"]}
            for key, value in normalizers.items()
        },
        "training_configuration": {
            "early_stop_min_epochs": args.min_epochs,
            "early_stop_patience": args.patience,
            "region_weight": args.region_weight,
            "reconstruction_weight": args.reconstruction_weight,
            "swea_energy_ev": ENERGY_EV.tolist(),
            "mag_swea_native_time_grid": True,
        },
        "train_history": history,
        **metrics
    }
    safe_json(folder/f"fold_{fold:02d}_metrics.json", result)
    write_csv(folder/f"fold_{fold:02d}_event_metrics.csv", event_rows)
    safe_checkpoint(folder/f"fold_{fold:02d}_final.pt", {
        "schema": 1, "fold": fold, "signature": signature,
        "best_epoch": best_epoch,
        "model": model.state_dict(),
        "normalizers": result["normalizers"],
        "monitor_threshold": threshold,
    })
    print(
        f"FOLD {fold}: median error={metrics['median_abs_peak_error_s']:.1f}s "
        f"evaluable={metrics['val_evaluable_events']}/{len(val)}, "
        f"within 60s={metrics['fraction_within_60s']:.1%}, "
        f"weak candidate F1={metrics['candidate_weak_f1']:.3f}.",
        flush=True
    )
    return result


def main():
    args = args_parser()
    torch.set_num_threads(args.torch_threads)
    torch.manual_seed(args.seed)
    random.seed(args.seed)
    np.random.seed(args.seed)
    device = torch.device("cpu")
    rows = pilot.read_manifest(args.manifest)
    for row in rows:
        if not row.get("bs_timestamp_utc"):
            raise ValueError("V7 needs associated BS timestamps for weak labels.")
        row["bs_time"] = bq.to_utc(row["bs_timestamp_utc"])
    all_requested = (
        [1, 2, 3, 4] if args.folds == "all" else [int(args.folds)]
    )
    fold_selection = {}
    for fold in all_requested:
        val = chosen_events(
            rows, fold, pilot_count=args.pilot_events_per_fold
        )
        train = sorted(
            [event for f in range(1, 5) if f != fold
             for event in chosen_events(
                 rows, f, pilot_count=args.pilot_events_per_fold
             )],
            key=lambda row: row["time"]
        )
        if len(train) < 3 or not val:
            raise ValueError("Insufficient paired MAG/SWEA events for fold.")
        previous.check_disjoint_supports(train, val)
        # Validate inner-month split before touching BigQuery.
        previous.split_inner_monitor_months(train, fold, args.seed)
        fold_selection[fold] = {"train": train, "val": val}
        print(
            f"FOLD {fold}: paired TRAIN={len(train)}, "
            f"outer VAL={len(val)}.", flush=True
        )
    unique = {
        row["time"]: row for groups in fold_selection.values()
        for subset in groups.values() for row in subset
    }
    if args.pilot_events_per_fold:
        print("PILOT SUBSET: results are NOT full cross validation.",
              flush=True)
    else:
        print(
            f"FULL paired MAG+SWEA catalog: {len(unique)} events "
            f"(all four fold count expected 559).", flush=True
        )
    args.output_dir.mkdir(parents=True, exist_ok=True)
    completed, remaining = {}, {}
    for fold, parts in fold_selection.items():
        metrics_path = args.output_dir/f"fold_{fold:02d}_metrics.json"
        final_path = args.output_dir/f"fold_{fold:02d}_final.pt"
        if args.resume and metrics_path.exists() and final_path.exists():
            saved = json.loads(metrics_path.read_text(encoding="utf-8"))
            if saved.get("architecture") != "dual-mag-swea-weak-regions-v1":
                raise ValueError("Different model in seventh output directory.")
            completed[fold] = saved
            print(f"FOLD {fold}: completed and reused.", flush=True)
        else:
            remaining[fold] = parts

    if remaining:
        events = {
            row["time"]: row for parts in remaining.values()
            for subset in parts.values() for row in subset
        }
        blocks = load_event_blocks(list(events.values()), args)
        if blocks is None:
            return
        for fold, parts in remaining.items():
            completed[fold] = fold_training(
                fold, parts, blocks, args, rows, device
            )
    if not args.execute:
        return
    summary = {
        "schema": 1, "instrument": "MAG+SWEA",
        "architecture": "dual-mag-swea-weak-regions-v1",
        "completed_folds": sorted(completed),
        "requested_folds": all_requested,
        "pilot_subset": bool(args.pilot_events_per_fold),
        "candidate_label_warning": (
            "MPB/BS weak region classes are not measured plasma state truth; "
            "unlabeled solar-wind/BS intervals must not be assumed negative."
        ),
        "evaluation_warning": (
            "Peak localization is conditioned on a catalog MPB in +/-30 min. "
            "Candidate F1 uses weak source labels in event neighborhoods, "
            "NOT full orbit precision/recall."
        ),
        "training_configuration": {
            "source": "BigQuery MAVEN MAG + SWEA svy3d",
            "swea_fixed_energy_ev": ENERGY_EV.tolist(),
            "evaluation_stride_s": args.evaluation_stride_s,
            "reconstruction_weight": args.reconstruction_weight,
            "region_weight": args.region_weight,
            "epochs_min": args.min_epochs,
            "patience": args.patience,
        },
        "fold_results": [
            {key: completed[fold].get(key) for key in (
                "fold", "best_epoch", "val_total_events",
                "val_evaluable_events", "median_abs_peak_error_s",
                "fraction_within_60s", "candidate_weak_precision",
                "candidate_weak_recall", "candidate_weak_f1",
                "weak_region_accuracy"
            )} for fold in sorted(completed)
        ]
    }
    safe_json(args.output_dir/"cv_summary.json", summary)
    print("V7 MAG+SWEA: CV summary saved.", flush=True)


if __name__ == "__main__":
    main()
