#!/usr/bin/env python3
"""Bounded real-data preprocessing pilot: one CV-training and one CV-val event.

DRY RUN by default. --execute:
  1. Queries one 70-minute BigQuery block for a training-fold MPB event.
  2. Fits channel normalization exclusively on this TRAINING block.
  3. Queries one held-out validation event independently.
  4. Reuses train normalization for both, then builds native-time padded batches.

Data are kept in RAM, no measurement CSV/NPZ files are written.
This is a wiring/sanity pilot: stats from ONE train event are NOT valid global
fold statistics, no CNN is trained, and negative labels remain provisional.
"""
from __future__ import annotations

import argparse
import sys
from datetime import timedelta
from importlib import import_module
from pathlib import Path

import numpy as np
from google.cloud import bigquery

ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

batching = import_module("scripts.4_preprocessing.4a_native_batches")
FoldChannelNormalizer = batching.FoldChannelNormalizer
collate_native_windows = batching.collate_native_windows
masked_reconstruction_mse = batching.masked_reconstruction_mse
weighted_proximity_mse = batching.weighted_proximity_mse

windowing = import_module("scripts.3_windowing.3a_windowing")
WindowStats = windowing.WindowStats
iter_native_windows = windowing.iter_native_windows

bq = import_module("scripts.1_bigquery.1c_bigquery_event_pilot")
pilot = import_module("scripts.3_windowing.3b_bigquery_window_pilot")


def options():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument(
        "--manifest", type=Path, default=Path("data/splits/cv4_monthly_manifest.csv")
    )
    p.add_argument("--fold", type=int, choices=(1, 2, 3, 4), default=1)
    p.add_argument("--instrument", choices=("mag", "swea"), default="mag")
    p.add_argument("--train-index", type=int, default=1)
    p.add_argument("--val-index", type=int, default=1)
    p.add_argument("--sigma-seconds", type=float, default=60)
    p.add_argument("--max-query-gib", type=float, default=1)
    p.add_argument("--max-total-gib", type=float, default=1)
    p.add_argument("--max-returned-rows", type=int, default=25000)
    p.add_argument("--execute", action="store_true")
    args = p.parse_args()
    if (
        min(args.train_index, args.val_index) < 1
        or args.sigma_seconds <= 0 or args.max_query_gib <= 0
        or args.max_total_gib <= 0 or args.max_returned_rows < 1
    ):
        p.error("Require positive event indices and safety limits.")
    return args


def choose_event(records, fold, instrument, *, is_train, index):
    filtered = [
        row for row in records
        if (row["fold"] != fold if is_train else row["fold"] == fold)
        and row[f"{instrument.lower()}_available"] == "1"
    ]
    if index > len(filtered):
        raise ValueError(f"Event index {index} exceeds {len(filtered)} available events.")
    return filtered[index - 1]


def fetch(client, plan, max_bytes, max_rows):
    event, instrument, sql, start, end, _estimate = plan
    result = client.query(
        sql,
        job_config=bq.query_config(
            start, end, dry_run=False, max_bytes=max_bytes
        ),
    )
    rows = []
    for record in result.result(page_size=250):
        rows.append(record)
        if len(rows) > max_rows:
            raise MemoryError("BigQuery row cap exceeded.")
    if instrument == "MAG":
        times, matrix, energies = pilot.read_mag(rows)
    else:
        times, matrix, energies = pilot.read_swea(rows)
    del rows
    return times, matrix, energies


def window_sample(times, x, event, label_times, sigma):
    stats = WindowStats()
    windows = list(iter_native_windows(
        times, x,
        center_label_s=event["time"].timestamp(),
        split_labels_s=label_times,
        sigma_seconds=sigma,
        stats=stats,
    ))
    if not windows:
        raise ValueError(
            f"No valid windows for {event['time'].isoformat()}; "
            f"rejections: {stats}"
        )
    nearest = min(range(len(windows)),
                  key=lambda i: abs(windows[i].offset_from_nearest_mpb_s))
    indices = sorted({0, len(windows)//2, len(windows)-1, nearest})
    selected = [windows[i] for i in indices]
    return stats, selected


def main():
    args = options()
    modality = args.instrument.upper()
    rows = pilot.read_manifest(args.manifest)
    train = choose_event(
        rows, args.fold, modality, is_train=True, index=args.train_index
    )
    val = choose_event(
        rows, args.fold, modality, is_train=False, index=args.val_index
    )
    labels_train = np.array(
        [row["time"].timestamp() for row in rows if row["fold"] != args.fold],
        dtype=np.float64
    )
    labels_val = np.array(
        [row["time"].timestamp() for row in rows if row["fold"] == args.fold],
        dtype=np.float64
    )
    sql = (bq.MAG_SQL.format(mag=bq.MAG)
           if modality == "MAG"
           else bq.SWEA_SQL.format(swea=bq.SWEA, axes=bq.AXES))
    client = bigquery.Client(project=bq.PROJECT)
    bq.validate_schema(client)
    plans = []
    print(f"Fold {args.fold}, instrument={modality}; training-only scaler PILOT.")
    for name, event in (("TRAIN", train), ("VAL", val)):
        center = event["time"]
        start = center-timedelta(minutes=35)
        end = center+timedelta(minutes=35)
        estimate_job = client.query(
            sql, job_config=bq.query_config(start, end, dry_run=True, max_bytes=1)
        )
        estimate = int(estimate_job.total_bytes_processed or 0)
        plans.append((event, modality, sql, start, end, estimate))
        print(
            f"  {name}: label={center.isoformat()}, catalog={event['source_group']}, "
            f"source fold={event['fold']}, est={estimate/bq.GIB:.5f} GiB"
        )
    total = sum(p[-1] for p in plans)
    print(f"Total estimate: {total/bq.GIB:.5f} GiB")
    if not args.execute:
        print("Dry run only; no MAVEN measurements transferred.")
        return

    cap = int(args.max_query_gib*bq.GIB)
    if any(p[-1] > cap for p in plans) or total > args.max_total_gib*bq.GIB:
        raise SystemExit("Query estimate exceeds BigQuery safety limits.")

    print("Loading training block into RAM...")
    t_train, x_train, e_train = fetch(
        client, plans[0], cap, args.max_returned_rows
    )
    if not len(t_train):
        raise ValueError("No training measurements.")
    norm = FoldChannelNormalizer(modality, held_out_fold=args.fold)
    clean_train = np.isfinite(x_train).all(axis=1)
    if modality == "SWEA":
        clean_train &= (x_train >= 0).all(axis=1)
    norm.update(x_train[clean_train], source_fold=train["fold"])
    norm.freeze()
    print(
        f"  Training epochs for normalization: {norm.count}/{len(x_train)}; "
        f"channels={norm.n_features}"
    )
    print(
        f"  Train-only feature mean: {np.array2string(norm.mean, precision=3)}"
    )
    print(
        f"  Train-only feature std:  {np.array2string(norm.scale, precision=3)}"
    )
    s_train, w_train = window_sample(
        t_train, x_train, train, labels_train, args.sigma_seconds
    )
    b_train = collate_native_windows(w_train, normalizer=norm)
    print(
        f"  TRAIN windows: {s_train.accepted}/{s_train.candidates}, "
        f"example batch tensor={tuple(b_train.x.shape)}, "
        f"time-mask valid={int(b_train.time_mask.sum())}, "
        f"Gaussian y={np.round(b_train.y.numpy().ravel(), 3).tolist()}"
    )
    del t_train, x_train, w_train, b_train

    print("Loading held-out validation block into RAM...")
    t_val, x_val, e_val = fetch(
        client, plans[1], cap, args.max_returned_rows
    )
    if modality == "SWEA" and (
        e_train is None or e_val is None or
        e_train.shape != e_val.shape or
        not np.allclose(e_train, e_val, rtol=1e-6, atol=1e-6)
    ):
        raise ValueError(
            "Training and validation SWEA energy grids differ; "
            "require explicit grid alignment before batching."
        )
    s_val, w_val = window_sample(
        t_val, x_val, val, labels_val, args.sigma_seconds
    )
    batch = collate_native_windows(w_val, normalizer=norm)
    print(
        f"  VAL windows: {s_val.accepted}/{s_val.candidates}, "
        f"batch tensor={tuple(batch.x.shape)}, "
        f"time-mask valid={int(batch.time_mask.sum())}, "
        f"Gaussian y={np.round(batch.y.numpy().ravel(), 3).tolist()}"
    )
    print(
        "  Sanity losses on a zero-output baseline (NOT trained): "
        f"reconstruction={masked_reconstruction_mse(batch.x*0, batch).item():.4g}, "
        f"weighted proximity={weighted_proximity_mse(batch.y*0, batch).item():.4g}"
    )
    print(
        "Done: applied training-only normalization to validation data in RAM; "
        "no scientific measurements or arrays saved to disk."
    )
    print("Next: stream ALL unique training epochs to fit stable fold-level statistics.")


if __name__ == "__main__":
    main()
