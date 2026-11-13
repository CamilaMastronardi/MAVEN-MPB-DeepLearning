#!/usr/bin/env python3
"""Fit a separate FULL-TRAINING-FOLD MAG/SWEA normalizer, incrementally.

Read only event neighborhoods for the selected CV TRAIN months, via a single
partition-pruned BigQuery query per UTC day. Overlapping +/-35-minute event
supports are unioned first: each native measurement timestamp contributes
once. SWEA is reduced in BigQuery to the original-grid 30–100 eV channels.

Default: DRY RUN of the next 3 UTC days, no scientific measurements fetched.
--execute: fetch those days (with BigQuery byte caps), stream them through the
training-only Welford scaler, atomically checkpoint only small statistics.
Repeat the same --execute command until the fold is complete, or use
--max-days 0 to select every remaining day, subject to total byte caps.

Scientific measurements are held in RAM only. Checkpoint JSON contains
counts, means, M2, energies, day IDs and configuration, NEVER measurement rows.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import os
import sys
from collections import defaultdict
from datetime import date, datetime, timedelta, timezone
from importlib import import_module
from pathlib import Path

import numpy as np
from google.cloud import bigquery

ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

bq = import_module("scripts.1_bigquery.1c_bigquery_event_pilot")
pilot = import_module("scripts.3_windowing.3b_bigquery_window_pilot")
FoldChannelNormalizer = import_module(
    "scripts.4_preprocessing.4a_native_batches"
).FoldChannelNormalizer

MANIFEST = Path("data/splits/cv4_monthly_manifest.csv")
OUTPUT = Path("results/metrics/fold_normalization")
SUPPORT_MINUTES = 35


def parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--manifest", type=Path, default=MANIFEST)
    p.add_argument("--fold", type=int, choices=(1, 2, 3, 4), default=1)
    p.add_argument("--instrument", choices=("mag", "swea"), default="mag")
    p.add_argument(
        "--max-days", type=int, default=3,
        help="Next N unprocessed UTC dates; 0 means all remaining dates."
    )
    p.add_argument("--max-query-gib", type=float, default=1.0)
    p.add_argument("--max-total-gib", type=float, default=2.0)
    p.add_argument(
        "--max-returned-rows", type=int, default=120_000,
        help="Maximum number of BigQuery records per UTC date."
    )
    p.add_argument("--output-dir", type=Path, default=OUTPUT)
    p.add_argument("--execute", action="store_true")
    return p


def merge_intervals(intervals):
    """Return sorted, disjoint half-open UTC intervals, merging intersections."""
    merged = []
    for start, end in sorted(intervals):
        if not start < end or start.tzinfo is None or end.tzinfo is None:
            raise ValueError("Intervals must be nonempty and timezone-aware.")
        if merged and start <= merged[-1][1]:
            merged[-1] = (merged[-1][0], max(end, merged[-1][1]))
        else:
            merged.append((start, end))
    return merged


def plan_training_days(manifest_rows, held_out_fold, instrument):
    """Select 70-min event neighborhoods and union by UTC day (no overlap)."""
    supports = defaultdict(list)
    source_folds = defaultdict(set)
    n_events = 0
    for event in manifest_rows:
        if event["fold"] == held_out_fold:
            continue
        if event[f"{instrument.lower()}_available"] != "1":
            continue
        source_fold = int(event["fold"])
        assert source_fold != held_out_fold
        n_events += 1
        center = event["time"]
        start = center - timedelta(minutes=SUPPORT_MINUTES)
        end = center + timedelta(minutes=SUPPORT_MINUTES)
        midnight = datetime.combine(start.date(), datetime.min.time(),
                                    tzinfo=timezone.utc)
        while midnight < end:
            next_midnight = midnight + timedelta(days=1)
            lo, hi = max(start, midnight), min(end, next_midnight)
            if lo < hi:
                key = midnight.date()
                supports[key].append((lo, hi))
                source_folds[key].add(source_fold)
            midnight = next_midnight
    return (
        {day: merge_intervals(intervals) for day, intervals in sorted(supports.items())},
        source_folds, n_events
    )


def planned_signature(manifest_path, fold, instrument, day_intervals):
    """Invalidate checkpoints if manifest CONTENT or requested support changes."""
    h = hashlib.sha256()
    h.update(manifest_path.read_bytes())
    h.update(f"fold={fold},instrument={instrument},support={SUPPORT_MINUTES},".encode())
    h.update(f"band={bq.SWEA_ENERGY_MIN_EV}-{bq.SWEA_ENERGY_MAX_EV}".encode())
    for day, intervals in day_intervals.items():
        h.update(str(day).encode())
        for start, end in intervals:
            h.update(f"{start.isoformat()}/{end.isoformat()}".encode())
    return h.hexdigest()


def day_sql_and_config(instrument, day, intervals, *, dry_run, max_bytes):
    """Parameterize one date and a union of bounded time intervals."""
    if not intervals:
        raise ValueError("At least one interval is required.")
    if instrument == "MAG":
        sql = bq.MAG_SQL.format(mag=bq.MAG)
        col = "datetime_utc"
    elif instrument == "SWEA":
        sql = bq.SWEA_SQL.format(swea=bq.SWEA, axes=bq.AXES)
        col = "d.datetime_utc"
    else:
        raise ValueError("Unknown instrument")
    original_filter = (
        f"WHERE DATE({col}) IN UNNEST(@days)\n"
        f"  AND {col} BETWEEN @start AND @end"
    )
    if original_filter not in sql:
        raise ValueError("The referenced BigQuery query changed unexpectedly.")
    time_predicates = [
        f"({col} >= @start_{i} AND {col} < @end_{i})"
        for i in range(len(intervals))
    ]
    sql = sql.replace(
        original_filter,
        f"WHERE DATE({col}) = @day\n  AND ("
        + " OR ".join(time_predicates) + ")"
    )
    params = [bigquery.ScalarQueryParameter("day", "DATE", day)]
    for i, (start, end) in enumerate(intervals):
        params.extend([
            bigquery.ScalarQueryParameter(f"start_{i}", "TIMESTAMP", start),
            bigquery.ScalarQueryParameter(f"end_{i}", "TIMESTAMP", end),
        ])
    if instrument == "SWEA":
        params.extend([
            bigquery.ScalarQueryParameter(
                "energy_min_ev", "FLOAT64", bq.SWEA_ENERGY_MIN_EV
            ),
            bigquery.ScalarQueryParameter(
                "energy_max_ev", "FLOAT64", bq.SWEA_ENERGY_MAX_EV
            ),
        ])
    config = bigquery.QueryJobConfig(
        dry_run=dry_run,
        use_query_cache=not dry_run,
        query_parameters=params,
        **({} if dry_run else {"maximum_bytes_billed": int(max_bytes)}),
    )
    return sql, config


def read_day_data(result, instrument, max_rows):
    """Receive at most one UTC-day result in transient RAM."""
    rows = []
    for row in result.result(page_size=250):
        rows.append(row)
        if len(rows) > max_rows:
            raise MemoryError(f"BigQuery row cap exceeded: {max_rows}")
    if instrument == "MAG":
        timestamps, values, energies = pilot.read_mag(rows)
    else:
        timestamps, values, energies = pilot.read_swea(rows)
    del rows
    if len(timestamps) > 1 and np.any(np.diff(timestamps) <= 0):
        raise ValueError(
            "Repeated or unsorted instrument timestamps within UTC day: "
            "refusing to double-count. Inspect source records first."
        )
    return timestamps, values, energies


def restore_normalizer(checkpoint, instrument, held_out_fold):
    norm = FoldChannelNormalizer(instrument, held_out_fold)
    if checkpoint is None:
        return norm
    if checkpoint["n_epochs"] > 0:
        norm.count = int(checkpoint["n_epochs"])
        norm.mean = np.array(checkpoint["mean"], dtype=np.float64)
        norm.m2 = np.array(checkpoint["m2"], dtype=np.float64)
        norm.n_features = len(norm.mean)
        if len(norm.m2) != norm.n_features:
            raise ValueError("Malformed normalizer checkpoint.")
    return norm


def checkpoint_payload(norm, signature, completed, invalid_count,
                       reference_energies, instrument, fold):
    return {
        "schema": 1,
        "signature": signature,
        "held_out_fold": fold,
        "instrument": instrument,
        "support_minutes_each_side": SUPPORT_MINUTES,
        "completed_days": sorted(completed),
        "n_epochs": norm.count,
        "skipped_invalid_epochs": invalid_count,
        "mean": norm.mean.tolist() if norm.mean is not None else [],
        "m2": norm.m2.tolist() if norm.m2 is not None else [],
        "swea_selected_energies_ev": (
            reference_energies.tolist()
            if reference_energies is not None else None
        ),
    }


def write_json_atomic(path, payload):
    """Only small, non-measurement statistics are persisted."""
    path.parent.mkdir(parents=True, exist_ok=True)
    temp = path.with_suffix(path.suffix + ".tmp")
    try:
        with temp.open("w", encoding="utf-8") as fh:
            json.dump(payload, fh, ensure_ascii=False, indent=2, sort_keys=True)
            fh.write("\n")
        os.replace(temp, path)
    finally:
        if temp.exists():
            temp.unlink()


def main():
    args = parser().parse_args()
    if (
        args.max_days < 0 or args.max_query_gib <= 0
        or args.max_total_gib <= 0 or args.max_returned_rows < 1
    ):
        raise SystemExit("Invalid --max-days or BigQuery safety limits.")
    instrument = args.instrument.upper()
    events = pilot.read_manifest(args.manifest)
    intervals, fold_provenance, n_events = plan_training_days(
        events, args.fold, instrument
    )
    if not intervals:
        raise ValueError("No eligible training intervals.")
    signature = planned_signature(args.manifest, args.fold, instrument, intervals)
    key = f"fold_{args.fold:02d}_{instrument.lower()}"
    checkpoint_file = args.output_dir / f"{key}_checkpoint.json"
    final_file = args.output_dir / f"{key}_normalizer.json"
    checkpoint = None
    if checkpoint_file.exists():
        checkpoint = json.loads(checkpoint_file.read_text(encoding="utf-8"))
        if (
            checkpoint.get("signature") != signature
            or checkpoint.get("instrument") != instrument
            or checkpoint.get("held_out_fold") != args.fold
            or checkpoint.get("schema") != 1
        ):
            raise ValueError(
                "Checkpoint metadata does not match current split/code settings. "
                "Remove or archive old checkpoint intentionally; do not mix data."
            )
    completed = set(checkpoint["completed_days"]) if checkpoint else set()
    available = {day.isoformat() for day in intervals}
    if not completed <= available:
        raise ValueError("Checkpoint includes days outside the current training plan.")
    norm = restore_normalizer(checkpoint, instrument, args.fold)
    invalid_count = int(checkpoint["skipped_invalid_epochs"]) if checkpoint else 0
    reference_energy = (
        np.asarray(checkpoint["swea_selected_energies_ev"], dtype=np.float64)
        if checkpoint and checkpoint["swea_selected_energies_ev"] is not None
        else None
    )
    pending = [day for day in intervals if day.isoformat() not in completed]
    chosen = pending[:args.max_days] if args.max_days else pending
    print(
        f"Fold {args.fold} TRAIN-only, {instrument}: "
        f"{n_events} labeled training events, {len(intervals)} unique UTC days, "
        f"{sum(map(len, intervals.values()))} merged time segments."
    )
    print(
        f"Already completed: {len(completed)} days; pending: {len(pending)}; "
        f"selected this run: {len(chosen)} days. "
        f"Accumulated training epochs: {norm.count}."
    )
    if not chosen:
        if checkpoint and norm.count >= 2 and not final_file.exists():
            norm.freeze()
            write_json_atomic(final_file, {
                "schema": 1, "signature": signature,
                "instrument": instrument,
                "held_out_fold": args.fold,
                "n_unique_train_epochs": norm.count,
                "skipped_invalid_epochs": invalid_count,
                "n_completed_days": len(completed),
                "feature_transform": (
                    "identity" if instrument == "MAG" else "log10(1+flux)"
                ),
                "mean": norm.mean.tolist(),
                "scale": norm.scale.tolist(),
                "swea_selected_energies_ev": (
                    reference_energy.tolist()
                    if reference_energy is not None else None
                ),
            })
            print(f"Complete, final training-only normalizer: {final_file}")
        else:
            print("No unprocessed dates remain.")
        return

    client = bigquery.Client(project=bq.PROJECT)
    bq.validate_schema(client)
    plans = []
    per_query_max = int(args.max_query_gib*bq.GIB)
    for day in chosen:
        sql, dry_cfg = day_sql_and_config(
            instrument, day, intervals[day],
            dry_run=True, max_bytes=per_query_max
        )
        estimate = client.query(sql, job_config=dry_cfg)
        size = int(estimate.total_bytes_processed or 0)
        print(
            f"  DRY RUN {day}: {len(intervals[day])} disjoint segments, "
            f"{size/bq.GIB:.5f} GiB"
        )
        plans.append((day, sql, size))
    estimated = sum(row[2] for row in plans)
    print(f"Selected batch estimate: {estimated/bq.GIB:.5f} GiB.")
    if not args.execute:
        print("Dry run: no MAG/SWEA measurement rows fetched or stored.")
        return
    if estimated > args.max_total_gib*bq.GIB or any(
        row[2] > per_query_max for row in plans
    ):
        raise SystemExit("Dry-run estimates exceed requested BigQuery byte caps.")

    for day, sql, estimated_day in plans:
        _, config = day_sql_and_config(
            instrument, day, intervals[day],
            dry_run=False, max_bytes=per_query_max
        )
        print(f"PROCESS {day}: {estimated_day/bq.GIB:.5f} GiB estimated...")
        query = client.query(sql, job_config=config)
        timestamps, values, energies = read_day_data(
            query, instrument, args.max_returned_rows
        )
        if instrument == "SWEA" and len(timestamps):
            if reference_energy is None:
                reference_energy = energies.copy()
            elif energies.shape != reference_energy.shape or not np.allclose(
                energies, reference_energy, rtol=1e-6, atol=1e-6
            ):
                raise ValueError(
                    f"SWEA energy grid mismatch on {day}; "
                    "refusing to mix different physical energy channels."
                )
        clean = np.isfinite(values).all(axis=1)
        if instrument == "SWEA":
            clean &= np.all(values >= 0, axis=1)
        invalid_day = int(len(clean) - np.count_nonzero(clean))
        if np.any(clean):
            source_fold = min(fold_provenance[day])
            norm.update(values[clean], source_fold=source_fold)
        invalid_count += invalid_day
        completed.add(day.isoformat())
        del timestamps, values, energies
        write_json_atomic(checkpoint_file, checkpoint_payload(
            norm, signature, completed, invalid_count,
            reference_energy, instrument, args.fold
        ))
        print(
            f"  Day completed and checkpointed. "
            f"Accumulated unique training epochs: {norm.count}, "
            f"skipped invalid: {invalid_count}, "
            f"days: {len(completed)}/{len(intervals)}."
        )

    if len(completed) != len(intervals):
        print(
            f"Partial fit checkpoint: {checkpoint_file}. "
            "Re-run the SAME command to process the next UTC days; "
            "do not use partial statistics for model training."
        )
        return
    norm.freeze()
    write_json_atomic(final_file, {
        "schema": 1,
        "signature": signature,
        "instrument": instrument,
        "held_out_fold": args.fold,
        "n_unique_train_epochs": norm.count,
        "skipped_invalid_epochs": invalid_count,
        "n_completed_days": len(completed),
        "feature_transform": (
            "identity" if instrument == "MAG" else "log10(1+flux)"
        ),
        "mean": norm.mean.tolist(),
        "scale": norm.scale.tolist(),
        "swea_selected_energies_ev": (
            reference_energy.tolist()
            if reference_energy is not None else None
        ),
    })
    print(
        f"FULL FOLD COMPLETE: {norm.count} unique training epochs, "
        f"normalizer summary: {final_file}"
    )
    print("No MAVEN raw values, flux arrays, or training windows saved locally.")


if __name__ == "__main__":
    main()
