#!/usr/bin/env python3
"""Fold-aware, bounded, in-RAM native-cadence window generation pilot.

Defaults to BigQuery DRY RUN for the first held-out event of CV fold 1.
--execute runs the (capped) BigQuery queries for up to three chosen events,
then constructs native MAG and 30–100 eV SWEA windows in RAM.

No scientific-data files, arrays, plots, or model tensors are written to disk.
This is a data-loader pilot, NOT model training or a negative-label sampler.
"""
from __future__ import annotations

import argparse
import csv
import sys
from datetime import timedelta, timezone, datetime
from importlib import import_module
from pathlib import Path

import numpy as np
from google.cloud import bigquery

# The numbered scripts live in stage-specific folders. Expose the repository
# root so direct CLI execution can resolve their namespace packages.
ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

windowing = import_module("scripts.3_windowing.3a_windowing")
WindowStats = windowing.WindowStats
iter_native_windows = windowing.iter_native_windows

# Reuse the verified SQL, safety guards, schema and spectral band from the
# existing BigQuery pilot rather than duplicating them here. Importing it
# performs NO queries: its main() is guarded by __name__ == '__main__'.
bq = import_module("scripts.1_bigquery.1c_bigquery_event_pilot")
MANIFEST = Path("data/splits/cv4_monthly_manifest.csv")
WINDOW_SECONDS = 600.0
CENTER_RANGE_SECONDS = 1800.0


def cli() -> argparse.Namespace:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--manifest", type=Path, default=MANIFEST)
    p.add_argument("--fold", type=int, choices=(1, 2, 3, 4), default=1)
    p.add_argument("--subset", choices=("train", "val"), default="val")
    p.add_argument("--instrument", choices=("both", "mag", "swea"), default="both")
    p.add_argument("--event-index", type=int, default=1,
                   help="1-based index in the chosen fold/subset.")
    p.add_argument("--limit-events", type=int, default=1,
                   help="How many consecutive events to inspect (max 3).")
    p.add_argument("--sigma-seconds", type=float, default=60.0,
                   help="Temporal width of proximity target, NOT MPB thickness.")
    p.add_argument("--max-query-gib", type=float, default=1.0)
    p.add_argument("--max-total-gib", type=float, default=2.0)
    p.add_argument("--max-returned-rows", type=int, default=25000)
    p.add_argument("--execute", action="store_true")
    args = p.parse_args()
    if args.event_index < 1 or not 1 <= args.limit_events <= 3:
        p.error("Require --event-index >= 1 and 1 <= --limit-events <= 3.")
    if (args.sigma_seconds <= 0 or args.max_query_gib <= 0
        or args.max_total_gib <= 0 or args.max_returned_rows < 1):
        p.error("All sigma and safety limits must be positive.")
    return args


def read_manifest(path: Path) -> list[dict]:
    with path.open(encoding="utf-8", newline="") as handle:
        reader = csv.DictReader(handle)
        expected = {"mpb_timestamp_utc", "fold", "mag_available",
                    "swea_available", "month_block"}
        if not expected <= set(reader.fieldnames or []):
            raise ValueError(f"Manifest is missing columns: {expected - set(reader.fieldnames or [])}")
        rows = list(reader)
    if not rows:
        raise ValueError("No MPB labels in manifest.")
    for row in rows:
        row["time"] = bq.to_utc(row["mpb_timestamp_utc"])
        row["fold"] = int(row["fold"])
        if row["month_block"] != row["time"].strftime("%Y-%m"):
            raise ValueError("Manifest month block mismatches its event timestamp.")
    if len({r["time"] for r in rows}) != len(rows):
        raise ValueError("Duplicate event timestamp in manifest.")
    return sorted(rows, key=lambda r: r["time"])


def read_mag(result_rows: list) -> tuple[np.ndarray, np.ndarray, np.ndarray | None]:
    dates = np.asarray(
        [bq.to_utc(row.datetime_utc).timestamp() for row in result_rows],
        dtype=np.float64
    )
    x = np.asarray([
        [row.bx_ss_nt, row.by_ss_nt, row.bz_ss_nt, row.b_total_nt]
        for row in result_rows
    ], dtype=np.float64)
    return dates, x.reshape(-1, 4), None


def read_swea(result_rows: list) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    timestamps = []
    values = []
    reference_energy = None
    selected_channels = None
    selected_energy = None
    for row in result_rows:
        dims = tuple(map(int, (row.n_elev, row.n_azim, row.n_energy)))
        if row.original_flux_count != int(np.prod(dims)):
            raise ValueError("SWEA original flux array does not match its axes.")
        energy = np.asarray(row.energy_ev, dtype=np.float64)
        if len(energy) != dims[-1]:
            raise ValueError("SWEA energy axis length mismatches the data cube.")
        if not np.isfinite(energy).all():
            raise ValueError("SWEA energy axis contains invalid values.")
        if reference_energy is None:
            reference_energy = energy
            selected_channels = np.flatnonzero(
                (energy >= bq.SWEA_ENERGY_MIN_EV) &
                (energy <= bq.SWEA_ENERGY_MAX_EV)
            )
            selected_energy = energy[selected_channels]
            if not selected_channels.size:
                raise ValueError("No energy channel lies within 30–100 eV.")
        elif energy.shape != reference_energy.shape or not np.allclose(
            reference_energy, energy, rtol=1e-6, atol=1e-6
        ):
            raise ValueError(
                "SWEA energy grid changes in event block; process by source/grid."
            )
        spectrum = np.full(len(selected_channels), np.nan)
        channels = {int(ch): i for i, ch in enumerate(selected_channels)}
        for b in row.energy_bins:
            idx = int(b["energy_index"])
            if idx not in channels:
                raise ValueError("Unexpected energy index outside 30–100 eV.")
            flux = b["mean_flux"]
            spectrum[channels[idx]] = float(flux) if flux is not None else np.nan
        timestamps.append(bq.to_utc(row.datetime_utc).timestamp())
        values.append(spectrum)

    if not timestamps:
        return np.empty(0), np.empty((0, 0)), np.empty(0)
    return (
        np.asarray(timestamps, dtype=np.float64),
        np.asarray(values, dtype=np.float64),
        selected_energy,
    )


def check_block(rows: list, max_rows: int) -> None:
    if len(rows) > max_rows:
        raise MemoryError("BigQuery result row cap exceeded.")
    if not rows:
        print("  No observations in this event block.")
        return


def describe_windows(times, data, selected_energy, event_time, labels, name, sigma):
    if not len(times):
        print(f"  {name}: no observations; skipped.")
        return

    if data.shape[0] != len(times):
        raise ValueError("Time and data dimensions disagree.")
    deltas = np.diff(times)
    if len(deltas) and np.any(deltas <= 0):
        raise ValueError("Non-increasing timestamps; investigate duplicated records.")
    median_dt = float(np.median(deltas)) if len(deltas) else float("nan")
    print(
        f"  {name} block: {len(times)} epochs, cadence median {median_dt:.3f} s, "
        f"features {data.shape[1]}"
    )
    if name == "SWEA":
        print(f"  Selected energies 30–100 eV: "
              f"{np.array2string(selected_energy, precision=2)}")
    invalid = int((~np.isfinite(data)).sum())
    print(f"  Invalid selected input values: {invalid}/{data.size}")

    stats = WindowStats()
    count = 0
    score_min = 1.0
    score_max = 0.0
    shape_min = float("inf")
    shape_max = 0
    closest = None
    strong_target = 0
    for window in iter_native_windows(
        times, data,
        center_label_s=event_time,
        split_labels_s=labels,
        window_seconds=WINDOW_SECONDS,
        center_range_seconds=CENTER_RANGE_SECONDS,
        sigma_seconds=sigma,
        stats=stats,
    ):
        count += 1
        score_min = min(score_min, window.detection_score)
        score_max = max(score_max, window.detection_score)
        shape_min = min(shape_min, len(window.data))
        shape_max = max(shape_max, len(window.data))
        strong_target += window.detection_score >= 0.5
        if closest is None or abs(window.offset_from_nearest_mpb_s) < abs(
            closest.offset_from_nearest_mpb_s
        ):
            closest = window

    print(
        f"  Candidate centers: {stats.candidates}; accepted windows: {count}; "
        f"rejected incomplete={stats.incomplete}, large_gap={stats.large_gap}, "
        f"invalid_flux_or_B={stats.invalid_data}"
    )
    if closest is not None:
        example_shape = (data.shape[1], len(closest.data)) if name == "MAG" else (
            1, data.shape[1], len(closest.data)
        )
        print(
            f"  Actual model input near MPB: {example_shape}; "
            f"native sample counts per window: {shape_min}..{shape_max}"
        )
        print(
            f"  Gaussian target: min={score_min:.4g}, max={score_max:.4g}; "
            f"centers with target >= 0.5: {strong_target}"
        )
        print(
            f"  Closest center offset to a labeled MPB: "
            f"{closest.offset_from_nearest_mpb_s:.2f} s, "
            f"target={closest.detection_score:.5f}"
        )
    print(
        "  Low Gaussian target means far from a known annotation; "
        "it does NOT certify a true physical negative."
    )


def main() -> None:
    args = cli()
    manifest = read_manifest(args.manifest)
    held_out = lambda row: row["fold"] == args.fold
    select = lambda row: held_out(row) if args.subset == "val" else not held_out(row)
    event_pool = [row for row in manifest if select(row)]
    if args.event_index + args.limit_events - 1 > len(event_pool):
        raise SystemExit(f"Event selection exceeds {len(event_pool)} labeled events.")
    chosen = event_pool[
        args.event_index-1:args.event_index-1+args.limit_events
    ]
    labels = np.asarray(
        [row["time"].timestamp() for row in event_pool], dtype=np.float64
    )
    instruments = ("MAG", "SWEA") if args.instrument == "both" else (
        args.instrument.upper(),)
    print(
        f"CV fold {args.fold}, subset={args.subset}, "
        f"selected {len(chosen)} event(s), instruments={','.join(instruments)}."
    )
    print("Window=600 s; centers ±1800 s; 1 native-sample stride.")
    print(f"Gaussian target sigma={args.sigma_seconds:g} s (proximity, not probability).")

    client = bigquery.Client(project=bq.PROJECT)
    bq.validate_schema(client)
    jobs = []
    for event in chosen:
        t0 = event["time"]
        start = t0 - timedelta(seconds=CENTER_RANGE_SECONDS + WINDOW_SECONDS/2)
        end = t0 + timedelta(seconds=CENTER_RANGE_SECONDS + WINDOW_SECONDS/2)
        print(f"EVENT {t0.isoformat()} | source={event['source_group']} | fold={event['fold']}")
        for name in instruments:
            if event[f"{name.lower()}_available"] != "1":
                print(f"  {name}: no timestamp coverage for this labeled event; skipped.")
                continue
            sql = (
                bq.MAG_SQL.format(mag=bq.MAG)
                if name == "MAG"
                else bq.SWEA_SQL.format(swea=bq.SWEA, axes=bq.AXES)
            )
            dry = client.query(
                sql, job_config=bq.query_config(start, end, dry_run=True, max_bytes=1)
            )
            estimated = int(dry.total_bytes_processed or 0)
            jobs.append((event, name, sql, start, end, estimated))
            print(
                f"  DRY RUN {name} [{start.isoformat()}..{end.isoformat()}]: "
                f"{estimated/bq.GIB:.5f} GiB"
            )
    total = sum(job[-1] for job in jobs)
    print(f"Total estimated BigQuery processing: {total/bq.GIB:.5f} GiB")
    if not args.execute:
        print("Dry run only. No observations queried or saved.")
        return
    limit_one = int(args.max_query_gib*bq.GIB)
    if total > args.max_total_gib*bq.GIB or any(
        job[-1] > limit_one for job in jobs
    ):
        raise SystemExit("Estimated BigQuery bytes exceed the configured limits.")

    for event, name, sql, start, end, estimated in jobs:
        print(f"RUN {name} {event['time'].isoformat()}...")
        query = client.query(
            sql, job_config=bq.query_config(
                start, end, dry_run=False, max_bytes=limit_one
            )
        )
        rows = []
        for record in query.result(page_size=250):
            rows.append(record)
            if len(rows) > args.max_returned_rows:
                raise MemoryError("Result row cap exceeded; aborting this query.")
        check_block(rows, args.max_returned_rows)
        if name == "MAG":
            times, measurements, energy = read_mag(rows)
        else:
            times, measurements, energy = read_swea(rows)
        del rows
        describe_windows(
            times, measurements, energy, event["time"].timestamp(),
            labels, name, args.sigma_seconds
        )
        del times, measurements

    print("Done. No MAG/SWEA measurement arrays were written to disk.")


if __name__ == "__main__":
    main()
