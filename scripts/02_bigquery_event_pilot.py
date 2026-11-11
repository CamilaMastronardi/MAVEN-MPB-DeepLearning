#!/usr/bin/env python3
"""Pilot extraction of 10-minute MAG and SWEA measurements at MPB labels.

Default: BigQuery dry-run only. Execute explicitly with --execute.
Preserves independent MAG/SWEA native timestamps; no cross-instrument resampling.
Stores raw SWEA angular-energy cubes and a provisional angle-mean spectrogram.
"""

from __future__ import annotations

import argparse
import csv
import json
from collections import Counter
from datetime import datetime, timedelta, timezone
from pathlib import Path

import numpy as np
from google.cloud import bigquery

PROJECT = "maven-vignes-crossings"
MAG_TABLE = f"{PROJECT}.maven.mag_fields"
SWEA_TABLE = f"{PROJECT}.maven.swea_svy3d"
SWEA_AXES = f"{PROJECT}.maven.swea_svy3d_axes"
LABELS = Path("data/labels/mpb_crossings.csv")
DEST = Path("data/raw/bigquery_pilot")
GIB = 1024 ** 3
HALF_WINDOW_SECONDS = 300

MAG_SQL = """
SELECT
  datetime_utc, bx_ss_nt, by_ss_nt, bz_ss_nt, b_total_nt,
  range_b, source_file
FROM `{table}`
WHERE DATE(datetime_utc) IN UNNEST(@days)
  AND datetime_utc BETWEEN @start_time AND @end_time
ORDER BY datetime_utc
"""

SWEA_SQL = """
SELECT
  d.datetime_utc, d.quality, d.source_file, d.diff_en_fluxes,
  a.energy_ev, a.elev_deg, a.azim_deg,
  a.n_elev, a.n_azim, a.n_energy
FROM `{data_table}` AS d
INNER JOIN `{axes_table}` AS a
  ON d.source_file = a.source_file
WHERE DATE(d.datetime_utc) IN UNNEST(@days)
  AND d.datetime_utc BETWEEN @start_time AND @end_time
ORDER BY d.datetime_utc
"""


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--labels", type=Path, default=LABELS)
    p.add_argument("--output-dir", type=Path, default=DEST)
    p.add_argument("--limit-events", type=int, default=1,
                   help="Number of chronological labels (default 1, max 3 for pilot).")
    p.add_argument("--execute", action="store_true",
                   help="Execute queries after estimating scanned bytes.")
    p.add_argument("--max-query-gib", type=float, default=1.0,
                   help="Safety limit per BigQuery data query.")
    p.add_argument("--max-total-gib", type=float, default=2.0,
                   help="Safety limit on estimated bytes for the complete pilot.")
    return p.parse_args()


def utc_time(value) -> datetime:
    if isinstance(value, str):
        value = datetime.fromisoformat(value.replace("Z", "+00:00"))
    if value.tzinfo is None:
        raise ValueError("Expected timezone-aware UTC timestamp.")
    return value.astimezone(timezone.utc)


def to_iso(value: datetime) -> str:
    return utc_time(value).strftime("%Y-%m-%dT%H:%M:%SZ")


def chosen_labels(path: Path, n: int) -> list[dict]:
    with path.open(newline="", encoding="utf-8") as file:
        labels = [
            {"time": utc_time(row["mpb_timestamp_utc"]),
             "source_group": row["source_group"]}
            for row in csv.DictReader(file)
        ]
    labels.sort(key=lambda e: e["time"])
    return labels[:n]


def query_config(start: datetime, end: datetime, *,
                 dry_run: bool, max_bytes: int) -> bigquery.QueryJobConfig:
    days = sorted({start.date(), end.date()})
    opts = {
        "query_parameters": [
            bigquery.ArrayQueryParameter("days", "DATE", days),
            bigquery.ScalarQueryParameter("start_time", "TIMESTAMP", start),
            bigquery.ScalarQueryParameter("end_time", "TIMESTAMP", end),
        ],
        "dry_run": dry_run,
        "use_query_cache": not dry_run,
    }
    # Do not set maximum_bytes_billed=None in a dry run: some clients
    # serialize it into a BadRequest-producing literal "None".
    if not dry_run:
        opts["maximum_bytes_billed"] = int(max_bytes)
    return bigquery.QueryJobConfig(**opts)


def validate_partitioning(client: bigquery.Client, table_id: str) -> None:
    table = client.get_table(table_id)
    if table.time_partitioning is None or table.time_partitioning.field != "datetime_utc":
        raise RuntimeError(f"Stopped: {table_id} is not partitioned by datetime_utc.")


def elapsed_statistics(times: list[datetime]) -> dict:
    if len(times) < 2:
        return {"n": len(times), "median_dt_s": None, "max_dt_s": None,
                "duplicates": 0}
    seconds = np.array([t.timestamp() for t in times], dtype=np.float64)
    dt = np.diff(seconds)
    positive = dt[dt > 0]
    return {
        "n": len(times),
        "median_dt_s": float(np.median(positive)) if positive.size else None,
        "max_dt_s": float(np.max(positive)) if positive.size else None,
        "duplicates": int(np.count_nonzero(dt == 0)),
    }


def save_mag(rows: list, output: Path) -> dict:
    if not rows:
        return {"available": False, "samples": 0}
    times = [utc_time(r.datetime_utc) for r in rows]
    vectors = np.asarray([
        [r.bx_ss_nt, r.by_ss_nt, r.bz_ss_nt, r.b_total_nt]
        for r in rows
    ], dtype=np.float64)
    if np.any(np.diff([x.timestamp() for x in times]) < 0):
        raise ValueError("MAG times not ordered.")
    np.savez_compressed(
        output,
        times_unix_s=np.asarray([t.timestamp() for t in times]),
        field_nt=vectors,
        range_b=np.array([
            r.range_b if r.range_b is not None else -1 for r in rows
        ], dtype=np.int64),
    )
    summary = elapsed_statistics(times)
    summary.update({
        "available": True,
        "samples": len(rows),
        "nonfinite_field_values": int((~np.isfinite(vectors)).sum()),
        "source_files": sorted({r.source_file for r in rows if r.source_file}),
    })
    return summary


def save_swea(rows: list, output: Path) -> dict:
    if not rows:
        return {"available": False, "samples": 0}

    reference_energy = None
    reference_elev = None
    reference_azim = None
    reference_shape = None
    cubes = []
    times = []
    quality = []
    sources = set()
    for row in rows:
        ne, na, nenergy = int(row.n_elev), int(row.n_azim), int(row.n_energy)
        shape = (ne, na, nenergy)
        if min(shape) < 1:
            raise ValueError("Invalid SWEA angular/energy dimensions.")
        energy = np.asarray(row.energy_ev, dtype=np.float64)
        elev = np.asarray(row.elev_deg, dtype=np.float64)
        azim = np.asarray(row.azim_deg, dtype=np.float64)
        if len(energy) != nenergy:
            raise ValueError("SWEA energy-axis length does not match n_energy.")
        if reference_energy is None:
            reference_energy = energy
            reference_elev = elev
            reference_azim = azim
            reference_shape = shape
        else:
            if (shape != reference_shape
                or not np.allclose(energy, reference_energy, rtol=1e-6, atol=1e-6,
                                   equal_nan=False)
                or not np.array_equal(elev, reference_elev, equal_nan=True)
                or not np.array_equal(azim, reference_azim, equal_nan=True)):
                raise ValueError(
                    "SWEA energy/angle axes or dimensions differ across epochs: "
                    "do not combine these frames without separate preprocessing."
                )
        flat = np.asarray(row.diff_en_fluxes, dtype=np.float64)
        if flat.size != ne * na * nenergy:
            raise ValueError(
                f"Invalid SWEA flux array length {flat.size}; expected {ne*na*nenergy}"
            )
        cubes.append(flat.reshape(shape))  # Original C-order: energy fastest.
        times.append(utc_time(row.datetime_utc))
        quality.append(int(row.quality) if row.quality is not None else -1)
        if row.source_file:
            sources.add(row.source_file)

    raw_cubes = np.stack(cubes)
    valid = np.isfinite(raw_cubes) & (raw_cubes >= 0)
    # Preliminary exploratory representation only; angular response is not
    # corrected. NOT an omnidirectional flux or validated SWEA feature.
    valid_counts = valid.sum(axis=(1, 2))
    finite_sum = np.where(valid, raw_cubes, 0).sum(axis=(1, 2))
    angle_mean = np.full(valid_counts.shape, np.nan, dtype=np.float64)
    np.divide(finite_sum, valid_counts, out=angle_mean, where=valid_counts > 0)

    np.savez_compressed(
        output,
        times_unix_s=np.asarray([t.timestamp() for t in times]),
        flux_angular_cube=raw_cubes,
        angle_mean_flux=angle_mean,
        valid_angle_count=valid_counts,
        energy_ev=reference_energy,
        elev_deg=reference_elev,
        azim_deg=reference_azim,
        quality=np.asarray(quality, dtype=np.int64),
    )
    summary = elapsed_statistics(times)
    summary.update({
        "available": True,
        "samples": len(rows),
        "shape_time_elev_azim_energy": list(raw_cubes.shape),
        "energy_min_ev": float(np.min(reference_energy)),
        "energy_max_ev": float(np.max(reference_energy)),
        "nonfinite_or_negative_flux_fraction":
            float(np.count_nonzero(~valid) / valid.size),
        "energy_time_cells_without_valid_angular_flux":
            int(np.count_nonzero(valid_counts == 0)),
        "quality_counts": dict(Counter(str(x) for x in quality)),
        "source_files": sorted(sources),
        "angle_mean_note":
            "Exploratory valid-bin average only; quality flags not filtered.",
    })
    return summary


def main() -> None:
    args = parse_args()
    if not (1 <= args.limit_events <= 3):
        raise SystemExit("Pilot uses --limit-events between 1 and 3.")
    if args.max_query_gib <= 0 or args.max_total_gib <= 0:
        raise SystemExit("BigQuery limits must be positive.")
    events = chosen_labels(args.labels, args.limit_events)
    if len(events) != args.limit_events:
        raise SystemExit("Fewer label events than requested.")

    client = bigquery.Client(project=PROJECT)
    validate_partitioning(client, MAG_TABLE)
    validate_partitioning(client, SWEA_TABLE)
    queries = []
    for idx, event in enumerate(events, start=1):
        t = event["time"]
        start = t - timedelta(seconds=HALF_WINDOW_SECONDS)
        end = t + timedelta(seconds=HALF_WINDOW_SECONDS)
        for name, sql in (
            ("mag", MAG_SQL.format(table=MAG_TABLE)),
            ("swea", SWEA_SQL.format(data_table=SWEA_TABLE, axes_table=SWEA_AXES)),
        ):
            dry = client.query(
                sql, job_config=query_config(start, end, dry_run=True, max_bytes=1)
            )
            size = int(dry.total_bytes_processed or 0)
            queries.append({
                "index": idx, "event": event, "start": start, "end": end,
                "instrument": name, "sql": sql, "bytes": size,
            })
            print(
                f"DRY RUN | event {idx} {name}: {to_iso(t)}, "
                f"estimated scanned={size / GIB:.4f} GiB"
            )
    total = sum(q["bytes"] for q in queries)
    print(f"TOTAL estimated scanned={total / GIB:.4f} GiB")
    if not args.execute:
        print("No data query executed. Re-run with --execute if estimates are acceptable.")
        return
    per_query_limit = int(args.max_query_gib * GIB)
    if total > args.max_total_gib * GIB or any(q["bytes"] > per_query_limit for q in queries):
        raise SystemExit(
            "Estimated scanned bytes exceed a safety cap. No data queries executed. "
            "Review --max-query-gib and --max-total-gib before proceeding."
        )

    args.output_dir.mkdir(parents=True, exist_ok=True)
    results = []
    for query in queries:
        i, event, name = query["index"], query["event"], query["instrument"]
        print(f"RUN | event {i} {name} ...")
        job = client.query(
            query["sql"],
            job_config=query_config(
                query["start"], query["end"], dry_run=False,
                max_bytes=per_query_limit,
            ),
        )
        rows = list(job.result(page_size=250))
        stamp = event["time"].strftime("%Y%m%dT%H%M%S")
        path = args.output_dir / f"event_{i:02d}_{stamp}_{name}.npz"
        summary = save_mag(rows, path) if name == "mag" else save_swea(rows, path)
        if not rows:
            path = None
        results.append({
            "event_time_utc": to_iso(event["time"]),
            "source_group": event["source_group"],
            "instrument": name,
            "file": str(path) if path else None,
            "scanned_gib_estimate": query["bytes"] / GIB,
            **summary,
        })
        print(f"  {name}: {summary.get('samples', 0)} time samples; "
              f"median dt={summary.get('median_dt_s')} s")
    report = args.output_dir / "pilot_summary.json"
    report.write_text(json.dumps(results, indent=2, allow_nan=False), encoding="utf-8")
    print(f"Saved pilot arrays and {report}")
    print("NOTE: SWEA angular mean and raw flux are unfiltered for instrument quality.")


if __name__ == "__main__":
    main()
