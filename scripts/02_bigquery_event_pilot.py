#!/usr/bin/env python3
"""Inspect one to three 10-minute MPB windows directly from BigQuery.

Dry-run by default. --execute processes bounded query results in RAM, not files.
SWEA's angular average is computed remotely in BigQuery per energy/time sample;
this is an exploratory feature, not calibrated omnidirectional electron flux.
"""

from __future__ import annotations

import argparse
import csv
from collections import Counter
from datetime import datetime, timedelta, timezone
from pathlib import Path

import numpy as np
from google.cloud import bigquery

PROJECT = "maven-vignes-crossings"
MAG = f"{PROJECT}.maven.mag_fields"
SWEA = f"{PROJECT}.maven.swea_svy3d"
AXES = f"{PROJECT}.maven.swea_svy3d_axes"
LABELS = Path("data/labels/mpb_crossings.csv")
GIB = 1024 ** 3

MAG_SQL = """
SELECT datetime_utc, bx_ss_nt, by_ss_nt, bz_ss_nt, b_total_nt, range_b
FROM `{mag}`
WHERE DATE(datetime_utc) IN UNNEST(@days)
  AND datetime_utc BETWEEN @start AND @end
ORDER BY datetime_utc
"""

# Angular bins stay on the BigQuery server. The client receives only one
# flux average and count per energy, for each SWEA epoch.
SWEA_SQL = """
SELECT d.datetime_utc, d.quality, d.source_file,
       a.energy_ev, a.n_elev, a.n_azim, a.n_energy,
       ARRAY_LENGTH(d.diff_en_fluxes) AS original_flux_count,
       ARRAY(
         SELECT AS STRUCT
           MOD(offset_index, a.n_energy) AS energy_index,
           AVG(IF(
             flux IS NOT NULL AND NOT IS_NAN(flux)
             AND NOT IS_INF(flux) AND flux >= 0,
             flux, NULL
           )) AS mean_flux,
           COUNTIF(
             flux IS NOT NULL AND NOT IS_NAN(flux)
             AND NOT IS_INF(flux) AND flux >= 0
           ) AS valid_count
         FROM UNNEST(d.diff_en_fluxes) AS flux WITH OFFSET AS offset_index
         GROUP BY energy_index
         ORDER BY energy_index
       ) AS energy_bins
FROM `{swea}` AS d
JOIN `{axes}` AS a USING (source_file)
WHERE DATE(d.datetime_utc) IN UNNEST(@days)
  AND d.datetime_utc BETWEEN @start AND @end
ORDER BY d.datetime_utc
"""


def args_parser():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--labels", type=Path, default=LABELS)
    p.add_argument("--limit-events", type=int, default=1)
    p.add_argument(
        "--instrument", choices=("both", "mag", "swea"), default="both",
        help="Run both instruments, or only one to avoid unnecessary queries.",
    )
    p.add_argument("--execute", action="store_true")
    p.add_argument("--max-query-gib", type=float, default=1.0)
    p.add_argument("--max-total-gib", type=float, default=2.0)
    return p


def to_utc(value):
    if isinstance(value, str):
        value = datetime.fromisoformat(value.replace("Z", "+00:00"))
    if value.tzinfo is None:
        raise ValueError("Time must have an explicit UTC offset.")
    return value.astimezone(timezone.utc)


def events_from_csv(path, number):
    with path.open(encoding="utf-8", newline="") as f:
        events = [
            to_utc(row["mpb_timestamp_utc"])
            for row in csv.DictReader(f)
        ]
    events.sort()
    if len(events) < number:
        raise ValueError(f"Only {len(events)} event labels found")
    return events[:number]


def query_config(start, end, *, dry_run, max_bytes):
    options = {
        "dry_run": dry_run,
        "use_query_cache": not dry_run,
        "query_parameters": [
            bigquery.ArrayQueryParameter(
                "days", "DATE", sorted({start.date(), end.date()})
            ),
            bigquery.ScalarQueryParameter("start", "TIMESTAMP", start),
            bigquery.ScalarQueryParameter("end", "TIMESTAMP", end),
        ],
    }
    # Never send maximum_bytes_billed=None to BigQuery in dry-run mode.
    if not dry_run:
        options["maximum_bytes_billed"] = int(max_bytes)
    return bigquery.QueryJobConfig(**options)


def validate_schema(client):
    for name in (MAG, SWEA):
        table = client.get_table(name)
        partitioning = table.time_partitioning
        if partitioning is None or partitioning.field != "datetime_utc":
            raise ValueError(f"Unsafe query: {name} not partitioned by datetime_utc")
    axis_fields = {field.name for field in client.get_table(AXES).schema}
    expected = {"source_file", "energy_ev", "n_elev", "n_azim", "n_energy"}
    if not expected <= axis_fields:
        raise ValueError(f"Missing SWEA axis columns: {sorted(expected - axis_fields)}")


def cadence(rows):
    timestamps = np.array(
        [to_utc(row.datetime_utc).timestamp() for row in rows],
        dtype=float,
    )
    dt = np.diff(timestamps)
    positive = dt[dt > 0]
    return (
        float(np.median(positive)) if positive.size else None,
        float(np.max(positive)) if positive.size else None,
        int(np.count_nonzero(dt == 0)),
    )


def inspect_mag(rows):
    if not rows:
        print("  MAG: no observations")
        return
    x = np.array(
        [[r.bx_ss_nt, r.by_ss_nt, r.bz_ss_nt, r.b_total_nt] for r in rows],
        dtype=float,
    )
    med, maximum, duplicates = cadence(rows)
    print(
        f"  MAG samples: {len(rows)}, field shape: {x.shape}, "
        f"median dt: {med} s, max dt: {maximum} s, "
        f"duplicate timestamps: {duplicates}, "
        f"invalid field values: {int((~np.isfinite(x)).sum())}"
    )


def inspect_swea(rows):
    if not rows:
        print("  SWEA: no observations")
        return
    shapes = set()
    reference_energy = None
    flags = Counter()
    spectra = []
    counts = []
    shape_mismatches = 0
    for r in rows:
        dims = (int(r.n_elev), int(r.n_azim), int(r.n_energy))
        shapes.add(dims)
        if r.original_flux_count != int(np.prod(dims)):
            shape_mismatches += 1
        energy = np.asarray(r.energy_ev, dtype=float)
        if len(energy) != dims[-1]:
            raise ValueError("SWEA energy axis has wrong dimension")
        if reference_energy is None:
            reference_energy = energy
        elif not np.allclose(
            energy, reference_energy, rtol=1e-6, atol=1e-6
        ):
            raise ValueError("SWEA energy axis changes between samples")
        spectrum = np.full(dims[-1], np.nan, dtype=float)
        valid_counts = np.zeros(dims[-1], dtype=int)
        for b in r.energy_bins:
            # BigQuery ARRAY<STRUCT> elements are returned as dicts by some
            # google-cloud-bigquery versions (rather than Row objects).
            # The mapping protocol works for both dict and bigquery.table.Row.
            idx = int(b["energy_index"])
            if idx < 0 or idx >= dims[-1]:
                raise ValueError("Invalid energy bin index")
            if b["mean_flux"] is not None:
                spectrum[idx] = float(b["mean_flux"])
            valid_counts[idx] = int(b["valid_count"])
        spectra.append(spectrum)
        counts.append(valid_counts)
        flags[str(r.quality)] += 1

    if len(shapes) != 1:
        raise ValueError(f"Inconsistent SWEA dimensions: {sorted(shapes)}")
    spectrogram = np.stack(spectra)
    count_matrix = np.stack(counts)
    med, maximum, duplicates = cadence(rows)
    print(
        f"  SWEA samples: {len(rows)}, cube dimensions: {next(iter(shapes))}, "
        f"energy-time shape: {spectrogram.shape}"
    )
    print(
        f"  SWEA median dt: {med} s, max dt: {maximum} s, "
        f"duplicate timestamps: {duplicates}"
    )
    print(
        f"  SWEA energy range: {reference_energy.min():.3g} .. "
        f"{reference_energy.max():.3g} eV, bad-length frames: {shape_mismatches}, "
        f"energy-time bins with no valid angular flux: "
        f"{int(np.count_nonzero(count_matrix == 0))}/{count_matrix.size}"
    )
    print(f"  SWEA quality flag counts: {dict(flags)}")
    print("  Angular mean is exploratory; not a validated omnidirectional flux.")


def main():
    args = args_parser().parse_args()
    if not 1 <= args.limit_events <= 3:
        raise SystemExit("--limit-events must be between 1 and 3")
    if args.max_query_gib <= 0 or args.max_total_gib <= 0:
        raise SystemExit("Positive BigQuery byte limits are required")

    events = events_from_csv(args.labels, args.limit_events)
    client = bigquery.Client(project=PROJECT)
    validate_schema(client)
    plans = []
    for n, tc in enumerate(events, start=1):
        start = tc - timedelta(seconds=300)
        end = tc + timedelta(seconds=300)
        queries = {
            "MAG": MAG_SQL.format(mag=MAG),
            "SWEA": SWEA_SQL.format(swea=SWEA, axes=AXES),
        }
        if args.instrument != "both":
            selected = args.instrument.upper()
            queries = {selected: queries[selected]}
        for name, sql in queries.items():
            estimate = client.query(
                sql, job_config=query_config(start, end, dry_run=True, max_bytes=1)
            )
            bytes_estimated = int(estimate.total_bytes_processed or 0)
            plans.append((n, tc, start, end, name, sql, bytes_estimated))
            print(
                f"DRY RUN event {n}, {name}: {tc.isoformat()}, "
                f"estimated {bytes_estimated / GIB:.5f} GiB"
            )

    total = sum(plan[-1] for plan in plans)
    print(f"Total estimated: {total / GIB:.5f} GiB")
    if not args.execute:
        print("No measurement queries performed; nothing saved locally.")
        return

    max_one = int(args.max_query_gib * GIB)
    if total > args.max_total_gib * GIB or any(
        plan[-1] > max_one for plan in plans
    ):
        raise SystemExit("Estimated bytes exceed query caps; no data fetched")

    for n, tc, start, end, name, sql, estimated in plans:
        print(f"RUN event {n} {name} ({tc.isoformat()})...")
        result = client.query(
            sql, job_config=query_config(
                start, end, dry_run=False, max_bytes=max_one
            )
        )
        # Only this 10-minute result exists in process memory.
        rows = list(result.result(page_size=250))
        if name == "MAG":
            inspect_mag(rows)
        else:
            inspect_swea(rows)
        del rows

    print("Done: no MAG or SWEA measurements were saved to disk.")


if __name__ == "__main__":
    main()
