#!/usr/bin/env python3
"""Verify that every V10 source event has valid MAG+SWEA caches, without BigQuery.

This is a read-only cache check performed before the V10-HP1 training launcher.
The actual V10 loader independently rechecks checksums and event provenance.
"""
from __future__ import annotations

import argparse
import sys
from importlib import import_module
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

v7 = import_module("scripts.5_training.5b_train_mag_swea_regions")
pilot = import_module("scripts.3_windowing.3b_bigquery_window_pilot")
bq = import_module("scripts.1_bigquery.1c_bigquery_event_pilot")


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--manifest", type=Path, default=v7.MANIFEST)
    parser.add_argument("--mag-cache-dir", type=Path, default=v7.MAG_CACHE)
    parser.add_argument("--swea-cache-dir", type=Path, default=v7.SWEA_CACHE)
    parser.add_argument("--pilot-events-per-fold", type=int, default=0)
    parser.add_argument("--max-mag-rows", type=int, default=25000)
    parser.add_argument("--max-swea-rows", type=int, default=2000)
    args = parser.parse_args()
    if (args.pilot_events_per_fold < 0 or
            args.max_mag_rows <= 0 or args.max_swea_rows <= 0):
        parser.error("Invalid limits or pilot count.")

    rows = pilot.read_manifest(args.manifest)
    selected = {}
    # Every V10 outer fold uses the other three folds as training sources,
    # hence all four folds are needed even when testing a single outer fold.
    for fold in (1, 2, 3, 4):
        for row in v7.chosen_events(
            rows, fold, pilot_count=args.pilot_events_per_fold
        ):
            selected[row["time"]] = row
    if not selected:
        raise SystemExit("No MAG+SWEA catalog events selected.")

    sqls = v7.query_sources()
    caches = {
        "MAG": v7.MagEventCache(
            args.mag_cache_dir, project=bq.PROJECT,
            table=bq.MAG, sql=sqls["MAG"]
        ),
        "SWEA": v7.SweaEventCache(
            args.swea_cache_dir, project=bq.PROJECT,
            table=bq.SWEA + " + " + bq.AXES, sql=sqls["SWEA"]
        ),
    }
    missing = []
    for row in sorted(selected.values(), key=lambda item: item["time"]):
        start, end = v7.event_bounds(row)
        for sensor in ("MAG", "SWEA"):
            cap = args.max_mag_rows if sensor == "MAG" else args.max_swea_rows
            cached = caches[sensor].load(
                row["time"], start, end, max_rows=cap
            )
            if cached is None or (sensor == "SWEA" and not len(cached[0])):
                missing.append((row["time"].isoformat(), sensor))

    print(
        f"V10-HP1 CACHE PREFLIGHT: {len(selected)} events; "
        f"{2 * len(selected) - len(missing)}/{2 * len(selected)} "
        f"valid sensor entries; {len(missing)} missing; "
        "no BigQuery clients or queries.",
        flush=True,
    )
    if missing:
        for timestamp, sensor in missing[:20]:
            print(f"MISSING {timestamp} {sensor}", file=sys.stderr)
        raise SystemExit(
            "Incomplete cache: aborting before training to avoid BigQuery. "
            "Use the existing cache/population workflow separately."
        )


if __name__ == "__main__":
    main()
