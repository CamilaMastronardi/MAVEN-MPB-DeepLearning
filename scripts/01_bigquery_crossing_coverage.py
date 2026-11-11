#!/usr/bin/env python3
"""Assess MAG/SWEA timestamp coverage around labeled MAVEN MPB crossings.

Only reads datetime_utc from the two partitioned BigQuery tables.
A dry run is the default; --execute explicitly enables billed queries.
No full MAG values or SWEA flux arrays are downloaded in this stage.
"""

from __future__ import annotations

import argparse
import csv
from datetime import datetime, timedelta, timezone
from pathlib import Path

from google.cloud import bigquery

PROJECT = "maven-vignes-crossings"
MAG_TABLE = f"{PROJECT}.maven.mag_fields"
SWEA_TABLE = f"{PROJECT}.maven.swea_svy3d"
AXES_TABLE = f"{PROJECT}.maven.swea_svy3d_axes"
POSITIONS_TABLE = f"{PROJECT}.maven.positions"
LABELS = Path("data/labels/mpb_crossings.csv")
OUTPUT_DIR = Path("results/metrics/bigquery_coverage")
WINDOW_SECONDS = 600
GIB = 1024 ** 3

SQL = """
WITH events AS (
  SELECT event_time FROM UNNEST(@event_times) AS event_time
),
matched AS (
  SELECT e.event_time, d.datetime_utc
  FROM `{table_id}` AS d
  INNER JOIN events AS e
    ON d.datetime_utc BETWEEN
      TIMESTAMP_SUB(e.event_time, INTERVAL 300 SECOND)
      AND TIMESTAMP_ADD(e.event_time, INTERVAL 300 SECOND)
  WHERE DATE(d.datetime_utc) IN UNNEST(@days)
),
with_deltas AS (
  SELECT
    event_time,
    datetime_utc,
    TIMESTAMP_DIFF(
      datetime_utc,
      LAG(datetime_utc) OVER (
        PARTITION BY event_time ORDER BY datetime_utc
      ),
      MICROSECOND
    ) / 1000000.0 AS dt_s
  FROM matched
)
SELECT
  event_time,
  COUNT(*) AS sample_count,
  MIN(datetime_utc) AS first_sample,
  MAX(datetime_utc) AS last_sample,
  COUNTIF(dt_s = 0) AS duplicate_intervals,
  COUNTIF(dt_s > 0) AS positive_intervals,
  APPROX_QUANTILES(IF(dt_s > 0, dt_s, NULL), 100)[OFFSET(50)]
    AS median_dt_s,
  MAX(IF(dt_s > 0, dt_s, NULL)) AS max_dt_s,
  AVG(IF(dt_s > 0, dt_s, NULL)) AS mean_dt_s
FROM with_deltas
GROUP BY event_time
ORDER BY event_time
"""


def args_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--labels", type=Path, default=LABELS)
    parser.add_argument("--output-dir", type=Path, default=OUTPUT_DIR)
    parser.add_argument("--instrument", choices=("both", "mag", "swea"), default="both")
    parser.add_argument("--batch-size", type=int, default=40)
    parser.add_argument(
        "--limit-events", type=int, default=0,
        help="Process only this many chronological events; 0 means all.",
    )
    parser.add_argument(
        "--execute", action="store_true",
        help="Actually execute billed queries. Default is dry-run estimates only.",
    )
    parser.add_argument(
        "--metadata-only", action="store_true",
        help="Print BigQuery schemas and partitioning without running SQL.",
    )
    parser.add_argument(
        "--max-billed-gib", type=float, default=1.0,
        help="Maximum estimated bytes per query in GiB, enforced for --execute.",
    )
    parser.add_argument(
        "--max-total-gib", type=float, default=5.0,
        help="Maximum sum of estimated query bytes, enforced for --execute.",
    )
    parser.add_argument(
        "--gap-factor", type=float, default=2.5,
        help="Preliminary QC threshold: dt and edge distance / median dt.",
    )
    return parser


def event_key(value: datetime) -> str:
    if value.tzinfo is None:
        raise ValueError("Event time is missing a timezone.")
    return value.astimezone(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ")


def read_events(path: Path, limit: int) -> list[dict]:
    events = []
    with path.open("r", newline="", encoding="utf-8") as handle:
        for row in csv.DictReader(handle):
            event_time = datetime.fromisoformat(
                row["mpb_timestamp_utc"].replace("Z", "+00:00")
            )
            if event_time.tzinfo is None:
                raise ValueError("Input annotations must contain UTC timestamps.")
            events.append({
                "event_time": event_time.astimezone(timezone.utc),
                "source_group": row["source_group"],
            })
    events.sort(key=lambda x: x["event_time"])
    keys = [event_key(x["event_time"]) for x in events]
    if len(keys) != len(set(keys)):
        raise ValueError("Duplicate MPB timestamps found in the annotation catalog.")
    if limit > 0:
        events = events[:limit]
    if not events:
        raise ValueError(f"No MPB labels were loaded from {path}")
    return events


def batches(seq: list[dict], size: int):
    for offset in range(0, len(seq), size):
        yield seq[offset : offset + size]


def utc_days_for(events: list[dict]) -> list:
    """Include both calendar dates if a 10-min window crosses midnight."""
    dates = set()
    half = timedelta(seconds=WINDOW_SECONDS / 2)
    for item in events:
        t = item["event_time"]
        dates.add((t - half).date())
        dates.add((t + half).date())
    return sorted(dates)


def schema_report(client: bigquery.Client) -> None:
    print("BigQuery table metadata (no data scanned):")
    for table_id in (MAG_TABLE, SWEA_TABLE, AXES_TABLE, POSITIONS_TABLE):
        table = client.get_table(table_id)
        tp = table.time_partitioning
        partitioned_by = (tp.field or "ingestion_time") if tp else "NO"
        print(f"\n{table_id}  |  partitioned_by={partitioned_by}")
        for field in table.schema:
            print(f"  {field.name}: {field.field_type} ({field.mode})")


def ensure_timestamp_partition(client: bigquery.Client, table_id: str) -> None:
    table = client.get_table(table_id)
    names = {field.name for field in table.schema}
    if "datetime_utc" not in names:
        raise RuntimeError(f"Missing datetime_utc in {table_id}")
    tp = table.time_partitioning
    if tp is None or tp.field != "datetime_utc":
        raise RuntimeError(
            f"{table_id} is not partitioned by datetime_utc. "
            "Stopped to avoid unbounded BigQuery scans."
        )


def job_config(
    events: list[dict],
    *,
    dry_run: bool,
    max_bytes: int | None = None,
) -> bigquery.QueryJobConfig:
    params = [
        bigquery.ArrayQueryParameter(
            "event_times", "TIMESTAMP", [e["event_time"] for e in events]
        ),
        bigquery.ArrayQueryParameter("days", "DATE", utc_days_for(events)),
    ]
    return bigquery.QueryJobConfig(
        query_parameters=params,
        dry_run=dry_run,
        use_query_cache=not dry_run,
        maximum_bytes_billed=max_bytes if not dry_run else None,
    )


def optional_seconds(value: object | None) -> float | None:
    return float(value) if value is not None else None


def analyze_row(
    event: dict,
    instrument: str,
    record: object | None,
    gap_factor: float,
) -> dict:
    time = event["event_time"]
    start = time - timedelta(seconds=WINDOW_SECONDS / 2)
    end = time + timedelta(seconds=WINDOW_SECONDS / 2)

    n = int(record.sample_count) if record else 0
    first = record.first_sample if record else None
    last = record.last_sample if record else None
    median = optional_seconds(record.median_dt_s) if record else None
    maximum = optional_seconds(record.max_dt_s) if record else None
    mean = optional_seconds(record.mean_dt_s) if record else None
    duplicates = int(record.duplicate_intervals) if record else 0
    left_edge = (first - start).total_seconds() if first else None
    right_edge = (end - last).total_seconds() if last else None
    span = (last - first).total_seconds() if first and last else None

    # This checks timestamp continuity, NOT MAG/SWEA physical validity.
    preliminary_ok = bool(
        n >= 3
        and median is not None
        and median > 0
        and maximum is not None
        and left_edge is not None
        and right_edge is not None
        and duplicates == 0
        and maximum <= gap_factor * median
        and left_edge <= gap_factor * median
        and right_edge <= gap_factor * median
    )

    return {
        "mpb_timestamp_utc": event_key(time),
        "source_group": event["source_group"],
        "instrument": instrument,
        "sample_count": n,
        "first_sample_utc": event_key(first) if first else "",
        "last_sample_utc": event_key(last) if last else "",
        "median_dt_s": median if median is not None else "",
        "mean_dt_s": mean if mean is not None else "",
        "max_dt_s": maximum if maximum is not None else "",
        "duplicate_intervals": duplicates,
        "observed_span_s": span if span is not None else "",
        "left_edge_s": left_edge if left_edge is not None else "",
        "right_edge_s": right_edge if right_edge is not None else "",
        "preliminary_time_coverage_ok": preliminary_ok,
    }


def write_csv(path: Path, rows: list[dict], columns: list[str]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=columns)
        writer.writeheader()
        writer.writerows(rows)


def summarize(results: list[dict], instruments: list[str], n_events: int) -> list[dict]:
    summary = []
    for instrument in instruments:
        subset = [r for r in results if r["instrument"] == instrument]
        summary.append({
            "instrument": instrument,
            "total_labeled_events": n_events,
            "events_with_samples": sum(r["sample_count"] > 0 for r in subset),
            "events_preliminary_coverage_ok": sum(
                r["preliminary_time_coverage_ok"] for r in subset
            ),
        })
    if len(instruments) == 2:
        usable = {}
        for r in results:
            usable.setdefault(r["mpb_timestamp_utc"], set())
            if r["preliminary_time_coverage_ok"]:
                usable[r["mpb_timestamp_utc"]].add(r["instrument"])
        both = sum(v == set(instruments) for v in usable.values())
        summary.append({
            "instrument": "both",
            "total_labeled_events": n_events,
            "events_with_samples": sum(
                all(
                    any(
                        r["instrument"] == inst
                        and r["mpb_timestamp_utc"] == event
                        and r["sample_count"] > 0
                        for r in results
                    )
                    for inst in instruments
                )
                for event in usable
            ),
            "events_preliminary_coverage_ok": both,
        })
    return summary


def main() -> None:
    args = args_parser().parse_args()
    if args.batch_size <= 0 or args.limit_events < 0:
        raise SystemExit("--batch-size must be positive; --limit-events >= 0")
    if args.max_billed_gib <= 0 or args.max_total_gib <= 0 or args.gap_factor <= 1:
        raise SystemExit("Byte limits must be positive and --gap-factor > 1")

    client = bigquery.Client(project=PROJECT)
    schema_report(client)
    if args.metadata_only:
        return

    events = read_events(args.labels, args.limit_events)
    instruments = (
        ["mag", "swea"] if args.instrument == "both" else [args.instrument]
    )
    tables = {"mag": MAG_TABLE, "swea": SWEA_TABLE}
    for instrument in instruments:
        ensure_timestamp_partition(client, tables[instrument])

    plans = []
    total_bytes = 0
    for instrument in instruments:
        for index, batch in enumerate(batches(events, args.batch_size), start=1):
            sql = SQL.format(table_id=tables[instrument])
            dry = client.query(sql, job_config=job_config(batch, dry_run=True))
            estimated_bytes = int(dry.total_bytes_processed or 0)
            total_bytes += estimated_bytes
            plans.append((instrument, index, batch, sql, estimated_bytes))
            print(
                f"DRY RUN | {instrument} batch {index}: "
                f"{len(batch)} events, {len(utc_days_for(batch))} UTC days, "
                f"{estimated_bytes / GIB:.4f} GiB"
            )
    print(f"\nTotal estimated query bytes: {total_bytes / GIB:.4f} GiB")
    if not args.execute:
        print("No data query executed. Re-run with --execute to compute coverage.")
        return

    if total_bytes > args.max_total_gib * GIB:
        raise SystemExit(
            f"Estimated total exceeds --max-total-gib={args.max_total_gib}. "
            "No data queries were executed."
        )
    if any(p[4] > args.max_billed_gib * GIB for p in plans):
        raise SystemExit(
            "At least one batch exceeds --max-billed-gib. "
            "No data queries were executed."
        )

    records = []
    for instrument, index, batch, sql, estimated_bytes in plans:
        config = job_config(
            batch,
            dry_run=False,
            max_bytes=int(args.max_billed_gib * GIB),
        )
        print(f"RUN | {instrument} batch {index} ({len(batch)} events)...")
        job = client.query(sql, job_config=config)
        observed = {
            event_key(row.event_time): row
            for row in job.result(page_size=1000)
        }
        for event in batch:
            records.append(
                analyze_row(
                    event,
                    instrument,
                    observed.get(event_key(event["event_time"])),
                    args.gap_factor,
                )
            )
    records.sort(key=lambda r: (r["mpb_timestamp_utc"], r["instrument"]))
    detail_cols = [
        "mpb_timestamp_utc", "source_group", "instrument",
        "sample_count", "first_sample_utc", "last_sample_utc",
        "median_dt_s", "mean_dt_s", "max_dt_s",
        "duplicate_intervals", "observed_span_s",
        "left_edge_s", "right_edge_s", "preliminary_time_coverage_ok",
    ]
    summary = summarize(records, instruments, len(events))
    detail_path = args.output_dir / "crossing_coverage_by_instrument.csv"
    summary_path = args.output_dir / "crossing_coverage_summary.csv"
    write_csv(detail_path, records, detail_cols)
    write_csv(
        summary_path, summary,
        [
            "instrument", "total_labeled_events", "events_with_samples",
            "events_preliminary_coverage_ok",
        ],
    )
    print(f"\nSaved {detail_path} and {summary_path}")
    print("Timestamp coverage only: SWEA flux validity and energy-grid checks are next.")
    for s in summary:
        print(
            f"  {s['instrument']:>4}: "
            f"with samples={s['events_with_samples']}/{s['total_labeled_events']}, "
            f"preliminary usable={s['events_preliminary_coverage_ok']}"
        )


if __name__ == "__main__":
    main()
