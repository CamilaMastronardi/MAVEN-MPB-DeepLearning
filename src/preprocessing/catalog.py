"""Lectura validada del manifiesto y conversion de MAG en cadencia nativa."""
from __future__ import annotations
import csv
from pathlib import Path
import numpy as np

from src.preprocessing import bigquery_source as bq

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

