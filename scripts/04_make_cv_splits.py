#!/usr/bin/env python3
"""Make leakage-aware, source-balanced monthly cross-validation folds for MPB.

Inputs are small, version-controlled label/coverage metadata CSV files ONLY.
NO BigQuery queries, raw MAG/SWEA downloads or model training are performed.

All events in the same YYYY-MM block are assigned to the same fold.
The default four folds are balanced by labeled-event counts and catalog group.
The script checks that potential raw signal supports of train/validation
events do not overlap for a declared extraction context.
"""
from __future__ import annotations

import argparse
import csv
import json
from collections import Counter, defaultdict
from datetime import datetime, timezone
from pathlib import Path

LABEL_PATH = Path("data/labels/mpb_crossings.csv")
MISSING_SWEA_PATH = Path("data/splits/swea_missing_event_timestamps.csv")
OUT = Path("data/splits")
DEFAULT_FOLDS = 4
WINDOW_MINUTES = 10
CENTER_RANGE_MINUTES = 30


def parse_time(value: str) -> datetime:
    result = datetime.fromisoformat(value.replace("Z", "+00:00"))
    if result.tzinfo is None:
        raise ValueError(f"Timestamp lacks timezone: {value}")
    return result.astimezone(timezone.utc)


def load_labels(path: Path) -> tuple[list[str], list[dict]]:
    with path.open(encoding="utf-8", newline="") as handle:
        reader = csv.DictReader(handle)
        headers = list(reader.fieldnames or [])
        required = {"mpb_timestamp_utc", "source_group"}
        if not required.issubset(headers):
            raise ValueError(f"Missing columns in labels: {required - set(headers)}")
        data = [dict(row) for row in reader]
    if not data:
        raise ValueError("The MPB catalog is empty.")
    data.sort(key=lambda r: parse_time(r["mpb_timestamp_utc"]))
    timestamps = [parse_time(r["mpb_timestamp_utc"]) for r in data]
    if len(timestamps) != len(set(timestamps)):
        raise ValueError("The catalog contains repeated MPB timestamps.")
    for row, date in zip(data, timestamps):
        row["_time"] = date
        row["_month"] = date.strftime("%Y-%m")
    return headers, data


def load_missing_swea(path: Path, known_times: set) -> set:
    with path.open(encoding="utf-8", newline="") as handle:
        rows = list(csv.DictReader(handle))
    missing = [parse_time(row["mpb_timestamp_utc"]) for row in rows]
    if len(missing) != len(set(missing)):
        raise ValueError("Duplicate missing SWEA entries.")
    unknown = set(missing) - known_times
    if unknown:
        raise ValueError(f"Missing SWEA list has unknown label times: {unknown}")
    return set(missing)


def build_folds(rows: list[dict], n_folds: int) -> tuple[dict, dict]:
    months = defaultdict(list)
    for row in rows:
        months[row["_month"]].append(row)

    by_catalog = defaultdict(list)
    for month, items in months.items():
        catalogs = {row["source_group"] for row in items}
        if len(catalogs) != 1:
            raise ValueError(
                f"Month {month} mixes source groups {catalogs}; "
                "revise the stratification algorithm before running."
            )
        catalog = next(iter(catalogs))
        by_catalog[catalog].append((month, len(items)))

    if any(len(month_blocks) < n_folds for month_blocks in by_catalog.values()):
        details = {k: len(v) for k, v in by_catalog.items()}
        raise ValueError(
            f"{n_folds} folds cannot represent every catalog group in each fold. "
            f"Number of monthly blocks per group: {details}."
        )

    total_count = [0] * n_folds
    block_count = [0] * n_folds
    per_source_count = {
        catalog: [0] * n_folds for catalog in by_catalog
    }
    assignment = {}

    # Process smaller source catalogs first, then their largest blocks.
    # During the first n_folds months of a catalog, place one in each fold.
    for catalog in sorted(by_catalog, key=lambda g: (len(by_catalog[g]), g)):
        group_months = sorted(by_catalog[catalog], key=lambda x: (-x[1], x[0]))
        for i, (month, number) in enumerate(group_months):
            candidates = list(range(n_folds))
            if i < n_folds:
                candidates = [
                    f for f in candidates if per_source_count[catalog][f] == 0
                ]
            f = min(
                candidates,
                key=lambda fold: (
                    per_source_count[catalog][fold],
                    total_count[fold],
                    block_count[fold],
                    fold,
                ),
            )
            assignment[month] = f
            total_count[f] += number
            per_source_count[catalog][f] += number
            block_count[f] += 1

    return assignment, {
        "total_events": total_count,
        "blocks": block_count,
        "source_counts": per_source_count,
    }


def minimum_interfold_gap(rows: list[dict], assignment: dict) -> float:
    # For events ordered in time, the nearest cross-fold pair is adjacent
    # somewhere in the sorted sequence. Comparing adjacent pairs suffices.
    gap_minutes = [
        (after["_time"] - before["_time"]).total_seconds() / 60.0
        for before, after in zip(rows, rows[1:])
        if assignment[before["_month"]] != assignment[after["_month"]]
    ]
    return min(gap_minutes) if gap_minutes else float("inf")


def save_csv(path: Path, columns: list[str], data: list[dict]) -> None:
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=columns, extrasaction="ignore")
        writer.writeheader()
        writer.writerows(data)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--labels", type=Path, default=LABEL_PATH)
    parser.add_argument("--swea-missing", type=Path, default=MISSING_SWEA_PATH)
    parser.add_argument("--out", type=Path, default=OUT)
    parser.add_argument("--folds", type=int, default=DEFAULT_FOLDS)
    parser.add_argument("--window-minutes", type=float, default=WINDOW_MINUTES)
    parser.add_argument(
        "--center-range-minutes", type=float, default=CENTER_RANGE_MINUTES,
        help="Max absolute offset of a sliding window center from its MPB label.",
    )
    args = parser.parse_args()

    if args.folds < 2 or args.window_minutes <= 0 or args.center_range_minutes < 0:
        parser.error("Require folds >=2, window-minutes >0, center-range-minutes >=0.")

    headers, rows = load_labels(args.labels)
    missing_swea = load_missing_swea(
        args.swea_missing, {row["_time"] for row in rows}
    )
    months, balance = build_folds(rows, args.folds)
    minimum_gap = minimum_interfold_gap(rows, months)
    # Input samples used by a sliding window centered up to center_range
    # minutes from a label extend up to half a window farther on each side.
    # Distinct folds need event-time separation greater than twice the support.
    required_gap = 2 * args.center_range_minutes + args.window_minutes
    if minimum_gap <= required_gap:
        raise RuntimeError(
            f"Potential train/validation overlap: closest cross-fold event "
            f"gap {minimum_gap:.2f} min <= required {required_gap:.2f} min. "
            "Change grouping or shrink the intended extraction context."
        )

    args.out.mkdir(parents=True, exist_ok=True)
    manifest = []
    for row in rows:
        values = {k: row[k] for k in headers}
        values.update({
            "month_block": row["_month"],
            "fold": months[row["_month"]] + 1,
            "mag_available": 1,
            "swea_available": int(row["_time"] not in missing_swea),
        })
        manifest.append(values)

    extra = ["month_block", "fold", "mag_available", "swea_available"]
    columns = headers + extra
    prefix = f"cv{args.folds}"
    save_csv(args.out / f"{prefix}_monthly_manifest.csv", columns, manifest)

    source_groups = sorted({row["source_group"] for row in rows})
    summary_rows = []
    for fold in range(1, args.folds + 1):
        val = [row for row in manifest if row["fold"] == fold]
        train = [row for row in manifest if row["fold"] != fold]
        save_csv(args.out / f"fold_{fold:02d}_train.csv", columns, train)
        save_csv(args.out / f"fold_{fold:02d}_val.csv", columns, val)
        summary_row = {
            "fold": fold,
            "train_events": len(train),
            "val_events": len(val),
            "mag_train": len(train),
            "mag_val": len(val),
            "swea_train": sum(r["swea_available"] for r in train),
            "swea_val": sum(r["swea_available"] for r in val),
            "val_months": len({r["month_block"] for r in val}),
        }
        for source in source_groups:
            summary_row[f"val_{source}"] = sum(
                r["source_group"] == source for r in val
            )
        summary_rows.append(summary_row)

    save_csv(
        args.out / f"{prefix}_summary.csv",
        list(summary_rows[0].keys()),
        summary_rows,
    )
    config = {
        "split_unit": "UTC calendar month",
        "folds": args.folds,
        "training_rule": "fold != validation fold",
        "validation_rule": "fold == validation fold",
        "window_minutes": args.window_minutes,
        "center_range_minutes": args.center_range_minutes,
        "minimum_required_interfold_event_gap_minutes": required_gap,
        "observed_minimum_interfold_event_gap_minutes": minimum_gap,
        "missing_swea_labels": len(missing_swea),
        "source_month_counts": dict(sorted(Counter(
            row["_month"] for row in rows
        ).items())),
        "warning": (
            "Leakage guarantee holds only if ALL fetched signals and any "
            "negative-window candidates stay within the declared event supports. "
            "For larger neighborhoods, regenerate and recheck. "
            "No unseen-year extrapolation is tested."
        ),
    }
    (args.out / f"{prefix}_config.json").write_text(
        json.dumps(config, indent=2) + "\n", encoding="utf-8"
    )
    print(f"Total events: {len(rows)}; distinct monthly blocks: {len(months)}")
    print(f"MAG available: {len(rows)}; SWEA available: {len(rows)-len(missing_swea)}")
    print(
        f"Minimum cross-fold separation: {minimum_gap:.1f} min "
        f"(required > {required_gap:.1f} min)"
    )
    for s in summary_rows:
        groups = ", ".join(
            f"{key.removeprefix('val_')}={value}"
            for key, value in s.items() if key.startswith("val_Group")
        )
        print(
            f"Fold {s['fold']}: train={s['train_events']}, val={s['val_events']}, "
            f"SWEA train={s['swea_train']}, val={s['swea_val']}; "
            f"validation: {groups}"
        )
    print(f"Saved fold manifests and summaries in {args.out}")
    print("No MAVEN scientific measurements or BigQuery data were used.")


if __name__ == "__main__":
    main()
