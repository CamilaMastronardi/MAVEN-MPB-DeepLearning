#!/usr/bin/env python3
"""Audit and compare four V7b region weights with fixed rank_weight=2 and rank_margin=0.25.

Checkpoint selection was already done within each run on its INNER-month
monitor. Hyperparameter selection here is based ONLY on inner-monitor
ranking top-1 accuracy (tie: pair accuracy, smaller region weight).
Outer-fold metrics remain exploratory for individual region weights; the per-fold
inner-selected combination is the nested-evaluation result.

No BigQuery, PyTorch, network access, or source data are read.
"""
from __future__ import annotations

import argparse
import csv
import json
import math
import statistics
from pathlib import Path

REGION_WEIGHTS = ((0.25, "region_0p25"), (0.5, "region_0p5"), (1.0, "region_1"), (2.0, "region_2"))
REQUIRED_TRAIN_CONFIG = (
    "seed", "learning_rate", "batch_size", "max_epochs",
    "evaluation_stride_s", "manifest_sha256",
    "early_stop_min_epochs", "early_stop_patience",
    "rank_margin", "reconstruction_weight", "rank_weight",
    "rank_control_step_s", "rank_positives_per_event",
    "rank_hard_negatives", "rank_random_negatives",
    "swea_energy_ev", "mag_swea_native_time_grid",
)
REQUIRED_SUMMARY_CONFIG = (
    "seed", "learning_rate", "batch_size", "max_epochs",
    "manifest_sha256", "swea_fixed_energy_ev", "evaluation_stride_s",
    "reconstruction_weight", "rank_margin", "rank_weight",
    "rank_control_step_s", "rank_positives_per_event",
    "rank_hard_negatives", "rank_random_negatives",
    "epochs_min", "patience",
)


def json_from(path: Path) -> dict:
    if not path.is_file():
        raise FileNotFoundError(f"Required result not found: {path}")
    return json.loads(path.read_text(encoding="utf-8"))


def csv_from(path: Path) -> list[dict]:
    if not path.is_file():
        raise FileNotFoundError(f"Required event CSV not found: {path}")
    with path.open(newline="", encoding="utf-8") as handle:
        return list(csv.DictReader(handle))


def save_json(path: Path, obj: dict) -> None:
    path.write_text(json.dumps(obj, indent=2, sort_keys=True, allow_nan=False) + "\n",
                    encoding="utf-8")


def save_csv(path: Path, rows: list[dict]) -> None:
    if not rows:
        raise ValueError(f"No rows to write at {path}")
    with path.open("w", newline="", encoding="utf-8") as handle:
        keys = list(dict.fromkeys(key for row in rows for key in row))
        writer = csv.DictWriter(handle, fieldnames=keys)
        writer.writeheader()
        writer.writerows(rows)


def expect_equal(value, expected, what: str) -> None:
    if value != expected:
        raise ValueError(f"{what}: expected {expected!r}, got {value!r}")


def monitor_selection_key(result: dict) -> tuple:
    """Common region-weight-independent score for all four model variants.

    Compare weight-independent top-1 and pair accuracy of the monitor,
    not the BCE-weighted checkpoint losses between different region weights.
    """
    top1 = result.get("monitor_event_top1_fraction")
    pair = result.get("monitor_event_pair_accuracy")
    rank = result.get("monitor_event_rank_loss")
    if any(value is None or not math.isfinite(float(value))
           for value in (top1, pair, rank)):
        raise ValueError("Missing or non-finite inner-monitor rank metrics")
    # Both monitor metrics are independent of the region-weight multiplier.
    return float(top1), float(pair)


def load_region_dir(root: Path, region: float, folder: str,
                    folds: list[int]) -> tuple[dict, dict]:
    run_dir = root / folder
    summary = json_from(run_dir / "cv_summary.json")
    if summary.get("architecture") != "dual-mag-swea-event-ranking-v1":
        raise ValueError(f"Not a V7b ranking run: {run_dir}")
    expect_equal(float(summary["training_configuration"]["region_weight"]),
                 region, f"{folder} summary region_weight")
    expect_equal(float(summary["training_configuration"]["rank_margin"]),
                 0.25, f"{folder} fixed rank_margin")
    expect_equal(float(summary["training_configuration"]["rank_weight"]),
                 2.0, f"{folder} fixed rank_weight")
    requested = [int(f) for f in summary.get("requested_folds", [])]
    completed = [int(f) for f in summary.get("completed_folds", [])]
    expect_equal(sorted(requested), sorted(folds), f"{folder} requested folds")
    expect_equal(sorted(completed), sorted(folds), f"{folder} completed folds")

    data = {}
    for fold in folds:
        fold_dir = run_dir
        metrics = json_from(fold_dir / f"fold_{fold:02d}_metrics.json")
        expect_equal(int(metrics["fold"]), fold, "metrics fold")
        expect_equal(metrics["architecture"], summary["architecture"],
                     "fold architecture")
        expect_equal(float(metrics["training_configuration"]["region_weight"]),
                     region, "fold region weight")
        expect_equal(float(metrics["training_configuration"]["rank_margin"]),
                     0.25, "fixed fold rank_margin")
        expect_equal(float(metrics["training_configuration"]["rank_weight"]),
                     2.0, "fixed fold rank_weight")
        raw = csv_from(fold_dir / f"fold_{fold:02d}_event_metrics.csv")
        expect_equal(len(raw), int(metrics["val_total_events"]),
                     f"{folder} fold {fold} event count")
        events = {}
        for row in raw:
            event = row["mpb_timestamp_utc"]
            if event in events:
                raise ValueError(f"Duplicate event {event}: {fold_dir}")
            error = None
            if row["absolute_error_s"] != "":
                error = float(row["absolute_error_s"])
                if not math.isfinite(error) or error < 0:
                    raise ValueError(f"Bad error {error} in {fold_dir}")
            events[event] = {"abs_error_s": error,
                             "n_windows": int(row["n_windows"]),
                             "source_group": row.get("source_group", "")}
        actual_evaluable = sum(row["abs_error_s"] is not None
                               for row in events.values())
        expect_equal(actual_evaluable, int(metrics["val_evaluable_events"]),
                     f"{folder} fold {fold} evaluable count")
        data[fold] = {"metrics": metrics, "events": events}
    return summary, data


def percentile_linear(values: list[float], p: float) -> float | None:
    """Percentile with linear interpolation on sorted observed errors."""
    if not values:
        return None
    ordered = sorted(values)
    position = p * (len(ordered) - 1)
    lower = int(position)
    upper = min(lower + 1, len(ordered) - 1)
    fraction = position - lower
    return ordered[lower] * (1 - fraction) + ordered[upper] * fraction


def error_stats(events: list[dict]) -> dict:
    errors = [ev["abs_error_s"] for ev in events
              if ev["abs_error_s"] is not None]
    n, evaluable = len(events), len(errors)
    if n == 0:
        raise ValueError("Empty event group")
    return {
        "n_events": n,
        "n_evaluable": evaluable,
        "evaluable_fraction": evaluable / n,
        "median_abs_error_s": statistics.median(errors) if errors else None,
        "p90_abs_error_s": percentile_linear(errors, 0.9),
        "within_60s_over_all": sum(e <= 60 for e in errors) / n,
        "within_120s_over_all": sum(e <= 120 for e in errors) / n,
        "within_60s_among_evaluable": (
            sum(e <= 60 for e in errors) / evaluable if evaluable else None),
        "within_120s_among_evaluable": (
            sum(e <= 120 for e in errors) / evaluable if evaluable else None),
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--results-root", type=Path,
                        default=Path("results/metrics/v7b_region_weight_sweep"))
    parser.add_argument("--folds", choices=("all", "1", "2", "3", "4"),
                        default="all")
    args = parser.parse_args()
    folds = [1, 2, 3, 4] if args.folds == "all" else [int(args.folds)]
    args.results_root.mkdir(parents=True, exist_ok=True)

    configurations, runs = {}, {}
    for region, folder in REGION_WEIGHTS:
        configurations[region], runs[region] = load_region_dir(
            args.results_root, region, folder, folds)

    # Compare only runs with identical manifest/splits, controlled parameters
    # and identical lists of catalog validation events.
    baseline = configurations[REGION_WEIGHTS[0][0]]
    for region, _ in REGION_WEIGHTS[1:]:
        other = configurations[region]
        expect_equal(other.get("pilot_subset"), baseline.get("pilot_subset"),
                     "pilot status")
        for key in REQUIRED_SUMMARY_CONFIG:
            if key not in baseline["training_configuration"] or key not in other["training_configuration"]:
                raise KeyError(f"Missing controlled summary hyperparameter {key}")
            expect_equal(other["training_configuration"][key],
                         baseline["training_configuration"][key],
                         f"configuration {key}")

    for fold in folds:
        ref = runs[REGION_WEIGHTS[0][0]][fold]
        for region, _ in REGION_WEIGHTS[1:]:
            cur = runs[region][fold]
            expect_equal(cur["metrics"]["train_events"],
                         ref["metrics"]["train_events"],
                         f"train count in fold {fold}")
            expect_equal(cur["metrics"]["inner_monitor_events"],
                         ref["metrics"]["inner_monitor_events"],
                         f"monitor count in fold {fold}")
            # Fold normalization is fitted on unique native TRAIN FIT epochs;
            # identical statistics and candidate counts detect dataset drift.
            for field in ("normalizers", "train_candidate_events",
                          "monitor_candidate_events", "train_skip_counts",
                          "monitor_skip_counts", "val_skip_counts",
                          "monitor_ranking_event_count"):
                expect_equal(cur["metrics"].get(field),
                             ref["metrics"].get(field),
                             f"{field} in fold {fold}")
            for field in REQUIRED_TRAIN_CONFIG:
                conf = cur["metrics"]["training_configuration"]
                ref_conf = ref["metrics"]["training_configuration"]
                if field not in conf or field not in ref_conf:
                    raise KeyError(f"Missing controlled fold hyperparameter {field}")
                expect_equal(conf[field], ref_conf[field],
                             f"{field} in fold {fold}")
            expect_equal(set(cur["events"]), set(ref["events"]),
                         f"event identity in fold {fold}")
            for event in ref["events"]:
                expect_equal(cur["events"][event]["source_group"],
                             ref["events"][event]["source_group"],
                             f"source group of {event}")
                expect_equal(cur["events"][event]["n_windows"],
                             ref["events"][event]["n_windows"],
                             f"input coverage of {event}")

    summary_rows = []
    selection_rows = []
    chosen_event_records = []
    paired_events = []
    common_evaluable = {}
    for fold in folds:
        ref_ids = sorted(runs[REGION_WEIGHTS[0][0]][fold]["events"])
        ids = [event for event in ref_ids
               if all(runs[region][fold]["events"][event]["abs_error_s"] is not None
                      for region, _ in REGION_WEIGHTS)]
        common_evaluable[fold] = set(ids)
        for event in ref_ids:
            record = {"fold": fold, "mpb_timestamp_utc": event,
                      "evaluable_all_region_weights": event in common_evaluable[fold]}
            for region, folder in REGION_WEIGHTS:
                record[f"{folder}_error_s"] = (
                    runs[region][fold]["events"][event]["abs_error_s"])
            paired_events.append(record)

        possible = [(region, monitor_selection_key(runs[region][fold]["metrics"]))
                    for region, _ in REGION_WEIGHTS]
        # Smaller region weight breaks complete ties; rank margin stays fixed.
        chosen_region, key = max(possible, key=lambda row: (*row[1], -row[0]))
        chosen = runs[chosen_region][fold]
        selection_rows.append({
            "fold": fold, "chosen_region_weight": chosen_region,
            "selection_metric": "INNER monitor event_top1, then pair_accuracy; tie = smaller region weight",
            "monitor_event_top1_fraction": key[0],
            "monitor_event_pair_accuracy": key[1],
            "monitor_event_rank_loss": chosen["metrics"]["monitor_event_rank_loss"],
            **error_stats(list(chosen["events"].values()))
        })
        chosen_event_records.extend(chosen["events"].values())

        for region, folder in REGION_WEIGHTS:
            run = runs[region][fold]
            inner = run["metrics"]
            summary_rows.append({
                "region_weight": region, "fold": fold,
                "best_epoch": inner["best_epoch"],
                "monitor_event_top1_fraction": inner["monitor_event_top1_fraction"],
                "monitor_event_pair_accuracy": inner["monitor_event_pair_accuracy"],
                "monitor_event_rank_loss": inner["monitor_event_rank_loss"],
                "candidate_weak_f1": inner.get("candidate_weak_f1"),
                "weak_region_accuracy": inner.get("weak_region_accuracy"),
                **error_stats(list(run["events"].values()))
            })

    for region, _ in REGION_WEIGHTS:
        all_events = [record for fold in folds
                      for record in runs[region][fold]["events"].values()]
        shared_events = [runs[region][fold]["events"][event]
                         for fold in folds for event in sorted(common_evaluable[fold])]
        summary_rows.append({
            "region_weight": region, "fold": "all_selected_folds",
            **error_stats(all_events),
            "paired_common_evaluable_count": len(shared_events),
            "paired_common_median_error_s": (statistics.median(
                x["abs_error_s"] for x in shared_events)
                if shared_events else None),
            "paired_common_within_120s_fraction": (
                sum(x["abs_error_s"] <= 120 for x in shared_events)
                / len(shared_events) if shared_events else None),
        })

    chosen_stats = error_stats(chosen_event_records)
    final = {
        "folds": folds,
        "tested_region_weights": [w for w, _ in REGION_WEIGHTS],
        "selected_region_weights_by_inner_monitor": {
            str(row["fold"]): row["chosen_region_weight"] for row in selection_rows
        },
        "inner_selected_outer_metrics": chosen_stats,
        "common_evaluable_events_all_region_weights": sum(len(ids)
                                                      for ids in common_evaluable.values()),
        "interpretation": (
            "Region weights selected separately inside each fold using INNER monitor "
            "event_top1 accuracy. Outer metrics for individual region weights are exploratory "
            "and were not used for checkpoint or hyperparameter selection. "
            "This remains conditional +/-30-minute MPB localization, not "
            "independent orbit-wide detection."
        ),
    }
    save_csv(args.results_root / "comparison_by_fold_and_region_weight.csv", summary_rows)
    save_csv(args.results_root / "paired_outer_events.csv", paired_events)
    save_csv(args.results_root / "chosen_region_weights_by_inner_monitor.csv", selection_rows)
    save_json(args.results_root / "nested_inner_selected_summary.json", final)

    print("V7b region-weight sweep: matched event identities and configurations validated")
    for row in selection_rows:
        print(f"Fold {row['fold']}: inner-monitor selected region_weight={row['chosen_region_weight']}; "
              f"outer within 120 s (all)={row['within_120s_over_all']:.1%}")
    print("Nested, inner-monitor-selected outcome:")
    print(json.dumps(chosen_stats, indent=2))
    print(f"Outputs: {args.results_root}")


if __name__ == "__main__":
    main()