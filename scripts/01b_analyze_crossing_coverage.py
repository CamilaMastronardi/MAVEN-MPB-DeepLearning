#!/usr/bin/env python3
"""Analyze saved MAG/SWEA timestamp coverage without running BigQuery queries.

Input: results/metrics/bigquery_coverage/crossing_coverage_by_instrument.csv
Output: event tables, group coverage, cadence summaries and diagnostic plots.
"""
from __future__ import annotations

import argparse
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

DEFAULT_INPUT = Path("results/metrics/bigquery_coverage/crossing_coverage_by_instrument.csv")
DEFAULT_OUTPUT = Path("results/metrics/coverage_analysis")


def parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--input", type=Path, default=DEFAULT_INPUT)
    p.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT)
    return p


def stats(series: pd.Series) -> dict:
    x = pd.to_numeric(series, errors="coerce").dropna()
    if x.empty:
        return {"n": 0, "mean_s": np.nan, "median_s": np.nan,
                "p05_s": np.nan, "p95_s": np.nan, "max_s": np.nan}
    return {
        "n": len(x),
        "mean_s": float(x.mean()),
        "median_s": float(x.median()),
        "p05_s": float(x.quantile(0.05)),
        "p95_s": float(x.quantile(0.95)),
        "max_s": float(x.max()),
    }


def main() -> None:
    args = parser().parse_args()
    if not args.input.exists():
        raise SystemExit(f"Input not found: {args.input}. Run coverage diagnostic --execute.")
    df = pd.read_csv(args.input)
    required = {
        "mpb_timestamp_utc", "source_group", "instrument", "sample_count",
        "median_dt_s", "max_dt_s", "preliminary_time_coverage_ok",
    }
    missing = required - set(df.columns)
    if missing:
        raise SystemExit(f"Required columns missing: {sorted(missing)}")
    if df.duplicated(["mpb_timestamp_utc", "instrument"]).any():
        raise SystemExit("Duplicate event/instrument rows found; check source file.")
    instruments = set(df["instrument"].dropna())
    if instruments != {"mag", "swea"}:
        raise SystemExit(f"Expected both MAG and SWEA coverage, got {sorted(instruments)}")

    df["preliminary_time_coverage_ok"] = (
        df["preliminary_time_coverage_ok"].astype(str).str.strip().str.lower().eq("true")
    )
    df["sample_count"] = pd.to_numeric(df["sample_count"], errors="coerce").fillna(0)
    df["median_dt_s"] = pd.to_numeric(df["median_dt_s"], errors="coerce")
    df["max_dt_s"] = pd.to_numeric(df["max_dt_s"], errors="coerce")
    df["mpb_timestamp_utc"] = pd.to_datetime(df["mpb_timestamp_utc"], utc=True)

    events = (
        df.pivot(index="mpb_timestamp_utc", columns="instrument",
                 values=["sample_count", "preliminary_time_coverage_ok", "median_dt_s"])
    )
    if events.isna().any().any():
        # Only counts and coverage flags must be nonmissing; cadence can be missing when n=0.
        if events["sample_count"].isna().any().any() or events["preliminary_time_coverage_ok"].isna().any().any():
            raise SystemExit("Some event/instrument combinations are missing in coverage file.")

    group = df.drop_duplicates("mpb_timestamp_utc").set_index("mpb_timestamp_utc")["source_group"]
    flat = pd.DataFrame(index=events.index)
    flat["source_group"] = group
    for ins in ("mag", "swea"):
        flat[f"{ins}_n_samples"] = events[("sample_count", ins)].fillna(0).astype(int)
        flat[f"{ins}_time_ok"] = events[("preliminary_time_coverage_ok", ins)].astype(bool)
        flat[f"{ins}_median_dt_s"] = events[("median_dt_s", ins)]
    flat["both_time_ok"] = flat["mag_time_ok"] & flat["swea_time_ok"]

    args.output_dir.mkdir(parents=True, exist_ok=True)
    flat.reset_index().to_csv(args.output_dir / "crossing_coverage_combined.csv", index=False)
    missing_swea = flat[flat["swea_n_samples"].eq(0)]
    missing_swea.reset_index().to_csv(args.output_dir / "events_without_swea.csv", index=False)
    failed = flat[~flat["both_time_ok"]]
    failed.reset_index().to_csv(args.output_dir / "events_failed_joint_coverage.csv", index=False)

    by_group = []
    for label, data in flat.groupby("source_group", sort=True):
        by_group.append({
            "source_group": label,
            "events": len(data),
            "mag_available": int((data["mag_n_samples"] > 0).sum()),
            "swea_available": int((data["swea_n_samples"] > 0).sum()),
            "mag_time_ok": int(data["mag_time_ok"].sum()),
            "swea_time_ok": int(data["swea_time_ok"].sum()),
            "both_time_ok": int(data["both_time_ok"].sum()),
        })
    pd.DataFrame(by_group).to_csv(args.output_dir / "coverage_by_group.csv", index=False)

    cadences = []
    for ins in ("mag", "swea"):
        subset = df[(df["instrument"] == ins) &
                    (df["preliminary_time_coverage_ok"])]
        for metric in ("median_dt_s", "max_dt_s"):
            cadences.append({"instrument": ins, "metric": metric,
                             **stats(subset[metric])})
        vals = subset["median_dt_s"].dropna()
        if not vals.empty:
            fig, ax = plt.subplots(figsize=(8, 5))
            ax.hist(vals.to_numpy(dtype=float), bins=min(40, max(5, int(np.sqrt(len(vals))))))
            ax.set_title(f"{ins.upper()} – median cadence in 10-minute event windows")
            ax.set_xlabel("Median positive sampling interval (s)")
            ax.set_ylabel("Number of labeled events")
            fig.tight_layout()
            fig.savefig(args.output_dir / f"{ins}_median_cadence.png", dpi=160)
            plt.close(fig)
    pd.DataFrame(cadences).to_csv(args.output_dir / "cadence_statistics.csv", index=False)

    n_events = len(flat)
    print(f"Total labeled events analyzed: {n_events}")
    for ins in ("mag", "swea"):
        print(f"  {ins.upper():>4}: sampled={(flat[f'{ins}_n_samples'] > 0).sum()}/{n_events}; "
              f"timestamp QC={flat[f'{ins}_time_ok'].sum()}/{n_events}")
    print(f"  BOTH: timestamp QC={flat['both_time_ok'].sum()}/{n_events}")
    print(f"  Events without SWEA samples: {len(missing_swea)}")
    if not missing_swea.empty:
        for t, row in missing_swea.iterrows():
            print(f"    {t.isoformat()}  ({row['source_group']})")
    print("\nMedian of event-wise median cadences (timestamp QC-passing windows):")
    for ins in ("mag", "swea"):
        d = flat.loc[flat[f"{ins}_time_ok"], f"{ins}_median_dt_s"].dropna()
        print(f"  {ins.upper()}: {d.median():.3f} s" if not d.empty else f"  {ins.upper()}: no valid windows")
    print(f"\nSaved diagnostics in: {args.output_dir}")


if __name__ == "__main__":
    main()
