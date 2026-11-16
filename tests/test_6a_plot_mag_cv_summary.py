"""Offline tests for the existing MAG CV figure script.

Only deliberately small, algebraic metric fixtures; no MAVEN measurements,
no BigQuery requests and no training.
"""
import csv
import importlib
import json
import tempfile
import unittest
from pathlib import Path

import numpy as np

figures = importlib.import_module("scripts.6_figures.6a_plot_mag_cv_summary")


class MagFiguresOfflineTests(unittest.TestCase):
    def test_histories_plot_three_losses_and_best_epoch(self):
        with tempfile.TemporaryDirectory() as folder:
            base = Path(folder)
            for fold in range(1, 5):
                metrics = {
                    "fold": fold,
                    "best_epoch": 2,
                    "train_history": [
                        {
                            "epoch": epoch,
                            "train_rec": 0.2 / epoch,
                            "val_rec": 0.3 / epoch,
                            "train_det": 0.5 / epoch,
                            "val_det": 0.6 / epoch,
                            "train_mag": 0.4 / epoch,
                            "val_mag": 0.7 / epoch,
                        } for epoch in (1, 2)
                    ],
                }
                (base / f"fold_{fold:02d}_metrics.json").write_text(
                    json.dumps(metrics), encoding="utf-8"
                )
            histories = figures.read_fold_histories(base)
            self.assertEqual(histories[3][1], 2)
            path = base / "losses.png"
            figures.plot_training_histories(histories, path)
            self.assertTrue(path.is_file())
            self.assertGreater(path.stat().st_size, 2000)

    def test_event_errors_global_median_uses_individual_events(self):
        with tempfile.TemporaryDirectory() as folder:
            base = Path(folder)
            errors = (0., 60., 900., 1800.)
            for fold, error in enumerate(errors, start=1):
                path = base / f"fold_{fold:02d}_event_metrics.csv"
                with path.open("w", newline="", encoding="utf-8") as handle:
                    writer = csv.DictWriter(handle, fieldnames=["absolute_error_s"])
                    writer.writeheader()
                    writer.writerow({"absolute_error_s": str(error)})
            per_fold = figures.read_event_errors(base)
            out, median, accurate, total = figures.plot_event_error_distribution(
                per_fold, base / "errors.png"
            )
            self.assertTrue(out.is_file())
            self.assertEqual(total, 4)
            self.assertEqual(accurate, 2)
            self.assertAlmostEqual(median, float(np.median(errors)))
            self.assertAlmostEqual(median, 480.)


    def test_second_third_comparison_uses_identical_fold_counts(self):
        with tempfile.TemporaryDirectory() as folder:
            previous = [
                {
                    "fold": k,
                    "val_evaluable_events": 5,
                    "median_abs_peak_error_s": 800. + k,
                    "fraction_within_60s": .05
                } for k in range(1, 5)
            ]
            third = [
                {
                    "fold": k,
                    "val_evaluable_events": 5,
                    "median_abs_peak_error_s": 700. + k,
                    "fraction_within_60s": .15
                } for k in range(1, 5)
            ]
            path = figures.plot_fold_comparison(
                third, previous, Path(folder) / "compare.png"
            )
            self.assertTrue(path.is_file())
            previous[0]["val_evaluable_events"] = 6
            with self.assertRaises(ValueError):
                figures.plot_fold_comparison(
                    third, previous, Path(folder) / "bad.png"
                )



    def test_paired_event_diagnostics_preserve_fold_and_timestamp(self):
        with tempfile.TemporaryDirectory() as directory:
            parent = Path(directory)
            previous = parent / "previous"
            current = parent / "current"
            previous.mkdir()
            current.mkdir()
            for fold in range(1, 5):
                for folder, offsets in (
                    (previous, [900., 30., -1600.]),
                    (current, [120., 90., -1500.])
                ):
                    with (
                        folder / f"fold_{fold:02d}_event_metrics.csv"
                    ).open("w", newline="", encoding="utf-8") as handle:
                        writer = csv.DictWriter(handle, fieldnames=[
                            "mpb_timestamp_utc", "source_group", "n_windows",
                            "peak_offset_s", "absolute_error_s"
                        ])
                        writer.writeheader()
                        for i, offset in enumerate(offsets):
                            writer.writerow({
                                "mpb_timestamp_utc":
                                    f"2020-0{fold}-{i + 1:02d}T00:00:00+00:00",
                                "source_group": "reference",
                                "n_windows": 600,
                                "peak_offset_s": offset,
                                "absolute_error_s": abs(offset),
                            })
            old = figures.load_localization_events(previous)
            new = figures.load_localization_events(current)
            stats, rows = figures.paired_localization_statistics(old, new)
            self.assertEqual(stats["n_paired_events"], 12)
            self.assertEqual(stats["improved_events"], 8)
            self.assertEqual(stats["worsened_events"], 4)
            self.assertEqual(stats["previous_within_60s"], 4)
            self.assertEqual(stats["current_within_60s"], 0)
            self.assertEqual(stats["previous_only_within_60s"], 4)
            self.assertEqual(stats["current_only_within_60s"], 0)
            self.assertEqual(
                stats["previous_peaks_near_interval_edges_25to30min"], 4
            )
            self.assertEqual(
                stats["current_peaks_near_interval_edges_25to30min"], 4
            )
            image, report = figures.save_paired_localization(
                rows, stats, parent, fifth=True
            )
            self.assertTrue(image.is_file())
            self.assertTrue(report.is_file())
            self.assertTrue(
                (parent / "mag_cv_paired_fourth_vs_fifth.csv").is_file()
            )

    def test_paired_comparison_rejects_unmatched_or_different_samples(self):
        key = (1, "2020-01-01T00:00:00+00:00")
        previous = {key: {
            "fold": 1, "source_group": "group1", "n_windows": 600,
            "absolute_error_s": 800., "peak_offset_s": 800.
        }}
        current = {}
        with self.assertRaisesRegex(ValueError, "Unmatched validation MPBs"):
            figures.paired_localization_statistics(previous, current)
        current[key] = dict(previous[key])
        current[key]["n_windows"] = 590
        with self.assertRaisesRegex(ValueError, "evaluation-center count"):
            figures.paired_localization_statistics(previous, current)


if __name__ == "__main__":
    unittest.main()
