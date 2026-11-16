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


if __name__ == "__main__":
    unittest.main()
