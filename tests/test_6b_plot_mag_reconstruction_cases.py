"""Offline tests for the MAG diagnostic figure metadata and selection only.

No synthetic MAVEN measurements, model training, or BigQuery requests.
"""
import importlib
import json
import tempfile
from types import SimpleNamespace
import numpy as np
import unittest
from pathlib import Path
from datetime import datetime, timezone

figure = importlib.import_module(
    "scripts.6_figures.6b_plot_mag_reconstruction_cases"
)


class MagCaseSelectionTests(unittest.TestCase):
    def test_choose_representative_good_and_bad(self):
        events = [
            {"fold": 1, "time": datetime(2020, 1, d, tzinfo=timezone.utc),
             "error_s": err, "saved_offset_s": err, "source_group": "test"}
            for d, err in [(1, 8.), (2, 28.), (3, 51.),
                           (4, 650.), (5, 1200.), (6, 1700.)]
        ]
        good, bad = figure.select_cases(events, 60., 600.)
        self.assertEqual(good["error_s"], 28.)
        self.assertEqual(bad["error_s"], 1200.)

    def test_recomputed_event_peak_must_match_saved_offset(self):
        t = datetime(2020, 1, 1, tzinfo=timezone.utc)
        event = {"fold": 2, "time": t, "saved_offset_s": 10.}
        predicted = figure.verify_saved_prediction(event, t.timestamp() + 10.)
        self.assertAlmostEqual(predicted, 10.)
        with self.assertRaises(RuntimeError):
            figure.verify_saved_prediction(event, t.timestamp() + 100.)

    def test_second_training_reads_600_centers_from_summary(self):
        with tempfile.TemporaryDirectory() as folder:
            base = Path(folder)
            summary = {
                "instrument": "MAG",
                "pilot_subset": False,
                "completed_folds": [1, 2, 3, 4],
                "training_configuration": {
                    "val_centers_per_event": 600, "sigma_seconds": 60.0
                },
            }
            (base / "cv_summary.json").write_text(
                json.dumps(summary), encoding="utf-8"
            )
            centers, sigma = figure.load_evaluation_config(base)
            self.assertEqual(centers, 600)
            self.assertEqual(sigma, 60.)

    def test_first_training_uses_saved_original_defaults(self):
        with tempfile.TemporaryDirectory() as folder:
            base = Path(folder)
            summary = {
                "instrument": "MAG", "pilot_subset": False,
                "completed_folds": [1, 2, 3, 4]
            }
            (base / "cv_summary.json").write_text(
                json.dumps(summary), encoding="utf-8"
            )
            centers, sigma = figure.load_evaluation_config(base)
            self.assertEqual(centers, 121)
            self.assertEqual(sigma, 60.)

    def test_missing_good_or_bad_case(self):
        t = datetime(2020, 1, 1, tzinfo=timezone.utc)
        events = [{"fold": 1, "time": t, "error_s": 900.}]
        with self.assertRaises(ValueError):
            figure.select_cases(events, 60., 600.)


    def test_magnitude_diagnostics_are_in_physical_nt(self):
        """Pure vector algebra fixture, not simulated or downloaded MAG data."""
        observed = np.array([[3., 4., 0., 5.],
                             [3., 4., 0., 5.]])
        reconstructed = np.array([[6., 8., 0., 5.],
                                  [np.nan, np.nan, np.nan, np.nan]])
        metrics = figure.magnitude_diagnostics(observed, reconstructed)
        self.assertAlmostEqual(metrics["rmse_decoded_magnitude_nt"], 0.)
        self.assertAlmostEqual(metrics["rmse_vector_magnitude_nt"], 5.)
        self.assertAlmostEqual(metrics["rmse_magnitude_consistency_nt"], 5.)



    def test_sixth_reconstruction_uses_native_epoch_grid(self):
        """Tiny arrays validate figure alignment, not synthetic physics data."""
        from src.detection.mag_dense_temporal_autoencoder import (
            MagDenseTemporalAutoencoder
        )
        torch = importlib.import_module("torch")
        torch.set_num_threads(1)

        class FixedNormalizer:
            instrument = "MAG"
            is_frozen = True
            n_features = 4
            mean = np.zeros(4)
            scale = np.ones(4)

            @staticmethod
            def transform(data):
                return data.astype(np.float32)

        times = np.arange(-7., 8.)
        values = np.zeros((len(times), 4), dtype=np.float64)
        windows = [
            SimpleNamespace(
                center_s=center, data=values,
                relative_times_s=times - center,
                detection_score=1., offset_from_nearest_mpb_s=center
            ) for center in (0., 1.)
        ]
        model = MagDenseTemporalAutoencoder()
        model.eval()
        output = figure.reconstruct_block(
            model, FixedNormalizer(), times, windows, batch_size=2,
            event_time_s=0., sigma_seconds=60.
        )
        self.assertEqual(output["n_scored_windows"], 2)
        self.assertEqual(len(output["scores"]), len(times))
        np.testing.assert_array_equal(output["centers_s"], times)
        self.assertTrue(np.isfinite(output["scores"]).all())


if __name__ == "__main__":
    unittest.main()
