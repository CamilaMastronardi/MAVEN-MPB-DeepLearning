"""Offline tests for the MAG diagnostic figure metadata and selection only.

No synthetic MAVEN measurements, model training, or BigQuery requests.
"""
import importlib
import unittest
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

    def test_missing_good_or_bad_case(self):
        t = datetime(2020, 1, 1, tzinfo=timezone.utc)
        events = [{"fold": 1, "time": t, "error_s": 900.}]
        with self.assertRaises(ValueError):
            figure.select_cases(events, 60., 600.)


if __name__ == "__main__":
    unittest.main()
