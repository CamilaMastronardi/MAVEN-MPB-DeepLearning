"""Pruebas offline de la morfología de score: sin red, cachés o BigQuery."""
import unittest
import numpy as np

from src.model.score_shape import score_shape_metrics


class ScoreShapeTests(unittest.TestCase):
    def test_peak_width_prominence_and_prediction_offset(self):
        x = np.arange(-128, 129, 32, dtype=float)
        y = np.array([.1, .2, .3, .6, .9, .6, .3, .2, .1])
        m = score_shape_metrics(x, y, competitor_distance_s=96)
        self.assertEqual(m["peak_offset_s"], 0.0)
        self.assertEqual(m["absolute_error_s"], 0.0)
        self.assertAlmostEqual(m["peak_prominence"], .8)
        self.assertAlmostEqual(m["width_half_prominence_s"], 85.33333333, places=5)
        self.assertAlmostEqual(m["peak_vs_competitor"], .7)
        self.assertEqual(m["n_segments"], 1)
        self.assertFalse(m["peak_at_segment_edge"])

    def test_secondary_local_peak_and_remote_gap(self):
        x = np.arange(-192, 193, 32, dtype=float)
        y = np.array([.1, .15, .65, .2, .1, .3, .95,
                      .25, .1, .2, .75, .1, .05])
        m = score_shape_metrics(x, y, competitor_distance_s=96)
        self.assertEqual(m["n_competing_peaks"], 2)
        self.assertAlmostEqual(m["peak_vs_competitor"], .20)

    def test_gap_edge_has_no_width(self):
        x = [-96, -64, -32, 160, 192, 224]
        y = [.1, .3, .95, .2, .4, .2]
        m = score_shape_metrics(x, y)
        self.assertEqual(m["n_segments"], 2)
        self.assertTrue(m["peak_at_segment_edge"])
        self.assertTrue(np.isnan(m["width_half_prominence_s"]))
        self.assertTrue(np.isnan(m["peak_prominence"]))

    def test_flat_and_single_center(self):
        m = score_shape_metrics([-32, 0, 32], [.5, .5, .5],
                                competitor_distance_s=30)
        self.assertTrue(np.isnan(m["width_half_prominence_s"]))
        self.assertEqual(m["peak_offset_s"], -32.0)
        self.assertEqual(m["peak_vs_competitor"], 0.)
        one = score_shape_metrics([64], [.8])
        self.assertEqual(one["absolute_error_s"], 64.)
        self.assertTrue(np.isnan(one["peak_vs_competitor"]))

    def test_invalid_time_duplicates_and_empty(self):
        with self.assertRaises(ValueError):
            score_shape_metrics([0, 0], [.1, .2])
        with self.assertRaises(ValueError):
            score_shape_metrics([0, 32], [.1])
        self.assertIsNone(score_shape_metrics([np.nan], [.3]))
        self.assertEqual(score_shape_metrics([32, -32, 0], [.2, .1, .9])
                         ["peak_offset_s"], 0.)


if __name__ == "__main__":
    unittest.main()
