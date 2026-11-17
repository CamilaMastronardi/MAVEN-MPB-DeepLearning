"""Unit tests for V10-HP2 monitor calibration and offline HP2A rows."""
from __future__ import annotations

import unittest
from importlib import import_module

v10=import_module("scripts.5_training.5f_train_mag_swea_refiner_v10")
hp2a=import_module("scripts.5_training.5m_recalibrate_v10_hp2a")


def record(coarse,offset):
    return {
        "coarse_offset_s":float(coarse),
        "refined_mean_offset_s":float(offset),
        "refined_peak_offset_s":float(offset),
    }


class HP2CalibrationTests(unittest.TestCase):
    def test_grid_includes_zero_and_one(self):
        grid=v10.calibration_alpha_grid(.1)
        self.assertEqual(len(grid),11)
        self.assertEqual(grid[0],0.)
        self.assertEqual(grid[-1],1.)
        self.assertAlmostEqual(grid[7],.7)
        self.assertEqual(v10.calibration_alpha_grid(.25),
                         v10.CALIBRATION_ALPHAS)

    def test_invalid_alpha_step(self):
        for invalid in (0.,-0.1,1.3,0.3):
            with self.assertRaises(ValueError):
                v10.calibration_alpha_grid(invalid)

    def test_within_60_optimized_before_mae(self):
        # A time correction helps three events cross the 60s threshold
        # but makes one large error worse. Distinct objectives must differ.
        monitor=[
            record(70.,-20.),
            record(-70.,20.),
            record(70.,-20.),
            record(500.,240.),
        ]
        legacy=v10.calibrate_refinement(monitor)
        tuned=v10.calibrate_refinement(
            monitor,goal="within_60",
            alphas=v10.calibration_alpha_grid(.1))
        self.assertEqual(legacy["alpha"],0.)
        self.assertGreater(tuned["alpha"],0.)
        self.assertEqual(legacy["monitor_within_60s"],0.)
        self.assertEqual(tuned["monitor_within_60s"],.75)
        self.assertEqual(tuned["calibration_goal"],"within_60")

    def test_fallback_when_all_corrections_harmful(self):
        monitor=[record(10.,200.),record(-15.,-210.),
                 record(25.,100.)]
        calibrated=v10.calibrate_refinement(
            monitor,goal="within_60",
            alphas=v10.calibration_alpha_grid(.1))
        self.assertEqual(calibrated["alpha"],0.)

    def test_paired_metrics_threshold(self):
        summary=hp2a.paired_summary([40,65,90],[70,55,100])
        self.assertEqual(summary["n_paired_events"],3)
        self.assertAlmostEqual(summary["hp2a_within_60s"],1/3)
        self.assertAlmostEqual(summary["hp1_within_60s"],1/3)
        self.assertEqual(summary["improved_over_hp1"],2)
        self.assertEqual(summary["worsened_over_hp1"],1)

    def test_outer_prediction_conversion(self):
        old=[{
            "mpb_timestamp_utc":"2020-01-01T00:00:00+00:00",
            "source_group":"Group1",
            "coarse_peak_offset_s":"42",
            "refiner_mean_offset_s":"-5",
            "refiner_peak_offset_s":"-12",
            "refiner_confidence":"0.7",
            "n_windows":"108"
        }]
        converted=hp2a.from_hp1_outer_rows(old)
        self.assertEqual(converted[0]["coarse_offset_s"],42.)
        self.assertEqual(converted[0]["refined_peak_offset_s"],-12.)


if __name__=="__main__":
    unittest.main()
