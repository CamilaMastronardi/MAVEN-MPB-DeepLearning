#!/usr/bin/env python3
"""Offline unit tests for 10-minute MAG and native-cadence SWEA windows.

No credentials, BigQuery calls, or MAVEN measurements required.
Run from repository root with: python -m unittest discover -s tests -v
"""
from __future__ import annotations

import sys
import unittest
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "scripts"))
from windowing import WindowStats, iter_native_windows, nearest_label_offset_s


class NativeWindowTests(unittest.TestCase):
    def test_mag_one_second_stride_and_gaussian_labels(self) -> None:
        times = np.arange(-2200.0, 2201.0, 1.0) + 1_500_000_000
        values = np.ones((len(times), 4))
        event = 1_500_000_000.0
        stats = WindowStats()
        windows = iter_native_windows(
            times, values, center_label_s=event,
            split_labels_s=np.array([event]), stats=stats
        )
        last = None
        central = None
        n = 0
        for window in windows:
            n += 1
            if last is not None:
                self.assertAlmostEqual(window.center_s-last, 1.0)
            last = window.center_s
            if window.center_s == event:
                central = window
        self.assertEqual(n, 3601)
        self.assertEqual(stats.candidates, 3601)
        self.assertEqual(stats.accepted, 3601)
        self.assertIsNotNone(central)
        self.assertEqual(central.data.shape, (601, 4))
        self.assertAlmostEqual(central.detection_score, 1.0)

    def test_swea_32_s_cadence_variable_native_length(self) -> None:
        event = 1_500_000_000.0
        times = np.arange(-2304.0, 2305.0, 32.0) + event
        values = np.full((len(times), 9), 5.0)
        windows = list(iter_native_windows(
            times, values, center_label_s=event,
            split_labels_s=np.array([event])
        ))
        self.assertGreater(len(windows), 50)
        self.assertTrue(all(w.data.shape[1] == 9 for w in windows))
        self.assertTrue(all(18 <= len(w.data) <= 19 for w in windows))
        self.assertEqual(windows[1].center_s-windows[0].center_s, 32.0)

    def test_bad_gap_rejected_without_interpolation(self) -> None:
        event = 1_500_000_000.0
        times = np.arange(-2200.0, 2201.0, 1.0) + event
        keep = (times < event - 25) | (times > event + 25)
        times = times[keep]
        values = np.ones((len(times), 4))
        stats = WindowStats()
        windows = list(iter_native_windows(
            times, values, center_label_s=event,
            split_labels_s=np.array([event]), stats=stats
        ))
        self.assertGreater(stats.large_gap, 0)
        self.assertEqual(
            len(windows), stats.candidates-stats.incomplete-stats.large_gap
        )
        for window in windows:
            self.assertLessEqual(np.diff(window.relative_times_s).max(), 2.5)

    def test_invalid_flux_rejected(self) -> None:
        event = 1_500_000_000.0
        times = np.arange(-2200.0, 2201.0, 1.0) + event
        flux = np.ones((len(times), 12))
        flux[np.argmin(abs(times-event)), 3] = np.nan
        stats = WindowStats()
        list(iter_native_windows(
            times, flux, center_label_s=event,
            split_labels_s=np.array([event]), stats=stats
        ))
        self.assertGreater(stats.invalid_data, 0)

    def test_incomplete_windows_rejected(self) -> None:
        event = 1_500_000_000.0
        times = np.arange(-200.0, 201.0, 1.0) + event
        values = np.zeros((len(times), 4))
        stats = WindowStats()
        self.assertEqual(len(list(iter_native_windows(
            times, values, center_label_s=event,
            split_labels_s=np.array([event]), stats=stats
        ))), 0)
        self.assertEqual(stats.incomplete, len(times))

    def test_nearest_label_avoids_mislabeling_close_events(self) -> None:
        event = 1_500_000_000.0
        labels = np.array([event, event+16.0])
        self.assertEqual(nearest_label_offset_s(event+15.0, labels), -1.0)
        self.assertEqual(nearest_label_offset_s(event+5.0, labels), 5.0)


if __name__ == "__main__":
    unittest.main()
