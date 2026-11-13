#!/usr/bin/env python3
"""Offline tests for merged train supports and resumable fold statistics.

No BigQuery jobs, source-measurement files or authentication are required.
"""
from __future__ import annotations

import sys
import unittest
from datetime import date, datetime, timedelta, timezone
from importlib import import_module
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
fit = import_module("scripts.4_preprocessing.4c_fit_fold_normalizers")


def event(time, fold, *, mag=True, swea=True):
    return {
        "time": datetime.fromisoformat(time.replace("Z", "+00:00")),
        "fold": fold,
        "mag_available": "1" if mag else "0",
        "swea_available": "1" if swea else "0",
    }


class FoldNormalizerPlanTests(unittest.TestCase):
    def test_overlapping_train_events_count_once_in_same_day(self):
        rows = [
            event("2018-04-12T12:00:00Z", 2),
            event("2018-04-12T12:15:00Z", 3),
            event("2018-04-12T12:30:00Z", 1),  # held-out; no inclusion
        ]
        plan, provenance, count = fit.plan_training_days(rows, 1, "MAG")
        self.assertEqual(count, 2)
        self.assertEqual(len(plan), 1)
        (start, end), = plan[date(2018, 4, 12)]
        self.assertEqual(start.isoformat(), "2018-04-12T11:25:00+00:00")
        self.assertEqual(end.isoformat(), "2018-04-12T12:50:00+00:00")
        self.assertEqual(provenance[date(2018, 4, 12)], {2, 3})

    def test_split_at_midnight_and_union_no_duplicate_timestamp(self):
        rows = [
            event("2018-12-31T23:50:00Z", 2),
            event("2019-01-01T00:10:00Z", 3),
        ]
        plan, _, count = fit.plan_training_days(rows, 1, "MAG")
        self.assertEqual(count, 2)
        self.assertEqual(list(plan), [date(2018,12,31), date(2019,1,1)])
        self.assertEqual(len(plan[date(2018,12,31)]), 1)
        self.assertEqual(len(plan[date(2019,1,1)]), 1)
        self.assertEqual(
            plan[date(2018,12,31)][0][1],
            datetime(2019,1,1,tzinfo=timezone.utc)
        )
        self.assertEqual(
            plan[date(2019,1,1)][0][0],
            datetime(2019,1,1,tzinfo=timezone.utc)
        )

    def test_missing_swea_event_is_excluded_not_mag(self):
        rows = [
            event("2020-03-01T12:00:00Z", 2, swea=False),
            event("2020-03-02T12:00:00Z", 3),
        ]
        swea_plan, _, swea_count = fit.plan_training_days(rows, 1, "SWEA")
        mag_plan, _, mag_count = fit.plan_training_days(rows, 1, "MAG")
        self.assertEqual((swea_count, mag_count), (1, 2))
        self.assertEqual(len(swea_plan), 1)
        self.assertEqual(len(mag_plan), 2)

    def test_sql_is_partition_pruned_and_half_open(self):
        day = date(2020, 1, 1)
        start = datetime(2020, 1, 1, 12, tzinfo=timezone.utc)
        intervals = [
            (start, start + timedelta(minutes=70)),
            (start + timedelta(hours=3), start + timedelta(hours=4)),
        ]
        for instrument, timestamp_col in (
            ("MAG", "datetime_utc"), ("SWEA", "d.datetime_utc")
        ):
            sql, conf = fit.day_sql_and_config(
                instrument, day, intervals, dry_run=True, max_bytes=2**30
            )
            self.assertIn(f"DATE({timestamp_col}) = @day", sql)
            self.assertIn(f"{timestamp_col} < @end_1", sql)
            self.assertIn(" OR ", sql)
            self.assertNotIn("BETWEEN @start AND @end", sql)
            self.assertNotIn("UNNEST(@days)", sql)
            self.assertTrue(conf.dry_run)
            self.assertEqual(len(conf.query_parameters), 5+(2 if instrument=="SWEA" else 0))

    def test_checkpoint_restores_accumulation_without_double_counting(self):
        a = np.array([[1.,2.,3.,4.], [2.,3.,4.,5.]])
        b = np.array([[10.,20.,30.,40.], [12.,22.,32.,42.]])
        n = fit.FoldChannelNormalizer("MAG", 1)
        n.update(a, source_fold=2)
        state = fit.checkpoint_payload(
            n, "signature", {"2018-01-01"}, 0, None, "MAG", 1
        )
        restored = fit.restore_normalizer(state, "MAG", 1)
        # Before freezing, reject held-out validation observations explicitly.
        with self.assertRaises(ValueError):
            restored.update(a, source_fold=1)
        # The invalid update must not alter accumulated training statistics.
        self.assertEqual(restored.count, len(a))
        restored.update(b, source_fold=3)
        restored.freeze()
        np.testing.assert_allclose(restored.mean, np.vstack([a,b]).mean(0))
        np.testing.assert_allclose(restored.scale, np.vstack([a,b]).std(0))
        self.assertEqual(restored.count, 4)
        # After freezing, every attempt to refit is rejected.
        with self.assertRaises(RuntimeError):
            restored.update(a, source_fold=2)


if __name__ == "__main__":
    unittest.main()
