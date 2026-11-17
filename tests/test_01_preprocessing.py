"""Chequeos de reproducibilidad de la division FIT/MONITOR de V7b."""
from __future__ import annotations
import unittest
from datetime import datetime, timezone
from src.preprocessing.cv_splits import check_disjoint_supports, split_inner_monitor_months

def event(month, day=15):
    t = datetime(2020, month, day, tzinfo=timezone.utc)
    return {"time": t, "month_block": t.strftime("%Y-%m")}

class PreprocessingTests(unittest.TestCase):
    def test_month_split_deterministic_and_disjoint(self):
        rows = [event(m) for m in range(1, 9)]
        a = split_inner_monitor_months(rows, fold=1, seed=42)
        b = split_inner_monitor_months(rows, fold=1, seed=42)
        self.assertEqual(a, b)
        fit, monitor, _ = a
        self.assertTrue(fit and monitor)
        self.assertTrue({r["month_block"] for r in fit}.isdisjoint(
            {r["month_block"] for r in monitor}))
        check_disjoint_supports(fit, monitor)

    def test_reject_overlapping_context(self):
        with self.assertRaises(ValueError):
            check_disjoint_supports([event(1, 15)], [event(1, 16)])

if __name__ == "__main__":
    unittest.main()
