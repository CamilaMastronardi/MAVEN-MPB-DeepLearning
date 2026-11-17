"""Regression tests for controlled V7b rank-margin sweep comparisons."""
from __future__ import annotations

import csv
import importlib.util
import json
import subprocess
import sys
import tempfile
import unittest
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
SCRIPT = ROOT / "scripts/5_training/compare_v7b_rank_margin_sweep.py"


def get_comparator():
    spec = importlib.util.spec_from_file_location("v7b_rank_comparison", SCRIPT)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


class SweepTest(unittest.TestCase):
    def test_margin_invariant_monitor_selection_and_event_pairing(self):
        with tempfile.TemporaryDirectory() as d:
            root = Path(d)
            for margin, folder in ((.25, "margin_0p25"), (.5, "margin_0p5"),
                                   (1., "margin_1"), (2., "margin_2")):
                run = root / folder
                run.mkdir()
                common = dict(
                    seed=42, learning_rate=.0003, batch_size=16,
                    max_epochs=100, manifest_sha256="test_digest",
                    reconstruction_weight=.15, region_weight=.5,
                    rank_weight=2., rank_control_step_s=120,
                    rank_positives_per_event=2,
                    rank_hard_negatives=2,
                    rank_random_negatives=2,
                )
                config = {**common, "rank_margin": margin,
                          "swea_fixed_energy_ev": [30., 300.],
                          "evaluation_stride_s": 32.,
                          "epochs_min": 10, "patience": 8}
                (run / "cv_summary.json").write_text(json.dumps({
                    "architecture": "dual-mag-swea-event-ranking-v1",
                    "training_configuration": config,
                    "requested_folds": [1], "completed_folds": [1],
                    "pilot_subset": True,
                }))
                config_fold = {**common, "rank_margin": margin,
                               "early_stop_min_epochs": 10,
                               "early_stop_patience": 8,
                               "evaluation_stride_s": 32.,
                               "swea_energy_ev": [30., 300.],
                               "mag_swea_native_time_grid": True}
                (run / "fold_01_metrics.json").write_text(json.dumps({
                    "fold": 1,
                    "architecture": "dual-mag-swea-event-ranking-v1",
                    "training_configuration": config_fold,
                    "val_total_events": 3, "val_evaluable_events": 2,
                    "monitor_event_top1_fraction": 1. - .2*abs(margin-1.),
                    "monitor_event_pair_accuracy": .8,
                    "monitor_event_rank_loss": .4,
                    "train_events": 20, "inner_monitor_events": 5,
                    "train_candidate_events": 20,
                    "monitor_candidate_events": 5,
                    "monitor_ranking_event_count": 5,
                    "train_skip_counts": {},
                    "monitor_skip_counts": {}, "val_skip_counts": {},
                    "normalizers": {"MAG": {"mean": [0], "scale": [1]},
                                    "SWEA": {"mean": [0], "scale": [1]}},
                    "best_epoch": 12,
                }))
                with (run / "fold_01_event_metrics.csv").open("w", newline="") as f:
                    writer = csv.DictWriter(f, fieldnames=[
                        "mpb_timestamp_utc", "absolute_error_s", "n_windows",
                        "source_group"])
                    writer.writeheader()
                    for i, error in enumerate((40*margin, 150*margin, "")):
                        writer.writerow({
                            "mpb_timestamp_utc": f"2020-01-0{i+1}T00:00:00+00:00",
                            "absolute_error_s": error, "n_windows": 45,
                            "source_group": "Group1"})
            subprocess.run([sys.executable, str(SCRIPT),
                            "--results-root", str(root), "--folds", "1"],
                           check=True, capture_output=True, text=True)
            summary = json.loads(
                (root / "nested_inner_selected_summary.json").read_text())
            self.assertEqual(summary["selected_margins_by_inner_monitor"],
                             {"1": 1.0})
            self.assertAlmostEqual(
                summary["inner_selected_outer_metrics"]["within_120s_over_all"],
                1/3)
            self.assertEqual(summary["common_evaluable_events_all_margins"], 2)
            # A different manifest hash must abort before comparing performance.
            result = root / "margin_2" / "cv_summary.json"
            changed = json.loads(result.read_text())
            changed["training_configuration"]["manifest_sha256"] = "different"
            result.write_text(json.dumps(changed))
            failed = subprocess.run([sys.executable, str(SCRIPT),
                                     "--results-root", str(root), "--folds", "1"],
                                    capture_output=True, text=True)
            self.assertNotEqual(failed.returncode, 0)
            self.assertIn("manifest_sha256", failed.stderr)


if __name__ == "__main__":
    unittest.main()