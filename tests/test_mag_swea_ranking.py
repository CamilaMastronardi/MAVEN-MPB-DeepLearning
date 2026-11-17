"""Offline tests for seventh MAG+SWEA v7b event ranking and mining."""
from __future__ import annotations

import unittest
import csv
import tempfile
from importlib import import_module
from pathlib import Path
from datetime import datetime, timedelta, timezone
from types import SimpleNamespace

import numpy as np
import torch

from src.data.mpb_weak_regions import (
    candidate_ranking_grid, classify_candidate
)
from src.detection.mpb_event_ranking import (
    event_pairwise_ranking_loss, numpy_event_rank_metrics,
    select_rank_event_groups, event_group_batches, update_hard_memory
)


class SeventhRankExperimentTests(unittest.TestCase):
    def setUp(self):
        self.event = datetime(2020, 1, 1, tzinfo=timezone.utc)

    def _example(self, label, shift):
        candidate = SimpleNamespace(
            transition=label,
            center_s=float(self.event.timestamp() + shift),
            left_region=0, right_region=(1 if label else 0),
        )
        return (candidate, object())

    def _candidate_pool(self):
        return {
            self.event: [
                self._example(1, -30),
                self._example(1, 0),
                self._example(1, 30),
                *(self._example(0, x) for x in (
                    -1500, -1200, -900, -600,
                    600, 900, 1200, 1500
                )),
            ]
        }

    def test_positive_above_negative_has_lower_ranking_loss(self):
        true = torch.tensor([2., 1., -1., -.5], requires_grad=True)
        labels = torch.tensor([1., 1., 0., 0.])
        ids = ["one"]*4
        good, accuracy, top1, count = event_pairwise_ranking_loss(
            true, labels, ids, margin=.5
        )
        incorrect, wrong_accuracy, _, _ = event_pairwise_ranking_loss(
            -true, labels, ids, margin=.5
        )
        self.assertLess(good.item(), incorrect.item())
        self.assertEqual(accuracy, 1.)
        self.assertEqual(wrong_accuracy, 0.)
        self.assertEqual(top1, 1.)
        self.assertEqual(count, 1)
        good.backward()
        self.assertTrue(torch.isfinite(true.grad).all())
        self.assertGreater(float(true.grad[2]), 0.)
        self.assertLess(float(true.grad[0]), 0.)

    def test_event_ranking_weights_events_not_number_of_negatives(self):
        logits = torch.tensor([2., -1., 1., 0., -1.])
        labels = torch.tensor([1., 0., 1., 0., 0.])
        events = ["a", "a", "b", "b", "b"]
        loss, _, _, n = event_pairwise_ranking_loss(
            logits, labels, events, margin=.5
        )
        one = np.logaddexp(0, .5-1-2)
        second = np.mean(np.logaddexp(0, .5+np.array([0., -1.])-1.))
        self.assertAlmostEqual(float(loss), float((one+second)/2), 6)
        self.assertEqual(n, 2)

    def test_single_class_event_cannot_be_silently_ranked(self):
        with self.assertRaisesRegex(ValueError, "positives and negatives"):
            event_pairwise_ranking_loss(
                torch.tensor([1., 2.]), torch.tensor([1., 1.]),
                ["event", "event"],
            )

    def test_numpy_inner_monitor_metric_matches_torch_ranking(self):
        logits = torch.tensor([2., -.1, 1., 0.])
        y = torch.tensor([1., 0., 1., 0.])
        events = ["a", "a", "b", "b"]
        expected, _, _, _ = event_pairwise_ranking_loss(
            logits, y, events, margin=.5
        )
        actual = numpy_event_rank_metrics(
            list(zip(events, y.tolist(), logits.tolist())),
            margin=.5
        )
        self.assertAlmostEqual(actual["rank"], float(expected), 6)
        self.assertEqual(actual["event_top1"], 1.)
        self.assertEqual(actual["n_rank_events"], 2)

    def test_training_grid_never_promotes_uncertain_mpb_or_bs(self):
        row = {
            "time": self.event,
            "bs_time": self.event-timedelta(minutes=30),
        }
        t = self.event.timestamp()
        bs = row["bs_time"].timestamp()
        labels = candidate_ranking_grid(
            row, all_mpb_s=np.array([t]),
            all_bs_s=np.array([bs]), step_s=120.
        )
        self.assertTrue(any(c.transition == 1 for c in labels))
        self.assertTrue(any(c.transition == 0 for c in labels))
        self.assertTrue(all(c.transition is not None for c in labels))
        self.assertTrue(all(
            abs(c.center_s - t) <= 35. or abs(c.center_s - t) >= 600.
            for c in labels
        ))
        contaminant = classify_candidate(
            row, t-1200.,
            all_mpb_s=[t, t-1300.],
            all_bs_s=[bs],
        )
        self.assertIsNone(contaminant.transition)

    def test_online_hard_negative_memory_beats_uniform_controls(self):
        pool = self._candidate_pool()
        groups = select_rank_event_groups(
            pool, epoch=1, seed=42, hard_memory={},
            n_positives=2, n_hard=2, n_random=2,
        )
        self.assertEqual(len(groups), 1)
        examples = groups[0][1]
        self.assertEqual(sum(x[3] == 1 for x in examples), 2)
        self.assertEqual(sum(x[3] == 0 for x in examples), 4)
        logits = torch.tensor([
            (5. if item[3] == 0 and item[1].center_s ==
             self.event.timestamp()-600 else float(i)/100.)
            for i, item in enumerate(examples)
        ])
        memory = {}
        update_hard_memory(memory, examples, logits, n_hard=2)
        self.assertEqual(len(memory[self.event.isoformat()]), 2)
        next_groups = select_rank_event_groups(
            pool, epoch=2, seed=42, hard_memory=memory,
            n_positives=2, n_hard=2, n_random=2,
        )
        next_neg = [
            item[1].center_s for item in next_groups[0][1]
            if item[3] == 0
        ]
        self.assertTrue(set(memory[self.event.isoformat()]) <= set(next_neg))
        self.assertEqual(len(set(next_neg)), 4)

    def test_whole_event_batching_never_breaks_pairs(self):
        first = self.event
        second = first+timedelta(days=1)
        events = [
            (first, [(first, object(), None, y, (0, y)) for y in (1, 0, 0)]),
            (second, [(second, object(), None, y, (0, y)) for y in (1, 1, 0)]),
        ]
        batches = list(event_group_batches(events, batch_size=4))
        self.assertEqual([len(x) for x in batches], [3, 3])
        self.assertEqual(len({x[0] for x in batches[0]}), 1)
        with self.assertRaisesRegex(ValueError, "Batch size too small"):
            list(event_group_batches(events, batch_size=2))

    def test_paired_plot_matches_utc_z_and_offset_spelling(self):
        plot = import_module("scripts.7_figures.7a_plot_mag_swea_results")
        with tempfile.TemporaryDirectory() as folder:
            for subpath, when in [
                ("old", "2020-01-01T00:00:00Z"),
                ("rank", "2020-01-01T00:00:00+00:00"),
            ]:
                location = Path(folder)/subpath
                location.mkdir()
                with (location/"fold_01_event_metrics.csv").open(
                    "w", newline="", encoding="utf-8"
                ) as output:
                    writer = csv.DictWriter(
                        output,
                        fieldnames=("mpb_timestamp_utc", "absolute_error_s")
                    )
                    writer.writeheader()
                    writer.writerow({
                        "mpb_timestamp_utc": when, "absolute_error_s": "20."
                    })
            old_events = plot.load_events(Path(folder)/"old")
            rank_events = plot.load_events(Path(folder)/"rank")
            self.assertEqual(old_events, rank_events)

    def test_hard_negative_memory_is_event_and_fold_local(self):
        groups = select_rank_event_groups(
            self._candidate_pool(), epoch=5, seed=7,
            hard_memory={self.event.isoformat(): [
                self.event.timestamp() - 1500.
            ]},
        )
        self.assertEqual(len(groups), 1)
        self.assertTrue(any(
            x[3] == 0 and x[1].center_s == self.event.timestamp()-1500.
            for x in groups[0][1]
        ))
        self.assertNotIn(
            (self.event+timedelta(days=1)).isoformat(),
            {self.event.isoformat(): [1.]},
        )


if __name__ == "__main__":
    unittest.main()
