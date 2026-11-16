"""Offline smoke tests for MAG CV training wiring; no BigQuery calls or training."""
import importlib
import unittest
from datetime import datetime, timedelta, timezone

import torch

MagSupervisedAutoencoder = importlib.import_module(
    "src.detection.mag_autoencoder"
).MagSupervisedAutoencoder
trainer = importlib.import_module("scripts.5_training.5a_train_mag_cv")


class MagCVOfflineTests(unittest.TestCase):
    def test_model_shapes_score_and_padding(self):
        model = MagSupervisedAutoencoder()
        # Numerical placeholders ONLY for software shape testing, not science.
        x = torch.zeros((2, 4, 65), dtype=torch.float32)
        mask = torch.ones((2, 65), dtype=torch.bool)
        mask[1, 56:] = False
        relative_time = torch.arange(-32., 33.).repeat(2, 1)
        latent, reconstructed, score = model(x, mask, relative_time)
        self.assertEqual(tuple(latent.shape), (2, 2, 65))
        self.assertEqual(tuple(reconstructed.shape), (2, 4, 65))
        self.assertEqual(tuple(score.shape), (2, 1))
        self.assertTrue(bool(torch.isfinite(score).all()))
        self.assertTrue(bool(((score >= 0) & (score <= 1)).all()))
        self.assertEqual(float(reconstructed[1, :, 56:].abs().sum()), 0.)

    def test_invalid_shapes_are_rejected(self):
        model = MagSupervisedAutoencoder()
        x = torch.zeros(2, 3, 65)
        mask = torch.ones(2, 65, dtype=torch.bool)
        time = torch.arange(65.).repeat(2, 1)
        with self.assertRaises(ValueError):
            model(x, mask, time)

    def test_uniform_selection_keeps_endpoints(self):
        selected = trainer.select_spaced(list(range(101)), 11)
        self.assertEqual(len(selected), 11)
        self.assertEqual(selected[0], 0)
        self.assertEqual(selected[-1], 100)

    def test_cross_fold_supports_must_not_overlap(self):
        t = datetime(2020, 1, 1, tzinfo=timezone.utc)
        train = [{"time": t}]
        safe = [{"time": t + timedelta(hours=4)}]
        unsafe = [{"time": t + timedelta(minutes=30)}]
        trainer.check_disjoint_supports(train, safe)
        with self.assertRaises(ValueError):
            trainer.check_disjoint_supports(train, unsafe)

    def test_event_selection_preserves_fold_assignment(self):
        t = datetime(2020, 1, 1, tzinfo=timezone.utc)
        rows = [
            {"time": t + timedelta(days=i), "fold": i % 4 + 1,
             "source_group": "Group1" if i % 2 else "Group2",
             "mag_available": "1"}
            for i in range(20)
        ]
        train = trainer.choose_events(rows, 1, True, 7, 42)
        val = trainer.choose_events(rows, 1, False, 3, 42)
        self.assertEqual(len(train), 7)
        self.assertEqual(len(val), 3)
        self.assertTrue(all(x["fold"] != 1 for x in train))
        self.assertTrue(all(x["fold"] == 1 for x in val))


if __name__ == "__main__":
    unittest.main()
