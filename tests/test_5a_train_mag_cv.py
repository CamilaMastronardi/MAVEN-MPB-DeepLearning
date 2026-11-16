"""Offline smoke tests for MAG CV training wiring; no BigQuery calls or training."""
import importlib
import unittest
from types import SimpleNamespace

import numpy as np
from datetime import datetime, timedelta, timezone

import torch

MagSupervisedAutoencoder = importlib.import_module(
    "src.detection.mag_autoencoder"
).MagSupervisedAutoencoder
trainer = importlib.import_module("scripts.5_training.5a_train_mag_cv")
batching = importlib.import_module("scripts.4_preprocessing.4a_native_batches")


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


    def test_inner_early_stopping_split_uses_distinct_months(self):
        rows = [
            {"month_block": f"2020-{month:02d}",
             "time": datetime(2020, month, 15, tzinfo=timezone.utc),
             "source_group": "Group1", "mag_available": "1",
             "fold": month % 4 + 1}
            for month in range(1, 13)
        ]
        fit, monitor, excluded = trainer.split_inner_monitor_months(
            rows, fold=1, seed=42
        )
        self.assertTrue(fit)
        self.assertTrue(monitor)
        self.assertEqual(len(fit) + len(monitor) + excluded, len(rows))
        self.assertFalse(
            {x["month_block"] for x in fit} &
            {x["month_block"] for x in monitor}
        )
        trainer.check_disjoint_supports(fit, monitor)

    def test_monitor_centers_include_near_and_far(self):
        from types import SimpleNamespace
        event = {"time": datetime(2020, 1, 15, tzinfo=timezone.utc)}
        offsets = [-1200., -600., -120., -60., 0., 60., 120., 600., 1200.]
        windows = [
            SimpleNamespace(offset_from_nearest_mpb_s=offset,
                            center_s=offset)
            for offset in offsets
        ]
        result = trainer.monitoring_windows([(event, windows)], 5)
        self.assertEqual(len(result), 5)
        self.assertTrue(any(abs(x.offset_from_nearest_mpb_s) <= 180 for x in result))
        self.assertTrue(any(abs(x.offset_from_nearest_mpb_s) > 180 for x in result))



    def test_600_targets_covered_after_25_epochs(self):
        all_seen = []
        for epoch in range(1, 26):
            ids = trainer.epoch_pool_indices(
                n=600, k=0, seed=42, epoch=epoch, count=24
            )
            self.assertEqual(len(ids), 24)
            self.assertEqual(len(set(ids.tolist())), 24)
            all_seen.extend(ids.tolist())
        self.assertEqual(len(all_seen), 600)
        self.assertEqual(set(all_seen), set(range(600)))

    def test_pool_targets_remain_gaussian_per_native_center(self):
        import math
        from types import SimpleNamespace
        windows = [
            SimpleNamespace(
                center_s=float(t),
                detection_score=math.exp(-0.5 * (t / 60.) ** 2),
                offset_from_nearest_mpb_s=float(t)
            )
            for t in range(-300, 300)
        ]
        pool = trainer.compact_event_pool(windows)
        self.assertEqual(pool.shape, (600, 3))
        self.assertAlmostEqual(pool[300, 1], 1., places=12)
        self.assertLess(pool[0, 1], 0.001)




    def test_magnitude_loss_is_zero_for_physical_consistent_reconstruction(self):
        """Only algebraic tensors: never synthetic training/validation MAVEN."""
        normalizer = SimpleNamespace(
            instrument="MAG", is_frozen=True,
            mean=np.array([1., -2., 1., 8.]),
            scale=np.array([2., 4., 5., 6.]),
        )
        # B=(3,4,0) nT, measured |B|=5 nT. Distinct feature scalers.
        observed = torch.tensor([1., 1.5, -0.2, -0.5]).reshape(1, 4, 1)
        observed = observed.repeat(1, 1, 3)
        valid = torch.tensor([[True, True, False]])
        batch = SimpleNamespace(x=observed, time_mask=valid)
        predicted = observed.clone()
        predicted[0, :, 2] = 1000.  # Padding must never affect physical loss.
        loss = batching.masked_mag_modulus_mse(
            predicted, batch, normalizer
        )
        self.assertAlmostEqual(loss.item(), 0., places=10)

    def test_magnitude_loss_penalizes_channel_and_vector_disagreement(self):
        normalizer = SimpleNamespace(
            instrument="MAG", is_frozen=True,
            mean=np.array([1., -2., 1., 8.]),
            scale=np.array([2., 4., 5., 6.]),
        )
        observed = torch.tensor([1., 1.5, -0.2, -0.5]).reshape(1, 4, 1)
        batch = SimpleNamespace(
            x=observed,
            time_mask=torch.ones((1, 1), dtype=torch.bool),
        )
        # Decode |B| channel 6 nT too high, while decoded xyz remain correct.
        decoded = observed.clone()
        decoded[0, 3, 0] += 1.
        channel_loss = batching.masked_mag_modulus_mse(
            decoded, batch, normalizer
        )
        self.assertAlmostEqual(channel_loss.item(), 0.5, places=6)

        # Decode incorrect Bx while keeping decoded |B| channel correct.
        predicted = observed.clone()
        predicted[0, 0, 0] += 1.
        predicted.requires_grad_(True)
        vector_loss = batching.masked_mag_modulus_mse(
            predicted, batch, normalizer
        )
        self.assertGreater(vector_loss.item(), 0.)
        vector_loss.backward()
        self.assertTrue(bool(torch.isfinite(predicted.grad).all()))
        self.assertNotEqual(float(predicted.grad[0, 0, 0]), 0.)

    def test_magnitude_loss_rejects_unfrozen_normalizer(self):
        normalizer = SimpleNamespace(
            instrument="MAG", is_frozen=False,
            mean=np.zeros(4), scale=np.ones(4),
        )
        x = torch.zeros((1, 4, 3))
        batch = SimpleNamespace(x=x, time_mask=torch.ones((1, 3), dtype=torch.bool))
        with self.assertRaises(ValueError):
            batching.masked_mag_modulus_mse(x, batch, normalizer)



if __name__ == "__main__":
    unittest.main()
