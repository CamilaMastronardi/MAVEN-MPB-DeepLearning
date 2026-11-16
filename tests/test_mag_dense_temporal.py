"""Offline tests for sixth MAG time-resolved MPB detection and scoring.

Small deterministic tensors test software algebra; never used as synthetic
science training data. No BigQuery request, model CV or data downloading.
"""
import unittest
from types import SimpleNamespace

import numpy as np
import torch

from src.detection.mag_dense_temporal_autoencoder import MagDenseTemporalAutoencoder
from src.detection.mag_residual_transition_autoencoder import (
    MagResidualTransitionAutoencoder,
)
from src.detection.mag_temporal_scoring import (
    DenseNativeScoreFusion, dense_gaussian_targets, dense_weighted_mse,
)
from importlib import import_module

trainer = import_module("scripts.5_training.5a_train_mag_cv")


class DenseTemporalTests(unittest.TestCase):
    def setUp(self):
        torch.set_num_threads(1)

    def test_v6_reuses_exact_v5_encoder_and_decoder_topology(self):
        fifth = MagResidualTransitionAutoencoder()
        sixth = MagDenseTemporalAutoencoder()
        self.assertEqual(
            [(k, tuple(v.shape)) for k, v in fifth.encoder.state_dict().items()],
            [(k, tuple(v.shape)) for k, v in sixth.encoder.state_dict().items()],
        )
        self.assertEqual(
            [(k, tuple(v.shape)) for k, v in fifth.decoder.state_dict().items()],
            [(k, tuple(v.shape)) for k, v in sixth.decoder.state_dict().items()],
        )
        self.assertEqual(sixth.receptive_field_samples, 515)

    def test_outputs_one_score_per_real_epoch_and_masked_gradients(self):
        model = MagDenseTemporalAutoencoder()
        x = torch.zeros((2, 4, 85), dtype=torch.float32, requires_grad=True)
        times = torch.arange(-42., 43.).repeat(2, 1)
        mask = torch.ones((2, 85), dtype=torch.bool)
        mask[1, 76:] = False
        times[1, 76:] = 0.
        z, reconstruction, scores = model(x, mask, times)
        self.assertEqual(tuple(z.shape), (2, 8, 85))
        self.assertEqual(tuple(reconstruction.shape), (2, 4, 85))
        self.assertEqual(tuple(scores.shape), (2, 85))
        self.assertTrue(bool(torch.isfinite(scores).all()))
        self.assertEqual(float(scores[1, 76:].abs().sum()), 0.)
        self.assertEqual(float(reconstruction[1, :, 76:].abs().sum()), 0.)
        (scores.mean() + reconstruction.square().mean()).backward()
        self.assertTrue(bool(torch.isfinite(x.grad).all()))

    def test_side_features_use_real_irregular_timestamps(self):
        model = MagDenseTemporalAutoencoder()
        times = torch.tensor([[-240., -120., 0., 120., 240., 0.]])
        mask = torch.tensor([[True, True, True, True, True, False]])
        z = torch.zeros((1, 8, 6), dtype=torch.float32)
        z[0, 0, :] = torch.tensor([1., 2., 5., 8., 10., 1e6])
        features = model.temporal_transition_features(z, mask, times)
        center, before, after, delta = torch.chunk(features, 4, dim=1)
        self.assertAlmostEqual(float(center[0, 0, 2]), 5., places=6)
        self.assertAlmostEqual(float(before[0, 0, 2]), 2., places=6)
        self.assertAlmostEqual(float(after[0, 0, 2]), 8., places=6)
        self.assertAlmostEqual(float(delta[0, 0, 2]), 6., places=6)
        self.assertEqual(float(features[0, :, -1].abs().sum()), 0.)

    def test_dense_target_uses_absolute_time_to_annotation(self):
        times = torch.tensor([[-60., 0., 60., 0.]])
        mask = torch.tensor([[True, True, True, False]])
        batch = SimpleNamespace(
            rel_time_s=times, time_mask=mask,
            label_offset_s=torch.tensor([0.]),
        )
        target = dense_gaussian_targets(batch, 60.)
        self.assertAlmostEqual(float(target[0, 1]), 1., places=6)
        self.assertAlmostEqual(
            float(target[0, 0]), float(np.exp(-0.5)), places=6
        )
        self.assertEqual(float(target[0, 3]), 0.)
        self.assertAlmostEqual(
            float(dense_weighted_mse(
                target, batch, sigma_seconds=60., near_label_weight=10.
            )), 0., places=8
        )

    def test_dense_fusion_averages_real_epoch_scores_not_window_centers(self):
        windows = [
            SimpleNamespace(
                center_s=-1., relative_times_s=np.array([-1., 0., 1.])
            ),
            SimpleNamespace(
                center_s=1., relative_times_s=np.array([-1., 0., 1.])
            ),
        ]
        fused = DenseNativeScoreFusion(windows, 0., radius_s=5.)
        fused.add_window(windows[0], [0.1, 0.2, 0.5])
        fused.add_window(windows[1], [0.9, 0.2, 0.1])
        result = fused.finish()
        self.assertEqual(len(result["times_s"]), 5)
        self.assertAlmostEqual(result["peak_offset_s"], 0., places=6)
        self.assertAlmostEqual(result["peak_score"], 0.7, places=6)
        self.assertEqual(int(result["coverage"][2]), 2)

    def test_forty_centers_per_epoch_cover_six_hundred_in_fifteen(self):
        all_indices = [
            trainer.epoch_pool_indices(
                n=600, k=0, seed=42, epoch=epoch, count=40
            )
            for epoch in range(1, 16)
        ]
        flat = np.concatenate(all_indices)
        self.assertEqual(len(flat), 600)
        self.assertEqual(len(set(flat.tolist())), 600)

    def test_v6_model_builder_is_not_old_architecture(self):
        fourth = trainer.build_mag_model(SimpleNamespace(
            architecture="transition",
            transition_side_offset_s=120.,
            transition_center_exclusion_s=15.
        ))
        sixth = trainer.build_mag_model(SimpleNamespace(
            architecture="dense-temporal",
            transition_side_offset_s=120.,
            transition_center_exclusion_s=15.
        ))
        self.assertEqual(fourth.detector[0].in_features, 8)
        self.assertIsInstance(sixth, MagDenseTemporalAutoencoder)
        self.assertEqual(sixth.detector[0].in_channels, 32)


if __name__ == "__main__":
    unittest.main()
