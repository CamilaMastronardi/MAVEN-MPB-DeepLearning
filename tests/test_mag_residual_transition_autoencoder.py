"""Offline structural/gradient tests for the fifth MAG residual encoder.

Uses small constant arrays ONLY to validate software mechanics. These are
not scientific MAG measurements and are never used in model training.
"""
import unittest

import torch

from src.detection.mag_transition_autoencoder import MagTransitionAutoencoder
from src.detection.mag_residual_transition_autoencoder import (
    MagResidualTransitionAutoencoder, ResidualDilatedBlock,
)


class ResidualMAGTests(unittest.TestCase):
    def setUp(self):
        torch.set_num_threads(1)

    def test_encoder_deeper_wider_and_receptive_field(self):
        old = MagTransitionAutoencoder()
        new = MagResidualTransitionAutoencoder()
        self.assertEqual(new.receptive_field_samples, 515)
        self.assertEqual(len(new.DILATIONS), 7)
        self.assertEqual(new.detector[0].in_features, 4 * 8)
        self.assertEqual(old.detector[0].in_features, 4 * 2)
        self.assertGreater(
            sum(p.numel() for p in new.parameters()),
            sum(p.numel() for p in old.parameters()),
        )
        self.assertEqual(sum(
            isinstance(m, ResidualDilatedBlock)
            for m in new.encoder.modules()
        ), 7)

    def test_forward_padding_scores_and_gradient(self):
        model = MagResidualTransitionAutoencoder()
        time = torch.arange(-64., 65.).repeat(2, 1)
        mask = torch.ones((2, 129), dtype=torch.bool)
        mask[1, 121:] = False
        x = torch.zeros((2, 4, 129), requires_grad=True)
        z, reconstruction, score = model(x, mask, time)
        self.assertEqual(tuple(z.shape), (2, 8, 129))
        self.assertEqual(tuple(reconstruction.shape), (2, 4, 129))
        self.assertEqual(tuple(score.shape), (2, 1))
        self.assertTrue(bool(torch.isfinite(reconstruction).all()))
        self.assertTrue(bool(torch.isfinite(score).all()))
        self.assertTrue(bool(((score >= 0) & (score <= 1)).all()))
        self.assertEqual(float(reconstruction[1, :, 121:].abs().sum()), 0.)
        (score.mean() + reconstruction.square().mean()).backward()
        self.assertTrue(bool(torch.isfinite(x.grad).all()))
        self.assertTrue(any(
            p.grad is not None and bool(torch.isfinite(p.grad).all())
            for p in model.encoder.parameters()
        ))

    def test_score_still_responds_to_before_after_features(self):
        model = MagResidualTransitionAutoencoder()
        z = torch.zeros((1, 8, 7), dtype=torch.float32)
        z[:, 0, :3] = -1.
        z[:, 0, 4:] = 1.
        times = torch.tensor([[-180., -120., -60., 0., 60., 120., 180.]])
        mask = torch.ones((1, 7), dtype=torch.bool)
        features = model.transition_features(z, mask, times)
        self.assertEqual(tuple(features.shape), (1, 32))
        center, before, after, delta = torch.chunk(features, 4, dim=1)
        self.assertLess(float(before[0, 0]), 0.)
        self.assertGreater(float(after[0, 0]), 0.)
        torch.testing.assert_close(delta, after - before)

    def test_residual_conv_time_length_preserved(self):
        block = ResidualDilatedBlock(channels=32, dilation=64)
        x = torch.zeros((2, 32, 71))
        self.assertEqual(tuple(block(x).shape), tuple(x.shape))


if __name__ == "__main__":
    unittest.main()
