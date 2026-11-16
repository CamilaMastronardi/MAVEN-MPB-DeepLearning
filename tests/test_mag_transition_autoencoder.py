"""Offline algebraic tests for the transition detector. No MAVEN data used."""
import unittest

import torch

from src.detection.mag_autoencoder import MagSupervisedAutoencoder
from src.detection.mag_transition_autoencoder import MagTransitionAutoencoder


class TransitionDetectorTests(unittest.TestCase):
    def setUp(self):
        torch.set_num_threads(1)

    def test_detector_has_explicit_four_part_latent_input(self):
        baseline = MagSupervisedAutoencoder(hidden=16, latent_channels=2)
        new = MagTransitionAutoencoder(hidden=16, latent_channels=2)
        self.assertEqual(baseline.detector[0].in_features, 4)
        self.assertEqual(new.detector[0].in_features, 8)
        self.assertEqual(len(baseline.encoder), len(new.encoder))
        self.assertEqual(len(baseline.decoder), len(new.decoder))

    def test_asymmetry_direction_and_delta_are_explicit(self):
        model = MagTransitionAutoencoder(
            latent_channels=2, side_offset_s=120., center_exclusion_s=15.
        )
        times = torch.tensor([[-180., -120., -60., 0., 60., 120., 180.]])
        mask = torch.ones((1, 7), dtype=torch.bool)
        z = torch.zeros((1, 2, 7))
        # Algebraic representation with distinguishable left and right regions.
        z[0, 0, :] = torch.tensor([-3., -2., -1., 0., 1., 2., 3.])
        parts = model.transition_features(z, mask, times)
        center, before, after, delta = torch.chunk(parts, 4, dim=1)
        self.assertAlmostEqual(center[0, 0].item(), 0., places=7)
        self.assertLess(before[0, 0].item(), 0.)
        self.assertGreater(after[0, 0].item(), 0.)
        self.assertAlmostEqual(
            delta[0, 0].item(),
            (after[0, 0] - before[0, 0]).item(),
            places=7,
        )
        self.assertEqual(tuple(parts.shape), (1, 8))

    def test_padded_latent_samples_never_enter_side_pools(self):
        model = MagTransitionAutoencoder(
            latent_channels=2, side_offset_s=120., center_exclusion_s=15.
        )
        times = torch.tensor([[-120., -60., 0., 60., 120., 240., 270.]])
        mask = torch.tensor([[True, True, True, True, True, False, False]])
        base = torch.arange(14., dtype=torch.float32).reshape(1, 2, 7)
        altered = base.clone()
        altered[:, :, 5:] = 1e7
        first = model.transition_features(base, mask, times)
        second = model.transition_features(altered, mask, times)
        torch.testing.assert_close(first, second)

    def test_forward_shapes_finite_scores_and_gradients(self):
        model = MagTransitionAutoencoder()
        x = torch.zeros((2, 4, 65), dtype=torch.float32, requires_grad=True)
        times = torch.arange(-32., 33.).repeat(2, 1)
        mask = torch.ones((2, 65), dtype=torch.bool)
        mask[1, 56:] = False
        z, reconstructed, score = model(x, mask, times)
        self.assertEqual(tuple(z.shape), (2, 2, 65))
        self.assertEqual(tuple(reconstructed.shape), (2, 4, 65))
        self.assertEqual(tuple(score.shape), (2, 1))
        self.assertTrue(bool(torch.isfinite(score).all()))
        self.assertTrue(bool(((score >= 0) & (score <= 1)).all()))
        self.assertEqual(float(reconstructed[1, :, 56:].abs().sum()), 0.)
        score.mean().backward()
        self.assertTrue(bool(torch.isfinite(x.grad).all()))
        self.assertTrue(any(p.grad is not None for p in model.encoder.parameters()))

    def test_missing_side_falls_back_to_center_without_nans(self):
        model = MagTransitionAutoencoder()
        mask = torch.ones((1, 3), dtype=torch.bool)
        time = torch.tensor([[0., 1., 2.]])  # No valid before region.
        latent = torch.tensor([[[2., 3., 4.], [7., 8., 9.]]])
        features = model.transition_features(latent, mask, time)
        center, before, after, delta = torch.chunk(features, 4, dim=1)
        torch.testing.assert_close(center, before)
        torch.testing.assert_close(center, after)
        self.assertTrue(bool(torch.isfinite(features).all()))
        self.assertTrue(bool((delta == 0).all()))

    def test_invalid_side_specification_rejected(self):
        with self.assertRaises(ValueError):
            MagTransitionAutoencoder(side_offset_s=10., center_exclusion_s=20.)


if __name__ == "__main__":
    unittest.main()
