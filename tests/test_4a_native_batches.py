#!/usr/bin/env python3
"""Offline unit tests: fold-isolated normalization and variable-length batches."""
from __future__ import annotations

import sys
import unittest
from importlib import import_module
from pathlib import Path

import numpy as np
import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
batching = import_module("scripts.4_preprocessing.4a_native_batches")
FoldChannelNormalizer = batching.FoldChannelNormalizer
collate_native_windows = batching.collate_native_windows
feature_transform = batching.feature_transform
masked_reconstruction_mse = batching.masked_reconstruction_mse
weighted_proximity_mse = batching.weighted_proximity_mse
Window = import_module("scripts.3_windowing.3a_windowing").Window


def example_window(values: np.ndarray, *, score: float = 0.5) -> Window:
    return Window(
        center_s=1_500_000_000.0,
        data=values,
        relative_times_s=np.arange(len(values), dtype=float),
        detection_score=score,
        offset_from_nearest_mpb_s=22.0,
    )


class PreprocessingTests(unittest.TestCase):
    def test_mag_normalization_uses_train_only_and_matches_full_batch(self) -> None:
        train_a = np.array([
            [-2., 1., 2., 3.], [-1., 2., 4., 5.], [1., 3., 6., 7.]
        ])
        train_b = np.array([
            [3., 4., 8., 9.], [6., 5., 10., 11.]
        ])
        normalizer = FoldChannelNormalizer("MAG", held_out_fold=1)
        normalizer.update(train_a, source_fold=2)
        with self.assertRaises(ValueError):
            normalizer.update(np.full((3, 4), 1000.0), source_fold=1)
        normalizer.update(train_b, source_fold=3)
        normalizer.freeze()
        all_train = np.vstack([train_a, train_b])
        normalized = normalizer.transform(all_train)
        np.testing.assert_allclose(normalized.mean(axis=0), np.zeros(4), atol=1e-6)
        np.testing.assert_allclose(normalized.std(axis=0), np.ones(4), atol=1e-6)
        validation = np.full((2, 4), 1000.0)
        val_transformed = normalizer.transform(validation)
        self.assertTrue((val_transformed > 10).all())
        with self.assertRaises(RuntimeError):
            normalizer.update(train_a, source_fold=2)

    def test_swea_log_and_per_energy_normalization(self) -> None:
        raw = np.array([
            [0., 9., 99.],
            [1., 99., 999.],
            [9., 999., 9999.],
        ])
        np.testing.assert_allclose(
            feature_transform(raw, "SWEA"), np.log10(1+raw)
        )
        n = FoldChannelNormalizer("SWEA", held_out_fold=4)
        n.update(raw, source_fold=1)
        n.freeze()
        self.assertEqual(n.transform(raw).shape, raw.shape)
        np.testing.assert_allclose(
            n.transform(raw).mean(axis=0), np.zeros(3), atol=1e-6
        )

    def test_collate_mag_padding_mask_and_loss(self) -> None:
        n = FoldChannelNormalizer("MAG", held_out_fold=1)
        n.update(np.arange(40., dtype=float).reshape(10, 4), source_fold=2)
        n.freeze()
        a = example_window(np.ones((5, 4)), score=0.9)
        b = example_window(np.ones((7, 4))*2, score=0.1)
        batch = collate_native_windows([a, b], normalizer=n)
        self.assertEqual(batch.x.shape, (2, 4, 7))
        self.assertEqual(batch.time_mask.sum().item(), 12)
        self.assertFalse(batch.time_mask[0, -1].item())
        self.assertEqual(batch.y.shape, (2, 1))
        reconstructed = batch.x.clone()
        reconstructed[0, :, 5:] = 123456.0  # wrong *only* on padding
        self.assertAlmostEqual(
            masked_reconstruction_mse(reconstructed, batch).item(), 0.0
        )
        self.assertGreater(
            weighted_proximity_mse(torch.zeros_like(batch.y), batch).item(), 0.0
        )

    def test_collate_swea_variable_cadence(self) -> None:
        n = FoldChannelNormalizer("SWEA", held_out_fold=2)
        n.update(np.ones((10, 11)), source_fold=1)
        n.freeze()
        a = example_window(np.full((18, 11), 2.0))
        b = example_window(np.full((19, 11), 3.0))
        batch = collate_native_windows([a, b], normalizer=n)
        self.assertEqual(batch.x.shape, (2, 1, 11, 19))
        self.assertEqual(batch.time_mask.sum().item(), 37)
        self.assertEqual(batch.rel_time_s.shape, (2, 19))
        self.assertAlmostEqual(
            masked_reconstruction_mse(batch.x.clone(), batch).item(), 0.0
        )

    def test_invalid_input_is_not_silently_imputed(self) -> None:
        n = FoldChannelNormalizer("SWEA", held_out_fold=3)
        with self.assertRaises(ValueError):
            n.update(np.array([[-1., 3.]]), source_fold=2)
        with self.assertRaises(ValueError):
            n.update(np.array([[np.nan, 3.]]), source_fold=2)


if __name__ == "__main__":
    unittest.main()
