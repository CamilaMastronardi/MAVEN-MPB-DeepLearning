"""Offline seventh architecture tests: no BigQuery or scientific simulations."""
import tempfile
import unittest
from datetime import datetime, timedelta, timezone
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import torch

from src.data.mpb_weak_regions import (
    SHEATH, INDUCED, classify_candidate, candidate_training_grid
)
from src.data.swea_event_cache import (
    ENERGY_EV, SweaEventCache, energy_features
)
from src.data.mpb_dual_windows import (
    extract_clip, collate_dual, fit_unique_epochs
)
from src.detection.mag_swea_region_model import (
    MagSweaRegionAutoencoder, masked_reconstruction_loss
)


class SeventhArchitectureOfflineTests(unittest.TestCase):
    def setUp(self):
        torch.set_num_threads(1)

    def example_event(self, inbound=True):
        epoch = datetime(2020, 2, 1, tzinfo=timezone.utc)
        return {
            "time": epoch,
            "bs_time": epoch+timedelta(
                seconds=(-1800 if inbound else 1800)
            ),
            "source_group": "Group1",
        }

    def test_inbound_outbound_region_orientation(self):
        for inbound in (True, False):
            event = self.example_event(inbound)
            center = event["time"].timestamp()
            bs = event["bs_time"].timestamp()
            pos = classify_candidate(
                event, center, all_mpb_s=[center], all_bs_s=[bs]
            )
            self.assertEqual(pos.transition, 1)
            self.assertEqual(
                (pos.left_region, pos.right_region),
                (SHEATH, INDUCED) if inbound else (INDUCED, SHEATH)
            )
            neg = classify_candidate(
                event, center + (-900 if inbound else 900),
                all_mpb_s=[center], all_bs_s=[bs]
            )
            self.assertEqual(neg.transition, 0)
            self.assertEqual((neg.left_region, neg.right_region),
                             (SHEATH, SHEATH))

    def test_bs_or_neighbor_annotation_never_becomes_negative(self):
        e = self.example_event(True)
        t = e["time"].timestamp()
        bs = e["bs_time"].timestamp()
        forbidden = classify_candidate(
            e, bs, all_mpb_s=[t], all_bs_s=[bs]
        )
        self.assertIsNone(forbidden.transition)
        other = classify_candidate(
            e, t + 900.,
            all_mpb_s=[t, t+1100.], all_bs_s=[bs]
        )
        self.assertIsNone(other.transition)
        distant = self.example_event(True)
        distant["bs_time"] = distant["time"]-timedelta(hours=8)
        label = classify_candidate(
            distant, t, all_mpb_s=[t],
            all_bs_s=[distant["bs_time"].timestamp()]
        )
        self.assertIsNone(label.transition)

    def test_joint_candidate_grid_has_positive_and_negative_examples(self):
        e = self.example_event(True)
        t = e["time"].timestamp()
        grid = candidate_training_grid(
            e, all_mpb_s=np.array([t]),
            all_bs_s=np.array([e["bs_time"].timestamp()])
        )
        self.assertGreaterEqual(sum(c.transition == 1 for c in grid), 1)
        self.assertGreaterEqual(sum(c.transition == 0 for c in grid), 1)

    def test_real_timestamp_clips_keep_independent_cadences(self):
        # Tiny deterministic coordinate fixture only, not MAVEN science data.
        center = 10000.
        mag_t = np.arange(center-350, center+351, 1., dtype=np.float64)
        swea_t = np.arange(center-350, center+351, 32., dtype=np.float64)
        mag = (mag_t, np.zeros((len(mag_t), 4), dtype=np.float64))
        swea = (swea_t, np.ones((len(swea_t), 12), dtype=np.float64))
        clip = extract_clip(mag, swea, center)
        self.assertIsNotNone(clip)
        self.assertGreater(len(clip.mag), len(clip.swea))
        norm = {
            "MAG": {"mean": np.zeros(4), "scale": np.ones(4)},
            "SWEA": {"mean": np.zeros(12), "scale": np.ones(12)},
        }
        batch = collate_dual([(clip, 1, (0, 1))], norm)
        self.assertEqual(tuple(batch["mag"].shape[:2]), (1, 4))
        self.assertEqual(tuple(batch["swea"].shape[:2]), (1, 12))
        self.assertTrue(
            bool(batch["mag_mask"].all() and batch["swea_mask"].all())
        )

    def test_dual_model_shapes_and_joint_gradients(self):
        model = MagSweaRegionAutoencoder()
        mag = torch.randn((2, 4, 130), requires_grad=True)
        swea = torch.randn((2, 12, 19), requires_grad=True)
        mt = torch.linspace(-300, 300, 130).repeat(2, 1)
        st = torch.linspace(-300, 300, 19).repeat(2, 1)
        mm = torch.ones((2, 130), dtype=torch.bool)
        sm = torch.ones((2, 19), dtype=torch.bool)
        output = model(mag, mm, mt, swea, sm, st)
        self.assertEqual(tuple(output["transition_logit"].shape), (2,))
        self.assertEqual(tuple(output["region_logits"].shape), (2, 2))
        self.assertEqual(
            tuple(output["mag_reconstruction"].shape), tuple(mag.shape)
        )
        self.assertEqual(
            tuple(output["swea_reconstruction"].shape), tuple(swea.shape)
        )
        loss = (
            output["transition_logit"].square().mean()
            + output["region_logits"].square().mean()
            + masked_reconstruction_loss(
                output["mag_reconstruction"], mag, mm
            )
        )
        loss.backward()
        self.assertTrue(bool(torch.isfinite(mag.grad).all()))
        self.assertTrue(bool(torch.isfinite(swea.grad).all()))

    def test_swea_spectral_grid_uses_real_energy_axis(self):
        # Artificial tiny arrays test energy axis bookkeeping only.
        t = datetime(2020, 2, 1, tzinfo=timezone.utc)
        energy = np.geomspace(3., 1000., 64)
        source = [{
            "energy_index": k,
            "mean_flux": float(k+1)
        } for k, e in enumerate(energy) if 25 <= e <= 330]
        record = SimpleNamespace(
            datetime_utc=t, n_elev=2, n_azim=2,
            n_energy=len(energy), original_flux_count=4*len(energy),
            energy_ev=list(energy), energy_bins=source
        )
        times, values = energy_features([record])
        self.assertEqual(times.shape, (1,))
        self.assertEqual(values.shape, (1, len(ENERGY_EV)))
        self.assertTrue(np.isfinite(values).all())

    def test_swea_cache_roundtrip_and_checksum(self):
        t = datetime(2020, 2, 1, tzinfo=timezone.utc)
        start = t-timedelta(minutes=35)
        end = t+timedelta(minutes=35)
        with tempfile.TemporaryDirectory() as folder:
            cache = SweaEventCache(
                Path(folder), project="test_project",
                table="test_dataset", sql="SELECT TEST"
            )
            times = np.array([t.timestamp()-32., t.timestamp()])
            values = np.arange(24., dtype=np.float64).reshape(2, 12)
            path = cache.save(t, start, end, times, values)
            actual = cache.load(t, start, end, max_rows=3)
            np.testing.assert_allclose(actual[1], values)
            with self.assertRaisesRegex(ValueError, "row cap"):
                cache.load(t, start, end, max_rows=1)
            path.write_bytes(b"corrupt")
            with self.assertRaises(ValueError):
                cache.load(t, start, end)

    def test_train_only_normalizer_deduplicates_overlapping_blocks(self):
        event1 = datetime(2020, 1, 1, tzinfo=timezone.utc)
        event2 = datetime(2020, 1, 2, tzinfo=timezone.utc)
        source = {
            event1: {"MAG": (
                np.array([1., 2., 3.]),
                np.array([[1., 1., 1., 1.],
                          [2., 2., 2., 2.],
                          [3., 3., 3., 3.]])
            )},
            event2: {"MAG": (
                np.array([2., 3., 4.]),
                np.array([[2., 2., 2., 2.],
                          [3., 3., 3., 3.],
                          [4., 4., 4., 4.]])
            )},
        }
        norm = fit_unique_epochs(
            source, [event1, event2], instrument="MAG"
        )
        self.assertEqual(norm["n_epochs"], 4)
        np.testing.assert_allclose(norm["mean"], [2.5]*4)


if __name__ == "__main__":
    unittest.main()
