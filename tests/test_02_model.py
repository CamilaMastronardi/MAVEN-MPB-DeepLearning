"""Pruebas offline del modelo V7b, sin BigQuery ni datos cientificos."""
from __future__ import annotations
import unittest
import torch
from src.model.v7b import MagSweaRegionAutoencoder
from src.model.ranking import event_pairwise_ranking_loss

class ModelTests(unittest.TestCase):
    def test_forward_shapes(self):
        model = MagSweaRegionAutoencoder().eval()
        batch = 2
        mag, swea = torch.randn(batch, 4, 120), torch.randn(batch, 12, 19)
        mag_mask = torch.ones(batch, 120, dtype=torch.bool)
        swea_mask = torch.ones(batch, 19, dtype=torch.bool)
        mt = torch.linspace(-300, 300, 120).repeat(batch, 1)
        st = torch.linspace(-288, 288, 19).repeat(batch, 1)
        with torch.no_grad():
            out = model(mag, mag_mask, mt, swea, swea_mask, st)
        self.assertEqual(tuple(out["transition_logit"].shape), (batch,))
        self.assertEqual(tuple(out["region_logits"].shape), (batch, 2))
        self.assertEqual(tuple(out["mag_reconstruction"].shape), (batch, 4, 120))
        self.assertEqual(tuple(out["swea_reconstruction"].shape), (batch, 12, 19))

    def test_event_ranking_gradients(self):
        logits = torch.tensor([2., 0., 0., 2.], requires_grad=True)
        labels = torch.tensor([1., 0., 0., 1.])
        loss, accuracy, top1, count = event_pairwise_ranking_loss(
            logits, labels, ["A", "A", "B", "B"], margin=0.25)
        self.assertEqual(count, 2)
        self.assertAlmostEqual(accuracy, 0.5)
        self.assertAlmostEqual(top1, 0.5)
        loss.backward()
        self.assertTrue(torch.isfinite(logits.grad).all())

if __name__ == "__main__":
    unittest.main()
