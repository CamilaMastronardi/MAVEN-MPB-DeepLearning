"""Offline checks for the independent V7b temporal localization objective."""
from __future__ import annotations

import importlib.util
import unittest
from datetime import datetime, timezone, timedelta
from pathlib import Path
from types import SimpleNamespace

import torch

ROOT = Path(__file__).resolve().parents[1]
SCRIPT = ROOT / "scripts/03_experiments/05_train_v7b_temporal_full.py"


def load_script():
    spec = importlib.util.spec_from_file_location("v7b_temporal_full", SCRIPT)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


class TemporalLossTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.module = load_script()

    def _examples(self, positions, event=None):
        if event is None:
            event = datetime(2016, 3, 16, tzinfo=timezone.utc)
        return [
            (event, SimpleNamespace(center_s=event.timestamp()+delta),
             None, 1 if delta == 0 else 0, (0, 1))
            for delta in positions
        ]

    def test_loss_prefers_a_high_score_at_real_mpb(self):
        examples = self._examples([-600., 0., 600.])
        center = torch.tensor([0., 5., 0.], requires_grad=True)
        outside = torch.tensor([5., 0., 0.], requires_grad=True)
        a = self.module.loss_temporal(
            center, examples, temperature=1., scale_s=600.
        )
        b = self.module.loss_temporal(
            outside, examples, temperature=1., scale_s=600.
        )
        self.assertLess(float(a), float(b))
        self.assertGreaterEqual(float(a), 0.)
        a.backward()
        self.assertTrue(torch.isfinite(center.grad).all())
        self.assertGreater(float(center.grad.abs().sum()), 0.)

    def test_events_are_equally_weighted(self):
        event1 = datetime(2016, 3, 16, tzinfo=timezone.utc)
        event2 = event1 + timedelta(days=1)
        first = self._examples([0., 600.], event1)
        second = self._examples([0., 1200., -1200.], event2)
        loss1 = self.module.loss_temporal(
            torch.tensor([0., 0.]), first, temperature=1., scale_s=600.
        )
        loss2 = self.module.loss_temporal(
            torch.zeros(3), second, temperature=1., scale_s=600.
        )
        pooled = self.module.loss_temporal(
            torch.zeros(5), first+second,
            temperature=1., scale_s=600.
        )
        self.assertAlmostEqual(
            float(pooled), float((loss1+loss2)/2), places=6
        )

    def test_softmax_loss_is_unchanged_by_common_logit_shift(self):
        examples = self._examples([-120., 0., 120.])
        x = torch.tensor([-1., 2., 0.])
        a = self.module.loss_temporal(
            x, examples, temperature=1.3, scale_s=600.
        )
        b = self.module.loss_temporal(
            x+3.7, examples, temperature=1.3, scale_s=600.
        )
        self.assertAlmostEqual(float(a), float(b), places=6)


if __name__ == '__main__':
    unittest.main()
