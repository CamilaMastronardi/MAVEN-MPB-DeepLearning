"""Offline V14 tests: frozen V7b, 8-layer 5D MAG/SWEA, native time."""
from __future__ import annotations

import sys
import unittest
from unittest.mock import patch
from importlib import import_module
from types import SimpleNamespace

import torch

from src.detection.mag_swea_region_model import MagSweaRegionAutoencoder
from src.detection.mpb_v14_dual_bottleneck_refiner import (
    V14DualBottleneckRefiner,center_pad_native,masked_mse
)

v14=import_module("scripts.5_training.5n_train_v14_dual_bottleneck")


class V14Tests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        torch.set_num_threads(2)

    def setUp(self):
        torch.manual_seed(14)
        self.teacher=MagSweaRegionAutoencoder().eval()
        self.model=V14DualBottleneckRefiner(
            self.teacher.state_dict()
        )
        self.mag=torch.randn(2,4,600)
        self.mag_mask=torch.ones(2,600,dtype=torch.bool)
        self.mag_time=torch.arange(-300.,300.)[None].repeat(2,1)
        self.swea=torch.randn(2,12,19)
        self.swea_mask=torch.ones(2,19,dtype=torch.bool)
        self.swea_time=torch.arange(-288.,289.,32.)[None].repeat(2,1)

    def forward(self):
        return self.model(
            self.mag,self.mag_mask,self.mag_time,
            self.swea,self.swea_mask,self.swea_time
        )

    def test_cli_supplies_every_v7b_cache_loader_setting(self):
        # Regress the missing refresh_swea_cache Namespace attribute
        # that previously crashed immediately after a successful preflight.
        with patch.object(sys,"argv",["v14"]):
            a=v14.parse_args()
        required=(
            "mag_cache_dir","swea_cache_dir",
            "refresh_swea_cache","max_mag_rows","max_swea_rows",
            "max_query_gib","max_total_gib","execute","output_dir"
        )
        for name in required:
            self.assertTrue(hasattr(a,name),name)
        self.assertIs(a.refresh_swea_cache,False)

    def test_eight_layers_and_scalar_latents(self):
        self.assertEqual(len(self.model.mag_ae.encoder_layers),8)
        self.assertEqual(len(self.model.swea_ae.encoder_layers),8)
        out=self.forward()
        self.assertEqual(tuple(out["mag_latent"].shape),(2,5))
        self.assertEqual(tuple(out["swea_latent"].shape),(2,5))
        self.assertEqual(self.model.fusion[0].in_features,13)
        self.assertEqual(tuple(self.model.mag_ae.decode(
            out["mag_latent"]).shape),(2,4,640))
        self.assertEqual(tuple(self.model.swea_ae.decode(
            out["swea_latent"]).shape),(2,12,256))
        self.assertEqual(tuple(out["time_logits"].shape),(2,600))
        self.assertEqual(tuple(out["expected_offset_s"].shape),(2,))
        self.assertEqual(tuple(out["peak_offset_s"].shape),(2,))
        torch.testing.assert_close(
            out["time_prob"].sum(-1),torch.ones(2)
        )
        self.assertLessEqual(float(out["peak_offset_s"].abs().max()),
                             250.)

    def test_teacher_is_exact_frozen_v7b(self):
        self.model.train()
        with torch.no_grad():
            teacher_output=self.teacher(
                self.mag,self.mag_mask,self.mag_time,
                self.swea,self.swea_mask,self.swea_time
            )
            coarse,region=self.model.teacher_logits(
                self.mag,self.mag_mask,self.mag_time,
                self.swea,self.swea_mask,self.swea_time
            )
        torch.testing.assert_close(
            coarse,teacher_output["transition_logit"]
        )
        torch.testing.assert_close(
            region,teacher_output["region_logits"]
        )
        self.assertFalse(self.model.teacher.training)
        self.assertFalse(any(p.requires_grad
                            for p in self.model.teacher.parameters()))

    def test_both_5d_encoders_receive_gradient_not_teacher(self):
        out=self.forward()
        batch={
            "mag":self.mag,"mag_mask":self.mag_mask,
            "mag_time":self.mag_time,
            "swea":self.swea,"swea_mask":self.swea_mask,
            "swea_time":self.swea_time,
            "transition":torch.tensor([1.,0.]),
            "target_offset":torch.tensor([70.,0.])
        }
        args=SimpleNamespace(
            gaussian_sigma_s=24.,localization_weight=3.,
            offset_weight=1.5,reconstruction_weight=.2
        )
        loss,metrics=v14.loss_step(self.model,batch,args)
        self.assertTrue(torch.isfinite(loss))
        self.assertEqual(metrics["n_positive"],1)
        self.assertGreater(metrics["reconstruction_mag"],0.)
        self.assertGreater(metrics["reconstruction_swea"],0.)
        loss.backward()
        self.assertTrue(any(p.grad is not None
                            for p in self.model.mag_ae.parameters()))
        self.assertTrue(any(p.grad is not None
                            for p in self.model.swea_ae.parameters()))
        self.assertTrue(any(p.grad is not None
                            for p in self.model.temporal_head.parameters()))
        self.assertFalse(any(p.grad is not None
                             for p in self.model.teacher.parameters()))

    def test_center_padding_native_lengths_preserved(self):
        x=torch.arange(2*4*600,dtype=torch.float32).reshape(2,4,600)
        mask=torch.ones(2,600,dtype=torch.bool)
        mask[1,550:]=False
        padded,valid=center_pad_native(x,mask,640,540)
        self.assertEqual(tuple(padded.shape),(2,4,640))
        self.assertEqual(valid.sum(-1).tolist(),[600,550])
        self.assertEqual(float(padded[1,:,45:595].sum()),
                         float(x[1,:,:550].sum()))
        self.assertEqual(float(padded[0,:,20:620].sum()),
                         float(x[0].sum()))

    def test_masked_reconstruction_ignores_padding(self):
        target=torch.randn(2,4,640)
        mask=torch.zeros(2,640,dtype=torch.bool)
        mask[:,20:620]=True
        same=target.clone()
        same[:,:,0]=999.
        self.assertAlmostEqual(float(masked_mse(same,target,mask)),0.)

    def test_bad_encoder_length_rejected(self):
        with self.assertRaises(ValueError):
            center_pad_native(torch.zeros(1,4,700),
                              torch.ones(1,700,dtype=torch.bool),640,540)


if __name__=="__main__":
    unittest.main()
