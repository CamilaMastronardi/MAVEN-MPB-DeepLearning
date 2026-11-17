"""Offline synthetic regression tests for V9 dense MAG+SWEA MPB localization."""
from __future__ import annotations

import unittest
from datetime import datetime,timezone
from importlib import import_module

import numpy as np
import torch

from src.detection.mag_swea_region_model import MagSweaRegionAutoencoder
from src.detection.mpb_temporal_localizer import (
    MagSweaTemporalLocalizer, temporal_loss, nearest_swea_latent
)

v9=import_module("scripts.5_training.5e_train_mag_swea_temporal_v9")


class TemporalV9Tests(unittest.TestCase):
    def setUp(self):
        torch.manual_seed(1)
        torch.set_num_threads(2)
        self.mag_t=torch.arange(-300.,300.)[None,:].repeat(2,1)
        self.swea_t=torch.arange(-288.,289.,32.)[None,:].repeat(2,1)
        self.mag=torch.randn(2,4,600)
        self.swea=torch.randn(2,12,self.swea_t.shape[-1])
        self.mag_mask=torch.ones(2,600,dtype=torch.bool)
        self.swea_mask=torch.ones(
            2,self.swea_t.shape[-1],dtype=torch.bool
        )

    def test_forward_density_and_confidence(self):
        model=MagSweaTemporalLocalizer()
        out=model(
            self.mag,self.mag_mask,self.mag_t,
            self.swea,self.swea_mask,self.swea_t
        )
        self.assertEqual(tuple(out["time_logits"].shape),(2,600))
        self.assertEqual(tuple(out["window_logit"].shape),(2,))
        self.assertTrue(torch.isfinite(out["time_prob"]).all())
        torch.testing.assert_close(
            out["time_prob"].sum(-1),torch.ones(2)
        )

    def test_time_kl_prefers_correct_peak(self):
        mask=torch.ones(2,600,dtype=torch.bool)
        target=torch.tensor([60.,-90.])
        times=self.mag_t
        correct=-.5*((times-target[:,None])/32.).square()
        displaced=-.5*((times-(target[:,None]+140.))/32.).square()
        def pack(x):
            return {"time_logits":x,
                    "window_logit":torch.tensor([3.,3.])}
        positive=torch.tensor([1.,1.])
        good,_=temporal_loss(
            pack(correct),times,mask,positive,target
        )
        bad,_=temporal_loss(
            pack(displaced),times,mask,positive,target
        )
        self.assertLess(float(good),float(bad))

    def test_negative_has_no_localization_supervision(self):
        model=MagSweaTemporalLocalizer()
        out=model(
            self.mag,self.mag_mask,self.mag_t,
            self.swea,self.swea_mask,self.swea_t
        )
        total,details=temporal_loss(
            out,self.mag_t,self.mag_mask,torch.zeros(2),
            torch.full((2,),999.)
        )
        self.assertEqual(details["time"],0.)
        self.assertEqual(details["n_positive"],0)
        total.backward()
        self.assertTrue(torch.isfinite(model.window_bias.grad))

    def test_masked_mag_time_samples_excluded(self):
        model=MagSweaTemporalLocalizer()
        magmask=self.mag_mask.clone()
        magmask[0,590:]=False
        out=model(
            self.mag,magmask,self.mag_t,
            self.swea,self.swea_mask,self.swea_t
        )
        self.assertLess(float(out["time_prob"][0,590:].max()),1e-8)
        torch.testing.assert_close(
            out["time_prob"][0].sum(),torch.ones(())
        )

    def test_nearest_swea_uses_true_timestamps(self):
        z=torch.tensor([[[1.,5.,9.]]])
        swea_time=torch.tensor([[-100.,0.,100.]])
        swea_mask=torch.tensor([[True,True,True]])
        mag_time=torch.tensor([[-90.,-49.,0.,51.,99.]])
        mag_mask=torch.ones((1,5),dtype=torch.bool)
        aligned=nearest_swea_latent(
            z,swea_time,swea_mask,mag_time,mag_mask
        )
        torch.testing.assert_close(
            aligned,torch.tensor([[[1.,5.,5.,9.,9.]]])
        )

    def test_load_pretrained_v7b_encoder_weights(self):
        prior=MagSweaRegionAutoencoder()
        new=MagSweaTemporalLocalizer()
        new.load_v7b_encoders(prior.state_dict())
        torch.testing.assert_close(
            new.mag.encoder[0].weight,prior.mag.encoder[0].weight
        )
        torch.testing.assert_close(
            new.swea.encoder[0].weight,prior.swea.encoder[0].weight
        )

    def test_pools_contain_shifted_positive_and_safe_negative(self):
        now=datetime(2020,1,1,12,tzinfo=timezone.utc)
        s=now.timestamp()
        mag_t=s+np.arange(-2100,2101,dtype=float)
        swea_t=s+np.arange(-2080,2081,32,dtype=float)
        blocks={
            now:{
                "MAG":(mag_t,np.zeros((len(mag_t),4))),
                "SWEA":(swea_t,np.zeros((len(swea_t),12)))
            }
        }
        bs=datetime.fromtimestamp(s-2000,tz=timezone.utc)
        pool,skip=v9.gather_training_pools(
            [{"time":now,"bs_time":bs}],blocks,
            np.asarray([s]),np.asarray([s-2000])
        )
        self.assertIn(now,pool)
        pos=pool[now]["positive"]
        neg=pool[now]["negative"]
        self.assertGreater(len(pos),3)
        self.assertGreater(len(neg),0)
        self.assertTrue(any(
            abs(item[2])>50 for item in pos
        ))
        self.assertTrue(all(item[1]==1 for item in pos))
        self.assertTrue(all(item[1]==0 for item in neg))


if __name__=="__main__":
    unittest.main()
