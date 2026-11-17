"""V13 phase-1 synthetic tests for 8-layer MAG AE, 3D grouping and labels."""
from __future__ import annotations

import unittest
from datetime import datetime,timezone
from importlib import import_module
from types import SimpleNamespace

import numpy as np
import torch
from sklearn.metrics import silhouette_score

from src.data.mpb_dual_windows import DualClip
from src.detection.mpb_cnn_autoencoder_v13 import (
    LENGTH,Conv3DAutoencoder,mag_collate,autoencoder_loss
)

v13=import_module("scripts.5_training.5i_train_mag_autoencoder_clusters_v13")


class V13ArchitectureTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        torch.set_num_threads(2)

    def setUp(self):
        torch.manual_seed(13)
        self.model=Conv3DAutoencoder()
        self.norm={
            "mean":np.zeros(4),
            "scale":np.ones(4),
        }

    def clip(self,n=600,amplitude=1.):
        return DualClip(
            mag=np.random.default_rng(n).normal(
                size=(n,4)
            ).astype(np.float32)*amplitude,
            swea=np.zeros((19,12),dtype=np.float32),
            mag_time_s=np.arange(n,dtype=np.float32)-n//2,
            swea_time_s=np.arange(19,dtype=np.float32)*32-288.
        )

    def test_encoder_exactly_eight_conv_and_three_scalar_variables(self):
        self.assertEqual(len(self.model.encoder_layers),8)
        self.assertEqual(len(self.model.decoder_layers),8)
        x=torch.randn(2,4,LENGTH)
        m=torch.zeros(2,LENGTH,dtype=torch.bool)
        m[:,20:620]=True
        out=self.model(x,m)
        self.assertEqual(tuple(out["latent"].shape),(2,3))
        self.assertEqual(tuple(out["reconstruction"].shape),(2,4,LENGTH))
        self.assertEqual(tuple(out["presence_logit"].shape),(2,))
        self.assertEqual(tuple(out["offset_s"].shape),(2,))
        self.assertLessEqual(float(out["offset_s"].abs().max()),240.)

    def test_two_windows_positive_negative_and_time_head_gradient(self):
        batch=mag_collate(
            [(self.clip(),1,180.),
             (self.clip(),0,0.)],
            self.norm
        )
        out=self.model(batch["mag"],batch["mask"])
        loss,metrics=autoencoder_loss(out,batch)
        self.assertTrue(bool(torch.isfinite(loss)))
        self.assertEqual(metrics["positive_examples"],1)
        loss.backward()
        self.assertTrue(any(
            p.grad is not None and bool(p.grad.abs().sum()>0)
            for p in self.model.encoder_layers[0].parameters()
        ))
        self.assertTrue(any(
            p.grad is not None and bool(p.grad.abs().sum()>0)
            for p in self.model.offset_head.parameters()
        ))

    def test_reconstruction_padding_is_not_supervised(self):
        batch=mag_collate(
            [(self.clip(600),1,0.)],self.norm
        )
        out=self.model(batch["mag"],batch["mask"])
        loss1,met1=autoencoder_loss(
            out,batch,detection_weight=0.,temporal_weight=0.
        )
        perturbed={**out,
                   "reconstruction":out["reconstruction"].clone()}
        perturbed["reconstruction"][
            ~batch["mask"][:,None,:].expand(-1,4,-1)
        ]+=1e4
        loss2,met2=autoencoder_loss(
            perturbed,batch,detection_weight=0.,temporal_weight=0.
        )
        torch.testing.assert_close(loss1,loss2)

    def test_601_endpoint_samples_allowed_without_interpolation(self):
        clip=self.clip(601)
        batch=mag_collate([(clip,1,32.)],self.norm)
        self.assertEqual(int(batch["mask"].sum()),601)
        self.assertEqual(tuple(batch["mag"].shape),(1,4,640))

    def test_loss_temporal_target_only_for_positive(self):
        batch=mag_collate(
            [(self.clip(),0,1000.)],self.norm
        )
        out=self.model(batch["mag"],batch["mask"])
        _,metrics=autoencoder_loss(out,batch)
        self.assertEqual(metrics["positive_examples"],0)
        self.assertEqual(metrics["time_huber"],0.)

    def test_window_time_is_not_encoder_input(self):
        batch1=mag_collate(
            [(self.clip(),1,0.)],self.norm
        )
        batch2=mag_collate(
            [(self.clip(),1,240.)],self.norm
        )
        self.model.eval()
        with torch.no_grad():
            out1=self.model(batch1["mag"],batch1["mask"])
            out2=self.model(batch2["mag"],batch2["mask"])
        torch.testing.assert_close(out1["latent"],out2["latent"])
        torch.testing.assert_close(out1["offset_s"],out2["offset_s"])


class V13ClusteringTests(unittest.TestCase):
    def test_cluster_fit_only_training_event_latents(self):
        rng=np.random.default_rng(13)
        center1=rng.normal(loc=(-3.,0.,0.),scale=.15,size=(9,3))
        center2=rng.normal(loc=(3.,0.,0.),scale=.15,size=(9,3))
        train=np.concatenate((center1,center2))
        rows=[]
        records={}
        for i,z in enumerate(train):
            when=datetime(2020,1,i+1,tzinfo=timezone.utc)
            rows.append({"time":when})
            records[when]={"latent":z}
        args=SimpleNamespace(
            cluster_k_min=2,cluster_k_max=4,seed=42
        )
        clustering=v13.choose_clustering(rows,records,args)
        self.assertEqual(clustering["selection"]["k"],2)
        self.assertGreater(clustering["selection"]["silhouette"],.8)
        old=clustering["estimator"].cluster_centers_.copy()
        # Assigning a holdout point should NOT refit cluster centers.
        holdout=(np.array([10000.,0.,0.])-
                 clustering["scaler_mean"])/clustering["scaler_scale"]
        _=clustering["estimator"].predict(holdout[None,:])
        np.testing.assert_allclose(
            old,clustering["estimator"].cluster_centers_
        )

    def test_group_assignment_retains_all_events_and_missing_flags(self):
        t1=datetime(2020,1,1,tzinfo=timezone.utc)
        t2=datetime(2020,1,2,tzinfo=timezone.utc)
        class Cluster:
            def predict(self,x):
                return np.zeros(len(x),dtype=int)
        data={
            "estimator":Cluster(),
            "scaler_mean":np.zeros(3),
            "scaler_scale":np.ones(3)
        }
        assigned=v13.assign_groups(
            [
                {"time":t1,"fold":1,"source_group":"Group1"},
                {"time":t2,"fold":4,"source_group":"Group3"}
            ],
            {t1:{
                "latent":np.array([0.,1.,2.]),
                "views":4,"window_mse":.2,
                "positive_prob":.9,"latent_spread":.4,
            }},
            data,
            {t1:"fit",t2:"outer_holdout"},
        )
        self.assertEqual(len(assigned),2)
        self.assertEqual(assigned[0]["cluster"],1)
        self.assertEqual(assigned[1]["cluster"],"")
        self.assertEqual(assigned[1]["split"],"outer_holdout")


if __name__=="__main__":
    unittest.main()
