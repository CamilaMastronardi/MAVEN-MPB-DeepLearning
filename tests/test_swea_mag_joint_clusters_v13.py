"""Synthetic SWEA-native-cadence and matched MAG+SWEA clustering tests."""
from __future__ import annotations

import csv
import tempfile
import unittest
from datetime import datetime,timedelta,timezone
from importlib import import_module
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import torch

from src.data.mpb_dual_windows import DualClip
from src.detection.mpb_swea_autoencoder_v13 import (
    LENGTH,SWEA3DAutoencoder,swea_collate,autoencoder_loss
)

joint=import_module("scripts.5_training.5j_train_swea_autoencoder_joint_v13")


class SweaAutoencoderV13Tests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        torch.set_num_threads(2)

    def setUp(self):
        self.model=SWEA3DAutoencoder()
        self.norm={
            "mean":np.zeros(12,dtype=np.float32),
            "scale":np.ones(12,dtype=np.float32),
        }

    @staticmethod
    def measured_clip(frames=18):
        rng=np.random.default_rng(16)
        return DualClip(
            mag=np.zeros((600,4),dtype=np.float32),
            swea=rng.normal(size=(frames,12)).astype(np.float32),
            mag_time_s=np.arange(600,dtype=np.float32)-300.,
            swea_time_s=np.arange(frames,dtype=np.float32)*32.-288.
        )

    def test_eight_layers_and_three_latent_variables(self):
        self.assertEqual(len(self.model.encoder_layers),8)
        self.assertEqual(len(self.model.decoder_layers),8)
        batch=swea_collate(
            [(self.measured_clip(),1,64.),
             (self.measured_clip(20),0,0.)],
            self.norm
        )
        out=self.model(batch["mag"],batch["mask"])
        self.assertEqual(tuple(out["latent"].shape),(2,3))
        self.assertEqual(
            tuple(out["reconstruction"].shape),(2,12,LENGTH)
        )
        self.assertEqual(tuple(out["offset_s"].shape),(2,))
        loss,metrics=autoencoder_loss(out,batch)
        self.assertTrue(bool(torch.isfinite(loss)))
        self.assertEqual(metrics["positive_examples"],1)
        loss.backward()
        self.assertTrue(any(
            p.grad is not None and
            bool(p.grad.abs().sum()>0)
            for p in self.model.encoder_layers[0].parameters()
        ))

    def test_no_temporal_interpolation_and_padding_mask(self):
        clip=self.measured_clip(frames=18)
        result=swea_collate([(clip,1,32.)],self.norm)
        self.assertEqual(int(result["mask"].sum()),18)
        self.assertEqual(result["mag"].shape,(1,12,LENGTH))
        self.assertEqual(
            torch.count_nonzero(
                result["mag"][:,:,~result["mask"][0]]
            ).item(),0
        )
        left=(LENGTH-18)//2
        np.testing.assert_array_equal(
            result["mag"][0,:,left:left+18].T.numpy(),clip.swea
        )

    def test_full_32_frame_window(self):
        batch=swea_collate(
            [(self.measured_clip(32),1,0.)],self.norm
        )
        self.assertEqual(int(batch["mask"].sum()),32)

    def test_real_38_frame_window_is_preserved_and_reconstructed(self):
        clip=self.measured_clip(38)
        batch=swea_collate([(clip,1,0.)],self.norm)
        self.assertEqual(int(batch["mask"].sum()),38)
        left=(LENGTH-38)//2
        np.testing.assert_array_equal(
            batch["mag"][0,:,left:left+38].T.cpu().numpy(),
            clip.swea
        )
        self.assertEqual(
            int(torch.count_nonzero(
                batch["mag"][:,:,~batch["mask"][0]]
            )),0
        )
        outputs=self.model(batch["mag"],batch["mask"])
        self.assertEqual(
            tuple(outputs["reconstruction"].shape),(1,12,LENGTH)
        )
        loss,metrics=autoencoder_loss(outputs,batch)
        self.assertTrue(bool(torch.isfinite(loss)))
        loss.backward()

    def test_long_128_frame_window_and_upper_capacity(self):
        for frames in (64,128,LENGTH):
            with self.subTest(frames=frames):
                clip=self.measured_clip(frames)
                batch=swea_collate([(clip,1,0.)],self.norm)
                self.assertEqual(int(batch["mask"].sum()),frames)
                reconstructed=self.model(
                    batch["mag"],batch["mask"]
                )["reconstruction"]
                self.assertEqual(
                    tuple(reconstructed.shape),(1,12,LENGTH)
                )
        with self.assertRaisesRegex(ValueError,"256"):
            swea_collate(
                [(self.measured_clip(LENGTH+1),1,0.)],self.norm
            )

    def test_invalid_frame_count_rejected(self):
        with self.assertRaises(ValueError):
            swea_collate(
                [(self.measured_clip(11),1,0.)],self.norm
            )

    def test_padding_is_ignored_in_reconstruction_loss(self):
        batch=swea_collate(
            [(self.measured_clip(),1,0.)],self.norm
        )
        out=self.model(batch["mag"],batch["mask"])
        a,_=autoencoder_loss(
            out,batch,detection_weight=0.,temporal_weight=0.
        )
        changed={**out,"reconstruction":out["reconstruction"].clone()}
        absent=~batch["mask"][:,None,:].expand(-1,12,-1)
        changed["reconstruction"][absent]+=1e3
        b,_=autoencoder_loss(
            changed,batch,detection_weight=0.,temporal_weight=0.
        )
        torch.testing.assert_close(a,b)

    def test_offset_label_not_model_input(self):
        clip=self.measured_clip()
        a=swea_collate([(clip,1,-180.)],self.norm)
        b=swea_collate([(clip,1,180.)],self.norm)
        self.model.eval()
        with torch.no_grad():
            z1=self.model(a["mag"],a["mask"])["latent"]
            z2=self.model(b["mag"],b["mask"])["latent"]
        torch.testing.assert_close(z1,z2)


class JointClusteringTests(unittest.TestCase):
    def test_mag_csv_exact_timestamp_join_and_split_protection(self):
        with tempfile.TemporaryDirectory() as tmp:
            path=Path(tmp)/"mag.csv"
            times=[
                datetime(2020,1,i+1,tzinfo=timezone.utc)
                for i in range(2)
            ]
            with path.open("w",newline="",encoding="utf-8") as f:
                w=csv.DictWriter(f,fieldnames=[
                    "mpb_timestamp_utc","split","cluster",
                    "z1","z2","z3"
                ])
                w.writeheader()
                # Deliberately shuffled CSV order.
                for t in times[::-1]:
                    w.writerow({
                        "mpb_timestamp_utc":t.isoformat(),
                        "split":"fit","cluster":1,
                        "z1":1.,"z2":2.,"z3":3.
                    })
            rows=[{"time":t} for t in times]
            source=joint.read_mag_reference(
                path,rows,strict_split={t:"fit" for t in times}
            )
            self.assertEqual(list(source),times)
            self.assertEqual(float(source[times[0]]["z3"]),3.)
            with self.assertRaises(RuntimeError):
                joint.read_mag_reference(
                    path,rows,strict_split={
                        times[0]:"outer_holdout",times[1]:"fit"
                    }
                )

    def test_joint_six_dimensional_fit_uses_only_train(self):
        rng=np.random.default_rng(33)
        start=datetime(2020,2,1,tzinfo=timezone.utc)
        times=[start+timedelta(days=i) for i in range(25)]
        rows=[]
        swea_emb={}
        with tempfile.TemporaryDirectory() as tmp:
            directory=Path(tmp)
            mag_csv=directory/"mag.csv"
            with mag_csv.open("w",newline="",encoding="utf-8") as f:
                writer=csv.DictWriter(f,fieldnames=[
                    "mpb_timestamp_utc","split","cluster",
                    "z1","z2","z3"
                ])
                writer.writeheader()
                for i,t in enumerate(times):
                    is_first=i<12
                    anchor=-3. if is_first else 3.
                    mag=rng.normal(anchor,.1,size=3)
                    swea=rng.normal(anchor,.1,size=3)
                    partition="fit" if i<20 else "outer_holdout"
                    writer.writerow({
                        "mpb_timestamp_utc":t.isoformat(),
                        "split":partition,"cluster":1 if is_first else 2,
                        "z1":mag[0],"z2":mag[1],"z3":mag[2]
                    })
                    swea_emb[t]={
                        "latent":swea,"views":9,
                        "window_mse":.15,"positive_prob":.7,
                        "latent_spread":.2
                    }
                    rows.append({
                        "time":t,"source_group":"Group1",
                        "fold":1 if i<20 else 4
                    })
            args=SimpleNamespace(
                mag_latents_csv=mag_csv,output_dir=directory,
                pilot_events_per_fold=0,cluster_k_min=2,
                cluster_k_max=3,seed=42
            )
            split={r["time"]:(
                "fit" if j<20 else "outer_holdout"
            ) for j,r in enumerate(rows)}
            report,params=joint.joint_analysis(
                args,rows,rows[:20],rows[20:],split,swea_emb
            )
            self.assertEqual(report["comparison"]["n_common_events"],25)
            self.assertEqual(report["joint_6d"]["selected_k"],2)
            self.assertGreater(report["joint_6d"]["fit_silhouette"],.7)
            self.assertEqual(
                params["joint_kmeans_centers"].shape[1],6
            )
            self.assertTrue(
                (directory/"mag_swea_joint_pca3d.png").exists()
            )
            self.assertTrue(
                (directory/"swea_latent_clusters_3d.png").exists()
            )
            self.assertTrue(
                (directory/"mag_swea_joint_groups.csv").exists()
            )
            self.assertEqual(
                len(list(csv.DictReader(
                    (directory/"mag_swea_joint_groups.csv").open()
                ))),25
            )


if __name__=="__main__":
    unittest.main()
