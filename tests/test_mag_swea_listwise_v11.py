"""Synthetic offline regression tests for V11 full-grid listwise reranking."""
from __future__ import annotations

import csv
import tempfile
import unittest
from datetime import datetime,timezone
from importlib import import_module
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import patch

import numpy as np
import torch

from src.data.mpb_dual_windows import DualClip
from src.detection.mpb_listwise_reranker_v11 import (
    MPBListwiseReranker,physical_candidate_features,
    fit_feature_normalizer,tensor_event,soft_event_loss,
    select_candidate,distinct_topk_coverage,N_FEATURES
)

v11=import_module("scripts.5_training.5g_train_mag_swea_listwise_v11")


class V11ModelTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        torch.set_num_threads(2)

    def setUp(self):
        torch.manual_seed(11)
        self.model=MPBListwiseReranker()
        self.features=np.random.default_rng(4).normal(
            size=(113,N_FEATURES)
        ).astype(np.float32)
        self.features[:,0]=np.linspace(-4,4,113)
        self.offsets=np.arange(-1800,1800.01,32,dtype=float)
        self.assertEqual(len(self.offsets),113)

    def test_model_full_sequence_and_gradients(self):
        norm=fit_feature_normalizer([self.features])
        x,base=tensor_event({
            "features":self.features,"offsets":self.offsets
        },norm,torch.device("cpu"))
        scores,delta=self.model(x,base)
        self.assertEqual(tuple(scores.shape),(113,))
        self.assertEqual(tuple(delta.shape),(113,))
        loss=soft_event_loss(scores,self.offsets)
        self.assertTrue(torch.isfinite(loss))
        loss.backward()
        self.assertTrue(
            any(p.grad is not None for p in self.model.parameters())
        )

    def test_no_catalog_center_feature_leak(self):
        # Reassigning event labels must not modify the features/scores.
        norm=fit_feature_normalizer([self.features])
        x,base=tensor_event({
            "features":self.features,
            "offsets":np.zeros(113,dtype=np.float32),
        },norm,torch.device("cpu"))
        self.model.eval()
        with torch.no_grad():
            first,_=self.model(x,base)
        x2,base2=tensor_event({
            "features":self.features,
            "offsets":np.arange(113,dtype=np.float32),
        },norm,torch.device("cpu"))
        with torch.no_grad():
            second,_=self.model(x2,base2)
        torch.testing.assert_close(first,second)

    def test_alpha_zero_exact_v7b_sigmoid_tie_break(self):
        # Two huge logits saturate to exactly 1.0 in float32 sigmoid.
        base=np.array([100.,200.,-.1],dtype=np.float32)
        other=np.array([101.,201.,300.])
        self.assertEqual(select_candidate(other,base,alpha=0.),0)
        self.assertEqual(select_candidate(other,base,alpha=1.),2)

    def test_soft_time_label_prefers_true_center(self):
        offsets=torch.arange(-320.,321.,32.)
        right=-0.5*(offsets/56.).square()
        wrong=-0.5*((offsets-224.)/56.).square()
        a=soft_event_loss(right,offsets.numpy())
        b=soft_event_loss(wrong,offsets.numpy())
        self.assertLess(float(a),float(b))

    def test_oracle_topk_is_metric_only(self):
        base=np.array([1.,4.,3.,0.,2.])
        offs=np.array([-500.,-240.,-32.,100.,350.])
        nearest=distinct_topk_coverage(
            base,offs,k=3,min_gap_s=100.
        )
        self.assertEqual(nearest,32.)


class V11ScanFeatureTests(unittest.TestCase):
    def test_physical_features_measurements_and_teacher_only(self):
        b=3
        mag_t=torch.arange(-300.,300.)[None,:].repeat(b,1)
        swea_t=torch.arange(-288.,289.,32.)[None,:].repeat(b,1)
        batch={
            "mag":torch.randn(b,4,600),
            "mag_mask":torch.ones(b,600,dtype=torch.bool),
            "mag_time":mag_t,
            "swea":torch.randn(b,12,len(swea_t[0])),
            "swea_mask":torch.ones(
                b,len(swea_t[0]),dtype=torch.bool
            ),
            "swea_time":swea_t,
        }
        out={
            "transition_logit":torch.tensor([1.,-1.,0.]),
            "region_logits":torch.tensor([
                [1.,-1.],[-2.,2.],[0.,0.]
            ])
        }
        values=physical_candidate_features(batch,out)
        self.assertEqual(values.shape,(3,N_FEATURES))
        np.testing.assert_allclose(values[:,0],[1.,-1.,0.])

    def test_feature_cache_is_restart_safe(self):
        with tempfile.TemporaryDirectory() as tmp:
            file=Path(tmp)/"cache.npz"
            record={
                "features":np.zeros((4,N_FEATURES),dtype=np.float32),
                "offsets":np.array([-64.,-32.,0.,32.])
            }
            v11.save_feature_record(file,record,"source-1")
            loaded=v11.cache_feature_record(file,"source-1")
            np.testing.assert_array_equal(
                loaded["features"],record["features"]
            )
            np.testing.assert_array_equal(
                loaded["offsets"],record["offsets"]
            )
            with self.assertRaises(RuntimeError):
                v11.cache_feature_record(file,"source-2")

    def test_blend_zero_reverts_to_v10_candidate(self):
        t=datetime(2020,1,1,12,tzinfo=timezone.utc)
        ts=t.timestamp()
        record={
            "features":np.zeros((3,N_FEATURES),dtype=np.float32),
            "offsets":np.array([-64.,0.,64.],dtype=np.float32)
        }
        record["features"][:,0]=[0.,3.,2.]
        t_mag=np.arange(-300,300,dtype=np.float32)
        t_swea=np.arange(-288,289,32,dtype=np.float32)
        clip=DualClip(
            mag=np.zeros((600,4),dtype=np.float32),
            swea=np.zeros((len(t_swea),12),dtype=np.float32),
            mag_time_s=t_mag,
            swea_time_s=t_swea,
        )
        with patch.object(v11,"extract_clip",return_value=clip):
            selected=v11.selected_proposals(
                [{"time":t,"source_group":"Group1"}],
                {t:record},
                {t:np.array([3.,1.,8.])},
                0.,{t:{"MAG":None,"SWEA":None}}
            )
        self.assertEqual(selected[0]["coarse_offset_s"],0.)
        self.assertAlmostEqual(selected[0]["candidate_s"],ts)

    def test_strict_paired_metrics(self):
        with tempfile.TemporaryDirectory() as tmp:
            p10=Path(tmp)/"v10.csv"
            p7=Path(tmp)/"v7.csv"
            for path,error,offset in (
                (p10,40.,0.),(p7,70.,64.)
            ):
                with path.open("w",newline="") as file:
                    writer=csv.DictWriter(
                        file,fieldnames=[
                            "mpb_timestamp_utc","absolute_error_s",
                            "peak_offset_s"
                        ]
                    )
                    writer.writeheader()
                    writer.writerow({
                        "mpb_timestamp_utc":"2020-01-01T12:00:00+00:00",
                        "absolute_error_s":error,
                        "peak_offset_s":offset
                    })
            measures,output=v11.paired_metrics(
                [{
                    "mpb_timestamp_utc":
                        "2020-01-01T12:00:00+00:00",
                    "source_group":"Group1",
                    "n_scanned_windows":112,
                    "coarse_offset_s":0.,
                    "refined_mean_offset_s":12.,
                    "refined_peak_offset_s":7.,
                    "selected_offset_s":12.,
                    "absolute_error_s":12.,
                }],p10,p7
            )
            self.assertEqual(measures["improved_vs_v10"],1)
            self.assertEqual(measures["median_error_s"],12.)

    def test_target_not_used_to_pick_candidate(self):
        # The proposal is based on predictions only; ground truth offsets
        # are used solely to reconstruct the physical timestamps and loss.
        t=datetime(2020,1,1,12,tzinfo=timezone.utc)
        f=np.zeros((3,N_FEATURES),dtype=np.float32)
        f[:,0]=[0.,0.5,0.2]
        records={t:{"features":f,"offsets":np.array([
            -64.,0.,64.
        ],dtype=np.float32)}}
        clip=DualClip(
            mag=np.zeros((600,4),dtype=np.float32),
            swea=np.zeros((19,12),dtype=np.float32),
            mag_time_s=np.arange(-300,300,dtype=np.float32),
            swea_time_s=np.arange(-288,289,32,dtype=np.float32),
        )
        with patch.object(v11,"extract_clip",return_value=clip):
            result=v11.selected_proposals(
                [{"time":t,"source_group":"Group1"}],records,
                {t:np.array([0.,0.,5.])},1.,
                {t:{"MAG":None,"SWEA":None}}
            )
        self.assertEqual(result[0]["coarse_offset_s"],64.)


if __name__=="__main__":
    unittest.main()
