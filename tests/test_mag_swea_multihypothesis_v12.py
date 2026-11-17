"""V12 synthetic tests: shortlist, frozen refinement, label safety, fallback."""
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
from src.detection.mpb_multi_hypothesis_v12 import (
    N_FEATURES,MultiHypothesisSelector,diverse_shortlist,
    assemble_features,consensus_features,fit_feature_norm,
    normalize,ranking_loss,selected_index,theoretical_coverage
)

v12=import_module("scripts.5_training.5h_train_mag_swea_multihypothesis_v12")


class V12SelectorTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        torch.set_num_threads(2)

    def setUp(self):
        torch.manual_seed(12)
        self.rng=np.random.default_rng(12)
        self.features=self.rng.normal(
            size=(12,N_FEATURES)
        ).astype(np.float32)

    def test_diverse_topk_and_v7b_ties(self):
        logits=np.asarray(
            [100.,200.,3.,2.,1.],dtype=np.float32
        )
        seconds=np.array([0.,300.,320.,600.,900.])
        # sigmoid(100) and sigmoid(200) are both 1.0 at float32.
        indices=diverse_shortlist(
            logits,seconds,topk=3,min_gap_s=120.
        )
        np.testing.assert_array_equal(indices,[0,1,3])

    def test_shift_of_physical_times_does_not_change_features(self):
        k=12
        physical=self.rng.normal(size=(k,16)).astype(np.float32)
        centers=np.arange(k,dtype=float)*180.+1.5e9
        fine=np.linspace(-60,60,k)
        presence=np.linspace(-1,3,k)
        args=dict(
            mean_delta=fine,peak_delta=fine+4,
            presence_logit=presence,
            entropy=np.full(k,.8),
            peak_prob=np.full(k,.035),
            std_seconds=np.full(k,35.),
            fine_alpha=1.,fine_method="mean"
        )
        f1,t1=assemble_features(physical,centers,**args)
        f2,t2=assemble_features(physical,centers+1234567.,**args)
        np.testing.assert_allclose(f1,f2,atol=1e-7)
        np.testing.assert_allclose(t2-t1,1234567.)
        self.assertEqual(f1.shape,(k,N_FEATURES))

    def test_permutation_equivariant_selector(self):
        model=MultiHypothesisSelector().eval()
        x=torch.tensor(self.features)
        order=torch.tensor([4,2,1,0,3,5,11,8,6,10,9,7])
        with torch.no_grad():
            a=model(x)
            b=model(x[order])
        torch.testing.assert_close(b,a[order],atol=1e-6,rtol=1e-6)

    def test_supervised_loss_prefers_true_best_of_k(self):
        labels=np.array([12.,50.,150.,300.])
        good=torch.tensor([4.,1.,-2.,-3.],requires_grad=True)
        bad=torch.tensor([-4.,-2.,1.,4.])
        goodloss=ranking_loss(good,labels)
        badloss=ranking_loss(bad,labels)
        self.assertLess(float(goodloss),float(badloss))
        goodloss.backward()
        self.assertIsNotNone(good.grad)

    def test_alpha_zero_restores_first_v7b_candidate(self):
        coarse=np.array([10.,8.,7.],dtype=np.float32)
        pred=np.array([-50.,3.,12.])
        self.assertEqual(selected_index(pred,coarse,0.),0)
        self.assertEqual(selected_index(pred,coarse,1.),2)

    def test_train_only_feature_normalizer(self):
        train=[{"features":self.features}]
        norm=fit_feature_norm(train)
        old=norm["mean"].copy()
        unseen={"features":self.features+1000}
        _=normalize(unseen,norm,torch.device("cpu"))
        np.testing.assert_array_equal(old,norm["mean"])

    def test_oracle_shortlist_diagnostic(self):
        records=[
            {"predicted_offsets":np.array([100.,65.,-40.])},
            {"predicted_offsets":np.array([110.,200.,-150.])}
        ]
        self.assertEqual(theoretical_coverage(records),.5)


class V12TrainingInfrastructureTests(unittest.TestCase):
    def test_cache_round_trip_and_wrong_checkpoint_rejected(self):
        with tempfile.TemporaryDirectory() as tmp:
            path=Path(tmp)/"f.npz"
            data={
                "features":np.zeros((2,N_FEATURES),dtype=np.float32),
                "candidate_offsets":np.array([0.,120.],np.float32),
                "predicted_offsets":np.array([-15.,130.],np.float32),
                "coarse_teacher":np.array([5.,0.],np.float32),
                "n_scanned_windows":4,
            }
            v12.save_feature(path,data,"model-a")
            got=v12.read_feature(path,"model-a")
            np.testing.assert_array_equal(
                got["features"],data["features"]
            )
            with self.assertRaises(RuntimeError):
                v12.read_feature(path,"model-b")

    def test_measured_scan_and_refinement_of_every_shortlist_item(self):
        t=datetime(2020,1,1,12,tzinfo=timezone.utc)
        timestamp=t.timestamp()
        centers=[timestamp-400.,timestamp-160.,
                 timestamp,timestamp+250.]
        candidates=[
            SimpleNamespace(center_s=center)
            for center in centers
        ]
        mag_time=np.arange(-300.,300.,dtype=np.float32)
        swea_time=np.arange(-288.,289.,32,dtype=np.float32)
        def fake_clip(mag,swea,center):
            level=centers.index(center)+1.
            return DualClip(
                mag=np.full((600,4),level,dtype=np.float32),
                swea=np.full(
                    (len(swea_time),12),level/2,dtype=np.float32
                ),
                mag_time_s=mag_time,swea_time_s=swea_time
            )

        class FakeTeacher:
            def eval(self):
                return self
            def __call__(self,mag,mag_mask,mag_time,
                         swea,swea_mask,swea_time):
                score=mag[:,0,:].mean(-1)
                return {
                    "transition_logit":score,
                    "region_logits":torch.stack((score,-score),dim=1)
                }

        class FakeFine:
            max_offset_s=250.
            def __init__(self):
                self.calls=0
                self.seen=0
            def __call__(self,mag,mag_mask,mag_time,
                         swea,swea_mask,swea_time):
                self.calls+=1
                self.seen+=mag.shape[0]
                unnormalized=torch.exp(
                    -.5*(mag_time/30.).square()
                )*mag_mask
                prob=unnormalized/unnormalized.sum(-1,keepdim=True)
                return {
                    "time_prob":prob,
                    "expected_offset_s":torch.zeros(
                        mag.shape[0]
                    ),
                    "peak_offset_s":torch.zeros(mag.shape[0]),
                    "presence_logit":mag[:,0,:].mean(-1)
                }
        teacher=FakeTeacher()
        fine=FakeFine()
        norm={
            "MAG":{"mean":np.zeros(4),"scale":np.ones(4)},
            "SWEA":{"mean":np.zeros(12),"scale":np.ones(12)}
        }
        args=SimpleNamespace(
            evaluation_stride_s=32.,batch_size=2,
            topk=3,min_gap_s=120.
        )
        row={"time":t}
        with patch.object(
            v12,"candidate_evaluation_grid",return_value=candidates
        ),patch.object(v12,"extract_clip",side_effect=fake_clip):
            record=v12.scan_refined_hypotheses(
                teacher,fine,row,
                {t:{"MAG":None,"SWEA":None}},
                norm,{"alpha":1.,"method":"mean"},
                args,torch.device("cpu"),
                np.array([timestamp]),np.array([timestamp-2000.])
            )
        self.assertEqual(record["features"].shape,(3,N_FEATURES))
        self.assertEqual(fine.seen,3)
        self.assertEqual(fine.calls,2)
        self.assertEqual(
            record["n_scanned_windows"],4
        )
        self.assertAlmostEqual(
            float(record["candidate_offsets"][0]),250.
        )

    def test_alpha_zero_identical_v10_metrics(self):
        with tempfile.TemporaryDirectory() as tmp:
            path10=Path(tmp)/"v10.csv"
            path7=Path(tmp)/"v7.csv"
            for path,err,offset in (
                (path10,29.7,29.7),(path7,64.,64.)
            ):
                with path.open("w",newline="",encoding="utf8") as fp:
                    writer=csv.DictWriter(fp,fieldnames=[
                        "mpb_timestamp_utc","absolute_error_s",
                        "selected_offset_s","peak_offset_s"
                    ])
                    writer.writeheader()
                    writer.writerow({
                        "mpb_timestamp_utc":"2020-01-01T12:00:00+00:00",
                        "absolute_error_s":err,
                        "selected_offset_s":offset,
                        "peak_offset_s":64.
                    })
            data=[{
                "mpb_timestamp_utc":"2020-01-01T12:00:00+00:00",
                "source_group":"Group1",
                "n_scanned_windows":113,"n_shortlisted":3,
                "chosen_hypothesis_rank":1,
                "candidate_offset_s":64.,
                "selected_offset_s":31.,"absolute_error_s":31.,
                "shortlist_nearest_refined_s":20.,
                "shortlist_nearest_coarse_s":64.,
            }]
            report,events=v12.paired_metrics(
                data,path10,path7,alpha=0.
            )
            self.assertAlmostEqual(report["median_error_s"],29.7)
            self.assertAlmostEqual(
                report["paired_v10_median_s"],29.7
            )
            self.assertEqual(report["tied_vs_v10"],1)
            self.assertAlmostEqual(
                events[0]["selected_offset_s"],29.7
            )


if __name__=="__main__":
    unittest.main()
