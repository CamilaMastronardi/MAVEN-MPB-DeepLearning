"""Offline V10 tests: frozen physical teacher, temporal offset and calibration."""
from __future__ import annotations

import csv
import tempfile
import unittest
from datetime import datetime,timezone
from importlib import import_module
from pathlib import Path

import numpy as np
import torch

from src.detection.mag_swea_region_model import MagSweaRegionAutoencoder
from src.detection.mpb_region_anchored_refiner import (
    RegionAnchoredTemporalRefiner,refiner_loss,shifted_context
)

v10=import_module("scripts.5_training.5f_train_mag_swea_refiner_v10")


class V10ArchitectureTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        torch.set_num_threads(2)

    def setUp(self):
        torch.manual_seed(102)
        self.original=MagSweaRegionAutoencoder().eval()
        self.refiner=RegionAnchoredTemporalRefiner(
            self.original.state_dict()
        )
        self.mag_t=torch.arange(-300.,300.)[None,:].repeat(2,1)
        self.swea_t=torch.arange(-288.,289.,32.)[None,:].repeat(2,1)
        self.mag=torch.randn(2,4,600)
        self.swea=torch.randn(2,12,self.swea_t.shape[-1])
        self.mag_mask=torch.ones(2,600,dtype=torch.bool)
        self.swea_mask=torch.ones(
            2,self.swea_t.shape[-1],dtype=torch.bool
        )

    def run_refiner(self):
        return self.refiner(
            self.mag,self.mag_mask,self.mag_t,
            self.swea,self.swea_mask,self.swea_t
        )

    def test_frozen_v7b_teacher_matches_original_both_heads(self):
        self.refiner.train()
        with torch.no_grad():
            expected=self.original(
                self.mag,self.mag_mask,self.mag_t,
                self.swea,self.swea_mask,self.swea_t
            )
            result=self.run_refiner()
        torch.testing.assert_close(
            result["teacher_logit"],expected["transition_logit"]
        )
        torch.testing.assert_close(
            result["teacher_region_logits"],expected["region_logits"]
        )
        self.assertFalse(self.refiner.teacher.training)
        self.assertTrue(
            all(not p.requires_grad
                for p in self.refiner.teacher.parameters())
        )

    def test_output_is_native_mag_timeline_inside_roi(self):
        out=self.run_refiner()
        self.assertEqual(tuple(out["time_logits"].shape),(2,600))
        self.assertEqual(tuple(out["presence_logit"].shape),(2,))
        self.assertEqual(tuple(out["expected_offset_s"].shape),(2,))
        self.assertEqual(tuple(out["peak_offset_s"].shape),(2,))
        torch.testing.assert_close(
            out["time_prob"].sum(-1),torch.ones(2)
        )
        self.assertLess(
            float(out["time_prob"][:,0].max()),1e-8
        )
        self.assertLessEqual(
            float(out["peak_offset_s"].abs().max()),250.
        )

    def test_backprop_only_through_refinement_heads(self):
        output=self.run_refiner()
        positive=torch.tensor([1.,0.])
        offset=torch.tensor([100.,0.])
        loss,metrics=refiner_loss(
            output,self.mag_t,self.mag_mask,positive,offset
        )
        self.assertTrue(torch.isfinite(loss))
        self.assertEqual(metrics["n_positive"],1)
        loss.backward()
        self.assertTrue(any(
            p.grad is not None for p in self.refiner.local_head.parameters()
        ))
        self.assertTrue(any(
            p.grad is not None
            for p in self.refiner.confidence_head.parameters()
        ))
        self.assertTrue(all(
            p.grad is None for p in self.refiner.teacher.parameters()
        ))

    def test_time_loss_prefers_correct_offset(self):
        target=torch.tensor([110.,-90.])
        valid=(self.mag_t.abs()<=250.)
        def make_prediction(shift_s):
            time_logits= -.5*(
                (self.mag_t-(target[:,None]+shift_s))/32.
            ).square()
            time_logits=time_logits.masked_fill(~valid,-1e4)
            p=torch.softmax(time_logits,dim=-1)
            return {
                "time_logits":time_logits,
                "expected_offset_s":(p*self.mag_t).sum(-1),
                "presence_logit":torch.tensor([2.,2.])
            }
        good,_=refiner_loss(
            make_prediction(0),self.mag_t,self.mag_mask,
            torch.tensor([1.,1.]),target
        )
        bad,_=refiner_loss(
            make_prediction(140),self.mag_t,self.mag_mask,
            torch.tensor([1.,1.]),target
        )
        self.assertLess(float(good),float(bad))

    def test_no_localization_target_assigned_to_far_negatives(self):
        model_out=self.run_refiner()
        loss,stats=refiner_loss(
            model_out,self.mag_t,self.mag_mask,torch.zeros(2),
            torch.tensor([1000.,-1000.])
        )
        self.assertEqual(stats["n_positive"],0)
        self.assertEqual(stats["temporal_kl"],0.)
        self.assertEqual(stats["offset"],0.)
        self.assertTrue(torch.isfinite(loss))

    def test_shifted_context_constant_zero_except_boundaries(self):
        data=torch.ones(1,2,100)
        result=shifted_context(data,5)
        self.assertTrue(
            torch.allclose(result[:,:,5:-5],torch.zeros(1,2,90))
        )


class V10CalibrationTests(unittest.TestCase):
    def row(self,coarse,mean,peak=0.):
        return {
            "coarse_offset_s":float(coarse),
            "refined_mean_offset_s":float(mean),
            "refined_peak_offset_s":float(peak),
        }

    def test_fallback_exact_v7b_when_correction_is_harmful(self):
        monitor=[
            self.row(20.,200.,200.),
            self.row(-35.,-190.,-190.),
            self.row(50.,180.,180.),
        ]
        cfg=v10.calibrate_refinement(monitor)
        self.assertEqual(cfg["alpha"],0.)
        for row in monitor:
            error,offset=v10.corrected_error(
                row,cfg["method"],cfg["alpha"]
            )
            self.assertEqual(offset,row["coarse_offset_s"])
            self.assertEqual(error,abs(row["coarse_offset_s"]))

    def test_calibration_can_choose_genuine_correction(self):
        monitor=[
            self.row(140.,-140.,-140.),
            self.row(-160.,160.,160.),
            self.row(190.,-190.,-190.),
        ]
        cfg=v10.calibrate_refinement(monitor)
        self.assertGreater(cfg["alpha"],0.)
        self.assertLess(cfg["monitor_median_error_s"],50.)

    def test_strict_paired_reference_and_fallback(self):
        with tempfile.TemporaryDirectory() as tmp:
            file=Path(tmp)/"base.csv"
            with file.open("w",newline="",encoding="utf-8") as handle:
                writer=csv.DictWriter(
                    handle,fieldnames=[
                        "mpb_timestamp_utc","peak_offset_s",
                        "absolute_error_s"
                    ]
                )
                writer.writeheader()
                writer.writerow({
                    "mpb_timestamp_utc":"2020-01-01T12:00:00+00:00",
                    "peak_offset_s":56.,
                    "absolute_error_s":56.,
                })
            data=[{
                "mpb_timestamp_utc":"2020-01-01T12:00:00+00:00",
                "source_group":"Group1",
                "coarse_offset_s":56.,
                "refined_mean_offset_s":-15.,
                "refined_peak_offset_s":-20.,
                "refiner_confidence":0.8,
                "n_scanned_windows":108,
            }]
            metrics,rows=v10.evaluate_fold(
                data,{"alpha":0.,"method":"mean"},file
            )
            self.assertEqual(metrics["median_abs_peak_error_s"],56.)
            self.assertEqual(metrics["paired_tied"],1)
            wrong=[{**data[0],"coarse_offset_s":88.}]
            with self.assertRaises(RuntimeError):
                v10.evaluate_fold(
                    wrong,{"alpha":1.,"method":"peak"},file
                )


if __name__=="__main__":
    unittest.main()
