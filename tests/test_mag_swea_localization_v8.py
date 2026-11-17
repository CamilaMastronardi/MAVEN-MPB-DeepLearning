"""Offline regression tests for V8's conditional localization objective."""
from __future__ import annotations

import unittest
from datetime import datetime, timezone

import numpy as np
import torch

from src.data.mpb_weak_regions import Candidate
from src.detection.mpb_localization_refinement import (
    _localization_rank, gather_near_candidates, select_localization_groups
)


class V8LocalizationTests(unittest.TestCase):
    def setUp(self):
        self.time=datetime(2021,1,2,12,0,0,tzinfo=timezone.utc)
        self.s=self.time.timestamp()

    def candidate(self,offset,label):
        return Candidate(
            self.s,self.s+offset,label,
            (0 if label is not None else None),
            (1 if label == 1 else 0) if label is not None else None,
            "positive" if label==1 else (
                "stable_control" if label==0 else "mpb_ambiguous"
            )
        )

    def test_near_rank_prefers_center(self):
        examples=[
            (self.time,self.candidate(0.,1),None,1,(0,1)),
            (self.time,self.candidate(160.,None),None,None,(0,0)),
            (self.time,self.candidate(-224.,None),None,None,(0,0)),
        ]
        well=torch.tensor([2.,-1.,-.5],requires_grad=True)
        poor=torch.tensor([-1.,2.,.5],requires_grad=True)
        good,good_win,n=_localization_rank(well,examples,margin=.1)
        bad,bad_win,n2=_localization_rank(poor,examples,margin=.1)
        self.assertLess(float(good),float(bad))
        self.assertEqual((good_win,bad_win,n,n2),(1.,0.,1,1))
        good.backward()
        self.assertTrue(torch.isfinite(well.grad).all())

    def test_center_is_always_selected_and_near_unknown(self):
        source=[
            (self.candidate(0.,1),"clip0"),
            (self.candidate(-30.,1),"clipm"),
            (self.candidate(30.,1),"clipp"),
        ]
        source += [
            (self.candidate(float(x),0),"clipfar")
            for x in (-900,-600,600,900,1200,1500)
        ]
        nearby=[
            (self.candidate(float(x),None),"clipnear")
            for x in (-320,-160,-96,96,160,320)
        ]
        for epoch in range(1,8):
            groups=select_localization_groups(
                {self.time:source},{self.time:nearby},
                epoch=epoch,seed=44,hard_memory={},
                n_positives=2,n_hard=2,n_random=2,
            )
            self.assertEqual(len(groups),1)
            examples=groups[0][1]
            positive=[x for x in examples if x[3]==1]
            far=[x for x in examples if x[3]==0]
            near=[x for x in examples if x[3] is None]
            self.assertEqual(len(positive),2)
            self.assertEqual(len(far),4)
            self.assertEqual(len(near),2)
            self.assertTrue(any(
                abs(x[1].center_s-self.s)<.1 for x in positive
            ))
            self.assertTrue(
                any(x[1].center_s<self.s for x in near)
            )
            self.assertTrue(
                any(x[1].center_s>self.s for x in near)
            )

    def test_near_grid_uses_real_measured_coverage(self):
        # Enough 1-s MAG and ~32-s SWEA samples to cover every 10-min clip.
        t_mag=self.s+np.arange(-2100,2101,dtype=float)
        t_swea=self.s+np.arange(-2080,2081,32,dtype=float)
        blocks={self.time:{
            "MAG":(t_mag,np.zeros((len(t_mag),4))),
            "SWEA":(t_swea,np.zeros((len(t_swea),12)))
        }}
        event={"time":self.time,"bs_time":datetime.fromtimestamp(
            self.s-1100,tz=timezone.utc)}
        pool,skipped=gather_near_candidates(
            [event],blocks,
            np.asarray([self.s]),
            np.asarray([self.s-1100])
        )
        self.assertTrue(pool[self.time])
        for c,clip in pool[self.time]:
            self.assertIsNone(c.transition)
            self.assertEqual(c.kind,"mpb_ambiguous")
            self.assertEqual(clip.mag.shape[1],4)
            self.assertEqual(clip.swea.shape[1],12)

    def test_near_rank_never_uses_far_binary_negatives(self):
        entries=[
            (self.time,self.candidate(0.,1),None,1,(0,1)),
            (self.time,self.candidate(900.,0),None,0,(1,1)),
        ]
        logits=torch.tensor([1.,-2.],requires_grad=True)
        loss,win,count=_localization_rank(logits,entries)
        self.assertEqual(count,0)
        self.assertIsNone(win)
        self.assertEqual(float(loss),0.)
        loss.backward()
        self.assertTrue(torch.isfinite(logits.grad).all())


if __name__=="__main__":
    unittest.main()
