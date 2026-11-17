"""Tests for the V13 MAG/SWEA physical interpretation (no live data)."""
from __future__ import annotations

import csv
import tempfile
import unittest
from importlib import import_module
from pathlib import Path

import numpy as np

from src.data.swea_event_cache import ENERGY_EV
from src.analysis.mpb_latent_physics_v13 import (
    flank_masks,mag_physical_features,swea_physical_features,
    spearman_pair,benjamini_hochberg,trajectory_statistics
)

script=import_module(
    "scripts.6_latent_analysis.6a_latent_physics_trajectories_v13"
)


class MAGPhysicsTests(unittest.TestCase):
    def test_measured_flanks_and_vector_rotation(self):
        seconds=np.arange(-320,321,dtype=float)
        vectors=np.zeros((len(seconds),4),dtype=float)
        vectors[:,0]=10.
        vectors[seconds>=60,0]=0.
        vectors[seconds>=60,1]=20.
        vectors[:,3]=np.linalg.norm(vectors[:,:3],axis=1)
        result=mag_physical_features(
            seconds,vectors,0.
        )
        self.assertAlmostEqual(result["mag_pre_mean_b_nt"],10.)
        self.assertAlmostEqual(result["mag_post_mean_b_nt"],20.)
        self.assertAlmostEqual(result["mag_delta_b_nt"],10.)
        self.assertAlmostEqual(result["mag_rotation_deg"],90.)
        self.assertEqual(result["mag_n_pre"],181)
        self.assertEqual(result["mag_n_post"],181)
        self.assertAlmostEqual(
            result["mag_total_b_consistency_max_nt"],0.
        )

    def test_gap_or_missing_flank_marks_physics_unavailable(self):
        seconds=np.arange(-320,-110,dtype=float)
        values=np.zeros((len(seconds),4),dtype=float)
        values[:,0]=5.;values[:,3]=5.
        result=mag_physical_features(seconds,values,0.)
        self.assertEqual(result["mag_n_post"],0)
        self.assertTrue(np.isnan(result["mag_rotation_deg"]))


class SWEAPhysicsTests(unittest.TestCase):
    def test_logflux_pre_post_are_native_and_energy_slope_valid(self):
        seconds=np.arange(-320,321,32,dtype=float)
        flux=np.ones((len(seconds),len(ENERGY_EV)),dtype=float)*2.
        flux[seconds>=60]=5.+np.log(ENERGY_EV)[None,:]
        result=swea_physical_features(
            seconds,flux,0.
        )
        self.assertGreaterEqual(result["swea_n_pre"],3)
        self.assertGreaterEqual(result["swea_n_post"],3)
        self.assertGreater(
            result["swea_delta_mean_logflux"],0.
        )
        self.assertGreater(
            result["swea_post_slope_logflux_loge"],.9
        )
        self.assertAlmostEqual(
            result["swea_pre_slope_logflux_loge"],0.,places=9
        )

    def test_swea_no_post_frames_gives_missing_not_fake_zeros(self):
        seconds=np.arange(-320,-80,32,dtype=float)
        flux=np.ones((len(seconds),12),dtype=float)
        result=swea_physical_features(seconds,flux,0.)
        self.assertTrue(np.isnan(
            result["swea_delta_mean_logflux"]
        ))


class StatisticsTests(unittest.TestCase):
    def test_spearman_ignores_missing_paired_values(self):
        x=np.arange(16,dtype=float)
        y=x*2.
        y[1]=np.nan
        result=spearman_pair(x,y)
        self.assertEqual(result["n"],15)
        self.assertAlmostEqual(result["rho"],1.)

    def test_fdr_bh_monotone_and_nan_safe(self):
        q=benjamini_hochberg([.01,.03,.03,np.nan])
        np.testing.assert_allclose(
            q[:3],[.03,.03,.03],atol=1e-12
        )
        self.assertTrue(np.isnan(q[3]))

    def test_latent_speed_skips_unobserved_gap(self):
        times=np.array([-60.,-30.,0.,90.,120.])
        path=np.zeros((5,3))
        path[:,0]=[0.,1.,2.,500.,501.]
        metrics=trajectory_statistics(times,path)
        self.assertEqual(metrics["n_adjacent_segments"],3)
        self.assertAlmostEqual(metrics["path_length"],3.)
        self.assertAlmostEqual(metrics["max_speed_per_s"],1./30.)

    def test_3d_features_not_affected_by_missing_first_csv_row(self):
        rows=[
            {"split":"fit","magz1":1.,
             "magz2":2.,"magz3":3.,
             "sweaz1":4.,"sweaz2":5.,"sweaz3":6.},
        ]
        for i in range(20):
            rows.append({
                "split":"fit",
                "magz1":float(i),"magz2":float(i*2),
                "magz3":float(-i),
                "sweaz1":float(i),"sweaz2":float(i*2),
                "sweaz3":float(-i),
                "mag_rotation_deg":float(i),
                "swea_abs_spectral_delta_logflux":float(i),
            })
        pairs=script.paired_correlations(rows)
        self.assertTrue(any(
            r["feature"]=="mag_rotation_deg" and
            r["latent_axis"]=="magz1" and r["n"]>=12
            for r in pairs
        ))
        self.assertTrue(any(
            r["feature"]=="swea_abs_spectral_delta_logflux"
            for r in pairs
        ))

    def test_original_embedding_matches_nine_shifted_views(self):
        track=[
            {
                "center_offset_s":float(d),
                "z1":float(i),"z2":float(i+1),"z3":float(i+2)
            }
            for i,d in enumerate(range(-240,241,30))
        ]
        subs=np.array([
            [r["z1"],r["z2"],r["z3"]]
            for r in track
            if int(r["center_offset_s"])%60==0
        ])
        avg=subs.mean(0)
        record={f"z{i+1}":str(avg[i]) for i in range(3)}
        error,n=script.original_embedding_differences(track,record)
        self.assertEqual(n,9)
        self.assertAlmostEqual(error,0.)


if __name__=="__main__":
    unittest.main()
