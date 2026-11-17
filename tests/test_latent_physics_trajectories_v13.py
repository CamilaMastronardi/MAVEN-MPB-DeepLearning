"""Tests for the V13 MAG/SWEA physical interpretation (no live data)."""
from __future__ import annotations

import csv
import tempfile
import unittest
from datetime import datetime,timezone
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
            and r["latent_axis"]=="sweaz1"
            and r["n"]>=12
            for r in pairs
        ))
        # Both instruments must have real, finite correlations, not
        # silent all-NaN outputs from a mistyped latent-column name.
        for axis,feature in (
            ("magz1","mag_rotation_deg"),
            ("sweaz1","swea_abs_spectral_delta_logflux")
        ):
            found=[
                r for r in pairs
                if r["latent_axis"]==axis and
                r["feature"]==feature and r["split"]=="fit"
            ]
            self.assertEqual(len(found),1)
            self.assertTrue(np.isfinite(found[0]["rho"]))
            self.assertTrue(np.isfinite(found[0]["q_fit_bh"]))

    def test_group_feature_boxplots_work_with_current_matplotlib(self):
        # Reproduce real Matplotlib API behavior without relying on
        # the removed boxplot(labels=...) argument.
        with tempfile.TemporaryDirectory() as tmp:
            output=Path(tmp)
            rows=[]
            for i in range(4):
                rows.append({
                    "split":"fit",
                    "mag_cluster":1 if i<2 else 2,
                    "swea_cluster":1 if i%2==0 else 2,
                    "mag_abs_delta_b_nt":float(i+1),
                    "mag_rotation_deg":float(20+i),
                    "mag_post_std_b_nt":float(i+2),
                    "mag_post_rms_db_nt":float(i+3),
                    "swea_abs_spectral_delta_logflux":float(i+.1),
                    "swea_delta_hardness_logdiff":float(i+.2),
                    "swea_post_temporal_std_logflux":float(i+.3),
                    "swea_delta_mean_logflux":float(i+.4),
                })
            script._plot_group_features(rows,output)
            self.assertTrue(
                (output/"mag_physical_group_boxplots.png").is_file()
            )
            self.assertTrue(
                (output/"swea_physical_group_boxplots.png").is_file()
            )

    def test_example_figure_uses_iso_timestamp_not_manifest_time(self):
        # The actual row comes from physical_features.csv and contains
        # mpb_timestamp_utc, not datetime-valued row["time"].
        t=datetime(2020,1,1,tzinfo=timezone.utc)
        center=t.timestamp()
        offsets=np.arange(-300.,301.,1.)
        mag_t=center+offsets
        b=np.zeros((len(offsets),4),dtype=np.float32)
        b[:,0]=10.+np.sin(offsets/30.)
        b[:,3]=np.abs(b[:,0])
        swea_offsets=np.arange(-288.,289.,32.)
        swea_t=center+swea_offsets
        swea_flux=np.ones((len(swea_t),12),dtype=np.float32)
        example={
            "mpb_timestamp_utc":t.isoformat(),
            "mag_cluster":1,
            "swea_cluster":2,
        }
        path=[
            {
                "center_offset_s":float(shift),
                "z1":float(shift/240.),
                "z2":float(shift/480.),
                "z3":0.,
            }
            for shift in (-120.,0.,120.)
        ]
        with tempfile.TemporaryDirectory() as temp:
            folder=Path(temp)
            script._plot_example(
                example,
                {"MAG":(mag_t,b),"SWEA":(swea_t,swea_flux)},
                {"MAG":path,"SWEA":path},
                folder
            )
            figures=list(folder.glob("*.png"))
            self.assertEqual(len(figures),1)
            self.assertGreater(figures[0].stat().st_size,1000)

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
