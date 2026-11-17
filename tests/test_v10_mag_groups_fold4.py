"""Synthetic tests for V10-MAG experts trained on folds 1-3, outer fold 4."""
from __future__ import annotations

import csv
import tempfile
import unittest
from datetime import datetime,timedelta,timezone
from importlib import import_module
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import patch

import numpy as np

grouped=import_module(
    "scripts.5_training.5k_train_v10_mag_groups_fold4"
)


def event(i,fold=4):
    return {
        "time":datetime(2020,1,1,tzinfo=timezone.utc)+
            timedelta(days=i),
        "fold":fold,
        "source_group":"Group1"
    }


class MAGExpertFold4Tests(unittest.TestCase):
    def test_group_csv_exact_matching_and_holdout_partition(self):
        with tempfile.TemporaryDirectory() as tmp:
            csv_path=Path(tmp)/"groups.csv"
            events=[
                event(0),event(1),event(2,fold=1)
            ]
            with csv_path.open("w",newline="",encoding="utf-8") as fp:
                writer=csv.DictWriter(fp,fieldnames=[
                    "mpb_timestamp_utc","fold","split","cluster"
                ])
                writer.writeheader()
                for ev,cluster in zip(events,(1,2,"")):
                    writer.writerow({
                        "mpb_timestamp_utc":ev["time"].isoformat(),
                        "fold":ev["fold"],
                        "split":(
                            "outer_holdout" if ev["fold"]==4 else "fit"
                        ),
                        "cluster":cluster
                    })
            labels=grouped.read_mag_groups(csv_path)
            subsets,missing=grouped.grouped_events(
                events,labels
            )
            self.assertEqual(
                [len(subsets[k]) for k in grouped.GROUPS],
                [1,1]
            )
            self.assertEqual(len(missing),1)
            self.assertEqual(
                subsets[1][0]["time"],events[0]["time"]
            )
            bad=event(4)
            with self.assertRaisesRegex(RuntimeError,"Missing exact"):
                grouped.grouped_events([bad],labels)
            corrupt=[dict(events[0],fold=1)]
            with self.assertRaisesRegex(RuntimeError,"fold mismatch"):
                grouped.grouped_events(corrupt,labels)

    def test_v13_fold4_must_be_outer_holdout(self):
        row=event(0)
        labels={
            row["time"].isoformat():{
                "group":1,"fold":4,"split":"fit"
            }
        }
        with self.assertRaisesRegex(RuntimeError,"holdout"):
            grouped.grouped_events([row],labels)

    def test_inner_monitor_is_group_specific_and_separate(self):
        # Mock existing month-based splitter. The selected monitor is
        # guaranteed not to contain events from the OUTER fold.
        sample=[event(i,fold=1) for i in range(30)]
        def split(rows,fold,seed):
            return rows[5:],rows[:5],0
        with patch.object(
            grouped.previous,"split_inner_monitor_months",
            side_effect=split
        ),patch.object(
            grouped.previous,"check_disjoint_supports",
            return_value=None
        ):
            fit,monitor,removed,seed=grouped.inner_group_split(
                sample,1,42
            )
        self.assertEqual(len(fit),25)
        self.assertEqual(len(monitor),5)
        self.assertEqual(removed,0)
        self.assertTrue(all(r["fold"]!=4 for r in fit+monitor))

    def test_paired_comparison_uses_identical_events(self):
        with tempfile.TemporaryDirectory() as tmp:
            path=Path(tmp)/"v10.csv"
            key=event(0)["time"].isoformat()
            with path.open("w",newline="",encoding="utf-8") as fp:
                writer=csv.DictWriter(fp,fieldnames=[
                    "mpb_timestamp_utc","absolute_error_s"
                ])
                writer.writeheader()
                writer.writerow({
                    "mpb_timestamp_utc":key,
                    "absolute_error_s":90.
                })
            metrics,rows=grouped.paired_with_v10(
                [{
                    "mpb_timestamp_utc":key,
                    "absolute_error_s":45.,
                    "selected_offset_s":45.,
                }],path,1
            )
            self.assertEqual(metrics["n_events"],1)
            self.assertEqual(metrics["improved_vs_v10"],1)
            self.assertEqual(metrics["within_60s"],1.)
            self.assertEqual(metrics["baseline_v10_within_60s"],0.)
            self.assertEqual(rows[0]["mag_cluster"],1)

    def test_fold4_summary_requires_complete_disjoint_groups(self):
        with tempfile.TemporaryDirectory() as tmp:
            args=SimpleNamespace(
                output_dir=Path(tmp),groups="all"
            )
            for group,offset in ((1,40.),(2,80.)):
                path=grouped.stem(args,group)
                csv_file=path.with_name(
                    path.name+"_event_metrics.csv"
                )
                with csv_file.open(
                    "w",newline="",encoding="utf-8"
                ) as fp:
                    writer=csv.DictWriter(fp,fieldnames=[
                        "mpb_timestamp_utc","absolute_error_s",
                        "v10_absolute_error_s"
                    ])
                    writer.writeheader()
                    writer.writerow({
                        "mpb_timestamp_utc":
                            event(group)["time"].isoformat(),
                        "absolute_error_s":offset,
                        "v10_absolute_error_s":60.,
                    })
            result=grouped.summarize(
                args,{
                    1:{"n_events":1,"median_error_s":40.,
                       "within_60s":1.,"best_epoch":10,
                       "baseline_v10_within_60s":1.,
                       "monitor_calibration":{"alpha":1.}},
                    2:{"n_events":1,"median_error_s":80.,
                       "within_60s":0.,"best_epoch":10,
                       "baseline_v10_within_60s":1.,
                       "monitor_calibration":{"alpha":.5}},
                },[event(1),event(2)]
            )
            self.assertEqual(result["n_events"],2)
            self.assertEqual(result["n_within_60s"],1)
            self.assertAlmostEqual(
                result["fraction_within_60s"],.5
            )
            with self.assertRaisesRegex(RuntimeError,"incomplete"):
                grouped.summarize(
                    args,{1:{
                        "n_events":1,"median_error_s":40.,
                        "within_60s":1.,"best_epoch":10,
                        "baseline_v10_within_60s":1.,
                        "monitor_calibration":{"alpha":1.}
                    }},[event(1),event(2)]
                )


if __name__=="__main__":
    unittest.main()
