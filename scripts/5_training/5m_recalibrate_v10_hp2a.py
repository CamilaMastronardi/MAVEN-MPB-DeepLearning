#!/usr/bin/env python3
"""V10-HP2A: recalibrate completed HP1 checkpoints on inner MONITOR only.

No training, no BigQuery: measured monitor MAG/SWEA blocks must exist in cache.
Use HP1's saved OUTER predictions for evaluation without ever tuning on them.
"""
from __future__ import annotations

import argparse
import csv
import json
import sys
from importlib import import_module
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import torch

ROOT=Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0,str(ROOT))

v7=import_module("scripts.5_training.5b_train_mag_swea_regions")
v9=import_module("scripts.5_training.5e_train_mag_swea_temporal_v9")
v10=import_module("scripts.5_training.5f_train_mag_swea_refiner_v10")
previous=import_module("scripts.5_training.5a_train_mag_cv")
pilot=import_module("scripts.3_windowing.3b_bigquery_window_pilot")
bq=import_module("scripts.1_bigquery.1c_bigquery_event_pilot")

from src.detection.mpb_region_anchored_refiner import RegionAnchoredTemporalRefiner


def parse_args():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument("--manifest",type=Path,default=v7.MANIFEST)
    p.add_argument("--hp1-dir",type=Path,
        default=Path("results/metrics/mag_swea_refiner_v10_hp1"))
    p.add_argument("--pretrained-dir",type=Path,default=v10.PRETRAINED)
    p.add_argument("--output-dir",type=Path,
        default=Path("results/metrics/mag_swea_refiner_v10_hp2a"))
    p.add_argument("--mag-cache-dir",type=Path,default=v7.MAG_CACHE)
    p.add_argument("--swea-cache-dir",type=Path,default=v7.SWEA_CACHE)
    p.add_argument("--folds",choices=("completed","all","1","2","3","4"),
                   default="completed")
    p.add_argument("--seed",type=int,default=42)
    p.add_argument("--batch-size",type=int,default=16)
    p.add_argument("--torch-threads",type=int,default=2)
    p.add_argument("--max-mag-rows",type=int,default=25000)
    p.add_argument("--max-swea-rows",type=int,default=2000)
    p.add_argument("--alpha-step",type=float,default=0.1)
    a=p.parse_args()
    v10.calibration_alpha_grid(a.alpha_step)
    if min(a.batch_size,a.torch_threads,a.max_mag_rows,a.max_swea_rows)<=0:
        p.error("All worker/batch/source limits must be positive.")
    for protected in (a.hp1_dir,a.pretrained_dir,
                      Path("results/metrics/mag_swea_refiner_v10"),
                      Path("results/metrics/mag_swea_refiner_v10_hp2b")):
        if a.output_dir.resolve()==protected.resolve():
            p.error("HP2A output must not overwrite a training/source directory.")
    return a


def read_csv(path):
    with path.open(newline="",encoding="utf-8") as fp:
        return list(csv.DictReader(fp))


def completed_hp1_folds(root):
    result=[]
    for fold in (1,2,3,4):
        stem=f"fold_{fold:02d}"
        needed=(root/f"{stem}_metrics.json",root/f"{stem}_final.pt",
                root/f"{stem}_event_metrics.csv")
        if all(f.is_file() for f in needed):
            result.append(fold)
    return result


def load_cached_blocks(rows,args):
    """Offline cache path with V7b's provenance checks, NEVER query BigQuery."""
    sqls=v7.query_sources()
    caches={
        "MAG":v7.MagEventCache(args.mag_cache_dir,project=bq.PROJECT,
                               table=bq.MAG,sql=sqls["MAG"]),
        "SWEA":v7.SweaEventCache(args.swea_cache_dir,project=bq.PROJECT,
                                 table=bq.SWEA+" + "+bq.AXES,
                                 sql=sqls["SWEA"])
    }
    blocks={}
    for row in rows:
        start,end=v7.event_bounds(row)
        record={}
        for sensor in ("MAG","SWEA"):
            cap=(args.max_mag_rows if sensor=="MAG" else args.max_swea_rows)
            hit=caches[sensor].load(row["time"],start,end,max_rows=cap)
            if hit is None or not len(hit[0]):
                raise RuntimeError(
                    f"Missing cached {sensor} for {row['time'].isoformat()}. "
                    "HP2A will not query BigQuery."
                )
            record[sensor]=hit
        blocks[row["time"]]=record
    print(f"HP2A: loaded {len(blocks)} cached monitor events, no BigQuery.",
          flush=True)
    return blocks


def hp1_signature_args(manifest,seed):
    """Default HP1 runner values; refuses to silently reinterpret variants."""
    return SimpleNamespace(
        manifest=manifest,max_epochs=150,min_epochs=20,patience=20,
        batch_size=16,learning_rate=0.0001,seed=seed,
        localization_weight=2.0,offset_weight=1.0,
        gaussian_sigma_s=32.0,train_positives_per_event=4,
        train_negatives_per_event=3,evaluation_stride_s=32.0,
        calibration_goal="clipped_mae",alpha_step=0.25
    )


def from_hp1_outer_rows(rows):
    """Transform saved OUTER HP1 predictions to V10 evaluation records."""
    records=[]
    for row in rows:
        records.append({
            "mpb_timestamp_utc":row["mpb_timestamp_utc"],
            "source_group":row["source_group"],
            "coarse_offset_s":float(row["coarse_peak_offset_s"]),
            "refined_mean_offset_s":float(row["refiner_mean_offset_s"]),
            "refined_peak_offset_s":float(row["refiner_peak_offset_s"]),
            "refiner_confidence":float(row["refiner_confidence"]),
            "n_scanned_windows":int(row["n_windows"])
        })
    return records


def paired_summary(new,old):
    new=np.asarray(new,dtype=float)
    old=np.asarray(old,dtype=float)
    if not len(new) or len(new)!=len(old):
        raise RuntimeError("Missing or unpaired HP1 predictions.")
    return {
        "n_paired_events":int(len(new)),
        "hp2a_within_60s":float(np.mean(new<=60)),
        "hp1_within_60s":float(np.mean(old<=60)),
        "delta_within_60_percentage_points":
            float(100*(np.mean(new<=60)-np.mean(old<=60))),
        "hp2a_median_error_s":float(np.median(new)),
        "hp1_median_error_s":float(np.median(old)),
        "improved_over_hp1":int(np.sum(new<old-1e-6)),
        "worsened_over_hp1":int(np.sum(new>old+1e-6)),
        "same_as_hp1":int(np.sum(np.abs(new-old)<=1e-6))
    }


def run_fold(fold,rows,args,device):
    parts={}
    for f in (1,2,3,4):
        parts[f]=v7.chosen_events(rows,f,pilot_count=0)
    train=sorted(
        [row for f in (1,2,3,4) if f!=fold for row in parts[f]],
        key=lambda r:r["time"])
    val=parts[fold]
    previous.check_disjoint_supports(train,val)
    fit,monitor,_=previous.split_inner_monitor_months(train,fold,args.seed)
    if not monitor or not val:
        raise RuntimeError("Missing monitor or outer events.")
    stem=f"fold_{fold:02d}"
    hp1_metrics=json.loads(
        (args.hp1_dir/f"{stem}_metrics.json").read_text(encoding="utf-8"))
    final=torch.load(args.hp1_dir/f"{stem}_final.pt",
                     map_location="cpu",weights_only=False)
    src_path=args.pretrained_dir/f"{stem}_final.pt"
    source=torch.load(src_path,map_location="cpu",weights_only=False)
    if (source.get("fold")!=fold or final.get("fold")!=fold or
            hp1_metrics.get("fold")!=fold):
        raise RuntimeError(f"FOLD {fold}: checkpoint fold mismatch.")

    # Ensure same month split, teacher, sample settings and architecture
    # as default HP1; do NOT silently recalibrate different experiments.
    expected_sig=v10.signature(
        hp1_signature_args(args.manifest,args.seed),
        fold,src_path,fit,monitor,val)
    if (final["signature"]!=expected_sig or
            hp1_metrics["signature"]!=expected_sig):
        raise RuntimeError(
            f"FOLD {fold}: HP1 signature does not match original HP1 "
            "defaults and manifest. Refuse to recalibrate a different run."
        )
    model=RegionAnchoredTemporalRefiner(source["model"]).to(device)
    model.load_state_dict(final["model"],strict=True)
    model.eval()
    normalizers=v9.normalizers_from_v7b(source)
    blocks=load_cached_blocks(monitor,args)
    mpb=np.asarray([r["time"].timestamp() for r in rows],dtype=np.float64)
    bs=np.asarray([r["bs_time"].timestamp() for r in rows],dtype=np.float64)
    coarse=v10.teacher_coarse_proposals(
        model,monitor,blocks,normalizers,mpb,bs,
        SimpleNamespace(batch_size=args.batch_size,evaluation_stride_s=32.),
        device)
    refined=v10.refine_proposals(
        model,coarse,normalizers,SimpleNamespace(batch_size=args.batch_size),
        device)
    del blocks,coarse

    original=v10.calibrate_refinement(refined)
    original_hp1=hp1_metrics["monitor_calibration"]
    if (original["alpha"]!=original_hp1["alpha"] or
            original["method"]!=original_hp1["method"]):
        raise RuntimeError(
            f"FOLD {fold}: could not reproduce original HP1 monitor "
            "calibration. Refuse to compare inconsistent predictions."
        )
    # Include HP1's selected alpha as well as 0.1 grid for fair
    # calibration comparison; the original might have alpha=0.75.
    alphas=tuple(sorted(set(
        v10.calibration_alpha_grid(args.alpha_step) +
        (float(original["alpha"]),))))
    calibration=v10.calibrate_refinement(
        refined,goal="within_60",alphas=alphas)

    hp1_outer=read_csv(args.hp1_dir/f"{stem}_event_metrics.csv")
    if len(hp1_outer)!=len(val):
        raise RuntimeError(f"FOLD {fold}: incomplete outer HP1 predictions.")
    outer=from_hp1_outer_rows(hp1_outer)
    if {r["mpb_timestamp_utc"] for r in outer} != {
            row["time"].isoformat() for row in val}:
        raise RuntimeError(f"FOLD {fold}: outer timestamps changed.")
    metrics,out=v10.evaluate_fold(
        outer,calibration,
        args.pretrained_dir/f"{stem}_event_metrics.csv")
    original_by_time={r["mpb_timestamp_utc"]:r for r in hp1_outer}
    for row in out:
        old=original_by_time[row["mpb_timestamp_utc"]]
        old_error=float(old["absolute_error_s"])
        row["hp1_absolute_error_s"]=old_error
        row["delta_hp2a_minus_hp1_s"]=row["absolute_error_s"]-old_error
    paired=paired_summary(
        [r["absolute_error_s"] for r in out],
        [r["hp1_absolute_error_s"] for r in out])
    metrics.update({
        "fold":fold,"experiment":"v10_hp2a_monitor_recalibration",
        "source_hp1_signature":expected_sig,
        "source_hp1_calibration":original_hp1,
        "monitor_calibration":calibration,
        "calibration_on":"inner_monitor_only",
        "outer_used_for_selection":False,
        "new_training":False,
        "comparison_with_hp1":paired,
    })
    v7.safe_json(args.output_dir/f"{stem}_metrics.json",metrics)
    v7.write_csv(args.output_dir/f"{stem}_event_metrics.csv",out)
    print(
        f"HP2A FOLD {fold}: <=60s HP1={paired['hp1_within_60s']:.1%} "
        f"HP2A={paired['hp2a_within_60s']:.1%}; "
        f"alpha {original['alpha']:.2f}->{calibration['alpha']:.2f}; "
        f"median={paired['hp2a_median_error_s']:.1f}s.",
        flush=True
    )
    return metrics


def main():
    a=parse_args()
    torch.set_num_threads(a.torch_threads)
    available=completed_hp1_folds(a.hp1_dir)
    if a.folds=="completed":
        folds=available
    elif a.folds=="all":
        folds=[1,2,3,4]
    else:
        folds=[int(a.folds)]
    if not folds:
        raise RuntimeError("No completed HP1 fold to recalibrate.")
    if any(f not in available for f in folds):
        raise RuntimeError(
            f"HP1 folds not complete: {sorted(set(folds)-set(available))}")
    rows=pilot.read_manifest(a.manifest)
    for row in rows:
        if not row.get("bs_timestamp_utc"):
            raise RuntimeError("No paired BS time in manifest.")
        row["bs_time"]=bq.to_utc(row["bs_timestamp_utc"])
    a.output_dir.mkdir(parents=True,exist_ok=True)
    reports={}
    for fold in folds:
        reports[fold]=run_fold(fold,rows,a,torch.device("cpu"))
    summary=v10.summarize(
        SimpleNamespace(output_dir=a.output_dir,pilot_events_per_fold=0),
        reports)
    summary["experiment"]="v10_hp2a_monitor_recalibration"
    summary["new_training"]=False
    summary["calibration_on"]="inner_monitor_only"
    new,old=[],[]
    for fold in folds:
        for row in read_csv(a.output_dir/f"fold_{fold:02d}_event_metrics.csv"):
            new.append(float(row["absolute_error_s"]))
            old.append(float(row["hp1_absolute_error_s"]))
    summary["comparison_with_hp1"]=paired_summary(new,old)
    v7.safe_json(a.output_dir/"cv_summary.json",summary)
    print("HP2A COMPLETE:",json.dumps(summary["comparison_with_hp1"],
          indent=2),flush=True)


if __name__=="__main__":
    main()
