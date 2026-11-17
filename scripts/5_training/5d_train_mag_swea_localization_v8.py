#!/usr/bin/env python3
"""V8: improve localization, GIVEN a known MPB inside the event neighborhood.

Warm-start four independent V7b fold checkpoints; preserve exact TRAIN-only
normalizers. Train near-boundary temporal ordering without labeling adjacent
ambiguous plasma as negatives. Evaluate the maximum in +/-30 minutes, at
the same 32-s grid as V7b, and compare on identical held-out events.

This is NOT MPB-presence detection; every test neighborhood has a catalog MPB.
"""
from __future__ import annotations

import argparse
import csv
import hashlib
import json
import random
import sys
from collections import defaultdict
from importlib import import_module
from pathlib import Path

import numpy as np
import torch

ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0,str(ROOT))

v7 = import_module("scripts.5_training.5b_train_mag_swea_regions")
pilot = import_module("scripts.3_windowing.3b_bigquery_window_pilot")
previous = import_module("scripts.5_training.5a_train_mag_cv")
bq = import_module("scripts.1_bigquery.1c_bigquery_event_pilot")
from src.detection.mag_swea_region_model import MagSweaRegionAutoencoder
from src.detection.mpb_localization_refinement import (
    gather_near_candidates, monitor_near_ranking,
    train_localization_epoch
)

DEFAULT_PRETRAINED=Path("results/metrics/mag_swea_regions_ranking_v1")
DEFAULT_OUTPUT=Path("results/metrics/mag_swea_localization_v8")


def parse_args():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument("--manifest",type=Path,default=v7.MANIFEST)
    p.add_argument("--pretrained-dir",type=Path,default=DEFAULT_PRETRAINED)
    p.add_argument("--output-dir",type=Path,default=DEFAULT_OUTPUT)
    p.add_argument("--mag-cache-dir",type=Path,default=v7.MAG_CACHE)
    p.add_argument("--swea-cache-dir",type=Path,default=v7.SWEA_CACHE)
    p.add_argument("--refresh-swea-cache",action="store_true")
    p.add_argument("--folds",default="all",choices=("all","1","2","3","4"))
    p.add_argument("--pilot-events-per-fold",type=int,default=0)
    p.add_argument("--max-epochs",type=int,default=28)
    p.add_argument("--min-epochs",type=int,default=8)
    p.add_argument("--patience",type=int,default=6)
    p.add_argument("--batch-size",type=int,default=16)
    p.add_argument("--torch-threads",type=int,default=2)
    p.add_argument("--seed",type=int,default=42)
    p.add_argument("--learning-rate",type=float,default=0.00008)
    p.add_argument("--region-weight",type=float,default=0.5)
    p.add_argument("--reconstruction-weight",type=float,default=0.15)
    p.add_argument("--rank-weight",type=float,default=1.0)
    p.add_argument("--rank-margin",type=float,default=0.5)
    p.add_argument("--rank-control-step-s",type=float,default=120.)
    p.add_argument("--rank-positives-per-event",type=int,default=2)
    p.add_argument("--rank-hard-negatives",type=int,default=2)
    p.add_argument("--rank-random-negatives",type=int,default=2)
    p.add_argument("--near-weight",type=float,default=0.7)
    p.add_argument("--near-margin",type=float,default=0.1)
    p.add_argument("--evaluation-stride-s",type=float,default=32.)
    p.add_argument("--max-query-gib",type=float,default=2.)
    p.add_argument("--max-total-gib",type=float,default=150.)
    p.add_argument("--max-mag-rows",type=int,default=25000)
    p.add_argument("--max-swea-rows",type=int,default=2000)
    p.add_argument("--resume",action="store_true")
    p.add_argument("--execute",action="store_true")
    a=p.parse_args()
    positive=(
        a.max_epochs,a.min_epochs,a.patience,a.batch_size,a.torch_threads,
        a.learning_rate,a.rank_weight,a.rank_control_step_s,
        a.rank_positives_per_event,a.rank_hard_negatives,
        a.rank_random_negatives,a.near_weight,a.evaluation_stride_s,
        a.max_query_gib,a.max_total_gib,a.max_mag_rows,a.max_swea_rows
    )
    if (min(positive)<=0 or a.min_epochs>a.max_epochs
            or a.rank_control_step_s>600
            or a.pilot_events_per_fold<0 or a.rank_margin<0
            or a.near_margin<0 or a.region_weight<0
            or a.reconstruction_weight<0
            or a.batch_size<(
                a.rank_positives_per_event+
                a.rank_hard_negatives+
                a.rank_random_negatives+2)):
        p.error("Invalid V8 localization configuration or batch too small.")
    return a


def fold_signature(args, fold, source, fit, monitor, val):
    config={
        "model":"v8-near-ordinal-refinement-v1",
        "fold":fold,"manifest_sha":v7.sha256_file(args.manifest),
        "source_sha":v7.sha256_file(source),
        "fit":[r["time"].isoformat() for r in fit],
        "monitor":[r["time"].isoformat() for r in monitor],
        "val":[r["time"].isoformat() for r in val],
        "epoch_limits":[args.max_epochs,args.min_epochs,args.patience],
        "batch_size":args.batch_size,"seed":args.seed,"lr":args.learning_rate,
        "region_weight":args.region_weight,
        "reconstruction_weight":args.reconstruction_weight,
        "rank_weight":args.rank_weight,"rank_margin":args.rank_margin,
        "rank_step":args.rank_control_step_s,
        "rank_counts":[args.rank_positives_per_event,
                       args.rank_hard_negatives,args.rank_random_negatives],
        "near_weight":args.near_weight,"near_margin":args.near_margin,
        "evaluation_stride":args.evaluation_stride_s
    }
    return hashlib.sha256(
        json.dumps(config,sort_keys=True).encode()
    ).hexdigest()


def normalizers_from_v7b(state):
    norms=state.get("normalizers")
    if not isinstance(norms,dict):
        raise RuntimeError("V7b checkpoint lacks fold-specific normalizers.")
    return {
        sensor:{
            "mean":np.asarray(norms[sensor]["mean"],dtype=np.float64),
            "scale":np.asarray(norms[sensor]["scale"],dtype=np.float64),
            "n_epochs":norms[sensor]["n_epochs"]
        } for sensor in ("MAG","SWEA")
    }


def with_baseline(rows, source_path):
    """Annotate identical outer events with V7b errors; no model selection."""
    with source_path.open(newline="",encoding="utf-8") as f:
        base={r["mpb_timestamp_utc"]:r for r in csv.DictReader(f)}
    result=[]
    for record in rows:
        record=dict(record)
        key=record["mpb_timestamp_utc"]
        baseline=base.get(key)
        if baseline is None or baseline["absolute_error_s"]=="":
            record.update({
                "v7b_absolute_error_s":"",
                "delta_v8_minus_v7b_s":"",
                "localization_better_than_v7b":""
            })
        else:
            old=float(baseline["absolute_error_s"])
            new=record["absolute_error_s"]
            if new=="":
                raise RuntimeError("V8 validation event has no usable clips.")
            delta=float(new)-old
            record.update({
                "v7b_absolute_error_s":old,
                "delta_v8_minus_v7b_s":delta,
                "localization_better_than_v7b":int(delta<0)
            })
        result.append(record)
    return result


def train_fold(fold, groups, blocks, all_rows, args, device):
    fit,monitor,removed=previous.split_inner_monitor_months(
        groups["train"],fold,args.seed
    )
    val=groups["val"]
    source=args.pretrained_dir/f"fold_{fold:02d}_final.pt"
    state=torch.load(source,map_location="cpu",weights_only=False)
    if (state.get("fold")!=fold or "model" not in state
            or "normalizers" not in state):
        raise RuntimeError("Wrong or incomplete V7b pretrained fold.")
    normalizers=normalizers_from_v7b(state)
    signature=fold_signature(args,fold,source,fit,monitor,val)

    mpb_s=np.asarray([r["time"].timestamp() for r in all_rows],
                     dtype=np.float64)
    bs_s=np.asarray([r["bs_time"].timestamp() for r in all_rows],
                    dtype=np.float64)
    known_fit, skip_fit=v7.gather_candidates(
        fit,blocks,mpb_s,bs_s,ranking=True,
        rank_control_step_s=args.rank_control_step_s
    )
    known_mon, skip_mon=v7.gather_candidates(
        monitor,blocks,mpb_s,bs_s,ranking=True,
        rank_control_step_s=args.rank_control_step_s
    )
    near_fit,near_fit_skip=gather_near_candidates(
        fit,blocks,mpb_s,bs_s
    )
    near_mon,near_mon_skip=gather_near_candidates(
        monitor,blocks,mpb_s,bs_s
    )
    print(
        f"FOLD {fold}: TRAIN={len(fit)} MONITOR={len(monitor)} "
        f"TEST={len(val)}; known pools={len(known_fit)}/"
        f"{len(known_mon)}; near centers="
        f"{sum(map(len,near_fit.values()))}/"
        f"{sum(map(len,near_mon.values()))}; "
        f"excluded overlapping train={removed}.",
        flush=True
    )
    if not known_fit or not known_mon:
        raise RuntimeError("No safe known-positive/far-control pool.")

    torch.manual_seed(args.seed+fold)
    model=MagSweaRegionAutoencoder().to(device)
    model.load_state_dict(state["model"])
    optimizer=torch.optim.AdamW(
        model.parameters(),lr=args.learning_rate,weight_decay=1e-3
    )
    folder=args.output_dir
    best_path=folder/f"fold_{fold:02d}_best.pt"
    last_path=folder/f"fold_{fold:02d}_last.pt"
    final_path=folder/f"fold_{fold:02d}_final.pt"
    best_epoch=0
    best_loss=float("inf")
    stale=0
    start_epoch=0
    history=[]
    hard_memory={}
    if last_path.exists():
        if not args.resume:
            raise RuntimeError(
                f"{last_path} exists; enable --resume or change output."
            )
        resumed=torch.load(
            last_path,map_location=device,weights_only=False
        )
        if resumed["signature"]!=signature:
            raise RuntimeError("V8 resume source or configuration mismatch.")
        model.load_state_dict(resumed["model"])
        optimizer.load_state_dict(resumed["optimizer"])
        best_epoch=resumed["best_epoch"]
        best_loss=resumed["best_loss"]
        stale=resumed["stale"]
        start_epoch=resumed["epoch"]
        history=resumed["history"]
        hard_memory=resumed["hard_memory"]
        print(f"FOLD {fold}: resumed at epoch {start_epoch}.",flush=True)

    for epoch in range(start_epoch+1,args.max_epochs+1):
        if epoch>=args.min_epochs and stale>=args.patience:
            break
        tr=train_localization_epoch(
            model,optimizer,known_fit,near_fit,normalizers,args,
            device,epoch=epoch,fold=fold,hard_memory=hard_memory
        )
        mon,_,_=v7.monitor_epoch(
            model,known_mon,normalizers,args,device
        )
        near_mon_metrics=monitor_near_ranking(
            model,known_mon,near_mon,normalizers,args,device
        )
        select_loss=(
            mon["select_loss"]+
            args.near_weight*near_mon_metrics["near_rank"]
        )
        if epoch>=args.min_epochs:
            if select_loss<best_loss-1e-4:
                best_loss,best_epoch,stale=select_loss,epoch,0
                v7.safe_checkpoint(best_path,{
                    "signature":signature,"epoch":epoch,
                    "model":model.state_dict()
                })
            else:
                stale+=1
        history.append({
            "epoch":epoch,
            "train_det":tr["det"],
            "train_region":tr["region"],
            "train_far_rank":tr["far_rank"],
            "train_near_rank":tr["near_rank"],
            "train_near_top1":tr["near_top1"],
            "monitor_det":mon["det"],
            "monitor_region":mon["region"],
            "monitor_far_rank":mon["rank"],
            "monitor_near_rank":near_mon_metrics["near_rank"],
            "monitor_near_top1":near_mon_metrics["near_top1"],
            "monitor_near_events":near_mon_metrics["near_events"],
            "monitor_select":select_loss,
            "best_epoch":best_epoch,"stale":stale
        })
        v7.safe_checkpoint(last_path,{
            "signature":signature,"epoch":epoch,
            "best_epoch":best_epoch,"best_loss":best_loss,
            "stale":stale,"history":history,
            "hard_memory":hard_memory,
            "model":model.state_dict(),"optimizer":optimizer.state_dict()
        })
        print(
            f"FOLD {fold} EPOCH {epoch}: "
            f"train_det={tr['det']:.4f} "
            f"train_far={tr['far_rank']:.4f} "
            f"train_near={tr['near_rank']:.4f} "
            f"val_det={mon['det']:.4f} "
            f"val_near={near_mon_metrics['near_rank']:.4f} "
            f"val_near_top1={near_mon_metrics['near_top1']:.3f} "
            f"val_select={select_loss:.4f} "
            f"best={best_epoch} stale={stale}",
            flush=True
        )
        if epoch>=args.min_epochs and stale>=args.patience:
            print(
                f"FOLD {fold}: early stop at {epoch}, best {best_epoch}.",
                flush=True
            )
            break

    if not best_path.exists():
        raise RuntimeError(f"FOLD {fold}: no V8 best checkpoint.")
    best=torch.load(best_path,map_location=device,weights_only=False)
    if best["signature"]!=signature:
        raise RuntimeError("Selected V8 checkpoint signature mismatch.")
    model.load_state_dict(best["model"])
    # Never select a threshold on outer fold.
    mon_metrics,mon_scores,mon_truth=v7.monitor_epoch(
        model,known_mon,normalizers,args,device
    )
    weak_threshold,monitor_f1=v7.monitor_threshold(
        mon_scores,mon_truth
    )
    del known_fit,known_mon,near_fit,near_mon

    val_pool,skip_val=v7.gather_candidates(
        val,blocks,mpb_s,bs_s,evaluate=True,
        stride_s=args.evaluation_stride_s
    )
    metrics,events=v7.evaluate_fold(
        model,val_pool,normalizers,args,device,weak_threshold,val
    )
    baseline_path=args.pretrained_dir/f"fold_{fold:02d}_event_metrics.csv"
    if not baseline_path.exists():
        raise FileNotFoundError(
            f"Required paired V7b comparison missing: {baseline_path}"
        )
    paired=with_baseline(events,baseline_path)
    deltas=[
        float(r["delta_v8_minus_v7b_s"])
        for r in paired if r["delta_v8_minus_v7b_s"]!=""
    ]
    comparisons={
        "n_paired":len(deltas),
        "paired_improved":int(sum(d<0 for d in deltas)),
        "paired_worsened":int(sum(d>0 for d in deltas)),
        "paired_tied":int(sum(d==0 for d in deltas)),
        "median_paired_error_delta_s":(
            float(np.median(deltas)) if deltas else None
        )
    }
    results={
        "schema":1,
        "architecture":"mag-swea-v8-near-ordinal-localizer",
        "fold":fold,"signature":signature,
        "v7b_source_checkpoint":str(source),
        "best_epoch":best_epoch,
        "epochs_completed":len(history),
        "train_events":len(fit),
        "inner_monitor_events":len(monitor),
        "test_events":len(val),
        "monitor_selected_loss":best_loss,
        "monitor_weak_candidate_threshold":weak_threshold,
        "monitor_weak_candidate_f1":monitor_f1,
        "normalizer_source":"V7b matching fold, fit-TRAIN only",
        "label_warning":(
            "Near-offset samples are unlabeled, ranked ordinally but NOT "
            "trained as false MPB or physical region negatives."
        ),
        "evaluation_warning":(
            "Maximum predicted score within +/-30 minutes around a "
            "cataloged MPB. This evaluates conditional localization, "
            "not detection of unknown boundaries."
        ),
        "train_history":history,
        "training_configuration":{
            "learning_rate":args.learning_rate,
            "rank_weight":args.rank_weight,
            "near_weight":args.near_weight,
            "near_margin":args.near_margin,
            "rank_control_step_s":args.rank_control_step_s,
            "rank_positives_per_event":args.rank_positives_per_event,
            "rank_hard_negatives":args.rank_hard_negatives,
            "rank_random_negatives":args.rank_random_negatives,
            "evaluation_stride_s":args.evaluation_stride_s,
        },
        "candidate_skips":{"fit":skip_fit,"monitor":skip_mon,
                           "near_fit":near_fit_skip,
                           "near_monitor":near_mon_skip,
                           "validation":skip_val},
        "paired_vs_v7b":comparisons,
        **metrics
    }
    v7.safe_json(folder/f"fold_{fold:02d}_metrics.json",results)
    v7.write_csv(folder/f"fold_{fold:02d}_event_metrics.csv",paired)
    v7.safe_checkpoint(final_path,{
        "schema":1,"architecture":results["architecture"],
        "signature":signature,"fold":fold,
        "best_epoch":best_epoch,
        "model":model.state_dict(),
        "normalizers":state["normalizers"],
        "monitor_threshold":weak_threshold,
    })
    print(
        f"FOLD {fold}: median error={metrics['median_abs_peak_error_s']:.1f}s "
        f"within 60s={metrics['fraction_within_60s']:.1%} "
        f"evaluable={metrics['val_evaluable_events']}/{len(val)} "
        f"paired improvements={comparisons['paired_improved']}/"
        f"{comparisons['n_paired']}.",
        flush=True
    )
    return results


def make_summary(args, completed):
    all_rows=[]
    for fold in sorted(completed):
        path=args.output_dir/f"fold_{fold:02d}_event_metrics.csv"
        with path.open(newline="",encoding="utf-8") as f:
            all_rows.extend(csv.DictReader(f))
    errors=np.asarray([
        float(r["absolute_error_s"]) for r in all_rows
        if r["absolute_error_s"]!=""
    ])
    old=np.asarray([
        float(r["v7b_absolute_error_s"]) for r in all_rows
        if r["v7b_absolute_error_s"]!=""
    ])
    aligned=[
        (float(r["absolute_error_s"]),float(r["v7b_absolute_error_s"]))
        for r in all_rows
        if r["absolute_error_s"]!="" and
        r["v7b_absolute_error_s"]!=""
    ]
    new_p=np.asarray([x[0] for x in aligned])
    old_p=np.asarray([x[1] for x in aligned])
    tp=sum(completed[f]["candidate_weak_tp"] for f in completed)
    fp=sum(completed[f]["candidate_weak_fp"] for f in completed)
    fn=sum(completed[f]["candidate_weak_fn"] for f in completed)
    return {
        "schema":1,
        "architecture":"mag-swea-v8-near-ordinal-localizer",
        "completed_folds":sorted(completed),
        "pilot_subset":bool(args.pilot_events_per_fold),
        "n_evaluable_events":len(errors),
        "global_median_absolute_error_s":(
            float(np.median(errors)) if len(errors) else None
        ),
        "fraction_within_60s":(
            float(np.mean(errors<=60)) if len(errors) else None
        ),
        "fraction_within_120s":(
            float(np.mean(errors<=120)) if len(errors) else None
        ),
        "fraction_within_300s":(
            float(np.mean(errors<=300)) if len(errors) else None
        ),
        "global_weak_candidate_f1":(
            2*tp/(2*tp+fp+fn) if 2*tp+fp+fn else None
        ),
        "paired_shared_events":len(aligned),
        "v8_shared_median_error_s":(
            float(np.median(new_p)) if len(aligned) else None
        ),
        "v7b_shared_median_error_s":(
            float(np.median(old_p)) if len(aligned) else None
        ),
        "v8_shared_within_60s":(
            float(np.mean(new_p<=60)) if len(aligned) else None
        ),
        "v7b_shared_within_60s":(
            float(np.mean(old_p<=60)) if len(aligned) else None
        ),
        "paired_improved":int(np.sum(new_p<old_p)),
        "paired_worsened":int(np.sum(new_p>old_p)),
        "paired_tied":int(np.sum(new_p==old_p)),
        "warning":(
            "Known-MPB conditional localization, NOT autonomous detector. "
            "These outer CV events were already consulted for choosing to "
            "try V8: future untouched temporal tests are needed for "
            "unbiased generalization estimates."
        ),
        "fold_results":[{
            "fold":f,
            "error_median_s":completed[f]["median_abs_peak_error_s"],
            "within_60s":completed[f]["fraction_within_60s"],
            "f1":completed[f]["candidate_weak_f1"],
            "best_epoch":completed[f]["best_epoch"]
        } for f in sorted(completed)]
    }


def main():
    a=parse_args()
    torch.set_num_threads(a.torch_threads)
    torch.manual_seed(a.seed)
    np.random.seed(a.seed)
    random.seed(a.seed)
    device=torch.device("cpu")
    rows=pilot.read_manifest(a.manifest)
    for r in rows:
        if not r.get("bs_timestamp_utc"):
            raise ValueError("V8 needs BS times to guard ambiguous zones.")
        r["bs_time"]=bq.to_utc(r["bs_timestamp_utc"])

    requested=[1,2,3,4] if a.folds=="all" else [int(a.folds)]
    parts={}
    for fold in requested:
        val=v7.chosen_events(
            rows,fold,pilot_count=a.pilot_events_per_fold
        )
        train=sorted([
            r for f in range(1,5) if f!=fold
            for r in v7.chosen_events(
                rows,f,pilot_count=a.pilot_events_per_fold
            )
        ],key=lambda r:r["time"])
        if len(train)<3 or not val:
            raise RuntimeError("Missing paired MAG+SWEA fold samples.")
        previous.check_disjoint_supports(train,val)
        previous.split_inner_monitor_months(train,fold,a.seed)
        source=a.pretrained_dir/f"fold_{fold:02d}_final.pt"
        baseline=a.pretrained_dir/f"fold_{fold:02d}_event_metrics.csv"
        if not source.is_file() or not baseline.is_file():
            raise FileNotFoundError(
                f"V7b full results missing fold {fold}: "
                f"{source} and {baseline}"
            )
        parts[fold]={"train":train,"val":val}

    a.output_dir.mkdir(parents=True,exist_ok=True)
    completed,todo={},{}
    for fold,groups in parts.items():
        fit,monitor,_=previous.split_inner_monitor_months(
            groups["train"],fold,a.seed
        )
        signature=fold_signature(
            a,fold,a.pretrained_dir/f"fold_{fold:02d}_final.pt",
            fit,monitor,groups["val"]
        )
        mpath=a.output_dir/f"fold_{fold:02d}_metrics.json"
        fpath=a.output_dir/f"fold_{fold:02d}_final.pt"
        if a.resume and mpath.exists() and fpath.exists():
            saved=json.loads(mpath.read_text(encoding="utf-8"))
            if (saved.get("signature")!=signature or
                    saved.get("architecture")!=
                    "mag-swea-v8-near-ordinal-localizer"):
                raise RuntimeError(
                    "Completed fold signature mismatch; change output path."
                )
            completed[fold]=saved
            print(f"FOLD {fold}: completed checkpoint verified; skip.",
                  flush=True)
        else:
            todo[fold]=groups

    if not a.execute:
        print(
            "Preflight only: V7b fold checkpoints and paired CSVs exist. "
            "Pass --execute to use MAG/SWEA caches and train.",
            flush=True
        )
        return

    if todo:
        unique={r["time"]:r for grp in todo.values()
                for subset in grp.values() for r in subset}
        print(
            f"V8: {len(unique)} MAG+SWEA event blocks; "
            f"pilot={bool(a.pilot_events_per_fold)}. "
            "Reuse cached 70-min sources.",
            flush=True
        )
        blocks=v7.load_event_blocks(list(unique.values()),a)
        if blocks is None:
            raise RuntimeError("Missing real source blocks.")
        for fold,groups in todo.items():
            completed[fold]=train_fold(
                fold,groups,blocks,rows,a,device
            )
    summary=make_summary(a,completed)
    v7.safe_json(a.output_dir/"cv_summary.json",summary)
    print(
        f"V8 GLOBAL: median={summary['global_median_absolute_error_s']}s "
        f"within60={summary['fraction_within_60s']:.1%} "
        f"V7b shared median={summary['v7b_shared_median_error_s']}s "
        f"V7b shared within60={summary['v7b_shared_within_60s']:.1%} "
        f"paired n={summary['paired_shared_events']}.",
        flush=True
    )


if __name__=="__main__":
    main()
