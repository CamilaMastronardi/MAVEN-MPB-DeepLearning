#!/usr/bin/env python3
"""V10: V7b physical-region teacher + trainable local temporal refinement.

The original V7b candidate score proposes a coarse MPB time. A new
multiscale head, trained on shifted 10-min measured MAG/SWEA windows,
predicts a small time correction. Both the teacher and the normalizers
come from the exact matching V7b fold, which never trained on that outer
fold's events. The teacher remains frozen.

The correction weight and mean-vs-peak estimator are calibrated ONLY on
inner-month monitor data. Include weight=0 (exact V7b baseline) as a
fallback. All evaluations are CONDITIONAL on known MPB in a +/-30 min
neighborhood, not autonomous orbit-level MPB detection.
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

ROOT=Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0,str(ROOT))

v7=import_module("scripts.5_training.5b_train_mag_swea_regions")
v9=import_module("scripts.5_training.5e_train_mag_swea_temporal_v9")
pilot=import_module("scripts.3_windowing.3b_bigquery_window_pilot")
previous=import_module("scripts.5_training.5a_train_mag_cv")
bq=import_module("scripts.1_bigquery.1c_bigquery_event_pilot")

from src.data.mpb_dual_windows import collate_dual,extract_clip
from src.data.mpb_weak_regions import candidate_evaluation_grid
from src.detection.mpb_region_anchored_refiner import (
    RegionAnchoredTemporalRefiner,refiner_loss
)

OUT=Path("results/metrics/mag_swea_refiner_v10")
PRETRAINED=Path("results/metrics/mag_swea_regions_ranking_v1")
CALIBRATION_ALPHAS=(0.,.25,.5,.75,1.)


def parse_args():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument("--manifest",type=Path,default=v7.MANIFEST)
    p.add_argument("--pretrained-dir",type=Path,default=PRETRAINED)
    p.add_argument("--output-dir",type=Path,default=OUT)
    p.add_argument("--mag-cache-dir",type=Path,default=v7.MAG_CACHE)
    p.add_argument("--swea-cache-dir",type=Path,default=v7.SWEA_CACHE)
    p.add_argument("--refresh-swea-cache",action="store_true")
    p.add_argument("--folds",choices=("all","1","2","3","4"),default="all")
    p.add_argument("--pilot-events-per-fold",type=int,default=0)
    p.add_argument("--max-epochs",type=int,default=45)
    p.add_argument("--min-epochs",type=int,default=10)
    p.add_argument("--patience",type=int,default=9)
    p.add_argument("--batch-size",type=int,default=16)
    p.add_argument("--torch-threads",type=int,default=2)
    p.add_argument("--seed",type=int,default=42)
    p.add_argument("--learning-rate",type=float,default=0.0002)
    p.add_argument("--localization-weight",type=float,default=1.5)
    p.add_argument("--offset-weight",type=float,default=0.5)
    p.add_argument("--gaussian-sigma-s",type=float,default=32.)
    p.add_argument("--train-positives-per-event",type=int,default=3)
    p.add_argument("--train-negatives-per-event",type=int,default=3)
    p.add_argument("--evaluation-stride-s",type=float,default=32.)
    p.add_argument("--max-query-gib",type=float,default=2.)
    p.add_argument("--max-total-gib",type=float,default=150.)
    p.add_argument("--max-mag-rows",type=int,default=25000)
    p.add_argument("--max-swea-rows",type=int,default=2000)
    p.add_argument("--resume",action="store_true")
    p.add_argument("--execute",action="store_true")
    a=p.parse_args()
    positive=(
        a.max_epochs,a.min_epochs,a.patience,a.batch_size,
        a.torch_threads,a.learning_rate,a.localization_weight,
        a.gaussian_sigma_s,a.train_positives_per_event,
        a.train_negatives_per_event,a.evaluation_stride_s,
        a.max_query_gib,a.max_total_gib,a.max_mag_rows,a.max_swea_rows
    )
    if (min(positive)<=0 or a.min_epochs>a.max_epochs
            or a.pilot_events_per_fold<0 or a.offset_weight<0):
        p.error("Invalid V10 configuration.")
    return a


def signature(args,fold,checkpoint,fit,monitor,val):
    spec={
        "architecture":"v10-v7b-region-anchored-temporal-refinement-v1",
        "fold":fold,
        "manifest_sha":v7.sha256_file(args.manifest),
        "source_checkpoint_sha":v7.sha256_file(checkpoint),
        "fit":[r["time"].isoformat() for r in fit],
        "monitor":[r["time"].isoformat() for r in monitor],
        "val":[r["time"].isoformat() for r in val],
        "max_epochs":args.max_epochs,"min_epochs":args.min_epochs,
        "patience":args.patience,"batch_size":args.batch_size,
        "lr":args.learning_rate,"seed":args.seed,
        "loc_weight":args.localization_weight,
        "offset_weight":args.offset_weight,
        "sigma":args.gaussian_sigma_s,
        "n_positive":args.train_positives_per_event,
        "n_negative":args.train_negatives_per_event,
        "eval_stride":args.evaluation_stride_s,
        "calibration_alphas":CALIBRATION_ALPHAS,
    }
    return hashlib.sha256(
        json.dumps(spec,sort_keys=True).encode()
    ).hexdigest()


def collate_examples(examples,normalizers,device):
    batch=collate_dual(
        [(clip,positive,(0,0))
         for clip,positive,_ in examples],
        normalizers,device
    )
    batch["target_offset"]=torch.tensor(
        [offset for _,_,offset in examples],
        device=device,dtype=torch.float32
    )
    return batch


def loss_step(model,batch,args):
    out=model(
        batch["mag"],batch["mag_mask"],batch["mag_time"],
        batch["swea"],batch["swea_mask"],batch["swea_time"]
    )
    return refiner_loss(
        out,batch["mag_time"],batch["mag_mask"],
        batch["transition"],batch["target_offset"],
        sigma_s=args.gaussian_sigma_s,
        localization_weight=args.localization_weight,
        offset_weight=args.offset_weight
    )


def train_epoch(model,optimizer,pools,normalizers,args,device,epoch,fold):
    # Same safe, measured shifted-positive/far-control windows as V9,
    # but a different pretrained physics-anchored network architecture.
    examples=v9.select_epoch_examples(
        pools,epoch,args.seed+fold*10000,args
    )
    if not examples:
        raise RuntimeError("No V10 training examples.")
    model.train()
    total=defaultdict(float)
    seen=0
    for i in range(0,len(examples),args.batch_size):
        batch=collate_examples(
            examples[i:i+args.batch_size],normalizers,device
        )
        optimizer.zero_grad(set_to_none=True)
        loss,metric=loss_step(model,batch,args)
        if not bool(torch.isfinite(loss)):
            raise FloatingPointError("Nonfinite V10 train loss.")
        loss.backward()
        torch.nn.utils.clip_grad_norm_(
            [p for p in model.parameters() if p.requires_grad],1.
        )
        optimizer.step()
        n=len(examples[i:i+args.batch_size])
        for key in ("bce","temporal_kl","offset","total"):
            total[key]+=metric[key]*n
        seen+=n
    return {k:value/seen for k,value in total.items()}


@torch.no_grad()
def monitor_epoch(model,pools,normalizers,args,device):
    samples=[
        example
        for group in pools.values()
        for key in ("positive","negative")
        for example in group[key]
    ]
    if not samples:
        raise RuntimeError("No inner-month monitor examples.")
    model.eval()
    aggregate=defaultdict(float)
    seen=0
    for i in range(0,len(samples),args.batch_size):
        group=samples[i:i+args.batch_size]
        batch=collate_examples(group,normalizers,device)
        loss,metric=loss_step(model,batch,args)
        for key in ("bce","temporal_kl","offset","total"):
            aggregate[key]+=metric[key]*len(group)
        seen+=len(group)
    return {key:value/seen for key,value in aggregate.items()}


@torch.no_grad()
def teacher_coarse_proposals(
    model,events,blocks,normalizers,mpb_s,bs_s,args,device
):
    """Find best V7b window in each known-MPB neighborhood WITHOUT labels.

    Catalog MPB only defines the fixed +/-30-min search interval, exactly
    as in V7b. All candidate centers (including weakly unknown) are scored.
    """
    model.eval()
    proposals=[]
    for idx,row in enumerate(events,1):
        t=row["time"]
        mag,swea=blocks[t]["MAG"],blocks[t]["SWEA"]
        grid=candidate_evaluation_grid(
            row,mpb_s,bs_s,step_s=args.evaluation_stride_s
        )
        windows=[]
        for candidate in grid:
            clip=extract_clip(mag,swea,candidate.center_s)
            if clip is not None:
                windows.append((candidate,clip))
        if not windows:
            raise RuntimeError(
                f"No complete MAG/SWEA candidates for {t}."
            )
        best_logit=float("-inf")
        best=None
        for start in range(0,len(windows),args.batch_size):
            items=windows[start:start+args.batch_size]
            batch=collate_dual(
                [(clip,0,(0,0)) for _,clip in items],
                normalizers,device
            )
            logits,_=model.teacher_logits(
                batch["mag"],batch["mag_mask"],batch["mag_time"],
                batch["swea"],batch["swea_mask"],batch["swea_time"]
            )
            # Match V7b evaluation exactly, including sigmoid ties.\n            values=torch.sigmoid(logits).cpu().numpy()
            max_index=int(np.argmax(values))
            if float(values[max_index])>best_logit:
                best_logit=float(values[max_index])
                best=items[max_index]
        candidate,clip=best
        proposals.append({
            "mpb_timestamp_utc":t.isoformat(),
            "source_group":row["source_group"],
            "target_s":t.timestamp(),
            "candidate_s":float(candidate.center_s),
            "coarse_offset_s":float(candidate.center_s-t.timestamp()),
            "n_scanned_windows":len(windows),
            "coarse_logit":best_logit,
            "clip":clip,
        })
        if idx%50==0 or idx==len(events):
            print(
                f"V7B COARSE SCAN {idx}/{len(events)}.",
                flush=True
            )
    return proposals


@torch.no_grad()
def refine_proposals(model,proposals,normalizers,args,device):
    model.eval()
    refined=[]
    for i in range(0,len(proposals),args.batch_size):
        group=proposals[i:i+args.batch_size]
        batch=collate_dual(
            [(item["clip"],0,(0,0)) for item in group],
            normalizers,device
        )
        out=model(
            batch["mag"],batch["mag_mask"],batch["mag_time"],
            batch["swea"],batch["swea_mask"],batch["swea_time"]
        )
        for j,row in enumerate(group):
            record={key:value for key,value in row.items()
                    if key!="clip"}
            record["refined_mean_offset_s"]=float(
                np.clip(
                    out["expected_offset_s"][j].item(),-240.,240.
                )
            )
            record["refined_peak_offset_s"]=float(
                np.clip(
                    out["peak_offset_s"][j].item(),-240.,240.
                )
            )
            record["refiner_confidence"]=float(torch.sigmoid(
                out["presence_logit"][j]
            ).item())
            refined.append(record)
    return refined


def corrected_error(record,method,alpha):
    delta=(record["refined_mean_offset_s"]
           if method=="mean"
           else record["refined_peak_offset_s"])
    time=float(record["coarse_offset_s"])+alpha*delta
    time=float(np.clip(time,-1800.,1800.))
    return abs(time),time


def calibrate_refinement(refined):
    """Pick the mean/peak residual strength from INNER monitor ONLY.

    Always include alpha=0, the exact V7b baseline; tie-break toward
    smaller corrections to avoid overfitting a small monitor sample.
    """
    if not refined:
        raise RuntimeError("Cannot calibrate without inner-month proposals.")
    best=None
    for alpha in CALIBRATION_ALPHAS:
        for method in ("mean","peak"):
            errors=np.asarray([
                corrected_error(row,method,alpha)[0]
                for row in refined
            ],dtype=float)
            # Optimize time localization on monitor; avoid outliers
            # dominating via 10-min winsorization and prefer modest alpha.
            score=(
                -float(np.mean(np.minimum(errors,600.))),
                float(np.mean(errors<=60.)),
                -float(np.median(errors)),
                -alpha,
                int(method=="mean"),
            )
            if best is None or score>best[0]:
                best=(
                    score,{"alpha":float(alpha),"method":method,
                           "monitor_median_error_s":float(np.median(errors)),
                           "monitor_within_60s":float(np.mean(errors<=60.)),
                           "monitor_clipped_mae_s":float(np.mean(
                               np.minimum(errors,600.)
                           ))}
                )
    return best[1]


def load_v7b_reference(path):
    with path.open(newline="",encoding="utf-8") as fp:
        return {
            row["mpb_timestamp_utc"]:row
            for row in csv.DictReader(fp)
        }


def evaluate_fold(refined,calibration,source_csv):
    baseline=load_v7b_reference(source_csv)
    results=[]
    for row in refined:
        key=row["mpb_timestamp_utc"]
        ref=baseline.get(key)
        if (ref is None or
                ref.get("absolute_error_s","")=="" or
                ref.get("peak_offset_s","")==""):
            raise RuntimeError(
                f"Missing V7b prediction on identical event {key}"
            )
        reference_offset=float(ref["peak_offset_s"])
        if abs(reference_offset-row["coarse_offset_s"])>0.1:
            raise RuntimeError(
                f"V7b coarse proposal mismatch on {key}: "
                f"recomputed {row['coarse_offset_s']} vs "
                f"baseline {reference_offset}. "
                "Never compare different candidate grids/checkpoints."
            )
        error,offset=corrected_error(
            row,calibration["method"],calibration["alpha"]
        )
        old_error=float(ref["absolute_error_s"])
        record={
            "mpb_timestamp_utc":key,
            "source_group":row["source_group"],
            "coarse_peak_offset_s":reference_offset,
            "v7b_absolute_error_s":old_error,
            "refiner_mean_offset_s":row["refined_mean_offset_s"],
            "refiner_peak_offset_s":row["refined_peak_offset_s"],
            "refiner_confidence":row["refiner_confidence"],
            "selected_offset_s":offset,
            "absolute_error_s":error,
            "delta_v10_minus_v7b_s":error-old_error,
            "n_windows":row["n_scanned_windows"],
            "calibration_alpha":calibration["alpha"],
            "calibration_method":calibration["method"],
        }
        results.append(record)
    new=np.asarray([r["absolute_error_s"] for r in results],dtype=float)
    old=np.asarray(
        [r["v7b_absolute_error_s"] for r in results],dtype=float
    )
    return {
        "val_evaluable_events":len(results),
        "median_abs_peak_error_s":float(np.median(new)),
        "mean_abs_peak_error_s":float(np.mean(new)),
        "p90_abs_peak_error_s":float(np.percentile(new,90)),
        "fraction_within_60s":float(np.mean(new<=60)),
        "fraction_within_120s":float(np.mean(new<=120)),
        "fraction_within_300s":float(np.mean(new<=300)),
        "paired_v7b_median_error_s":float(np.median(old)),
        "paired_v7b_within_60s":float(np.mean(old<=60)),
        "paired_improved":int(np.sum(new<old)),
        "paired_worsened":int(np.sum(new>old)),
        "paired_tied":int(np.sum(new==old)),
        "median_paired_delta_s":float(np.median(new-old)),
    },results


def train_fold(fold,parts,blocks,all_rows,args,device):
    fit,monitor,removed=previous.split_inner_monitor_months(
        parts["train"],fold,args.seed
    )
    val=parts["val"]
    pretrained=args.pretrained_dir/f"fold_{fold:02d}_final.pt"
    source=torch.load(pretrained,map_location="cpu",weights_only=False)
    if (source.get("fold")!=fold or
            "model" not in source or
            "normalizers" not in source):
        raise RuntimeError("V10 V7b source fold is incomplete or mismatched.")
    normalizers=v9.normalizers_from_v7b(source)
    sig=signature(args,fold,pretrained,fit,monitor,val)
    all_mpb_s=np.asarray(
        [r["time"].timestamp() for r in all_rows],dtype=np.float64
    )
    all_bs_s=np.asarray(
        [r["bs_time"].timestamp() for r in all_rows],dtype=np.float64
    )
    train_pool,train_skips=v9.gather_training_pools(
        fit,blocks,all_mpb_s,all_bs_s
    )
    monitor_pool,monitor_skips=v9.gather_training_pools(
        monitor,blocks,all_mpb_s,all_bs_s
    )
    if not train_pool or not monitor_pool:
        raise RuntimeError("Missing V10 real positive/far-negative clips.")
    print(
        f"FOLD {fold}: fit={len(fit)} (usable={len(train_pool)}) "
        f"monitor={len(monitor)} (usable={len(monitor_pool)}) "
        f"outer={len(val)}; excluded overlaps={removed}.",
        flush=True
    )
    torch.manual_seed(args.seed+fold)
    model=RegionAnchoredTemporalRefiner(source["model"]).to(device)
    optimizer=torch.optim.AdamW(
        [p for p in model.parameters() if p.requires_grad],
        lr=args.learning_rate,weight_decay=1e-3
    )
    root=args.output_dir
    last=root/f"fold_{fold:02d}_last.pt"
    best=root/f"fold_{fold:02d}_best.pt"
    final=root/f"fold_{fold:02d}_final.pt"
    start,best_epoch,stale=0,0,0
    best_loss=float("inf")
    history=[]
    if last.exists():
        if not args.resume:
            raise RuntimeError(
                f"{last} exists. Use --resume or another output dir."
            )
        old=torch.load(last,map_location=device,weights_only=False)
        if old["signature"]!=sig:
            raise RuntimeError("V10 resume signature mismatch.")
        model.load_state_dict(old["model"])
        optimizer.load_state_dict(old["optimizer"])
        start=old["epoch"]
        best_epoch=old["best_epoch"]
        stale=old["stale"]
        best_loss=old["best_loss"]
        history=old["history"]
        print(f"FOLD {fold}: resumed from epoch {start}.",flush=True)
    for epoch in range(start+1,args.max_epochs+1):
        if epoch>=args.min_epochs and stale>=args.patience:
            break
        tr=train_epoch(
            model,optimizer,train_pool,normalizers,args,
            device,epoch,fold
        )
        mon=monitor_epoch(
            model,monitor_pool,normalizers,args,device
        )
        if epoch>=args.min_epochs:
            if mon["total"]<best_loss-1e-4:
                best_loss=mon["total"]
                best_epoch=epoch
                stale=0
                v7.safe_checkpoint(best,{
                    "signature":sig,"epoch":epoch,
                    "model":model.state_dict()
                })
            else:
                stale+=1
        history.append({
            "epoch":epoch,"train_bce":tr["bce"],
            "train_time":tr["temporal_kl"],
            "train_offset":tr["offset"],
            "train_total":tr["total"],
            "val_bce":mon["bce"],"val_time":mon["temporal_kl"],
            "val_offset":mon["offset"],"val_total":mon["total"],
            "best_epoch":best_epoch,"stale":stale,
        })
        v7.safe_checkpoint(last,{
            "signature":sig,"epoch":epoch,
            "best_epoch":best_epoch,"best_loss":best_loss,
            "stale":stale,"history":history,
            "model":model.state_dict(),
            "optimizer":optimizer.state_dict()
        })
        print(
            f"FOLD {fold} EPOCH {epoch}: "
            f"train_bce={tr['bce']:.4f} train_time={tr['temporal_kl']:.4f} "
            f"val_bce={mon['bce']:.4f} val_time={mon['temporal_kl']:.4f} "
            f"val_total={mon['total']:.4f} "
            f"best={best_epoch} stale={stale}.",
            flush=True
        )
        if epoch>=args.min_epochs and stale>=args.patience:
            print(
                f"FOLD {fold}: early stopped at {epoch}; best {best_epoch}.",
                flush=True
            )
            break
    if not best.exists():
        raise RuntimeError(f"FOLD {fold}: no V10 best checkpoint.")
    best_state=torch.load(best,map_location=device,weights_only=False)
    if best_state["signature"]!=sig:
        raise RuntimeError("V10 best signature mismatch.")
    model.load_state_dict(best_state["model"])
    del train_pool,monitor_pool

    # Calibration only uses inner-monitor months; never the outer test.
    monitor_coarse=teacher_coarse_proposals(
        model,monitor,blocks,normalizers,all_mpb_s,all_bs_s,
        args,device
    )
    monitor_refined=refine_proposals(
        model,monitor_coarse,normalizers,args,device
    )
    calibration=calibrate_refinement(monitor_refined)
    print(
        f"FOLD {fold}: INNER MONITOR selected refinement "
        f"method={calibration['method']} "
        f"alpha={calibration['alpha']:.2f} "
        f"MAE<=600s={calibration['monitor_clipped_mae_s']:.1f}.",
        flush=True
    )
    del monitor_coarse,monitor_refined
    outer_coarse=teacher_coarse_proposals(
        model,val,blocks,normalizers,all_mpb_s,all_bs_s,
        args,device
    )
    outer_refined=refine_proposals(
        model,outer_coarse,normalizers,args,device
    )
    metrics,events=evaluate_fold(
        outer_refined,calibration,
        args.pretrained_dir/f"fold_{fold:02d}_event_metrics.csv"
    )
    result={
        "schema":1,
        "architecture":"v10-frozen-region-teacher-temporal-refiner",
        "fold":fold,"signature":sig,
        "best_epoch":best_epoch,
        "epochs_completed":len(history),
        "train_events":len(fit),"monitor_events":len(monitor),
        "val_total_events":len(val),
        "train_skips":train_skips,
        "monitor_skips":monitor_skips,
        "monitor_calibration":calibration,
        "train_history":history,
        "normalizer_source":"matching V7b fit-TRAIN normalization",
        "training_configuration":{
            "learning_rate":args.learning_rate,
            "gaussian_sigma_s":args.gaussian_sigma_s,
            "localization_weight":args.localization_weight,
            "offset_weight":args.offset_weight,
            "n_train_positive":args.train_positives_per_event,
            "n_train_negative":args.train_negatives_per_event,
            "evaluation_stride_s":args.evaluation_stride_s,
            "teacher_frozen":True
        },
        "evaluation_warning":(
            "Conditional MPB localization around known catalog events; "
            "coarse proposals from V7b score in +/-30 min, not orbital "
            "autonomous detection. Alpha selected using inner TRAIN months."
        ),
        **metrics
    }
    v7.safe_json(root/f"fold_{fold:02d}_metrics.json",result)
    v7.write_csv(root/f"fold_{fold:02d}_event_metrics.csv",events)
    v7.safe_checkpoint(final,{
        "schema":1,"fold":fold,"signature":sig,
        "architecture":result["architecture"],
        "best_epoch":best_epoch,"model":model.state_dict(),
        "normalizers":source["normalizers"],
        "calibration":calibration,
    })
    print(
        f"FOLD {fold}: median error={metrics['median_abs_peak_error_s']:.1f}s "
        f"within 60s={metrics['fraction_within_60s']:.1%} "
        f"evaluable={len(events)}/{len(val)} "
        f"improved_vs_v7b={metrics['paired_improved']}/{len(events)} "
        f"(alpha={calibration['alpha']:.2f}).",
        flush=True
    )
    return result


def summarize(args,results):
    rows=[]
    for fold in sorted(results):
        path=args.output_dir/f"fold_{fold:02d}_event_metrics.csv"
        with path.open(newline="",encoding="utf-8") as f:
            rows.extend(csv.DictReader(f))
    curr=np.asarray(
        [float(r["absolute_error_s"]) for r in rows],dtype=float
    )
    base=np.asarray(
        [float(r["v7b_absolute_error_s"]) for r in rows],dtype=float
    )
    return {
        "schema":1,
        "architecture":"v10-frozen-region-teacher-temporal-refiner",
        "completed_folds":sorted(results),
        "pilot_subset":bool(args.pilot_events_per_fold),
        "n_evaluable_events":len(rows),
        "global_median_absolute_error_s":float(np.median(curr)),
        "global_mean_absolute_error_s":float(np.mean(curr)),
        "global_p90_absolute_error_s":float(np.percentile(curr,90)),
        "fraction_within_60s":float(np.mean(curr<=60)),
        "fraction_within_120s":float(np.mean(curr<=120)),
        "fraction_within_300s":float(np.mean(curr<=300)),
        "paired_v7b_median_error_s":float(np.median(base)),
        "paired_v7b_within_60s":float(np.mean(base<=60)),
        "paired_improved":int(np.sum(curr<base)),
        "paired_worsened":int(np.sum(curr>base)),
        "paired_tied":int(np.sum(curr==base)),
        "fold_results":[{
            "fold":fold,
            "n_events":results[fold]["val_evaluable_events"],
            "median_error_s":results[fold]["median_abs_peak_error_s"],
            "within_60s":results[fold]["fraction_within_60s"],
            "alpha":results[fold]["monitor_calibration"]["alpha"],
            "method":results[fold]["monitor_calibration"]["method"],
            "improved_vs_v7b":results[fold]["paired_improved"],
        } for fold in sorted(results)],
        "warning":(
            "Exploratory development compared on the same outer folds "
            "already consulted for V7b/V8/V9. New untouched temporal "
            "events are needed for an unbiased final generalization test. "
            "This remains conditional localization, not full-orbit detection."
        )
    }


def main():
    a=parse_args()
    torch.set_num_threads(a.torch_threads)
    torch.manual_seed(a.seed)
    random.seed(a.seed)
    np.random.seed(a.seed)
    device=torch.device("cpu")
    rows=pilot.read_manifest(a.manifest)
    for r in rows:
        if not r.get("bs_timestamp_utc"):
            raise RuntimeError("V10 requires catalog BS for safe controls.")
        r["bs_time"]=bq.to_utc(r["bs_timestamp_utc"])
    folds=[1,2,3,4] if a.folds=="all" else [int(a.folds)]
    selections={}
    for fold in folds:
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
            raise RuntimeError("Insufficient paired MAG/SWEA data.")
        previous.check_disjoint_supports(train,val)
        previous.split_inner_monitor_months(train,fold,a.seed)
        source=a.pretrained_dir/f"fold_{fold:02d}_final.pt"
        source_csv=a.pretrained_dir/f"fold_{fold:02d}_event_metrics.csv"
        if not source.is_file() or not source_csv.is_file():
            raise FileNotFoundError(
                f"Missing full V7b sources: {source}, {source_csv}"
            )
        selections[fold]={"train":train,"val":val}
    a.output_dir.mkdir(parents=True,exist_ok=True)
    finished,to_do={},{}
    for fold,groups in selections.items():
        fit,monitor,_=previous.split_inner_monitor_months(
            groups["train"],fold,a.seed
        )
        sig=signature(
            a,fold,
            a.pretrained_dir/f"fold_{fold:02d}_final.pt",
            fit,monitor,groups["val"]
        )
        metrics_path=a.output_dir/f"fold_{fold:02d}_metrics.json"
        final_path=a.output_dir/f"fold_{fold:02d}_final.pt"
        if a.resume and metrics_path.is_file() and final_path.is_file():
            stored=json.loads(metrics_path.read_text(encoding="utf-8"))
            if (stored.get("signature")!=sig or
                    stored.get("architecture")!=
                    "v10-frozen-region-teacher-temporal-refiner"):
                raise RuntimeError("Completed V10 signature mismatch.")
            finished[fold]=stored
            print(f"FOLD {fold}: completed; verified and skipped.",flush=True)
        else:
            to_do[fold]=groups
    if not a.execute:
        print(
            "V10 preflight: all V7b checkpoints and paired CSVs exist; "
            "use --execute to train. No BigQuery in this preflight.",
            flush=True
        )
        return
    if to_do:
        unique={
            r["time"]:r for grp in to_do.values()
            for group in grp.values() for r in group
        }
        print(
            f"V10 {len(unique)} measured MAG/SWEA event blocks "
            "(cache first), teacher frozen throughout.",
            flush=True
        )
        blocks=v7.load_event_blocks(list(unique.values()),a)
        if blocks is None:
            raise RuntimeError("No measured sources.")
        for fold,groups in to_do.items():
            finished[fold]=train_fold(
                fold,groups,blocks,rows,a,device
            )
    summary=summarize(a,finished)
    v7.safe_json(a.output_dir/"cv_summary.json",summary)
    print(
        f"V10 GLOBAL: median={summary['global_median_absolute_error_s']:.1f}s "
        f"within60={summary['fraction_within_60s']:.1%} "
        f"V7b same events median={summary['paired_v7b_median_error_s']:.1f}s "
        f"within60={summary['paired_v7b_within_60s']:.1%} "
        f"improved={summary['paired_improved']}/"
        f"{summary['n_evaluable_events']}.",
        flush=True
    )


if __name__=="__main__":
    main()
