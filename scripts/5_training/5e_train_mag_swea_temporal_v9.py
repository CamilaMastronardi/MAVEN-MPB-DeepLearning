#!/usr/bin/env python3
"""V9: dense-in-time MPB localizer, conditioned on a known catalog crossing.

Pretrained V7b fold-specific MAG/SWEA encoders; new dense temporal head.
Positive 10-min windows contain the labeled MPB at varied offsets (not always
the center). Conservative negative 10-min windows are far from annotated
boundaries. Gaussian soft labels supervise the exact temporal location using
native MAG timestamps. An independent SWEA latent is fused by sample time.

Outer evaluation: scan each existing 70-min event block with overlapping
10-min windows and combine each window's dense probability over physical
time; choose the global density maximum in +/-30min. This is conditional
localization and does NOT evaluate autonomous MPB presence detection.
"""
from __future__ import annotations

import argparse
import csv
import hashlib
import json
import os
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
pilot=import_module("scripts.3_windowing.3b_bigquery_window_pilot")
previous=import_module("scripts.5_training.5a_train_mag_cv")
bq=import_module("scripts.1_bigquery.1c_bigquery_event_pilot")

from src.data.mpb_dual_windows import collate_dual, extract_clip
from src.data.mpb_weak_regions import (
    classify_candidate, candidate_evaluation_grid
)
from src.detection.mpb_temporal_localizer import (
    MagSweaTemporalLocalizer, temporal_loss
)

PRETRAINED=Path("results/metrics/mag_swea_regions_ranking_v1")
OUT=Path("results/metrics/mag_swea_temporal_v9")
POSITIVE_OFFSETS_S=(-240.,-180.,-120.,-60.,0.,60.,120.,180.,240.)
NEGATIVE_OFFSETS_S=(-1500.,-1200.,-900.,-600.,
                       600.,900.,1200.,1500.)


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
    p.add_argument("--max-epochs",type=int,default=40)
    p.add_argument("--min-epochs",type=int,default=8)
    p.add_argument("--patience",type=int,default=7)
    p.add_argument("--batch-size",type=int,default=16)
    p.add_argument("--seed",type=int,default=42)
    p.add_argument("--learning-rate",type=float,default=0.00015)
    p.add_argument("--localization-weight",type=float,default=1.5)
    p.add_argument("--gaussian-sigma-s",type=float,default=32.)
    p.add_argument("--train-positives-per-event",type=int,default=3)
    p.add_argument("--train-negatives-per-event",type=int,default=3)
    p.add_argument("--evaluation-stride-s",type=float,default=32.)
    p.add_argument("--evaluation-smoothing-s",type=int,default=9)
    p.add_argument("--torch-threads",type=int,default=2)
    p.add_argument("--max-query-gib",type=float,default=2.)
    p.add_argument("--max-total-gib",type=float,default=150.)
    p.add_argument("--max-mag-rows",type=int,default=25000)
    p.add_argument("--max-swea-rows",type=int,default=2000)
    p.add_argument("--resume",action="store_true")
    p.add_argument("--execute",action="store_true")
    a=p.parse_args()
    checks=(a.max_epochs,a.min_epochs,a.patience,a.batch_size,
            a.learning_rate,a.localization_weight,a.gaussian_sigma_s,
            a.train_positives_per_event,a.train_negatives_per_event,
            a.evaluation_stride_s,a.torch_threads,a.max_query_gib,
            a.max_total_gib,a.max_mag_rows,a.max_swea_rows)
    if (min(checks)<=0 or a.min_epochs>a.max_epochs
            or a.pilot_events_per_fold<0
            or a.evaluation_smoothing_s<1 or
            a.evaluation_smoothing_s%2!=1):
        p.error("Invalid V9 parameters or smoothing is not an odd length.")
    return a


def signature(args,fold,source,fit,monitor,val):
    spec={
        "model":"v9-dense-native-mag-time-swea-latent-v1",
        "fold":fold,
        "manifest_sha":v7.sha256_file(args.manifest),
        "v7b_checkpoint_sha":v7.sha256_file(source),
        "fit":[r["time"].isoformat() for r in fit],
        "monitor":[r["time"].isoformat() for r in monitor],
        "val":[r["time"].isoformat() for r in val],
        "epochs":[args.min_epochs,args.max_epochs,args.patience],
        "lr":args.learning_rate,"batch_size":args.batch_size,
        "seed":args.seed,"localization_weight":args.localization_weight,
        "sigma":args.gaussian_sigma_s,
        "train_examples":[args.train_positives_per_event,
                          args.train_negatives_per_event],
        "train_offset_s":POSITIVE_OFFSETS_S,
        "far_offsets_s":NEGATIVE_OFFSETS_S,
        "eval_stride":args.evaluation_stride_s,
        "eval_smooth":args.evaluation_smoothing_s,
    }
    return hashlib.sha256(
        json.dumps(spec,sort_keys=True).encode()
    ).hexdigest()


def normalizers_from_v7b(source):
    norms=source.get("normalizers")
    if not isinstance(norms,dict):
        raise RuntimeError("V7b fold source has no TRAIN-only normalizers.")
    return {
        sensor:{
            "mean":np.asarray(norms[sensor]["mean"],dtype=np.float64),
            "scale":np.asarray(norms[sensor]["scale"],dtype=np.float64),
            "n_epochs":norms[sensor]["n_epochs"],
        } for sensor in ("MAG","SWEA")
    }


def gather_training_pools(events,blocks,all_mpb_s,all_bs_s):
    """Only measured 10-min clips; near positive labels and safe far controls.

    Near positive windows can contain the MPB at arbitrary relative times;
    we DO NOT use V7b's +/-35s 'candidate center' binary classification
    to incorrectly label these as background.
    """
    result={}
    skipped=defaultdict(int)
    for row in events:
        t=row["time"]
        center=t.timestamp()
        mag,swea=blocks[t]["MAG"],blocks[t]["SWEA"]
        positives,negatives=[],[]
        for delta in POSITIVE_OFFSETS_S+NEGATIVE_OFFSETS_S:
            candidate=classify_candidate(
                row,center+delta,
                all_mpb_s=all_mpb_s,all_bs_s=all_bs_s
            )
            is_positive=delta in POSITIVE_OFFSETS_S
            if is_positive:
                # Allow near unknown (the MPB is genuinely inside this
                # 10-min window), but never near another labeled boundary.
                if candidate.kind not in ("positive","mpb_ambiguous"):
                    skipped["positive_"+candidate.kind]+=1
                    continue
            elif candidate.transition!=0:
                skipped["negative_"+candidate.kind]+=1
                continue
            clip=extract_clip(mag,swea,candidate.center_s)
            if clip is None:
                skipped["no_joint_coverage"]+=1
                continue
            example=(clip,int(is_positive),-float(delta))
            (positives if is_positive else negatives).append(example)
        if positives and negatives:
            result[t]={"positive":positives,"negative":negatives}
        else:
            skipped["events_without_both_labels"]+=1
    return result,dict(skipped)


def select_epoch_examples(pools,epoch,seed,args):
    rng=random.Random(seed+epoch)
    examples=[]
    for parts in pools.values():
        positives=parts["positive"]
        negatives=parts["negative"]
        examples.extend(rng.sample(
            positives,min(args.train_positives_per_event,len(positives))
        ))
        examples.extend(rng.sample(
            negatives,min(args.train_negatives_per_event,len(negatives))
        ))
    rng.shuffle(examples)
    return examples


def collate_examples(examples,normalizers,device):
    batch=collate_dual(
        [(clip,int(is_positive),(0,0))
         for clip,is_positive,_ in examples],
        normalizers,device
    )
    batch["target_offset"]=torch.tensor(
        [offset for _,_,offset in examples],
        device=device,dtype=torch.float32
    )
    return batch


def run_step(model,batch,args):
    output=model(
        batch["mag"],batch["mag_mask"],batch["mag_time"],
        batch["swea"],batch["swea_mask"],batch["swea_time"]
    )
    return temporal_loss(
        output,batch["mag_time"],batch["mag_mask"],
        batch["transition"],batch["target_offset"],
        sigma_s=args.gaussian_sigma_s,
        localization_weight=args.localization_weight
    )


def train_epoch(model,optimizer,pools,normalizers,args,device,epoch,fold):
    samples=select_epoch_examples(
        pools,epoch,args.seed+fold*10000,args
    )
    if not samples:
        raise RuntimeError("V9 has no train samples.")
    model.train()
    weighted=defaultdict(float)
    seen=0
    for start in range(0,len(samples),args.batch_size):
        items=samples[start:start+args.batch_size]
        batch=collate_examples(items,normalizers,device)
        optimizer.zero_grad(set_to_none=True)
        loss,metrics=run_step(model,batch,args)
        if not bool(torch.isfinite(loss)):
            raise FloatingPointError("Nonfinite V9 training loss.")
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(),1.0)
        optimizer.step()
        for k in ("det","time","total"):
            weighted[k]+=metrics[k]*len(items)
        seen+=len(items)
    result={k:v/seen for k,v in weighted.items()}
    result["n_examples"]=seen
    return result


@torch.no_grad()
def monitor_epoch(model,pools,normalizers,args,device):
    # Deterministic, distinct inner-month holdout samples.
    items=[]
    for group in pools.values():
        items.extend(group["positive"])
        items.extend(group["negative"])
    if not items:
        raise RuntimeError("No inner-month monitor examples.")
    model.eval()
    scores=defaultdict(float)
    n=0
    for start in range(0,len(items),args.batch_size):
        block=items[start:start+args.batch_size]
        batch=collate_examples(block,normalizers,device)
        loss,metrics=run_step(model,batch,args)
        for key in ("det","time","total"):
            scores[key]+=metrics[key]*len(block)
        n+=len(block)
    result={key:value/n for key,value in scores.items()}
    result["n_examples"]=n
    return result


@torch.no_grad()
def scan_event(model,row,blocks,normalizers,args,device):
    """Aggregate dense MPB distributions from all overlapping 10-min windows.

    The catalog MPB time is ONLY used to define the already-annotated 70-min
    block and evaluate prediction error. No labels/times are model features.
    """
    model.eval()
    target=row["time"].timestamp()
    mag,swea=blocks[row["time"]]["MAG"],blocks[row["time"]]["SWEA"]
    grid=candidate_evaluation_grid(
        row,np.asarray([],dtype=float),np.asarray([],dtype=float),
        step_s=args.evaluation_stride_s
    )
    clips=[]
    for candidate in grid:
        clip=extract_clip(mag,swea,candidate.center_s)
        if clip is not None:
            clips.append((candidate,clip))
    if not clips:
        raise RuntimeError(
            f"No usable MAG+SWEA scanning windows for {row['time']}."
        )
    # Reference grid 1-sec relative to label, for evaluation only.
    density=np.zeros(3601,dtype=np.float64)
    n_fused=0
    max_conf=0.
    for start in range(0,len(clips),args.batch_size):
        group=clips[start:start+args.batch_size]
        batch=collate_dual(
            [(clip,0,(0,0)) for _,clip in group],
            normalizers,device
        )
        output=model(
            batch["mag"],batch["mag_mask"],batch["mag_time"],
            batch["swea"],batch["swea_mask"],batch["swea_time"]
        )
        probabilities=output["time_prob"].cpu().numpy()
        confidences=torch.sigmoid(
            output["window_logit"]
        ).cpu().numpy()
        native_times=batch["mag_time"].cpu().numpy()
        native_masks=batch["mag_mask"].cpu().numpy()
        for j,(candidate,clip) in enumerate(group):
            mask=native_masks[j]
            relative_abs=(candidate.center_s-target)+native_times[j,mask]
            indices=np.rint(relative_abs).astype(np.int64)+1800
            weights=probabilities[j,mask]*float(confidences[j])
            inside=(indices>=0)&(indices<len(density))
            np.add.at(density,indices[inside],weights[inside])
            max_conf=max(max_conf,float(confidences[j]))
            n_fused+=1
    width=args.evaluation_smoothing_s
    density=np.convolve(density,np.ones(width)/width,mode="same")
    peak=int(np.argmax(density))-1800
    error=abs(peak)
    return {
        "mpb_timestamp_utc":row["time"].isoformat(),
        "source_group":row["source_group"],
        "peak_offset_s":peak,
        "absolute_error_s":error,
        "n_windows":n_fused,
        "peak_density":float(density[peak+1800]),
        "max_window_confidence":max_conf
    }


def paired_results(event_rows,baseline_file):
    with baseline_file.open(newline="",encoding="utf-8") as f:
        originals={
            r["mpb_timestamp_utc"]:r for r in csv.DictReader(f)
        }
    compared=[]
    for record in event_rows:
        key=record["mpb_timestamp_utc"]
        old=originals.get(key)
        if old is None or old.get("absolute_error_s","")=="":
            raise RuntimeError(
                f"No matched V7b reference error for V9 event {key}"
            )
        prev=float(old["absolute_error_s"])
        record=dict(record)
        record["v7b_absolute_error_s"]=prev
        record["delta_v9_minus_v7b_s"]=record["absolute_error_s"]-prev
        compared.append(record)
    return compared


def train_fold(fold,parts,blocks,all_rows,args,device):
    fit,monitor,removed=previous.split_inner_monitor_months(
        parts["train"],fold,args.seed
    )
    val=parts["val"]
    source_path=args.pretrained_dir/f"fold_{fold:02d}_final.pt"
    baseline_path=args.pretrained_dir/f"fold_{fold:02d}_event_metrics.csv"
    source=torch.load(source_path,map_location="cpu",weights_only=False)
    if source.get("fold")!=fold or "model" not in source:
        raise RuntimeError("Wrong V7b source fold.")
    normalizers=normalizers_from_v7b(source)
    sig=signature(args,fold,source_path,fit,monitor,val)
    all_mpb_s=np.asarray(
        [r["time"].timestamp() for r in all_rows],dtype=np.float64
    )
    all_bs_s=np.asarray(
        [r["bs_time"].timestamp() for r in all_rows],dtype=np.float64
    )
    fit_pool,fit_skips=gather_training_pools(
        fit,blocks,all_mpb_s,all_bs_s
    )
    mon_pool,mon_skips=gather_training_pools(
        monitor,blocks,all_mpb_s,all_bs_s
    )
    if not fit_pool or not mon_pool:
        raise RuntimeError("Missing positive and negative MAG+SWEA windows.")
    print(
        f"FOLD {fold}: fit={len(fit)} usable={len(fit_pool)} "
        f"monitor={len(monitor)} usable={len(mon_pool)} "
        f"test={len(val)} excluded_train_overlaps={removed}.",
        flush=True
    )
    torch.manual_seed(args.seed+fold)
    model=MagSweaTemporalLocalizer().to(device)
    model.load_v7b_encoders(source["model"])
    optimizer=torch.optim.AdamW(
        model.parameters(),lr=args.learning_rate,weight_decay=1e-3
    )
    folder=args.output_dir
    last_path=folder/f"fold_{fold:02d}_last.pt"
    best_path=folder/f"fold_{fold:02d}_best.pt"
    final_path=folder/f"fold_{fold:02d}_final.pt"
    start_epoch,best_epoch,stale=0,0,0
    best_loss=float("inf")
    history=[]
    if last_path.exists():
        if not args.resume:
            raise RuntimeError(f"{last_path} exists; use --resume.")
        state=torch.load(last_path,map_location=device,weights_only=False)
        if state["signature"]!=sig:
            raise RuntimeError("V9 checkpoint signature mismatch.")
        model.load_state_dict(state["model"])
        optimizer.load_state_dict(state["optimizer"])
        start_epoch,best_epoch,stale=(
            state["epoch"],state["best_epoch"],state["stale"]
        )
        best_loss,history=state["best_loss"],state["history"]
        print(f"FOLD {fold}: resumed from epoch {start_epoch}.",flush=True)
    for epoch in range(start_epoch+1,args.max_epochs+1):
        if epoch>=args.min_epochs and stale>=args.patience:
            break
        tr=train_epoch(
            model,optimizer,fit_pool,normalizers,args,device,epoch,fold
        )
        mon=monitor_epoch(model,mon_pool,normalizers,args,device)
        if epoch>=args.min_epochs:
            if mon["total"]<best_loss-1e-4:
                best_loss=mon["total"]
                best_epoch=epoch
                stale=0
                v7.safe_checkpoint(best_path,{
                    "signature":sig,"epoch":epoch,
                    "model":model.state_dict()
                })
            else:
                stale+=1
        history.append({
            "epoch":epoch,
            "train_det":tr["det"],
            "train_time":tr["time"],
            "train_total":tr["total"],
            "train_n":tr["n_examples"],
            "val_det":mon["det"],
            "val_time":mon["time"],
            "val_total":mon["total"],
            "val_n":mon["n_examples"],
            "best_epoch":best_epoch,"stale":stale
        })
        v7.safe_checkpoint(last_path,{
            "signature":sig,"epoch":epoch,
            "best_epoch":best_epoch,"best_loss":best_loss,"stale":stale,
            "history":history,"model":model.state_dict(),
            "optimizer":optimizer.state_dict()
        })
        print(
            f"FOLD {fold} EPOCH {epoch}: "
            f"train_det={tr['det']:.4f} train_time={tr['time']:.4f} "
            f"val_det={mon['det']:.4f} val_time={mon['time']:.4f} "
            f"val_total={mon['total']:.4f} "
            f"best={best_epoch} stale={stale}.",
            flush=True
        )
        if epoch>=args.min_epochs and stale>=args.patience:
            print(
                f"FOLD {fold}: early stopping at {epoch}, best {best_epoch}.",
                flush=True
            )
            break
    if not best_path.exists():
        raise RuntimeError(f"FOLD {fold}: missing selected V9 checkpoint.")
    best=torch.load(best_path,map_location=device,weights_only=False)
    if best["signature"]!=sig:
        raise RuntimeError("V9 selected checkpoint signature mismatch.")
    model.load_state_dict(best["model"])
    del fit_pool,mon_pool
    event_rows=[
        scan_event(model,row,blocks,normalizers,args,device)
        for row in val
    ]
    event_rows=paired_results(event_rows,baseline_path)
    errors=np.asarray(
        [r["absolute_error_s"] for r in event_rows],dtype=float
    )
    old=np.asarray(
        [r["v7b_absolute_error_s"] for r in event_rows],dtype=float
    )
    metrics={
        "schema":1,"architecture":"mag-swea-v9-dense-temporal-localizer",
        "fold":fold,"signature":sig,
        "best_epoch":best_epoch,
        "epochs_completed":len(history),
        "train_events":len(fit),"inner_monitor_events":len(monitor),
        "val_total_events":len(val),
        "val_evaluable_events":len(errors),
        "median_abs_peak_error_s":float(np.median(errors)),
        "mean_abs_peak_error_s":float(np.mean(errors)),
        "p90_abs_peak_error_s":float(np.percentile(errors,90)),
        "fraction_within_60s":float(np.mean(errors<=60)),
        "fraction_within_120s":float(np.mean(errors<=120)),
        "fraction_within_300s":float(np.mean(errors<=300)),
        "paired_v7b_median_abs_error_s":float(np.median(old)),
        "paired_v7b_fraction_within_60s":float(np.mean(old<=60)),
        "paired_improved":int(np.sum(errors<old)),
        "paired_worsened":int(np.sum(errors>old)),
        "paired_tied":int(np.sum(errors==old)),
        "median_paired_delta_s":float(np.median(errors-old)),
        "train_history":history,
        "skipped_fit":fit_skips,"skipped_monitor":mon_skips,
        "normalizer_source":"matching V7b fold TRAIN-only normalizers",
        "training_configuration":{
            "positive_offsets_s":POSITIVE_OFFSETS_S,
            "negative_offsets_s":NEGATIVE_OFFSETS_S,
            "gaussian_sigma_s":args.gaussian_sigma_s,
            "localization_weight":args.localization_weight,
            "learning_rate":args.learning_rate,
            "evaluation_stride_s":args.evaluation_stride_s,
            "evaluation_smoothing_s":args.evaluation_smoothing_s,
        },
        "evaluation_warning":(
            "CONDITIONAL localization: all +/-30-min evaluation "
            "neighborhoods contain catalog MPB. No independent orbital "
            "presence/absence detection is measured."
        )
    }
    v7.safe_json(folder/f"fold_{fold:02d}_metrics.json",metrics)
    v7.write_csv(
        folder/f"fold_{fold:02d}_event_metrics.csv",event_rows
    )
    v7.safe_checkpoint(final_path,{
        "schema":1,"fold":fold,"signature":sig,
        "architecture":metrics["architecture"],
        "best_epoch":best_epoch,
        "model":model.state_dict(),
        "normalizers":source["normalizers"],
        "evaluation_stride_s":args.evaluation_stride_s,
        "evaluation_smoothing_s":args.evaluation_smoothing_s,
    })
    print(
        f"FOLD {fold}: median error={metrics['median_abs_peak_error_s']:.1f}s "
        f"within 60s={metrics['fraction_within_60s']:.1%} "
        f"evaluable={len(errors)}/{len(val)} "
        f"improved_vs_v7b={metrics['paired_improved']}/{len(val)}.",
        flush=True
    )
    return metrics


def summarize(args,results):
    rows=[]
    for fold in sorted(results):
        path=args.output_dir/f"fold_{fold:02d}_event_metrics.csv"
        with path.open(newline="",encoding="utf-8") as f:
            rows.extend(csv.DictReader(f))
    now=np.asarray(
        [float(r["absolute_error_s"]) for r in rows],dtype=float
    )
    old=np.asarray(
        [float(r["v7b_absolute_error_s"]) for r in rows],dtype=float
    )
    return {
        "schema":1,
        "architecture":"mag-swea-v9-dense-temporal-localizer",
        "completed_folds":sorted(results),
        "pilot_subset":bool(args.pilot_events_per_fold),
        "n_evaluable_events":len(now),
        "global_median_absolute_error_s":float(np.median(now)),
        "global_mean_absolute_error_s":float(np.mean(now)),
        "global_p90_absolute_error_s":float(np.percentile(now,90)),
        "fraction_within_60s":float(np.mean(now<=60)),
        "fraction_within_120s":float(np.mean(now<=120)),
        "fraction_within_300s":float(np.mean(now<=300)),
        "paired_v7b_median_error_s":float(np.median(old)),
        "paired_v7b_within_60s":float(np.mean(old<=60)),
        "paired_improved":int(np.sum(now<old)),
        "paired_worsened":int(np.sum(now>old)),
        "paired_tied":int(np.sum(now==old)),
        "median_paired_delta_s":float(np.median(now-old)),
        "fold_results":[{
            "fold":f,
            "n_events":results[f]["val_evaluable_events"],
            "median_error_s":results[f]["median_abs_peak_error_s"],
            "within_60s":results[f]["fraction_within_60s"],
            "improved_vs_v7b":results[f]["paired_improved"]
        } for f in sorted(results)],
        "warning":(
            "Same external folds already used for prior model selection: "
            "this exploratory benchmark is not a new untouched test. "
            "All scans are conditioned on known catalog MPB events."
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
    for row in rows:
        if not row.get("bs_timestamp_utc"):
            raise ValueError("Missing BS label needed to exclude ambiguities.")
        row["bs_time"]=bq.to_utc(row["bs_timestamp_utc"])
    requested=[1,2,3,4] if a.folds=="all" else [int(a.folds)]
    selections={}
    for fold in requested:
        val=v7.chosen_events(
            rows,fold,pilot_count=a.pilot_events_per_fold
        )
        train=sorted(
            [r for f in range(1,5) if f!=fold
             for r in v7.chosen_events(
                 rows,f,pilot_count=a.pilot_events_per_fold
             )],
            key=lambda r:r["time"]
        )
        if len(train)<3 or not val:
            raise RuntimeError("Insufficient MAG+SWEA paired fold data.")
        previous.check_disjoint_supports(train,val)
        previous.split_inner_monitor_months(train,fold,a.seed)
        checkpoint=a.pretrained_dir/f"fold_{fold:02d}_final.pt"
        baseline=a.pretrained_dir/f"fold_{fold:02d}_event_metrics.csv"
        if not checkpoint.is_file() or not baseline.is_file():
            raise FileNotFoundError(
                f"Full V7b checkpoint/reference missing for fold {fold}: "
                f"{checkpoint}, {baseline}"
            )
        selections[fold]={"train":train,"val":val}
    a.output_dir.mkdir(parents=True,exist_ok=True)
    done,todo={},{}
    for fold,selection in selections.items():
        fit,monitor,_=previous.split_inner_monitor_months(
            selection["train"],fold,a.seed
        )
        sig=signature(
            a,fold,
            a.pretrained_dir/f"fold_{fold:02d}_final.pt",
            fit,monitor,selection["val"]
        )
        mp=a.output_dir/f"fold_{fold:02d}_metrics.json"
        fp=a.output_dir/f"fold_{fold:02d}_final.pt"
        if a.resume and mp.is_file() and fp.is_file():
            saved=json.loads(mp.read_text(encoding="utf-8"))
            if (saved.get("signature")!=sig or
                    saved.get("architecture")!=
                    "mag-swea-v9-dense-temporal-localizer"):
                raise RuntimeError("Completed V9 signature mismatch.")
            done[fold]=saved
            print(f"FOLD {fold}: validated complete, skip.",flush=True)
        else:
            todo[fold]=selection
    if not a.execute:
        print(
            "V9 PREFLIGHT: verified V7b checkpoints and reference event "
            "CSV. No BigQuery or training without --execute.",
            flush=True
        )
        return
    if todo:
        unique={
            r["time"]:r for selection in todo.values()
            for part in selection.values() for r in part
        }
        print(
            f"V9: {len(unique)} cached MAG+SWEA source events; "
            f"pilot={bool(a.pilot_events_per_fold)}.",
            flush=True
        )
        blocks=v7.load_event_blocks(list(unique.values()),a)
        if blocks is None:
            raise RuntimeError("No usable source data.")
        for fold,selection in todo.items():
            done[fold]=train_fold(
                fold,selection,blocks,rows,a,device
            )
    summary=summarize(a,done)
    v7.safe_json(a.output_dir/"cv_summary.json",summary)
    print(
        f"V9 GLOBAL: median={summary['global_median_absolute_error_s']:.1f}s "
        f"within60={summary['fraction_within_60s']:.1%} "
        f"V7b paired median={summary['paired_v7b_median_error_s']:.1f}s "
        f"within60={summary['paired_v7b_within_60s']:.1%} "
        f"improved={summary['paired_improved']}/{summary['n_evaluable_events']}.",
        flush=True
    )


if __name__=="__main__":
    main()
