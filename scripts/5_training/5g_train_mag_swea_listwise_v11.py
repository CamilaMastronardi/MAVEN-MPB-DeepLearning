#!/usr/bin/env python3
"""V11: re-rank ALL valid MPB candidates, then refine the selected one with V10.

V7b (frozen) extracts physical candidate features in 10-minute native-cadence
MAG+SWEA clips at 32-s stride across each KNOWN-MPB 70-min neighborhood.
A compact Conv1d model learns full-event/listwise selection, not merely
the top-1 shift. The winning candidate is passed to the FROZEN V10
temporal refiner. Mixing weight 0 falls back exactly to V10.

All supervision, early stopping, hyperparameter calibration and feature
standardization remain within outer TRAIN months; outer folds are untouched
until the final evaluation. Event-relative target offsets appear ONLY
in loss and metrics, NEVER in the reranker inputs. Scans assume an MPB
is already known to lie in the search interval, not orbit detection.
"""
from __future__ import annotations

import argparse
import csv
import hashlib
import json
import os
import random
import sys
from importlib import import_module
from pathlib import Path

import numpy as np
import torch

ROOT=Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0,str(ROOT))

v7=import_module("scripts.5_training.5b_train_mag_swea_regions")
v9=import_module("scripts.5_training.5e_train_mag_swea_temporal_v9")
v10=import_module("scripts.5_training.5f_train_mag_swea_refiner_v10")
pilot=import_module("scripts.3_windowing.3b_bigquery_window_pilot")
previous=import_module("scripts.5_training.5a_train_mag_cv")
bq=import_module("scripts.1_bigquery.1c_bigquery_event_pilot")

from src.data.mpb_dual_windows import collate_dual,extract_clip
from src.data.mpb_weak_regions import candidate_evaluation_grid
from src.detection.mag_swea_region_model import MagSweaRegionAutoencoder
from src.detection.mpb_region_anchored_refiner import RegionAnchoredTemporalRefiner
from src.detection.mpb_listwise_reranker_v11 import (
    MPBListwiseReranker,physical_candidate_features,
    fit_feature_normalizer,tensor_event,soft_event_loss,
    select_candidate,distinct_topk_coverage
)

V7B=Path("results/metrics/mag_swea_regions_ranking_v1")
V10=Path("results/metrics/mag_swea_refiner_v10")
OUT=Path("results/metrics/mag_swea_listwise_v11")
CALIBRATION_WEIGHTS=(0.,0.5,1.,1.5,2.)


def parse_args():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument("--manifest",type=Path,default=v7.MANIFEST)
    p.add_argument("--v7b-dir",type=Path,default=V7B)
    p.add_argument("--v10-dir",type=Path,default=V10)
    p.add_argument("--output-dir",type=Path,default=OUT)
    p.add_argument("--mag-cache-dir",type=Path,default=v7.MAG_CACHE)
    p.add_argument("--swea-cache-dir",type=Path,default=v7.SWEA_CACHE)
    p.add_argument("--refresh-swea-cache",action="store_true")
    p.add_argument("--folds",choices=("all","1","2","3","4"),
                   default="all")
    p.add_argument("--pilot-events-per-fold",type=int,default=0)
    p.add_argument("--max-epochs",type=int,default=70)
    p.add_argument("--min-epochs",type=int,default=10)
    p.add_argument("--patience",type=int,default=12)
    p.add_argument("--batch-size",type=int,default=16)
    p.add_argument("--events-per-update",type=int,default=12)
    p.add_argument("--torch-threads",type=int,default=2)
    p.add_argument("--seed",type=int,default=42)
    p.add_argument("--learning-rate",type=float,default=0.0005)
    p.add_argument("--target-sigma-s",type=float,default=56.)
    p.add_argument("--softmax-temperature",type=float,default=2.)
    p.add_argument("--evaluation-stride-s",type=float,default=32.)
    p.add_argument("--max-query-gib",type=float,default=2.)
    p.add_argument("--max-total-gib",type=float,default=150.)
    p.add_argument("--max-mag-rows",type=int,default=25000)
    p.add_argument("--max-swea-rows",type=int,default=2000)
    p.add_argument("--resume",action="store_true")
    p.add_argument("--execute",action="store_true")
    a=p.parse_args()
    checks=(
        a.max_epochs,a.min_epochs,a.patience,a.batch_size,
        a.events_per_update,a.torch_threads,a.learning_rate,
        a.target_sigma_s,a.softmax_temperature,
        a.evaluation_stride_s,a.max_query_gib,
        a.max_total_gib,a.max_mag_rows,a.max_swea_rows,
    )
    if (min(checks)<=0 or a.max_epochs<a.min_epochs or
            a.pilot_events_per_fold<0 or
            a.evaluation_stride_s!=32.):
        p.error("Invalid parameters. V11 must use 32-s V7b scan grid.")
    return a


def experiment_signature(args,fold,v7b_checkpoint,v10_checkpoint,
                         fit,monitor,val):
    data={
        "model":"v11-listwise-full-grid-frozen-V7b-V10-v1",
        "fold":fold,"manifest_sha":v7.sha256_file(args.manifest),
        "v7b_sha":v7.sha256_file(v7b_checkpoint),
        "v10_sha":v7.sha256_file(v10_checkpoint),
        "fit":[r["time"].isoformat() for r in fit],
        "monitor":[r["time"].isoformat() for r in monitor],
        "val":[r["time"].isoformat() for r in val],
        "max_epochs":args.max_epochs,
        "min_epochs":args.min_epochs,"patience":args.patience,
        "batch_size":args.batch_size,
        "events_per_update":args.events_per_update,
        "seed":args.seed,"lr":args.learning_rate,
        "target_sigma":args.target_sigma_s,
        "temperature":args.softmax_temperature,
        "stride":args.evaluation_stride_s,
        "calibration_weights":CALIBRATION_WEIGHTS
    }
    return hashlib.sha256(json.dumps(
        data,sort_keys=True
    ).encode("utf-8")).hexdigest()


def cache_path(args,fold,event):
    date=event.strftime("%Y%m%dT%H%M%S%f")
    return args.output_dir/"features"/f"fold_{fold:02d}"/f"{date}.npz"


def cache_feature_record(path,expected_sha):
    if not path.is_file():
        return None
    with np.load(path,allow_pickle=False) as packed:
        source=str(packed["source_sha"].item())
        if source!=expected_sha:
            raise RuntimeError(
                f"Cached V11 features from different model/config: {path}"
            )
        return {
            "features":packed["features"].astype(np.float32),
            "offsets":packed["offsets"].astype(np.float32),
        }


def save_feature_record(path,record,source_sha):
    path.parent.mkdir(parents=True,exist_ok=True)
    tmp=path.with_suffix(".npz.tmp")
    with tmp.open("wb") as fh:
        np.savez_compressed(
            fh,source_sha=np.asarray(source_sha),
            features=record["features"],
            offsets=record["offsets"]
        )
    os.replace(tmp,path)


@torch.no_grad()
def measured_event_features(
    teacher,event,blocks,norm,args,device,mpb_s,bs_s
):
    """Score complete measured V7b scan without access to target index."""
    teacher.eval()
    t=event["time"]
    mag,swea=blocks[t]["MAG"],blocks[t]["SWEA"]
    candidates=candidate_evaluation_grid(
        event,mpb_s,bs_s,step_s=args.evaluation_stride_s
    )
    records=[]
    for candidate in candidates:
        clip=extract_clip(mag,swea,candidate.center_s)
        if clip is not None:
            records.append((candidate,clip))
    if len(records)<2:
        return None
    features=[]
    offsets=[]
    for i in range(0,len(records),args.batch_size):
        sub=records[i:i+args.batch_size]
        batch=collate_dual([
            (clip,0,(0,0)) for _,clip in sub
        ],norm,device)
        output=teacher(
            batch["mag"],batch["mag_mask"],batch["mag_time"],
            batch["swea"],batch["swea_mask"],batch["swea_time"]
        )
        features.append(physical_candidate_features(batch,output))
        offsets.extend(
            c.center_s-t.timestamp() for c,_ in sub
        )
    data={
        "features":np.concatenate(features,axis=0).astype(np.float32),
        # OFFSETS ARE SUPERVISION / EVALUATION METADATA ONLY.
        "offsets":np.asarray(offsets,dtype=np.float32)
    }
    if not np.isfinite(data["features"]).all():
        raise RuntimeError("Nonfinite V11 teacher physics features.")
    return data


def get_feature_records(
    teacher,fold,events,blocks,norm,args,device,mpb_s,bs_s,source_sha
):
    result={}
    n_cached=n_new=n_skipped=0
    for i,event in enumerate(events,1):
        path=cache_path(args,fold,event["time"])
        record=cache_feature_record(path,source_sha)
        if record is None:
            record=measured_event_features(
                teacher,event,blocks,norm,args,device,mpb_s,bs_s
            )
            if record is None:
                n_skipped+=1
                continue
            save_feature_record(path,record,source_sha)
            n_new+=1
        else:
            n_cached+=1
        result[event["time"]]=record
        if i%50==0 or i==len(events):
            print(
                f"V11 FOLD {fold} FEATURES {i}/{len(events)} "
                f"cached={n_cached} fresh={n_new} missing={n_skipped}.",
                flush=True
            )
    return result


def train_epoch(model,optimizer,events,records,norm,args,fold,epoch,device):
    model.train()
    rng=random.Random(args.seed+fold*10000+epoch)
    active=[r for r in events if r["time"] in records and
            np.any(np.abs(records[r["time"]]["offsets"])<=120.)]
    rng.shuffle(active)
    if not active:
        raise RuntimeError("No valid target-bearing V11 TRAIN scans.")
    optimizer.zero_grad(set_to_none=True)
    total=0.
    pending=0
    count=0
    for i,event in enumerate(active,1):
        record=records[event["time"]]
        features,base=tensor_event(record,norm,device)
        scores,resid=model(features,base)
        loss=soft_event_loss(
            scores,record["offsets"],
            sigma_s=args.target_sigma_s,
            temperature=args.softmax_temperature,
        )+0.0005*resid.square().mean()
        if not bool(torch.isfinite(loss)):
            raise FloatingPointError("Nonfinite V11 event ranking loss.")
        # Accumulate event gradients without storing all event graphs.
        (loss/args.events_per_update).backward()
        total+=float(loss.detach())
        count+=1
        pending+=1
        if pending==args.events_per_update or i==len(active):
            torch.nn.utils.clip_grad_norm_(model.parameters(),1.)
            optimizer.step()
            optimizer.zero_grad(set_to_none=True)
            pending=0
    return {"event_nll":total/count,"train_events":count}


@torch.no_grad()
def monitor_epoch(model,events,records,norm,args,device):
    model.eval()
    losses=[]
    errors=[]
    for event in events:
        record=records.get(event["time"])
        if record is None or not np.any(
            np.abs(record["offsets"])<=120.
        ):
            continue
        features,base=tensor_event(record,norm,device)
        scores,resid=model(features,base)
        loss=soft_event_loss(
            scores,record["offsets"],
            sigma_s=args.target_sigma_s,
            temperature=args.softmax_temperature,
        )
        losses.append(float(loss))
        j=select_candidate(
            scores.cpu().numpy(),record["features"][:,0],alpha=1.
        )
        errors.append(abs(float(record["offsets"][j])))
    if not losses:
        raise RuntimeError("No valid INNER monitor V11 scan events.")
    err=np.asarray(errors,dtype=float)
    return {
        "event_nll":float(np.mean(losses)),
        "coarse_median_s":float(np.median(err)),
        "coarse_within_60s":float(np.mean(err<=60.)),
        "coarse_clipped_mae_s":float(np.minimum(err,600.).mean()),
        "events":len(err),
    }


@torch.no_grad()
def score_records(model,events,records,norm,device):
    model.eval()
    result={}
    for event in events:
        when=event["time"]
        record=records.get(when)
        if record is None:
            continue
        x,base=tensor_event(record,norm,device)
        scores,_=model(x,base)
        result[when]=scores.cpu().numpy()
    return result


def selected_proposals(events,records,scores,alpha,blocks):
    """No label/target used to choose candidate; only scores + measured clip."""
    selected=[]
    for event in events:
        when=event["time"]
        if when not in records or when not in scores:
            continue
        record=records[when]
        j=select_candidate(
            scores[when],record["features"][:,0],alpha=alpha
        )
        offset=float(record["offsets"][j])
        center=when.timestamp()+offset
        clip=extract_clip(
            blocks[when]["MAG"],blocks[when]["SWEA"],center
        )
        if clip is None:
            raise RuntimeError("Valid feature scan candidate lost its data.")
        selected.append({
            "mpb_timestamp_utc":when.isoformat(),
            "source_group":event["source_group"],
            "target_s":when.timestamp(),
            "candidate_s":center,
            "coarse_offset_s":offset,
            "n_scanned_windows":len(record["offsets"]),
            "coarse_logit":float(
                record["features"][j,0]
            ),
            "candidate_index":j,
            "clip":clip,
        })
    return selected


def finalized_predictions(refined,calibration):
    """Apply strictly earlier V10 monitor-chosen offset correction."""
    records=[]
    for item in refined:
        shift=(
            item["refined_mean_offset_s"]
            if calibration["method"]=="mean" else
            item["refined_peak_offset_s"]
        )
        total=float(np.clip(
            item["coarse_offset_s"]+calibration["alpha"]*shift,
            -1800.,1800.
        ))
        records.append({
            **item,
            "selected_offset_s":total,
            "absolute_error_s":abs(total),
        })
    return records


@torch.no_grad()
def calibrate_alpha(
    model,events,records,blocks,norm,feature_norm,
    fine_model,fine_calibration,args,device
):
    """Pick candidate score blend using INNER months and V10 frozen fine stage.

    Alpha=0 reproduces the V10 selection (exact sigmoid V7b rank). Outer
    validation targets are never passed to this calibration function.
    """
    scores=score_records(model,events,records,feature_norm,device)
    candidates={}
    for alpha in CALIBRATION_WEIGHTS:
        proposals=selected_proposals(
            events,records,scores,alpha,blocks
        )
        fine=v10.refine_proposals(
            fine_model,proposals,norm,args,device
        )
        predictions=finalized_predictions(fine,fine_calibration)
        errors=np.asarray(
            [r["absolute_error_s"] for r in predictions],dtype=float
        )
        if len(errors)==0:
            raise RuntimeError("No INNER monitor predictions.")
        candidates[alpha]={
            "alpha":float(alpha),
            "events":len(errors),
            "within_60s":float(np.mean(errors<=60.)),
            "within_120s":float(np.mean(errors<=120.)),
            "median_error_s":float(np.median(errors)),
            "clipped_mean_error_s":float(np.mean(
                np.minimum(errors,600.)
            ))
        }
    # 60-s localization is the primary intended objective.
    # With identical accuracy prefer lower clipped error, then alpha=0.
    best=max(
        candidates.values(),
        key=lambda row:(
            row["within_60s"],-row["clipped_mean_error_s"],
            -row["median_error_s"],-row["alpha"]
        )
    )
    return {"selected":best,"trials":list(candidates.values())}


def paired_metrics(predicted,baseline_v10,baseline_v7b):
    with baseline_v10.open(newline="",encoding="utf-8") as f:
        v10_by_time={
            r["mpb_timestamp_utc"]:r for r in csv.DictReader(f)
        }
    with baseline_v7b.open(newline="",encoding="utf-8") as f:
        v7_by_time={
            r["mpb_timestamp_utc"]:r for r in csv.DictReader(f)
        }
    output=[]
    for r in predicted:
        key=r["mpb_timestamp_utc"]
        v10row=v10_by_time.get(key)
        v7row=v7_by_time.get(key)
        if v10row is None or v7row is None:
            raise RuntimeError(f"Missing V7b/V10 paired event: {key}")
        old10=float(v10row["absolute_error_s"])
        old7=float(v7row["absolute_error_s"])
        oldcoarse=float(v7row["peak_offset_s"])
        # No change in candidate-search grid/teacher is allowed.
        if not np.isfinite(oldcoarse):
            raise RuntimeError("V7b baseline has no finite peak offset.")
        output.append({
            "mpb_timestamp_utc":key,
            "source_group":r["source_group"],
            "n_windows":r["n_scanned_windows"],
            "v7b_coarse_offset_s":oldcoarse,
            "reranked_candidate_offset_s":r["coarse_offset_s"],
            "refiner_mean_offset_s":r["refined_mean_offset_s"],
            "refiner_peak_offset_s":r["refined_peak_offset_s"],
            "selected_offset_s":r["selected_offset_s"],
            "absolute_error_s":r["absolute_error_s"],
            "v7b_absolute_error_s":old7,
            "v10_absolute_error_s":old10,
            "delta_v11_minus_v10_s":r["absolute_error_s"]-old10,
            "delta_v11_minus_v7b_s":r["absolute_error_s"]-old7,
        })
    new=np.asarray([r["absolute_error_s"] for r in output])
    old10=np.asarray([r["v10_absolute_error_s"] for r in output])
    old7=np.asarray([r["v7b_absolute_error_s"] for r in output])
    return {
        "val_evaluable_events":len(new),
        "median_error_s":float(np.median(new)),
        "mean_error_s":float(np.mean(new)),
        "p90_error_s":float(np.percentile(new,90)),
        "fraction_within_60s":float(np.mean(new<=60.)),
        "fraction_within_120s":float(np.mean(new<=120.)),
        "fraction_within_300s":float(np.mean(new<=300.)),
        "paired_v10_median_s":float(np.median(old10)),
        "paired_v10_within_60s":float(np.mean(old10<=60.)),
        "paired_v7b_median_s":float(np.median(old7)),
        "paired_v7b_within_60s":float(np.mean(old7<=60.)),
        "improved_vs_v10":int(np.sum(new<old10)),
        "worsened_vs_v10":int(np.sum(new>old10)),
        "tied_vs_v10":int(np.sum(new==old10)),
        "improved_vs_v7b":int(np.sum(new<old7)),
        "reached_target_80pct":bool(np.mean(new<=60.)>=.8),
    },output


def train_fold(fold,selection,blocks,all_rows,args,device):
    fit,monitor,removed=previous.split_inner_monitor_months(
        selection["train"],fold,args.seed
    )
    val=selection["val"]
    path7=args.v7b_dir/f"fold_{fold:02d}_final.pt"
    path10=args.v10_dir/f"fold_{fold:02d}_final.pt"
    v7state=torch.load(path7,map_location="cpu",weights_only=False)
    v10state=torch.load(path10,map_location="cpu",weights_only=False)
    if v7state.get("fold")!=fold or v10state.get("fold")!=fold:
        raise RuntimeError("V11 teacher/refiner fold mismatch.")
    if "model" not in v7state or "model" not in v10state:
        raise RuntimeError("Missing source V7b/V10 model state.")
    if "normalizers" not in v7state or "calibration" not in v10state:
        raise RuntimeError("Missing fold V7b norms or V10 calibration.")
    norm=v9.normalizers_from_v7b(v7state)
    sig=experiment_signature(
        args,fold,path7,path10,fit,monitor,val
    )
    source_sha=hashlib.sha256(
        (v7.sha256_file(path7)+v7.sha256_file(args.manifest)+
         str(args.evaluation_stride_s)).encode()
    ).hexdigest()
    teacher=MagSweaRegionAutoencoder().to(device)
    teacher.load_state_dict(v7state["model"])
    teacher.eval().requires_grad_(False)
    all_mpb=np.asarray(
        [r["time"].timestamp() for r in all_rows],dtype=np.float64
    )
    all_bs=np.asarray(
        [r["bs_time"].timestamp() for r in all_rows],dtype=np.float64
    )
    union={r["time"]:r for r in fit+monitor+val}
    records=get_feature_records(
        teacher,fold,list(union.values()),blocks,norm,args,
        device,all_mpb,all_bs,source_sha
    )
    del teacher
    norm_events=[
        records[r["time"]]["features"] for r in fit
        if r["time"] in records
    ]
    if not norm_events:
        raise RuntimeError("No V11 TRAIN features after scan validation.")
    feature_norm=fit_feature_normalizer(norm_events)
    print(
        f"FOLD {fold}: fit={len(fit)} "
        f"available={sum(r['time'] in records for r in fit)}, "
        f"monitor={len(monitor)} "
        f"available={sum(r['time'] in records for r in monitor)}, "
        f"outer={len(val)} "
        f"available={sum(r['time'] in records for r in val)}; "
        f"excluded TRAIN overlaps={removed}.",
        flush=True
    )
    if any(r["time"] not in records for r in val):
        raise RuntimeError(
            "Outer evaluation has incomplete event scans. "
            "Do not silently drop holdout events."
        )
    torch.manual_seed(args.seed+fold)
    model=MPBListwiseReranker().to(device)
    optimizer=torch.optim.AdamW(
        model.parameters(),lr=args.learning_rate,weight_decay=1e-3
    )
    root=args.output_dir
    last=root/f"fold_{fold:02d}_last.pt"
    best=root/f"fold_{fold:02d}_best.pt"
    final=root/f"fold_{fold:02d}_final.pt"
    start_epoch,best_epoch,stale=0,0,0
    best_loss=float("inf")
    history=[]
    if last.is_file():
        if not args.resume:
            raise RuntimeError(f"{last} exists; use --resume.")
        snapshot=torch.load(last,map_location=device,weights_only=False)
        if snapshot.get("signature")!=sig:
            raise RuntimeError("V11 resume configuration mismatch.")
        model.load_state_dict(snapshot["model"])
        optimizer.load_state_dict(snapshot["optimizer"])
        start_epoch=snapshot["epoch"]
        best_epoch=snapshot["best_epoch"]
        best_loss=snapshot["best_loss"]
        stale=snapshot["stale"]
        history=snapshot["history"]
        print(
            f"FOLD {fold}: resumed at epoch {start_epoch}.",flush=True
        )
    for epoch in range(start_epoch+1,args.max_epochs+1):
        if epoch>=args.min_epochs and stale>=args.patience:
            break
        tr=train_epoch(
            model,optimizer,fit,records,feature_norm,args,
            fold,epoch,device
        )
        mon=monitor_epoch(
            model,monitor,records,feature_norm,args,device
        )
        if epoch>=args.min_epochs:
            if mon["event_nll"]<best_loss-1e-4:
                best_loss=mon["event_nll"]
                best_epoch=epoch
                stale=0
                v7.safe_checkpoint(best,{
                    "signature":sig,"epoch":epoch,
                    "model":model.state_dict()
                })
            else:
                stale+=1
        history.append({
            "epoch":epoch,
            "train_event_nll":tr["event_nll"],
            "val_event_nll":mon["event_nll"],
            "val_coarse_median_s":mon["coarse_median_s"],
            "val_coarse_within_60s":mon["coarse_within_60s"],
            "val_coarse_clipped_mae_s":mon["coarse_clipped_mae_s"],
            "best_epoch":best_epoch,"stale":stale,
        })
        v7.safe_checkpoint(last,{
            "signature":sig,"epoch":epoch,
            "best_epoch":best_epoch,"best_loss":best_loss,
            "stale":stale,"history":history,
            "model":model.state_dict(),
            "optimizer":optimizer.state_dict(),
        })
        print(
            f"FOLD {fold} EPOCH {epoch}: "
            f"train_listwise={tr['event_nll']:.4f} "
            f"val_listwise={mon['event_nll']:.4f} "
            f"val_coarse_within60={mon['coarse_within_60s']:.1%} "
            f"best={best_epoch} stale={stale}.",
            flush=True
        )
        if epoch>=args.min_epochs and stale>=args.patience:
            print(
                f"FOLD {fold}: early stop epoch {epoch}; "
                f"best epoch {best_epoch}.",flush=True
            )
            break
    if not best.is_file():
        raise RuntimeError("V11 produced no selected reranker.")
    state=torch.load(best,map_location=device,weights_only=False)
    if state["signature"]!=sig:
        raise RuntimeError("V11 selected checkpoint signature mismatch.")
    model.load_state_dict(state["model"])
    fine=RegionAnchoredTemporalRefiner(
        v7state["model"]
    ).to(device)
    fine.load_state_dict(v10state["model"],strict=True)
    fine.eval().requires_grad_(False)
    fine_calibration=v10state["calibration"]
    calibration=calibrate_alpha(
        model,monitor,records,blocks,norm,feature_norm,
        fine,fine_calibration,args,device
    )
    alpha=calibration["selected"]["alpha"]
    print(
        f"FOLD {fold}: INNER MONITOR selected ranking alpha={alpha:.2f} "
        f"within60={calibration['selected']['within_60s']:.1%} "
        f"(monitor N={calibration['selected']['events']}).",
        flush=True
    )
    outer_scores=score_records(
        model,val,records,feature_norm,device
    )
    proposals=selected_proposals(
        val,records,outer_scores,alpha,blocks
    )
    fine_predictions=v10.refine_proposals(
        fine,proposals,norm,args,device
    )
    predicted=finalized_predictions(
        fine_predictions,fine_calibration
    )
    metrics,events=paired_metrics(
        predicted,
        args.v10_dir/f"fold_{fold:02d}_event_metrics.csv",
        args.v7b_dir/f"fold_{fold:02d}_event_metrics.csv"
    )
    oracle=np.asarray([
        distinct_topk_coverage(
            records[row["time"]]["features"][:,0],
            records[row["time"]]["offsets"]
        ) for row in val
    ])
    result={
        "schema":1,
        "architecture":"v11-all-candidate-listwise-V7b-V10",
        "fold":fold,"signature":sig,
        "best_epoch":best_epoch,"epochs_completed":len(history),
        "train_events":len(fit),"inner_monitor_events":len(monitor),
        "outer_events":len(val),
        "train_history":history,
        "inner_monitor_ranking_calibration":calibration,
        "v10_fine_calibration":fine_calibration,
        "oracle_top5_v7b_candidate_within_240s":float(
            np.mean(oracle<=240.)
        ),
        "oracle_top5_v7b_candidate_within_60s":float(
            np.mean(oracle<=60.)
        ),
        "training_configuration":{
            "teacher_frozen":True,
            "refiner_frozen":True,
            "max_epochs":args.max_epochs,
            "target_sigma_s":args.target_sigma_s,
            "softmax_temperature":args.softmax_temperature,
            "evaluation_stride_s":args.evaluation_stride_s,
        },
        "evaluation_warning":(
            "All 70-min held-out test blocks contain known catalog MPB. "
            "The reranker never receives label-based candidate offsets "
            "as input. Top5 oracle is diagnostic only, not model selection."
        ),
        **metrics
    }
    v7.safe_json(root/f"fold_{fold:02d}_metrics.json",result)
    v7.write_csv(
        root/f"fold_{fold:02d}_event_metrics.csv",events
    )
    v7.safe_checkpoint(final,{
        "schema":1,"architecture":result["architecture"],
        "fold":fold,"signature":sig,
        "best_epoch":best_epoch,
        "model":model.state_dict(),
        "feature_normalizer":{
            k:v.tolist() for k,v in feature_norm.items()
        },
        "ranking_calibration":calibration,
        "source_v7b_checkpoint":str(path7),
        "source_v10_checkpoint":str(path10),
    })
    print(
        f"FOLD {fold}: median={metrics['median_error_s']:.1f}s "
        f"within 60s={metrics['fraction_within_60s']:.1%} "
        f"V10 within 60s={metrics['paired_v10_within_60s']:.1%} "
        f"improved={metrics['improved_vs_v10']}/{len(events)}, "
        f"alpha={alpha:.2f}.",
        flush=True
    )
    return result


def summarize(args,results):
    rows=[]
    for fold in sorted(results):
        file=args.output_dir/f"fold_{fold:02d}_event_metrics.csv"
        with file.open(newline="",encoding="utf-8") as f:
            rows.extend(csv.DictReader(f))
    new=np.asarray([float(r["absolute_error_s"]) for r in rows])
    old10=np.asarray([
        float(r["v10_absolute_error_s"]) for r in rows
    ])
    old7=np.asarray([
        float(r["v7b_absolute_error_s"]) for r in rows
    ])
    if len(new)==0:
        raise RuntimeError("No complete V11 validation events.")
    return {
        "schema":1,
        "architecture":"v11-all-candidate-listwise-V7b-V10",
        "completed_folds":sorted(results),
        "pilot_subset":bool(args.pilot_events_per_fold),
        "n_evaluable_events":len(new),
        "global_median_error_s":float(np.median(new)),
        "global_mean_error_s":float(np.mean(new)),
        "global_p90_error_s":float(np.percentile(new,90)),
        "fraction_within_60s":float(np.mean(new<=60.)),
        "fraction_within_120s":float(np.mean(new<=120.)),
        "fraction_within_300s":float(np.mean(new<=300.)),
        "target_80pct_met":bool(np.mean(new<=60.)>=.8),
        "n_within_60s":int(np.sum(new<=60.)),
        "paired_v10_median_s":float(np.median(old10)),
        "paired_v10_within_60s":float(np.mean(old10<=60.)),
        "paired_v7b_median_s":float(np.median(old7)),
        "paired_v7b_within_60s":float(np.mean(old7<=60.)),
        "improved_vs_v10":int(np.sum(new<old10)),
        "worsened_vs_v10":int(np.sum(new>old10)),
        "tied_vs_v10":int(np.sum(new==old10)),
        "oracle_top5_v7b_candidate_within_240s_by_fold":{
            str(f):results[f]["oracle_top5_v7b_candidate_within_240s"]
            for f in sorted(results)
        },
        "fold_results":[{
            "fold":fold,
            "n_events":results[fold]["val_evaluable_events"],
            "median_error_s":results[fold]["median_error_s"],
            "within_60s":results[fold]["fraction_within_60s"],
            "v10_within_60s":results[fold]["paired_v10_within_60s"],
            "alpha":results[fold][
                "inner_monitor_ranking_calibration"
            ]["selected"]["alpha"],
        } for fold in sorted(results)],
        "warning":(
            "This benchmark is conditional on MPB being known in a "
            "70-min neighborhood. Existing outer folds have already "
            "influenced iterative architecture selection; an untouched "
            "time-separated test is necessary for final generalization."
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
            raise RuntimeError("V11 needs paired BS to exclude risky windows.")
        r["bs_time"]=bq.to_utc(r["bs_timestamp_utc"])
    requested=[1,2,3,4] if a.folds=="all" else [int(a.folds)]
    groups={}
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
            raise RuntimeError("Insufficient MAG+SWEA paired events.")
        previous.check_disjoint_supports(train,val)
        previous.split_inner_monitor_months(train,fold,a.seed)
        for folder in (a.v7b_dir,a.v10_dir):
            for suffix in ("final.pt","event_metrics.csv"):
                path=folder/f"fold_{fold:02d}_{suffix}"
                if not path.is_file():
                    raise FileNotFoundError(
                        f"V11 needs completed fold-matched sources: {path}"
                    )
        groups[fold]={"train":train,"val":val}
    a.output_dir.mkdir(parents=True,exist_ok=True)
    finished,remaining={},{}
    for fold,grp in groups.items():
        fit,monitor,_=previous.split_inner_monitor_months(
            grp["train"],fold,a.seed
        )
        sig=experiment_signature(
            a,fold,
            a.v7b_dir/f"fold_{fold:02d}_final.pt",
            a.v10_dir/f"fold_{fold:02d}_final.pt",
            fit,monitor,grp["val"]
        )
        metrics_path=a.output_dir/f"fold_{fold:02d}_metrics.json"
        final_path=a.output_dir/f"fold_{fold:02d}_final.pt"
        if a.resume and metrics_path.is_file() and final_path.is_file():
            saved=json.loads(metrics_path.read_text(encoding="utf-8"))
            if (saved.get("signature")!=sig or
                    saved.get("architecture")!=
                    "v11-all-candidate-listwise-V7b-V10"):
                raise RuntimeError("V11 completed fold signature mismatch.")
            finished[fold]=saved
            print(
                f"FOLD {fold}: completed signature verified; skip.",
                flush=True
            )
        else:
            remaining[fold]=grp
    if not a.execute:
        print(
            "V11 PREFLIGHT: V7b and V10 paired sources available. "
            "Use --execute to load cached source measurements and train.",
            flush=True
        )
        return
    if remaining:
        unique={
            r["time"]:r for grp in remaining.values()
            for group in grp.values() for r in group
        }
        print(
            f"V11: {len(unique)} measured MAG/SWEA event blocks. "
            "V7b feature cache is per fold and restart-safe; "
            "V10 refiners are frozen.",
            flush=True
        )
        blocks=v7.load_event_blocks(
            list(unique.values()),a
        )
        if blocks is None:
            raise RuntimeError("No source measurements returned.")
        for fold,grp in remaining.items():
            finished[fold]=train_fold(
                fold,grp,blocks,rows,a,device
            )
    summary=summarize(a,finished)
    v7.safe_json(a.output_dir/"cv_summary.json",summary)
    print(
        f"V11 GLOBAL: {summary['n_within_60s']}/"
        f"{summary['n_evaluable_events']} within 60s = "
        f"{summary['fraction_within_60s']:.1%}; "
        f"median={summary['global_median_error_s']:.1f}s. "
        f"V10 same events={summary['paired_v10_within_60s']:.1%}. "
        f"80% target met={summary['target_80pct_met']}.",
        flush=True
    )


if __name__=="__main__":
    main()
