#!/usr/bin/env python3
"""V12: multi-hypothesis MPB localization (frozen V7b + frozen V10).

V7b scans all measured 10-min MAG/SWEA windows within +/-30 min of a
KNOWN catalog MPB. A diverse shortlist of K score-ranked hypotheses
is EACH refined by V10. The NEW permutation-invariant selector learns
which refined time is credible, using measured physics, V7b scores,
V10 confidence/uncertainty and translation-invariant consensus.

Only candidate selection weights are trained; V7b and V10 stay frozen.
Time relative to the catalog MPB enters solely as supervision/evaluation.
Inner TRAIN-month monitor chooses checkpoints and mixture alpha.
Outer 4-fold comparisons are strictly paired against V10/V7b.
Conditional time localization, NOT autonomous full-orbit MPB detection.
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
from src.detection.mpb_listwise_reranker_v11 import physical_candidate_features
from src.detection.mpb_multi_hypothesis_v12 import (
    N_FEATURES,MultiHypothesisSelector,diverse_shortlist,
    assemble_features,fit_feature_norm,normalize,ranking_loss,
    selected_index,theoretical_coverage
)

V7B_DIR=Path("results/metrics/mag_swea_regions_ranking_v1")
V10_DIR=Path("results/metrics/mag_swea_refiner_v10")
OUT_DIR=Path("results/metrics/mag_swea_multi_hypothesis_v12")
BLENDS=(0.,.25,.5,.75,1.)


def args_parser():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument("--manifest",type=Path,default=v7.MANIFEST)
    p.add_argument("--v7b-dir",type=Path,default=V7B_DIR)
    p.add_argument("--v10-dir",type=Path,default=V10_DIR)
    p.add_argument("--output-dir",type=Path,default=OUT_DIR)
    p.add_argument("--mag-cache-dir",type=Path,default=v7.MAG_CACHE)
    p.add_argument("--swea-cache-dir",type=Path,default=v7.SWEA_CACHE)
    p.add_argument("--refresh-swea-cache",action="store_true")
    p.add_argument("--folds",choices=("all","1","2","3","4"),default="all")
    p.add_argument("--pilot-events-per-fold",type=int,default=0)
    p.add_argument("--topk",type=int,default=12)
    p.add_argument("--min-gap-s",type=float,default=120.)
    p.add_argument("--evaluation-stride-s",type=float,default=32.)
    p.add_argument("--max-epochs",type=int,default=65)
    p.add_argument("--min-epochs",type=int,default=8)
    p.add_argument("--patience",type=int,default=12)
    p.add_argument("--batch-size",type=int,default=16)
    p.add_argument("--events-per-update",type=int,default=16)
    p.add_argument("--learning-rate",type=float,default=0.0004)
    p.add_argument("--target-sigma-s",type=float,default=90.)
    p.add_argument("--seed",type=int,default=42)
    p.add_argument("--torch-threads",type=int,default=2)
    p.add_argument("--max-query-gib",type=float,default=2.)
    p.add_argument("--max-total-gib",type=float,default=150.)
    p.add_argument("--max-mag-rows",type=int,default=25000)
    p.add_argument("--max-swea-rows",type=int,default=2000)
    p.add_argument("--resume",action="store_true")
    p.add_argument("--execute",action="store_true")
    a=p.parse_args()
    positive=(
        a.topk,a.min_gap_s,a.evaluation_stride_s,
        a.max_epochs,a.min_epochs,a.patience,a.batch_size,
        a.events_per_update,a.learning_rate,a.target_sigma_s,
        a.torch_threads,a.max_query_gib,a.max_total_gib,
        a.max_mag_rows,a.max_swea_rows
    )
    if (min(positive)<=0 or a.min_epochs>a.max_epochs or
            a.pilot_events_per_fold<0 or
            a.evaluation_stride_s!=32.):
        p.error("Invalid V12 configuration; 32-s V7b grid is required.")
    return a


def digest(*items):
    h=hashlib.sha256()
    for item in items:
        h.update(str(item).encode("utf-8"))
        h.update(b"\x00")
    return h.hexdigest()


def fold_signature(a,fold,source7,source10,fit,monitor,val):
    return digest(
        "V12-frozen-V7b-V10-multihypothesis-v1",
        v7.sha256_file(a.manifest),
        v7.sha256_file(source7),v7.sha256_file(source10),
        [(r["time"].isoformat()) for r in fit],
        [(r["time"].isoformat()) for r in monitor],
        [(r["time"].isoformat()) for r in val],
        a.topk,a.min_gap_s,a.evaluation_stride_s,
        a.max_epochs,a.min_epochs,a.patience,a.batch_size,
        a.events_per_update,a.learning_rate,a.target_sigma_s,
        a.seed,BLENDS,
        v7.sha256_file(
            ROOT/"src/detection/mpb_multi_hypothesis_v12.py"
        ),
        v7.sha256_file(
            ROOT/"scripts/5_training/5h_train_mag_swea_multihypothesis_v12.py"
        ),
    )


def feature_source_sha(a,path7,path10):
    return digest(
        "V12-full-coarse-and-K-refined-v1",
        v7.sha256_file(a.manifest),
        v7.sha256_file(path7),v7.sha256_file(path10),
        v7.sha256_file(
            ROOT/"src/detection/mpb_multi_hypothesis_v12.py"
        ),
        v7.sha256_file(
            ROOT/"src/detection/mpb_listwise_reranker_v11.py"
        ),
        v7.sha256_file(
            ROOT/"src/detection/mpb_region_anchored_refiner.py"
        ),
        v7.sha256_file(
            ROOT/"scripts/5_training/5h_train_mag_swea_multihypothesis_v12.py"
        ),
        a.topk,a.min_gap_s,a.evaluation_stride_s
    )


def feature_path(a,fold,time):
    return (
        a.output_dir/"features"/f"fold_{fold:02d}"/
        (time.strftime("%Y%m%dT%H%M%S%f")+".npz")
    )


def read_feature(path,source_sha):
    if not path.exists():
        return None
    with np.load(path,allow_pickle=False) as data:
        if str(data["source_sha"].item())!=source_sha:
            raise RuntimeError(
                f"V12 cache signature mismatch at {path}. "
                "Use another output directory or clear V12 feature cache."
            )
        record={
            "features":data["features"].astype(np.float32),
            "candidate_offsets":data["candidate_offsets"].astype(np.float32),
            "predicted_offsets":data["predicted_offsets"].astype(np.float32),
            "coarse_teacher":data["coarse_teacher"].astype(np.float32),
            "n_scanned_windows":int(data["n_scanned_windows"].item()),
        }
    if (record["features"].ndim!=2 or
            record["features"].shape[-1]!=N_FEATURES or
            not np.isfinite(record["features"]).all()):
        raise ValueError(f"Corrupt V12 cached feature file: {path}")
    return record


def save_feature(path,record,source_sha):
    path.parent.mkdir(parents=True,exist_ok=True)
    tmp=path.with_suffix(".tmp")
    with tmp.open("wb") as fp:
        np.savez_compressed(
            fp,
            source_sha=np.asarray(source_sha),
            **record,
        )
    os.replace(tmp,path)


@torch.no_grad()
def scan_refined_hypotheses(
    teacher,fine,event,blocks,norm,fine_calibration,a,device,
    mpb_s,bs_s
):
    """Complete V7b scan; topK diversified, THEN V10 fine on each.

    The known MPB timestamp is used to DEFINE the benchmark +/-30-min
    scan and to SAVE labels; no model sees it as a feature.
    """
    when=event["time"]
    mag,swea=blocks[when]["MAG"],blocks[when]["SWEA"]
    grid=candidate_evaluation_grid(
        event,mpb_s,bs_s,step_s=a.evaluation_stride_s
    )
    prepared=[]
    for candidate in grid:
        clip=extract_clip(mag,swea,candidate.center_s)
        if clip is not None:
            prepared.append((candidate,clip))
    if len(prepared)<2:
        return None
    phys=[]
    for i in range(0,len(prepared),a.batch_size):
        sub=prepared[i:i+a.batch_size]
        batch=collate_dual(
            [(clip,0,(0,0)) for _,clip in sub],norm,device
        )
        outputs=teacher(
            batch["mag"],batch["mag_mask"],batch["mag_time"],
            batch["swea"],batch["swea_mask"],batch["swea_time"]
        )
        phys.append(physical_candidate_features(batch,outputs))
    phys=np.concatenate(phys,axis=0)
    centers=np.asarray([
        candidate.center_s for candidate,_ in prepared
    ],dtype=np.float64)
    shortlist=diverse_shortlist(
        phys[:,0],centers,topk=a.topk,min_gap_s=a.min_gap_s
    )
    selected_phys=phys[shortlist]
    selected_centers=centers[shortlist]
    mean_delta=[]
    peak_delta=[]
    presence_logit=[]
    entropy=[]
    peak_prob=[]
    std_seconds=[]
    for i in range(0,len(shortlist),a.batch_size):
        ids=shortlist[i:i+a.batch_size]
        batch=collate_dual(
            [(prepared[int(j)][1],0,(0,0)) for j in ids],
            norm,device
        )
        outputs=fine(
            batch["mag"],batch["mag_mask"],batch["mag_time"],
            batch["swea"],batch["swea_mask"],batch["swea_time"]
        )
        prob=outputs["time_prob"]
        t=batch["mag_time"]
        expectation=outputs["expected_offset_s"]
        # Normalized entropy, max probability and temporal spread are
        # measured from the frozen V10 distribution, never targets.
        count=(batch["mag_mask"] &
               (t.abs()<=fine.max_offset_s)).sum(-1).clamp_min(2)
        ent=-(prob*prob.clamp_min(1e-12).log()).sum(-1)/count.log()
        variance=(prob*(t-expectation[:,None]).square()).sum(-1)
        mean_delta.extend(
            expectation.cpu().numpy().astype(float)
        )
        peak_delta.extend(
            outputs["peak_offset_s"].cpu().numpy().astype(float)
        )
        presence_logit.extend(
            outputs["presence_logit"].cpu().numpy().astype(float)
        )
        entropy.extend(ent.cpu().numpy().astype(float))
        peak_prob.extend(prob.max(-1).values.cpu().numpy().astype(float))
        std_seconds.extend(
            variance.clamp_min(0).sqrt().cpu().numpy().astype(float)
        )
    features,predicted_t=assemble_features(
        selected_phys,selected_centers,
        np.asarray(mean_delta),np.asarray(peak_delta),
        np.asarray(presence_logit),np.asarray(entropy),
        np.asarray(peak_prob),np.asarray(std_seconds),
        fine_alpha=float(fine_calibration["alpha"]),
        fine_method=str(fine_calibration["method"])
    )
    return {
        "features":features,
        "candidate_offsets":(selected_centers-when.timestamp()).astype(
            np.float32
        ),
        # TRUE label offsets: NEVER fed into candidate neural net.
        "predicted_offsets":(predicted_t-when.timestamp()).astype(
            np.float32
        ),
        "coarse_teacher":selected_phys[:,0].astype(np.float32),
        "n_scanned_windows":len(prepared),
    }


def load_fold_features(
    teacher,fine,fold,events,blocks,norm,fine_calibration,
    a,device,mpb_s,bs_s,source_sha
):
    records={}
    hit=generated=missing=0
    for i,row in enumerate(events,1):
        path=feature_path(a,fold,row["time"])
        data=read_feature(path,source_sha)
        if data is None:
            data=scan_refined_hypotheses(
                teacher,fine,row,blocks,norm,fine_calibration,
                a,device,mpb_s,bs_s
            )
            if data is None:
                missing+=1
                continue
            save_feature(path,data,source_sha)
            generated+=1
        else:
            hit+=1
        records[row["time"]]=data
        if i%50==0 or i==len(events):
            print(
                f"FOLD {fold} V12 CACHE {i}/{len(events)} "
                f"reused={hit} new={generated} missing={missing}.",
                flush=True
            )
    return records


def predict_scores(model,record,norm,device):
    x=normalize(record,norm,device)
    return model(x)


def best_error(record,scores,alpha):
    j=selected_index(
        scores,record["coarse_teacher"],alpha
    )
    return float(abs(record["predicted_offsets"][j])),j


def train_epoch(model,opt,train_rows,records,norm,a,fold,epoch,device):
    active=[r for r in train_rows if r["time"] in records]
    if not active:
        raise RuntimeError("No V12 TRAIN records available.")
    rng=random.Random(a.seed+fold*10000+epoch)
    rng.shuffle(active)
    model.train()
    opt.zero_grad(set_to_none=True)
    n=0
    loss_sum=0.
    pending=0
    for i,row in enumerate(active,1):
        record=records[row["time"]]
        scores=predict_scores(model,record,norm,device)
        err=np.abs(record["predicted_offsets"])
        loss=ranking_loss(
            scores,err,sigma_s=a.target_sigma_s
        )
        if not bool(torch.isfinite(loss)):
            raise FloatingPointError("Nonfinite V12 loss.")
        (loss/a.events_per_update).backward()
        loss_sum+=float(loss.detach())
        n+=1
        pending+=1
        if pending==a.events_per_update or i==len(active):
            torch.nn.utils.clip_grad_norm_(model.parameters(),1.)
            opt.step()
            opt.zero_grad(set_to_none=True)
            pending=0
    return {"train_loss":loss_sum/n,"train_events":n}


@torch.no_grad()
def evaluate_monitor(model,monitor,records,norm,a,device):
    model.eval()
    losses=[]
    errors=[]
    for row in monitor:
        record=records.get(row["time"])
        if record is None:
            continue
        scores=predict_scores(model,record,norm,device)
        error,j=best_error(
            record,scores.cpu().numpy(),1.
        )
        losses.append(float(ranking_loss(
            scores,np.abs(record["predicted_offsets"]),
            sigma_s=a.target_sigma_s
        )))
        errors.append(error)
    if not errors:
        raise RuntimeError("No V12 inner monitor records.")
    err=np.asarray(errors,dtype=float)
    return {
        "loss":float(np.mean(losses)),
        "median_error_s":float(np.median(err)),
        "clipped_mean_error_s":float(np.minimum(err,600).mean()),
        "within_60s":float(np.mean(err<=60.)),
        "events":len(err),
    }


@torch.no_grad()
def calibrate_alpha(model,monitor,records,norm,device):
    model.eval()
    outcomes=[]
    for alpha in BLENDS:
        errors=[]
        for row in monitor:
            rec=records.get(row["time"])
            if rec is None:
                continue
            scores=predict_scores(
                model,rec,norm,device
            ).cpu().numpy()
            error,_=best_error(rec,scores,alpha)
            errors.append(error)
        if not errors:
            raise RuntimeError("V12 cannot calibrate empty monitor.")
        x=np.asarray(errors,dtype=float)
        outcomes.append({
            "alpha":float(alpha),"n":len(x),
            "within_60s":float(np.mean(x<=60.)),
            "median_error_s":float(np.median(x)),
            "clipped_mean_error_s":float(np.minimum(x,600).mean())
        })
    selected=max(outcomes,key=lambda r:(
        r["within_60s"],-r["clipped_mean_error_s"],
        -r["median_error_s"],-r["alpha"]
    ))
    return {"selected":selected,"trials":outcomes}


@torch.no_grad()
def final_predictions(model,rows,records,norm,alpha,device):
    model.eval()
    out=[]
    for row in rows:
        rec=records.get(row["time"])
        if rec is None:
            raise RuntimeError(
                "V12 holdout missing measured candidate windows."
            )
        score=predict_scores(model,rec,norm,device).cpu().numpy()
        err,j=best_error(rec,score,alpha)
        out.append({
            "mpb_timestamp_utc":row["time"].isoformat(),
            "source_group":row["source_group"],
            "n_scanned_windows":rec["n_scanned_windows"],
            "n_shortlisted":len(rec["coarse_teacher"]),
            "chosen_hypothesis_rank":j+1,
            "candidate_offset_s":float(rec["candidate_offsets"][j]),
            "selected_offset_s":float(rec["predicted_offsets"][j]),
            "absolute_error_s":float(err),
            "shortlist_nearest_refined_s":float(
                np.min(np.abs(rec["predicted_offsets"]))
            ),
            "shortlist_nearest_coarse_s":float(
                np.min(np.abs(rec["candidate_offsets"]))
            )
        })
    return out


def paired_metrics(predicted,base10_csv,base7_csv,alpha):
    base10=v10.load_v7b_reference(base10_csv)
    base7=v10.load_v7b_reference(base7_csv)
    result=[]
    for row in predicted:
        when=row["mpb_timestamp_utc"]
        r10=base10.get(when)
        r7=base7.get(when)
        if r10 is None or r7 is None:
            raise RuntimeError(f"Missing matched V10/V7b event: {when}")
        old_coarse=float(r7["peak_offset_s"])
        old10=float(r10["absolute_error_s"])
        old7=float(r7["absolute_error_s"])
        # Shortlist first item MUST reproduce the V7b coarse peak.
        # (Verifying the full shortlist scan happens in train_fold.)
        if alpha==0.:
            # Preserve the literal V10 CSV prediction; eliminates any
            # batching/roundoff discrepancy in this exact fallback.
            row={**row,
                 "selected_offset_s":float(r10["selected_offset_s"]),
                 "absolute_error_s":old10}
        result.append({
            **row,
            "v7b_coarse_offset_s":old_coarse,
            "v10_absolute_error_s":old10,
            "v7b_absolute_error_s":old7,
            "delta_v12_minus_v10_s":row["absolute_error_s"]-old10,
            "delta_v12_minus_v7b_s":row["absolute_error_s"]-old7,
        })
    new=np.asarray([r["absolute_error_s"] for r in result],dtype=float)
    old10=np.asarray([r["v10_absolute_error_s"] for r in result])
    old7=np.asarray([r["v7b_absolute_error_s"] for r in result])
    return {
        "val_evaluable_events":len(result),
        "median_error_s":float(np.median(new)),
        "mean_error_s":float(np.mean(new)),
        "p90_error_s":float(np.percentile(new,90)),
        "fraction_within_60s":float(np.mean(new<=60.)),
        "fraction_within_120s":float(np.mean(new<=120.)),
        "fraction_within_300s":float(np.mean(new<=300.)),
        "paired_v10_median_s":float(np.median(old10)),
        "paired_v10_within_60s":float(np.mean(old10<=60.)),
        "paired_v7b_within_60s":float(np.mean(old7<=60.)),
        "improved_vs_v10":int(np.sum(new<old10)),
        "worsened_vs_v10":int(np.sum(new>old10)),
        "tied_vs_v10":int(np.sum(new==old10)),
        "oracle_shortlist_refined_within_60s":float(np.mean([
            r["shortlist_nearest_refined_s"]<=60. for r in result
        ])),
        "oracle_shortlist_coarse_within_240s":float(np.mean([
            r["shortlist_nearest_coarse_s"]<=240. for r in result
        ])),
        "target_80pct_met":bool(np.mean(new<=60.)>=.8),
    },result


def train_fold(fold,selection,blocks,all_rows,a,device):
    fit,monitor,removed=previous.split_inner_monitor_months(
        selection["train"],fold,a.seed
    )
    outer=selection["val"]
    path7=a.v7b_dir/f"fold_{fold:02d}_final.pt"
    path10=a.v10_dir/f"fold_{fold:02d}_final.pt"
    src7=torch.load(path7,map_location="cpu",weights_only=False)
    src10=torch.load(path10,map_location="cpu",weights_only=False)
    if (src7.get("fold")!=fold or src10.get("fold")!=fold or
            "model" not in src7 or "model" not in src10 or
            "calibration" not in src10):
        raise RuntimeError("V12 source checkpoint fold mismatch.")
    norm=v9.normalizers_from_v7b(src7)
    fine_cal=src10["calibration"]
    teacher=MagSweaRegionAutoencoder().to(device)
    teacher.load_state_dict(src7["model"])
    teacher.eval().requires_grad_(False)
    fine=RegionAnchoredTemporalRefiner(
        src7["model"]
    ).to(device)
    fine.load_state_dict(src10["model"],strict=True)
    fine.eval().requires_grad_(False)
    mpb_s=np.asarray([
        r["time"].timestamp() for r in all_rows
    ],dtype=np.float64)
    bs_s=np.asarray([
        r["bs_time"].timestamp() for r in all_rows
    ],dtype=np.float64)
    sig=fold_signature(a,fold,path7,path10,fit,monitor,outer)
    source_sha=feature_source_sha(a,path7,path10)
    union={r["time"]:r for r in fit+monitor+outer}
    records=load_fold_features(
        teacher,fine,fold,list(union.values()),blocks,
        norm,fine_cal,a,device,mpb_s,bs_s,source_sha
    )
    del teacher,fine
    if not records:
        raise RuntimeError("No V12 candidate data.")
    if any(r["time"] not in records for r in outer):
        raise RuntimeError("V12 incomplete outer-fold data: do not drop events.")
    # Exact teacher coarse parity check BEFORE trying any new selection.
    reference=v10.load_v7b_reference(
        a.v7b_dir/f"fold_{fold:02d}_event_metrics.csv"
    )
    for row in outer:
        record=records[row["time"]]
        old=reference.get(row["time"].isoformat())
        if old is None:
            raise RuntimeError("V12 cannot match original V7b event.")
        observed=float(record["candidate_offsets"][0])
        expected=float(old["peak_offset_s"])
        if abs(observed-expected)>.1:
            raise RuntimeError(
                f"V12 V7b fallback mismatch for {row['time']}: "
                f"new {observed}, V7b {expected}. "
                "Check V12 cached feature provenance and scan order."
            )
    train_data=[
        records[r["time"]] for r in fit if r["time"] in records
    ]
    feat_norm=fit_feature_norm(train_data)
    print(
        f"FOLD {fold}: TRAIN={len(fit)} "
        f"available={len(train_data)}; INNER={len(monitor)} "
        f"available={sum(r['time'] in records for r in monitor)}; "
        f"OUTER={len(outer)}, excluded_overlaps={removed}.",
        flush=True
    )
    if sum(r["time"] in records for r in monitor)<1:
        raise RuntimeError("V12 missing all inner-monitor event scans.")
    train_coverage=theoretical_coverage(train_data)
    print(
        f"FOLD {fold}: TRAIN shortlist oracle <=60s={train_coverage:.1%}; "
        "this is an upper bound, NOT observed model accuracy.",
        flush=True
    )
    torch.manual_seed(a.seed+fold)
    model=MultiHypothesisSelector().to(device)
    optimizer=torch.optim.AdamW(
        model.parameters(),lr=a.learning_rate,weight_decay=1e-3
    )
    best=a.output_dir/f"fold_{fold:02d}_best.pt"
    last=a.output_dir/f"fold_{fold:02d}_last.pt"
    final=a.output_dir/f"fold_{fold:02d}_final.pt"
    start=best_epoch=stale=0
    best_metric=None
    history=[]
    if last.is_file():
        if not a.resume:
            raise RuntimeError(f"{last} exists; pass --resume.")
        previous_state=torch.load(last,map_location=device,weights_only=False)
        if previous_state.get("signature")!=sig:
            raise RuntimeError("V12 resume signature mismatch.")
        model.load_state_dict(previous_state["model"])
        optimizer.load_state_dict(previous_state["optimizer"])
        start=int(previous_state["epoch"])
        best_epoch=int(previous_state["best_epoch"])
        stale=int(previous_state["stale"])
        best_metric=previous_state["best_metric"]
        history=previous_state["history"]
        print(
            f"FOLD {fold}: reanudando desde época {start}.",flush=True
        )
    for epoch in range(start+1,a.max_epochs+1):
        if epoch>=a.min_epochs and stale>=a.patience:
            break
        tr=train_epoch(
            model,optimizer,fit,records,feat_norm,
            a,fold,epoch,device
        )
        mon=evaluate_monitor(
            model,monitor,records,feat_norm,a,device
        )
        # Objective order reflects requested <=60s criterion. Then
        # favor smaller clipped MAE and smaller cross entropy.
        monitor_key=(
            mon["within_60s"],
            -mon["clipped_mean_error_s"],
            -mon["loss"],
        )
        if epoch>=a.min_epochs:
            if best_metric is None or monitor_key>best_metric:
                best_metric=monitor_key
                best_epoch=epoch
                stale=0
                v7.safe_checkpoint(best,{
                    "signature":sig,"epoch":epoch,
                    "model":model.state_dict()
                })
            else:
                stale+=1
        history.append({
            "epoch":epoch,"train_loss":tr["train_loss"],
            "val_loss":mon["loss"],
            "val_within_60s":mon["within_60s"],
            "val_median_s":mon["median_error_s"],
            "val_clipped_mae_s":mon["clipped_mean_error_s"],
            "best_epoch":best_epoch,"stale":stale
        })
        v7.safe_checkpoint(last,{
            "signature":sig,"epoch":epoch,
            "best_epoch":best_epoch,"best_metric":best_metric,
            "stale":stale,"history":history,
            "model":model.state_dict(),
            "optimizer":optimizer.state_dict(),
        })
        print(
            f"FOLD {fold} EPOCH {epoch}: "
            f"train_loss={tr['train_loss']:.4f} "
            f"val_loss={mon['loss']:.4f} "
            f"val_within60={mon['within_60s']:.1%} "
            f"val_med={mon['median_error_s']:.1f}s "
            f"best={best_epoch} stale={stale}.",
            flush=True
        )
        if epoch>=a.min_epochs and stale>=a.patience:
            print(
                f"FOLD {fold}: early stopping; best epoch {best_epoch}.",
                flush=True
            )
            break
    if not best.is_file():
        raise RuntimeError("V12 missing best checkpoint.")
    selection=torch.load(best,map_location=device,weights_only=False)
    if selection["signature"]!=sig:
        raise RuntimeError("V12 selected checkpoint signature mismatch.")
    model.load_state_dict(selection["model"])
    calibration=calibrate_alpha(
        model,monitor,records,feat_norm,device
    )
    alpha=float(calibration["selected"]["alpha"])
    print(
        f"FOLD {fold}: MONITOR chose alpha={alpha:.2f}, "
        f"<=60s={calibration['selected']['within_60s']:.1%} "
        f"n={calibration['selected']['n']}.",
        flush=True
    )
    outer_predictions=final_predictions(
        model,outer,records,feat_norm,alpha,device
    )
    metrics,events=paired_metrics(
        outer_predictions,
        a.v10_dir/f"fold_{fold:02d}_event_metrics.csv",
        a.v7b_dir/f"fold_{fold:02d}_event_metrics.csv",
        alpha,
    )
    report={
        "schema":1,
        "architecture":"v12-diverse-multihypothesis-refined-selector",
        "fold":fold,"signature":sig,
        "train_total_events":len(fit),
        "monitor_total_events":len(monitor),
        "val_total_events":len(outer),
        "best_epoch":best_epoch,"epochs_completed":len(history),
        "source_v7b":str(path7),"source_v10":str(path10),
        "shortlist_topk":a.topk,
        "min_gap_s":a.min_gap_s,
        "inner_monitor_selection":calibration,
        "v10_fine_calibration":fine_cal,
        "history":history,
        "warning":(
            "Conditional MPB timing within a catalog-centered +/-30-min "
            "search. Oracle ceiling uses actual MPB time only for "
            "diagnostics, never neural-network input or calibration."
        ),
        **metrics,
    }
    v7.safe_json(
        a.output_dir/f"fold_{fold:02d}_metrics.json",report
    )
    v7.write_csv(
        a.output_dir/f"fold_{fold:02d}_event_metrics.csv",events
    )
    v7.safe_checkpoint(final,{
        "schema":1,"fold":fold,"signature":sig,
        "architecture":report["architecture"],
        "best_epoch":best_epoch,
        "model":model.state_dict(),
        "feature_normalizer":{
            k:v.tolist() for k,v in feat_norm.items()
        },
        "inner_monitor_selection":calibration,
        "v10_fine_calibration":fine_cal,
    })
    print(
        f"FOLD {fold}: median={metrics['median_error_s']:.1f}s "
        f"within 60s={metrics['fraction_within_60s']:.1%} "
        f"V10={metrics['paired_v10_within_60s']:.1%} "
        f"improved={metrics['improved_vs_v10']}/{len(events)} "
        f"alpha={alpha:.2f}; "
        f"ORACLE shortlist<=60s="
        f"{metrics['oracle_shortlist_refined_within_60s']:.1%}.",
        flush=True
    )
    return report


def global_summary(a,reports):
    all_rows=[]
    for fold in sorted(reports):
        path=a.output_dir/f"fold_{fold:02d}_event_metrics.csv"
        with path.open(newline="",encoding="utf-8") as fp:
            all_rows.extend(csv.DictReader(fp))
    v12=np.asarray([
        float(r["absolute_error_s"]) for r in all_rows
    ],dtype=float)
    v10old=np.asarray([
        float(r["v10_absolute_error_s"]) for r in all_rows
    ],dtype=float)
    v7old=np.asarray([
        float(r["v7b_absolute_error_s"]) for r in all_rows
    ],dtype=float)
    oracle=np.asarray([
        float(r["shortlist_nearest_refined_s"]) for r in all_rows
    ])
    if len(v12)==0:
        raise RuntimeError("No V12 fold evaluation produced.")
    return {
        "schema":1,
        "architecture":"v12-diverse-multihypothesis-refined-selector",
        "folds_completed":sorted(reports),
        "pilot_subset":bool(a.pilot_events_per_fold),
        "n_evaluable_events":len(v12),
        "n_within_60s":int(np.sum(v12<=60.)),
        "fraction_within_60s":float(np.mean(v12<=60.)),
        "global_median_error_s":float(np.median(v12)),
        "global_mean_error_s":float(np.mean(v12)),
        "global_p90_error_s":float(np.percentile(v12,90)),
        "fraction_within_120s":float(np.mean(v12<=120.)),
        "fraction_within_300s":float(np.mean(v12<=300.)),
        "paired_v10_median_s":float(np.median(v10old)),
        "paired_v10_within_60s":float(np.mean(v10old<=60.)),
        "paired_v7b_within_60s":float(np.mean(v7old<=60.)),
        "improved_vs_v10":int(np.sum(v12<v10old)),
        "worsened_vs_v10":int(np.sum(v12>v10old)),
        "tied_vs_v10":int(np.sum(v12==v10old)),
        "oracle_shortlist_refined_within_60s":float(
            np.mean(oracle<=60.)
        ),
        "target_80pct_met":bool(np.mean(v12<=60.)>=.8),
        "fold_results":[{
            "fold":f,
            "n_events":reports[f]["val_evaluable_events"],
            "median_error_s":reports[f]["median_error_s"],
            "within_60s":reports[f]["fraction_within_60s"],
            "v10_within_60s":reports[f]["paired_v10_within_60s"],
            "oracle_refined_within_60s":
                reports[f]["oracle_shortlist_refined_within_60s"],
            "alpha":reports[f][
                "inner_monitor_selection"
            ]["selected"]["alpha"],
            "best_epoch":reports[f]["best_epoch"]
        } for f in sorted(reports)],
        "warning":(
            "Known-MPB-conditioned experiment; full orbit MPB detection "
            "not tested. V7b/V10 outer folds have already influenced "
            "experiment design; unbiased final conclusions require "
            "new untouched time-separated events."
        )
    }


def main():
    a=args_parser()
    torch.set_num_threads(a.torch_threads)
    torch.manual_seed(a.seed)
    random.seed(a.seed)
    np.random.seed(a.seed)
    device=torch.device("cpu")
    rows=pilot.read_manifest(a.manifest)
    for row in rows:
        if not row.get("bs_timestamp_utc"):
            raise RuntimeError("V12 requires paired catalog BS for QC.")
        row["bs_time"]=bq.to_utc(row["bs_timestamp_utc"])
    folds=[1,2,3,4] if a.folds=="all" else [int(a.folds)]
    groups={}
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
        if not val or len(train)<3:
            raise RuntimeError("V12 insufficient events for this fold.")
        previous.check_disjoint_supports(train,val)
        previous.split_inner_monitor_months(train,fold,a.seed)
        for directory in (a.v7b_dir,a.v10_dir):
            for suffix in ("final.pt","event_metrics.csv"):
                source=directory/f"fold_{fold:02d}_{suffix}"
                if not source.is_file():
                    raise FileNotFoundError(
                        f"V12 needs complete V7b/V10 source: {source}"
                    )
        groups[fold]={"train":train,"val":val}
    a.output_dir.mkdir(parents=True,exist_ok=True)
    finished={}
    to_do={}
    for fold,group in groups.items():
        fit,monitor,_=previous.split_inner_monitor_months(
            group["train"],fold,a.seed
        )
        sign=fold_signature(
            a,fold,
            a.v7b_dir/f"fold_{fold:02d}_final.pt",
            a.v10_dir/f"fold_{fold:02d}_final.pt",
            fit,monitor,group["val"]
        )
        report=a.output_dir/f"fold_{fold:02d}_metrics.json"
        final=a.output_dir/f"fold_{fold:02d}_final.pt"
        if a.resume and report.is_file() and final.is_file():
            data=json.loads(report.read_text(encoding="utf-8"))
            if data.get("signature")!=sign:
                raise RuntimeError(
                    f"V12 completed fold {fold} configuration changed."
                )
            finished[fold]=data
            print(
                f"FOLD {fold}: completed with matching signature; skip.",
                flush=True
            )
        else:
            to_do[fold]=group
    if not a.execute:
        print(
            "V12 PREFLIGHT: V7b+V10 fold checkpoints and CSVs exist; "
            "pass --execute for full measured feature extraction.",
            flush=True
        )
        return
    if to_do:
        unique={
            r["time"]:r for groupset in to_do.values()
            for group in groupset.values() for r in group
        }
        print(
            f"V12: {len(unique)} unique MAG/SWEA catalog event blocks. "
            "Teacher+fine frozen; restart-safe per-fold refined shortlist "
            "feature caches.",
            flush=True
        )
        blocks=v7.load_event_blocks(list(unique.values()),a)
        if blocks is None:
            raise RuntimeError("No event sources.")
        for fold,group in to_do.items():
            finished[fold]=train_fold(
                fold,group,blocks,rows,a,device
            )
    result=global_summary(a,finished)
    v7.safe_json(a.output_dir/"cv_summary.json",result)
    print(
        f"V12 GLOBAL: {result['n_within_60s']}/"
        f"{result['n_evaluable_events']} = "
        f"{result['fraction_within_60s']:.1%} within60; "
        f"V10 matched={result['paired_v10_within_60s']:.1%}; "
        f"median={result['global_median_error_s']:.1f}s; "
        f"oracle shortlist ceiling="
        f"{result['oracle_shortlist_refined_within_60s']:.1%}; "
        f"80% goal met={result['target_80pct_met']}.",
        flush=True
    )


if __name__=="__main__":
    main()
