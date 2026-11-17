#!/usr/bin/env python3
"""V10 MAG GROUP SPECIALISTS — FOLD 4 ONLY.

Train two INDEPENDENT V10 refiners, one for each previously defined V13
MAG cluster (1,2), using only events from folds 1-3 belonging to that
group. A frozen V7b FOLD-4 teacher, its source normalizers, 10-minute
MAG+SWEA native-cadence clips, loss and coarse search are unchanged
from the original V10 implementation.

For each group, choose a group-specific month-block FIT/MONITOR split
using ONLY folds 1-3. A small V13 monitor (only two events for MAG1)
must not be reused as a calibration set. Train/monitor split and model
selection are kept separate from the fold-4 evaluation.

IMPORTANT: the V13 group labels were obtained using catalog-centered
positive crossing windows. Routing fold-4 events by these labels is an
ORACLE, retrospective group assignment and is NOT deployable until
group prediction can work without knowing the actual MPB time.
No 80%-within-60s result is claimed before external evaluation.
"""
from __future__ import annotations

import argparse
import csv
import hashlib
import json
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
previous=import_module("scripts.5_training.5a_train_mag_cv")
pilot=import_module("scripts.3_windowing.3b_bigquery_window_pilot")
bq=import_module("scripts.1_bigquery.1c_bigquery_event_pilot")

from src.detection.mpb_region_anchored_refiner import (
    RegionAnchoredTemporalRefiner
)

GROUP_CSV=Path(
    "results/metrics/mag_autoencoder_clusters_v13/crossing_latents_groups.csv"
)
V7B=Path("results/metrics/mag_swea_regions_ranking_v1")
V10=Path("results/metrics/mag_swea_refiner_v10")
OUT=Path("results/metrics/v10_mag_group_specialists_fold4")
FOLD=4
GROUPS=(1,2)
ARCHITECTURE="v10-mag-group-specific-fold4-v1"


def parse_args():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument("--manifest",type=Path,default=v7.MANIFEST)
    p.add_argument("--mag-groups-csv",type=Path,default=GROUP_CSV)
    p.add_argument("--pretrained-dir",type=Path,default=V7B)
    p.add_argument("--baseline-v10-dir",type=Path,default=V10)
    p.add_argument("--output-dir",type=Path,default=OUT)
    p.add_argument("--mag-cache-dir",type=Path,default=v7.MAG_CACHE)
    p.add_argument("--swea-cache-dir",type=Path,default=v7.SWEA_CACHE)
    p.add_argument("--refresh-swea-cache",action="store_true")
    p.add_argument("--groups",choices=("all","1","2"),default="all")
    p.add_argument("--max-epochs",type=int,default=45)
    p.add_argument("--min-epochs",type=int,default=10)
    p.add_argument("--patience",type=int,default=9)
    p.add_argument("--batch-size",type=int,default=16)
    p.add_argument("--torch-threads",type=int,default=2)
    p.add_argument("--seed",type=int,default=42)
    p.add_argument("--learning-rate",type=float,default=.0002)
    p.add_argument("--localization-weight",type=float,default=1.5)
    p.add_argument("--offset-weight",type=float,default=.5)
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
        a.max_query_gib,a.max_total_gib,
        a.max_mag_rows,a.max_swea_rows,
    )
    if (min(positive)<=0 or a.offset_weight<0 or
            a.min_epochs>a.max_epochs or a.evaluation_stride_s!=32.):
        p.error("Invalid V10 expert configuration.")
    return a


def read_mag_groups(path):
    """Load V13 labels using exact UTC event time and original fold."""
    if not path.is_file():
        raise FileNotFoundError(
            f"V13 MAG cluster CSV not found: {path}"
        )
    records={}
    with path.open(newline="",encoding="utf-8") as fp:
        for row in csv.DictReader(fp):
            when=row["mpb_timestamp_utc"]
            if when in records:
                raise RuntimeError(f"Duplicate V13 cluster timestamp: {when}")
            raw=row["cluster"]
            if raw in ("",None,"NaN","nan"):
                label=None
            else:
                numeric=float(raw)
                if not np.isfinite(numeric) or numeric not in (1.,2.):
                    raise RuntimeError(
                        f"Unexpected V13 MAG cluster {raw} at {when}."
                    )
                label=int(numeric)
            records[when]={
                "group":label,
                "fold":int(row["fold"]),
                "split":str(row["split"]),
            }
    return records


def grouped_events(events,labels):
    subsets={1:[],2:[]}
    unavailable=[]
    for event in events:
        key=event["time"].isoformat()
        item=labels.get(key)
        if item is None:
            raise RuntimeError(
                f"Missing exact MAG grouping for {key}."
            )
        if item["fold"]!=event["fold"]:
            raise RuntimeError(
                f"Manifest/V13 fold mismatch at {key}."
            )
        if event["fold"]==FOLD and item["split"]!="outer_holdout":
            raise RuntimeError(
                "The V13 MAG holdout MUST be fold 4. "
                f"Found {item['split']} for {key}."
            )
        if item["group"] is None:
            unavailable.append(event)
        else:
            subsets[item["group"]].append(event)
    expected={event["time"] for event in events}
    recovered={
        event["time"]
        for group in GROUPS for event in subsets[group]
    }|{event["time"] for event in unavailable}
    if recovered!=expected or sum(map(len,subsets.values()))+len(
        unavailable
    )!=len(events):
        raise RuntimeError("MAG grouping lost or duplicated events.")
    return subsets,unavailable


def inner_group_split(events,group,seed):
    """Month-block split selected using only group TRAIN events.

    V13's original 41-event inner monitor has only 2 MAG1 examples;
    therefore calibrating a group-1 model directly on that monitor
    would be unstable. This split reselects month blocks in folds 1-3.
    """
    if len(events)<16:
        raise RuntimeError(
            f"MAG group {group}: {len(events)} TRAIN events is too few."
        )
    valid=[]
    for trial in range(120):
        trial_seed=seed+group*1009+trial*43
        try:
            fit,mon,excluded=previous.split_inner_monitor_months(
                events,FOLD,trial_seed
            )
        except ValueError:
            continue
        if len(fit)<12 or len(mon)<5:
            continue
        objective=(
            abs(len(mon)/len(events)-.15),
            abs(len(mon)-max(5,round(len(events)*.15))),
            trial,
        )
        valid.append((objective,fit,mon,excluded,trial_seed))
    if not valid:
        raise RuntimeError(
            f"Cannot find >=5 month-separated MONITOR events in "
            f"MAG group {group}. Inspect group monthly coverage."
        )
    _,fit,mon,excluded,trial_seed=min(valid,key=lambda x:x[0])
    previous.check_disjoint_supports(fit,mon)
    return fit,mon,excluded,trial_seed


def stem(a,group):
    return a.output_dir/f"fold_04_maggroup_{group}"


def source_files(a):
    return (
        a.pretrained_dir/"fold_04_final.pt",
        a.pretrained_dir/"fold_04_event_metrics.csv",
        a.baseline_v10_dir/"fold_04_event_metrics.csv",
    )


def experiment_signature(a,group,source_path,fit,monitor,outer,split_seed):
    spec={
        "architecture":ARCHITECTURE,"fold":FOLD,"group":group,
        "manifest_sha":v7.sha256_file(a.manifest),
        "group_sha":v7.sha256_file(a.mag_groups_csv),
        "teacher_sha":v7.sha256_file(source_path),
        "source_v10_code_sha":v7.sha256_file(
            ROOT/"scripts/5_training/5f_train_mag_swea_refiner_v10.py"
        ),
        "architecture_code_sha":v7.sha256_file(
            ROOT/"src/detection/mpb_region_anchored_refiner.py"
        ),
        "fit":[r["time"].isoformat() for r in fit],
        "monitor":[r["time"].isoformat() for r in monitor],
        "holdout":[r["time"].isoformat() for r in outer],
        "inner_seed":split_seed,
        "epochs":(a.max_epochs,a.min_epochs,a.patience),
        "batch_size":a.batch_size,"lr":a.learning_rate,
        "loss_weights":(
            a.localization_weight,a.offset_weight,a.gaussian_sigma_s
        ),
        "sampling":(
            a.train_positives_per_event,a.train_negatives_per_event
        ),
        "scan_step":a.evaluation_stride_s,
        "seed":a.seed,
    }
    return hashlib.sha256(
        json.dumps(spec,sort_keys=True).encode()
    ).hexdigest()


def paired_with_v10(results,baseline_path,group):
    """Compare identical fold-4 events with unchanged V10 baseline."""
    baseline=v10.load_v7b_reference(baseline_path)
    paired=[]
    for row in results:
        ref=baseline.get(row["mpb_timestamp_utc"])
        if ref is None:
            raise RuntimeError(
                f"Missing V10 fold4 event {row['mpb_timestamp_utc']}."
            )
        old=float(ref["absolute_error_s"])
        new=float(row["absolute_error_s"])
        paired.append({
            **row,
            "mag_cluster":group,
            "v10_absolute_error_s":old,
            "delta_grouped_minus_v10_s":new-old,
        })
    if not paired:
        raise RuntimeError("Empty MAG group holdout evaluation.")
    curr=np.asarray([x["absolute_error_s"] for x in paired])
    old=np.asarray([x["v10_absolute_error_s"] for x in paired])
    metrics={
        "n_events":len(paired),
        "median_error_s":float(np.median(curr)),
        "mean_error_s":float(np.mean(curr)),
        "p90_error_s":float(np.percentile(curr,90)),
        "within_60s":float(np.mean(curr<=60.)),
        "within_120s":float(np.mean(curr<=120.)),
        "within_300s":float(np.mean(curr<=300.)),
        "baseline_v10_median_error_s":float(np.median(old)),
        "baseline_v10_within_60s":float(np.mean(old<=60.)),
        "baseline_v10_within_300s":float(np.mean(old<=300.)),
        "improved_vs_v10":int(np.sum(curr<old)),
        "worsened_vs_v10":int(np.sum(curr>old)),
        "tied_vs_v10":int(np.sum(curr==old)),
        "delta_within_60_pp":float(
            100*(np.mean(curr<=60.)-np.mean(old<=60.))
        ),
    }
    return metrics,paired


def run_one_group(a,group,fit,monitor,outer,blocks,rows,device,split_seed):
    src7,src7csv,baseline=source_files(a)
    state=torch.load(src7,map_location="cpu",weights_only=False)
    if (state.get("fold")!=FOLD or "model" not in state
            or "normalizers" not in state):
        raise RuntimeError("Missing matching FOLD 4 V7b teacher.")
    norm=v9.normalizers_from_v7b(state)
    signature=experiment_signature(
        a,group,src7,fit,monitor,outer,split_seed
    )
    t_mpb=np.asarray([
        r["time"].timestamp() for r in rows
    ],dtype=np.float64)
    t_bs=np.asarray([
        r["bs_time"].timestamp() for r in rows
    ],dtype=np.float64)
    fit_pool,fit_skips=v9.gather_training_pools(
        fit,blocks,t_mpb,t_bs
    )
    monitor_pool,monitor_skips=v9.gather_training_pools(
        monitor,blocks,t_mpb,t_bs
    )
    if len(fit_pool)<10 or len(monitor_pool)<3:
        raise RuntimeError(
            f"MAG group {group} lacks enough measured FIT/MONITOR "
            f"positive and negative clips: {len(fit_pool)}/"
            f"{len(monitor_pool)}."
        )
    print(
        f"FOLD 4 GROUP {group}: fit={len(fit)} "
        f"usable={len(fit_pool)}, monitor={len(monitor)} "
        f"usable={len(monitor_pool)}, holdout={len(outer)}.",
        flush=True
    )
    torch.manual_seed(a.seed+100*FOLD+group)
    model=RegionAnchoredTemporalRefiner(state["model"]).to(device)
    optimizer=torch.optim.AdamW(
        [p for p in model.parameters() if p.requires_grad],
        lr=a.learning_rate,weight_decay=1e-3
    )
    prefix=stem(a,group)
    last=prefix.with_name(prefix.name+"_last.pt")
    best=prefix.with_name(prefix.name+"_best.pt")
    final=prefix.with_name(prefix.name+"_final.pt")
    start=best_epoch=stale=0
    best_loss=float("inf")
    history=[]
    if last.is_file():
        if not a.resume:
            raise RuntimeError(f"{last} exists; use --resume.")
        snapshot=torch.load(
            last,map_location=device,weights_only=False
        )
        if snapshot.get("signature")!=signature:
            raise RuntimeError(
                f"MAG {group} checkpoint/resume signature mismatch."
            )
        model.load_state_dict(snapshot["model"])
        optimizer.load_state_dict(snapshot["optimizer"])
        start=snapshot["epoch"]
        best_epoch=snapshot["best_epoch"]
        stale=snapshot["stale"]
        best_loss=snapshot["best_loss"]
        history=snapshot["history"]
        print(
            f"FOLD 4 MAG GROUP {group}: resumed at epoch {start}.",
            flush=True
        )
    for epoch in range(start+1,a.max_epochs+1):
        if epoch>=a.min_epochs and stale>=a.patience:
            break
        tr=v10.train_epoch(
            model,optimizer,fit_pool,norm,a,device,epoch,
            FOLD+10*group
        )
        mon=v10.monitor_epoch(
            model,monitor_pool,norm,a,device
        )
        if epoch>=a.min_epochs:
            if mon["total"]<best_loss-1e-4:
                best_loss=mon["total"]
                best_epoch=epoch
                stale=0
                v7.safe_checkpoint(best,{
                    "signature":signature,
                    "epoch":epoch,"model":model.state_dict()
                })
            else:
                stale+=1
        history.append({
            "epoch":epoch,
            "train":tr,
            "monitor":mon,
            "best_epoch":best_epoch,"stale":stale
        })
        v7.safe_checkpoint(last,{
            "signature":signature,
            "epoch":epoch,"best_epoch":best_epoch,
            "best_loss":best_loss,"stale":stale,
            "history":history,
            "model":model.state_dict(),
            "optimizer":optimizer.state_dict()
        })
        print(
            f"FOLD 4 MAG GROUP {group} EPOCH {epoch}: "
            f"train={tr['total']:.4f}, monitor={mon['total']:.4f}, "
            f"best={best_epoch} stale={stale}.",
            flush=True
        )
        if epoch>=a.min_epochs and stale>=a.patience:
            break
    if not best.is_file():
        raise RuntimeError(
            f"MAG {group}: no best V10 checkpoint."
        )
    best_state=torch.load(
        best,map_location=device,weights_only=False
    )
    if best_state.get("signature")!=signature:
        raise RuntimeError("Best grouped V10 signature mismatch.")
    model.load_state_dict(best_state["model"])
    del fit_pool,monitor_pool
    monitor_coarse=v10.teacher_coarse_proposals(
        model,monitor,blocks,norm,t_mpb,t_bs,a,device
    )
    monitor_fine=v10.refine_proposals(
        model,monitor_coarse,norm,a,device
    )
    calibration=v10.calibrate_refinement(monitor_fine)
    outer_coarse=v10.teacher_coarse_proposals(
        model,outer,blocks,norm,t_mpb,t_bs,a,device
    )
    outer_fine=v10.refine_proposals(
        model,outer_coarse,norm,a,device
    )
    _,result_rows=v10.evaluate_fold(
        outer_fine,calibration,src7csv
    )
    metrics,paired=paired_with_v10(
        result_rows,baseline,group
    )
    report={
        "schema":1,"architecture":ARCHITECTURE,
        "fold":FOLD,"mag_cluster":group,
        "signature":signature,
        "train_events":len(fit),"monitor_events":len(monitor),
        "holdout_events":len(outer),
        "fit_usable_events":len([1 for x in fit if x["time"] in blocks]),
        "best_epoch":best_epoch,
        "epochs_completed":len(history),
        "group_inner_split_seed":split_seed,
        "monitor_calibration":calibration,
        "history":history,
        "fit_skips":fit_skips,
        "monitor_skips":monitor_skips,
        "teacher_frozen":True,
        "warning":(
            "Fold-4 events excluded from V13 MAG encoder and all V10 "
            "expert weight/calibration fits, but routing uses group "
            "labels computed with catalog MPB times. Therefore this is "
            "conditional/oracle group assignment, not a deployable "
            "end-to-end MPB detector."
        ),
        **metrics,
    }
    metrics_path=prefix.with_name(prefix.name+"_metrics.json")
    events_path=prefix.with_name(prefix.name+"_event_metrics.csv")
    v7.safe_json(metrics_path,report)
    v7.write_csv(events_path,paired)
    v7.safe_checkpoint(final,{
        "schema":1,"architecture":ARCHITECTURE,
        "fold":FOLD,"mag_cluster":group,
        "signature":signature,
        "best_epoch":best_epoch,
        "model":model.state_dict(),
        "normalizers":state["normalizers"],
        "calibration":calibration,
        "group_inner_split_seed":split_seed,
    })
    print(
        f"FOLD 4 MAG GROUP {group}: "
        f"within60={metrics['within_60s']:.1%}, "
        f"V10={metrics['baseline_v10_within_60s']:.1%}, "
        f"median={metrics['median_error_s']:.1f}s, "
        f"improved={metrics['improved_vs_v10']}/{len(paired)}, "
        f"alpha={calibration['alpha']:.2f}, "
        f"method={calibration['method']}.",
        flush=True
    )
    return report


def summarize(a,reports,expected):
    all_rows=[]
    for group in sorted(reports):
        prefix=stem(a,group)
        path=prefix.with_name(prefix.name+"_event_metrics.csv")
        with path.open(newline="",encoding="utf-8") as fp:
            all_rows.extend(csv.DictReader(fp))
    keys=[r["mpb_timestamp_utc"] for r in all_rows]
    expected_keys={r["time"].isoformat() for r in expected}
    if len(keys)!=len(set(keys)):
        raise RuntimeError("A fold-4 event appears in both MAG groups.")
    if a.groups=="all" and set(keys)!=expected_keys:
        raise RuntimeError(
            f"Fold-4 evaluation incomplete: {len(keys)} of "
            f"{len(expected_keys)}. Do not silently drop events."
        )
    curr=np.asarray([
        float(r["absolute_error_s"]) for r in all_rows
    ])
    old=np.asarray([
        float(r["v10_absolute_error_s"]) for r in all_rows
    ])
    if len(curr)==0:
        raise RuntimeError("No completed group metrics.")
    summary={
        "schema":1,"architecture":ARCHITECTURE,
        "fold":FOLD,"groups_evaluated":sorted(reports),
        "n_events":len(curr),
        "n_within_60s":int(np.sum(curr<=60.)),
        "fraction_within_60s":float(np.mean(curr<=60.)),
        "median_error_s":float(np.median(curr)),
        "mean_error_s":float(np.mean(curr)),
        "p90_error_s":float(np.percentile(curr,90)),
        "fraction_within_120s":float(np.mean(curr<=120.)),
        "fraction_within_300s":float(np.mean(curr<=300.)),
        "paired_v10_within_60s":float(np.mean(old<=60.)),
        "paired_v10_median_error_s":float(np.median(old)),
        "improved_vs_v10":int(np.sum(curr<old)),
        "worsened_vs_v10":int(np.sum(curr>old)),
        "tied_vs_v10":int(np.sum(curr==old)),
        "target_80pct_met":bool(np.mean(curr<=60.)>=.8),
        "group_results":[{
            "mag_cluster":g,
            "n_events":reports[g]["n_events"],
            "median_error_s":reports[g]["median_error_s"],
            "within_60s":reports[g]["within_60s"],
            "v10_within_60s":reports[g]["baseline_v10_within_60s"],
            "best_epoch":reports[g]["best_epoch"],
            "alpha":reports[g]["monitor_calibration"]["alpha"],
        } for g in sorted(reports)],
        "warning":(
            "Validating V10 group specialists ONLY on fold4. "
            "The V13 MAG groups were inferred from windows centered "
            "using the actual MPB annotation. Results therefore "
            "assume ORACLE group assignment and are NOT autonomous "
            "localization of unknown MPBs."
        ),
    }
    return summary


def main():
    a=parse_args()
    torch.set_num_threads(a.torch_threads)
    random.seed(a.seed)
    np.random.seed(a.seed)
    torch.manual_seed(a.seed)
    device=torch.device("cpu")
    rows=pilot.read_manifest(a.manifest)
    for row in rows:
        if not row.get("bs_timestamp_utc"):
            raise RuntimeError(
                "Grouped V10 requires BS catalog for negative filtering."
            )
        row["bs_time"]=bq.to_utc(row["bs_timestamp_utc"])
    selected=[
        row for fold in (1,2,3,4)
        for row in v7.chosen_events(
            rows,fold,pilot_count=0
        )
    ]
    source7,source7csv,base10=source_files(a)
    for path in (source7,source7csv,base10):
        if not path.is_file():
            raise FileNotFoundError(
                f"Missing required fold-4 source: {path}"
            )
    labels=read_mag_groups(a.mag_groups_csv)
    train=[r for r in selected if r["fold"]!=FOLD]
    outer=[r for r in selected if r["fold"]==FOLD]
    previous.check_disjoint_supports(train,outer)
    g_train,missing_train=grouped_events(train,labels)
    g_outer,missing_outer=grouped_events(outer,labels)
    if missing_outer:
        raise RuntimeError(
            f"Fold 4 has {len(missing_outer)} MAG-unassigned crossings. "
            "Do not silently omit events from the evaluation."
        )
    if a.groups=="all":
        groups=GROUPS
    else:
        groups=(int(a.groups),)
    for group in groups:
        if not g_outer[group]:
            raise RuntimeError(
                f"Missing fold-4 crossings of MAG group {group}."
            )
    a.output_dir.mkdir(parents=True,exist_ok=True)
    prepared={}
    completed={}
    for group in groups:
        fit,monitor,excluded,split_seed=inner_group_split(
            g_train[group],group,a.seed
        )
        signature=experiment_signature(
            a,group,source7,fit,monitor,g_outer[group],
            split_seed
        )
        prefix=stem(a,group)
        metrics_path=prefix.with_name(prefix.name+"_metrics.json")
        final_path=prefix.with_name(prefix.name+"_final.pt")
        if a.resume and metrics_path.is_file() and final_path.is_file():
            stored=json.loads(
                metrics_path.read_text(encoding="utf-8")
            )
            if (stored.get("signature")!=signature
                    or stored.get("architecture")!=ARCHITECTURE):
                raise RuntimeError(
                    f"MAG group {group} completed checkpoint mismatch."
                )
            completed[group]=stored
            print(
                f"FOLD 4 MAG GROUP {group}: previously completed, "
                "verified and skipped.",flush=True
            )
        else:
            prepared[group]={
                "fit":fit,"monitor":monitor,
                "outer":g_outer[group],
                "seed":split_seed,
            }
        print(
            f"FOLD 4 MAG GROUP {group}: "
            f"TRAIN={len(g_train[group])}, "
            f"FIT={len(fit)}, MONITOR={len(monitor)}, "
            f"OUTER={len(g_outer[group])}, "
            f"excluded_overlaps={excluded}.",
            flush=True
        )
    if not a.execute:
        print(
            "PREFLIGHT ONLY: exactly FOLD 4 is reserved for evaluation. "
            "Run with --execute to train 2 group-specific V10 refiners.",
            flush=True
        )
        return
    if prepared:
        needed={
            r["time"]:r
            for section in prepared.values()
            for subset in ("fit","monitor","outer")
            for r in section[subset]
        }
        print(
            f"Loading {len(needed)} event MAG/SWEA blocks for "
            f"{len(prepared)} FOLD-4 group experts (cache first).",
            flush=True
        )
        blocks=v7.load_event_blocks(list(needed.values()),a)
        if blocks is None:
            raise RuntimeError("Could not load measured event blocks.")
        for group in sorted(prepared):
            args=prepared[group]
            completed[group]=run_one_group(
                a,group,args["fit"],args["monitor"],
                args["outer"],blocks,rows,device,args["seed"]
            )
    expected=[
        event for group in groups for event in g_outer[group]
    ]
    summary=summarize(a,completed,expected)
    v7.safe_json(a.output_dir/"fold_04_summary.json",summary)
    print(
        f"FOLD 4 MAG GROUPED V10: "
        f"{summary['n_within_60s']}/{summary['n_events']} "
        f"within60={summary['fraction_within_60s']:.1%}, "
        f"original V10={summary['paired_v10_within_60s']:.1%}, "
        f"median={summary['median_error_s']:.1f}s, "
        f"target80={summary['target_80pct_met']}.",
        flush=True
    )


if __name__=="__main__":
    main()
