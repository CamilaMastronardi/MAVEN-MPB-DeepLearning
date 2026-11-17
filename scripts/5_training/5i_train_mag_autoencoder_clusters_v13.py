#!/usr/bin/env python3
"""V13, PHASE 1: MAG 8-layer CNN autoencoder -> 3D latent -> crossing groups.

Deliberately NOT another V7b/V10/V12 ranking architecture. Train one
discovery encoder using catalog folds 1,2,3, reserving fold 4 and
separating an inner-month monitor for early stop. Only TRAIN event
embeddings fit the KMeans clusters; fold 4 is ASSIGNED, not fit.

The AE has a reconstruction-primary loss with auxiliary binary
positive-window identification and positive-window offset regression.
This PHASE DOES NOT YET CLAIM 60-second full-event MPB localization.
Positive shifted clips and conservative controls come from the
existing measured native-cadence MAG/SWEA-safe window code, but only
actual 1-Hz MAG samples are used as inputs to the new 8-layer CNN.
No SWEA flux is interpolated to MAG cadence.

Clusters describe MPB-crossing morphology using average bottleneck
coordinates over several shifted positive-window views of each event.
Those views use catalog time to DEFINE the exploratory grouping;
therefore this clustering by itself is NOT a deployable autonomous
event-group assignment. A later experiment must validate group
inference without known MPB timing and train group-specific localizers.
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
from sklearn.cluster import KMeans
from sklearn.metrics import (
    silhouette_score,adjusted_rand_score,precision_recall_fscore_support,
    roc_auc_score,average_precision_score,
)

ROOT=Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0,str(ROOT))

v7=import_module("scripts.5_training.5b_train_mag_swea_regions")
v9=import_module("scripts.5_training.5e_train_mag_swea_temporal_v9")
pilot=import_module("scripts.3_windowing.3b_bigquery_window_pilot")
previous=import_module("scripts.5_training.5a_train_mag_cv")
bq=import_module("scripts.1_bigquery.1c_bigquery_event_pilot")

from src.data.mpb_dual_windows import extract_clip,fit_unique_epochs
from src.data.mpb_weak_regions import classify_candidate
from src.detection.mpb_cnn_autoencoder_v13 import (
    Conv3DAutoencoder,mag_collate,autoencoder_loss
)

OUT=Path("results/metrics/mag_autoencoder_clusters_v13")
OFFSETS_FOR_EMBED=(-240.,-180.,-120.,-60.,0.,60.,120.,180.,240.)


def get_args():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument("--manifest",type=Path,default=v7.MANIFEST)
    p.add_argument("--output-dir",type=Path,default=OUT)
    p.add_argument("--mag-cache-dir",type=Path,default=v7.MAG_CACHE)
    p.add_argument("--swea-cache-dir",type=Path,default=v7.SWEA_CACHE)
    p.add_argument("--refresh-swea-cache",action="store_true")
    p.add_argument("--pilot-events-per-fold",type=int,default=0)
    p.add_argument("--discovery-holdout-fold",type=int,default=4,
                   choices=(1,2,3,4))
    p.add_argument("--max-epochs",type=int,default=65)
    p.add_argument("--min-epochs",type=int,default=8)
    p.add_argument("--patience",type=int,default=10)
    p.add_argument("--batch-size",type=int,default=32)
    p.add_argument("--train-positives-per-event",type=int,default=3)
    p.add_argument("--train-negatives-per-event",type=int,default=3)
    p.add_argument("--learning-rate",type=float,default=5e-4)
    p.add_argument("--detection-weight",type=float,default=.30)
    p.add_argument("--temporal-weight",type=float,default=.10)
    p.add_argument("--cluster-k-min",type=int,default=2)
    p.add_argument("--cluster-k-max",type=int,default=6)
    p.add_argument("--seed",type=int,default=42)
    p.add_argument("--torch-threads",type=int,default=2)
    p.add_argument("--max-query-gib",type=float,default=2.)
    p.add_argument("--max-total-gib",type=float,default=150.)
    p.add_argument("--max-mag-rows",type=int,default=25000)
    p.add_argument("--max-swea-rows",type=int,default=2000)
    p.add_argument("--resume",action="store_true")
    p.add_argument("--execute",action="store_true")
    a=p.parse_args()
    if (min(a.max_epochs,a.min_epochs,a.patience,a.batch_size,
            a.train_positives_per_event,a.train_negatives_per_event,
            a.learning_rate,a.cluster_k_min,a.torch_threads,
            a.max_query_gib,a.max_total_gib,a.max_mag_rows,
            a.max_swea_rows)<=0 or
            a.min_epochs>a.max_epochs or
            a.cluster_k_min>a.cluster_k_max or
            a.pilot_events_per_fold<0 or
            min(a.detection_weight,a.temporal_weight)<0):
        p.error("Invalid V13 discovery configuration.")
    return a


def signature(a,fit,monitor,outer):
    payload={
        "architecture":"v13-MAG-8-conv-3D-bottleneck-discovery-v1",
        "manifest_sha":v7.sha256_file(a.manifest),
        "code_sha":v7.sha256_file(
            ROOT/"src/detection/mpb_cnn_autoencoder_v13.py"
        ),
        "fit":[r["time"].isoformat() for r in fit],
        "monitor":[r["time"].isoformat() for r in monitor],
        "outer":[r["time"].isoformat() for r in outer],
        "training":{
            "epochs":(a.max_epochs,a.min_epochs,a.patience),
            "batch_size":a.batch_size,
            "learning_rate":a.learning_rate,
            "n_positive":a.train_positives_per_event,
            "n_negative":a.train_negatives_per_event,
            "det_weight":a.detection_weight,
            "temporal_weight":a.temporal_weight,
            "seed":a.seed,
        },
    }
    return hashlib.sha256(
        json.dumps(payload,sort_keys=True).encode()
    ).hexdigest()


def gather_samples(pools,selected=False):
    examples=[]
    for parts in pools.values():
        examples.extend(parts["positive"])
        examples.extend(parts["negative"])
    return examples


def compute_loss(model,batch,args):
    out=model(batch["mag"],batch["mask"])
    return autoencoder_loss(
        out,batch,detection_weight=args.detection_weight,
        temporal_weight=args.temporal_weight
    )


def train_epoch(model,optimizer,pools,norm,a,device,epoch):
    examples=v9.select_epoch_examples(
        pools,epoch,a.seed,a
    )
    model.train()
    n=0
    totals=defaultdict(float)
    for begin in range(0,len(examples),a.batch_size):
        sub=examples[begin:begin+a.batch_size]
        batch=mag_collate(sub,norm,device)
        optimizer.zero_grad(set_to_none=True)
        loss,metrics=compute_loss(model,batch,a)
        if not bool(torch.isfinite(loss)):
            raise FloatingPointError("V13 nonfinite training loss.")
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(),1.)
        optimizer.step()
        for key in ("reconstruction_mse","identification_bce",
                    "time_huber","total"):
            totals[key]+=metrics[key]*len(sub)
        n+=len(sub)
    if n==0:
        raise RuntimeError("V13 no training examples.")
    return {key:total/n for key,total in totals.items()}


@torch.no_grad()
def monitor_epoch(model,pools,norm,a,device):
    model.eval()
    items=gather_samples(pools)
    if not items:
        raise RuntimeError("V13 no inner-month monitor samples.")
    losses=defaultdict(float)
    truth=[]
    score=[]
    offsets=[]
    predicted=[]
    count=0
    for begin in range(0,len(items),a.batch_size):
        sub=items[begin:begin+a.batch_size]
        batch=mag_collate(sub,norm,device)
        outputs=model(batch["mag"],batch["mask"])
        _,metrics=autoencoder_loss(
            outputs,batch,detection_weight=a.detection_weight,
            temporal_weight=a.temporal_weight
        )
        for key in ("reconstruction_mse","identification_bce",
                    "time_huber","total"):
            losses[key]+=metrics[key]*len(sub)
        truth.extend(batch["positive"].cpu().numpy().tolist())
        score.extend(
            torch.sigmoid(outputs["presence_logit"]).cpu().numpy().tolist()
        )
        hit=batch["positive"]>.5
        offsets.extend(batch["offset_s"][hit].cpu().numpy().tolist())
        predicted.extend(
            outputs["offset_s"][hit].cpu().numpy().tolist()
        )
        count+=len(sub)
    report={k:v/count for k,v in losses.items()}
    label=np.asarray(truth,dtype=int)
    prob=np.asarray(score,dtype=float)
    dt=np.abs(
        np.asarray(predicted,dtype=float)-
        np.asarray(offsets,dtype=float)
    )
    predicted_class=(prob>=.5).astype(int)
    precision,recall,f1,_=precision_recall_fscore_support(
        label,predicted_class,average="binary",zero_division=0
    )
    report["window_precision_at_0_5"]=float(precision)
    report["window_recall_at_0_5"]=float(recall)
    report["window_f1_at_0_5"]=float(f1)
    report["window_accuracy_at_0_5"]=float(
        np.mean(predicted_class==label)
    )
    report["window_roc_auc"]=(
        float(roc_auc_score(label,prob))
        if len(np.unique(label))==2 else None
    )
    report["window_pr_auc"]=(
        float(average_precision_score(label,prob))
        if label.any() else None
    )
    report["positive_window_offset_median_error_s"]=(
        float(np.median(dt)) if len(dt) else None
    )
    report["positive_window_within_60s"]=(
        float(np.mean(dt<=60.)) if len(dt) else None
    )
    report["n_examples"]=count
    return report


@torch.no_grad()
def event_embedding(
    model,row,blocks,norm,device,batch_size,mpb_s,bs_s
):
    """Aggregate shifted positive views; labels used ONLY to define views.

    Clusters are descriptive, not directly deployable before MPB is known.
    The averaged z is less tied to whether MPB was at clip center.
    """
    mag,swea=blocks[row["time"]]["MAG"],blocks[row["time"]]["SWEA"]
    center=row["time"].timestamp()
    examples=[]
    for delta in OFFSETS_FOR_EMBED:
        candidate=classify_candidate(
            row,center+delta,all_mpb_s=mpb_s,all_bs_s=bs_s
        )
        # Exclude windows confounded by other catalog boundaries.
        if candidate.kind not in ("positive","mpb_ambiguous"):
            continue
        clip=extract_clip(mag,swea,candidate.center_s)
        if clip is not None:
            examples.append((clip,1,-delta))
    if not examples:
        return None
    zs=[]
    recon=[]
    positives=[]
    for begin in range(0,len(examples),batch_size):
        subset=examples[begin:begin+batch_size]
        batch=mag_collate(subset,norm,device)
        out=model(batch["mag"],batch["mask"])
        z=out["latent"].cpu().numpy()
        zs.extend(z.tolist())
        valid=batch["mask"][:,None,:].to(
            out["reconstruction"].dtype
        )
        mse=(
            (out["reconstruction"]-batch["mag"]).square()*valid
        ).sum(dim=(1,2))/(
            valid.sum(dim=(1,2))*4.
        ).clamp_min(1)
        recon.extend(mse.cpu().numpy().tolist())
        positives.extend(
            torch.sigmoid(out["presence_logit"]).cpu().numpy().tolist()
        )
    zs=np.asarray(zs,dtype=float)
    return {
        "latent":zs.mean(axis=0),
        "latent_spread":float(zs.std(axis=0).mean()),
        "window_mse":float(np.mean(recon)),
        "positive_prob":float(np.mean(positives)),
        "views":len(zs),
    }


def choose_clustering(fit_rows,records,a):
    """KMeans on 3D TRAIN event embeddings, not on windows or holdout."""
    timestamps=[
        r["time"] for r in fit_rows if r["time"] in records
    ]
    z=np.asarray([
        records[t]["latent"] for t in timestamps
    ],dtype=np.float64)
    if len(z)<a.cluster_k_min+2:
        raise RuntimeError(
            f"V13 needs >={a.cluster_k_min+2} fit event embeddings."
        )
    mu=z.mean(0)
    scale=z.std(0)
    scale=np.where(scale>1e-6,scale,1.)
    standardized=(z-mu)/scale
    candidates=[]
    for k in range(a.cluster_k_min,min(a.cluster_k_max,len(z)-1)+1):
        estimator=KMeans(n_clusters=k,n_init=20,random_state=a.seed)
        labels=estimator.fit_predict(standardized)
        counts=np.bincount(labels,minlength=k)
        if min(counts)<3:
            continue
        silhouette=float(silhouette_score(standardized,labels))
        candidates.append({
            "k":k,"silhouette":silhouette,
            "min_size":int(counts.min()),
            "max_size":int(counts.max()),
        })
    if not candidates:
        raise RuntimeError(
            "V13 clusters were too small/degenerate for reliable "
            "silhouette comparison. Increase the sample size."
        )
    selected=max(
        candidates,key=lambda item:(item["silhouette"],-item["k"])
    )
    estimator=KMeans(
        n_clusters=selected["k"],n_init=40,random_state=a.seed
    ).fit(standardized)
    return {
        "estimator":estimator,
        "scaler_mean":mu,
        "scaler_scale":scale,
        "selection":selected,
        "candidates":candidates
    }


def assign_groups(rows,records,clustering,split_map):
    assigned=[]
    for row in rows:
        when=row["time"]
        info=records.get(when)
        if info is None:
            assigned.append({
                "mpb_timestamp_utc":when.isoformat(),
                "fold":row["fold"],
                "split":split_map[when],
                "catalog_group":row.get("source_group",""),
                "cluster":"", "z1":"", "z2":"", "z3":"",
                "n_views":0,
                "reconstruction_mse":"",
                "presence_probability":"",
                "latent_spread":""
            })
            continue
        z=np.asarray(info["latent"],dtype=float)
        scaled=(
            z-clustering["scaler_mean"]
        )/clustering["scaler_scale"]
        label=int(clustering["estimator"].predict(
            scaled[None,:]
        )[0])
        assigned.append({
            "mpb_timestamp_utc":when.isoformat(),
            "fold":row["fold"],
            "split":split_map[when],
            "catalog_group":row.get("source_group",""),
            "cluster":label+1,
            "z1":float(z[0]),"z2":float(z[1]),"z3":float(z[2]),
            "n_views":info["views"],
            "reconstruction_mse":info["window_mse"],
            "presence_probability":info["positive_prob"],
            "latent_spread":info["latent_spread"],
        })
    return assigned


def save_scatter(rows,path):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    valid=[r for r in rows if r["cluster"]!=""]
    if not valid:
        return
    labels=np.asarray([int(r["cluster"]) for r in valid])
    x=np.asarray([
        [float(r[k]) for k in ("z1","z2","z3")]
        for r in valid
    ])
    fig=plt.figure(figsize=(9,7))
    ax=fig.add_subplot(111,projection="3d")
    for label in sorted(set(labels)):
        points=x[labels==label]
        ax.scatter(
            points[:,0],points[:,1],points[:,2],s=22,
            alpha=.72,label=f"Grupo {label}"
        )
    ax.set_xlabel("z1")
    ax.set_ylabel("z2")
    ax.set_zlabel("z3")
    ax.set_title("V13: grupos exploratorios MAG / espacio latente 3D")
    ax.legend(loc="best")
    fig.tight_layout()
    fig.savefig(path,dpi=160)
    plt.close(fig)


def summarize_groups(rows,clustering):
    valid=[r for r in rows if r["cluster"]!=""]
    groups={}
    for k in sorted({int(r["cluster"]) for r in valid}):
        selected=[r for r in valid if r["cluster"]==k]
        catalog={}
        for r in selected:
            label=str(r["catalog_group"])
            catalog[label]=catalog.get(label,0)+1
        groups[str(k)]={
            "n":len(selected),
            "by_manifest_fold":{
                str(f):sum(r["fold"]==f for r in selected)
                for f in (1,2,3,4)
            },
            "by_catalog_group":catalog,
            "mean_reconstruction_mse":float(np.mean(
                [r["reconstruction_mse"] for r in selected]
            )),
            "mean_presence_probability":float(np.mean(
                [r["presence_probability"] for r in selected]
            )),
        }
    return {
        "selected_k":clustering["selection"]["k"],
        "fit_silhouette":clustering["selection"]["silhouette"],
        "k_comparison":clustering["candidates"],
        "n_events_with_embedding":len(valid),
        "n_events_unavailable":len(rows)-len(valid),
        "groups":groups,
    }


def compare_clusters_to_v10(assigned,baseline_dir):
    """Post-hoc group diagnostics; never used in AE or KMeans fit.

    Old V10 errors are compared on identical catalog event timestamps.
    Missing previous output is not an error for V13 exploratory grouping.
    """
    matched={}
    for fold in (1,2,3,4):
        path=baseline_dir/f"fold_{fold:02d}_event_metrics.csv"
        if not path.is_file():
            continue
        with path.open(newline="",encoding="utf-8") as fp:
            for row in csv.DictReader(fp):
                try:
                    matched[row["mpb_timestamp_utc"]]=float(
                        row["absolute_error_s"]
                    )
                except (ValueError,KeyError):
                    pass
    by_cluster=defaultdict(list)
    no_baseline=0
    for row in assigned:
        if row["cluster"]=="":
            continue
        err=matched.get(row["mpb_timestamp_utc"])
        if err is None:
            no_baseline+=1
            continue
        by_cluster[int(row["cluster"])].append(err)
    return {
        "reference_model":"V10, exact event timestamp matches",
        "matched":sum(map(len,by_cluster.values())),
        "without_v10_reference":no_baseline,
        "groups":{
            str(group):{
                "n":len(err),
                "median_v10_error_s":float(np.median(err)),
                "v10_fraction_within_60s":float(np.mean(
                    np.asarray(err)<=60.
                )),
                "v10_fraction_over_300s":float(np.mean(
                    np.asarray(err)>300.
                )),
            } for group,err in sorted(by_cluster.items())
        },
        "caution":(
            "Diagnostic after unsupervised grouping; do not use V10 error "
            "or future localization quality as KMeans input."
        )
    }


def main():
    a=get_args()
    torch.set_num_threads(a.torch_threads)
    torch.manual_seed(a.seed)
    np.random.seed(a.seed)
    random.seed(a.seed)
    all_rows=pilot.read_manifest(a.manifest)
    for row in all_rows:
        if not row.get("bs_timestamp_utc"):
            raise RuntimeError("V13 requires BS annotations to filter negatives.")
        row["bs_time"]=bq.to_utc(row["bs_timestamp_utc"])
    selected=sorted([
        row for fold in (1,2,3,4)
        for row in v7.chosen_events(
            all_rows,fold,pilot_count=a.pilot_events_per_fold
        )
    ],key=lambda row:row["time"])
    outer=[
        r for r in selected
        if r["fold"]==a.discovery_holdout_fold
    ]
    training=[
        r for r in selected
        if r["fold"]!=a.discovery_holdout_fold
    ]
    previous.check_disjoint_supports(training,outer)
    fit,monitor,removed=previous.split_inner_monitor_months(
        training,a.discovery_holdout_fold,a.seed
    )
    if not a.execute:
        print(
            f"V13 preflight: {len(selected)} catalog events; "
            f"fit={len(fit)} monitor={len(monitor)} "
            f"heldout={len(outer)}. Run with --execute.",
            flush=True
        )
        return
    a.output_dir.mkdir(parents=True,exist_ok=True)
    sig=signature(a,fit,monitor,outer)
    device=torch.device("cpu")
    # Reuse measured source caches only. The V7 source loader checks
    # BigQuery cost caps BEFORE querying any missing 70-min blocks.
    blocks=v7.load_event_blocks(selected,a)
    if blocks is None:
        raise RuntimeError("No measured MAG/SWEA sources.")
    mag_norm=fit_unique_epochs(
        blocks,[row["time"] for row in fit],instrument="MAG"
    )
    mpb_s=np.asarray([
        r["time"].timestamp() for r in all_rows
    ],dtype=np.float64)
    bs_s=np.asarray([
        r["bs_time"].timestamp() for r in all_rows
    ],dtype=np.float64)
    fit_pool,fit_skips=v9.gather_training_pools(
        fit,blocks,mpb_s,bs_s
    )
    monitor_pool,monitor_skips=v9.gather_training_pools(
        monitor,blocks,mpb_s,bs_s
    )
    if not fit_pool or not monitor_pool:
        raise RuntimeError("V13 needs measured positive and safe negative clips.")
    print(
        f"V13 MAG discovery: fit={len(fit)} "
        f"usable={len(fit_pool)} monitor={len(monitor)} "
        f"usable={len(monitor_pool)} heldout={len(outer)}.",
        flush=True
    )
    model=Conv3DAutoencoder().to(device)
    optimizer=torch.optim.AdamW(
        model.parameters(),lr=a.learning_rate,weight_decay=1e-4
    )
    last=a.output_dir/"discovery_last.pt"
    best=a.output_dir/"discovery_best.pt"
    final=a.output_dir/"discovery_final.pt"
    start=best_epoch=stale=0
    best_loss=float("inf")
    history=[]
    if a.resume and last.is_file():
        old=torch.load(last,map_location=device,weights_only=False)
        if old.get("signature")!=sig:
            raise RuntimeError("V13 resume signature mismatch.")
        model.load_state_dict(old["model"])
        optimizer.load_state_dict(old["optimizer"])
        start=old["epoch"]
        best_epoch=old["best_epoch"]
        stale=old["stale"]
        best_loss=old["best_loss"]
        history=old["history"]
        print(
            f"V13 resumed at epoch {start}.",flush=True
        )
    elif last.is_file() and not a.resume:
        raise RuntimeError("V13 last checkpoint exists; use --resume.")
    for epoch in range(start+1,a.max_epochs+1):
        if epoch>=a.min_epochs and stale>=a.patience:
            break
        tr=train_epoch(
            model,optimizer,fit_pool,mag_norm,a,device,epoch
        )
        val=monitor_epoch(
            model,monitor_pool,mag_norm,a,device
        )
        if epoch>=a.min_epochs:
            if val["total"]<best_loss-1e-4:
                best_loss=val["total"]
                best_epoch=epoch
                stale=0
                v7.safe_checkpoint(best,{
                    "signature":sig,"model":model.state_dict(),
                    "epoch":epoch,
                })
            else:
                stale+=1
        history.append({
            "epoch":epoch,
            "train":tr,
            "monitor":val,
            "best_epoch":best_epoch,
            "stale":stale,
        })
        v7.safe_checkpoint(last,{
            "signature":sig,"epoch":epoch,
            "best_epoch":best_epoch,"stale":stale,
            "best_loss":best_loss,"history":history,
            "model":model.state_dict(),
            "optimizer":optimizer.state_dict(),
        })
        print(
            f"V13 EPOCH {epoch}: "
            f"train_rec={tr['reconstruction_mse']:.4f} "
            f"monitor_rec={val['reconstruction_mse']:.4f} "
            f"monitor_auc={val['window_roc_auc']} "
            f"monitor_positive_within60={val['positive_window_within_60s']} "
            f"best={best_epoch} stale={stale}.",
            flush=True
        )
        if epoch>=a.min_epochs and stale>=a.patience:
            break
    if not best.is_file():
        raise RuntimeError("V13 best checkpoint missing.")
    ckpt=torch.load(best,map_location=device,weights_only=False)
    if ckpt.get("signature")!=sig:
        raise RuntimeError("V13 best checkpoint mismatch.")
    model.load_state_dict(ckpt["model"])
    model.eval()
    del fit_pool,monitor_pool
    embeddings={}
    for i,row in enumerate(selected,1):
        info=event_embedding(
            model,row,blocks,mag_norm,device,a.batch_size,
            mpb_s,bs_s
        )
        if info is not None:
            embeddings[row["time"]]=info
        if i%50==0 or i==len(selected):
            print(
                f"V13 EMBEDDINGS {i}/{len(selected)} "
                f"usable={len(embeddings)}.",
                flush=True
            )
    # Evaluate MPB-window identification and conditional offset on a
    # held-out MONTH-block fold, not only on the inner training monitor.
    # These metrics still do NOT test an autonomous full 70-min scan.
    outer_pool,outer_skips=v9.gather_training_pools(
        outer,blocks,mpb_s,bs_s
    )
    outer_window_metrics=(
        monitor_epoch(model,outer_pool,mag_norm,a,device)
        if outer_pool else None
    )
    clustering=choose_clustering(fit,embeddings,a)
    fit_times={r["time"] for r in fit}
    monitor_times={r["time"] for r in monitor}
    split={
        r["time"]:(
            "fit" if r["time"] in fit_times else
            "inner_monitor" if r["time"] in monitor_times else
            "outer_holdout"
        ) for r in selected
    }
    results=assign_groups(
        selected,embeddings,clustering,split
    )
    v7.write_csv(
        a.output_dir/"crossing_latents_groups.csv",results
    )
    save_scatter(
        results,a.output_dir/"latent_clusters_3d.png"
    )
    summary={
        "schema":1,
        "architecture":"v13-mag-8-conv-ae-3D-bottleneck",
        "signature":sig,
        "n_catalog_events":len(selected),
        "n_fit_events":len(fit),
        "n_monitor_events":len(monitor),
        "n_outer_events":len(outer),
        "n_excluded_support_overlap":removed,
        "best_epoch":best_epoch,
        "monitor_at_best":history[best_epoch-1]["monitor"],
        "outer_window_diagnostics":outer_window_metrics,
        "outer_skips":outer_skips,
        "train_skips":fit_skips,
        "monitor_skips":monitor_skips,
        "normalizer":"MAG TRAIN-only unique epochs",
        "window_definition":"actual MAG samples, 10-min clips",
        "latent_dimensions":3,
        "encoder_cnn_layers":8,
        "decoder_cnn_layers":8,
        "loss_weights":{
            "reconstruction":1.,
            "positive_window_identification":a.detection_weight,
            "positive_window_time_offset":a.temporal_weight
        },
        "grouping":summarize_groups(results,clustering),
        "limitations":(
            "Exploratory clustering uses catalog-timed positive views "
            "to describe crossings. Not deployable event detection. "
            "Window-level identification AUC and time offset accuracy "
            "are not the full-event 60-second localization rate. "
            "Do not compare cluster labels across independently "
            "trained encoders without aligning their latent spaces."
        )
    }
    summary["posthoc_v10_by_cluster"]=compare_clusters_to_v10(
        results,Path("results/metrics/mag_swea_refiner_v10")
    )
    v7.safe_json(
        a.output_dir/"discovery_summary.json",summary
    )
    v7.safe_checkpoint(final,{
        "schema":1,"signature":sig,
        "model":model.state_dict(),
        "normalizer":{
            "mean":mag_norm["mean"],
            "scale":mag_norm["scale"],
            "n_epochs":mag_norm["n_epochs"]
        },
        "latent_feature_standardizer":{
            "mean":clustering["scaler_mean"],
            "scale":clustering["scaler_scale"]
        },
        "kmeans_centers":clustering["estimator"].cluster_centers_,
        "selected_k":clustering["selection"]["k"],
        "best_epoch":best_epoch,
    })
    print(
        f"V13 discovery: {summary['grouping']['n_events_with_embedding']}/"
        f"{len(selected)} event embeddings, "
        f"K={summary['grouping']['selected_k']}, "
        f"silhouette={summary['grouping']['fit_silhouette']:.3f}, "
        f"window-pr-auc={summary['monitor_at_best']['window_pr_auc']}. "
        "Group assignments saved; no final 80% localization claim.",
        flush=True
    )


if __name__=="__main__":
    main()
