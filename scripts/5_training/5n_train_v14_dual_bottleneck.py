#!/usr/bin/env python3
"""V14: V10 conditional MPB refinement with eight-convolution 3D AEs.

A frozen matching-fold V7b model proposes the coarse event time.
Two NEW 8-layer convolutional encoders compress measured MAG and SWEA
windows independently to THREE scalar latent coordinates each; their
decoders reconstruct the observations. The fine time head ONLY sees
the two 3D codes and the frozen V7b coarse/region scores.

Fold-safe, monitored, resumable CPU training. Produces OUTER event
errors AND 3D embeddings without BigQuery if source caches are complete.
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

v10=import_module("scripts.5_training.5f_train_mag_swea_refiner_v10")
v9=import_module("scripts.5_training.5e_train_mag_swea_temporal_v9")
v7=import_module("scripts.5_training.5b_train_mag_swea_regions")
pilot=import_module("scripts.3_windowing.3b_bigquery_window_pilot")
previous=import_module("scripts.5_training.5a_train_mag_cv")
bq=import_module("scripts.1_bigquery.1c_bigquery_event_pilot")

from src.data.mpb_dual_windows import collate_dual, extract_clip
from src.detection.mpb_v14_dual_bottleneck_refiner import V14DualBottleneckRefiner

OUT=Path("results/metrics/v14_dual_3d_refiner")
METRIC_KEYS=("bce","temporal_kl","offset","reconstruction_mag",
             "reconstruction_swea","total")


def parse_args():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument("--manifest",type=Path,default=v7.MANIFEST)
    p.add_argument("--pretrained-dir",type=Path,default=v10.PRETRAINED,
                   help="Frozen matching-fold V7b source checkpoints.")
    p.add_argument("--output-dir",type=Path,default=OUT)
    p.add_argument("--mag-cache-dir",type=Path,default=v7.MAG_CACHE)
    p.add_argument("--swea-cache-dir",type=Path,default=v7.SWEA_CACHE)
    # v7.load_event_blocks uses this flag. V14 is deliberately cache-only:
    # never request a SWEA refresh (which would trigger BigQuery).
    p.set_defaults(refresh_swea_cache=False)
    p.add_argument("--folds",choices=("all","1","2","3","4"),default="all")
    p.add_argument("--pilot-events-per-fold",type=int,default=0)
    p.add_argument("--max-epochs",type=int,default=150)
    p.add_argument("--min-epochs",type=int,default=20)
    p.add_argument("--patience",type=int,default=20)
    p.add_argument("--batch-size",type=int,default=16)
    p.add_argument("--torch-threads",type=int,default=2)
    p.add_argument("--seed",type=int,default=42)
    p.add_argument("--learning-rate",type=float,default=1e-4)
    p.add_argument("--localization-weight",type=float,default=3.)
    p.add_argument("--offset-weight",type=float,default=1.5)
    p.add_argument("--reconstruction-weight",type=float,default=.2)
    p.add_argument("--gaussian-sigma-s",type=float,default=24.)
    p.add_argument("--train-positives-per-event",type=int,default=6)
    p.add_argument("--train-negatives-per-event",type=int,default=3)
    p.add_argument("--evaluation-stride-s",type=float,default=32.)
    p.add_argument("--alpha-step",type=float,default=.1)
    p.add_argument("--calibration-goal",choices=("within_60","clipped_mae"),
                   default="within_60")
    p.add_argument("--max-query-gib",type=float,default=2.)
    p.add_argument("--max-total-gib",type=float,default=150.)
    p.add_argument("--max-mag-rows",type=int,default=25000)
    p.add_argument("--max-swea-rows",type=int,default=2000)
    p.add_argument("--resume",action="store_true")
    p.add_argument("--execute",action="store_true")
    a=p.parse_args()
    positive=(a.max_epochs,a.min_epochs,a.patience,a.batch_size,
              a.torch_threads,a.learning_rate,a.localization_weight,
              a.gaussian_sigma_s,a.train_positives_per_event,
              a.train_negatives_per_event,a.evaluation_stride_s,
              a.max_query_gib,a.max_total_gib,a.max_mag_rows,a.max_swea_rows)
    if (min(positive)<=0 or a.min_epochs>a.max_epochs or
            a.pilot_events_per_fold<0 or
            a.offset_weight<0 or a.reconstruction_weight<0):
        p.error("Invalid V14 training configuration.")
    try:
        v10.calibration_alpha_grid(a.alpha_step)
    except ValueError as exc:
        p.error(str(exc))
    protected=[
        a.pretrained_dir,
        Path("results/metrics/mag_swea_refiner_v10"),
        Path("results/metrics/mag_swea_refiner_v10_hp1"),
        Path("results/metrics/mag_swea_refiner_v10_hp2a"),
        Path("results/metrics/mag_swea_refiner_v10_hp2b"),
        Path("results/metrics/mag_autoencoder_clusters_v13"),
        Path("results/metrics/swea_mag_joint_clusters_v13_frames256"),
    ]
    if a.output_dir.resolve() in {p.resolve() for p in protected}:
        p.error("V14 output cannot overwrite prior experiment checkpoints.")
    return a


def signature(a,fold,checkpoint,fit,monitor,val):
    files=[
        ROOT/"src/detection/mpb_v14_dual_bottleneck_refiner.py",
        ROOT/"scripts/5_training/5n_train_v14_dual_bottleneck.py",
        ROOT/"src/detection/mpb_cnn_autoencoder_v13.py",
        ROOT/"src/detection/mpb_swea_autoencoder_v13.py",
        ROOT/"src/detection/mpb_region_anchored_refiner.py",
    ]
    spec={
        "architecture":V14DualBottleneckRefiner.ARCHITECTURE,
        "manifest_sha":v7.sha256_file(a.manifest),
        "v7b_checkpoint_sha":v7.sha256_file(checkpoint),
        "source_code_shas":{p.name:v7.sha256_file(p) for p in files},
        "fold":fold,
        "fit":[row["time"].isoformat() for row in fit],
        "monitor":[row["time"].isoformat() for row in monitor],
        "val":[row["time"].isoformat() for row in val],
        "settings":{name:getattr(a,name) for name in (
            "max_epochs","min_epochs","patience","batch_size",
            "seed","learning_rate","localization_weight","offset_weight",
            "reconstruction_weight","gaussian_sigma_s",
            "train_positives_per_event","train_negatives_per_event",
            "evaluation_stride_s","alpha_step","calibration_goal",
        )},
    }
    return hashlib.sha256(json.dumps(spec,sort_keys=True).encode()).hexdigest()


def loss_step(model,batch,a):
    out=model(
        batch["mag"],batch["mag_mask"],batch["mag_time"],
        batch["swea"],batch["swea_mask"],batch["swea_time"]
    )
    localization,metrics=v10.refiner_loss(
        out,batch["mag_time"],batch["mag_mask"],
        batch["transition"],batch["target_offset"],
        sigma_s=a.gaussian_sigma_s,
        localization_weight=a.localization_weight,
        offset_weight=a.offset_weight,
    )
    mag=out["mag_reconstruction_mse"]
    swea=out["swea_reconstruction_mse"]
    total=localization+a.reconstruction_weight*(mag+swea)
    return total,{
        **metrics,
        "reconstruction_mag":float(mag.detach()),
        "reconstruction_swea":float(swea.detach()),
        "total":float(total.detach()),
    }


def process_epoch(model,optimizer,pools,normalizers,a,device,epoch,fold,
                  *,monitor):
    if monitor:
        examples=[
            item for group in pools.values()
            for key in ("positive","negative") for item in group[key]
        ]
        model.eval()
    else:
        examples=v9.select_epoch_examples(pools,epoch,a.seed+fold*10000,a)
        model.train()
    if not examples:
        raise RuntimeError("No usable V14 training/monitor windows.")
    metrics=defaultdict(float)
    seen=0
    context=torch.no_grad() if monitor else torch.enable_grad()
    with context:
        for start in range(0,len(examples),a.batch_size):
            selection=examples[start:start+a.batch_size]
            batch=v10.collate_examples(selection,normalizers,device)
            if not monitor:
                optimizer.zero_grad(set_to_none=True)
            loss,values=loss_step(model,batch,a)
            if not bool(torch.isfinite(loss)):
                raise FloatingPointError("Nonfinite V14 loss.")
            if not monitor:
                loss.backward()
                torch.nn.utils.clip_grad_norm_(
                    [p for p in model.parameters() if p.requires_grad],1.
                )
                optimizer.step()
            n=len(selection)
            for key in METRIC_KEYS:
                metrics[key]+=values[key]*n
            seen+=n
    return {key:metrics[key]/seen for key in METRIC_KEYS}


@torch.no_grad()
def export_outer_latents(model,val,blocks,normalizers,event_rows,
                         fold,out_path,device):
    """Export 9 MPB-relative views and one predicted-centered view.

    MPB label is used ONLY to select/align views for retrospective
    scientific analysis; the CNN does NOT receive catalog timestamps.
    The predicted view is available for inference without a catalog time
    once the search neighborhood is specified.
    """
    model.eval()
    by_event={row["mpb_timestamp_utc"]:row for row in event_rows}
    result=[]
    skipped=0
    for event in val:
        t=event["time"]
        key=t.isoformat()
        pred=by_event[key]
        offsets=[
            ("catalog_centered" if delta==0 else "shifted",float(delta))
            for delta in (-240,-180,-120,-60,0,60,120,180,240)
        ]
        offsets.append(("predicted_centered",
                        float(pred["selected_offset_s"])))
        clips=[]
        for view,delta in offsets:
            clip=extract_clip(
                blocks[t]["MAG"],blocks[t]["SWEA"],
                t.timestamp()+delta
            )
            if clip is None:
                skipped+=1
                continue
            clips.append((view,delta,clip))
        for start in range(0,len(clips),16):
            part=clips[start:start+16]
            batch=collate_dual(
                [(clip,0,(0,0)) for _,_,clip in part],
                normalizers,device)
            zmag,zswea=model.encode_latents(
                batch["mag"],batch["mag_mask"],
                batch["swea"],batch["swea_mask"]
            )
            for (view,delta,clip),zm,zs in zip(
                part,zmag.cpu().tolist(),zswea.cpu().tolist()
            ):
                result.append({
                    "mpb_timestamp_utc":key,
                    "source_group":event["source_group"],
                    "fold":fold,
                    "split":"outer",
                    "view":view,
                    "center_offset_s":delta,
                    "absolute_error_s":float(pred["absolute_error_s"]),
                    "mag_z1":zm[0],"mag_z2":zm[1],"mag_z3":zm[2],
                    "swea_z1":zs[0],"swea_z2":zs[1],"swea_z3":zs[2],
                    "mag_samples":len(clip.mag),
                    "swea_frames":len(clip.swea),
                })
    if not result:
        raise RuntimeError("V14 generated no OUTER latent embeddings.")
    v7.write_csv(out_path,result)
    return len(result),skipped


def train_fold(fold,parts,blocks,rows,a,device):
    fit,monitor,removed=previous.split_inner_monitor_months(
        parts["train"],fold,a.seed
    )
    val=parts["val"]
    teacher_path=a.pretrained_dir/f"fold_{fold:02d}_final.pt"
    teacher=torch.load(teacher_path,map_location="cpu",weights_only=False)
    if (teacher.get("fold")!=fold or
            "model" not in teacher or "normalizers" not in teacher):
        raise RuntimeError("Incorrect or incomplete matching V7b teacher.")
    normalizers=v9.normalizers_from_v7b(teacher)
    sig=signature(a,fold,teacher_path,fit,monitor,val)
    all_mpb=np.asarray(
        [r["time"].timestamp() for r in rows],dtype=np.float64
    )
    all_bs=np.asarray(
        [r["bs_time"].timestamp() for r in rows],dtype=np.float64
    )
    train_pool,train_skips=v9.gather_training_pools(
        fit,blocks,all_mpb,all_bs
    )
    mon_pool,mon_skips=v9.gather_training_pools(
        monitor,blocks,all_mpb,all_bs
    )
    if not train_pool or not mon_pool:
        raise RuntimeError("Missing safe positive/far-negative event clips.")
    torch.manual_seed(a.seed+fold)
    model=V14DualBottleneckRefiner(teacher["model"]).to(device)
    params=[p for p in model.parameters() if p.requires_grad]
    optimizer=torch.optim.AdamW(
        params,lr=a.learning_rate,weight_decay=1e-3
    )
    root=a.output_dir
    stem=f"fold_{fold:02d}"
    last=root/f"{stem}_last.pt"
    best=root/f"{stem}_best.pt"
    final=root/f"{stem}_final.pt"
    start,best_epoch,stale=0,0,0
    best_loss=float("inf")
    history=[]
    if last.exists():
        if not a.resume:
            raise RuntimeError("V14 checkpoint exists; resume or change output.")
        old=torch.load(last,map_location=device,weights_only=False)
        if old.get("signature")!=sig:
            raise RuntimeError("V14 resume signature mismatch.")
        model.load_state_dict(old["model"])
        optimizer.load_state_dict(old["optimizer"])
        start=int(old["epoch"])
        best_epoch=int(old["best_epoch"])
        best_loss=float(old["best_loss"])
        stale=int(old["stale"])
        history=old["history"]
        print(f"V14 fold {fold}: resumed epoch {start}.",flush=True)
    print(f"V14 fold {fold}: fit={len(fit)} usable={len(train_pool)} "
          f"monitor={len(monitor)} usable={len(mon_pool)} "
          f"outer={len(val)} excluded_overlaps={removed}",flush=True)
    for epoch in range(start+1,a.max_epochs+1):
        if epoch>=a.min_epochs and stale>=a.patience:
            break
        tr=process_epoch(model,optimizer,train_pool,normalizers,
                         a,device,epoch,fold,monitor=False)
        mon=process_epoch(model,None,mon_pool,normalizers,
                          a,device,epoch,fold,monitor=True)
        if epoch>=a.min_epochs:
            if mon["total"]<best_loss-1e-4:
                best_loss=mon["total"]
                best_epoch=epoch
                stale=0
                v7.safe_checkpoint(best,{
                    "signature":sig,"epoch":epoch,
                    "model":model.state_dict(),
                })
            else:
                stale+=1
        history.append({
            "epoch":epoch,"train":tr,"monitor":mon,
            "best_epoch":best_epoch,"stale":stale,
        })
        v7.safe_checkpoint(last,{
            "signature":sig,"epoch":epoch,
            "best_epoch":best_epoch,"best_loss":best_loss,
            "stale":stale,"history":history,
            "model":model.state_dict(),
            "optimizer":optimizer.state_dict(),
        })
        print(f"V14 fold {fold} epoch {epoch}: "
              f"train={tr['total']:.4f} monitor={mon['total']:.4f} "
              f"reconMAG={mon['reconstruction_mag']:.4f} "
              f"reconSWEA={mon['reconstruction_swea']:.4f} "
              f"best={best_epoch} stale={stale}",flush=True)
        if epoch>=a.min_epochs and stale>=a.patience:
            print(f"V14 fold {fold}: early stopped at epoch {epoch}.",
                  flush=True)
            break
    if not best.exists():
        raise RuntimeError("V14 no selected checkpoint.")
    checkpoint=torch.load(best,map_location=device,weights_only=False)
    if checkpoint.get("signature")!=sig:
        raise RuntimeError("V14 best signature mismatch.")
    model.load_state_dict(checkpoint["model"])
    del train_pool,mon_pool

    monitor_coarse=v10.teacher_coarse_proposals(
        model,monitor,blocks,normalizers,all_mpb,all_bs,a,device
    )
    monitor_pred=v10.refine_proposals(
        model,monitor_coarse,normalizers,a,device
    )
    calibration=v10.calibrate_refinement(
        monitor_pred,goal=a.calibration_goal,
        alphas=v10.calibration_alpha_grid(a.alpha_step)
    )
    del monitor_coarse,monitor_pred
    outer_coarse=v10.teacher_coarse_proposals(
        model,val,blocks,normalizers,all_mpb,all_bs,a,device
    )
    outer_pred=v10.refine_proposals(
        model,outer_coarse,normalizers,a,device
    )
    metrics,event_rows=v10.evaluate_fold(
        outer_pred,calibration,
        a.pretrained_dir/f"{stem}_event_metrics.csv"
    )
    n_latent,n_skips=export_outer_latents(
        model,val,blocks,normalizers,event_rows,
        fold,root/f"{stem}_latent_embeddings.csv",device
    )
    report={
        "schema":1,
        "architecture":V14DualBottleneckRefiner.ARCHITECTURE,
        "fold":fold,"signature":sig,
        "train_events":len(fit),"monitor_events":len(monitor),
        "val_total_events":len(val),
        "train_skips":train_skips,"monitor_skips":mon_skips,
        "best_epoch":best_epoch,"epochs_completed":len(history),
        "train_history":history,
        "monitor_calibration":calibration,
        "latent_dimensions":{"MAG":3,"SWEA":3},
        "encoder_conv_layers":{"MAG":8,"SWEA":8},
        "latent_export_rows":n_latent,
        "latent_export_skipped_views":n_skips,
        "normalizer_source":"matching frozen V7b fold TRAIN",
        "teacher_frozen":True,
        "training_configuration":{k:getattr(a,k) for k in (
            "learning_rate","localization_weight","offset_weight",
            "reconstruction_weight","gaussian_sigma_s",
            "train_positives_per_event","train_negatives_per_event",
            "evaluation_stride_s","calibration_goal","alpha_step",
        )},
        "limitations":(
            "Conditional MPB localization within catalog-defined +/-30min. "
            "3D embeddings use observed data only. Catalog-centered exported "
            "views use true MPB labels for retrospective analysis, never "
            "for inference. No orbital autonomous F1."
        ),
        **metrics
    }
    v7.safe_json(root/f"{stem}_metrics.json",report)
    v7.write_csv(root/f"{stem}_event_metrics.csv",event_rows)
    v7.safe_checkpoint(final,{
        "schema":1,"fold":fold,"signature":sig,
        "architecture":V14DualBottleneckRefiner.ARCHITECTURE,
        "best_epoch":best_epoch,
        "model":model.state_dict(),
        "normalizers":teacher["normalizers"],
        "calibration":calibration,
        "latent_dimensions":{"MAG":3,"SWEA":3},
    })
    print(f"V14 fold {fold}: median={metrics['median_abs_peak_error_s']:.1f}s "
          f"within60={metrics['fraction_within_60s']:.1%} "
          f"evaluable={len(event_rows)}/{len(val)} "
          f"latent_rows={n_latent} alpha={calibration['alpha']:.2f}",
          flush=True)
    return report


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
            raise RuntimeError("V14 requires catalog BS for safe controls.")
        row["bs_time"]=bq.to_utc(row["bs_timestamp_utc"])
    folds=[1,2,3,4] if a.folds=="all" else [int(a.folds)]
    selections={}
    for fold in folds:
        val=v7.chosen_events(
            rows,fold,pilot_count=a.pilot_events_per_fold
        )
        train=sorted([
            r for f in (1,2,3,4) if f!=fold
            for r in v7.chosen_events(
                rows,f,pilot_count=a.pilot_events_per_fold
            )
        ],key=lambda r:r["time"])
        if len(train)<3 or not val:
            raise RuntimeError("Insufficient V14 paired MAG/SWEA data.")
        previous.check_disjoint_supports(train,val)
        previous.split_inner_monitor_months(train,fold,a.seed)
        if not (a.pretrained_dir/f"fold_{fold:02d}_final.pt").exists():
            raise FileNotFoundError("V14 requires complete matching V7b folds.")
        if not (a.pretrained_dir/f"fold_{fold:02d}_event_metrics.csv").exists():
            raise FileNotFoundError("Missing V7b paired reference metrics.")
        selections[fold]={"train":train,"val":val}
    a.output_dir.mkdir(parents=True,exist_ok=True)
    finished,to_do={},{}
    for fold,parts in selections.items():
        fit,monitor,_=previous.split_inner_monitor_months(
            parts["train"],fold,a.seed
        )
        sig=signature(a,fold,
                      a.pretrained_dir/f"fold_{fold:02d}_final.pt",
                      fit,monitor,parts["val"])
        report=a.output_dir/f"fold_{fold:02d}_metrics.json"
        final=a.output_dir/f"fold_{fold:02d}_final.pt"
        latents=a.output_dir/f"fold_{fold:02d}_latent_embeddings.csv"
        if a.resume and all(p.exists() for p in (report,final,latents)):
            stored=json.loads(report.read_text(encoding="utf-8"))
            if (stored.get("signature")!=sig or
                    stored.get("architecture")!=
                    V14DualBottleneckRefiner.ARCHITECTURE):
                raise RuntimeError("Completed V14 signature mismatch.")
            finished[fold]=stored
            print(f"V14 fold {fold}: verified complete, skipping.",flush=True)
        else:
            to_do[fold]=parts
    if not a.execute:
        print(f"V14 preflight: {len(to_do)} pending folds; "
              "no measurements fetched. Use --execute.",flush=True)
        return
    if to_do:
        unique={
            row["time"]:row
            for part in to_do.values()
            for group in part.values() for row in group
        }
        blocks=v7.load_event_blocks(list(unique.values()),a)
        if blocks is None:
            raise RuntimeError("No measured event blocks.")
        for fold,parts in to_do.items():
            finished[fold]=train_fold(
                fold,parts,blocks,rows,a,device
            )
    summary=v10.summarize(a,finished)
    summary["architecture"]=V14DualBottleneckRefiner.ARCHITECTURE
    summary["latent_dimensions"]={"MAG":3,"SWEA":3}
    summary["encoder_conv_layers"]={"MAG":8,"SWEA":8}
    v7.safe_json(a.output_dir/"cv_summary.json",summary)
    print(f"V14 DONE: folds={summary['completed_folds']} "
          f"n={summary['n_evaluable_events']} "
          f"median={summary['global_median_absolute_error_s']:.1f}s "
          f"within60={summary['fraction_within_60s']:.1%}.",flush=True)


if __name__=="__main__":
    main()
