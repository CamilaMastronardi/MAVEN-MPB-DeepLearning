#!/usr/bin/env python3
"""V13 interpretation: physical MAG/SWEA features and latent trajectories.

Read the existing MAG and SWEA 3D encoder checkpoints and crossing CSVs.
For each measured catalog event, derive physical pre/post flank features
from native data and encode shifted 10-min windows around the MPB.
No training, no clustering fit, no MAG/SWEA time interpolation.

Spearman FIT-only correlations with BH FDR; OUTER fold4 correlations
reported separately as descriptive confirmation, never used to select
features. Sliding-window latent trajectories describe the change in the
CNN representation as the WINDOW CENTER moves; they are NOT velocities
of plasma motion. All event timestamps are catalog-known: diagnostic,
not autonomous MPB detection or localization.
"""
from __future__ import annotations

import argparse
import csv
import json
import random
import sys
from collections import Counter,defaultdict
from importlib import import_module
from pathlib import Path

import numpy as np
import torch

ROOT=Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0,str(ROOT))

v7=import_module("scripts.5_training.5b_train_mag_swea_regions")
pilot=import_module("scripts.3_windowing.3b_bigquery_window_pilot")
bq=import_module("scripts.1_bigquery.1c_bigquery_event_pilot")

from src.data.mpb_dual_windows import extract_clip
from src.data.mpb_weak_regions import classify_candidate
from src.data.swea_event_cache import ENERGY_EV
from src.detection.mpb_cnn_autoencoder_v13 import (
    Conv3DAutoencoder,mag_collate
)
from src.detection.mpb_swea_autoencoder_v13 import (
    SWEA3DAutoencoder,swea_collate
)
from src.analysis.mpb_latent_physics_v13 import (
    mag_physical_features,swea_physical_features,
    spearman_pair,benjamini_hochberg,trajectory_statistics
)

MAG_V13=Path("results/metrics/mag_autoencoder_clusters_v13")
SWEA_V13=Path("results/metrics/swea_mag_joint_clusters_v13_frames256")
OUT=Path("results/metrics/v13_latent_physics_trajectories")
OFFSETS=np.arange(-240.,241.,30.,dtype=float)
LATENT_NAMES=("z1","z2","z3")


def parse_args():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument("--manifest",type=Path,default=v7.MANIFEST)
    p.add_argument("--mag-dir",type=Path,default=MAG_V13)
    p.add_argument("--swea-dir",type=Path,default=SWEA_V13)
    p.add_argument("--output-dir",type=Path,default=OUT)
    p.add_argument("--mag-cache-dir",type=Path,default=v7.MAG_CACHE)
    p.add_argument("--swea-cache-dir",type=Path,default=v7.SWEA_CACHE)
    p.add_argument("--events-per-fold",type=int,default=0)
    p.add_argument("--plot-examples",type=int,default=4)
    p.add_argument("--batch-size",type=int,default=32)
    p.add_argument("--torch-threads",type=int,default=2)
    p.add_argument("--require-complete-cache",action="store_true")
    p.add_argument("--execute",action="store_true")
    a=p.parse_args()
    if (a.events_per_fold<0 or a.plot_examples<0 or
            a.batch_size<=0 or a.torch_threads<=0):
        p.error("Invalid V13 interpretation parameters.")
    return a


def csv_by_time(path):
    if not path.is_file():
        raise FileNotFoundError(f"Missing prior V13 event table: {path}")
    result={}
    with path.open(newline="",encoding="utf-8") as fp:
        for row in csv.DictReader(fp):
            t=row["mpb_timestamp_utc"]
            if t in result:
                raise RuntimeError(f"Duplicate crossing timestamp in {path}: {t}")
            result[t]=row
    return result


def _latent(row):
    if row is None or not all(row.get(k) not in ("",None) for k in LATENT_NAMES):
        return None
    try:
        z=np.array([float(row[k]) for k in LATENT_NAMES])
    except (TypeError,ValueError):
        return None
    return z if np.isfinite(z).all() else None


def _group(row):
    if not row or row.get("cluster","") in ("",None):
        return ""
    try:
        x=int(float(row["cluster"]))
    except (ValueError,TypeError):
        raise ValueError("Invalid original V13 group label.")
    return x


def load_existing_model(path,sensor):
    if not path.is_file():
        raise FileNotFoundError(f"Missing trained {sensor} V13 encoder: {path}")
    checkpoint=torch.load(path,map_location="cpu",weights_only=False)
    if "model" not in checkpoint or "normalizer" not in checkpoint:
        raise RuntimeError(f"Incomplete {sensor} model checkpoint.")
    model=Conv3DAutoencoder() if sensor=="MAG" else SWEA3DAutoencoder()
    model.load_state_dict(checkpoint["model"])
    model.eval()
    norm=checkpoint["normalizer"]
    expected=4 if sensor=="MAG" else 12
    for field in ("mean","scale"):
        x=np.asarray(norm[field],dtype=float)
        if x.shape!=(expected,) or not np.isfinite(x).all():
            raise RuntimeError(f"Invalid {sensor} normalizer {field}.")
    if np.any(np.asarray(norm["scale"])<=0):
        raise RuntimeError(f"Non-positive {sensor} scale.")
    return model,norm


def load_only_cached_events(selected,a):
    """Use provenance-checking cache APIs; never query BigQuery implicitly."""
    sql=v7.query_sources()
    caches={
        "MAG":v7.MagEventCache(
            a.mag_cache_dir,project=bq.PROJECT,
            table=bq.MAG,sql=sql["MAG"]
        ),
        "SWEA":v7.SweaEventCache(
            a.swea_cache_dir,project=bq.PROJECT,
            table=bq.SWEA+" + "+bq.AXES,
            sql=sql["SWEA"]
        ),
    }
    good={}
    missing=[]
    for row in selected:
        begin,end=v7.event_bounds(row)
        parts={}
        for sensor in ("MAG","SWEA"):
            values=caches[sensor].load(
                row["time"],begin,end,max_rows=(
                    25000 if sensor=="MAG" else 2000
                )
            )
            if values is None or len(values[0])==0:
                missing.append({
                    "mpb_timestamp_utc":row["time"].isoformat(),
                    "sensor":sensor,
                    "reason":"missing_or_empty_native_cache"
                })
            else:
                parts[sensor]=values
        if len(parts)==2:
            good[row["time"]]=parts
    if a.require_complete_cache and missing:
        raise RuntimeError(
            f"{len(missing)} cached instrument blocks are missing. "
            "No BigQuery was queried. Check cache coverage."
        )
    return good,missing


def _float(value):
    try:
        x=float(value)
        return x if np.isfinite(x) else np.nan
    except (ValueError,TypeError):
        return np.nan


@torch.no_grad()
def event_latent_paths(row,blocks,models,normalizers,batch_size,mpb_s,bs_s):
    """Apply the SAME catalog safeguard as V13 when shifting clip centers."""
    center=row["time"].timestamp()
    examples=[]
    skipped=Counter()
    for shift in OFFSETS:
        candidate=classify_candidate(
            row,center+shift,all_mpb_s=mpb_s,all_bs_s=bs_s
        )
        if candidate.kind not in ("positive","mpb_ambiguous"):
            skipped[str(candidate.kind)]+=1
            continue
        clip=extract_clip(
            blocks["MAG"],blocks["SWEA"],candidate.center_s
        )
        if clip is None:
            skipped["no_joint_clip"]+=1
            continue
        examples.append((float(shift),clip))
    paths={}
    for sensor in ("MAG","SWEA"):
        if not examples:
            paths[sensor]=[]
            continue
        paths[sensor]=[]
        model=models[sensor]
        collate=mag_collate if sensor=="MAG" else swea_collate
        for start in range(0,len(examples),batch_size):
            subset=examples[start:start+batch_size]
            batch=collate([
                (clip,1.,-shift) for shift,clip in subset
            ],normalizers[sensor])
            out=model(batch["mag"],batch["mask"])
            pred=out["reconstruction"]
            target=batch["mag"]
            mask=batch["mask"][:,None,:].to(pred.dtype)
            mse=((pred-target)**2*mask).sum((1,2))/(
                mask.sum((1,2))*target.shape[1]
            ).clamp_min(1)
            z=out["latent"].numpy()
            prob=torch.sigmoid(out["presence_logit"]).numpy()
            rms=mse.numpy()
            for j,(delta,clip) in enumerate(subset):
                paths[sensor].append({
                    "center_offset_s":float(delta),
                    "z1":float(z[j,0]),
                    "z2":float(z[j,1]),
                    "z3":float(z[j,2]),
                    "presence_probability":float(prob[j]),
                    "reconstruction_mse":float(rms[j]),
                    "n_native_samples":int(
                        len(clip.mag) if sensor=="MAG" else len(clip.swea)
                    ),
                })
    return paths,dict(skipped)


def grouped_summary(records,prefix,group_key):
    summary={}
    groups=sorted({
        int(row[group_key]) for row in records
        if row.get(group_key,"") not in ("",None)
    })
    fields=sorted({
        key for row in records for key in row
        if key.startswith(prefix) and key not in (
            "mag_n_pre","mag_n_post","swea_n_pre","swea_n_post",
            "mag_total_b_consistency_max_nt",
            "mag_cluster","swea_cluster",
        ) and not key.endswith("_9view_n")
    })
    for group in groups:
        subset=[row for row in records if row.get(group_key)==group]
        metrics={}
        for feature in fields:
            values=np.array([_float(r.get(feature)) for r in subset])
            valid=values[np.isfinite(values)]
            if len(valid):
                metrics[feature]={
                    "n":int(len(valid)),
                    "median":float(np.median(valid)),
                    "q25":float(np.quantile(valid,.25)),
                    "q75":float(np.quantile(valid,.75)),
                }
        summary[str(group)]={"n":len(subset),"features":metrics}
    return summary


def paired_correlations(rows,sensors=("MAG","SWEA")):
    correlations=[]
    for sensor in sensors:
        prefix=sensor.lower()+"_"
        # Physical features only: exclude encoder errors, pads, labels
        # and any columns added as embedding-reproduction diagnostics.
        excluded={
            "mag_n_pre","mag_n_post","swea_n_pre","swea_n_post",
            "mag_total_b_consistency_max_nt",
            "mag_cluster","swea_cluster",
        }
        fields=sorted({
            k for row in rows for k in row
            if k.startswith(prefix) and k not in excluded
            and not k.startswith(prefix+"z")
            and not k.endswith("_reconstruction_mse")
            and not k.endswith("_latent_spread")
            and "_9view_" not in k
        })
        targets=[f"{sensor.lower()}z{i}" for i in (1,2,3)]
        for split,subset in (
            ("fit",[r for r in rows if r["split"]=="fit"]),
            ("outer_holdout",[
                r for r in rows if r["split"]=="outer_holdout"
            ]),
        ):
            local=[]
            for target in targets:
                for field in fields:
                    measure=spearman_pair(
                        [_float(r.get(target)) for r in subset],
                        [_float(r.get(field)) for r in subset]
                    )
                    local.append({
                        "split":split,"sensor":sensor,
                        "latent_axis":target,"feature":field,
                        **measure
                    })
            if split=="fit":
                corrected=benjamini_hochberg([
                    r["pvalue"] for r in local
                ])
                for r,q in zip(local,corrected):
                    r["q_fit_bh"]=float(q) if np.isfinite(q) else ""
            else:
                for r in local:
                    r["q_fit_bh"]=""
            correlations.extend(local)
    return correlations


def original_embedding_differences(paths,source_row):
    """Check whether 9 original +/-240 s views reproduce stored event z."""
    z=_latent(source_row)
    if z is None:
        return np.nan,0
    chosen=[
        [row[f"z{i}"] for i in (1,2,3)]
        for row in paths
        if abs(row["center_offset_s"]/60.-
               round(row["center_offset_s"]/60.))<1e-6
    ]
    if not chosen:
        return np.nan,0
    return float(np.linalg.norm(
        np.mean(np.asarray(chosen),axis=0)-z
    )),len(chosen)


def _save_table(path,records):
    path.parent.mkdir(parents=True,exist_ok=True)
    if not records:
        return
    cols=list(dict.fromkeys(k for record in records for k in record))
    with path.open("w",newline="",encoding="utf-8") as fh:
        writer=csv.DictWriter(fh,fieldnames=cols,extrasaction="ignore")
        writer.writeheader()
        writer.writerows(records)


def _plot_heatmap(corrs,split,sensor,path):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    selected=[r for r in corrs if r["split"]==split and r["sensor"]==sensor]
    if not selected:
        return
    features=sorted({r["feature"] for r in selected})
    axes=[f"{sensor.lower()}z{i}" for i in (1,2,3)]
    matrix=np.full((len(features),3),np.nan)
    by_key={(r["feature"],r["latent_axis"]):r for r in selected}
    for i,feature in enumerate(features):
        for j,axis in enumerate(axes):
            matrix[i,j]=by_key[(feature,axis)]["rho"]
    fig,ax=plt.subplots(figsize=(9,max(5,.36*len(features))))
    im=ax.imshow(matrix,aspect="auto",vmin=-1,vmax=1,cmap="coolwarm")
    ax.set_yticks(np.arange(len(features)),features,fontsize=8)
    ax.set_xticks(range(3),axes)
    ax.set_title(f"{sensor} / Spearman {split} (solo muestra descriptiva)")
    fig.colorbar(im,ax=ax,label="rho")
    fig.tight_layout()
    fig.savefig(path,dpi=160)
    plt.close(fig)


def _plot_group_features(rows,folder):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    definitions=[
        ("mag_cluster",[
            "mag_abs_delta_b_nt","mag_rotation_deg",
            "mag_post_std_b_nt","mag_post_rms_db_nt"
        ],"MAG"),
        ("swea_cluster",[
            "swea_abs_spectral_delta_logflux",
            "swea_delta_hardness_logdiff",
            "swea_post_temporal_std_logflux",
            "swea_delta_mean_logflux"
        ],"SWEA"),
    ]
    fit=[row for row in rows if row["split"]=="fit"]
    for gkey,features,sensor in definitions:
        groups=sorted(set(
            int(r[gkey]) for r in fit
            if r.get(gkey) not in ("",None)
        ))
        if not groups:
            continue
        fig,axes=plt.subplots(2,2,figsize=(11,8))
        for ax,feature in zip(axes.flatten(),features):
            series=[]
            for group in groups:
                x=[_float(r.get(feature)) for r in fit if r[gkey]==group]
                x=np.asarray(x)
                series.append(x[np.isfinite(x)])
            available=[(g,x) for g,x in zip(groups,series) if len(x)>0]
            if available:
                ax.boxplot(
                    [x for _,x in available],
                    labels=[f"G{g}" for g,_ in available],
                    showfliers=False
                )
            ax.set_title(feature,fontsize=10)
            ax.grid(alpha=.2)
        fig.suptitle(f"{sensor}: rasgos medidos por grupo (FIT)")
        fig.tight_layout()
        fig.savefig(folder/f"{sensor.lower()}_physical_group_boxplots.png",dpi=170)
        plt.close(fig)


def _plot_group_trajectories(paths,folder):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    for sensor,gfield in (("MAG","mag_cluster"),("SWEA","swea_cluster")):
        rows=[r for r in paths if r["sensor"]==sensor
              and r["split"]=="fit" and r[gfield]!=""]
        labels=sorted(set(r[gfield] for r in rows))
        if not labels:
            continue
        fig,axs=plt.subplots(1,3,figsize=(13,4),sharex=True)
        for group in labels:
            sub=[r for r in rows if r[gfield]==group]
            for j in range(3):
                pos=defaultdict(list)
                for r in sub:
                    pos[float(r["center_offset_s"])].append(
                        r[f"z{j+1}"]
                    )
                x=[];median=[];q25=[];q75=[]
                for t in sorted(pos):
                    values=np.asarray(pos[t])
                    if len(values)<3:
                        continue
                    x.append(t)
                    median.append(float(np.median(values)))
                    q25.append(float(np.percentile(values,25)))
                    q75.append(float(np.percentile(values,75)))
                if x:
                    line=axs[j].plot(
                        x,median,label=f"Grupo {group}")[0]
                    axs[j].fill_between(
                        x,q25,q75,color=line.get_color(),alpha=.14
                    )
                axs[j].axvline(0,linestyle="--",linewidth=1)
                axs[j].set_title(f"{sensor} z{j+1}")
                axs[j].set_xlabel("Desplazamiento del centro (s)")
                axs[j].grid(alpha=.2)
        axs[0].legend()
        fig.suptitle(
            f"{sensor} / trayectoria mediana con Q25-Q75 (FIT)"
        )
        fig.tight_layout()
        fig.savefig(
            folder/f"{sensor.lower()}_median_latent_trajectories.png",
            dpi=170
        )
        plt.close(fig)


def _plot_example(row,block,paths,folder):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    from matplotlib.colors import Normalize
    fig=plt.figure(figsize=(12,8))
    t=row["time"].timestamp()
    mag_t,mag_b=block["MAG"]
    swea_t,swea_flux=block["SWEA"]
    mag_mask=(mag_t>=t-300)&(mag_t<=t+300)
    swea_mask=(swea_t>=t-300)&(swea_t<=t+300)
    ax1=fig.add_subplot(221)
    if mag_mask.any():
        tr=mag_t[mag_mask]-t
        x=mag_b[mag_mask]
        for j,label in enumerate(("Bx","By","Bz","|B|")):
            ax1.plot(tr,x[:,j],label=label,linewidth=.9)
        ax1.legend(fontsize=8,ncol=2)
    ax1.axvline(0,linestyle="--")
    ax1.set_xlabel("Tiempo respecto a MPB (s)")
    ax1.set_ylabel("B (nT)")
    ax1.set_title("MAG medido: 10 minutos")
    ax2=fig.add_subplot(222)
    if swea_mask.any():
        times=swea_t[swea_mask]-t
        values=swea_flux[swea_mask]
        xx=np.repeat(times,len(ENERGY_EV))
        yy=np.tile(ENERGY_EV,len(times))
        sc=ax2.scatter(
            xx,yy,c=values.ravel(),s=9,marker="s"
        )
        fig.colorbar(sc,ax=ax2,label="log1p(flujo SWEA)")
    ax2.axvline(0,linestyle="--")
    ax2.set_yscale("log")
    ax2.set_xlabel("Tiempo respecto a MPB (s)")
    ax2.set_ylabel("Energía (eV)")
    ax2.set_title("SWEA: muestras reales, sin interpolar tiempo")
    for i,sensor in enumerate(("MAG","SWEA")):
        ax=fig.add_subplot(2,2,i+3,projection="3d")
        points=paths[sensor]
        if len(points)>1:
            z=np.asarray([
                [r["z1"],r["z2"],r["z3"]] for r in points
            ])
            offsets=np.asarray([r["center_offset_s"] for r in points])
            ax.plot(*z.T,alpha=.6)
            sc=ax.scatter(
                *z.T,c=offsets,cmap="viridis",s=30,
                norm=Normalize(vmin=-240,vmax=240)
            )
            fig.colorbar(
                sc,ax=ax,label="Centro de ventana vs MPB (s)",
                fraction=.035
            )
        ax.set_xlabel("z1")
        ax.set_ylabel("z2")
        ax.set_zlabel("z3")
        ax.set_title(f"Trayectoria {sensor}")
    fig.suptitle(
        f"{row['mpb_timestamp_utc']} / MAG {row['mag_cluster']} / "
        f"SWEA {row['swea_cluster']}"
    )
    fig.tight_layout()
    filename=row["mpb_timestamp_utc"].replace(":","").replace("+","_")
    fig.savefig(folder/f"{filename}.png",dpi=160)
    plt.close(fig)


def choose_representatives(records,paths,how_many):
    """Pick typical held-out MAG/SWEA groups; no optimization on fold4."""
    lookup=defaultdict(list)
    for r in records:
        lookup[(r["mpb_timestamp_utc"],"MAG")]=[]
        lookup[(r["mpb_timestamp_utc"],"SWEA")]=[]
    for p in paths:
        lookup[(p["mpb_timestamp_utc"],p["sensor"])].append(p)
    valid=[r for r in records if len(
        lookup[(r["mpb_timestamp_utc"],"MAG")])>=8 and len(
        lookup[(r["mpb_timestamp_utc"],"SWEA")])>=8
    ]
    picks=[]
    candidates=[
        ("mag_cluster",1,"mag"),("mag_cluster",2,"mag"),
        ("swea_cluster",3,"swea"),("swea_cluster",4,"swea"),
    ]
    for gkey,group,prefix in candidates:
        matched=[r for r in valid if r[gkey]==group and
                 r["split"]=="outer_holdout" and
                 r not in picks]
        if not matched:
            matched=[r for r in valid if r[gkey]==group and
                     r not in picks]
        if not matched:
            continue
        z=np.asarray([
            [r[f"{prefix}z{i}"] for i in (1,2,3)]
            for r in matched
        ],dtype=float)
        center=np.nanmedian(z,axis=0)
        distances=np.linalg.norm(z-center,axis=1)
        picks.append(matched[int(np.argmin(distances))])
        if len(picks)>=how_many:
            break
    return picks


def main():
    args=parse_args()
    torch.set_num_threads(args.torch_threads)
    random.seed(42)
    np.random.seed(42)
    torch.manual_seed(42)
    mag_csv=csv_by_time(args.mag_dir/"crossing_latents_groups.csv")
    swea_csv=csv_by_time(args.swea_dir/"swea_latents_groups.csv")
    all_rows=pilot.read_manifest(args.manifest)
    for row in all_rows:
        if not row.get("bs_timestamp_utc"):
            raise RuntimeError("Physical diagnostics require BS catalog times.")
        row["bs_time"]=bq.to_utc(row["bs_timestamp_utc"])
    selected=sorted([
        r for fold in (1,2,3,4)
        for r in v7.chosen_events(
            all_rows,fold,pilot_count=args.events_per_fold
        )
    ],key=lambda r:r["time"])
    for row in selected:
        iso=row["time"].isoformat()
        mag=mag_csv.get(iso)
        swea=swea_csv.get(iso)
        if mag is None or swea is None:
            raise RuntimeError(f"Prior V13 CSV does not contain {iso}.")
        if str(mag["split"])!=str(swea["split"]):
            raise RuntimeError(f"MAG/SWEA split mismatch at {iso}.")
        if int(mag["fold"])!=int(swea["fold"]) or int(mag["fold"])!=row["fold"]:
            raise RuntimeError(f"Manifest fold mismatch at {iso}.")
    if not args.execute:
        print(
            f"PREFLIGHT: {len(selected)} V13 crossing labels matched. "
            "No models loaded, no BigQuery, no analysis executed. "
            "Use --execute.",
            flush=True
        )
        return
    args.output_dir.mkdir(parents=True,exist_ok=True)
    models={}
    norms={}
    for sensor,base in (("MAG",args.mag_dir),("SWEA",args.swea_dir)):
        models[sensor],norms[sensor]=load_existing_model(
            base/"discovery_final.pt",sensor
        )
    blocks,missing=load_only_cached_events(selected,args)
    print(
        f"CACHE: {len(blocks)}/{len(selected)} complete native "
        f"MAG+SWEA blocks. Missing source entries: {len(missing)}. "
        "No BigQuery queries.",
        flush=True
    )
    _save_table(args.output_dir/"missing_cached_events.csv",missing)
    mpb_s=np.asarray([
        r["time"].timestamp() for r in all_rows
    ],dtype=float)
    bs_s=np.asarray([
        r["bs_time"].timestamp() for r in all_rows
    ],dtype=float)

    physics=[]
    trajectories=[]
    geometry=[]
    safety=Counter()
    for n,row in enumerate(selected,1):
        t=row["time"]
        key=t.isoformat()
        original_mag=mag_csv[key]
        original_swea=swea_csv[key]
        magz=_latent(original_mag)
        sweaz=_latent(original_swea)
        base={
            "mpb_timestamp_utc":key,
            "year":t.year,
            "fold":row["fold"],
            "split":original_mag["split"],
            "catalog_group":row.get("source_group",""),
            "mag_cluster":_group(original_mag),
            "swea_cluster":_group(original_swea),
            "magz1":magz[0] if magz is not None else np.nan,
            "magz2":magz[1] if magz is not None else np.nan,
            "magz3":magz[2] if magz is not None else np.nan,
            "sweaz1":sweaz[0] if sweaz is not None else np.nan,
            "sweaz2":sweaz[1] if sweaz is not None else np.nan,
            "sweaz3":sweaz[2] if sweaz is not None else np.nan,
            "mag_reconstruction_mse":_float(
                original_mag["reconstruction_mse"]
            ),
            "swea_reconstruction_mse":_float(
                original_swea["reconstruction_mse"]
            ),
            "mag_latent_spread":_float(original_mag["latent_spread"]),
            "swea_latent_spread":_float(original_swea["latent_spread"]),
            "nearest_bs_dt_s":float(np.min(np.abs(bs_s-t.timestamp()))),
        }
        record=dict(base)
        record["cache_complete"]=t in blocks
        if t not in blocks:
            physics.append(record)
            continue
        for sensor,fn in (
            ("MAG",mag_physical_features),
            ("SWEA",swea_physical_features)
        ):
            times,values=blocks[t][sensor]
            record.update(fn(times,values,t.timestamp()))
        paths,skips=event_latent_paths(
            row,blocks[t],models,norms,args.batch_size,
            mpb_s,bs_s
        )
        safety.update(skips)
        for sensor,source in (
            ("MAG",original_mag),("SWEA",original_swea)
        ):
            path=paths[sensor]
            difference,nine=original_embedding_differences(
                path,source
            )
            record[f"{sensor.lower()}_9view_embedding_l2_delta"]=difference
            record[f"{sensor.lower()}_9view_n"]=nine
            for item in path:
                trajectories.append({
                    **{
                        k:base[k] for k in (
                            "mpb_timestamp_utc","year","fold",
                            "split","mag_cluster","swea_cluster"
                        )
                    },
                    "sensor":sensor,**item
                })
            stat=trajectory_statistics(
                [item["center_offset_s"] for item in path],
                [[item["z1"],item["z2"],item["z3"]]
                 for item in path]
            )
            geometry.append({
                **{
                    k:base[k] for k in (
                        "mpb_timestamp_utc","fold","split",
                        "mag_cluster","swea_cluster"
                    )
                },
                "sensor":sensor,
                "original_embedding_l2_delta":difference,
                **stat
            })
        physics.append(record)
        if n%50==0 or n==len(selected):
            print(
                f"LATENT PHYSICS {n}/{len(selected)}: "
                f"trajectories={len(trajectories)}.",
                flush=True
            )
    _save_table(args.output_dir/"physical_features.csv",physics)
    _save_table(args.output_dir/"latent_trajectories.csv",trajectories)
    _save_table(args.output_dir/"trajectory_geometry.csv",geometry)
    if not trajectories:
        raise RuntimeError("No measured, safe crossing windows were available.")

    correlations=paired_correlations(physics)
    _save_table(args.output_dir/"latent_physics_spearman.csv",correlations)
    for sensor in ("MAG","SWEA"):
        for split in ("fit","outer_holdout"):
            _plot_heatmap(
                correlations,split,sensor,
                args.output_dir/f"{sensor.lower()}_spearman_{split}.png"
            )
    _plot_group_features(physics,args.output_dir)
    _plot_group_trajectories(trajectories,args.output_dir)
    examples=choose_representatives(
        physics,trajectories,args.plot_examples
    )
    folder=args.output_dir/"trajectory_examples"
    folder.mkdir(exist_ok=True)
    trajectory_index=defaultdict(lambda:{"MAG":[],"SWEA":[]})
    for r in trajectories:
        trajectory_index[r["mpb_timestamp_utc"]][r["sensor"]].append(r)
    for item in examples:
        import datetime
        when=datetime.datetime.fromisoformat(item["mpb_timestamp_utc"])
        _plot_example(
            item,blocks[when],
            trajectory_index[item["mpb_timestamp_utc"]],
            folder
        )
    fit_correlations=[
        r for r in correlations if r["split"]=="fit"
        and np.isfinite(_float(r["rho"]))
    ]
    top=sorted(fit_correlations,key=lambda r:-abs(r["rho"]))[:20]
    diffs={}
    for sensor in ("mag","swea"):
        values=[
            _float(r.get(f"{sensor}_9view_embedding_l2_delta"))
            for r in physics
        ]
        arr=np.array(values,dtype=float)
        arr=arr[np.isfinite(arr)]
        diffs[sensor]={
            "n_reconstructed":int(len(arr)),
            "median_l2_discrepancy":float(np.median(arr))
                if len(arr) else None,
            "max_l2_discrepancy":float(arr.max())
                if len(arr) else None,
        }
    report={
        "schema":1,
        "analysis":"V13 3D latent physics and shifted-window trajectories",
        "n_catalog_selected":len(selected),
        "n_complete_cached":len(blocks),
        "n_missing_cache_records":len(missing),
        "n_trajectory_rows":len(trajectories),
        "n_events_with_trajectories":len({
            r["mpb_timestamp_utc"] for r in trajectories
        }),
        "n_event_sensor_trajectories":len(geometry),
        "group_mag_fit_physics":grouped_summary(
            [r for r in physics if r["split"]=="fit"],
            "mag_","mag_cluster"
        ),
        "group_swea_fit_physics":grouped_summary(
            [r for r in physics if r["split"]=="fit"],
            "swea_","swea_cluster"
        ),
        "top20_fit_absolute_spearman":top,
        "embedding_reproduction_diagnostics":diffs,
        "windows_skipped_by_safety":dict(safety),
        "sample_time_offsets_s":OFFSETS.tolist(),
        "feature_intervals_s":{
            "pre":[-240,-60],"post":[60,240]
        },
        "swea_energy_ev":ENERGY_EV.tolist(),
        "n_example_figures":len(examples),
        "example_crossings":[r["mpb_timestamp_utc"] for r in examples],
        "limitations":[
            "Correlations are associations, not causal effects.",
            "Latent axes are model-dependent and not physical quantities.",
            "FIT is exploratory; outer fold 4 correlations are separately reported.",
            "A latent trajectory is parameterized by sliding window CENTER "
            "relative to the catalog MPB; its derivative is not plasma speed.",
            "SWEA cache values are log1p angular-mean flux interpolated in "
            "ENERGY only to twelve 30-300 eV bins; original time cadence used.",
            "A catalog MPB timestamp is required to select flanks/windows. "
            "This is not autonomous MPB localization.",
            "Different solar wind states, orbital coverage and acquisition "
            "years may confound cluster-physics associations.",
            "A pre/post magnetic difference cannot be interpreted as a "
            "signed sheath-to-ionosphere jump without crossing direction.",
        ]
    }
    v7.safe_json(args.output_dir/"latent_physics_report.json",report)
    print(
        f"DONE: {len(blocks)} cached events, "
        f"{len(geometry)} sensor trajectories. "
        f"Report: {args.output_dir/'latent_physics_report.json'}",
        flush=True
    )


if __name__=="__main__":
    main()
