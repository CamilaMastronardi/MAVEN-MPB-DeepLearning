#!/usr/bin/env python3
"""V14 per-fold 3D latent visualizations and catalog-relative trajectories.

Important: each outer fold trained DISTINCT MAG/SWEA encoders.
Never pool coordinates across folds before model alignment.
This tool reads only existing latent CSVs; no BigQuery / inference.
"""
from __future__ import annotations

import argparse
import csv
from collections import defaultdict
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np


def load_csv(path):
    with path.open(newline="",encoding="utf-8") as stream:
        return list(csv.DictReader(stream))


def f(row,key):
    return float(row[key])


def scatter_3d(rows,sensor,path,fold):
    # Use only *predicted-centered* views for prospective grouping.
    selected=[r for r in rows if r["view"]=="predicted_centered"]
    if not selected:
        raise ValueError("No predicted-centered embeddings for this fold.")
    groups=sorted({r["source_group"] for r in selected})
    fig=plt.figure(figsize=(8,6))
    ax=fig.add_subplot(111,projection="3d")
    colors=plt.get_cmap("tab10")
    for i,group in enumerate(groups):
        part=[r for r in selected if r["source_group"]==group]
        p=np.asarray([[f(r,f"{sensor}_z{k}") for k in (1,2,3)]
                      for r in part])
        ax.scatter(p[:,0],p[:,1],p[:,2],alpha=.75,
                   s=25,label=f"{group} (n={len(part)})",
                   color=colors(i%10))
    ax.set_xlabel(f"{sensor.upper()} z1")
    ax.set_ylabel(f"{sensor.upper()} z2")
    ax.set_zlabel(f"{sensor.upper()} z3")
    ax.set_title(f"V14 fold {fold}: {sensor.upper()} latent at predicted center")
    ax.legend(loc="best",fontsize=8)
    fig.tight_layout()
    fig.savefig(path,dpi=170)
    plt.close(fig)


def errors_3d(rows,sensor,path,fold):
    selected=[r for r in rows if r["view"]=="predicted_centered"]
    coords=np.asarray([[f(r,f"{sensor}_z{k}") for k in (1,2,3)]
                       for r in selected])
    errors=np.asarray([f(r,"absolute_error_s") for r in selected])
    fig=plt.figure(figsize=(8,6))
    ax=fig.add_subplot(111,projection="3d")
    pts=ax.scatter(coords[:,0],coords[:,1],coords[:,2],
                   c=np.minimum(errors,600.),cmap="viridis",
                   alpha=.85,s=30)
    fig.colorbar(pts,ax=ax,shrink=.7,label="Absolute MPB error (s, capped at 600)")
    ax.set_xlabel(f"{sensor.upper()} z1")
    ax.set_ylabel(f"{sensor.upper()} z2")
    ax.set_zlabel(f"{sensor.upper()} z3")
    ax.set_title(f"V14 fold {fold}: error and {sensor.upper()} latent")
    fig.tight_layout()
    fig.savefig(path,dpi=170)
    plt.close(fig)


def median_trajectories(rows,sensor,path,fold):
    curves=defaultdict(list)
    for row in rows:
        if row["view"] in ("shifted","catalog_centered"):
            curves[f(row,"center_offset_s")].append(
                [f(row,f"{sensor}_z{k}") for k in (1,2,3)]
            )
    if not curves:
        return
    offsets=sorted(curves)
    med,q25,q75=[],[],[]
    for delta in offsets:
        sample=np.asarray(curves[delta])
        med.append(np.median(sample,axis=0))
        q25.append(np.percentile(sample,25,axis=0))
        q75.append(np.percentile(sample,75,axis=0))
    med,q25,q75=map(np.asarray,(med,q25,q75))
    fig,axes=plt.subplots(3,1,figsize=(9,8),sharex=True)
    for i,ax in enumerate(axes):
        ax.plot(offsets,med[:,i],label=f"z{i+1}")
        ax.fill_between(offsets,q25[:,i],q75[:,i],alpha=.22)
        ax.axvline(0,linestyle="--",color="gray",linewidth=1)
        ax.set_ylabel(f"{sensor.upper()} z{i+1}")
        ax.legend(loc="best")
    axes[-1].set_xlabel("Window center - catalog MPB (s)")
    fig.suptitle(
        f"V14 fold {fold}: {sensor.upper()} median latent trajectories "
        "(catalog-aligned, retrospective)"
    )
    fig.tight_layout()
    fig.savefig(path,dpi=170)
    plt.close(fig)


def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument("--results-dir",type=Path,
                   default=Path("results/metrics/v14_dual_3d_refiner"))
    p.add_argument("--output-dir",type=Path,default=None)
    args=p.parse_args()
    paths=sorted(args.results_dir.glob("fold_*_latent_embeddings.csv"))
    if not paths:
        p.error("No V14 latent CSVs. Complete at least one fold first.")
    root=args.output_dir or args.results_dir/"figures"
    for path in paths:
        fold=path.name.split("_")[1]
        rows=load_csv(path)
        folds={r["fold"] for r in rows}
        if folds!={str(int(fold))}:
            raise RuntimeError("Mixed fold embeddings would be invalid.")
        target=root/f"fold_{fold}"
        target.mkdir(parents=True,exist_ok=True)
        for sensor in ("mag","swea"):
            scatter_3d(rows,sensor,target/f"{sensor}_latent_3d.png",fold)
            errors_3d(rows,sensor,target/f"{sensor}_latent_error.png",fold)
            median_trajectories(
                rows,sensor,target/f"{sensor}_latent_trajectories.png",fold
            )
        print(f"V14 fold {fold}: plotted {len(rows)} latent views to {target}")


if __name__=="__main__":
    main()
