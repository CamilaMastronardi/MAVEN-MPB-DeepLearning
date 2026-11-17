#!/usr/bin/env python3
"""Plot V14 MAG/SWEA five-dimensional bottlenecks per outer fold.

PCA is descriptive and fitted on predicted-centered OUTER embeddings,
never on training targets or across folds; it does NOT calibrate V14.
Latent trajectories use the five ORIGINAL learned coordinates and
catalog-relative views (retrospective, not autonomous localization).
Reads only saved CSVs; no BigQuery, no model inference.
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
from sklearn.decomposition import PCA

LATENT=5


def read_csv(path):
    with path.open(newline="",encoding="utf-8") as handle:
        return list(csv.DictReader(handle))


def vector(row,sensor):
    return np.asarray([float(row[f"{sensor}_z{k}"])
                       for k in range(1,LATENT+1)],dtype=float)


def plot_pca(rows,sensor,fold,directory):
    selected=[r for r in rows if r["view"]=="predicted_centered"]
    if len(selected)<3:
        print(f"V14 fold {fold} {sensor}: PCA skipped (<3 predicted views).")
        return
    x=np.stack([vector(r,sensor) for r in selected])
    if not np.isfinite(x).all():
        raise ValueError("Non-finite 5D V14 embedding.")
    pca=PCA(n_components=3)
    z=pca.fit_transform(x)
    ratios=pca.explained_variance_ratio_
    groups=sorted({r["source_group"] for r in selected})
    colors=plt.get_cmap("tab10")
    fig,ax=plt.subplots(figsize=(8,6))
    for i,group in enumerate(groups):
        index=[k for k,r in enumerate(selected)
               if r["source_group"]==group]
        ax.scatter(z[index,0],z[index,1],s=38,alpha=.8,
                   color=colors(i%10),label=f"{group} (n={len(index)})")
    ax.set_xlabel(f"PC1 ({100*ratios[0]:.1f}% variance)")
    ax.set_ylabel(f"PC2 ({100*ratios[1]:.1f}% variance)")
    ax.set_title(f"V14 fold {fold}: {sensor.upper()} 5D latent — PCA (OUTER)")
    ax.legend(fontsize=9)
    fig.tight_layout()
    fig.savefig(directory/f"{sensor}_pca_groups.png",dpi=180)
    plt.close(fig)

    fig=plt.figure(figsize=(8,6))
    ax=fig.add_subplot(111,projection="3d")
    for i,group in enumerate(groups):
        index=[k for k,r in enumerate(selected)
               if r["source_group"]==group]
        ax.scatter(z[index,0],z[index,1],z[index,2],
                   s=38,alpha=.8,
                   color=colors(i%10),label=f"{group} (n={len(index)})")
    for i in range(3):
        getattr(ax,f"set_{'xyz'[i]}label")(
            f"PC{i+1} ({100*ratios[i]:.1f}%)")
    ax.set_title(f"V14 fold {fold}: {sensor.upper()} projected 5D latent")
    ax.legend(fontsize=8)
    fig.tight_layout()
    fig.savefig(directory/f"{sensor}_pca_3d.png",dpi=180)
    plt.close(fig)

    errors=np.asarray([float(r["absolute_error_s"]) for r in selected])
    fig,ax=plt.subplots(figsize=(8,6))
    dot=ax.scatter(z[:,0],z[:,1],c=np.minimum(errors,600.),
                   cmap="viridis",s=42)
    fig.colorbar(dot,ax=ax,label="Error absoluto MPB (s; truncado en 600)")
    ax.set_xlabel(f"PC1 ({100*ratios[0]:.1f}%)")
    ax.set_ylabel(f"PC2 ({100*ratios[1]:.1f}%)")
    ax.set_title(f"V14 fold {fold}: {sensor.upper()} PCA and localization error")
    fig.tight_layout()
    fig.savefig(directory/f"{sensor}_pca_error.png",dpi=180)
    plt.close(fig)
    print(f"V14 fold {fold} {sensor.upper()}: PCA variances "
          f"{ratios[0]:.3f},{ratios[1]:.3f},{ratios[2]:.3f}")


def plot_trajectories(rows,sensor,fold,directory):
    grouped=defaultdict(list)
    for row in rows:
        if row["view"] in ("shifted","catalog_centered"):
            grouped[float(row["center_offset_s"])].append(
                vector(row,sensor))
    if not grouped:
        return
    offsets=sorted(grouped)
    med=np.asarray([np.median(grouped[t],axis=0) for t in offsets])
    q25=np.asarray([np.percentile(grouped[t],25,axis=0) for t in offsets])
    q75=np.asarray([np.percentile(grouped[t],75,axis=0) for t in offsets])
    fig,axes=plt.subplots(LATENT,1,figsize=(9,11),sharex=True)
    for i,ax in enumerate(axes):
        ax.plot(offsets,med[:,i],label=f"z{i+1}")
        ax.fill_between(offsets,q25[:,i],q75[:,i],alpha=.20)
        ax.axvline(0,color="gray",linestyle="--",linewidth=1)
        ax.set_ylabel(f"{sensor.upper()} z{i+1}")
        ax.legend(loc="best")
    axes[-1].set_xlabel("Centro de ventana - t_MPB catalogado (s)")
    fig.suptitle(f"V14 fold {fold}: five {sensor.upper()} latent coordinates\n"
                 "Median and IQR, retrospective label alignment")
    fig.tight_layout()
    fig.savefig(directory/f"{sensor}_five_latent_trajectories.png",dpi=180)
    plt.close(fig)


def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument("--results-dir",type=Path,
                   default=Path("results/metrics/v14_dual_5d_refiner"))
    p.add_argument("--output-dir",type=Path,default=None)
    args=p.parse_args()
    files=sorted(args.results_dir.glob("fold_*_latent_embeddings.csv"))
    if not files:
        p.error("No V14 5D latent CSV files available yet.")
    root=args.output_dir or args.results_dir/"figures"
    for path in files:
        fold=int(path.stem.split("_")[1])
        rows=read_csv(path)
        if not rows or {int(r["fold"]) for r in rows}!={fold}:
            raise RuntimeError("Missing embeddings or mixed fold coordinates.")
        for sensor in ("mag","swea"):
            fields={f"{sensor}_z{i}" for i in range(1,LATENT+1)}
            if not fields.issubset(rows[0].keys()):
                raise RuntimeError(f"{path}: missing {sensor} 5D columns.")
        target=root/f"fold_{fold:02d}"
        target.mkdir(parents=True,exist_ok=True)
        for sensor in ("mag","swea"):
            plot_pca(rows,sensor,fold,target)
            plot_trajectories(rows,sensor,fold,target)
        print(f"V14 fold {fold}: plotted {len(rows)} views in {target}")


if __name__=="__main__":
    main()
