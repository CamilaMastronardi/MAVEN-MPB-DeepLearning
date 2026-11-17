#!/usr/bin/env python3
"""V8: detection-oriented fine-tuning of V7b, then thresholded MPB scans.

Initial benchmark: 70-minute MAG+SWEA blocks centered on catalog crossings,
NOT complete orbits. Sliding 10-min windows are scored at native cadences.
No event timestamp, BS time or weak labels enter model inputs. A peak must
clear a threshold calibrated ONLY on inner-month training events. It may miss
the target; unmatched peaks in weakly verified background are counted.

The full-CV V7b fold checkpoints initialize the network. More hard/random
negative windows with a denser 64-s control grid improve the chance of
rejecting non-MPB fluctuations. All MAG/SWEA sources reuse existing caches.
"""
from __future__ import annotations

import argparse
import csv
import hashlib
import json
import os
import random
import sys
from datetime import datetime
from importlib import import_module
from pathlib import Path

import numpy as np
import torch

ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

v7 = import_module("scripts.5_training.5b_train_mag_swea_regions")
pilot = import_module("scripts.3_windowing.3b_bigquery_window_pilot")
previous = import_module("scripts.5_training.5a_train_mag_cv")
bq = import_module("scripts.1_bigquery.1c_bigquery_event_pilot")
from src.data.mpb_dual_windows import collate_dual, extract_clip
from src.data.mpb_weak_regions import candidate_evaluation_grid
from src.detection.mag_swea_region_model import MagSweaRegionAutoencoder
from src.detection.mpb_scan_detection import (
    evaluate_scans, calibrate_scan_threshold
)

DEFAULT_PRETRAINED = Path("results/metrics/mag_swea_regions_ranking_v1")
DEFAULT_OUTPUT = Path("results/metrics/mag_swea_detection_v8")


def parse_args():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--manifest", type=Path, default=v7.MANIFEST)
    p.add_argument("--pretrained-dir", type=Path, default=DEFAULT_PRETRAINED)
    p.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT)
    p.add_argument("--mag-cache-dir", type=Path, default=v7.MAG_CACHE)
    p.add_argument("--swea-cache-dir", type=Path, default=v7.SWEA_CACHE)
    p.add_argument("--refresh-swea-cache", action="store_true")
    p.add_argument("--folds", default="all", choices=("all","1","2","3","4"))
    p.add_argument("--pilot-events-per-fold", type=int, default=0)
    p.add_argument("--max-epochs", type=int, default=35)
    p.add_argument("--min-epochs", type=int, default=8)
    p.add_argument("--patience", type=int, default=6)
    p.add_argument("--batch-size", type=int, default=16)
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--learning-rate", type=float, default=0.0001)
    p.add_argument("--torch-threads", type=int, default=2)
    p.add_argument("--rank-weight", type=float, default=1.0)
    p.add_argument("--rank-margin", type=float, default=0.5)
    p.add_argument("--rank-control-step-s", type=float, default=64.)
    p.add_argument("--rank-positives-per-event", type=int, default=3)
    p.add_argument("--rank-hard-negatives", type=int, default=5)
    p.add_argument("--rank-random-negatives", type=int, default=5)
    p.add_argument("--region-weight", type=float, default=0.5)
    p.add_argument("--reconstruction-weight", type=float, default=0.15)
    p.add_argument("--evaluation-stride-s", type=float, default=32.)
    p.add_argument("--max-query-gib", type=float, default=2.)
    p.add_argument("--max-total-gib", type=float, default=150.)
    p.add_argument("--max-mag-rows", type=int, default=25000)
    p.add_argument("--max-swea-rows", type=int, default=2000)
    p.add_argument("--resume", action="store_true")
    p.add_argument("--execute", action="store_true")
    args = p.parse_args()
    positive = (
        args.max_epochs, args.min_epochs, args.patience, args.batch_size,
        args.learning_rate, args.evaluation_stride_s, args.torch_threads,
        args.max_query_gib, args.max_total_gib, args.max_mag_rows,
        args.max_swea_rows, args.rank_weight, args.rank_margin,
        args.rank_control_step_s, args.rank_positives_per_event,
        args.rank_hard_negatives, args.rank_random_negatives
    )
    if (min(positive) <= 0 or args.min_epochs > args.max_epochs
            or args.pilot_events_per_fold < 0
            or args.rank_control_step_s > 600.
            or args.region_weight < 0 or args.reconstruction_weight < 0
            or args.batch_size < (
                args.rank_positives_per_event +
                args.rank_hard_negatives +
                args.rank_random_negatives
            )):
        p.error("Invalid training parameters or incomplete ranking minibatch.")
    return args


def json_signature(payload):
    return hashlib.sha256(
        json.dumps(payload, sort_keys=True).encode("utf-8")
    ).hexdigest()


def fold_signature(args, fold, pretrained_path, fit, monitor, val):
    return json_signature({
        "model": "v8-dual-sensor-dense-control-detection-v1",
        "fold": fold, "manifest_sha": v7.sha256_file(args.manifest),
        "pretrained_sha256": v7.sha256_file(pretrained_path),
        "fit": [r["time"].isoformat() for r in fit],
        "monitor": [r["time"].isoformat() for r in monitor],
        "validation": [r["time"].isoformat() for r in val],
        "epochs": [args.max_epochs,args.min_epochs,args.patience],
        "batch_size": args.batch_size,
        "lr": args.learning_rate, "seed": args.seed,
        "loss_weights": [args.rank_weight,args.rank_margin,
                         args.region_weight,args.reconstruction_weight],
        "negatives": [args.rank_control_step_s,
                      args.rank_positives_per_event,
                      args.rank_hard_negatives,args.rank_random_negatives],
        "scan_stride_s": args.evaluation_stride_s
    })


def normalizers_from_checkpoint(state):
    if state.get("normalizers") is None:
        raise ValueError("V7b checkpoint has no fold-specific normalizers.")
    return {
        sensor: {
            "mean": np.asarray(state["normalizers"][sensor]["mean"],
                               dtype=np.float64),
            "scale": np.asarray(state["normalizers"][sensor]["scale"],
                                dtype=np.float64),
            "n_epochs": state["normalizers"][sensor].get("n_epochs"),
        } for sensor in ("MAG","SWEA")
    }


@torch.no_grad()
def scan_events(model, events, blocks, norm, all_mpb_s, all_bs_s, args, device):
    """Score full available 70-min blocks; never choose a target-centered max."""
    model.eval()
    scans = []
    for i, row in enumerate(events, 1):
        when = row["time"]
        mag = blocks[when]["MAG"]
        swea = blocks[when]["SWEA"]
        candidates = candidate_evaluation_grid(
            row, all_mpb_s, all_bs_s, step_s=args.evaluation_stride_s
        )
        kept_times, kept_scores = [], []
        valid = []
        for candidate in candidates:
            clip = extract_clip(mag, swea, candidate.center_s)
            if clip is not None:
                valid.append((candidate, clip))
        for start in range(0, len(valid), args.batch_size):
            group = valid[start:start+args.batch_size]
            items = [(clip, 0, (0, 0)) for _,clip in group]
            batch = collate_dual(items, norm, device)
            out = model(
                batch["mag"], batch["mag_mask"], batch["mag_time"],
                batch["swea"], batch["swea_mask"], batch["swea_time"]
            )
            scores = torch.sigmoid(out["transition_logit"]).cpu().numpy()
            kept_times.extend(float(candidate.center_s)
                              for candidate,_ in group)
            kept_scores.extend(map(float, scores))
        scans.append({
            "event_id": when.isoformat(),
            "target_s": when.timestamp(),
            "times_s": kept_times,
            "scores": kept_scores,
        })
        if i%50 == 0 or i==len(events):
            print(f"SCANNED {i}/{len(events)} 70-min event blocks.",
                  flush=True)
    return scans


def train_fold(fold, parts, blocks, rows, args, device):
    fit, monitor, removed = previous.split_inner_monitor_months(
        parts["train"], fold, args.seed
    )
    val = parts["val"]
    pre_path = args.pretrained_dir / f"fold_{fold:02d}_final.pt"
    if not pre_path.exists():
        raise FileNotFoundError(
            f"{pre_path}: run the completed V7b full training first."
        )
    signature = fold_signature(
        args, fold, pre_path, fit, monitor, val
    )
    saved = torch.load(pre_path, map_location="cpu", weights_only=False)
    if saved.get("fold") != fold or "model" not in saved:
        raise RuntimeError("V7b source checkpoint fold/model mismatch.")
    norm = normalizers_from_checkpoint(saved)
    all_mpb_s = np.asarray([r["time"].timestamp() for r in rows],
                           dtype=np.float64)
    all_bs_s = np.asarray([r["bs_time"].timestamp() for r in rows],
                          dtype=np.float64)
    fit_pool, fit_skips = v7.gather_candidates(
        fit, blocks, all_mpb_s, all_bs_s, ranking=True,
        rank_control_step_s=args.rank_control_step_s
    )
    mon_pool, mon_skips = v7.gather_candidates(
        monitor, blocks, all_mpb_s, all_bs_s, ranking=True,
        rank_control_step_s=args.rank_control_step_s
    )
    if not fit_pool or not mon_pool:
        raise RuntimeError(f"FOLD {fold}: insufficient labeled TRAIN controls.")
    torch.manual_seed(args.seed + fold)
    model = MagSweaRegionAutoencoder().to(device)
    model.load_state_dict(saved["model"])
    optimizer = torch.optim.AdamW(
        model.parameters(), lr=args.learning_rate, weight_decay=1e-3
    )
    folder = args.output_dir
    last_path = folder / f"fold_{fold:02d}_last.pt"
    best_path = folder / f"fold_{fold:02d}_best.pt"
    final_path = folder / f"fold_{fold:02d}_final.pt"
    start_epoch, best_epoch, stale, best_loss = 0, 0, 0, float("inf")
    history, hard_memory = [], {}
    if last_path.exists():
        if not args.resume:
            raise RuntimeError(f"{last_path} exists; use --resume.")
        state = torch.load(last_path, map_location=device, weights_only=False)
        if state["signature"] != signature:
            raise RuntimeError("V8 resume signature mismatch.")
        model.load_state_dict(state["model"])
        optimizer.load_state_dict(state["optimizer"])
        start_epoch = state["epoch"]
        best_epoch, stale = state["best_epoch"], state["stale"]
        best_loss = state["best_loss"]
        history, hard_memory = state["history"], state["hard_memory"]
        print(f"FOLD {fold}: resumed at epoch {start_epoch}.", flush=True)
    print(
        f"FOLD {fold}: fit {len(fit)} (usable {len(fit_pool)}), "
        f"monitor {len(monitor)} (usable {len(mon_pool)}), "
        f"outer test {len(val)}; excluded overlapping train {removed}.",
        flush=True
    )
    for epoch in range(start_epoch+1,args.max_epochs+1):
        if epoch >= args.min_epochs and stale >= args.patience:
            break
        tr = v7.train_epoch(
            model, optimizer, fit_pool, norm, args, device,
            epoch, fold, hard_memory=hard_memory
        )
        mon, _, _ = v7.monitor_epoch(
            model, mon_pool, norm, args, device
        )
        if epoch >= args.min_epochs:
            if mon["select_loss"] < best_loss-1e-4:
                best_loss, best_epoch, stale = mon["select_loss"], epoch, 0
                v7.safe_checkpoint(best_path, {
                    "signature": signature, "epoch": epoch,
                    "model": model.state_dict()
                })
            else:
                stale += 1
        history.append({
            "epoch": epoch, "train_det": tr["det"],
            "train_rank": tr["rank"],
            "train_region": tr["region"],
            "monitor_det": mon["det"],
            "monitor_rank": mon["rank"],
            "monitor_region": mon["region"],
            "monitor_select": mon["select_loss"],
            "monitor_event_top1": mon["event_top1"],
            "best_epoch": best_epoch,
        })
        v7.safe_checkpoint(last_path, {
            "signature": signature, "epoch": epoch,
            "best_epoch": best_epoch, "stale": stale,
            "best_loss": best_loss, "history": history,
            "hard_memory": hard_memory,
            "model": model.state_dict(),
            "optimizer": optimizer.state_dict(),
        })
        print(
            f"FOLD {fold} EPOCH {epoch}: train_det={tr['det']:.4f} "
            f"train_rank={tr['rank']:.4f} monitor_det={mon['det']:.4f} "
            f"monitor_rank={mon['rank']:.4f} "
            f"monitor_select={mon['select_loss']:.4f} "
            f"best={best_epoch} stale={stale}", flush=True
        )
        if epoch >= args.min_epochs and stale >= args.patience:
            break

    if not best_path.exists():
        raise RuntimeError(f"FOLD {fold}: no valid V8 best checkpoint.")
    best = torch.load(best_path,map_location=device,weights_only=False)
    if best["signature"] != signature:
        raise RuntimeError("V8 selected checkpoint signature mismatch.")
    model.load_state_dict(best["model"])
    del fit_pool, mon_pool

    # Do NOT calibrate on outer-fold validation scans.
    mon_scans = scan_events(
        model, monitor, blocks, norm, all_mpb_s, all_bs_s, args, device
    )
    threshold, calibration = calibrate_scan_threshold(
        mon_scans, all_mpb_s, all_bs_s
    )
    print(
        f"FOLD {fold}: inner-monitor scan threshold={threshold:.3f}; "
        f"monitor weak F1={calibration['weak_detection_f1']:.3f}.",
        flush=True
    )
    del mon_scans
    outer_scans = scan_events(
        model, val, blocks, norm, all_mpb_s, all_bs_s, args, device
    )
    measured, event_records = evaluate_scans(
        outer_scans, all_mpb_s, all_bs_s, threshold=threshold
    )
    del outer_scans
    metrics = {
        "schema": 1,
        "architecture": "dual-mag-swea-dense-control-v8",
        "fold": fold, "signature": signature,
        "v7b_source_checkpoint": str(pre_path),
        "best_epoch": best_epoch, "epochs_completed": len(history),
        "train_events": len(fit), "monitor_events": len(monitor),
        "outer_events": len(val),
        "fit_skips": fit_skips, "monitor_skips": mon_skips,
        "calibrated_inner_threshold": threshold,
        "inner_monitor_scan_metrics": calibration,
        "training_history": history,
        "train_configuration": {
            "rank_step_s": args.rank_control_step_s,
            "n_positive": args.rank_positives_per_event,
            "n_hard_negative": args.rank_hard_negatives,
            "n_random_negative": args.rank_random_negatives,
            "scan_stride_s": args.evaluation_stride_s,
            "scan_min_peak_separation_s": 300.,
            "detection_match_tolerance_s": 120.,
            "unknown_label_exclusion_s": 390.,
            "normalizers": "from the matching V7b TRAIN-only fold",
        },
        **measured
    }
    v7.safe_json(folder / f"fold_{fold:02d}_metrics.json", metrics)
    v7.write_csv(folder / f"fold_{fold:02d}_detection_events.csv",
                 event_records)
    v7.safe_checkpoint(final_path, {
        "schema": 1, "architecture": metrics["architecture"],
        "signature": signature, "fold": fold, "best_epoch": best_epoch,
        "model": model.state_dict(),
        "normalizers": saved["normalizers"],
        "scan_threshold": threshold,
        "scan_min_separation_s": 300.,
    })
    print(
        f"FOLD {fold} DETECTION: TP={measured['target_tp']}/"
        f"{measured['n_target_events']} FN={measured['target_fn']} "
        f"weak-FP={measured['weak_background_fp']} "
        f"unknown-peaks={measured['unverified_peaks']} "
        f"weak-F1={measured['weak_detection_f1']:.3f} "
        f"median_detected_error={measured['median_detected_abs_error_s']}s.",
        flush=True
    )
    return metrics


def main():
    args = parse_args()
    torch.set_num_threads(args.torch_threads)
    torch.manual_seed(args.seed)
    random.seed(args.seed)
    np.random.seed(args.seed)
    device = torch.device("cpu")
    rows = pilot.read_manifest(args.manifest)
    for row in rows:
        if not row.get("bs_timestamp_utc"):
            raise ValueError("V8 requires catalog BS dates to exclude unknowns.")
        row["bs_time"] = bq.to_utc(row["bs_timestamp_utc"])
    folds = [1,2,3,4] if args.folds == "all" else [int(args.folds)]
    parts = {}
    for fold in folds:
        val = v7.chosen_events(
            rows, fold, pilot_count=args.pilot_events_per_fold
        )
        train = sorted(
            [e for f in range(1,5) if f!=fold
             for e in v7.chosen_events(
                 rows,f,pilot_count=args.pilot_events_per_fold
             )],
            key=lambda e:e["time"]
        )
        if len(train)<3 or not val:
            raise RuntimeError("Not enough MAG+SWEA events for this fold.")
        previous.check_disjoint_supports(train,val)
        parts[fold]={"train":train,"val":val}
        path=args.pretrained_dir/f"fold_{fold:02d}_final.pt"
        if not path.exists():
            raise FileNotFoundError(
                f"V7b pretrained source not found: {path}"
            )
    args.output_dir.mkdir(parents=True,exist_ok=True)
    completed, todo={},{}
    for fold, group in parts.items():
        fit, monitor, _ = previous.split_inner_monitor_months(
            group["train"],fold,args.seed
        )
        signature = fold_signature(
            args,fold,
            args.pretrained_dir/f"fold_{fold:02d}_final.pt",
            fit,monitor,group["val"]
        )
        metrics_path = args.output_dir/f"fold_{fold:02d}_metrics.json"
        final_path = args.output_dir/f"fold_{fold:02d}_final.pt"
        if args.resume and metrics_path.exists() and final_path.exists():
            state=json.loads(metrics_path.read_text())
            if state.get("signature")!=signature:
                raise RuntimeError("Completed V8 fold signature mismatch.")
            completed[fold]=state
            print(f"FOLD {fold}: completed, skipping.",flush=True)
        else:
            todo[fold]=group
    if not args.execute:
        print(
            "DRY RUN: source-fold checkpoints verified; use --execute to "
            "train. Cache is never queried in this mode.",
            flush=True
        )
        return
    if todo:
        unique={r["time"]:r for group in todo.values()
                for half in group.values() for r in half}
        print(
            f"V8 {len(unique)} paired source blocks, "
            f"pilot={bool(args.pilot_events_per_fold)}; "
            "reusing V7b source-bound caches when complete.",
            flush=True
        )
        blocks=v7.load_event_blocks(list(unique.values()),args)
        if blocks is None:
            raise RuntimeError("No measured source blocks.")
        for fold,group in todo.items():
            completed[fold]=train_fold(
                fold,group,blocks,rows,args,device
            )
    all_events=[]
    for fold in sorted(completed):
        path=args.output_dir/f"fold_{fold:02d}_detection_events.csv"
        with path.open(newline="",encoding="utf-8") as fp:
            all_events.extend(csv.DictReader(fp))
    n=len(all_events)
    tps=sum(int(r["detected"]) for r in all_events)
    fps=sum(int(r["n_weak_background_false_peaks"]) for r in all_events)
    unverified=sum(int(r["n_unverified_peaks"]) for r in all_events)
    errs=[float(r["absolute_error_s"]) for r in all_events
          if r["absolute_error_s"]!=""]
    p=tps/(tps+fps) if tps+fps else 0.
    rec=tps/n if n else 0.
    summary={
        "schema":1,"model":"v8-MAG-SWEA-dense-control-detector",
        "completed_folds":sorted(completed),
        "pilot":bool(args.pilot_events_per_fold),
        "n_blocks":n,"detected_targets":tps,"missed_targets":n-tps,
        "weak_background_false_peaks":fps,
        "unverified_peaks":unverified,
        "weak_detection_precision":p,"target_recall":rec,
        "weak_detection_f1":2*p*rec/(p+rec) if p+rec else 0.,
        "median_error_detected_only_s":(
            float(np.median(errs)) if errs else None
        ),
        "fraction_all_targets_within_60s":(
            sum(err<=60 for err in errs)/n if n else 0.
        ),
        "fold_results":[{
            "fold":fold,
            "target_tp":completed[fold]["target_tp"],
            "target_fn":completed[fold]["target_fn"],
            "weak_background_fp":completed[fold]["weak_background_fp"],
            "weak_detection_f1":completed[fold]["weak_detection_f1"],
            "calibrated_inner_threshold":
                completed[fold]["calibrated_inner_threshold"]
        } for fold in sorted(completed)],
        "warning":(
            "70-min positive-centered event blocks, NOT autonomous "
            "whole-orbit detection. Catalog-based unknown exclusions make "
            "precision optimistic. Report median only over detections "
            "and FN separately; do not compare this median directly with "
            "V7b oracle-peak median."
        )
    }
    v7.safe_json(args.output_dir/"cv_summary.json",summary)
    print(
        f"V8 GLOBAL DETECTION: targets={tps}/{n} "
        f"weak-FP={fps} unverified={unverified} "
        f"weak-F1={summary['weak_detection_f1']:.3f} "
        f"median detected error={summary['median_error_detected_only_s']}s.",
        flush=True
    )


if __name__ == "__main__":
    main()
