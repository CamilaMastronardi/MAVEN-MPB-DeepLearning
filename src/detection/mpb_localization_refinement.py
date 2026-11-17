"""V8 localization refinement: known-MPB event, improve temporal precision.

Warm-start each corresponding completed V7b fold, maintain its normalizers
and temporal train/monitor/test split. NEW: ordinal near-boundary ranking
between annotated center and neighboring candidate centers (96..448 s).
These ambiguous windows are NOT given binary negative or region labels.
No presence/absence classification or full-orbit detection is claimed.
"""
from __future__ import annotations

import random
from collections import defaultdict

import numpy as np
import torch
from torch.nn import functional as F

from src.data.mpb_dual_windows import collate_dual, extract_clip
from src.data.mpb_weak_regions import classify_candidate
from src.detection.mag_swea_region_model import masked_reconstruction_loss
from src.detection.mpb_event_ranking import (
    event_pairwise_ranking_loss, event_group_batches,
    select_rank_event_groups, update_hard_memory,
)

# Deliberately outside the positive radius (+/-35 s), but within 10 min.
# Soft ordinal constraints only: these are NOT declared physical negatives.
NEAR_OFFSETS_S = (-448., -320., -224., -160., -96.,
                   96., 160., 224., 320., 448.)


def gather_near_candidates(events, blocks, all_mpb_s, all_bs_s):
    """Return only ambiguous near-MPB windows without adjacent boundaries."""
    prepared, exclusions = {}, defaultdict(int)
    for row in events:
        t = row["time"]
        event_s = t.timestamp()
        mag, swea = blocks[t]["MAG"], blocks[t]["SWEA"]
        candidates = []
        for shift in NEAR_OFFSETS_S:
            c = classify_candidate(
                row, event_s+shift, all_mpb_s=all_mpb_s, all_bs_s=all_bs_s
            )
            if c.kind != "mpb_ambiguous":
                exclusions[c.kind] += 1
                continue
            clip = extract_clip(mag, swea, c.center_s)
            if clip is None:
                exclusions["no_joint_coverage"] += 1
                continue
            candidates.append((c, clip))
        prepared[t] = candidates
    return prepared, dict(exclusions)


def _find_center_positive(event, pool):
    """Reference exactly the catalog center when it is a valid weak positive."""
    for candidate, clip in pool.get(event, []):
        if (candidate.transition == 1 and
                abs(candidate.center_s-event.timestamp()) < 0.1):
            return (event, candidate, clip, 1,
                    (candidate.left_region, candidate.right_region))
    return None


def select_localization_groups(
    known_pool, near_pool, *, epoch, seed, hard_memory,
    n_positives=2, n_hard=2, n_random=2
):
    """Sample full event groups: known positives/far controls + near unknowns.

    Draw at most one ambiguous candidate from each temporal side, ensuring
    sign balance as available. Force the exact catalog-center candidate
    into every group for the ordinal objective; do not relabel near windows.
    """
    groups = select_rank_event_groups(
        known_pool, epoch=epoch, seed=seed, hard_memory=hard_memory,
        n_positives=n_positives, n_hard=n_hard, n_random=n_random
    )
    rng = random.Random(seed+epoch+481)
    result = []
    for event, base in groups:
        center = _find_center_positive(event, known_pool)
        selected = list(base)
        if center is not None and not any(
            (item[1].transition == 1 and
             abs(item[1].center_s-event.timestamp()) < 0.1)
            for item in selected
        ):
            pos_idx = next(
                i for i, item in enumerate(selected) if item[3] == 1
            )
            selected[pos_idx] = center
        near = near_pool.get(event, [])
        choices = []
        for sign in (-1,1):
            half = [(c,clip) for c,clip in near
                    if (c.center_s-event.timestamp())*sign > 0]
            if half:
                choices.append(rng.choice(half))
        # Unknown near labels are sentinel None, NOT binary negatives.
        selected.extend(
            (event, c, clip, None, (0,0)) for c,clip in choices
        )
        rng.shuffle(selected)
        result.append((event, selected))
    return result


def _localization_rank(logits, examples, *, margin=0.1):
    """Center should beat sampled near candidate on the same event.

    Margin is deliberately gentle because catalog MPB times have finite
    uncertainty and MPB crossings can be spatially extended.
    """
    per_event = defaultdict(lambda: {"center": [], "near": []})
    for i, (event, candidate, _, label, _) in enumerate(examples):
        if (label == 1 and
                abs(candidate.center_s-event.timestamp()) < .1):
            per_event[event]["center"].append(logits[i])
        elif label is None:
            per_event[event]["near"].append(logits[i])
    losses, wins = [], []
    for pair in per_event.values():
        if not pair["center"] or not pair["near"]:
            continue
        center = torch.stack(pair["center"]).mean()
        near = torch.stack(pair["near"])
        losses.append(F.softplus(margin + near - center).mean())
        wins.append(float((center.detach()>near.detach().max()).item()))
    if not losses:
        # Differentiable zero on same device.
        return logits.sum()*0., None, 0
    return torch.stack(losses).mean(), float(np.mean(wins)), len(wins)


def train_localization_epoch(
    model, optimizer, known_pool, near_pool, normalizers, args,
    device, *, epoch, fold, hard_memory
):
    groups = select_localization_groups(
        known_pool, near_pool, epoch=epoch, seed=args.seed+fold*10000,
        hard_memory=hard_memory,
        n_positives=args.rank_positives_per_event,
        n_hard=args.rank_hard_negatives,
        n_random=args.rank_random_negatives,
    )
    if not groups:
        raise RuntimeError("No complete V8 localization TRAIN events.")
    model.train()
    acc = defaultdict(float)
    total_steps = 0
    n_near_events = 0
    for examples in event_group_batches(groups, args.batch_size):
        batch = collate_dual([
            (item[2], (item[3] if item[3] is not None else 0), item[4])
            for item in examples
        ], normalizers, device)
        known_indices = [
            i for i,item in enumerate(examples) if item[3] is not None
        ]
        known = torch.as_tensor(
            known_indices, device=device, dtype=torch.long
        )
        optimizer.zero_grad(set_to_none=True)
        out = model(
            batch["mag"], batch["mag_mask"], batch["mag_time"],
            batch["swea"], batch["swea_mask"], batch["swea_time"]
        )
        logits = out["transition_logit"]
        det = F.binary_cross_entropy_with_logits(
            logits.index_select(0,known),
            batch["transition"].index_select(0,known)
        )
        region = F.binary_cross_entropy_with_logits(
            out["region_logits"].index_select(0,known),
            batch["regions"].index_select(0,known)
        )
        mag_rec = masked_reconstruction_loss(
            out["mag_reconstruction"], batch["mag"], batch["mag_mask"]
        )
        swea_rec = masked_reconstruction_loss(
            out["swea_reconstruction"], batch["swea"], batch["swea_mask"]
        )
        known_examples = [examples[i] for i in known_indices]
        rank, _, _, _ = event_pairwise_ranking_loss(
            logits.index_select(0,known),
            batch["transition"].index_select(0,known),
            [item[0].isoformat() for item in known_examples],
            margin=args.rank_margin
        )
        near_rank, near_top1, n_events = _localization_rank(
            logits, examples, margin=args.near_margin
        )
        loss = (
            det + args.region_weight*region +
            args.reconstruction_weight*(mag_rec+swea_rec) +
            args.rank_weight*rank + args.near_weight*near_rank
        )
        if not bool(torch.isfinite(loss)):
            raise FloatingPointError("Nonfinite V8 localization loss.")
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), 1.)
        optimizer.step()
        update_hard_memory(
            hard_memory, known_examples, logits.index_select(0,known),
            n_hard=args.rank_hard_negatives
        )
        total_steps += 1
        n_near_events += n_events
        for key,value in (
            ("det",det),("region",region),("far_rank",rank),
            ("near_rank",near_rank),("total",loss)
        ):
            acc[key] += float(value.detach())
        if near_top1 is not None:
            acc["near_top1_weighted"] += near_top1*n_events
    metrics = {key: val/total_steps for key,val in acc.items()
               if key!="near_top1_weighted"}
    metrics["near_top1"] = (
        acc["near_top1_weighted"]/n_near_events
        if n_near_events else None
    )
    metrics["near_event_groups"] = n_near_events
    return metrics


@torch.no_grad()
def monitor_near_ranking(
    model, known_pool, near_pool, normalizers, args, device
):
    """Monitor near ranking ONLY on the reserved inner TRAIN months."""
    model.eval()
    losses, top1 = [], []
    for event in known_pool:
        center = _find_center_positive(event,known_pool)
        near = near_pool.get(event,[])
        if center is None or not near:
            continue
        examples = [center] + [
            (event,c,clip,None,(0,0)) for c,clip in near
        ]
        # All candidates belong to one event, chunk for CPU memory.
        all_logits = []
        for start in range(0,len(examples),args.batch_size):
            part=examples[start:start+args.batch_size]
            batch=collate_dual([
                (item[2], item[3] if item[3] is not None else 0, item[4])
                for item in part
            ],normalizers,device)
            out=model(
                batch["mag"],batch["mag_mask"],batch["mag_time"],
                batch["swea"],batch["swea_mask"],batch["swea_time"]
            )
            all_logits.append(out["transition_logit"])
        logits=torch.cat(all_logits)
        loss, win, count=_localization_rank(
            logits,examples,margin=args.near_margin
        )
        if count:
            losses.append(float(loss))
            top1.append(win)
    if not losses:
        raise RuntimeError("No near-MPB monitor groups; cannot select V8.")
    return {
        "near_rank":float(np.mean(losses)),
        "near_top1":float(np.mean(top1)),
        "near_events":len(losses)
    }
