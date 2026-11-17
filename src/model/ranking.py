"""Event-wise MPB ranking and cheap online hard-negative mining.

An event is the ranking unit: compare annotated-MPB candidates with safe
weak-negative candidates from THE SAME measured MAG/SWEA event. The
positive/negative labels are NOT used as network inputs. Online mining
remembers high-scoring TRAIN negatives observed on previous epochs; no
validation or outer-fold samples enter this memory.
"""
from __future__ import annotations

import random
from collections import defaultdict

import numpy as np
import torch
from torch.nn import functional as F


def event_pairwise_ranking_loss(logits, labels, event_ids, *, margin=0.5):
    """Differentiable mean softplus(margin + negative - positive) per event.

    Equal event weighting avoids favoring events with many safe controls.
    Uses *logits*, not sigmoid probabilities, to avoid saturation. An event
    with only one class is an error, never silently scored as easy.
    """
    if logits.ndim != 1 or labels.shape != logits.shape:
        raise ValueError("Ranking requires aligned 1D logits and labels.")
    if len(event_ids) != logits.numel() or margin < 0:
        raise ValueError("Invalid event ids or ranking margin.")
    event_positions = defaultdict(list)
    for idx, event in enumerate(event_ids):
        event_positions[event].append(idx)
    if not event_positions:
        raise ValueError("Ranking requires at least one event.")
    event_losses, pair_accuracies, top1 = [], [], []
    for indices in event_positions.values():
        idx = torch.as_tensor(indices, dtype=torch.long, device=logits.device)
        v = logits.index_select(0, idx)
        y = labels.index_select(0, idx)
        positive = v[y > .5]
        negative = v[y < .5]
        if not positive.numel() or not negative.numel():
            raise ValueError("Each ranking group needs positives and negatives.")
        pair_differences = negative[None, :] - positive[:, None]
        event_losses.append(F.softplus(pair_differences + margin).mean())
        pair_accuracies.append(
            (pair_differences.detach() < 0).float().mean()
        )
        top1.append(
            (positive.detach().max() > negative.detach().max()).float()
        )
    return (
        torch.stack(event_losses).mean(),
        float(torch.stack(pair_accuracies).mean().item()),
        float(torch.stack(top1).mean().item()),
        len(event_losses),
    )


def numpy_event_rank_metrics(entries, *, margin=0.5):
    """Monitor-only event ranking, each tuple (event_id, weak_label, logit)."""
    groups = defaultdict(list)
    for event, label, score in entries:
        groups[event].append((int(label), float(score)))
    losses, accuracies, top1 = [], [], []
    for records in groups.values():
        positive = np.asarray(
            [score for label, score in records if label == 1],
            dtype=float,
        )
        negative = np.asarray(
            [score for label, score in records if label == 0],
            dtype=float,
        )
        if not len(positive) or not len(negative):
            continue
        differences = negative[None, :] - positive[:, None]
        losses.append(
            float(np.logaddexp(0., differences + margin).mean())
        )
        accuracies.append(float(np.mean(differences < 0)))
        top1.append(float(positive.max() > negative.max()))
    if not losses:
        raise ValueError("No events with both weak positive and negative labels.")
    return {
        "rank": float(np.mean(losses)),
        "pair_accuracy": float(np.mean(accuracies)),
        "event_top1": float(np.mean(top1)),
        "n_rank_events": len(losses),
    }


def select_rank_event_groups(
    candidate_pool, *, epoch, seed, hard_memory,
    n_positives=2, n_hard=2, n_random=2
):
    """TRAIN-only candidate selection, retaining individual event groups.

    An example tuple has (event_utc, Candidate, DualClip, label, regions).
    Remembers high-scoring negative *candidate centers* from previous
    epochs; exploration draws other safe negatives at random each epoch.
    """
    if min(n_positives, n_hard, n_random) < 0 or n_positives < 1:
        raise ValueError("Invalid per-event sample counts.")
    rng = random.Random(seed + epoch)
    groups = []
    for event, pairs in candidate_pool.items():
        positives = [
            (event, candidate, clip, 1,
             (candidate.left_region, candidate.right_region))
            for candidate, clip in pairs if candidate.transition == 1
        ]
        negatives = [
            (event, candidate, clip, 0,
             (candidate.left_region, candidate.right_region))
            for candidate, clip in pairs if candidate.transition == 0
        ]
        if not positives or not negatives:
            continue
        selected_positives = rng.sample(
            positives, min(n_positives, len(positives))
        )
        memorized = set(hard_memory.get(event.isoformat(), []))
        # Native centers are stored as original float64 seconds.
        known_hard = [
            x for x in negatives if x[1].center_s in memorized
        ]
        if len(known_hard) > n_hard:
            known_hard = rng.sample(known_hard, n_hard)
        taken = {x[1].center_s for x in known_hard}
        unexplored = [
            x for x in negatives if x[1].center_s not in taken
        ]
        random_neg = rng.sample(
            unexplored, min(n_random, len(unexplored))
        )
        taken.update(x[1].center_s for x in random_neg)
        # At the first epoch, fill the unavailable hard slots randomly.
        remaining = [x for x in negatives if x[1].center_s not in taken]
        extra = rng.sample(
            remaining, min(max(0, n_hard-len(known_hard)), len(remaining))
        )
        selected = selected_positives + known_hard + random_neg + extra
        if not any(x[3] == 0 for x in selected):
            raise ValueError("Ranking event has no negative samples.")
        rng.shuffle(selected)
        groups.append((event, selected))
    rng.shuffle(groups)
    return groups


def event_group_batches(groups, batch_size):
    """Pack whole events together, never split event pairs across batches."""
    if batch_size < 1:
        raise ValueError("Batch size must be positive.")
    current, n = [], 0
    for _, group in groups:
        if current and n + len(group) > batch_size:
            yield current
            current, n = [], 0
        if len(group) > batch_size:
            raise ValueError(
                "Batch size too small to hold a complete ranking event."
            )
        current.extend(group)
        n += len(group)
    if current:
        yield current


def update_hard_memory(memory, examples, logits, *, n_hard=2):
    """Store highest scoring TRAIN-only negatives observed in this step.

    Scores are detached from the graph, and event identities are never
    exchanged across folds. This is online *approximate* hard mining:
    prior-epoch hard centers are mixed with fresh random safe controls.
    """
    if len(examples) != logits.numel() or n_hard < 0:
        raise ValueError("Mining logits or memory count disagree.")
    by_event = defaultdict(list)
    scores = logits.detach().float().cpu().tolist()
    for item, score in zip(examples, scores):
        event, candidate, _, label, _ = item
        if label == 0:
            by_event[event.isoformat()].append((
                float(score), float(candidate.center_s)
            ))
    for event, ranked in by_event.items():
        selected = sorted(
            ranked, key=lambda row: (-row[0], row[1])
        )[:n_hard]
        memory[event] = [center for _, center in selected]
