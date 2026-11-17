"""V11: compact context-aware listwise candidate reranker.

A frozen V7b physics classifier scores every 10-minute MAG+SWEA candidate
inside a known-MPB 70-minute neighborhood. This module computes features
without catalog time/group/direction as network inputs and trains a small
translation-equivariant Conv1d reranker across the *complete candidate
sequence*. Unlike V10 it can select a DIFFERENT candidate rather than
only shifting the original top-1.

The input feature vector is physical and uses only fold-normalized measured
MAG/SWEA and V7b predictions. Target offsets are used only in supervised
loss/evaluation, NEVER concatenated to features.
"""
from __future__ import annotations

import numpy as np
import torch
from torch import nn
from torch.nn import functional as F

from src.detection.mag_swea_region_model import ResidualTimeBlock

N_FEATURES=16


def _masked_mean(x, mask):
    """x: [batch, channels, time], mask: [batch, time]."""
    w=mask[:,None,:].to(x.dtype)
    return (x*w).sum(-1)/w.sum(-1).clamp_min(1.)


@torch.no_grad()
def physical_candidate_features(batch, teacher_outputs):
    """16 strictly local features; no catalog offsets, dates or event index."""
    mag=batch["mag"]
    mag_t=batch["mag_time"]
    mag_m=batch["mag_mask"]
    swea=batch["swea"]
    swea_t=batch["swea_time"]
    swea_m=batch["swea_mask"]
    coarse=teacher_outputs["transition_logit"]
    region=teacher_outputs["region_logits"]
    ml=_masked_mean(mag,mag_m&(mag_t < -90.))
    mr=_masked_mean(mag,mag_m&(mag_t > 90.))
    mc=_masked_mean(mag,mag_m&(mag_t.abs() < 60.))
    mbefore=_masked_mean(mag,mag_m&(mag_t >= -60.)&(mag_t < 0.))
    mafter=_masked_mean(mag,mag_m&(mag_t >= 0.)&(mag_t <= 60.))
    sl=_masked_mean(swea,swea_m&(swea_t < -90.))
    sr=_masked_mean(swea,swea_m&(swea_t > 90.))
    sc=_masked_mean(swea,swea_m&(swea_t.abs() < 60.))
    rl=torch.sigmoid(region[:,0])
    rr=torch.sigmoid(region[:,1])
    sdiff=sr-sl
    mdiff=mr-ml
    fields=(
        coarse,
        region[:,0],
        region[:,1],
        region[:,1]-region[:,0],
        (region[:,1]-region[:,0]).abs(),
        rl*(1.-rr)+(1.-rl)*rr,
        torch.linalg.vector_norm(mdiff[:,:3],dim=-1),
        mdiff[:,3],
        mafter[:,3]-mbefore[:,3],
        mc[:,3]-.5*(ml[:,3]+mr[:,3]),
        (mag[:,3,:].square()*mag_m).sum(-1)/
            mag_m.sum(-1).clamp_min(1),
        sdiff.mean(-1),
        sdiff.abs().mean(-1),
        (sc-.5*(sl+sr)).abs().mean(-1),
        (sr[:,:4]-sl[:,:4]).mean(-1)-
            (sr[:,-4:]-sl[:,-4:]).mean(-1),
        (mr[:,:3].abs()+ml[:,:3].abs()).mean(-1),
    )
    feature=torch.stack(fields,dim=-1)
    if feature.shape[-1]!=N_FEATURES or not torch.isfinite(feature).all():
        raise ValueError("Invalid physical feature shape or nonfinite data.")
    return feature.cpu().numpy().astype(np.float32)


class MPBListwiseReranker(nn.Module):
    """Relative 10-minute evidence + nearby context + global event context.

    No absolute event-center index or positional embedding is supplied.
    Convolutions read neighboring ordered candidate scores/features, while
    global mean context is pooled without location.
    """

    def __init__(self,n_features=N_FEATURES,hidden=32,dropout=.12):
        super().__init__()
        self.embed=nn.Sequential(
            nn.Conv1d(n_features,hidden,1),nn.GELU(),
        )
        self.global_fuse=nn.Sequential(
            nn.Conv1d(2*hidden,hidden,1),nn.GELU()
        )
        self.blocks=nn.Sequential(
            ResidualTimeBlock(hidden,1),
            ResidualTimeBlock(hidden,2),
            nn.Dropout(dropout),
        )
        self.readout=nn.Conv1d(hidden,1,1)

    def forward(self,standardized_features,base_logits):
        if (standardized_features.ndim!=2 or
                standardized_features.shape[1]!=N_FEATURES or
                base_logits.ndim!=1 or
                len(base_logits)!=len(standardized_features)):
            raise ValueError("Candidate feature/logit shape mismatch.")
        if len(base_logits)<2:
            raise ValueError("Listwise ranking requires >=2 candidates.")
        x=standardized_features.T.unsqueeze(0)
        h=self.embed(x)
        context=h.mean(-1,keepdim=True).expand_as(h)
        h=self.global_fuse(torch.cat((h,context),dim=1))
        residual=12.*torch.tanh(
            self.readout(self.blocks(h)).reshape(-1)/12.
        )
        return base_logits+residual,residual


def fit_feature_normalizer(feature_arrays):
    """Fold TRAIN only. Never fit on monitor or external fold."""
    if not feature_arrays:
        raise ValueError("Empty feature fit sample.")
    x=np.concatenate(feature_arrays,axis=0).astype(np.float64)
    if x.ndim!=2 or x.shape[1]!=N_FEATURES or not np.isfinite(x).all():
        raise ValueError("Bad V11 training feature matrix.")
    mean=x.mean(axis=0)
    sigma=x.std(axis=0)
    sigma=np.where(sigma>1e-5,sigma,1.)
    return {"mean":mean.astype(np.float32),
            "scale":sigma.astype(np.float32)}


def tensor_event(record,norm,device):
    x=(record["features"]-norm["mean"])/norm["scale"]
    return (
        torch.as_tensor(x,dtype=torch.float32,device=device),
        torch.as_tensor(
            record["features"][:,0],dtype=torch.float32,device=device
        )
    )


def soft_event_loss(scores,offsets_s,*,sigma_s=56.,temperature=2.0):
    """Gaussian target around annotated time, listwise probability loss.

    Target offsets are NOT model inputs, only target labels. Every candidate
    in a labeled event participates: the loss discourages false distant
    peaks while preserving a finite-width target rather than an exact delta.
    """
    if sigma_s<=0 or temperature<=0:
        raise ValueError("Invalid target Gaussian/score temperature.")
    offsets=torch.as_tensor(
        offsets_s,dtype=scores.dtype,device=scores.device
    )
    if len(offsets)!=len(scores):
        raise ValueError("Score/time-label length mismatch.")
    if not bool((offsets.abs()<=120.).any()):
        raise ValueError("No candidate with catalog MPB within +/-120 s.")
    labels=torch.softmax(-.5*(offsets/sigma_s).square(),dim=0)
    prediction=F.log_softmax(scores/temperature,dim=0)
    return -(labels*prediction).sum()


def select_candidate(scores,base_logits,alpha=1.):
    """No oracle use; alpha=0 is *exact* sigmoid ranking of V7b scores."""
    if len(scores)!=len(base_logits):
        raise ValueError("Scores and teacher predictions differ.")
    if alpha==0:
        probs=torch.sigmoid(torch.as_tensor(
            base_logits,dtype=torch.float32
        )).cpu().numpy()
        return int(np.argmax(probs))
    score=np.asarray(base_logits,dtype=float)+alpha*(
        np.asarray(scores,dtype=float)-
        np.asarray(base_logits,dtype=float)
    )
    return int(np.argmax(score))


def distinct_topk_coverage(base_logits,offsets,*,k=5,min_gap_s=192.):
    """Diagnostic oracle ceiling: never called for model selection."""
    probabilities=torch.sigmoid(torch.as_tensor(
        base_logits,dtype=torch.float32
    )).cpu().numpy()
    order=np.argsort(-probabilities,kind="stable")
    selected=[]
    for j in order:
        if all(abs(offsets[j]-offsets[prev])>=min_gap_s
               for prev in selected):
            selected.append(int(j))
        if len(selected)==k:
            break
    return min(abs(float(offsets[j])) for j in selected)
