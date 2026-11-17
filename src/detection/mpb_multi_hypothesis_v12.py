"""V12 multi-hypothesis MPB selector: compare V10-refined candidate times.

V7b (frozen) generates a diverse shortlist of physical transition
hypotheses. V10 (frozen) refines EACH hypothesis, including predicted
presence and uncertainty. This network only selects among those
refined hypotheses using permutation-invariant context. It cannot
access the event's annotated center/time or absolute chronology.

Offsets from the catalog are LABELS for training/evaluation ONLY.
Only pairwise differences of predicted physical times enter consensus.
"""
from __future__ import annotations

import numpy as np
import torch
from torch import nn
from torch.nn import functional as F

N_PHYS=16
N_FEATURES=28


def diverse_shortlist(teacher_logit, candidate_times, *, topk=12,
                      min_gap_s=120.):
    """Select teacher-ranked candidates separated in physical time.

    The same teacher sigmoid/float32 tie rules as V7b are used.
    No timestamp relative to a catalog crossing is consulted.
    """
    logits=np.asarray(teacher_logit,dtype=np.float32)
    times=np.asarray(candidate_times,dtype=np.float64)
    if (logits.ndim!=1 or len(logits)!=len(times) or len(logits)<1
            or not np.isfinite(logits).all()
            or not np.isfinite(times).all()
            or topk<1 or min_gap_s<=0):
        raise ValueError("Invalid V12 coarse candidate score/times.")
    probs=torch.sigmoid(torch.as_tensor(logits)).numpy()
    order=np.argsort(-probs,kind="stable")
    keep=[]
    for j in order:
        if all(abs(times[j]-times[i])>=min_gap_s for i in keep):
            keep.append(int(j))
        if len(keep)==topk:
            break
    if not keep:
        raise RuntimeError("V12 could not form candidate shortlist.")
    return np.asarray(keep,dtype=np.int64)


def consensus_features(predicted_times,teacher_logits,presence):
    """Four location-translation-invariant features per hypothesis.

    Invariance under a shared shift is critical: a known catalog MPB
    must not be inferred from the center of the search interval.
    """
    t=np.asarray(predicted_times,dtype=np.float64)
    score=np.asarray(teacher_logits,dtype=np.float64)
    presence=np.asarray(presence,dtype=np.float64)
    if (t.ndim!=1 or len(t)!=len(score) or
            len(t)!=len(presence) or
            not np.isfinite(t).all() or
            not np.isfinite(score).all() or
            not np.isfinite(presence).all()):
        raise ValueError("Invalid V12 timestamp consensus inputs.")
    delta=t[:,None]-t[None,:]
    np.fill_diagonal(delta,np.inf)
    # Other hypotheses only, with confidence derived from data.
    teacher_rank=np.argsort(
        np.argsort(-score,kind="stable"),kind="stable"
    )
    confidence=np.exp(-teacher_rank/4.)*np.clip(presence,.01,1.)
    contributions=[]
    for width in (60.,120.,240.):
        weights=np.exp(-.5*(delta/width)**2)
        contributions.append(
            (weights*confidence[None,:]).sum(axis=1)/
            max(1.,confidence.sum())
        )
    # Strongest non-self agreement across hypotheses.
    near=np.min(np.abs(delta),axis=1)
    nearest=np.exp(-np.minimum(near,2000.)/120.)
    return np.stack((*contributions,nearest),axis=1).astype(np.float32)


def assemble_features(physical,centers,mean_delta,peak_delta,
                      presence_logit,entropy,peak_prob,std_seconds,
                      *,fine_alpha,fine_method):
    """Build measured/model-only features and refined candidate timestamps.

    centers may be Unix seconds or arbitrarily shifted seconds. Their
    *absolute* values are NEVER appended to the neural-network inputs.
    """
    physical=np.asarray(physical,dtype=np.float32)
    centers=np.asarray(centers,dtype=np.float64)
    mean_delta=np.asarray(mean_delta,dtype=np.float64)
    peak_delta=np.asarray(peak_delta,dtype=np.float64)
    presence_logit=np.asarray(presence_logit,dtype=np.float64)
    entropy=np.asarray(entropy,dtype=np.float64)
    peak_prob=np.asarray(peak_prob,dtype=np.float64)
    std_seconds=np.asarray(std_seconds,dtype=np.float64)
    k=len(centers)
    if (physical.shape!=(k,N_PHYS) or
            any(len(x)!=k for x in (
                mean_delta,peak_delta,presence_logit,
                entropy,peak_prob,std_seconds
            )) or k<1):
        raise ValueError("Wrong V12 physical/fine feature dimensions.")
    correction=mean_delta if fine_method=="mean" else peak_delta
    if fine_method not in ("mean","peak") or not 0<=fine_alpha<=1:
        raise ValueError("Invalid calibrated V10 refinement.")
    correction=np.clip(correction,-240.,240.)
    refined_times=centers+fine_alpha*correction
    confidence=1./(1.+np.exp(-np.clip(presence_logit,-30,30)))
    consensus=consensus_features(
        refined_times,physical[:,0],confidence
    )
    order=np.argsort(
        np.argsort(-physical[:,0],kind="stable"),kind="stable"
    )
    rank=order.astype(np.float32)/max(1,k-1)
    local=np.stack((
        np.clip(mean_delta/240.,-1.2,1.2),
        np.clip(peak_delta/240.,-1.2,1.2),
        np.clip(presence_logit/6.,-4.,4.),
        np.clip(entropy,0,1),
        np.clip(peak_prob*100.,0,10),
        np.clip(std_seconds/240.,0,3),
        np.clip(physical[:,0]-np.max(physical[:,0]),-30,0)/10.,
        rank,
    ),axis=1).astype(np.float32)
    features=np.concatenate((physical,local,consensus),axis=1)
    if features.shape!=(k,N_FEATURES) or not np.isfinite(features).all():
        raise ValueError("Invalid assembled V12 features.")
    return features,refined_times.astype(np.float64)


class MultiHypothesisSelector(nn.Module):
    """Permutation-equivariant candidate scorer with event-level context."""

    def __init__(self,n_features=N_FEATURES,hidden=48,dropout=.12):
        super().__init__()
        self.encoder=nn.Sequential(
            nn.Linear(n_features,hidden),nn.GELU(),
            nn.Linear(hidden,hidden),nn.GELU()
        )
        self.head=nn.Sequential(
            nn.Linear(hidden*3,hidden),nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(hidden,1)
        )

    def forward(self,features):
        if features.ndim!=2 or features.shape[-1]!=N_FEATURES:
            raise ValueError("V12 expects a [K,28] candidate feature matrix.")
        h=self.encoder(features)
        mean=h.mean(0,keepdim=True).expand_as(h)
        maximum=h.max(0,keepdim=True).values.expand_as(h)
        return self.head(torch.cat((h,mean,maximum),dim=-1)).squeeze(-1)


def fit_feature_norm(train_records):
    if not train_records:
        raise ValueError("No fold training records for V12.")
    all_features=np.concatenate([
        item["features"] for item in train_records
    ],axis=0).astype(np.float64)
    if (all_features.ndim!=2 or
            all_features.shape[1]!=N_FEATURES or
            not np.isfinite(all_features).all()):
        raise ValueError("Invalid TRAIN features.")
    mean=all_features.mean(0)
    std=np.where(all_features.std(0)>1e-5,all_features.std(0),1.)
    return {"mean":mean.astype(np.float32),
            "scale":std.astype(np.float32)}


def normalize(record,norm,device):
    x=(record["features"]-norm["mean"])/norm["scale"]
    return torch.as_tensor(
        x,dtype=torch.float32,device=device
    )


def ranking_loss(logits,errors_s,*,temperature=1.,sigma_s=90.):
    """Soft best-of-K supervision; no label-derived features.

    The Gaussian/absolute loss uses the FINAL V10-corrected timestamp,
    not the original coarse V7b candidate center.
    """
    error=torch.as_tensor(
        errors_s,device=logits.device,dtype=logits.dtype
    )
    if (logits.ndim!=1 or len(logits)!=len(error) or
            temperature<=0 or sigma_s<=0 or
            not torch.isfinite(error).all()):
        raise ValueError("Invalid V12 ranking labels.")
    # Soft labels remain defined even when all hypotheses are far away.
    target=torch.softmax(
        -(error-error.min())/sigma_s,dim=0
    )
    logp=F.log_softmax(logits/temperature,dim=0)
    listwise=-(target*logp).sum()
    # Additional learning signal directed at the <=60-s objective.
    known_good=error<=60.
    clear_bad=error>=120.
    supervised=known_good|clear_bad
    if bool(supervised.any()):
        bce=F.binary_cross_entropy_with_logits(
            logits[supervised],
            known_good[supervised].to(logits.dtype)
        )
    else:
        bce=logits.sum()*0.
    return listwise+.30*bce


def selected_index(learner_scores,coarse_scores,alpha):
    """Alpha=0 is EXACT top V7b candidate, not approximate blending."""
    scores=np.asarray(learner_scores,dtype=np.float64)
    baseline=np.asarray(coarse_scores,dtype=np.float32)
    if len(scores)!=len(baseline) or len(scores)==0:
        raise ValueError("V12 score lengths differ.")
    if not np.isfinite(scores).all() or not 0<=alpha<=1:
        raise ValueError("Invalid V12 scores/mixture coefficient.")
    if alpha==0.:
        # Candidates are retained in the original V7b-ranked order.
        return 0
    order=np.argsort(
        np.argsort(-baseline,kind="stable"),kind="stable"
    )
    base=1.-order/max(1,len(order)-1)
    mix=(1.-alpha)*base+alpha*scores
    return int(np.argmax(mix))


def theoretical_coverage(records,threshold=60.):
    """Oracle ceiling for shortlist, diagnostics ONLY. Never model input."""
    return float(np.mean([
        np.min(np.abs(r["predicted_offsets"]))<=threshold
        for r in records
    ]))
