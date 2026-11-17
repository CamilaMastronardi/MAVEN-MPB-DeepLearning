"""V9 dense temporal MPB localizer, warm-started from V7b MAG/SWEA encoders.

Each measured 600-s window produces a logit at every *native MAG sample*
and a window-level boundary confidence. SWEA is encoded at its native cadence
then its latent representation (NOT the measured flux) is sampled by nearest
time for fusion. This preserves timestamp alignment without fabricating
additional SWEA measurements. No region averaging or central 60-s blackout.
"""
from __future__ import annotations

import torch
from torch import nn
from torch.nn import functional as F

from src.detection.mag_swea_region_model import (
    SensorEncoderDecoder, ResidualTimeBlock
)


def masked_local_mean(z, mask, kernel):
    """Centered, mask-aware temporal average including padded minibatches."""
    if kernel%2!=1 or kernel<1:
        raise ValueError("Pooling window must be positive and odd.")
    w=mask[:,None,:].to(z.dtype)
    numerator=F.avg_pool1d(z*w,kernel,stride=1,padding=kernel//2)
    denominator=F.avg_pool1d(w,kernel,stride=1,padding=kernel//2)
    return numerator/denominator.clamp_min(1./kernel)


def nearest_swea_latent(z, swea_time, swea_mask, mag_time, mag_mask):
    """Nearest valid native SWEA frame in *latent* space at MAG sample times."""
    if (z.ndim!=3 or z.shape[-1]!=swea_time.shape[-1] or
            swea_time.shape!=swea_mask.shape or
            mag_time.shape!=mag_mask.shape):
        raise ValueError("Incompatible MAG/SWEA tensors and time masks.")
    outputs=[]
    for i in range(z.shape[0]):
        valid=torch.nonzero(swea_mask[i],as_tuple=False).flatten()
        if len(valid)==0:
            raise ValueError("SWEA time series has no valid frames.")
        times=swea_time[i].index_select(0,valid).contiguous()
        if len(times)>1 and bool(torch.any(times[1:]<=times[:-1])):
            raise ValueError("SWEA frame timestamps must increase.")
        right=torch.searchsorted(times,mag_time[i].contiguous())
        right=right.clamp(0,len(times)-1)
        left=(right-1).clamp(min=0)
        choose_left=(
            (mag_time[i]-times[left]).abs() <=
            (times[right]-mag_time[i]).abs()
        )
        nearest=torch.where(choose_left,left,right)
        indices=valid.index_select(0,nearest)
        aligned=z[i].index_select(1,indices)
        outputs.append(aligned*mag_mask[i][None,:].to(aligned.dtype))
    return torch.stack(outputs,dim=0)


class MagSweaTemporalLocalizer(nn.Module):
    """Dense MPB time logits and independent 10-min window confidence."""

    def __init__(self, dropout=0.12):
        super().__init__()
        self.mag=SensorEncoderDecoder(4,24,16,(1,2,4,8))
        self.swea=SensorEncoderDecoder(12,24,16,(1,2))
        self.fuse=nn.Sequential(
            nn.Conv1d(64,48,kernel_size=5,padding=2),
            nn.GELU(),
            ResidualTimeBlock(48,4),
            ResidualTimeBlock(48,16),
            nn.Dropout(dropout),
            nn.Conv1d(48,1,kernel_size=1),
        )
        self.window_bias=nn.Parameter(torch.tensor(-1.0))

    def load_v7b_encoders(self, source_state):
        """Only reuse V7b encoders; new localization head starts fresh."""
        for name,module in (("mag",self.mag),("swea",self.swea)):
            weights={
                k[len(name)+1:]:value
                for k,value in source_state.items()
                if k.startswith(name+".")
            }
            if not weights:
                raise ValueError(f"V7b checkpoint missing {name} encoder.")
            module.load_state_dict(weights,strict=True)

    def forward(
        self, mag, mag_mask, mag_time,
        swea, swea_mask, swea_time
    ):
        if (mag.ndim!=3 or swea.ndim!=3 or
                mag.shape[0]!=swea.shape[0] or
                mag_mask.shape!=mag_time.shape or
                swea_mask.shape!=swea_time.shape):
            raise ValueError("Missing or invalid native-cadence inputs.")
        zm=self.mag.encoder(mag*mag_mask[:,None,:])
        zs=self.swea.encoder(swea*swea_mask[:,None,:])
        aligned=nearest_swea_latent(
            zs,swea_time,swea_mask,mag_time,mag_mask
        )
        features=torch.cat((
            zm,aligned,
            masked_local_mean(zm,mag_mask,31),
            masked_local_mean(zm,mag_mask,121)
        ),dim=1)
        logits=self.fuse(features).squeeze(1)
        logits=logits.masked_fill(~mag_mask,-1e4)
        # Log-average-exp allows a sparse strong peak to raise confidence
        # while retaining a calibrated distinction between known positive
        # windows and physically safe far controls.
        count=mag_mask.sum(-1).clamp_min(1).to(logits.dtype)
        confidence=torch.logsumexp(logits,dim=-1)-count.log()
        return {
            "time_logits":logits,
            "window_logit":confidence+self.window_bias,
            "time_prob":torch.softmax(logits,dim=-1),
        }


def temporal_loss(
    out, mag_time, mag_mask, positive, target_offset_s,
    *, sigma_s=32., localization_weight=1.0
):
    """Positive windows: Gaussian soft time labels. Negatives: BCE only.

    Positive offsets must genuinely lie inside the observed MAG clip.
    Missing samples/padding never receive probability mass or supervision.
    """
    if sigma_s<=0 or localization_weight<0:
        raise ValueError("Invalid localization loss configuration.")
    positive=positive.to(torch.bool)
    detection=F.binary_cross_entropy_with_logits(
        out["window_logit"],positive.to(torch.float32)
    )
    if not bool(positive.any()):
        return detection, {
            "det":float(detection.detach()),"time":0.,
            "total":float(detection.detach()),
            "n_positive":0,
        }
    t=mag_time[positive]
    mask=mag_mask[positive]
    target=target_offset_s[positive,None]
    if not bool(
        ((t-target).abs().masked_fill(~mask,float("inf")).min(-1).values
         <=sigma_s*3).all()
    ):
        raise ValueError("Annotated MPB not supported by positive clip.")
    target_scores= -0.5*((t-target)/sigma_s).square()
    target_scores=target_scores.masked_fill(~mask,-1e4)
    target_prob=torch.softmax(target_scores,dim=-1)
    log_pred=F.log_softmax(out["time_logits"][positive],dim=-1)
    # Minimize KL, with entropy baseline independent of model parameters.
    time_kl=F.kl_div(
        log_pred,target_prob,reduction="batchmean"
    )
    total=detection+localization_weight*time_kl
    return total, {
        "det":float(detection.detach()),
        "time":float(time_kl.detach()),
        "total":float(total.detach()),
        "n_positive":int(positive.sum().item()),
    }
