"""V10 MPB coarse-to-fine temporal refinement with a FROZEN V7b teacher.

Stage 1 (unchanged): V7b locates a plausible MPB transition using physical
magnetosheath/induced-region information in MAG and SWEA.

Stage 2 (new): a multiscale, region-conditioned temporal head sees the
complete 600-s candidate window and predicts a small relative offset.
The V7b weights and region/transition classifier are frozen; the local
head alone learns a Gaussian boundary-time distribution plus an auxiliary
window-positive confidence. Training samples explicitly vary where the
MPB falls in the window so the network cannot infer a fixed center.

No catalog timestamp, group, orbit ID, or BS association enters model input.
"""
from __future__ import annotations

import torch
from torch import nn
from torch.nn import functional as F

from src.detection.mag_swea_region_model import (
    MagSweaRegionAutoencoder, ResidualTimeBlock, masked_region_pool
)
from src.detection.mpb_temporal_localizer import (
    masked_local_mean, nearest_swea_latent
)


def shifted_context(x, shift):
    """Return right-minus-left context centered around each MAG epoch."""
    if shift <= 0 or x.ndim != 3:
        raise ValueError("Shift must be positive for [B,C,T] features.")
    # The padded edges are never considered as physical samples in the
    # training target; near-window candidate support is limited to +/-250s.
    left = F.pad(x[:, :, :-shift], (shift, 0))
    right = F.pad(x[:, :, shift:], (0, shift))
    return right-left


class RegionAnchoredTemporalRefiner(nn.Module):
    """Frozen region-aware V7b model plus trainable fine temporal head."""

    def __init__(self, v7b_state, *, dropout=0.12, max_offset_s=250.):
        super().__init__()
        if max_offset_s <= 0 or max_offset_s >= 300.:
            raise ValueError("Refinement support must lie within 10-min clip.")
        self.max_offset_s=float(max_offset_s)
        self.teacher=MagSweaRegionAutoencoder()
        self.teacher.load_state_dict(v7b_state,strict=True)
        self.teacher.requires_grad_(False)
        self.teacher.eval()
        # 16 MAG + 16 SWEA + 16 short + 16 long change-point signals
        # + two V7b physical region logits + V7b transition score
        # + actual relative MAG time: 68 total channels.
        self.local_head=nn.Sequential(
            nn.Conv1d(68,48,kernel_size=5,padding=2),
            nn.GELU(),
            ResidualTimeBlock(48,1),
            ResidualTimeBlock(48,4),
            ResidualTimeBlock(48,16),
            nn.Dropout(dropout),
            nn.Conv1d(48,1,kernel_size=1),
        )
        self.confidence_head=nn.Sequential(
            nn.Linear(5,16), nn.GELU(), nn.Linear(16,1)
        )

    def train(self, mode=True):
        super().train(mode)
        # The frozen V7b teacher must never activate dropout.
        self.teacher.eval()
        return self

    @torch.no_grad()
    def teacher_logits(self, mag, mag_mask, mag_time,
                       swea, swea_mask, swea_time):
        """Exactly the original V7b candidate classification output."""
        self.teacher.eval()
        pred=self.teacher(
            mag,mag_mask,mag_time,
            swea,swea_mask,swea_time
        )
        return pred["transition_logit"],pred["region_logits"]

    @torch.no_grad()
    def _teacher_features(self, mag, mag_mask, mag_time,
                          swea, swea_mask, swea_time):
        self.teacher.eval()
        zm=self.teacher.mag.encoder(
            mag*mag_mask[:,None,:].to(mag.dtype)
        )
        zs=self.teacher.swea.encoder(
            swea*swea_mask[:,None,:].to(swea.dtype)
        )
        zm=zm*mag_mask[:,None,:].to(zm.dtype)
        zs=zs*swea_mask[:,None,:].to(zs.dtype)
        gap=self.teacher.region_gap_s
        left=torch.cat((
            masked_region_pool(
                zm,mag_mask,mag_time,side="left",gap_s=gap
            ),
            masked_region_pool(
                zs,swea_mask,swea_time,side="left",gap_s=gap
            )
        ),dim=1)
        right=torch.cat((
            masked_region_pool(
                zm,mag_mask,mag_time,side="right",gap_s=gap
            ),
            masked_region_pool(
                zs,swea_mask,swea_time,side="right",gap_s=gap
            )
        ),dim=1)
        region_logits=torch.stack((
            self.teacher.region_head(left).squeeze(-1),
            self.teacher.region_head(right).squeeze(-1)
        ),dim=1)
        coarse_features=torch.cat((
            left,right,right-left,(right-left).abs()
        ),dim=1)
        coarse_logit=self.teacher.transition_head(
            coarse_features
        ).squeeze(-1)
        aligned=nearest_swea_latent(
            zs,swea_time,swea_mask,mag_time,mag_mask
        )
        return zm,aligned,coarse_logit,region_logits

    def forward(self, mag, mag_mask, mag_time,
                swea, swea_mask, swea_time):
        zm,aligned,coarse,region=self._teacher_features(
            mag,mag_mask,mag_time,
            swea,swea_mask,swea_time
        )
        short=shifted_context(
            masked_local_mean(zm,mag_mask,31),15
        )
        long=shifted_context(
            masked_local_mean(zm,mag_mask,121),60
        )
        t=zm.shape[-1]
        temporal_features=torch.cat((
            zm, aligned, short, long,
            region[:,:,None].expand(-1,-1,t),
            coarse[:,None,None].expand(-1,1,t),
            (mag_time/300.)[:,None,:]
        ),dim=1)
        time_logits=self.local_head(temporal_features).squeeze(1)
        valid=mag_mask & (mag_time.abs()<=self.max_offset_s)
        if not bool(valid.any(dim=1).all()):
            raise ValueError("No valid MAG samples inside refinement support.")
        time_logits=time_logits.masked_fill(~valid,-1e4)
        probabilities=torch.softmax(time_logits,dim=-1)
        expected=(probabilities*mag_time).sum(-1)
        index=torch.argmax(time_logits,dim=-1)
        peak_offset=mag_time.gather(1,index[:,None]).squeeze(1)
        n=valid.sum(-1).to(time_logits.dtype)
        mean_exp=torch.logsumexp(time_logits,dim=-1)-n.log()
        max_logit=time_logits.max(-1).values
        confidence_features=torch.stack((
            coarse,region[:,0],region[:,1],mean_exp,max_logit
        ),dim=-1)
        presence_logit=self.confidence_head(
            confidence_features
        ).squeeze(-1)
        return {
            "time_logits":time_logits,
            "time_prob":probabilities,
            "expected_offset_s":expected,
            "peak_offset_s":peak_offset,
            "presence_logit":presence_logit,
            "teacher_logit":coarse,
            "teacher_region_logits":region,
        }


def refiner_loss(
    output, mag_time, mag_mask, is_positive, target_relative_s, *,
    sigma_s=32., localization_weight=1.5, offset_weight=0.5
):
    """Soft time supervision only for windows known to contain a MPB.

    Far negative windows receive a BCE loss only; they are not assigned
    an invented boundary timestamp or a physical region label.
    """
    if sigma_s<=0 or localization_weight<0 or offset_weight<0:
        raise ValueError("Invalid V10 loss weights or time scale.")
    is_positive=is_positive.to(torch.bool)
    bce=F.binary_cross_entropy_with_logits(
        output["presence_logit"],is_positive.to(torch.float32)
    )
    if not bool(is_positive.any()):
        return bce,{
            "bce":float(bce.detach()),"temporal_kl":0.,
            "offset":0.,"total":float(bce.detach()),
            "n_positive":0,
        }
    times=mag_time[is_positive]
    valid=mag_mask[is_positive] & (
        times.abs() <= float(times.new_tensor(250.))
    )
    targets=target_relative_s[is_positive,None]
    dist=(times-targets).abs()
    if not bool(
        (dist.masked_fill(~valid,float("inf")).min(-1).values
         <sigma_s*3.).all()
    ):
        raise ValueError("Positive MPB time outside measured refinement ROI.")
    gt_logit=-0.5*((times-targets)/sigma_s).square()
    gt_logit=gt_logit.masked_fill(~valid,-1e4)
    gt=torch.softmax(gt_logit,dim=-1)
    kl=F.kl_div(
        F.log_softmax(output["time_logits"][is_positive],dim=-1),
        gt,reduction="batchmean"
    )
    regression=F.smooth_l1_loss(
        output["expected_offset_s"][is_positive]/60.,
        target_relative_s[is_positive]/60.,
    )
    loss=bce+localization_weight*kl+offset_weight*regression
    return loss,{
        "bce":float(bce.detach()),
        "temporal_kl":float(kl.detach()),
        "offset":float(regression.detach()),
        "total":float(loss.detach()),
        "n_positive":int(is_positive.sum().item()),
    }
