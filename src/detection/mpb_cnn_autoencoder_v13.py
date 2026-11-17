"""V13 exploratory 3D CNN autoencoder for MAG crossing morphology.

Eight strided Conv1d encoder layers compress a measured 10-minute MAG
window to THREE scalar coordinates. Eight upsampling Conv1d decoder
layers reconstruct Bx, By, Bz and |B|. Two intentionally lightweight
auxiliary heads assess (a) whether an MPB falls INSIDE the window and
(b) where inside that positive window its catalog time lies.

The 3D bottleneck is meant for physical morphology exploration and
subsequent clustering. A window's center timestamp is NEVER input to
the model. The relative-offset prediction can be converted to an
absolute UTC time by the caller: t_MPB = t_window_center + offset_s.

This does not yet validate full-orbit autonomous MPB detection.
"""
from __future__ import annotations

import torch
from torch import nn
from torch.nn import functional as F

LENGTH=640
LATENT=3


class Conv3DAutoencoder(nn.Module):
    def __init__(self,dropout=.10):
        super().__init__()
        encoder_dims=(4,16,24,32,40,48,56,64,64)
        self.encoder_layers=nn.ModuleList([
            nn.Sequential(
                nn.Conv1d(encoder_dims[i],encoder_dims[i+1],
                          kernel_size=5,stride=2,padding=2),
                nn.GELU(),
            ) for i in range(8)
        ])
        # 640 -> 320 -> 160 -> 80 -> 40 -> 20 -> 10 -> 5 -> 3
        self.to_latent=nn.Linear(64*3,LATENT)
        self.from_latent=nn.Linear(LATENT,64*3)
        sizes=(5,10,20,40,80,160,320,640)
        decode_dims=(64,64,56,48,40,32,24,16,4)
        self.decoder_layers=nn.ModuleList([
            nn.Conv1d(decode_dims[i],decode_dims[i+1],5,padding=2)
            for i in range(8)
        ])
        self.decode_sizes=sizes
        self.presence_head=nn.Sequential(
            nn.Linear(LATENT,16),nn.GELU(),nn.Dropout(dropout),
            nn.Linear(16,1)
        )
        self.offset_head=nn.Sequential(
            nn.Linear(LATENT,16),nn.GELU(),
            nn.Linear(16,1),nn.Tanh()
        )

    def encode(self,x,mask):
        if (x.ndim!=3 or x.shape[1:]!=(4,LENGTH) or
                mask.shape!=(x.shape[0],LENGTH)):
            raise ValueError("V13 MAG windows must be [B,4,640].")
        h=x*mask[:,None,:].to(x.dtype)
        for layer in self.encoder_layers:
            h=layer(h)
        if h.shape[1:]!=(64,3):
            raise RuntimeError("Unexpected V13 encoder temporal shape.")
        return self.to_latent(h.flatten(1))

    def decode(self,z):
        h=self.from_latent(z).reshape(-1,64,3)
        for i,(size,conv) in enumerate(zip(
            self.decode_sizes,self.decoder_layers
        )):
            h=F.interpolate(h,size=size,mode="nearest")
            h=conv(h)
            if i<7:
                h=F.gelu(h)
        return h

    def forward(self,x,mask):
        z=self.encode(x,mask)
        return {
            "latent":z,
            "reconstruction":self.decode(z),
            "presence_logit":self.presence_head(z).squeeze(-1),
            "offset_s":240.*self.offset_head(z).squeeze(-1),
        }


def mag_collate(examples,mag_normalizer,device="cpu"):
    """Measured MAG samples only. Padding is explicit, never interpolated.

    Every example is (DualClip, is_positive, target_offset_s) from
    the same physically safe V9 event pools. The data may contain SWEA
    but this MAG-only exploratory V13 deliberately ignores it.
    """
    import numpy as np

    if not examples:
        raise ValueError("Empty V13 batch.")
    mu=torch.as_tensor(
        mag_normalizer["mean"],dtype=torch.float32,device=device
    )
    std=torch.as_tensor(
        mag_normalizer["scale"],dtype=torch.float32,device=device
    )
    if mu.shape!=(4,) or std.shape!=(4,) or bool((std<=0).any()):
        raise ValueError("Invalid TRAIN-only MAG normalizer.")
    x=torch.zeros(len(examples),4,LENGTH,device=device)
    mask=torch.zeros(len(examples),LENGTH,device=device,dtype=torch.bool)
    targets=[]
    positive=[]
    for i,(clip,hit,offset) in enumerate(examples):
        values=torch.as_tensor(
            clip.mag,dtype=torch.float32,device=device
        )
        n=len(values)
        if values.ndim!=2 or values.shape[1]!=4 or not 540<=n<=LENGTH-40:
            raise ValueError(
                f"Invalid V13 measured MAG clip of length {n}."
            )
        # Symmetric padding avoids a strong time-axis asymmetry;
        # reconstruction and loss ignore every unmeasured padded sample.
        start=(LENGTH-n)//2
        x[i,:,start:start+n]=((values-mu)/std).T
        mask[i,start:start+n]=True
        targets.append(float(offset))
        positive.append(float(hit))
    return {
        "mag":x,
        "mask":mask,
        "positive":torch.as_tensor(
            positive,dtype=torch.float32,device=device
        ),
        "offset_s":torch.as_tensor(
            targets,dtype=torch.float32,device=device
        ),
    }


def autoencoder_loss(
    out,batch,*,detection_weight=.3,temporal_weight=.1
):
    """Reconstruction-primary objective plus MPB-identification diagnostics.

    Temporal regression is supervised on positive windows only;
    far physically safe controls have NO invented MPB target.
    """
    mask=batch["mask"][:,None,:].to(
        out["reconstruction"].dtype
    )
    target=batch["mag"]
    recon=((out["reconstruction"]-target).square()*mask).sum()/(
        mask.sum()*target.shape[1]
    ).clamp_min(1)
    bce=F.binary_cross_entropy_with_logits(
        out["presence_logit"],batch["positive"]
    )
    positives=batch["positive"]>0.5
    if bool(positives.any()):
        time=F.smooth_l1_loss(
            out["offset_s"][positives]/60.,
            batch["offset_s"][positives]/60.,
        )
    else:
        time=out["offset_s"].sum()*0.
    loss=recon+detection_weight*bce+temporal_weight*time
    metrics={
        "reconstruction_mse":float(recon.detach()),
        "identification_bce":float(bce.detach()),
        "time_huber":float(time.detach()),
        "total":float(loss.detach()),
        "positive_examples":int(positives.sum().item()),
    }
    return loss,metrics
