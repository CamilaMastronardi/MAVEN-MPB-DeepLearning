"""V13-SWEA: independent eight-convolution 3D bottleneck autoencoder.

SWEA measured native-cadence spectra: 12 preprocessed energy channels
and up to 32 real frames. No temporal interpolation; shorter clips are
masked and symmetrically padded. TRAIN SWEA normalizer only.
"""
from __future__ import annotations

import torch
from torch import nn
from torch.nn import functional as F

from src.detection.mpb_cnn_autoencoder_v13 import autoencoder_loss

LENGTH=32
CHANNELS=12
LATENT=3


class SWEA3DAutoencoder(nn.Module):
    def __init__(self,dropout=.10):
        super().__init__()
        widths=(12,16,24,32,40,48,56,64,64)
        strides=(2,2,2,2,1,1,1,1)
        self.encoder_layers=nn.ModuleList([
            nn.Sequential(
                nn.Conv1d(widths[i],widths[i+1],kernel_size=5,
                          stride=strides[i],padding=2),
                nn.GELU(),
            ) for i in range(8)
        ])
        # 32 samples ->16->8->4->2->2->2->2->2.
        self.to_latent=nn.Linear(64*2,LATENT)
        self.from_latent=nn.Linear(LATENT,64*2)
        dec_widths=(64,64,56,48,40,32,24,16,12)
        self.decode_sizes=(2,2,2,2,4,8,16,32)
        self.decoder_layers=nn.ModuleList([
            nn.Conv1d(dec_widths[i],dec_widths[i+1],5,padding=2)
            for i in range(8)
        ])
        self.presence_head=nn.Sequential(
            nn.Linear(LATENT,16),nn.GELU(),
            nn.Dropout(dropout),nn.Linear(16,1)
        )
        self.offset_head=nn.Sequential(
            nn.Linear(LATENT,16),nn.GELU(),
            nn.Linear(16,1),nn.Tanh()
        )

    def encode(self,swea,mask):
        if (swea.ndim!=3 or swea.shape[1:]!=(CHANNELS,LENGTH)
                or mask.shape!=(len(swea),LENGTH)):
            raise ValueError("SWEA requires [B,12,32] and [B,32] mask.")
        h=swea*mask[:,None,:].to(swea.dtype)
        for layer in self.encoder_layers:
            h=layer(h)
        if h.shape[1:]!=(64,2):
            raise RuntimeError("Unexpected SWEA CNN latent temporal shape.")
        return self.to_latent(h.flatten(1))

    def decode(self,z):
        h=self.from_latent(z).reshape(-1,64,2)
        for i,(size,layer) in enumerate(zip(
            self.decode_sizes,self.decoder_layers
        )):
            h=F.interpolate(h,size=size,mode="nearest")
            h=layer(h)
            if i<7:
                h=F.gelu(h)
        return h

    def forward(self,swea,mask):
        z=self.encode(swea,mask)
        return {
            "latent":z,
            "reconstruction":self.decode(z),
            "presence_logit":self.presence_head(z).squeeze(-1),
            "offset_s":240.*self.offset_head(z).squeeze(-1),
        }


def swea_collate(examples,normalizer,device="cpu"):
    """Measured SWEA frames only, with mask for padded positions."""
    if not examples:
        raise ValueError("Cannot collate empty SWEA windows.")
    mu=torch.as_tensor(
        normalizer["mean"],dtype=torch.float32,device=device
    )
    scale=torch.as_tensor(
        normalizer["scale"],dtype=torch.float32,device=device
    )
    if (mu.shape!=(CHANNELS,) or scale.shape!=(CHANNELS,)
            or not bool((scale>0).all())):
        raise ValueError("Invalid SWEA TRAIN-only normalizer.")
    x=torch.zeros(len(examples),CHANNELS,LENGTH,device=device)
    mask=torch.zeros(len(examples),LENGTH,device=device,dtype=torch.bool)
    labels=[]
    relative_offsets=[]
    for i,(clip,is_positive,offset) in enumerate(examples):
        values=torch.as_tensor(
            clip.swea,dtype=torch.float32,device=device
        )
        n=len(values)
        if (values.ndim!=2 or values.shape[1]!=CHANNELS
                or not 12<=n<=LENGTH):
            raise ValueError(
                f"Invalid measured SWEA window {tuple(values.shape)}"
            )
        start=(LENGTH-n)//2
        x[i,:,start:start+n]=((values-mu)/scale).T
        mask[i,start:start+n]=True
        labels.append(float(is_positive))
        relative_offsets.append(float(offset))
    return {
        "mag":x,  # Shared V13 objective uses this generic input key
        "mask":mask,
        "positive":torch.as_tensor(
            labels,dtype=torch.float32,device=device
        ),
        "offset_s":torch.as_tensor(
            relative_offsets,dtype=torch.float32,device=device
        ),
    }
