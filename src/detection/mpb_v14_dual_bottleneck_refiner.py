"""V14: V10 coarse-to-fine MPB locator with two 8-conv 5D bottlenecks.

Keep V7b strictly frozen as the coarse candidate proposer. Unlike V10's
16-channel temporal teacher representation, the trainable FINE model uses
only two independent five-SCALAR embeddings (MAG and SWEA) to predict its
temporal correction. Each embedding is produced by eight Conv1d layers
and regularized by a decoder reconstructing the measured input.

Do not confuse the old frozen V7b encoder (16 temporal channels) with
the NEW interpretable V14 bottlenecks (five coordinates per sensor).
The catalog MPB time is never part of the neural-network input.
"""
from __future__ import annotations

import torch
from torch import nn
from torch.nn import functional as F

from src.detection.mag_swea_region_model import MagSweaRegionAutoencoder
# Keep the original V13 3D autoencoders unchanged. V14 now uses the
# same eight-convolution layout with a five-SCALAR latent vector.
MAG_LENGTH=640
SWEA_LENGTH=256
LATENT_DIM=5


class SensorFiveLatentAE(nn.Module):
    """V13-style eight-convolution encoder/decoder with z in R^5."""

    def __init__(self,sensor):
        super().__init__()
        if sensor=="MAG":
            features,length=4,MAG_LENGTH
            widths=(4,16,24,32,40,48,56,64,64)
            strides=(2,)*8
            conv_end=3
            decode_sizes=(5,10,20,40,80,160,320,640)
            decode_widths=(64,64,56,48,40,32,24,16,4)
        elif sensor=="SWEA":
            features,length=12,SWEA_LENGTH
            widths=(12,16,24,32,40,48,56,64,64)
            strides=(2,2,2,2,2,2,2,1)
            conv_end=2
            decode_sizes=(2,4,8,16,32,64,128,256)
            decode_widths=(64,64,56,48,40,32,24,16,12)
        else:
            raise ValueError("Unknown V14 instrument")
        self.features=features
        self.length=length
        self.conv_end=conv_end
        self.decode_sizes=decode_sizes
        self.encoder_layers=nn.ModuleList([
            nn.Sequential(
                nn.Conv1d(widths[i],widths[i+1],
                          kernel_size=5,stride=strides[i],padding=2),
                nn.GELU(),
            ) for i in range(8)
        ])
        self.to_latent=nn.Linear(64*conv_end,LATENT_DIM)
        self.from_latent=nn.Linear(LATENT_DIM,64*conv_end)
        self.decoder_layers=nn.ModuleList([
            nn.Conv1d(decode_widths[i],decode_widths[i+1],5,padding=2)
            for i in range(8)
        ])

    def encode(self,x,mask):
        if (x.ndim!=3 or x.shape[1:]!=(self.features,self.length)
                or mask.shape!=(len(x),self.length)):
            raise ValueError("Invalid padded V14 sensor samples or mask")
        h=x*mask[:,None,:].to(x.dtype)
        for layer in self.encoder_layers:
            h=layer(h)
        if h.shape[1:]!=(64,self.conv_end):
            raise RuntimeError("Unexpected V14 convolutional time shape")
        return self.to_latent(h.flatten(1))

    def decode(self,z):
        if z.ndim!=2 or z.shape[1]!=LATENT_DIM:
            raise ValueError("V14 latent must be [batch,5]")
        h=self.from_latent(z).reshape(-1,64,self.conv_end)
        for i,(size,layer) in enumerate(zip(
            self.decode_sizes,self.decoder_layers
        )):
            h=F.interpolate(h,size=size,mode="nearest")
            h=layer(h)
            if i<7:
                h=F.gelu(h)
        return h


def center_pad_native(x, mask, target_length, min_count):
    """Center real native-time samples in fixed AE length, without interpolation.

    Input minibatches are right-padded by collate_dual; each example is
    individually centered to avoid encoding spurious batch pad positions.
    Returns tensor, mask; gradients can flow through the padded input.
    """
    if (x.ndim!=3 or mask.shape!=(len(x),x.shape[-1])
            or mask.dtype!=torch.bool):
        raise ValueError("Invalid instrument batch or boolean time mask.")
    result=x.new_zeros((len(x),x.shape[1],target_length))
    valid=torch.zeros((len(x),target_length),device=x.device,dtype=torch.bool)
    for i in range(len(x)):
        positions=torch.nonzero(mask[i],as_tuple=False).flatten()
        n=int(positions.numel())
        if not min_count<=n<=target_length:
            raise ValueError(f"Invalid native sensor length {n}.")
        first=(target_length-n)//2
        result[i,:,first:first+n]=x[i,:,positions]
        valid[i,first:first+n]=True
    return result,valid


def masked_mse(pred,target,mask):
    if pred.shape!=target.shape or mask.shape!=(len(pred),pred.shape[-1]):
        raise ValueError("Reconstruction/mask shapes differ.")
    valid=mask[:,None,:].to(pred.dtype)
    return ((pred-target).square()*valid).sum()/(
        valid.sum()*pred.shape[1]).clamp_min(1.)


class V14DualBottleneckRefiner(nn.Module):
    """Two trainable 5D AEs + time refinement anchored on frozen V7b."""

    ARCHITECTURE="v14-v10-coarse-8conv-5d-mag-8conv-5d-swea-v1"

    def __init__(self,v7b_state,*,dropout=.1,max_offset_s=250.,
                 basis_width_s=24.):
        super().__init__()
        if not 0.<max_offset_s<300. or basis_width_s<=0.:
            raise ValueError("Invalid V14 temporal support.")
        self.max_offset_s=float(max_offset_s)
        self.basis_width_s=float(basis_width_s)
        self.teacher=MagSweaRegionAutoencoder()
        self.teacher.load_state_dict(v7b_state,strict=True)
        self.teacher.requires_grad_(False)
        self.teacher.eval()
        # Eight Conv1d encoder layers, each with FIVE scalar coordinates.
        self.mag_ae=SensorFiveLatentAE("MAG")
        self.swea_ae=SensorFiveLatentAE("SWEA")
        # [z_MAG(5),z_SWEA(5),teacher transition(1),teacher sides(2)]
        self.fusion=nn.Sequential(
            nn.Linear(2*LATENT_DIM+3,64),nn.GELU(),nn.Dropout(dropout),
            nn.Linear(64,64),nn.GELU(),
        )
        # 21 overlapping Gaussian time basis functions at native MAG times;
        # no interpolation of MAG or SWEA measurement samples.
        self.temporal_head=nn.Linear(64,21)
        self.confidence_head=nn.Linear(64,1)
        self.register_buffer("basis_centers_s",
                             torch.linspace(-250.,250.,21))

    def train(self,mode=True):
        super().train(mode)
        self.teacher.eval()
        return self

    @torch.no_grad()
    def teacher_logits(self,mag,mag_mask,mag_time,
                       swea,swea_mask,swea_time):
        self.teacher.eval()
        out=self.teacher(mag,mag_mask,mag_time,
                         swea,swea_mask,swea_time)
        return out["transition_logit"],out["region_logits"]

    def _padded_inputs(self,mag,mag_mask,swea,swea_mask):
        pm,mm=center_pad_native(mag,mag_mask,MAG_LENGTH,540)
        ps,ms=center_pad_native(swea,swea_mask,SWEA_LENGTH,12)
        if pm.shape[1]!=4 or ps.shape[1]!=12:
            raise ValueError("MAG must have 4 and SWEA 12 channels.")
        return pm,mm,ps,ms

    def encode_latents(self,mag,mag_mask,swea,swea_mask):
        """Return a five-scalar vector [z1,...,z5] for each sensor."""
        pm,mm,ps,ms=self._padded_inputs(mag,mag_mask,swea,swea_mask)
        return self.mag_ae.encode(pm,mm),self.swea_ae.encode(ps,ms)

    def forward(self,mag,mag_mask,mag_time,
                swea,swea_mask,swea_time):
        if mag_time.shape!=mag_mask.shape or swea_time.shape!=swea_mask.shape:
            raise ValueError("Native instrument time and mask shapes differ.")
        pm,mm,ps,ms=self._padded_inputs(mag,mag_mask,swea,swea_mask)
        zm=self.mag_ae.encode(pm,mm)
        zs=self.swea_ae.encode(ps,ms)
        rm=self.mag_ae.decode(zm)
        rs=self.swea_ae.decode(zs)
        with torch.no_grad():
            coarse,region=self.teacher_logits(
                mag,mag_mask,mag_time,swea,swea_mask,swea_time
            )
        fused=torch.cat((zm,zs,coarse[:,None],region),dim=-1)
        h=self.fusion(fused)
        basis_logits=self.temporal_head(h)
        centers=self.basis_centers_s[None,:,None]
        native_t=mag_time[:,None,:]
        smooth_log_density= -0.5*((native_t-centers)/self.basis_width_s).square()
        log_weights=F.log_softmax(basis_logits,dim=-1)[:,:,None]
        logits=torch.logsumexp(log_weights+smooth_log_density,dim=1)
        valid=mag_mask & (mag_time.abs()<=self.max_offset_s)
        if not bool(valid.any(dim=1).all()):
            raise ValueError("No measured MAG samples in V14 temporal ROI.")
        logits=logits.masked_fill(~valid,-1e4)
        probabilities=torch.softmax(logits,dim=-1)
        expected=(probabilities*mag_time).sum(-1)
        index=logits.argmax(dim=-1)
        peak=mag_time.gather(1,index[:,None]).squeeze(1)
        presence=self.confidence_head(h).squeeze(-1)
        return {
            "time_logits":logits,
            "time_prob":probabilities,
            "expected_offset_s":expected,
            "peak_offset_s":peak,
            "presence_logit":presence,
            "teacher_logit":coarse,
            "teacher_region_logits":region,
            "mag_latent":zm,
            "swea_latent":zs,
            "mag_reconstruction_mse":masked_mse(rm,pm,mm),
            "swea_reconstruction_mse":masked_mse(rs,ps,ms),
        }
