"""Pure physical diagnostics for V13 MAG/SWEA latent exploration.

These features use measured instrument values and catalog MPB times.
All pre/post intervals exclude +/-60 seconds around the annotated MPB,
so they describe two flanks rather than the boundary transition itself.

MAG components and vector magnitude are in nT; SWEA input is
log1p(angular-mean electron flux), interpolated in ENERGY only to the
12 fixed physical channels 30-300 eV. Temporal samples stay native.
"""
from __future__ import annotations

import numpy as np
from scipy.stats import spearmanr

from src.data.swea_event_cache import ENERGY_EV

FLANK_INNER_S=60.
FLANK_OUTER_S=240.


def flank_masks(time_s, mpb_s):
    """Return actual pre/post measured-sample masks, never resample."""
    delta=np.asarray(time_s,dtype=float)-float(mpb_s)
    left=(delta>=-FLANK_OUTER_S)&(delta<=-FLANK_INNER_S)
    right=(delta>=FLANK_INNER_S)&(delta<=FLANK_OUTER_S)
    return left,right


def _rms_b_difference(times, vectors):
    """RMS magnitude of consecutive vector increments in nT (dt<=2.5s)."""
    if len(vectors)<3:
        return np.nan
    dt=np.diff(times)
    valid=(dt>0.)&(dt<=2.5)
    if not valid.any():
        return np.nan
    increments=np.linalg.norm(np.diff(vectors,axis=0),axis=1)
    return float(np.sqrt(np.mean(increments[valid]**2)))


def mag_physical_features(times,values,mpb_s):
    """Independent magnetic morphology descriptors around a catalog MPB.

    Never derive MAG physical parameters from reconstructed CNN output.
    The 4th channel is the measured total B, used as a consistency check
    against vector magnitude; features use sqrt(Bx^2+By^2+Bz^2).
    """
    times=np.asarray(times,dtype=float)
    values=np.asarray(values,dtype=float)
    if values.ndim!=2 or values.shape!=(len(times),4):
        raise ValueError("MAG block must have [N,4] measured values.")
    if not np.isfinite(times).all() or not np.isfinite(values).all():
        raise ValueError("MAG physical diagnostics require finite data.")
    pre,post=flank_masks(times,mpb_s)
    p=values[pre,:3]
    q=values[post,:3]
    out={
        "mag_n_pre":int(len(p)),
        "mag_n_post":int(len(q)),
        "mag_total_b_consistency_max_nt":float(np.max(
            np.abs(np.linalg.norm(values[:,:3],axis=1)-values[:,3])
        )) if len(values) else np.nan,
    }
    names=(
        "mag_pre_mean_b_nt","mag_post_mean_b_nt",
        "mag_delta_b_nt","mag_abs_delta_b_nt",
        "mag_b_ratio_post_pre",
        "mag_pre_std_b_nt","mag_post_std_b_nt",
        "mag_pre_rms_db_nt","mag_post_rms_db_nt",
        "mag_rotation_deg",
        "mag_pre_mean_bx_nt","mag_pre_mean_by_nt",
        "mag_pre_mean_bz_nt","mag_post_mean_bx_nt",
        "mag_post_mean_by_nt","mag_post_mean_bz_nt",
    )
    if len(p)<50 or len(q)<50:
        out.update({name:np.nan for name in names})
        return out
    bp=np.linalg.norm(p,axis=1)
    bq=np.linalg.norm(q,axis=1)
    meanp=p.mean(axis=0)
    meanq=q.mean(axis=0)
    dp=np.linalg.norm(meanp)*np.linalg.norm(meanq)
    rotation=float(np.degrees(np.arccos(
        np.clip(np.dot(meanp,meanq)/dp,-1.,1.)
    ))) if dp>1e-8 else np.nan
    out.update({
        "mag_pre_mean_b_nt":float(bp.mean()),
        "mag_post_mean_b_nt":float(bq.mean()),
        "mag_delta_b_nt":float(bq.mean()-bp.mean()),
        "mag_abs_delta_b_nt":float(abs(bq.mean()-bp.mean())),
        "mag_b_ratio_post_pre":float(bq.mean()/bp.mean())
            if bp.mean()>1e-8 else np.nan,
        "mag_pre_std_b_nt":float(np.std(bp)),
        "mag_post_std_b_nt":float(np.std(bq)),
        "mag_pre_rms_db_nt":_rms_b_difference(times[pre],p),
        "mag_post_rms_db_nt":_rms_b_difference(times[post],q),
        "mag_rotation_deg":rotation,
    })
    for side,vector in (("pre",meanp),("post",meanq)):
        for j,axis in enumerate("xyz"):
            out[f"mag_{side}_mean_b{axis}_nt"]=float(vector[j])
    return out


def _spectral_slope(logflux):
    """Empirical log1p(flux)-versus-log(E) slope; not a plasma power law."""
    if not np.isfinite(logflux).all():
        return np.nan
    return float(np.polyfit(np.log(ENERGY_EV),logflux,1)[0])


def swea_physical_features(times,values,mpb_s):
    """Pre/post native SWEA flux proxies in original cache coordinates.

    The 12 cache channels are log1p of energy-interpolated angular mean
    flux, not raw instrument energy channels and not particle moments.
    """
    times=np.asarray(times,dtype=float)
    values=np.asarray(values,dtype=float)
    if values.ndim!=2 or values.shape!=(len(times),12):
        raise ValueError("SWEA block must have [N,12] spectral values.")
    if not np.isfinite(times).all() or not np.isfinite(values).all():
        raise ValueError("SWEA diagnostics require finite samples.")
    before,after=flank_masks(times,mpb_s)
    p=values[before]
    q=values[after]
    out={
        "swea_n_pre":int(len(p)),
        "swea_n_post":int(len(q)),
    }
    names=(
        "swea_pre_mean_logflux","swea_post_mean_logflux",
        "swea_delta_mean_logflux",
        "swea_abs_spectral_delta_logflux",
        "swea_pre_temporal_std_logflux",
        "swea_post_temporal_std_logflux",
        "swea_pre_hardness_logdiff",
        "swea_post_hardness_logdiff",
        "swea_delta_hardness_logdiff",
        "swea_pre_slope_logflux_loge",
        "swea_post_slope_logflux_loge",
        "swea_delta_slope_logflux_loge",
    )
    if len(p)<3 or len(q)<3:
        out.update({name:np.nan for name in names})
        return out
    meanp=p.mean(axis=0)
    meanq=q.mean(axis=0)
    low=ENERGY_EV<=80.
    high=ENERGY_EV>=120.
    hp=float(meanp[high].mean()-meanp[low].mean())
    hq=float(meanq[high].mean()-meanq[low].mean())
    sp=_spectral_slope(meanp)
    sq=_spectral_slope(meanq)
    out.update({
        "swea_pre_mean_logflux":float(p.mean()),
        "swea_post_mean_logflux":float(q.mean()),
        "swea_delta_mean_logflux":float(q.mean()-p.mean()),
        "swea_abs_spectral_delta_logflux":float(
            np.mean(np.abs(meanq-meanp))
        ),
        "swea_pre_temporal_std_logflux":float(
            np.mean(np.std(p,axis=0))
        ),
        "swea_post_temporal_std_logflux":float(
            np.mean(np.std(q,axis=0))
        ),
        "swea_pre_hardness_logdiff":hp,
        "swea_post_hardness_logdiff":hq,
        "swea_delta_hardness_logdiff":hq-hp,
        "swea_pre_slope_logflux_loge":sp,
        "swea_post_slope_logflux_loge":sq,
        "swea_delta_slope_logflux_loge":sq-sp,
    })
    return out


def spearman_pair(x,y,min_count=12):
    """Spearman coefficient and asymptotic p-value on finite paired rows."""
    x=np.asarray(x,dtype=float)
    y=np.asarray(y,dtype=float)
    use=np.isfinite(x)&np.isfinite(y)
    x=x[use];y=y[use]
    if (len(x)<min_count or np.ptp(x)==0. or np.ptp(y)==0.):
        return {"n":int(len(x)),"rho":np.nan,"pvalue":np.nan}
    rho,p=spearmanr(x,y)
    return {"n":int(len(x)),"rho":float(rho),"pvalue":float(p)}


def benjamini_hochberg(pvalues):
    """BH adjusted q values, retaining NaNs for untestable contrasts."""
    raw=np.asarray(pvalues,dtype=float)
    result=np.full(raw.shape,np.nan)
    indices=np.flatnonzero(np.isfinite(raw))
    if not len(indices):
        return result
    order=indices[np.argsort(raw[indices])]
    m=len(indices)
    ranked=raw[order]*m/np.arange(1,m+1)
    result[order]=np.minimum.accumulate(ranked[::-1])[::-1].clip(0,1)
    return result


def trajectory_statistics(time_offsets,z):
    """Geometry of a sliding-window latent path (not plasma velocity).

    Break at a gap >1.5*median center step; never equate a skipped unsafe
    window with an observed smooth segment.
    """
    t=np.asarray(time_offsets,dtype=float)
    z=np.asarray(z,dtype=float)
    if (z.ndim!=2 or z.shape!=(len(t),3)
            or len(t)<2 or not np.isfinite(t).all()
            or not np.isfinite(z).all()
            or np.any(np.diff(t)<=0)):
        return {
            "n_windows":int(len(t)),"path_length":np.nan,
            "max_speed_per_s":np.nan,
            "peak_speed_center_s":np.nan,
            "center_to_edges_displacement":np.nan,
            "n_adjacent_segments":0,
        }
    d=np.diff(t)
    reference=float(np.min(d))
    valid=d<=reference*1.5+1e-8
    distance=np.linalg.norm(np.diff(z,axis=0),axis=1)
    speed=distance[valid]/d[valid]
    midpoint=(t[:-1]+t[1:])[valid]*.5
    peak=int(np.argmax(speed)) if len(speed) else None
    return {
        "n_windows":int(len(t)),
        "path_length":float(distance[valid].sum()),
        "max_speed_per_s":float(speed[peak])
            if peak is not None else np.nan,
        "peak_speed_center_s":float(midpoint[peak])
            if peak is not None else np.nan,
        "center_to_edges_displacement":float(np.linalg.norm(z[-1]-z[0])),
        "n_adjacent_segments":int(valid.sum()),
    }
