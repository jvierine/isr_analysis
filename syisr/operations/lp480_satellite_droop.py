"""Estimate the effective pulse envelope using independent LP480 satellite echoes."""
import argparse
from collections import defaultdict
import json
from pathlib import Path

import h5py
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from scipy.stats import theilslopes

from lp480_satellite_delay import read_part


def obvious_rail(parts, left, right):
    """Look for repeated high extrema in raw I or Q during the measured track."""
    for part in parts:
        times, _, iq = read_part(part)
        iq = iq[(times >= left) & (times <= right)]
        if iq.size == 0:
            continue
        for v in (iq.real, -iq.real, iq.imag, -iq.imag):
            ceiling = v.max()
            if ceiling > 20 * np.median(np.abs(v)) and np.count_nonzero(v == ceiling) >= 8:
                return True
    return False


def main():
    p = argparse.ArgumentParser(__doc__)
    p.add_argument("audit", type=Path)
    p.add_argument("index", type=Path)
    p.add_argument("output", type=Path)
    a = p.parse_args()
    with h5py.File(a.index) as f:
        index = json.loads(bytes(f["json_utf8"][:]))
    parts = defaultdict(list)
    for job in index["jobs"]:
        for part in job["parts"]:
            parts[job["az"], job["el"], part["block"]].append(part)
    t = np.arange(0., 471., 2.)
    records = []
    with h5py.File(a.audit) as f:
        x = f["delay_axis_us"][:]
        beams = f["beam_az_el_deg"][:]
        for g in f["crossing"].values():
            m = json.loads(g.attrs["metrics_json"])
            if not m["accepted"] or min(m["split_snr"]) < 25. or abs(m["measured_minus_predicted_us"]-162)>10:
                continue
            trace = g["profile"][:]
            left_edge = m["measured_minus_predicted_us"]-235.
            signal = np.interp(t+left_edge, x, (trace*np.exp(-1j*m["phase_rad"])).real)
            level = np.median(signal[(t >= 60)&(t <= 130)])
            if level <= 0:
                continue
            records.append((m, signal/level, float(level), g["recorded_utc_range_rate_separation"][:,0]))
    independent = {}
    for rec in sorted(records,key=lambda r:min(r[0]["split_snr"]),reverse=True):
        independent.setdefault(rec[0]["norad"],rec)
    records = sorted(independent.values(),key=lambda r:r[2])
    amplitudes = np.array([r[2] for r in records])
    limit = np.quantile(amplitudes,.75)
    kept = []
    rail = 0
    for m,power,level,times in records:
        if level > limit:
            continue
        az,el=beams[m["beam"]]
        left,right=times.min(),times.max()
        candidates=[part for k in range(int(left)//10*10,int(right)//10*10+1,10)
                    for part in parts[az,el,k]]
        if obvious_rail(candidates,left,right):
            rail += 1
            continue
        kept.append((m,power,level))
    if len(kept)<10:
        raise SystemExit("Too few independently supported moderate echoes")
    y=np.array([r[1] for r in kept])
    central=(t>=60)&(t<=410)
    gain=np.median(y,axis=0)
    bounds=np.percentile(y,[25,75],axis=0)
    x1=(t>=60)&(t<=130)
    x2=(t>=350)&(t<=420)
    droop=10*np.log10(np.median(y[:,x2],axis=1)/np.median(y[:,x1],axis=1))
    slopes=np.array([theilslopes(10*np.log10(z[central]),t[central])[0]
                     for z in y if np.all(z[central]>0)])
    rng=np.random.default_rng(20240101)
    good=droop[np.isfinite(droop)]
    ci=np.percentile(np.median(rng.choice(good,size=(5000,len(good)),replace=True),axis=1),[2.5,97.5])
    weak,strong=np.array_split(np.array([r[1] for r in kept]),2)
    med=lambda z:float(np.nanmedian(10*np.log10(np.median(z[:,x2],axis=1)/np.median(z[:,x1],axis=1))))
    fig,ax=plt.subplots(figsize=(7.4,3.2),constrained_layout=True)
    ax.fill_between(t,*bounds,color="#2a7389",alpha=.19,label="Middle 50% of echoes")
    ax.plot(t,gain,color="#2a7389",label="Median")
    ax.plot(t,np.median(weak,axis=0),color="#a34c32",ls="--",lw=1,label="Weaker half")
    ax.axvspan(60,130,color=".5",alpha=.10)
    ax.axvspan(350,420,color=".5",alpha=.10)
    ax.set(xlabel="Time within first-lag overlap (µs)",ylabel="Relative first-lag power",xlim=(0,470))
    ax.legend(frameon=False,loc="lower center",ncol=3)
    a.output.mkdir(parents=True,exist_ok=True)
    fig.savefig(a.output/"pulse-droop.png",dpi=200)
    plt.close(fig)
    result=dict(eligible=len(records), selected=len(kept), clipped=rail,
                upper_quartile_excluded=int((amplitudes>limit).sum()), median_droop_db=float(np.median(good)),
                ci_db=ci.tolist(), weak_db=med(weak), strong_db=med(strong),
                median_slope_db_per_us=float(np.median(slopes)),
                early_us=[60,130],late_us=[350,420],
                limitation="Raw-code repeated extrema screen; analog compression cannot be excluded")
    with h5py.File(a.output/"pulse-droop.h5","w") as f:
        f["time_us"]=t
        f["normalized_envelopes"]=y
        f["median_envelope"]=gain
        f["norad_id"]=[r[0]["norad"] for r in kept]
        f.attrs["summary_json"]=json.dumps(result)
    print(json.dumps(result),flush=True)


if __name__=="__main__":
    main()
