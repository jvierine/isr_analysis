"""Review crossing delays and first-lag envelopes from the LP480 timing audit."""
import argparse
import json
from pathlib import Path

import h5py
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np


def main():
    parser = argparse.ArgumentParser(__doc__)
    parser.add_argument("audit", type=Path)
    parser.add_argument("output", type=Path)
    args = parser.parse_args()
    with h5py.File(args.audit) as f:
        x = f["delay_axis_us"][:]
        rows = []
        for g in f["crossing"].values():
            metrics = json.loads(g.attrs["metrics_json"])
            if metrics["accepted"]:
                rows.append((metrics, g["profile"][:]))
    if not rows:
        raise SystemExit("No crossings passed the independent half-track and phase checks")
    # Keep one independent sample per catalog object for the timing summary.
    by_object = {}
    for row in sorted(rows, key=lambda pair: min(pair[0]["split_snr"]), reverse=True):
        by_object.setdefault(row[0]["norad"], row)
    rows = sorted(by_object.values(), key=lambda pair: pair[0]["median_utc"])
    metrics = [r for r, _ in rows]
    y = np.array([r["measured_minus_predicted_us"] for r in metrics])
    median = np.median(y)
    mad = 1.4826*np.median(np.abs(y-median))
    rng = np.random.default_rng(20240101)
    ci = np.percentile(np.median(rng.choice(y, size=(5000,len(y)), replace=True), axis=1), [2.5,97.5])
    print(f"Unique objects {len(y)}; median {median:.2f} us; scaled MAD {mad:.2f} us;"
          f" object bootstrap interval [{ci[0]:.2f}, {ci[1]:.2f}] us")
    args.output.mkdir(parents=True, exist_ok=True)
    fig, ax = plt.subplots(figsize=(9,4), constrained_layout=True)
    time = (np.array([r["median_utc"] for r in metrics])-1704067200)/3600
    ax.axhline(0,color=".65",lw=1)
    ax.axhline(median,color="#286b82",lw=1)
    ax.axhspan(*ci,color="#286b82",alpha=.12)
    for t,r in zip(time,metrics):
        ax.plot([t,t], sorted(r["split_delay_us"]), color=".65", lw=.8)
    ax.scatter(time,y,s=19,color="#286b82")
    ax.set(xlabel="UT hour on 1 January 2024", ylabel="Observed − predicted two-way delay (µs)",
           title=f"SYISR first-lag timing · {len(y)} distinct satellites")
    fig.savefig(args.output/"delay-summary.png",dpi=180)
    plt.close(fig)
    best = sorted(rows,key=lambda pair:min(pair[0]["split_snr"]),reverse=True)[:20]
    fig, axes = plt.subplots((len(best)+3)//4,4,figsize=(12,2.0*((len(best)+3)//4)),
                             sharex=True,squeeze=False,constrained_layout=True)
    for ax in axes.flat:
        ax.set_visible(False)
    for ax,(r,p) in zip(axes.flat,best):
        ax.set_visible(True)
        use=np.abs(x)<=650
        ax.plot(x[use],np.abs(p[use]),color="#286b82",lw=.8)
        edge=r["measured_minus_predicted_us"]
        ax.axvline(0,color=".5",ls=":",lw=.8)
        ax.axvspan(edge-235,edge+235,color="#286b82",alpha=.12)
        ax.set_title(f"NORAD {r['norad']} · {edge:+.0f} µs",fontsize=9)
        ax.ticklabel_format(axis="y",style="sci",scilimits=(0,0))
    fig.text(.5,.005,"Delay relative to predicted pulse center (µs)",ha="center")
    fig.savefig(args.output/"first-lag-envelopes.png",dpi=165)
    plt.close(fig)
    with h5py.File(args.output/"summary.h5","w") as f:
        f["delay_us"] = y
        f["bootstrap_median_interval_us"] = ci
        f.attrs.update(unique_objects=len(y),median_us=median,scaled_mad_us=mad,
                       median_slant_range_km=median*299792.458/2e6,
                       selection_json=json.dumps(metrics))


if __name__ == "__main__":
    main()
