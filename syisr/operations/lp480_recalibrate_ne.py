"""Estimate a fixed beam density scale from corrected LP480/L2 comparisons."""
import argparse
from datetime import datetime, timezone
from pathlib import Path

import h5py
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np


def daily_log_ratios(inversion, l2, count, altitude):
    region = (altitude >= 500) & (altitude <= 650)
    ratios = np.full(inversion.shape[:2], np.nan)
    for k in range(inversion.shape[0]):
        for j in range(inversion.shape[1]):
            a, b, n = inversion[k, j, region], l2[k, j, region], count[k, j, region]
            keep = (a >= 1e10) & (b >= 1e10) & (n >= 2)
            if keep.sum() >= 5:
                ratios[k, j] = np.median(np.log(b[keep] / a[keep]))
    return ratios


def main():
    p = argparse.ArgumentParser(__doc__)
    p.add_argument("input", type=Path)
    p.add_argument("output", type=Path)
    a = p.parse_args()
    with h5py.File(a.input) as f:
        if f.attrs.get("range_delay_correction_us") != -162.:
            raise ValueError("Use the delay-corrected comparison")
        z = f["altitude_km"][:]
        beams, time = f["beam_az_el_deg"][:], f["utc_window_start_s"][:]
        inv, l2, count = (f[k][:, 0] for k in ("inversion", "l2_mean", "l2_count"))
    ratios = daily_log_ratios(inv, l2, count, z)
    factors, days, holdout = [], [], []
    for row in ratios:
        day_medians = [np.median(d[np.isfinite(d)])
                       for d in row[48:].reshape(-1, 48) if np.isfinite(d).sum() >= 8]
        days.append(len(day_medians))
        factors.append(np.exp(np.median(day_medians)) if len(day_medians) >= 5 else np.nan)
        holdout.append(np.exp(np.nanmedian(row[:48])))
    factors = np.asarray(factors)
    corrected = inv * factors[:, None, None]
    a.output.mkdir(parents=True, exist_ok=True)
    pending = a.output / "density-recalibrated.partial.h5"
    with h5py.File(pending, "w") as f:
        for name, val in (
            ("altitude_km", z), ("beam_az_el_deg", beams), ("utc_window_start_s", time),
            ("ne_inversion_original_m3", inv), ("ne_inversion_recalibrated_m3", corrected),
            ("ne_l2_mean_m3", l2), ("l2_dwell_count", count), ("density_scale", factors),
            ("training_days", days), ("holdout_l2_over_original", holdout),
            ("holdout_l2_over_recalibrated", holdout/factors),
        ):
            f.create_dataset(name, data=val, compression="gzip")
        f.attrs["calibration"] = ("Median daily log(L2/inversion) after 162 us correction; "
            "500–650 km, at least two accepted L2 dwells per cell; "
            "median over altitude per window and then over windows; first UTC day withheld")
        f.attrs["interpretation"] = "Topside only; post-fit density scale, no change to the fitted forward model"
        f.attrs["source_comparison"] = str(a.input)
    pending.replace(a.output / "density-recalibrated.h5")
    k = int(np.argmax(beams[:, 1]))
    print(f"Zenith scale {factors[k]:.4f}; training days {days[k]}; "
          f"holdout L2/inversion {holdout[k]:.4f} -> {holdout[k]/factors[k]:.4f}", flush=True)
    fig, axs = plt.subplots(1, 5, figsize=(14, 6), sharex=True, sharey=True)
    for ax, j in zip(axs, (0, 10, 20, 30, 40)):
        ax.axhspan(z[0], 500, color=".92")
        ax.semilogx(inv[k,j], z, color=".55", lw=1, label="Before density rescaling")
        ax.semilogx(corrected[k,j], z, color="#176581", lw=1.8, label="Recalibrated inversion")
        ax.semilogx(l2[k,j], z, color="#b05229", ls="--", lw=1.5, label="Sanya L2")
        ax.set_title(datetime.fromtimestamp(int(time[j]), timezone.utc).strftime("%H:%M UTC"))
        ax.set(ylim=(z[0],1800), xlim=(1e9,1e13), xticks=[1e9,1e11,1e13])
        ax.grid(True, color=".87", lw=.6)
    axs[0].set_ylabel("Altitude (km)")
    fig.suptitle("SYISR zenith · 1 January 2024 · Held-out density profiles")
    fig.text(.5,.12,r"Electron density $N_e$ (m$^{-3}$)",ha="center")
    fig.legend(*axs[0].get_legend_handles_labels(), loc="lower center",
               bbox_to_anchor=(.5,.035), ncol=3, frameon=False)
    fig.subplots_adjust(left=.075,right=.985,top=.87,bottom=.18,wspace=.14)
    fig.savefig(a.output/"day01-density-recalibrated.png",dpi=260)
    plt.close(fig)


if __name__ == "__main__":
    main()
