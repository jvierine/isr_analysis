"""Compare completed LP480 inversions and supplied Sanya L2 fits on UTC bins."""
import argparse
import calendar
from datetime import datetime, timezone
import json
from pathlib import Path

import h5py
import matplotlib
matplotlib.use("Agg")
import matplotlib.dates as mdates
import matplotlib.pyplot as plt
from matplotlib.colors import LinearSegmentedColormap, LogNorm, Normalize
import numpy as np

HEIGHT = np.arange(100., 1801., 10.)
FIELDS = ("ne_m3", "ti_k", "te_k", "velocity_m_s")
LABELS = (r"$N_e$ (m$^{-3}$)", r"$T_i$ (K)", r"$T_e$ (K)", r"$v_i$ (m/s)")
LIMITS = ((1e9, 1e13), (150, 3000), (150, 6000), (-100, 100))
VELOCITY = LinearSegmentedColormap.from_list(
    "muted velocity", ["#21406b", "#548ba7", "#f4f0e5", "#cd7954", "#781f30"]
)


def read_fit(path):
    with h5py.File(path) as source:
        return json.loads(bytes(source["json_utf8"][:]))


def interpolate(height, values, max_gap):
    """Interpolate between adjacent valid gates, with no extrapolation."""
    height = np.asarray(height, dtype=float)
    values = np.asarray(values, dtype=float)
    order = np.argsort(height)
    height, values = height[order], values[order]
    right = np.searchsorted(height, HEIGHT)
    out = np.full(HEIGHT.size, np.nan)
    if not height.size:
        return out
    r = right.clip(0, height.size - 1)
    exact = (HEIGHT == height[r]) & np.isfinite(values[r])
    out[exact] = values[r[exact]]
    left = (right - 1).clip(0, height.size - 1)
    delta = height[r] - height[left]
    good = (right > 0) & (right < height.size) & (delta > 0) & (delta <= max_gap)
    good &= np.isfinite(values[r]) & np.isfinite(values[left]) & ~exact
    out[good] = values[left[good]] + (values[r[good]] - values[left[good]]) * (
        (HEIGHT[good] - height[left[good]]) / delta[good]
    )
    return out


def l2_rows(path, beams, start, inversion, total, count, seen):
    with h5py.File(path) as f:
        dirs = f["Beamdir"][:]
        dates = f["Time"][:]
        shape = f["Fittedparameter"].shape
        if shape != (*f["Fitstate"].shape, 4) or dates.shape != shape[:2] or f["GEO"].shape != (*shape[:3], 3):
            raise ValueError(f"Unsupported L2 layout in {path}")
        for b, (az, el) in enumerate(dirs):
            matches = [k for k, (a, e) in enumerate(beams)
                       if abs((a - az + 180) % 360 - 180) < .05 and abs(e - el) < .05]
            if len(matches) != 1:
                continue
            k = matches[0]
            params, state, geo = f["Fittedparameter"][b], f["Fitstate"][b], f["GEO"][b]
            for t, stamp in enumerate(dates[b]):
                try:
                    utc = datetime.strptime(stamp.decode().strip(), "%d-%b-%Y %H:%M:%S").replace(tzinfo=timezone.utc)
                except ValueError:
                    continue
                index = (int(utc.timestamp()) - start) // 1800
                if not 0 <= index < inversion.shape[2] or not np.isfinite(inversion[k, 0, index]).any():
                    continue
                key = (k, int(utc.timestamp()))
                if key in seen:
                    continue
                if np.any(state[t] == 0):
                    seen.add(key)
                z = geo[t, :, 2]
                geometry = np.isfinite(z) & (z > 0)
                for q in range(4):
                    v = params[t, :, q].copy()
                    v[(state[t] != 0) | ~np.isfinite(v) | ((v <= 0) if q < 3 else False)] = np.nan
                    profile = interpolate(z[geometry], v[geometry], 35.)
                    good = np.isfinite(profile)
                    total[k, q, index, good] += profile[good]
                    count[k, q, index, good] += 1


def render(out, beam, start, stop, inversion, l2, day_only):
    fig, axes = plt.subplots(4, 2, figsize=(11, 10), sharex=True, sharey=True, constrained_layout=True)
    x = np.arange(start, stop + 1800, 1800) / 86400
    paired = np.any(np.isfinite(l2), axis=(0, 2)).sum()
    for q, row in enumerate(axes):
        norm = (LogNorm if q == 0 else Normalize)(*LIMITS[q])
        for ax, values in zip(row, (inversion, l2)):
            mesh = ax.pcolormesh(x, np.r_[HEIGHT - 5, HEIGHT[-1] + 5],
                                 values[q].T, norm=norm, cmap=VELOCITY if q == 3 else "turbo")
            ax.set_ylim(100, 1800)
            ax.set_xlim(x[0], x[-1])
        row[0].set_ylabel("Altitude (km)")
        fig.colorbar(mesh, ax=row, shrink=.9, pad=.025, label=LABELS[q])
    axes[0, 0].set_title("Full-profile inversion")
    axes[0, 1].set_title("SYISR L2 · 30 min means")
    for ax in axes[-1]:
        ax.xaxis.set_major_locator(mdates.HourLocator(interval=4) if day_only else mdates.DayLocator(interval=5))
        ax.xaxis.set_major_formatter(mdates.DateFormatter("%H:%M" if day_only else "%d %b"))
        ax.set_xlabel("UTC")
    az, el = beam
    date_format = "%d %B %Y" if day_only else "%B %Y"
    date = datetime.fromtimestamp(start, timezone.utc).strftime(date_format)
    fig.suptitle(f"SYISR LP480 · {date} · Az {az:.2f}°, El {el:.2f}°\n"
                 f"{paired} windows with matched L2 · LOS velocities as stored")
    if not paired:
        axes[0, 1].text(.5, .5, "No matched L2 windows", transform=axes[0, 1].transAxes, ha="center")
    fig.savefig(out, dpi=155)
    plt.close(fig)


def main():
    parser = argparse.ArgumentParser(__doc__)
    parser.add_argument("campaign", type=Path)
    parser.add_argument("month")
    parser.add_argument("l2", type=Path)
    parser.add_argument("output", type=Path)
    args = parser.parse_args()
    year, month = int(args.month[:4]), int(args.month[4:])
    start = int(datetime(year, month, 1, tzinfo=timezone.utc).timestamp())
    length = calendar.monthrange(year, month)[1]
    stop = start + length * 86400
    fits = {}
    for day in sorted((args.campaign / "days").glob(f"{args.month}??")):
        if not (day / "DAILY_COMPLETE").is_file():
            continue
        for p in (day / "snapshot").glob("*.h5"):
            if p.stem.isdigit() and start <= int(p.stem) < stop:
                fits[int(p.stem)] = read_fit(p)["beams"]
    beams = sorted({(round(b["azimuth_deg"], 2), round(b["elevation_deg"], 2))
                    for profiles in fits.values() for b in profiles}, key=lambda b: (-b[1], b[0]))
    if not beams:
        raise RuntimeError("No completed inversion snapshots")
    shape = (len(beams), 4, length * 48, HEIGHT.size)
    inversion = np.full(shape, np.nan)
    for time, profiles in fits.items():
        if 1726732800 <= time < 1726799400:  # Verified September low-power display mask.
            continue
        for b in profiles:
            key = (round(b["azimuth_deg"], 2), round(b["elevation_deg"], 2))
            for q, field in enumerate(FIELDS):
                v = np.asarray(b[field], dtype=float)
                if q < 3:
                    v[v <= 0] = np.nan
                inversion[beams.index(key), q, (time - start) // 1800] = interpolate(b["altitude_km"], v, 100.)
    total, count = np.zeros(shape), np.zeros(shape, dtype=np.uint16)
    sources = sorted(args.l2.glob(f"OSAYA_TISR01_IFIT_L2_DAY_{args.month}*_*_LP_30sec.h5"))
    seen = set()
    for path in sources:
        l2_rows(path, beams, start, inversion, total, count, seen)
    l2 = np.divide(total, count, out=np.full(shape, np.nan), where=count > 0)
    args.output.mkdir(parents=True, exist_ok=True)
    with h5py.File(args.output / "matched-l2.h5", "w") as dst:
        dst["utc_window_start_s"] = np.arange(start, stop, 1800)
        dst["altitude_km"] = HEIGHT
        dst["beam_az_el_deg"] = beams
        for name, array in (("inversion", inversion), ("l2_mean", l2), ("l2_count", count)):
            dst.create_dataset(name, data=array, compression="gzip")
        dst.attrs["fields"] = ",".join(FIELDS)
        dst.attrs["l2_files"] = json.dumps([str(p) for p in sources])
        dst.attrs["method"] = "Recorded timestamps in UTC, 30 min half-open bins; Fitstate=0; equal accepted dwell weights; bounded adjacent-gate interpolation"
        dst.attrs["velocity_sign"] = "As stored in each source"
    for k, (az, el) in enumerate(beams):
        stem = f"az{round(az*100):05d}-el{round(el*100):04d}"
        render(args.output / f"month-{stem}.png", beams[k], start, stop, inversion[k], l2[k], False)
        render(args.output / f"day01-{stem}.png", beams[k], start, start+86400,
               inversion[k, :, :48], l2[k, :, :48], True)
        print(f"RENDERED {args.month} {stem}", flush=True)


if __name__ == "__main__":
    main()
