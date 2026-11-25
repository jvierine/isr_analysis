"""Southern LFM reconstruction and its physical-altitude echo predictions."""
import argparse
from pathlib import Path
import h5py
import matplotlib.pyplot as plt
import numpy as np
from matplotlib.colors import ListedColormap

parser = argparse.ArgumentParser()
parser.add_argument("directory", type=Path)
parser.add_argument("palette", type=Path, help="Rustplotlib Storm RGB lookup")
args = parser.parse_args()
palette = ListedColormap(np.loadtxt(args.palette))
with h5py.File(args.directory / "south_observations.h5") as h:
    rows = h["observations"][()]
with h5py.File(args.directory / "south.h5") as h:
    prediction = h["predicted_ratio"][()]
    x, y = h["x_km"][()], h["y_km"][()]
    speed, north_speed, _, _, reference_time, *_ = h["parameters"][()]
    support = h["support_weight"][()]
    relative = np.where(support > .005 * support.max(), 1 + h["contrast"][()], np.nan)
fig, (ax, echo_ax) = plt.subplots(1, 2, figsize=(15, 6), sharey=True, layout="constrained")
art = ax.pcolormesh(x, y, relative, cmap=palette, vmin=0, vmax=1, shading="nearest")
fig.colorbar(art, ax=ax, label="Fitted relative echo")
# Bin the forward predictions at the mapped beam centres. Their amplitudes
# retain the local quiet layer, unlike the fractional depletion map.
ok = np.isfinite(prediction)
dt = rows[ok, 0] - reference_time
coords = (rows[ok, 5] - north_speed*dt, rows[ok, 4] - speed*dt)
edges = lambda grid: np.r_[grid - (grid[1]-grid[0])/2, grid[-1] + (grid[1]-grid[0])/2]
bins = (edges(y), edges(x))
power = prediction[ok]*rows[ok, 8]*rows[ok, 2]**2
count = np.histogram2d(*coords, bins=bins)[0]
total = np.histogram2d(*coords, bins=bins, weights=power)[0]
scale = np.nanpercentile(rows[ok, 8]*rows[ok, 2]**2, 95)
echo = np.divide(total, count*scale, out=np.full_like(total, np.nan), where=count>0)
art = echo_ax.pcolormesh(x, y, 10*np.log10(np.maximum(echo, 1e-3)), cmap=palette,
                        vmin=-25, vmax=3, shading="nearest")
fig.colorbar(art, ax=echo_ax, label="Range-squared echo (dB relative to quiet F region)")
ax.set(ylabel="Cross-field upward (km)", title="Southern cut · fitted depletion")
echo_ax.set(title="Predicted echo at mapped beam positions")
iy, ix = np.where(np.isfinite(relative) | (count > 0))
for panel in (ax, echo_ax):
    panel.set(xlabel="Moving cross-field east (km)", xlim=(x[ix.min()]-10, x[ix.max()]+10),
              ylim=(y[iy.min()]-10, y[iy.max()]+10))
fig.savefig(args.directory / "southern_cut.png", dpi=180)

fig, axes = plt.subplots(2, 3, figsize=(15, 7), sharex=True, sharey=True, layout="constrained")
for j, beam in enumerate((6, 4, 3)):
    select = rows[:, 1] == beam - 1
    p = rows[select]
    t = np.unique(p[:, 0])
    nr = len(p)//len(t)
    z = p[::len(t), 3]
    # Use linear range-squared echo power to retain the actual layer envelope.
    measured = p[:, 9]*p[:, 2]**2
    predicted = prediction[select]*p[:, 8]*p[:, 2]**2
    scale = np.nanpercentile(p[:, 8]*p[:, 2]**2, 95)
    for i, power in enumerate((measured, predicted)):
        # Display floor applies to signed power only after reconstruction.
        db = 10*np.log10(np.maximum(power/scale, 1e-3))
        art = axes[i, j].pcolormesh(t/3600 % 24, z, db.reshape(nr, len(t)),
                                  cmap=palette, vmin=-25, vmax=3, shading="nearest")
    axes[0, j].set_title(f"Beam {beam}")
    axes[1, j].set_xlabel("UTC hour")
axes[0, 0].set_ylabel("Measured · altitude (km)")
axes[1, 0].set_ylabel("Predicted · altitude (km)")
axes[1, 0].set_ylim(100, 700)
fig.colorbar(art, ax=axes, label="Range-squared echo (dB relative to quiet F region)")
fig.savefig(args.directory / "southern_bottomside_rtis.png", dpi=180)
