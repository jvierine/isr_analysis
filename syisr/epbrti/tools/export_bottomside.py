"""Extend each editor model below its original support on the same 2 km grid."""
import argparse
from pathlib import Path
import h5py
import numpy as np

parser = argparse.ArgumentParser()
parser.add_argument("fits", type=Path)
parser.add_argument("baseline", type=Path)
parser.add_argument("output", type=Path)
args = parser.parse_args()
shape = (3, 351, 701)
seed = np.fromfile(args.baseline / "seed.bin", dtype="<f4").reshape(shape)
support = np.fromfile(args.baseline / "support.bin", dtype="<f4").reshape(shape)
state = np.fromfile(args.baseline / "state.bin", dtype="<f4").reshape((2, *shape))
new_seed, new_support, new_state = seed.copy(), support.copy(), state.copy()
rows = np.arange(shape[1])[:, None]
for g, group in enumerate(("west", "central", "east")):
    with h5py.File(args.fits / f"{group}.h5") as h:
        ratio = 1 + h["contrast"][()]
        weight = h["support_weight"][()]
        y = h["y_km"][()]
    assert ratio.shape == weight.shape == shape[1:]
    assert np.isfinite(ratio).all() and np.isfinite(weight).all()
    old = support[g] > 0
    first = np.argmax(old, axis=0)
    # Stay inside the original zonal domain. Add only cells below its bottom
    # boundary; a smaller support floor is needed as the F-region echo fades.
    added = (rows < first) & old.any(axis=0) & (weight > .001 * weight.max())
    assert added.any(), group
    col = np.arange(shape[2])
    delta = seed[g, first, col] - ratio[first, col]
    taper = np.exp(-np.maximum(first - rows, 0) / 8)
    extended = np.clip(ratio + taper * delta, 0, 1)
    new_seed[g, added] = extended[added]
    new_support[g, added] = 1
    new_state[0, g, added] = extended[added]
    assert np.array_equal(new_state[0, g, old], state[0, g, old])
    print(group, "added cells", added.sum(), "lowest v", y[added.any(axis=1)].min())
args.output.mkdir(parents=True, exist_ok=True)
for name, values in (("seed", new_seed), ("support", new_support), ("state", new_state)):
    values.astype("<f4").tofile(args.output / f"{name}.bin")
