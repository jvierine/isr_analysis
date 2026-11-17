#!/usr/bin/env python3
"""Generate fixed IRI2020 temperature/gravity shapes for zenith ULP windows.
Run with base conda Python. External model sources/data are never vendored.
The Fortran driver interface matches syisr/operations/iri_shape_driver.f90
at alternating-code commit 0dbc990. Outputs are HDF5, with source hashes.
"""
import argparse
import datetime as dt
import hashlib
import json
import subprocess
from pathlib import Path
import h5py
import numpy as np


def main():
    parser = argparse.ArgumentParser(__doc__)
    parser.add_argument("driver", type=Path)
    parser.add_argument("model_directory", type=Path)
    parser.add_argument("source_root", type=Path)
    parser.add_argument("output_directory", type=Path)
    args = parser.parse_args()
    args.output_directory.mkdir(parents=True, exist_ok=True)
    driver = args.driver.resolve()
    model = args.model_directory.resolve()
    hashes = {p.name: hashlib.sha256(p.read_bytes()).hexdigest()
              for p in model.iterdir() if p.is_file()}
    for index in range(48):
        with h5py.File(args.source_root / f"windows/{index:03d}/lpm.h5") as f:
            times = f["time_utc_s"][:]
        midpoint = (times[0] + times[-1] + 0.04) / 2
        when = dt.datetime.fromtimestamp(midpoint, dt.timezone.utc)
        command = [str(driver), *[str(v) for v in
                   (when.year, when.month, when.day, when.hour, when.minute,
                    when.second, 18.3492, 109.6222, 500, 2000, 25)], "."]
        result = subprocess.run(command, cwd=model, text=True, capture_output=True, check=True)
        rows = np.array([[float(v) for v in line.split()]
                         for line in result.stdout.splitlines()[:61]])
        if rows.shape != (61, 12) or not np.all(np.isfinite(rows[:, 3])) or not np.all(rows[:, 3] > 100):
            raise ValueError(f"Invalid IRI ion temperatures in window {index}")
        z, ti = rows[:, 0], rows[:, 3]
        gravity = 9.80665 * (6371 / (6371 + z)) ** 2
        raw = 1.380649e-23 * ti / (15 * 1.66053906660e-27 * gravity) / 1000
        scale = np.maximum.accumulate(raw)
        record = dict(altitude_km=z.tolist(), scale_height_km=scale.tolist(),
                      source=f"IRI2020 Ti/g differential H/O scale height; monotone cumulative maximum; UTC {when.isoformat()}; zenith lat18.3492 lon109.6222; linear continuation above2000km",
                      iri_ti_k=ti.tolist(), unmodified_scale_height_km=raw.tolist(),
                      time_utc_s=midpoint, model_directory=str(model), sha256=hashes,
                      driver_sha256=hashlib.sha256(driver.read_bytes()).hexdigest())
        with h5py.File(args.output_directory / f"{index:03d}.h5", "w") as f:
            f.attrs["metadata_json"] = np.frombuffer(json.dumps(record).encode(), dtype=np.uint8)
            for name, values in [("altitude_km", z), ("iri_ti_k", ti),
                                 ("scale_height_km", scale), ("unmodified_scale_height_km", raw)]:
                f.create_dataset(name, data=values)
    print("Saved 48 IRI composition-reference HDF5 products")

if __name__ == "__main__":
    main()
