"""Independent end-to-end checks; run with conda run -n base python.

Processing and fitting remain native Rust. This audit saves numerical QA in HDF5.
"""
import hashlib
import json
from pathlib import Path
import sys

import h5py
import numpy as np


def metadata(h):
    value = h.attrs["metadata_json"]
    return json.loads(value if isinstance(value, (str, bytes)) else bytes(value))


old, raw, fits = map(Path, sys.argv[1:4])
records = []
for t in range(48):
    source = raw / f"windows/{t:03}"
    result = fits / f"windows/{t:03}"
    cv = json.loads((raw / f"covariance-validation/window-{t:03}.json").read_text())
    with h5py.File(source / "lpm.h5") as h, h5py.File(old / f"windows/{t:03}/lpm.h5") as previous:
        m = metadata(h)
        keep, previous_keep = h["pulse_keep"][:], previous["pulse_keep"][:]
        counts = h["block_retained_pulses"][:]
        assert len(keep) == len(previous_keep)
        assert not np.any((keep > 0) & (previous_keep == 0)), "Reaccepted rejected pulses"
        assert keep.sum() == counts.sum() == m["retained_pulse_count"]
        assert m["screen_end_sample_exclusive"] == 3782
        assert np.all((counts == 0) | (counts >= 25))
        input_count, retained, old_retained = len(keep), keep.sum(), previous_keep.sum()
    with h5py.File(result / "profile.h5") as h, h5py.File(result / "covariance.h5") as c:
        m = metadata(h)
        fit = m["fit"]
        assert fit["status"] == "converged" and fit["normalized_gradient"] < 1e-3
        assert m["requested_observation_max_slant_range_km"] == 5000
        assert len(h["slant_range_km"]) == 23
        assert abs(h["observed_slant_range_km"][-1] - 5000) < 4
        assert h["quadrature_edges_slant_range_km"][-1] == 5500
        assert Path(m["provenance"]["lpm"]).resolve() == (source / "lpm.h5").resolve()
        assert Path(m["measurement_covariance"]["product"]).resolve() == (result / "covariance.h5").resolve()
        assert m["measurement_covariance"]["fixed_during_fit"] is True
        y = np.stack([h[f"observed_{part}_adu2"][:] for part in ("real", "imag")], axis=-1).ravel()
        sigma = np.stack([h[f"sem_{part}_adu2"][:] for part in ("real", "imag")], axis=-1).ravel()
        assert y.size == 6840 and c["structured_cholesky_lower"].shape == (6840, 6840)
        np.testing.assert_array_equal(c["measurement_sigma"][:], sigma)
        assert np.max(np.abs(c["measurement_mean"][:] - y) / sigma) < 1e-6
        assert c.attrs["diagonal_nugget"] == 0.1
        assert c.attrs["structured_fraction"] == cv["selected_alpha"]
        assert cv["selected_score_gain_vs_diagonal"] > 0
        assert .7 < cv["selected_heldout_energy"] < 1.5
        records.append((t, input_count, retained, old_retained, cv["selected_heldout_energy"],
                        cv["selected_score_gain_vs_diagonal"], cv["selected_alpha"],
                        fit["iterations"], fit["normalized_gradient"]))

names = ("window", "input_pulses", "retained_pulses", "previous_retained_pulses",
         "heldout_covariance_energy", "covariance_score_gain", "structured_fraction",
         "final_solve_iterations", "normalized_gradient")
values = np.asarray(records)
assert values[:, 1].sum() == 2154424
with h5py.File(fits / "end-to-end-audit.h5", "w") as h:
    for i, name in enumerate(names):
        h[name] = values[:, i]
    h.attrs["checks"] = "48 converged fits; full-range masks are subsets of original masks; pulse-count conservation; same partial/complete products; 5000-km observations; complete upper support; full covariance belongs to the actual fitted data; held-out covariance prediction; unchanged gradient tolerance"
    h.attrs["script_sha256"] = hashlib.sha256(Path(__file__).read_bytes()).hexdigest()
    h.attrs["uncertainty_scope"] = "Conditional estimated covariance; instrumental, forward-model and covariance-selection uncertainty remain"
print(json.dumps({"windows": 48, "input_pulses": int(values[:, 1].sum()),
                  "retained_pulses": int(values[:, 2].sum()),
                  "additional_rejected_pulses": int((values[:, 3] - values[:, 2]).sum()),
                  "heldout_energy_min_median_max": np.percentile(values[:, 4], [0, 50, 100]).tolist(),
                  "maximum_normalized_gradient": float(values[:, 8].max())}, indent=2))
