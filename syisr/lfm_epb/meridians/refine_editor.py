"""Resolution experiment: signed-linear 15s x 1.5km inputs, finite beams.

One whole beam per meridian is withheld to select regularization. Final maps
use every eligible beam (9 is excluded throughout). HDF5 keeps all candidates,
predictions, validation metrics and exact preparation metadata.
"""
from pathlib import Path
import os
import subprocess
import sys
import h5py
import numpy as np
from scipy.interpolate import interp1d
from mpi4py import MPI

GROUPS = {'west': ([8, 7, 6], 7), 'central': ([10, 11, 5, 4], 11),
          'east': ([1, 2, 3], 2)}
LAMBDAS = [100., 25., 6.25, 1.5625]
ROOT = Path(sys.argv[1])
NATIVE = Path(sys.argv[2])
OUT = ROOT / 'editor_refinement'
OUT.mkdir(exist_ok=True)
COMM = MPI.COMM_WORLD
assert COMM.size <= 16


def prepare(name, beam_ids):
    with h5py.File(ROOT / f'meridian_cutouts/{name}_observations.h5') as h:
        old = h['observations'][()]
        stats = h['beam_statistics'][()]
    rows = []
    with h5py.File(ROOT / 'geometry.h5') as geo:
        names = sorted(k for k in geo if k.startswith('beam_'))
        for beam in beam_ids:
            g = geo[names[beam-1]]
            # Average signed display estimates, never dB or clipped powers.
            with h5py.File(ROOT / 'beams' / names[beam-1] / 'display.h5') as h:
                te = h['time_edges_unix_s'][()]
                r = h['range_km'][()]
                it = np.flatnonzero((te[:-1] >= 1772365500.) &
                                    (te[1:] <= 1772372700.))
                raw = h['median_signal'][it[0]:it[-1]+1]
                nt, nr = raw.shape[0]//3, raw.shape[1]//4
                blocks = raw[:nt*3, :nr*4].reshape(nt, 3, nr, 4)
                count = np.isfinite(blocks).sum(axis=(1, 3))
                sig = np.divide(np.where(np.isfinite(blocks), blocks, 0).sum(axis=(1, 3)),
                                count, out=np.full((nt, nr), np.nan), where=count>0)
                t = te[it[0]] + (np.arange(nt)+.5)*15.
                r = r[:nr*4].reshape(nr, 4).mean(axis=1)
            rr = old[old[:, 1] == beam-1]
            unique, ix = np.unique(rr[:, 2], return_index=True)
            reference = np.interp(r, unique, rr[ix, 8], left=np.nan, right=np.nan)
            reference[np.min(abs(r[:, None]-unique), axis=1) > 7.] = np.nan
            ot, oi = np.unique(rr[:, 0], return_index=True)
            noise = np.interp(t, ot, rr[oi, 10])
            z = g['geodetic_lat_lon_alt'][()][:, 2]
            gr = g['range_km'][()]
            center = g['field_line_reference_plane_km'][()][:, :2]
            straight = g['straight_B_projection_km'][()][:, :2]
            quad = g['beam_quadrature_field_line_km'][()][:, :, :2].reshape(len(gr), 18)
            aspect = g['aspect_deviation_from_perpendicular_deg'][()]
            mapped = interp1d(gr, np.column_stack([z, center, straight, quad, aspect]),
                              axis=0, bounds_error=True)(r)
            width = g['oneway_fwhm_elevation_azimuth_deg'][()]
            for k in np.flatnonzero((mapped[:, 0]>=200) & (mapped[:, 0]<=900) &
                                    np.isfinite(reference) & (reference>0)):
                p = np.empty((nt, 34))
                p[:, 0] = t; p[:, 1] = beam-1; p[:, 2] = r[k]
                p[:, 3:6] = mapped[k, :3]
                p[:, 6] = sig[:, k]/reference[k]
                # Eight times as many samples as the former 30s x 6km input.
                # Preserve total empirical data weight rather than silently
                # reducing regularization merely by increasing sample count.
                p[:, 7] = np.sqrt((.08*noise/reference[k])**2+.1**2)*np.sqrt(8)
                p[:, 8] = reference[k]; p[:, 9] = sig[:, k]; p[:, 10] = noise
                p[:, 11:13] = mapped[k, 3:5]
                p[:, 13:31] = mapped[k, 5:23]
                p[:, 31] = mapped[k, 23]; p[:, 32:34] = width
                rows.append(p)
    obs = np.concatenate(rows)
    eligible = np.isfinite(obs).all(axis=1) & (obs[:, 6]<6)
    with h5py.File(OUT / f'{name}_observations.h5', 'w') as h:
        h.create_dataset('observations', data=obs, compression='gzip', shuffle=True)
        h['eligible'] = eligible.astype(float); h['beam_statistics'] = stats
        h.attrs['preparation'] = '15s x 1.499km finite-aware arithmetic mean of signed 5s x 0.37474km display pulse medians. Same quiet reference as former fit; no logarithmic averaging. Empirical weights divided by8 for sample-count invariance.'
        h.attrs['limitations'] = 'No claim of independent 1.5km resolution; matched-filter response, finite beam, noise, frozen-pattern and field-alignment assumptions still limit recovery.'
    print('PREPARED', name, obs.shape, flush=True)


def fit(name, lam, fold, target):
    env = dict(os.environ, SYISR_CAVITY_DEPLETION='.2', SYISR_CAVITY_ITERATIONS='8000')
    subprocess.run([str(NATIVE), 'fit', str(OUT/f'{name}_observations.h5'),
                    str(target), '.12', '0', str(lam), str(fold)], env=env, check=True)


def metrics(name, held, path):
    with h5py.File(OUT/f'{name}_observations.h5') as h:
        p = h['observations'][()]; use = h['eligible'][()].astype(bool)
    with h5py.File(path) as h:
        pred = h['predicted_ratio'][()]; diag = h['diagnostics'][()]
    # Event depletion morphology is the target, not bright enhanced echoes.
    use &= (p[:, 1]==held-1) & np.isfinite(pred) & (abs(p[:, 31])>=2)
    use &= (p[:, 0]>=1772366400) & (p[:, 0]<1772370900) & (p[:, 3]>=220) & (p[:, 3]<=600)
    e = pred-p[:, 6]
    depleted = use & (p[:, 6]<.6)
    rho = np.where(e>=-.2, e*e, -.4*e-.04)
    return np.array([np.sqrt(np.mean(e[depleted]**2)), np.mean(rho[use]),
                     np.sqrt(np.mean(e[use]**2)), depleted.sum(), use.sum(),
                     diag[1], diag[2]])


for j, (name, (ids, held)) in enumerate(GROUPS.items()):
    if j % COMM.size == COMM.rank:
        prepare(name, ids)
COMM.Barrier()
jobs = [(name, lam, held) for name, (_, held) in GROUPS.items() for lam in LAMBDAS]
for j, (name, lam, held) in enumerate(jobs):
    if j % COMM.size != COMM.rank:
        continue
    target = OUT/f'{name}_cv_{lam:g}.h5'
    fit(name, lam, 100+held-1, target)
    print('VALIDATION', name, lam, metrics(name, held, target), flush=True)
COMM.Barrier()
for j, (name, (_, held)) in enumerate(GROUPS.items()):
    if j % COMM.size != COMM.rank:
        continue
    scores = np.array([metrics(name, held, OUT/f'{name}_cv_{lam:g}.h5') for lam in LAMBDAS])
    # Require the robust all-event score not to worsen by more than 2%.
    valid = scores[:, 1] <= scores[0, 1]*1.02
    chosen = int(np.argmin(np.where(valid, scores[:, 0], np.inf)))
    lam = LAMBDAS[chosen]
    fit(name, lam, -1, OUT/f'{name}.h5')
    with h5py.File(OUT/f'{name}.h5', 'a') as h:
        h['candidate_lambdas'] = LAMBDAS; h['heldout_metrics'] = scores
        h.attrs['heldout_beam'] = held; h.attrs['selected_lambda'] = lam
        h['heldout_metrics'].attrs['columns'] = 'depleted_RMSE,asymmetric_Huber_mean,all_RMSE,n_depleted,n_all,iterations,max_iteration_change'
        h.attrs['selection'] = 'Minimize heldout depleted RMSE among fits within2percent of lambda100 all-event asymmetric Huber score. Entire beam withheld for validation, then refit all eligible beams. Beam9 never fitted.'
    print('SELECTED', name, lam, scores, flush=True)
COMM.Barrier()
if COMM.rank == 0:
    print('REFINEMENT_COMPLETE', flush=True)
