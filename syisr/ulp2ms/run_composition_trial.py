#!/usr/bin/env python3
"""Run a controlled ULP composition comparison; numerical fitting stays in Rust.
Use conda run -n base python. Existing baseline data/covariance are read-only.
"""
import argparse
import concurrent.futures
import json
import os
import subprocess
from pathlib import Path
import h5py
import numpy as np

GRID = '80,100,150,200,250,300,350,400,500,600,700,800,1000,1250,1500,1800,2150,2550,3000,3550,4200,5000,5500'


def metadata(path):
    with h5py.File(path) as f:
        return json.loads(bytes(f.attrs['metadata_json']))


def main():
    parser = argparse.ArgumentParser(__doc__)
    parser.add_argument('source_root', type=Path)
    parser.add_argument('baseline_root', type=Path)
    parser.add_argument('output_root', type=Path)
    parser.add_argument('--windows', type=int, nargs='+', default=[0, 16, 24, 40])
    parser.add_argument('--workers', type=int, default=4)
    parser.add_argument('--threads', type=int, default=3)
    parser.add_argument('--iterations', type=int, default=500)
    parser.add_argument('--bin-directory', type=Path, default=Path('target/release'))
    parser.add_argument('--summarize-only', action='store_true')
    args = parser.parse_args()
    for name in ['source_root', 'baseline_root', 'output_root', 'bin_directory']:
        setattr(args, name, getattr(args, name).resolve())
    if args.source_root == args.output_root or args.baseline_root == args.output_root:
        raise ValueError('Trial output must be separate from inputs')
    if not args.windows or any(w < 0 or w >= 48 for w in args.windows) or len(set(args.windows)) != len(args.windows):
        raise ValueError('Provide unique window indices in 0..47')
    args.output_root.mkdir(parents=True, exist_ok=True)
    env = os.environ.copy()
    # Explicitly reject unrelated inherited fitting switches.
    for name in ['SYISR_BOTTOMSIDE_CHAPMAN', 'SYISR_COVARIANCE_SHRINKAGE', 'OMPI_COMM_WORLD_RANK', 'OMPI_COMM_WORLD_SIZE']:
        env.pop(name, None)
    env.update(SYISR_OBSERVED_MAX_KM='5000', SYISR_PROFILE_OPTIMIZER='bfgs',
               SYISR_PROFILE_ITERATIONS=str(args.iterations), SYISR_PROFILE_THREADS=str(args.threads),
               SYISR_COMPOSITION_SHAPE_ROOT=str(args.output_root / 'shapes'),
               SYISR_PARTIAL_ROOT=str(args.source_root), SYISR_INITIAL_ROOT=str(args.baseline_root),
               SYISR_COVARIANCE_ROOT=str(args.baseline_root), SYISR_PLOT_MAX_KM='2500')
    common = [str(args.source_root), str(args.output_root), str(args.baseline_root / 'windows/024/profile.h5'), GRID]

    def run(window):
        current = env.copy()
        current.update(SYISR_WINDOW_START=str(window), SYISR_WINDOW_LIMIT=str(window + 1))
        with (args.output_root / f'trial-{window:03d}.log').open('w') as log:
            subprocess.run([str(args.bin_directory / 'profile-campaign'), 'run', *common],
                           env=current, stdout=log, stderr=log, check=True)
        return window

    if not args.summarize_only:
        with concurrent.futures.ThreadPoolExecutor(max_workers=args.workers) as pool:
            for window in pool.map(run, args.windows):
                print(f'Fit finished for window {window:03d}', flush=True)
    records = []
    for w in args.windows:
        old = args.baseline_root / f'windows/{w:03d}/profile.h5'
        new = args.output_root / f'windows/{w:03d}/profile.h5'
        a, b = metadata(old), metadata(new)
        if a['fit']['status'] != 'converged' or b['fit']['status'] != 'converged':
            raise ValueError(f'Window {w} requires convergence review')
        if a['measurement_covariance']['product'] != b['measurement_covariance']['product']:
            raise ValueError('Covariance differs between comparison fits')
        with h5py.File(old) as fa, h5py.File(new) as fb:
            for key in ['observed_real_adu2', 'observed_imag_adu2', 'sem_real_adu2', 'sem_imag_adu2', 'observed_slant_range_km', 'lag_us', 'time_utc_s']:
                np.testing.assert_array_equal(fa[key][:], fb[key][:])
        subprocess.run([str(args.bin_directory / 'compare-composition'), str(old), str(new), str(args.output_root / f'comparison-{w:03d}')], check=True, env=env)
        h, he = b['h_transition'], b['helium_layer']
        record = dict(window=w, old_data_chi_square=a['fit']['data_chi_square'],
                      new_data_chi_square=b['fit']['data_chi_square'],
                      old_penalty=a['fit']['penalty'], new_penalty=b['fit']['penalty'],
                      normalized_gradient=b['fit']['normalized_gradient'],
                      h50_altitude_km=h['crossing50_altitude_km'], h_local_width_km=h['local_scale_width_km'],
                      he_non_h_amplitude=he['peak_non_h_fraction'], he_center_km=he['center_km'], he_sigma_km=he['sigma_km'])
        records.append(record)
    subprocess.run([str(args.bin_directory / 'profile-validate'), str(args.output_root), ','.join(map(str,args.windows))], check=True, env=env)
    with h5py.File(args.output_root / 'comparison.h5', 'w') as f:
        f.attrs['interpretation'] = 'Matched measured products, full covariance, fixed gain and thermodynamic regularization. Five global composition parameters replace 46 knot coordinates and their roughness prior. Compare data costs separately; nominal solutions and conditional errors do not establish model correctness or total uncertainty.'
        f.attrs['baseline_root'] = str(args.baseline_root)
        f.attrs['source_root'] = str(args.source_root)
        for key in records[0]:
            f.create_dataset(key, data=[r[key] for r in records])
    for record in records:
        print(json.dumps(record))


if __name__ == '__main__':
    main()
