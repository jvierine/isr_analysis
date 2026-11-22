"""Stage alternative *initial guesses* and inspect immutable LP480 lag products.

All fitted parameters remain free in the existing native Rust solver. This is
an isolated diagnostic, not a replacement campaign or a confidence calculation.
"""
import argparse
import copy
import json
import os
from pathlib import Path
import subprocess

import h5py
import numpy as np


def load(path):
    with h5py.File(path) as f:
        return json.loads(bytes(f['json_utf8'][:]))


def save(path, value):
    path.parent.mkdir(parents=True, exist_ok=True)
    with h5py.File(path, 'w') as f:
        f.create_dataset('json_utf8', data=np.frombuffer(json.dumps(value).encode(), dtype='u1'), compression='gzip')


def stage(month, output):
    cases = []
    for date in ['20240520', '20240531']:
        day = month/'days'/date
        fits = sorted(day.joinpath('final').glob('[0-9]*.h5'))
        candidates = [p for p in fits if int(p.stem) % 86400 == 18*3600]
        if len(candidates) != 1:
            raise ValueError(f'missing nighttime observation: {date}')
        original = load(candidates[0])
        time = original['start_utc_s']
        prior = output/date/'neighbors'
        prior.mkdir(parents=True, exist_ok=True)
        for other in [time-1800, time+1800]:
            source = day/'final'/f'{other}.h5'
            if source.exists():
                link = prior/source.name
                if link.exists():
                    if link.resolve() != source.resolve():
                        raise ValueError('changed diagnostic neighbor')
                else:
                    link.symlink_to(source.resolve())
        for label, center, width, cool in [
            ('nominal', None, None, False),
            ('h1000-w150', 1000, 150, False),
            ('h1400-w250', 1400, 250, False),
            ('h1000-w150-cool', 1000, 150, True),
        ]:
            trial = copy.deepcopy(original)
            if center is not None:
                q = [(center-550)/1450, np.log(width/20)/np.log(15)]
                trial['shared_parameters'][:2] = q
                for b in trial['beams']:
                    b['controls'][0]['h_fraction'], b['controls'][1]['h_fraction'] = q
                    if cool:
                        ti = np.interp(450, b['altitude_km'], b['ti_k'])
                        te = np.interp(450, b['altitude_km'], b['te_k'])
                        vi = np.interp(450, b['altitude_km'], b['velocity_m_s'])
                        knots = b['spline']['knots']
                        for i, node in enumerate(b['controls']):
                            z = sum(knots[i+1:i+4])/3
                            if z > 450:
                                node['ti_k'] = float(ti*(z/450)**0.25)
                                node['te_k'] = float(te*(z/450)**0.4)
                                node['velocity_m_s'] = float(vi)
            directory = output/date/label
            seed = directory/'seed'/f'{time}.h5'
            save(seed, trial)
            cases.append(dict(time=time, label=label, input=str(day/'input'),
                              prior=str(prior), directory=str(directory),
                              nominal=str(candidates[0].resolve())))
        # Actual, signed, background-subtracted lag observations and predictions.
        with h5py.File(output/f'acfs-{date}.h5', 'w') as f:
            f.attrs['time_utc_s'] = time
            f.attrs['note'] = 'Nominal immutable fit; signed products, no normalization or clipping'
            for b in original['beams']:
                decoded = load(day/'input'/'products'/f"{b['id']}.h5")
                g = f.create_group(b['id'])
                g.attrs['azimuth_deg'] = b['azimuth_deg']
                g.attrs['elevation_deg'] = b['elevation_deg']
                altitudes = np.array([v['altitude_km'] for v in decoded['gates']])
                gate = np.array(b['measurement_gate'])
                lag = np.array(b['measurement_lag'])
                chosen = sorted(set(int(np.argmin(abs(altitudes-z))) for z in [700,900,1100,1300,1500,1700]))
                for index in chosen:
                    sel = gate == index
                    if not sel.any():
                        continue
                    row = g.create_group(f'gate-{index}')
                    row.attrs['altitude_km'] = altitudes[index]
                    row.attrs['gate_definition'] = json.dumps(decoded['gates'][index])
                    row.create_dataset('lag_us', data=lag[sel]*10.)
                    for name in ['observed','predicted','assigned_sigma','residual']:
                        row.create_dataset(name, data=np.array(b[name])[sel], compression='gzip')
    save(output/'cases.h5', cases)


def run(output, binary):
    rank = int(os.environ['SLURM_PROCID'])
    size = int(os.environ['SLURM_NTASKS'])
    for i, case in enumerate(load(output/'cases.h5')):
        if i % size != rank:
            continue
        directory = Path(case['directory'])
        env = os.environ.copy()
        env.update(SLURM_PROCID='0', SLURM_NTASKS='1', SYISR_INITIAL_PROFILE_DIR=str(directory/'seed'))
        with (directory/'fit.log').open('w') as log:
            subprocess.run([str(binary), 'fit', case['input'], str(case['time']), str(directory/'fits'), case['prior'], '0'],
                           env=env, stdout=log, stderr=subprocess.STDOUT, check=True)


if __name__ == '__main__':
    p = argparse.ArgumentParser(__doc__)
    p.add_argument('mode', choices=['stage','run'])
    p.add_argument('month', type=Path)
    p.add_argument('output', type=Path)
    p.add_argument('binary', type=Path)
    a = p.parse_args()
    a.output.mkdir(parents=True, exist_ok=True)
    if a.mode == 'stage':
        stage(a.month, a.output)
    else:
        run(a.output, a.binary)
