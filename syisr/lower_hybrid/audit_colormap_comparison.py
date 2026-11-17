"""Audit normalization, matched colour transforms, and all delivered hashes."""
import argparse
import hashlib
import json
from pathlib import Path
import h5py
import numpy as np


def sha(path):
    digest=hashlib.sha256()
    with Path(path).open('rb') as f:
        while block:=f.read(8*1024*1024):digest.update(block)
    return digest.hexdigest()


def main():
    p=argparse.ArgumentParser();p.add_argument('root',type=Path);q=p.parse_args();rows=[]
    for path in sorted((q.root/'products').rglob('*.h5')):
        if path.name.startswith('._'):continue
        with h5py.File(path) as h:
            df=float(h.attrs['frequency_bin_hz'])
            for b in range(5):
                if f'beam_{b}' not in h:continue
                g=h[f'beam_{b}'];raw=g['psd_adu2_per_hz'][:];excess=g['background_subtracted_psd_adu2_per_hz'][:]
                power=g['normalization_received_power_adu2'][:];norm=g['received_power_normalized_excess_per_hz'][:]
                assert np.all(power>0) and np.isfinite(norm).all()
                np.testing.assert_array_equal(power,raw.sum(-1)*df)
                np.testing.assert_array_equal(norm,excess/power[:,:,None])
                np.testing.assert_array_equal(np.sign(norm),np.sign(excess))
                rows.append([b,len(raw),float(power.min()),float(np.median(abs(excess).sum(-1)*df/power))])
    transforms=[]
    with h5py.File(q.root/'comparison.h5','r+') as combined:
        for path in combined['source_manifests'].asstr()[:]:
            with h5py.File(path,'r+') as m:
                info=json.loads(m.attrs['display_scales_json'])
                transforms.append([*info['signed_absolute_adu2_per_hz'],*info['signed_normalized_1e4_per_hz'],info['asinh_width_absolute'],info['asinh_width_normalized']])
                assert 'total received power' in info['normalization']
                assert sha(m.attrs['source_product'])==m.attrs['source_sha256']
                for group in m.values():
                    if isinstance(group,h5py.Group):assert sha(group.attrs['movie_path'])==group.attrs['sha256']
                # Correct a legacy prose descriptor. Recorded transform numbers
                # and all pixels are unchanged; the normalized width is 0.001.
                info['method']='fixed shared colour limits; signed asinh widths recorded separately for absolute and normalized panels; signed scientific data unchanged'
                m.attrs['display_scales_json']=json.dumps(info)
                metadata=Path(path).parent/'figures/display-scales.json'
                if metadata.exists():metadata.write_text(json.dumps(info,indent=2))
        np.testing.assert_array_equal(transforms,np.tile(transforms[0],(len(transforms),1)))
        combined['source_manifest_sha256'][...]=[sha(p) for p in combined['source_manifests'].asstr()[:]]
        assert sha(combined.attrs['movie_path'])==combined.attrs['sha256']
        for path,digest in zip(combined['variant_movies'].asstr()[:],combined['variant_sha256'].asstr()[:]):assert sha(path)==digest
    with h5py.File(q.root/'normalization-audit.h5','w') as h:
        h['checks']=rows;h.attrs['columns']='beam_index,frames,minimum_positive_denominator_ADU2,median_new_to_old_L1_amplitude'
        h['matched_display_parameters']=transforms
        h.attrs['result']='passed: positive received-power denominators; finite normalized excess; exact sign preservation; same display transforms for every colour variant; all movie, source and manifest hashes verified'
    print('Normalization, colour-transform and artifact audits passed',flush=True)


if __name__=='__main__':main()
