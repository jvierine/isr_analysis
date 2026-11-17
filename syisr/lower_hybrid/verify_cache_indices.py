"""Require cache-derived indices to match the original sequential-copy hashes."""
import argparse
from pathlib import Path
import h5py
import numpy as np


def main():
    p=argparse.ArgumentParser();p.add_argument('products',type=Path);p.add_argument('original_indices',type=Path);q=p.parse_args()
    for session in sorted(q.products.glob('session_*')):
        if not (session/'recording-index.h5').is_file():continue
        with h5py.File(session/'recording-index.h5','r+') as a,h5py.File(q.original_indices/session.name/'recording-index.h5') as b:
            assert a.attrs['raw_sha256']==b.attrs['raw_sha256']
            for name in ['source_sha256','source_record_counts','time_utc_s','beam_index','beam_geometry']:
                np.testing.assert_array_equal(a[name][:],b[name][:])
            a.attrs['cache_reindex_status']='verified: per-file and concatenated source SHA256, pulse timestamps, beam assignments and geometry agree exactly with original sequential-copy index'
        print('Original/cache agreement:',session.name,flush=True)


if __name__=='__main__':main()
