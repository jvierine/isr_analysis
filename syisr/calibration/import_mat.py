from pathlib import Path
import hashlib
import argparse
import h5py
import numpy as np
from scipy.io import loadmat
parser=argparse.ArgumentParser(description='Preserve the supplied SYISR voltage FIR in HDF5')
parser.add_argument('source', type=Path)
parser.add_argument('--output', type=Path, default=Path(__file__).with_name('receiver_100khz.h5'))
args=parser.parse_args()
p=args.source
h=loadmat(p)['ht'].reshape(-1)
with h5py.File(args.output,'w') as f:
    f.attrs['schema']='syisr-voltage-impulse-response-v1'
    f.attrs['source_filename']=p.name
    f.attrs['source_sha256']=hashlib.sha256(p.read_bytes()).hexdigest()
    f.attrs['source_variable']='ht'
    f.attrs['output_sample_rate_hz']=100000.
    f.attrs['coefficient_interval_s']=1e-6
    f.attrs['coefficient_interval_status']='Inferred from supplied delay-axis plot and existing supplier-file analysis; MAT file contains no timing metadata'
    f.attrs['timing_origin_status']='Unspecified; caller must explicitly select origin and avoid duplicating satellite-calibrated instrument delay'
    f.attrs['coefficient_convention']='Discrete voltage convolution weights, preserved exactly; no dt multiplier'
    f.attrs['dc_gain']=h.sum()
    f.attrs['energy_centroid_index']=np.dot(np.arange(h.size),h*h)/np.dot(h,h)
    f.create_dataset('voltage_impulse_response',data=h)
    f.create_dataset('tap_index',data=np.arange(h.size,dtype='i4'))
with h5py.File(args.output) as f:
    assert np.array_equal(f['voltage_impulse_response'][:],h)
print('Exact float64 round-trip:', h.size, 'coefficients')
