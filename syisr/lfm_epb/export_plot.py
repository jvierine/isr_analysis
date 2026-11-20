"""Compact display copy; full precision signed science products stay in the RTI."""
import argparse
import h5py
import numpy as np
p=argparse.ArgumentParser();p.add_argument('input');p.add_argument('output');a=p.parse_args()
keys=['time_unix_s','time_edges_unix_s','altitude_km','altitude_edges_km','range_km',
      'range_edges_km','noise_power','median_signal','pulse_count']
with h5py.File(a.input) as hi,h5py.File(a.output,'w') as ho:
    for k,v in hi.attrs.items():ho.attrs[k]=v
    ho.attrs['artifact_role']='compact RTI display copy; two-dimensional arrays stored as float32'
    for k in keys:
        v=hi[k][()]
        if v.ndim==2:v=v.astype(np.float32)
        ho.create_dataset(k,data=v,compression='gzip',shuffle=True)
