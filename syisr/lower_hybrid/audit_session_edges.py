"""Check near-gate coherent edges independently in every source file/beam.

No ion-line alignment is used. The zero-range leakage interpretation remains
provisional; the HDF5 records the sampled-pulse evidence for each source.
"""
import argparse
from pathlib import Path
import hashlib
import h5py
import numpy as np


def main():
    p=argparse.ArgumentParser();p.add_argument('raw_directory',type=Path);p.add_argument('output',type=Path);q=p.parse_args()
    rows=[];traces=[];coherent=[];names=[]
    files=sorted(q.raw_directory.glob('*.dat*'),key=lambda p:(p.name.split('.dat')[0],int(p.name.rsplit('-',1)[1])))
    for f in files:
        header=f.open('rb').read(1024);n=int.from_bytes(header[768:772],'little');stride=1024+(n+1)*8
        assert int.from_bytes(header[988:990],'little')==480 and int.from_bytes(header[766:768],'little')==533
        count=f.stat().st_size//stride;mm=np.memmap(f,mode='r',dtype='u1')
        angles=np.ndarray((count,),dtype='<u2',buffer=mm,offset=858,strides=(stride,))
        iq=np.ndarray((count,n,2),dtype='<i4',buffer=mm,offset=1024,strides=(stride,8,4))
        # Limit header reads to a deterministic, spread-out pulse sample.
        candidates=np.unique(np.linspace(100,min(count-1,10000),500,dtype=int));els=np.unique(angles[candidates])[::-1]
        for el in els:
            chosen=candidates[angles[candidates]==el];raw=iq[chosen,:80,:].astype(float);z=raw[:,:,1]+1j*raw[:,:,0]
            power=np.mean(abs(z)**2,axis=0);norm=power/np.median(power[:5]);k=int(np.flatnonzero(norm<.5)[0])
            assert k>0
            bounds=[533+(k-1)*10-480,533+k*10-480]
            rows.append([len(names),float(el)*360/65536,len(chosen),*bounds]);traces.append(norm);coherent.append(abs(z.mean(axis=0))**2/power)
        names.append(str(f.resolve()));print(f.name,rows[-len(els):],flush=True)
    with h5py.File(q.output,'w') as h:
        h['source_paths']=np.asarray(names,dtype=h5py.string_dtype());h['edge_summary']=rows
        h.attrs['columns']='source_index,elevation_deg,sampled_pulses,last_above_half_delay_us,first_below_half_delay_us'
        h['time_after_commanded_pulse_end_us']=533+np.arange(80)*10-480
        h['normalized_power']=traces;h['coherent_fraction']=coherent
        h.attrs['status']='Zero-range transmitter-leakage interpretation is provisional; exact hardware timing reference unverified'
        h.attrs['script_sha256']=hashlib.sha256(Path(__file__).read_bytes()).hexdigest()
    print('Edge brackets:',np.unique(np.asarray(rows)[:,-2:],axis=0))


if __name__=='__main__':main()
