"""Coherent pulse-edge timing diagnostic, with an older pulse-width control.

The edge is only a timing candidate: identifying it as zero-range transmitter
leakage requires hardware confirmation. It is not calibrated from ion lines.
"""
import argparse
import hashlib
from pathlib import Path

import h5py
import numpy as np


def edge(power,front,pulse,dt):
    plateau=np.median(power[:5]);normalized=power/plateau
    k=int(np.flatnonzero(normalized<0.5)[0]);assert k>=1
    return normalized,np.array([front+(k-1)*dt-pulse,front+k*dt-pulse])


def main():
    p=argparse.ArgumentParser();p.add_argument('index',type=Path);p.add_argument('audit',type=Path);p.add_argument('controls',type=Path)
    q=p.parse_args()
    with h5py.File(q.index) as ix,h5py.File(q.audit,'r+') as h:
        if 'pulse_edge' in h:del h['pulse_edge']
        out=h.create_group('pulse_edge')
        n=len(ix['beam_index']);beam=ix['beam_index'][:];stride=int(ix.attrs['record_stride_bytes']);ns=int(ix.attrs['sample_count'])
        mm=np.memmap(ix.attrs['raw_path'],mode='r',dtype='u1');iq=np.ndarray((n,ns,2),dtype='<i4',buffer=mm,offset=1024,strides=(stride,8,4))
        traces=[];coherent=[];bounds=[]
        for b in range(5):
            chosen=np.flatnonzero(beam==b)[100:4100:2]
            raw=iq[chosen,:80,:].astype(float);z=raw[:,:,1]+1j*raw[:,:,0]
            power=np.mean(abs(z)**2,axis=0);coh=abs(z.mean(axis=0))**2
            norm,bound=edge(power,533.,480.,10.)
            traces.append(norm);coherent.append(coh/power);bounds.append(bound)
        out['new_time_after_commanded_pulse_end_us']=533.+np.arange(80)*10.-480.
        out['new_normalized_power']=traces;out['new_coherent_fraction']=coherent;out['new_edge_delay_bracket_us']=bounds
        names=[];prefix_hashes=[];old=[];old_times=[];old_bounds=[]
        for file in sorted(q.controls.glob('*.dat*'))[:12]:
            with file.open('rb') as f:header=f.read(1024);body=f.read(80*8)
            u16=lambda i:int.from_bytes(header[i:i+2],'little')
            assert header[36]==0x41 and u16(988)==2000 and u16(766)==2001
            raw=np.frombuffer(body,dtype='<i4').reshape(-1,2).astype(float);z=raw[:,1]+1j*raw[:,0]
            norm,bound=edge(abs(z)**2,2001.,2000.,10.)
            names.append(str(file.resolve()));prefix_hashes.append(hashlib.sha256(header+body).hexdigest())
            old.append(norm);old_times.append(2001.+np.arange(80)*10.-2000.);old_bounds.append(bound)
        out['control_files']=np.asarray(names,dtype=h5py.string_dtype())
        out['control_prefix_sha256']=np.asarray(prefix_hashes,dtype=h5py.string_dtype())
        out['control_normalized_power']=old;out['control_time_after_commanded_pulse_end_us']=old_times
        out['control_edge_delay_bracket_us']=old_bounds
        out.attrs['candidate_correction_us']=160.
        out.attrs['sampling_interval_us']=10.
        out.attrs['method']='first drop below half the initial plateau power; edge bracket is last above/first below sample; no sub-sample calibration'
        out.attrs['status']='provisional zero-range transmitter-leakage interpretation; receiver blanking/switch transients and exact emission timing not independently measured'
        out.attrs['independent_of_ion_line_narrowing']=True
        out.attrs['script_sha256']=hashlib.sha256(Path(__file__).read_bytes()).hexdigest()
        print('480 us pulse, five beam edge brackets (us)',np.asarray(bounds))
        print('2000 us pulse, twelve control edge brackets (us)',np.asarray(old_bounds))
        print('Candidate correction: 160 us, 23.9834 km; hardware interpretation provisional')


if __name__=='__main__':main()
