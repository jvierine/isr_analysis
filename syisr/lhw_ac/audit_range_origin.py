"""Byte-level real-voltage trace and independent full-file point-echo injections.

This tests software indexing conditional on the header time origin. It cannot
calibrate transmitter/receiver clock offsets from an unknown-range plasma echo.
"""
from pathlib import Path
from datetime import datetime, timezone, timedelta
import argparse, json, subprocess, os
import h5py
import numpy as np

p = argparse.ArgumentParser()
p.add_argument('run', type=Path)
p.add_argument('output', type=Path)
q = p.parse_args()
q.output.mkdir(parents=True, exist_ok=True)
with h5py.File(q.run/'index.h5') as h:
    paths = json.loads(bytes(h['files_json_utf8'][:]))
    bi = h['beam_index'][:]
    offsets = h['frame_offset'][:]
    ids = h['file_index'][:]
    angles = h['beam_az_el_deg'][:]
    meta = json.loads(bytes(h['metadata_json_utf8'][:]))
n = meta['samples']
stride = 1024+(n+1)*8
headers = np.empty((5,3,64,1024), dtype='u1')
voltage = np.empty((5,3,64,n+1,2), dtype='<i4')
selected = np.empty((5,3), int)
for b in range(5):
    ix = np.flatnonzero(bi == b)
    selected[b] = ix[[0,len(ix)//2,-1]]
    for j, ci in enumerate(selected[b]):
        for c in range(64):
            off = int(offsets[ci,c])
            with open(paths[int(ids[ci,c])], 'rb') as f:
                f.seek(off)
                frame = f.read(stride)
                next_magic = f.read(4)
            h = frame[:1024]
            assert int.from_bytes(h[:4],'little') == 0xaaaa5555
            assert int.from_bytes(h[1020:1024],'little') == 0xaa5555aa
            assert not next_magic or int.from_bytes(next_magic,'little') == 0xaaaa5555
            assert int.from_bytes(h[942:944],'little') == c
            az = int.from_bytes(h[856:858],'little')*360/65536
            el = int.from_bytes(h[858:860],'little')*360/65536
            np.testing.assert_allclose([az,el], angles[b], atol=1e-10)
            headers[b,j,c] = np.frombuffer(h,'u1')
            voltage[b,j,c] = np.frombuffer(frame[1024:],'<i4').reshape(n+1,2)
with h5py.File(q.output/'raw-trace.h5','w') as out:
    out.create_dataset('headers',data=headers,compression='gzip')
    out.create_dataset('voltage_q_i_int32',data=voltage,compression='gzip')
    out['selected_cycle_indices'] = selected
    out['frame_offsets_bytes'] = offsets[selected]
    out['source_file_indices'] = ids[selected]
    out['source_paths_json_utf8'] = np.frombuffer(json.dumps(paths).encode(),'u1')
    out['sample_index'] = np.arange(n+1)
    out['nominal_receive_delay_us'] = meta['gate_front_us']+np.arange(n+1)*10
    out['median_received_power_by_beam_sample'] = np.median(np.sum(voltage.astype(float)**2,axis=-1),axis=(1,2))
    out.attrs['checks'] = '960 actual frames: sentinels, following header stride, all code identities and beam directions; original signed Q,I words preserved'
    out.attrs['sample_zero_byte'] = 1024
    out.attrs['frame_stride_bytes'] = stride

codes = np.array([[1 if s=='+' else -1 for s in line.strip()] for line in
                  (q.run/'source/syisr/m32-reference.bc').read_text().splitlines() if line.strip()])
raw = q.output/'fixture'
raw.mkdir(exist_ok=True)
targets = np.array([40,65,90,115,140])
anchor = datetime(2026,11,19,12,tzinfo=timezone.utc)
with open(raw/'fixture.dat1-1-0','wb') as out:
    for start in [10,20,310]:
        for c in range(64):
            for b in range(5):
                h = bytearray(headers[b,0,c].tobytes())
                t = anchor+timedelta(hours=8,seconds=start+(c*5+b)*.016)
                h[12:18] = bytes([t.month,t.year-2000,t.hour,t.day,t.second,t.minute])
                h[18:20] = round(t.microsecond/25).to_bytes(2,'little')
                x = np.zeros(n+1,complex)
                # Independent forward model: an echo is a delayed transmitted
                # voltage waveform, not an index expression copied from decoder.
                tx = np.repeat(codes[c],2)*10000
                source = int(targets[b]*2)
                x[source:source+len(tx)] = tx*np.exp(2j*np.pi*1562.5*np.arange(64)*1e-5)
                v = np.rint(np.stack([x.imag,x.real],axis=-1)).astype('<i4')
                out.write(h)
                out.write(v.tobytes())
env = dict(os.environ, RAYON_NUM_THREADS='2', SLURM_PROCID='0', SLURM_NTASKS='1')
program = q.run/'bin/process'
subprocess.run([str(program),'index',str(raw),'fixture',str(q.output/'fixture-index.h5')],check=True,env=env)
subprocess.run([str(program),'worker',str(q.output/'fixture-index.h5'),str(q.run/'source/syisr/m32-reference.bc'),str(q.output/'fixture-blocks')],check=True,env=env)
records = []
with h5py.File(q.output/'fixture-blocks/0000.h5') as h:
    ranges = h['slant_range_km'][:]
    for b,target in enumerate(targets):
        a = h[f'beam_{b}/unscreened_acf_adu2'][:]
        for lag in [1,2,3,20,40,62]:
            power = np.hypot(a[:,lag,0],a[:,lag,1])
            peak = int(np.argmax(power))
            centroid = np.dot(np.arange(len(power)),power)/power.sum()
            assert peak == target, (b,lag,target,peak)
            assert abs(centroid-target)<1e-8, (b,lag,target,centroid)
            expected = (meta['gate_front_us']+target*20)*.149896229
            assert abs(ranges[peak]-expected)<1e-9
            records.append([b,lag,target,peak,centroid,expected])
with h5py.File(q.output/'software-injection.h5','w') as h:
    h['beam_lag_injected_gate_recovered_gate_centroid_range_km'] = records
    h.attrs['result'] = 'PASS: full binary frames through production index and worker; no 64-sample/32-gate shift'
    h.attrs['qualification'] = 'Conditional on receive delay being relative to transmitted leading edge; does not establish physical hardware time origin'
print('PASS raw byte/beam/code trace and 30 production point-echo lag profiles',flush=True)
