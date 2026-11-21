from pathlib import Path
import sys, hashlib, os
import h5py
r = Path(sys.argv[1])
d = dict(line.split('=',1) for line in (r/'decode.txt').read_text().splitlines() if '=' in line)
with h5py.File(r/'sectors.h5') as h:
    n = len(h['window_start_utc_s'])
assert d['progress']=='end' and int(d['frame'])==n*3
assert abs(float(d['out_time_us'])/1e6-n/8)<.05
with h5py.File(r/'movie-verification.h5','w') as h:
    h.attrs.update(encoded_frames=n*3,duration_s=n/8,git_commit=os.environ['SOURCE_GIT_COMMIT'],movie_sha256=hashlib.sha256((r/'syisr-lhw-ac-stitched-sectors-30min-4gate.mp4').read_bytes()).hexdigest(),result='PASS full decode, frame count and duration')
print('PASS full composite decode',n*3,'frames',n/8,'s',flush=True)
