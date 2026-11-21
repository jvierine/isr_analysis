"""Check FFmpeg full-decode progress and retain movie/source provenance."""
from pathlib import Path
import sys,hashlib,json
import h5py,numpy as np
r=Path(sys.argv[1]);label=sys.argv[2] if len(sys.argv)>2 else '15min_step5min';records=[]
with h5py.File(r/'movie-input.h5') as h:n=len(h['window_start_utc_s']);geo=h['beam_geometry'][:];integration=float(h.attrs['integration_wall_s']);smoothing=int(h.attrs.get('range_smoothing_gates',1));delay=float(h.attrs.get('range_delay_correction_us',0))
for name,mult in [(f'{a}km',1) for a in [150,200,250,300,350]]+[('all',5)]:
 d=dict(line.split('=',1) for line in (r/'logs'/f'decode-{name}.txt').read_text().splitlines() if '=' in line)
 assert d['progress']=='end';assert int(d['frame'])==n*3*mult,(name,d);assert abs(int(d['out_time_us'])/1e6-n/8*mult)<.05
 records.append([name,int(d['frame']),float(d['out_time_us'])/1e6])
joined=r/f'movies/syisr_lhw_ac_20261119_all_beams_{label}.mp4'
with h5py.File(r/'movies/movies.h5','w') as h:
 h['beam_geometry']=geo;h['full_decode_checks_json_utf8']=np.frombuffer(json.dumps(records).encode(),dtype='u1');h.attrs.update(scientific_frames_per_beam=n,scientific_frames_per_s=8,encoded_frames=n*3*5,duration_s=n*5/8,integration_wall_s=integration,range_smoothing_gates=smoothing,range_delay_correction_us=delay,step_wall_s=300,processor_git_commit=(r/'source-commit.txt').read_text().strip(),renderer_source_sha256=hashlib.sha256((r/'source/syisr/lhw_ac/plot/src/main.rs').read_bytes()).hexdigest(),renderer_binary_sha256=hashlib.sha256((r/'bin/renderer').read_bytes()).hexdigest(),movie_sha256=hashlib.sha256(joined.read_bytes()).hexdigest(),validation='all five beam movies and concatenation fully decoded; frame counts and duration checked')
print('PASS all movies',records,flush=True)
