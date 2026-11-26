"""Concatenate verified November phase-table views: pseudo first, then beams."""
from pathlib import Path
import argparse,subprocess,hashlib,json
import h5py,numpy as np
p=argparse.ArgumentParser(description=__doc__);p.add_argument('root',type=Path);a=p.parse_args();r=a.root.resolve()
with h5py.File(r/'spectra.h5') as h:targets=h['beam_geometry'][:,0].astype(int)
if not np.array_equal(targets,[150,200,250,300,350]):raise ValueError('Unsupported beam layout')
products=[r/'pseudo-phase-table/sectors.h5']+[r/'individual-phase-table'/f'beam-{n}km.h5' for n in targets]
with h5py.File(products[0]) as h:starts=h['window_start_utc_s'][:];stops=h['window_end_utc_s'][:]
processing=[]
for product in products:
 with h5py.File(product) as h:
  assert h.attrs['pointing_model']=='phase-table'
  np.testing.assert_array_equal(h['window_start_utc_s'][:],starts);np.testing.assert_array_equal(h['window_end_utc_s'][:],stops)
  processing.append(str(h.attrs['git_commit']))
files=[r/'pseudo-phase-table/syisr-lhw-ac-stitched-sectors-30min-4gate.mp4']+[r/'individual-phase-table'/f'syisr-ac-{n}km-corrected.mp4' for n in targets]
assert all(f.exists() for f in files)
if any("'" in str(f) or '\n' in str(f) for f in files):raise ValueError('Unsupported path characters in FFmpeg manifest')
out=r/'syisr-lhw-ac-phase-table-all-beams.mp4';manifest=r/'phase-movie-concat.txt';manifest.write_text(''.join("file '"+str(f)+"'\n" for f in files))
subprocess.run(['ffmpeg','-v','error','-y','-f','concat','-safe','0','-i',str(manifest),'-c','copy','-movflags','+faststart',str(out)],check=True)
progress=r/'phase-movie-decode.txt'
subprocess.run(['ffmpeg','-v','error','-progress',str(progress),'-i',str(out),'-f','null','-'],check=True)
d=dict(line.split('=',1) for line in progress.read_text().splitlines() if '=' in line);n=len(starts);expected_frames=n*3*len(files);expected_duration=n/8*len(files)
assert d['progress']=='end' and int(d['frame'])==expected_frames
assert abs(float(d['out_time_us'])/1e6-expected_duration)<.05
with h5py.File(r/'phase-movie-verification.h5','w') as h:
 h['source_movie']=np.array([str(f) for f in files],dtype='S512');h['source_movie_sha256']=np.array([hashlib.sha256(f.read_bytes()).hexdigest() for f in files],dtype='S64');h['source_processing_git_commit']=np.array(processing,dtype='S64')
 h.attrs.update(program='syisr-phase-movie-assembly',version='0.1.0',git_commit=subprocess.check_output(['git','rev-parse','HEAD'],cwd=Path(__file__).parent,text=True).strip(),source_sha256=hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),frame_count=expected_frames,duration_s=expected_duration,movie_sha256=hashlib.sha256(out.read_bytes()).hexdigest(),status='PASS pseudo-first ordered concatenation and full decode',pointing_model='phase-table')
print('PASS phase-table movie',out,flush=True)
