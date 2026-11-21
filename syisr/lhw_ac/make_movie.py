"""Stream inferno movies, concatenate all beams, and fully decode outputs."""
from pathlib import Path
from concurrent.futures import ThreadPoolExecutor
import argparse,subprocess,json,hashlib
import h5py,numpy as np
P=argparse.ArgumentParser();P.add_argument('input',type=Path);P.add_argument('output',type=Path);P.add_argument('--renderer',type=Path);q=P.parse_args();root=Path(__file__).resolve().parent;renderer=(q.renderer or root/'plot/target/release/syisr-lhw-ac-movie').resolve();q.output.mkdir(parents=True,exist_ok=True);qa=q.output/'qa';qa.mkdir(exist_ok=True)
with h5py.File(q.input) as h:n=len(h['window_start_utc_s']);geo=h['beam_geometry'][:]
def render(b):
 alt=int(geo[b,0]);path=q.output/f'syisr_lhw_ac_{alt}km_15min_step5min.mp4'
 with (q.output/f'render-{alt}km.log').open('w') as log:subprocess.run([str(renderer),str(q.input.resolve()),str(q.output/'figures'),f'beam={b}','colormap=inferno','cuts',f'movie={path.resolve()}'],cwd=root/'plot',stdout=log,stderr=subprocess.STDOUT,check=True)
 info=json.loads(subprocess.check_output(['ffprobe','-v','error','-select_streams','v:0','-show_entries','stream=width,height,nb_frames,duration','-of','json',str(path)],text=True))['streams'][0]
 assert (info['width'],info['height'])==(1920,1080);assert int(info['nb_frames'])==n*3;assert abs(float(info['duration'])-n/8)<.05
 subprocess.run(['ffmpeg','-v','error','-i',str(path),'-f','null','-'],check=True)
 for tag,t in [('first',0),('middle',n/16),('last',n/8-.125)]:subprocess.run(['ffmpeg','-v','error','-y','-ss',str(t),'-i',str(path),'-frames:v','1',str(qa/f'{alt}km-{tag}.png')],check=True)
 print('Verified',alt,'km',flush=True);return path
with ThreadPoolExecutor(max_workers=3) as pool:movies=list(pool.map(render,range(5)))
manifest=q.output/'concat.txt';manifest.write_text(''.join("file '"+str(f.resolve())+"'\n" for f in movies));joined=q.output/'syisr_lhw_ac_20261119_all_beams_15min_step5min.mp4';subprocess.run(['ffmpeg','-v','error','-y','-f','concat','-safe','0','-i',str(manifest),'-c','copy','-movflags','+faststart',str(joined)],check=True)
subprocess.run(['ffmpeg','-v','error','-i',str(joined),'-f','null','-'],check=True)
info=json.loads(subprocess.check_output(['ffprobe','-v','error','-select_streams','v:0','-show_entries','stream=nb_frames,duration','-of','json',str(joined)],text=True))['streams'][0];assert int(info['nb_frames'])==n*3*5;assert abs(float(info['duration'])-n*5/8)<.05
with h5py.File(q.output/'movies.h5','w') as h:
 h['beam_geometry']=geo;h.attrs.update(scientific_frames_per_beam=n,scientific_frames_per_s=8,encoded_frames=int(info['nb_frames']),duration_s=float(info['duration']),integration_wall_s=900,step_wall_s=300,source_product=str(q.input.resolve()),renderer_source_sha256=hashlib.sha256((root/'plot/src/main.rs').read_bytes()).hexdigest(),renderer_binary_sha256=hashlib.sha256(renderer.read_bytes()).hexdigest(),movie_sha256=hashlib.sha256(joined.read_bytes()).hexdigest(),validation='all per-beam and concatenated movies fully decoded; frame counts/dimensions/duration checked')
print('COMPLETE',joined,info,flush=True)
