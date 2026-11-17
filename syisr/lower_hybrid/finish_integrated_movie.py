"""Publish session products and join their single-beam movies in UTC order."""
import argparse
import hashlib
import json
from pathlib import Path
import subprocess
import sys
import tempfile
import h5py
import numpy as np


def sha(path):return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def main():
    p=argparse.ArgumentParser();p.add_argument('scratch',type=Path);p.add_argument('destination',type=Path);p.add_argument('--altitude-km',type=int,default=650);p.add_argument('--name',default='syisr_20261116_650km_30min_step5min.mp4');q=p.parse_args()
    subprocess.run([sys.executable,str(Path(__file__).with_name('publish_batch.py')),str(q.scratch),str(q.destination)],check=True)
    sessions=[]
    for manifest in q.destination.glob('session_*/movies.h5'):
        with h5py.File(manifest,'r+') as m:
            product=Path(m.attrs['source_product']);display=json.loads(m.attrs['display_scales_json'])
            with h5py.File(product,'r+') as h:
                chain=json.loads(h.attrs['processing_chain_json']);chain=[c for c in chain if c['program']!='syisr-rd-render']
                chain.append(dict(program='syisr-rd-render',version=display['render_version'],git_commit=m.attrs['renderer_git_commit']))
                h.attrs['processing_chain_json']=json.dumps(chain)
                h.attrs['renderer_version']=display['render_version'];h.attrs['renderer_git_commit']=m.attrs['renderer_git_commit']
                h.attrs['renderer_source_sha256']=m.attrs['renderer_source_sha256'];h.attrs['final_renderer_binary_sha256']=m.attrs['renderer_binary_sha256']
                h.attrs['display_normalization']=display['normalization'];h.attrs['display_colormap']=display['color_map']
                h.attrs['video_encoding_mode']=m.attrs['encoding_mode']
                assert float(h.attrs['integration_window_s'])==1800. and float(h.attrs['integration_step_s'])==300.
                assert 'independent_validation_json' in h.attrs
            m.attrs['source_sha256']=sha(product)
            g=m[f'beam_{q.altitude_km}km'];movie=Path(g.attrs['movie_path']);assert sha(movie)==g.attrs['sha256']
            sessions.append((float(m['window_start_utc_s'][0]),manifest,movie,int(g.attrs['nb_frames']),float(g.attrs['duration']),m['window_start_utc_s'][:],m['window_end_utc_s'][:]))
    sessions.sort();assert len(sessions)==2
    movie=q.destination/q.name
    with tempfile.NamedTemporaryFile(mode='w',suffix='.txt') as listing:
        for _,_,path,*_ in sessions:listing.write("file '"+str(path).replace("'","'\\''")+"'\n")
        listing.flush();subprocess.run(['ffmpeg','-y','-v','error','-f','concat','-safe','0','-i',listing.name,'-c','copy','-movflags','+faststart',str(movie)],check=True)
    info=json.loads(subprocess.check_output(['ffprobe','-v','error','-select_streams','v:0','-show_entries','stream=width,height,nb_frames,duration','-of','json',str(movie)],text=True))['streams'][0]
    assert info['width']==1920 and info['height']==1080 and int(info['nb_frames'])==sum(s[3] for s in sessions)
    assert abs(float(info['duration'])-sum(s[4] for s in sessions))<.05
    subprocess.run(['ffmpeg','-v','error','-i',str(movie),'-f','null','-'],check=True)
    with h5py.File(q.destination/'movie.h5','w') as h:
        h.attrs['movie_path']=str(movie.resolve());h.attrs['sha256']=sha(movie);h.attrs['duration_s']=float(info['duration']);h.attrs['encoded_frames']=int(info['nb_frames'])
        h.attrs['nominal_target_altitude_km']=q.altitude_km;h.attrs['window_s']=1800.;h.attrs['step_s']=300.;h.attrs['scientific_frames_per_second']=8.
        h.attrs['gap_handling']='Separate sessions joined chronologically; windows never cross restart; no invented frames; UTC labels retained'
        h['source_manifests']=np.asarray([str(s[1]) for s in sessions],dtype=h5py.string_dtype());h['source_manifest_sha256']=np.asarray([sha(s[1]) for s in sessions],dtype=h5py.string_dtype())
        h['window_start_utc_s']=np.concatenate([s[5] for s in sessions]);h['window_end_utc_s']=np.concatenate([s[6] for s in sessions])
    for label,t in [('first',0.),('middle',float(info['duration'])/2),('last',float(info['duration'])-.125)]:
        subprocess.run(['ffmpeg','-y','-v','error','-ss',str(t),'-i',str(movie),'-frames:v','1',str(q.destination/f'{label}.png')],check=True)
    print('Published and fully decoded:',movie,info['duration'],'seconds',flush=True)


if __name__=='__main__':main()
