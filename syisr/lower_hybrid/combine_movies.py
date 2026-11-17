"""Join verified sessions chronologically without fabricating gap frames."""
import argparse
import hashlib
import json
from pathlib import Path
import subprocess
import tempfile
import h5py
import numpy as np


def sha(path):return hashlib.sha256(path.read_bytes()).hexdigest()


def main():
    p=argparse.ArgumentParser();p.add_argument('root',type=Path);q=p.parse_args()
    sessions=[]
    for path in sorted(q.root.glob('session_*/provisional_delay160us/movies.h5')):
        with h5py.File(path) as h:
            if float(h.attrs['window_s'])<299.:continue
            sessions.append((float(h['window_start_utc_s'][0]),path))
    sessions.sort();assert len(sessions)>=2
    starts=[];ends=[];manifest=q.root/'movies-all.h5';qa=q.root/'qa';qa.mkdir(exist_ok=True)
    with h5py.File(manifest,'w') as out:
        for _,path in sessions:
            with h5py.File(path) as h:starts.extend(h['window_start_utc_s'][:]);ends.extend(h['window_end_utc_s'][:])
        out['window_start_utc_s']=starts;out['window_end_utc_s']=ends
        out['session_manifests']=np.asarray([str(p.resolve()) for _,p in sessions],dtype=h5py.string_dtype())
        out['session_manifest_sha256']=np.asarray([sha(p) for _,p in sessions],dtype=h5py.string_dtype())
        out.attrs['gap_handling']='sessions concatenated in UTC order; actual dates/times displayed on every frame; no interpolation or invented measurements across acquisition gaps; short test omitted'
        out.attrs['window_s']=300.;out.attrs['nominal_step_s']=150.;out.attrs['range_delay_correction_us']=160.
        out.attrs['source_git_commit']=subprocess.check_output(['git','rev-parse','HEAD'],cwd=Path(__file__).parent,text=True).strip()
        for altitude in [250,350,450,550,650]:
            paths=[];frames=0;duration=0.
            for _,path in sessions:
                with h5py.File(path) as h:
                    g=h[f'beam_{altitude}km'];paths.append(Path(g.attrs['movie_path']));frames+=int(g.attrs['nb_frames']);duration+=float(g.attrs['duration'])
            movie=q.root/f'syisr_20261116_{altitude}km_5min.mp4'
            with tempfile.NamedTemporaryFile(mode='w',suffix='.txt') as listing:
                for path in paths:listing.write("file '"+str(path).replace("'","'\\''")+"'\n")
                listing.flush()
                subprocess.run(['ffmpeg','-y','-v','error','-f','concat','-safe','0','-i',listing.name,'-c','copy','-movflags','+faststart',str(movie)],check=True)
            info=json.loads(subprocess.check_output(['ffprobe','-v','error','-select_streams','v:0','-show_entries','stream=width,height,nb_frames,duration','-of','json',str(movie)],text=True))['streams'][0]
            assert info['width']==1920 and info['height']==1080 and int(info['nb_frames'])==frames
            assert abs(float(info['duration'])-duration)<.06
            subprocess.run(['ffmpeg','-v','error','-i',str(movie),'-f','null','-'],check=True)
            for label,second in [('first',0.),('middle',duration/2),('last',duration-.25)]:
                subprocess.run(['ffmpeg','-y','-v','error','-ss',str(second),'-i',str(movie),'-frames:v','1',str(qa/f'{altitude}km-{label}.png')],check=True)
            g=out.create_group(f'beam_{altitude}km');g.attrs['movie_path']=str(movie.resolve());g.attrs['sha256']=sha(movie)
            g.attrs['duration_s']=duration;g.attrs['encoded_frames']=frames
            print('Combined and decoded:',movie.name,duration,'seconds',flush=True)


if __name__=='__main__':main()
