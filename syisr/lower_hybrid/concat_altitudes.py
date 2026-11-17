"""Full altitude-by-altitude movie, retaining every frame and UTC label."""
import argparse
import hashlib
import json
from pathlib import Path
import subprocess
import tempfile
import h5py
import numpy as np


def main():
 p=argparse.ArgumentParser();p.add_argument('root',type=Path);q=p.parse_args();altitudes=[250,350,450,550,650]
 paths=[q.root/f'syisr_20261116_{a}km_5min.mp4' for a in altitudes]
 infos=[json.loads(subprocess.check_output(['ffprobe','-v','error','-select_streams','v:0','-show_entries','stream=nb_frames,duration','-of','json',str(path)],text=True))['streams'][0] for path in paths]
 durations=np.asarray([float(i['duration']) for i in infos]);frames=sum(int(i['nb_frames']) for i in infos);duration=float(durations.sum())
 output=q.root/'syisr_20261116_all_altitudes_5min.mp4'
 with tempfile.NamedTemporaryFile(mode='w',suffix='.txt') as listing:
  for path in paths:listing.write("file '"+str(path.resolve()).replace("'","'\\''")+"'\n")
  listing.flush();subprocess.run(['ffmpeg','-y','-v','error','-f','concat','-safe','0','-i',listing.name,'-c','copy','-movflags','+faststart',str(output)],check=True)
 info=json.loads(subprocess.check_output(['ffprobe','-v','error','-select_streams','v:0','-show_entries','stream=width,height,nb_frames,duration','-of','json',str(output)],text=True))['streams'][0]
 assert info['width']==1920 and info['height']==1080 and int(info['nb_frames'])==frames and abs(float(info['duration'])-duration)<.05
 subprocess.run(['ffmpeg','-v','error','-i',str(output),'-f','null','-'],check=True)
 with h5py.File(q.root/'movie-all-altitudes.h5','w') as h:
  h['nominal_target_altitudes_km']=altitudes;h['segment_start_s']=np.r_[0.,np.cumsum(durations)[:-1]];h['segment_end_s']=np.cumsum(durations)
  h['source_movies']=np.asarray([str(p.resolve()) for p in paths],dtype=h5py.string_dtype())
  h['source_sha256']=np.asarray([hashlib.sha256(p.read_bytes()).hexdigest() for p in paths],dtype=h5py.string_dtype())
  h.attrs['movie_path']=str(output.resolve());h.attrs['sha256']=hashlib.sha256(output.read_bytes()).hexdigest()
  h.attrs['encoded_frames']=frames;h.attrs['duration_s']=duration;h.attrs['concatenation']='All scientific frames; ascending nominal target altitude; original UTC labels retained; no interpolation across acquisition gaps'
 print('Full altitude concatenation verified:',output,duration,'seconds',flush=True)

if __name__=='__main__':main()
