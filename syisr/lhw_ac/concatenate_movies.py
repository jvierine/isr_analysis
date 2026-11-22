"""Join verified, identically encoded movies without re-encoding; HDF5 manifest."""
import argparse,hashlib,json,os,subprocess,tempfile
from pathlib import Path
import h5py,numpy as np

p=argparse.ArgumentParser();p.add_argument('output',type=Path);p.add_argument('inputs',nargs='+',type=Path);q=p.parse_args()
assert not q.output.exists(),'output already exists'
def probe(path):
    v=json.loads(subprocess.check_output(['ffprobe','-v','error','-select_streams','v:0','-show_streams','-of','json',str(path)]))['streams'][0]
    return v
def digest(path):
    h=hashlib.sha256()
    with path.open('rb') as f:
        for chunk in iter(lambda:f.read(1048576),b''):h.update(chunk)
    return h.hexdigest()
streams=[probe(path) for path in q.inputs]
for s in streams[1:]:
    for key in ['codec_name','width','height','pix_fmt','r_frame_rate','time_base']:assert s[key]==streams[0][key],key
duration=np.array([float(s['duration']) for s in streams]);frames=np.array([int(s['nb_frames']) for s in streams])
partial=q.output.with_name(q.output.stem+'.partial.mp4')
with tempfile.NamedTemporaryFile(mode='w',suffix='.txt',dir=q.output.parent) as listing:
    for path in q.inputs:
        assert "'" not in str(path.resolve())
        listing.write("file '"+str(path.resolve())+"'\n")
    listing.flush()
    subprocess.run(['ffmpeg','-v','error','-y','-f','concat','-safe','0','-i',listing.name,'-c','copy','-movflags','+faststart',str(partial)],check=True)
actual=probe(partial);assert int(actual['nb_frames'])==frames.sum();assert abs(float(actual['duration'])-duration.sum())<.05
subprocess.run(['ffmpeg','-v','error','-i',str(partial),'-f','null','-'],check=True)
partial.rename(q.output)
with h5py.File(q.output.with_suffix('.h5'),'w') as h:
    h['input_paths']=np.array([str(path.resolve()) for path in q.inputs],dtype=h5py.string_dtype())
    h['input_sha256']=np.array([digest(path) for path in q.inputs],dtype=h5py.string_dtype())
    h['start_s']=np.r_[0,np.cumsum(duration)[:-1]];h['duration_s']=duration;h['frames']=frames
    h.attrs.update(result='PASS codec agreement, exact concatenated frame count, duration and full decode',program='syisr-lhw-movie-concat',version='0.1.0',git_commit=os.environ.get('SOURCE_GIT_COMMIT','working'),output_sha256=digest(q.output),output_duration_s=float(actual['duration']),output_frames=int(actual['nb_frames']))
print('PASS concatenated movie',q.output,float(actual['duration']),int(actual['nb_frames']),flush=True)
