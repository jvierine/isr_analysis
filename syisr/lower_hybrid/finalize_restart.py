"""Verify published registered products and refresh relocated manifest hashes."""
import argparse
import hashlib
import json
import subprocess
from pathlib import Path
import h5py

def sha(path):
 digest=hashlib.sha256()
 with Path(path).open('rb') as f:
  while block:=f.read(8*1024*1024):digest.update(block)
 return digest.hexdigest()

def main():
 p=argparse.ArgumentParser();p.add_argument('root',type=Path);q=p.parse_args()
 for session in ['190252','205701']:
  path=q.root/f'session_{session}/restart_registered/range-doppler.h5'
  with h5py.File(path.parent/'movies.h5') as m:
   display=json.loads(m.attrs['display_scales_json']);encoding=m.attrs.get('encoding_mode','legacy PNG sequence')
  tool=Path(__file__).resolve().parent
  render_commit=subprocess.check_output(['git','log','-1','--format=%H','--',':(top)syisr/lower_hybrid/src/bin/render.rs'],cwd=tool,text=True).strip()
  with h5py.File(path,'r+') as h:
   chain=json.loads(h.attrs.get('processing_chain_json','[]'))
   if not chain:
    for program,source in [('syisr-rd','src/bin/process.rs'),('lp-transient-screen','src/lib.rs'),('syisr-rd-background','subtract_background.py'),('syisr-rd-render','src/bin/render.rs')]:
     commit=subprocess.check_output(['git','log','-1','--format=%H','--',':(top)syisr/lower_hybrid/'+source],cwd=tool,text=True).strip()
     chain.append(dict(program=program,version='0.2.0',git_commit=commit))
   for component in chain:
    if component['program']=='syisr-rd-render':component.update(version=display['render_version'],git_commit=render_commit)
   if not any(c['program']=='syisr-restart-register' for c in chain):
    chain.append(dict(program=h.attrs['registration_program'],version=h.attrs['registration_version'],git_commit=h.attrs['registration_git_commit']))
   h.attrs['processing_chain_json']=json.dumps(chain)
   h.attrs['renderer_version']=display['render_version'];h.attrs['renderer_git_commit']=render_commit
   h.attrs['final_renderer_binary_sha256']=sha(tool/'target/release/render')
   h.attrs['renderer_source_sha256']=sha(tool/'src/bin/render.rs')
   h.attrs['display_colormap']=display['color_map'];h.attrs['display_normalization']=display['normalization']
   h.attrs['video_encoding_mode']=encoding;h.attrs['scientific_frames_per_playback_second']=display['frame_rate_fps']
  with h5py.File(path.parent/'movies.h5','r+') as m:
   assert Path(m.attrs['source_product'])==path
   m.attrs['source_sha256']=sha(path)
   for g in m.values():
    if isinstance(g,h5py.Group):assert sha(g.attrs['movie_path'])==g.attrs['sha256']
  print('Verified published',session,flush=True)
 with h5py.File(q.root/'movies-all.h5','r+') as h:
  h['session_manifest_sha256'][...]=[sha(p) for p in h['session_manifests'].asstr()[:]]
  for g in h.values():
   if isinstance(g,h5py.Group):assert sha(g.attrs['movie_path'])==g.attrs['sha256']
 with h5py.File(q.root/'movie-all-altitudes.h5') as h:
  assert sha(h.attrs['movie_path'])==h.attrs['sha256']
  for p,digest in zip(h['source_movies'].asstr()[:],h['source_sha256'].asstr()[:]):assert sha(p)==digest
 print('Published movie and scientific-product hashes verified',flush=True)

if __name__=='__main__':main()
