"""Verify published registered products and refresh relocated manifest hashes."""
import argparse
import hashlib
import json
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
  with h5py.File(path,'r+') as h:
   chain=json.loads(h.attrs.get('processing_chain_json','[]'))
   if not any(c['program']=='syisr-restart-register' for c in chain):
    chain.append(dict(program=h.attrs['registration_program'],version=h.attrs['registration_version'],git_commit=h.attrs['registration_git_commit']))
   h.attrs['processing_chain_json']=json.dumps(chain)
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
