"""Reindex verified cache bytes on fast storage; retain original source paths."""
import argparse
import hashlib
from pathlib import Path
import subprocess
import sys
import h5py


def main():
    p=argparse.ArgumentParser();p.add_argument('cache',type=Path);p.add_argument('original',type=Path);p.add_argument('plan',type=Path);p.add_argument('prefix');p.add_argument('output',type=Path);q=p.parse_args()
    tool=Path(__file__).resolve().parent
    subprocess.run([sys.executable,str(tool/'index_recording.py'),'--session',q.prefix,str(q.cache),str(q.plan),str(q.output)],check=True)
    with h5py.File(q.output,'r+') as h:
        cached=h['source_paths'].asstr()[:]
        h['source_paths'][:]=[str(q.original.resolve()/Path(path).name) for path in cached]
        h.attrs['raw_path']=str(q.original.resolve()/Path(cached[0]).name)
        h.attrs['cache_reindex_source_sha256']=hashlib.sha256(Path(__file__).read_bytes()).hexdigest()
        h.attrs['cache_reindex_status']='cache byte identities must agree with the original sequential-copy index before final publication'


if __name__=='__main__':main()
