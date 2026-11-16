"""Attach exact source identities to HDF5 after the scientific/video checks."""
import argparse
import hashlib
from importlib.metadata import version
import json
from pathlib import Path
import platform
import subprocess

import h5py


def sha(path):return hashlib.sha256(path.read_bytes()).hexdigest()


def main():
    p=argparse.ArgumentParser();p.add_argument('output',type=Path);q=p.parse_args()
    here=Path(__file__).resolve().parent;root=here.parent.parent
    revision=subprocess.check_output(['git','-C',str(root),'rev-parse','HEAD'],text=True).strip()
    remote=subprocess.check_output(['git','-C',str(root),'remote','get-url','origin'],text=True).strip()
    paths=['src/lib.rs','src/bin/process.rs','src/bin/render.rs','index_recording.py',
           'subtract_background.py','verify.py','make_movies.py','finalize.py','Cargo.lock']
    source_hashes={path:sha(here/path) for path in paths}
    for path in paths:
        committed=subprocess.check_output(['git','-C',str(root),'show',revision+':syisr/lower_hybrid/'+path])
        assert hashlib.sha256(committed).hexdigest()==source_hashes[path],path
    data=q.output/'range-doppler.h5';index=q.output/'recording-index.h5'
    with h5py.File(data,'r+') as h,h5py.File(index) as ix:
        assert 'independent_validation_json' in h.attrs and 'background_audit_json' in h.attrs
        assert ix.attrs['index_script_sha256']==source_hashes['index_recording.py']
        h.attrs['program_name']='SYISR lower-hybrid range-Doppler'
        h.attrs['program_version']='0.1.0'
        h.attrs['source_git_url']=remote;h.attrs['source_git_commit']=revision
        h.attrs['source_sha256_json']=json.dumps(source_hashes)
        h.attrs['reader_sha256']=sha(root/'syisr/src/reader.rs')
        h.attrs['process_binary_sha256']=sha(here/'target/release/process')
        h.attrs['renderer_binary_sha256']=sha(here/'target/release/render')
        h.attrs['rust_compiler']=subprocess.check_output(['rustc','-Vv'],text=True)
        h.attrs['python_version']=platform.python_version()
        h.attrs['python_packages_json']=json.dumps({name:version(name) for name in ['numpy','scipy','h5py','ppigrf','Pillow']})
        h.attrs['background_script_sha256']=source_hashes['subtract_background.py']
        h.attrs['memo050_geometry_sha256']=sha(Path(ix.attrs['plan_path']))
        h.attrs['movie_visual_review']='decoded first/middle/last frames from all five movies reviewed; fixed 1920x1080 frames, legible labels, signed scales and correct beam-dependent marker ranges'
    with h5py.File(q.output/'movies.h5','r+') as m:
        m.attrs['source_git_url']=remote;m.attrs['source_git_commit']=revision
        m.attrs['source_sha256']=sha(data)
        for g in m.values():
            if isinstance(g,h5py.Group) and 'movie_path' in g.attrs:
                assert sha(Path(g.attrs['movie_path']))==g.attrs['sha256']
    print('Final HDF5 provenance and all five movie hashes verified at',revision)


if __name__=='__main__':main()
