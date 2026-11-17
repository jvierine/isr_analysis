"""Pin processing-chain identities and verify all delivered movie hashes."""
import argparse
import hashlib
import json
from pathlib import Path
import platform
import subprocess
import tomllib
import h5py


def sha(path):
    digest=hashlib.sha256()
    with path.open('rb') as f:
        while block:=f.read(8*1024*1024):digest.update(block)
    return digest.hexdigest()


def main():
    p=argparse.ArgumentParser();p.add_argument('root',type=Path);q=p.parse_args();tool=Path(__file__).resolve().parent
    commit=subprocess.check_output(['git','rev-parse','HEAD'],cwd=tool,text=True).strip()
    assert not subprocess.check_output(['git','status','--porcelain'],cwd=tool,text=True).strip()
    version=tomllib.loads((tool/'Cargo.toml').read_text())['package']['version']
    names=['index_recording.py','src/lib.rs','src/bin/process.rs','src/bin/render.rs','subtract_background.py','verify.py','make_movies.py','audit_session_edges.py','prepare_display.py','screen_review.py','src/bin/plot-screen-review.rs','finalize_batch.py','reindex_cache.py','verify_cache_indices.py','combine_movies.py','Cargo.lock']
    names.append('publish_batch.py')
    hashes={name:sha(tool/name) for name in names}
    chain=[dict(program=name,version=version,git_commit=commit) for name in ['syisr-rd','lp-transient-screen','syisr-rd-render']]
    with h5py.File(q.root/'pulse-edge-audit.h5','r+') as a:a.attrs['audit_source_git_commit']=commit
    for session in sorted(q.root.glob('session_*')):
        if not session.is_dir():continue
        with h5py.File(session/'recording-index.h5') as ix:
            assert ix.attrs['index_script_sha256']==hashes['index_recording.py']
            assert ix.attrs.get('cache_reindex_status','').startswith('verified:')
        for data in [session/'range-doppler.h5',session/'provisional_delay160us/range-doppler.h5']:
            with h5py.File(data,'r+') as h:
                assert 'independent_validation_json' in h.attrs
                h.attrs['program_name']='syisr-rd';h.attrs['program_version']=version
                h.attrs['source_git_commit']=commit;h.attrs['source_git_url']=subprocess.check_output(['git','remote','get-url','origin'],cwd=tool,text=True).strip()
                h.attrs['processing_chain_json']=json.dumps(chain);h.attrs['source_sha256_json']=json.dumps(hashes)
                h.attrs['compute_host']=platform.node();h.attrs['python_version']=platform.python_version()
                h.attrs['timing_audit_sha256']=sha(q.root/'pulse-edge-audit.h5')
                h.attrs['process_binary_sha256']=sha(tool/'target/release/process')
                h.attrs['renderer_binary_sha256']=sha(tool/'target/release/render')
                h.attrs['screen_limitations']='Morphology screen can miss weak/long-lived satellites or reject physical transients resembling point targets; not a calibrated detection probability; unscreened PSD and time/range exposure retained'
        manifest=session/'provisional_delay160us/movies.h5'
        with h5py.File(manifest,'r+') as m:
            m.attrs['source_git_commit']=commit;m.attrs['source_sha256']=sha(session/'provisional_delay160us/range-doppler.h5')
            for g in m.values():
                if isinstance(g,h5py.Group) and 'movie_path' in g.attrs:assert sha(Path(g.attrs['movie_path']))==g.attrs['sha256']
        print('Pinned and verified',session.name,commit,flush=True)


if __name__=='__main__':main()
