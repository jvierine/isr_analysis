"""Attach correction provenance without relabelling the original IQ processing."""
import argparse
import hashlib
from pathlib import Path
import subprocess
import h5py


def sha(path):return hashlib.sha256(path.read_bytes()).hexdigest()


def main():
    p=argparse.ArgumentParser();p.add_argument('audit',type=Path);p.add_argument('candidate_directory',type=Path);q=p.parse_args()
    tool=Path(__file__).resolve().parent
    commit=subprocess.check_output(['git','rev-parse','HEAD'],cwd=tool,text=True).strip()
    assert not subprocess.check_output(['git','status','--porcelain'],cwd=tool,text=True).strip()
    with h5py.File(q.audit,'r+') as h:
        h.attrs['audit_source_git_commit']=commit
        for name in ['audit_gating.py','audit_pulse_edge.py','src/bin/audit-geometry.rs','src/bin/plot-gating-audit.rs','src/bin/plot-pulse-edge.rs']:
            h.attrs['source_sha256_'+name.replace('/','_')]=sha(tool/name)
    data=q.candidate_directory/'range-doppler.h5'
    with h5py.File(data,'r+') as h:
        assert h.attrs['range_delay_correction_us']==160.
        h.attrs['range_correction_source_git_commit']=commit
        h.attrs['timing_audit_sha256']=sha(q.audit)
        status=h.attrs['range_delay_status']
        original_commit=h.attrs['source_git_commit']
    with h5py.File(q.candidate_directory/'movies.h5','r+') as h:
        h.attrs['source_sha256']=sha(data)
        h.attrs['original_processing_source_git_commit']=original_commit
        h.attrs['renderer_source_git_commit']=commit
        h.attrs['range_delay_status']=status
        for key in h:
            if key.startswith('beam_'):
                assert sha(Path(h[key].attrs['movie_path']))==h[key].attrs['sha256']
    print('Correction and rendering provenance attached; original processing commit preserved:',original_commit)


if __name__=='__main__':main()
