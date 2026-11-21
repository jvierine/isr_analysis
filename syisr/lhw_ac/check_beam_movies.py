from pathlib import Path
import sys,hashlib,os,json
import h5py,numpy as np
r=Path(sys.argv[1]);records=[]
for name,mult in [(str(a),1) for a in [150,200,250,300,350]]+[('all',5)]:
    p=dict(line.split('=',1) for line in (r/f'decode-{name}.txt').read_text().splitlines() if '=' in line)
    assert p['progress']=='end' and int(p['frame'])==615*mult
    assert abs(float(p['out_time_us'])/1e6-25.625*mult)<.05
    file=r/(f'syisr-ac-{name}km-corrected.mp4' if name!='all' else 'syisr-ac-all-beams-corrected.mp4')
    records.append(dict(beam=name,frames=int(p['frame']),duration_s=float(p['out_time_us'])/1e6,filename=file.name,sha256=hashlib.sha256(file.read_bytes()).hexdigest()))
with h5py.File(r/'movies-verification.h5','w') as h:
    h['records_json_utf8']=np.frombuffer(json.dumps(records).encode(),'u1')
    h.attrs.update(result='PASS all five full decodes plus concatenation',git_commit=os.environ['SOURCE_GIT_COMMIT'])
print('PASS individual beam movies',records,flush=True)
