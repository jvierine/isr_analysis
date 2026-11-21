"""Audit supplied L2 calibration candidates without replacing physical validation.
Original values/quality retained. No density upper cutoff or uncertainty claim.
"""
import argparse
from pathlib import Path
from datetime import datetime,timezone
import h5py,numpy as np
p=argparse.ArgumentParser();p.add_argument('source');p.add_argument('output');a=p.parse_args()
out=Path(a.output);out.parent.mkdir(parents=True,exist_ok=True)
with h5py.File(a.source) as h,h5py.File(out,'w') as o:
    print('ROOT',[(k,v.shape,str(v.dtype)) for k,v in h.items() if isinstance(v,h5py.Dataset)],flush=True)
    o.attrs['source']=a.source;o.attrs['clock']='L2 text parsed as UTC provisionally; metadata and raw text retained for checking'
    for k,v in h.attrs.items():print('ATTR',k,str(v)[:500],flush=True)
    for k in ['Time','Beamdir','GEO','Fittedparameter','Fitstate','Error']:
        d=h[k];x=d[()];o[k]=x
        for key,val in d.attrs.items():o[k].attrs[key]=val;print('DATASET_ATTR',k,key,str(val)[:500],flush=True)
    times=h['Time'][()];ts=np.full(times.shape,np.nan)
    for ix in np.ndindex(times.shape):
        raw=times[ix].decode() if isinstance(times[ix],bytes) else str(times[ix])
        try:ts[ix]=datetime.strptime(raw.strip(),'%d-%b-%Y %H:%M:%S').replace(tzinfo=timezone.utc).timestamp()
        except ValueError:pass
    o['time_unix_s_provisional_UTC']=ts
    ne=h['Fittedparameter'][()][...,0];err=h['Error'][()][...,0];state=h['Fitstate'][()];z=h['GEO'][()][...,2]
    basic=(state==0)&np.isfinite(ne)&(ne>0)&np.isfinite(err)&(err>=0)&(err/ne<=.5)&(z>=150)&(z<=650)
    o['candidate_quality_mask']=basic
    dirs=h['Beamdir'][()]
    for b in range(len(dirs)):
        print('BEAM',b+1,'DIRECTION',dirs[b], 'TIME',times[b,0],times[b,-1], 'PROFILE_COUNT',np.sum(np.any(basic[b],axis=1)),flush=True)
        sel=basic[b] & (ts[b,:,None]>=1772362800)&(ts[b,:,None]<1772373600)
        vals=ne[b][sel]
        print('EVENT_CANDIDATES',len(vals),'Ne_percentiles',np.percentile(vals,[0,5,50,95,99,100]) if len(vals) else [],'GT1e13',np.sum(vals>1e13),flush=True)
    o.attrs['candidate_rule']='Fitstate=0, positive finite density, reported relative error<=0.5, height150..650km; this is numerical screening only, not validated calibration'
print('AUDIT_COMPLETE',flush=True)
