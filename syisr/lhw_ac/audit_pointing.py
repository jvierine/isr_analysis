"""Independent geocentric IGRF check and raw TX/RX pointing replay."""
import argparse, importlib.util, json
from pathlib import Path
from datetime import datetime
import h5py
import numpy as np
import ppigrf
from scipy.optimize import brentq

p=argparse.ArgumentParser();p.add_argument('index');p.add_argument('output');q=p.parse_args()
sp=importlib.util.spec_from_file_location('geo',Path(__file__).parent.parent/'lower_hybrid/index_recording.py')
g=importlib.util.module_from_spec(sp);sp.loader.exec_module(g)
date=datetime(2026,11,19,12)
site=g.ecef(18.34,109.62,0);basis=g.basis(18.34,109.62)
def field_gc(x):
    r=np.linalg.norm(x);theta=np.arccos(x[2]/r);phi=np.arctan2(x[1],x[0])
    br,bt,bp=np.asarray(ppigrf.igrf_gc(r,np.rad2deg(theta),np.rad2deg(phi),date)).reshape(3)
    er=np.array([np.sin(theta)*np.cos(phi),np.sin(theta)*np.sin(phi),np.cos(theta)])
    et=np.array([np.cos(theta)*np.cos(phi),np.cos(theta)*np.sin(phi),-np.sin(theta)])
    ep=np.array([-np.sin(phi),np.cos(phi),0])
    return br*er+bt*et+bp*ep
rows=[];headers=[]
with h5py.File(q.index) as h:
    angles=h['beam_az_el_deg'][:];bi=h['beam_index'][:];off=h['frame_offset'][:];fi=h['file_index'][:]
    paths=json.loads(bytes(h['files_json_utf8'][:]))
    for b,(az,el) in enumerate(angles):
        selected=np.flatnonzero(bi==b)
        for ci in selected[np.linspace(0,len(selected)-1,9).astype(int)]:
            for c in [0,31,63]:
                with open(paths[int(fi[ci,c])],'rb') as f:f.seek(int(off[ci,c]));raw=f.read(1024)
                pair=np.array([int.from_bytes(raw[k:k+2],'little')*360/65536 for k in [856,858,868,870]])
                np.testing.assert_allclose(pair,[az,el,az,el],atol=1e-10)
                headers.append(np.frombuffer(raw,'u1'))
        a,e=np.deg2rad([az,el]);u=np.array([np.sin(a)*np.cos(e),np.cos(a)*np.cos(e),np.sin(e)])@basis
        r=brentq(lambda r:np.dot(u,field_gc(site+r*u)),0,1000)
        x=site+r*u;bg=field_gc(x);bo=g.field(x,date)
        np.testing.assert_allclose(bg,bo,rtol=0,atol=1e-3)
        observed={0:245.,1:275.,4:425.}.get(b,np.nan)
        aspect=np.nan;required=np.nan;delay=np.nan
        if np.isfinite(observed):
            ro=brentq(lambda r:g.geod(site+r*u)[2]-observed,0,1000)
            delay=(ro-r)/.149896229
            bf=field_gc(site+ro*u);aspect=np.rad2deg(np.arcsin(np.dot(u,bf)/np.linalg.norm(bf)))
            def dot_at_height(elevation):
                e=np.deg2rad(elevation);v=np.array([np.sin(a)*np.cos(e),np.cos(a)*np.cos(e),np.sin(e)])@basis
                rr=brentq(lambda r:g.geod(site+r*v)[2]-observed,0,1000)
                return np.dot(v,field_gc(site+rr*v))
            required=brentq(dot_at_height,50,70)
        row=[150+50*b,az,el,r,g.geod(x)[2],observed,aspect,required,required-el,delay]
        rows.append(row);print(row,flush=True)
with h5py.File(q.output,'w') as h:
    h['pointing_checks']=rows;h['raw_headers']=np.array(headers)
    h.attrs['columns']='nominal_label_km,header_az_deg,header_el_deg,independent_perp_slant_km,independent_perp_altitude_km,user_feature_altitude_km,aspect_at_feature_deg,elevation_required_at_feature_deg,required_minus_header_el_deg,apparent_excess_two_way_delay_us'
    h.attrs['checks']='135 raw headers across 9 epochs per beam; TX and RX angles agree with index. Independent geocentric spherical IGRF14 synthesis agrees with deployed geodetic ENU synthesis within 1e-3 nT.'
    h.attrs['site_lat_lon_alt_km']=[18.34,109.62,0.]
    h.attrs['field_model']=str(ppigrf.igrf_gc.__defaults__)
