"""Validate every SYISR LP header and compute actual-beam IGRF geometry.

Format reference: the MATLAB/Python readers supplied with the raw recording.
Site and target plan: committed Memo 50 geometry. The recording clock is
Beijing time, confirmed by the data owner on 16 November 2026.
"""
import argparse
from datetime import datetime, timezone
import hashlib
from pathlib import Path

import h5py
import numpy as np
import ppigrf
from scipy.optimize import brentq

C=299792.458
E2=6.6943799901413165e-3


def basis(lat,lon):
    p,l=np.deg2rad([lat,lon])
    return np.array([[-np.sin(l),np.cos(l),0],[-np.sin(p)*np.cos(l),-np.sin(p)*np.sin(l),np.cos(p)],
                     [np.cos(p)*np.cos(l),np.cos(p)*np.sin(l),np.sin(p)]])


def ecef(lat,lon,h):
    p,l=np.deg2rad([lat,lon]);n=6378.137/np.sqrt(1-E2*np.sin(p)**2)
    return np.array([(n+h)*np.cos(p)*np.cos(l),(n+h)*np.cos(p)*np.sin(l),(n*(1-E2)+h)*np.sin(p)])


def geod(x):
    r=np.hypot(x[0],x[1]);p=np.arctan2(x[2],r*(1-E2))
    for _ in range(15):
        n=6378.137/np.sqrt(1-E2*np.sin(p)**2);p=np.arctan2(x[2]+E2*n*np.sin(p),r)
    return np.rad2deg(p),np.rad2deg(np.arctan2(x[1],x[0])),r/np.cos(p)-n


def field(x,date):
    lat,lon,h=geod(x)
    enu=np.asarray(ppigrf.igrf(lon,lat,h,date)).reshape(3)
    return enu@basis(lat,lon)


def main():
    a=argparse.ArgumentParser()
    a.add_argument('raw',type=Path);a.add_argument('plan',type=Path);a.add_argument('output',type=Path)
    q=a.parse_args()
    first=q.raw.open('rb').read(1024)
    n_sample=int.from_bytes(first[768:772],'little')
    stride=1024+(n_sample+1)*8
    assert q.raw.stat().st_size%stride==0
    n=q.raw.stat().st_size//stride
    mm=np.memmap(q.raw,mode='r',dtype=np.uint8)
    def v(offset,dtype): return np.ndarray((n,),dtype=dtype,buffer=mm,offset=offset,strides=(stride,))
    assert np.all(v(0,'<u4')==0xaaaa5555) and np.all(v(1020,'<u4')==0xaa5555aa)
    for offset,dtype,value in [(36,'<u2',0x2041),(764,'u1',1),(768,'<u4',n_sample),
                               (766,'<u2',533),(980,'<u2',10000),(988,'<u2',480)]:
        assert np.all(v(offset,dtype)==value),(offset,np.unique(v(offset,dtype)))
    assert np.all(v(856,'<u2')==v(868,'<u2')) and np.all(v(858,'<u2')==v(870,'<u2'))
    days=np.stack([v(13,'u1'),v(12,'u1'),v(15,'u1')],axis=1)
    dates=np.unique(days,axis=0)
    t=np.zeros(n)
    for year,month,day in dates:
        selected=np.all(days==[year,month,day],axis=1)
        midnight=datetime(2000+int(year),int(month),int(day),tzinfo=timezone.utc).timestamp()
        t[selected]=midnight
    t+=v(14,'u1').astype(float)*3600+v(17,'u1').astype(float)*60+v(16,'u1').astype(float)+v(18,'<u2')*25e-6
    t-=8*3600 # confirmed Beijing header clock -> UTC
    dt=np.diff(t)
    assert np.all((dt>=0.00995)&(dt<=0.01005)),np.quantile(dt,[0,.5,1])
    angle=np.stack([v(856,'<u2'),v(858,'<u2')],axis=1)
    beams=np.unique(angle,axis=0)
    assert len(beams)==5
    beams=beams[np.argsort(-beams[:,1].astype(float))]
    with h5py.File(q.plan) as f:
        planned=f['pointings'][:];site_lat,site_lon=f.attrs['site_lat_lon_deg']
    assert list(planned[:,0])==[250,350,450,550,650]
    site=ecef(site_lat,site_lon,0.);enu=basis(site_lat,site_lon)
    date=datetime.fromtimestamp(t[0],timezone.utc).replace(tzinfo=None)
    beam_index=np.zeros(n,dtype=np.uint8)
    rows=[]
    for i,(raw_az,raw_el) in enumerate(beams):
        beam_index[np.all(angle==[raw_az,raw_el],axis=1)]=i
        az,el=np.asarray([raw_az,raw_el],float)*360/65536
        assert abs(el-planned[i,2])<0.1 and abs(az-planned[i,1])<0.3
        aa,ee=np.deg2rad([az,el]);u=np.array([np.sin(aa)*np.cos(ee),np.cos(aa)*np.cos(ee),np.sin(ee)])@enu
        def dot(r):
            b=field(site+r*u,date);return np.dot(u,b)/np.linalg.norm(b)
        r=brentq(dot,max(0,planned[i,3]-200),planned[i,3]+200,xtol=1e-8)
        point=site+r*u;b=field(point,date)
        row=[planned[i,0],az,el,r,geod(point)[2],np.linalg.norm(b)/1000,np.rad2deg(np.arcsin(dot(r)))]
        rows.append(row)
        print('beam',i,row,'pulses',np.count_nonzero(beam_index[:n] == i) if i==4 else np.count_nonzero(np.all(angle==[raw_az,raw_el],axis=1)),flush=True)
    sha=hashlib.sha256()
    with q.raw.open('rb') as f:
        while block:=f.read(8*1024*1024):sha.update(block)
    q.output.parent.mkdir(parents=True,exist_ok=True)
    with h5py.File(q.output,'w') as h:
        h['time_utc_s']=t;h['beam_index']=beam_index;h['beam_raw_angles']=beams
        h['beam_geometry']=rows;h['planned_geometry']=planned
        h.attrs['geometry_columns']='planned_altitude_km,recorded_az_deg,recorded_el_deg,perp_slant_range_km,perp_geodetic_altitude_km,B_uT,signed_aspect_deg'
        h.attrs['raw_path']=str(q.raw.resolve());h.attrs['raw_sha256']=sha.hexdigest()
        h.attrs['raw_bytes']=q.raw.stat().st_size;h.attrs['record_stride_bytes']=stride
        h.attrs['sample_count']=n_sample;h.attrs['sample_rate_hz']=100000.
        h.attrs['gate_front_us']=533.;h.attrs['pulse_width_us']=480.;h.attrs['ipp_us']=10000.
        h.attrs['header_clock_offset_to_utc_s']=-28800.
        h.attrs['clock_basis']='Beijing time UTC+8, confirmed by data owner'
        h.attrs['site_lat_lon_deg']=[site_lat,site_lon]
        h.attrs['geometry_model']='IGRF-14, WGS84 straight ray; provisional Memo 50 site coordinates; actual recorded angles'
        h.attrs['geometry_epoch_utc']=date.isoformat()+'Z'
        h.attrs['plan_path']=str(q.plan.resolve())
        h.attrs['index_script_sha256']=hashlib.sha256(Path(__file__).read_bytes()).hexdigest()
    print('UTC',datetime.fromtimestamp(t[0],timezone.utc),datetime.fromtimestamp(t[-1],timezone.utc),'records',n,'span_s',t[-1]-t[0],flush=True)


if __name__=='__main__':main()
