"""Rank real beam approaches using a historical orbital snapshot; HDF5 only.

SGP4 TEME states rotate with UT1 GMST1982 to terrestrial coordinates.
Polar motion is omitted (<arcsecond scale); checked against Skyfield altaz.
No range or timing correction is fitted by this prediction stage.
"""
import argparse,gzip,json,os,hashlib
from pathlib import Path
from datetime import datetime,timezone
from concurrent.futures import ProcessPoolExecutor
import numpy as np
import h5py
from scipy.optimize import minimize_scalar
from sgp4.api import Satrec
from skyfield.api import load,EarthSatellite,wgs84
from skyfield.sgp4lib import theta_GMST1982

LAT,LON,ALT=18.34,109.62,0.
AZ=357.9949951171875
ELS=np.array([61.58935546875,60.897216796875,60.1776123046875,59.425048828125,58.64501953125])
ts=load.timescale(builtin=True)
station=wgs84.latlon(LAT,LON,elevation_m=ALT*1000)
p,l=np.deg2rad([LAT,LON]);n=6378.137/np.sqrt(1-6.6943799901413165e-3*np.sin(p)**2)
SITE=np.array([(n+ALT)*np.cos(p)*np.cos(l),(n+ALT)*np.cos(p)*np.sin(l),(n*(1-6.6943799901413165e-3)+ALT)*np.sin(p)])
ENU=np.array([[-np.sin(l),np.cos(l),0],[-np.sin(p)*np.cos(l),-np.sin(p)*np.sin(l),np.cos(p)],[np.cos(p)*np.cos(l),np.cos(p)*np.sin(l),np.sin(p)]])
a=np.deg2rad(AZ);e=np.deg2rad(ELS)
BEAMS=np.array([np.sin(a)*np.cos(e),np.cos(a)*np.cos(e),np.sin(e)]).T
START=datetime(2026,11,19,11,39,59,tzinfo=timezone.utc).timestamp()
STOP=datetime(2026,11,20,5,11,33,tzinfo=timezone.utc).timestamp()
COARSE=np.arange(START,STOP+20,20.)
def time_arrays(unix):
    jd=np.asarray(unix)/86400+2440587.5;whole=np.floor(jd);fraction=jd-whole
    t=ts.ut1_jd(jd) # seed UTC -> UT1 correction using bundled leap/DeltaT tables
    # UT1-UTC is small, but apply the modeled offset consistently.
    dut1=np.asarray(t.dut1)/86400
    theta,_=theta_GMST1982(whole,fraction+dut1)
    return whole,fraction,np.cos(theta),np.sin(theta)
CT=time_arrays(COARSE)
def observe(sat,unix,cache=None):
    whole,fraction,c,s=cache if cache is not None else time_arrays(np.atleast_1d(unix))
    err,r,v=sat.sgp4_array(whole,fraction)
    x=c*r[:,0]+s*r[:,1];y=-s*r[:,0]+c*r[:,1]
    xyz=np.array([x,y,r[:,2]]).T
    vel=np.array([c*v[:,0]+s*v[:,1],-s*v[:,0]+c*v[:,1],v[:,2]]).T
    vel+=np.array([7.29211514670698e-5*y,-7.29211514670698e-5*x,np.zeros_like(x)]).T
    d=xyz-SITE;ran=np.linalg.norm(d,axis=1);unit=d/ran[:,None];enu=unit@ENU.T
    sep=np.rad2deg(np.arccos(np.clip(enu@BEAMS.T,-1,1)))
    sep[err!=0]=np.nan
    az=np.rad2deg(np.arctan2(enu[:,0],enu[:,1]))%360;el=np.rad2deg(np.arcsin(enu[:,2]))
    rr=np.sum(unit*vel,axis=1)
    return sep,ran,rr,az,el
def work(record):
    try:sat=Satrec.twoline2rv(record['TLE_LINE1'],record['TLE_LINE2'])
    except Exception:return []
    sep,ran,rr,az,el=observe(sat,COARSE,CT)
    result=[]
    for b in range(5):
        angle=sep[:,b]
        minima=np.flatnonzero((angle[1:-1]<angle[:-2])&(angle[1:-1]<=angle[2:])&(angle[1:-1]<25))+1
        for i in minima:
            centre=COARSE[i]
            opt=minimize_scalar(lambda dt:observe(sat,[centre+dt])[0][0,b],bounds=(-20.,20.),method='bounded',options={'xatol':.001})
            t=centre+opt.x;ang,r,vr,az1,el1=observe(sat,[t])
            if START<=t<=STOP and ang[0,b]<5 and 0<r[0]<2300:
                result.append((b,int(record['NORAD_CAT_ID']),t,ang[0,b],r[0],vr[0],az1[0],el1[0],record['EPOCH'],record['OBJECT_NAME'],record['TLE_LINE1'],record['TLE_LINE2']))
    return result
def main():
    p=argparse.ArgumentParser();p.add_argument('snapshot');p.add_argument('output');p.add_argument('--workers',type=int,default=16);q=p.parse_args()
    raw=Path(q.snapshot).read_bytes();records=json.loads(gzip.decompress(raw) if q.snapshot.endswith('.gz') else raw)
    records=[r for r in records if r.get('TLE_LINE1') and r.get('TLE_LINE2')]
    # Validate frame/sign conventions against independent Skyfield topocentric output.
    rec=records[0];sat=Satrec.twoline2rv(rec['TLE_LINE1'],rec['TLE_LINE2']);times=np.linspace(START,STOP,3)
    _,r,vr,az,el=observe(sat,times)
    sky=EarthSatellite(rec['TLE_LINE1'],rec['TLE_LINE2'],ts=ts)
    t=ts.from_datetimes([datetime.fromtimestamp(v,timezone.utc) for v in times])
    alt,azi,dist=(sky-station).at(t).altaz()
    np.testing.assert_allclose(r,dist.km,atol=.02,rtol=0)
    np.testing.assert_allclose(el,alt.degrees,atol=.002,rtol=0)
    np.testing.assert_allclose((az-azi.degrees+180)%360-180,0,atol=.002,rtol=0)
    print('PASS independent Skyfield frame check; propagating',len(records),flush=True)
    rows=[]
    with ProcessPoolExecutor(max_workers=q.workers) as pool:
        for i,passes in enumerate(pool.map(work,records,chunksize=32)):
            rows.extend(passes)
            if i%2000==0:print(i,'objects',len(rows),'crossings',flush=True)
    rows.sort(key=lambda r:(r[0],r[3]));dt=np.dtype([(k,'f8') for k in ['beam_index','norad_id','utc_s','separation_deg','range_km','range_rate_km_s','azimuth_deg','elevation_deg']])
    with h5py.File(q.output,'w') as h:
        h['crossings']=np.array([tuple(r[:8]) for r in rows],dtype=dt)
        for j,key in enumerate(['tle_epoch','object_name','tle_line1','tle_line2'],8):h[key]=np.array([r[j] for r in rows],dtype=h5py.string_dtype())
        h['beam_az_el_deg']=np.c_[np.full(5,AZ),ELS]
        h.attrs.update(snapshot_sha256=hashlib.sha256(raw).hexdigest(),source_snapshot=q.snapshot,site_lat_lon_alt_km=[LAT,LON,ALT],utc_start=START,utc_stop=STOP,maximum_separation_deg=5.,maximum_range_km=2300.,coarse_step_s=20.,coarse_candidate_angle_deg=25.,frame_check='Skyfield altaz within 0.002 deg; distance within 0.02 km',git_commit=os.environ.get('SOURCE_GIT_COMMIT','working'))
    for b in range(5):
        sub=[r for r in rows if r[0]==b];print('BEAM',150+50*b,'crossings',len(sub),'closest',[(r[1],round(r[3],3),round(r[4],1),datetime.fromtimestamp(r[2],timezone.utc).isoformat()) for r in sub[:10]],flush=True)
if __name__=='__main__':main()
