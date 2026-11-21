"""Actual five-beam geometry; nominal timing and provisional phase centre."""
from pathlib import Path
from datetime import datetime
import sys,importlib.util
import h5py,numpy as np
from scipy.optimize import brentq
P=Path(__file__).resolve().parent
sp=importlib.util.spec_from_file_location('rdgeo',P.parent/'lower_hybrid/index_recording.py');m=importlib.util.module_from_spec(sp);sp.loader.exec_module(m)
with h5py.File(sys.argv[1]) as h:
 if 'beam_az_el_deg' in h:angles=h['beam_az_el_deg'][:]
 else:
  b=h['initial_headers'][:];angles=np.unique(np.stack([b[:,856].astype(int)+256*b[:,857],b[:,858].astype(int)+256*b[:,859]],axis=1),axis=0)*360/65536.;angles=angles[np.argsort(-angles[:,1])]
assert angles.shape==(5,2)
site=m.ecef(18.34,109.62,0);enu=m.basis(18.34,109.62);date=datetime(2026,11,19,12);rows=[]
for i,(az,el) in enumerate(angles):
 aa,ee=np.deg2rad([az,el]);u=np.array([np.sin(aa)*np.cos(ee),np.cos(aa)*np.cos(ee),np.sin(ee)])@enu
 def aspect(r):
  b=m.field(site+r*u,date);return np.dot(u,b)/np.linalg.norm(b)
 r=brentq(aspect,100.,600.,xtol=1e-8);x=site+r*u;b=m.field(x,date);rows.append([150+i*50,az,el,r,m.geod(x)[2],np.linalg.norm(b)/1000,np.rad2deg(np.arcsin(aspect(r)))])
with h5py.File(sys.argv[2],'w') as h:
 h['beam_geometry']=rows;h.attrs['columns']='nominal_target_altitude_km,recorded_az_deg,recorded_el_deg,perpendicular_slant_range_km,perpendicular_WGS84_altitude_km,B_uT,signed_aspect_deg';h.attrs['site']='18.34N109.62E0km provisional';h.attrs['IGRF_epoch_UTC']='2026-11-19T12:00:00Z'
print(np.array(rows))
