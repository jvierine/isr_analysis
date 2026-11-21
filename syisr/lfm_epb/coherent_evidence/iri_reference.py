"""Explicitly assumed IRI2020 Ne/Te/Ti at physical beam sampling locations.
Run with base conda; never substitute this model for an independent density anchor.
"""
import argparse,hashlib
from pathlib import Path
from datetime import datetime,timezone
import h5py,numpy as np
import iri20py
from iri20py.base import DATADIR
p=argparse.ArgumentParser();p.add_argument('geometry');p.add_argument('output');a=p.parse_args()
model=iri20py.Iri2020();times=np.arange(1772362800,1772373601,1800.)
with h5py.File(a.geometry) as g,h5py.File(a.output,'w') as h:
    h.attrs['model']='IRI2020 empirical assumption, not measured temperatures or density';h.attrs['package_version']=iri20py.__version__
    h.attrs['source_geometry']=a.geometry;h['time_unix_s']=times
    for name in ['apf107.dat','ig_rz.dat']:
        path=DATADIR/name
        if path.exists():h.attrs[name+'_sha256']=hashlib.sha256(path.read_bytes()).hexdigest()
    for name in sorted(k for k in g if k.startswith('beam_')):
        xyz=g[name]['geodetic_lat_lon_alt'][()];ids=[]
        for z in np.arange(200,901,20):
            i=np.argmin(abs(xyz[:,2]-z))
            if i not in ids:ids.append(i)
        coords=xyz[ids];o=h.create_group(name);o['geodetic_lat_lon_alt']=coords;o['geometry_range_index']=ids
        vals=np.empty((len(times),len(ids),3))
        for j,t in enumerate(times):
            for k,(lat,lon,z) in enumerate(coords):
                settings,ds=model.evaluate(datetime.fromtimestamp(t,timezone.utc),lat,lon,np.array([z]),tzaware=True)
                vals[j,k]=[float(ds.Ne.values[0])*1e6,float(ds.Te.values[0]),float(ds.Ti.values[0])]
        o['ne_te_ti']=vals;o.attrs['columns']='Ne_m^-3, Te_K, Ti_K'
        print(name,'Ne',np.nanmin(vals[:,:,0]),np.nanmax(vals[:,:,0]),'Te/Ti',np.nanmin(vals[:,:,1]/vals[:,:,2]),np.nanmax(vals[:,:,1]/vals[:,:,2]),flush=True)
    h.attrs['settings_note']='iri20py Settings defaults retained; input indices and package recorded. Output Ne converted explicitly from library cm^-3 to m^-3.'
