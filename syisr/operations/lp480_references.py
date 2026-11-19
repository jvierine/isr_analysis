#!/usr/bin/env python3
"""External official IRI tables for shared LP480 composition shape/calibration.
Run with the base conda environment. Fortran supplies the scientific model;
Python transports/validates the tables and writes HDF5, never CSV.
"""
import argparse,datetime as dt,hashlib,io,json,subprocess
from pathlib import Path
import h5py,numpy as np

def main():
 p=argparse.ArgumentParser(__doc__);p.add_argument('iri_directory',type=Path);p.add_argument('output',type=Path);p.add_argument('--start',default='2023-11-15T00:00:00+00:00');p.add_argument('--end',default='2023-11-17T00:00:00+00:00');a=p.parse_args();a.output.mkdir(parents=True,exist_ok=True)
 hashes={n:hashlib.sha256((a.iri_directory/n).read_bytes()).hexdigest() for n in ['irisub.for','irifun.for','apf107.dat','ig_rz.dat']}
 start=dt.datetime.fromisoformat(a.start);end=dt.datetime.fromisoformat(a.end)
 for time in range(int(start.timestamp()),int(end.timestamp()),1800):
  out=a.output/f'{time}.h5'
  if out.exists():continue
  t=dt.datetime.fromtimestamp(time+900,dt.timezone.utc)
  cmd=[str((a.iri_directory/'iri_shape_driver').resolve()),*[str(v) for v in [t.year,t.month,t.day,t.hour,t.minute,t.second,18.3492,109.6222,100,2000,25]],'.']
  text=subprocess.check_output(cmd,cwd=a.iri_directory,text=True)
  v=np.genfromtxt(io.StringIO(text),max_rows=77)
  if v.shape!=(77,12) or np.any(v[:,1:5]<=0) or np.any(~np.isfinite(v)):raise ValueError(f'Invalid IRI table at {t}')
  z=v[:,0];heavy=np.maximum(v[:,[5,9,8,11]],0);den=heavy.sum(axis=1);heavy=heavy/np.maximum(den[:,None],1e-30);heavy[den<=0]=[1,0,0,0]
  mask=z>=500;g=9.80665*(6371.05/(6371.05+z[mask]))**2
  scale=np.maximum.accumulate(1.380649e-23*v[mask,3]/(15*1.66053906660e-27*g)/1000)
  shape=dict(altitude_km=z[mask].tolist(),scale_height_km=scale.tolist(),source='IRI2020 Ti/g differential H/O scale-height coordinate; monotone cumulative maximum; shared Sanya geometry')
  value=dict(altitude_km=z.tolist(),ne_m3=v[:,1].tolist(),ti_k=v[:,3].tolist(),te_k=v[:,4].tolist(),heavy_fractions=heavy.tolist(),shape=shape,provenance=dict(utc_center=t.isoformat(),latitude_deg=18.3492,longitude_deg=109.6222,heavy_species_order=['O+','NO+','O2+','N+'],hashes=hashes,density='Empirical calibration reference; not observed density or an absolute system calibration'))
  pending=out.with_suffix('.pending.h5')
  with h5py.File(pending,'w') as f:f.create_dataset('json_utf8',data=np.frombuffer(json.dumps(value).encode(),dtype='u1'),compression='gzip')
  pending.replace(out)
  print('IRI_REFERENCE',time,flush=True)
if __name__=='__main__':main()
