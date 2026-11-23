"""Display levels average signed linear estimates before logs or saturation.
Apparent density uses the background-equalized calibrated arithmetic mean product.
Power levels average native signed display estimates (per-bin pulse medians).
"""
from pathlib import Path
import sys,json
import h5py,numpy as np
root=Path(sys.argv[1]);out=Path(sys.argv[2]);pub=out/'public';pub.mkdir(parents=True,exist_ok=True)
manifest=json.loads((pub/'manifest.json').read_text());manifest['measurement_levels']={};science=h5py.File(out/'measurement_levels.h5','w')
DENSITY=root/'coherent_scatter_evidence/background_equalized.h5';SOURCE=root/'multibeam_expanded/meridian_cutouts/cutouts.h5'
def reduce(a,ft,fr):
 nt=a.shape[0]//ft;nr=a.shape[1]//fr;b=a[:nt*ft,:nr*fr].reshape(nt,ft,nr,fr)
 with np.errstate(invalid='ignore'):return np.nanmean(b,axis=(1,3))
def layer(obs,name,level,a,t,r,dt,dr,beam):
 limits=[10.,14.] if obs=='density' else [78.,105.]
 with np.errstate(divide='ignore',invalid='ignore'):v=np.where(a>0,np.log10(a)*(1 if obs=='density' else 10),limits[0]);pix=(np.clip(np.rint((v-limits[0])/(limits[1]-limits[0])*254),0,254)+1).astype('u1')
 pix[~np.isfinite(a)]=0;filename=f'{obs}_{beam:02}_{level}.bin';(pub/filename).write_bytes(pix.T.copy().tobytes())
 g=science.require_group(f'{obs}/beam{beam:02}').create_group(level);g.create_dataset('signed_linear_estimate',data=a.astype('f4'),compression='gzip');g['time_unix_s']=t;g['range_km']=r
 return {'file':filename,'nt':len(t),'nr':len(r),'time0':float(t[0]),'dt':dt,'r0':float(r[0]),'dr':dr,'time_edges':[float(t[0]-dt/2),float(t[-1]+dt/2)],'range_edges':[float(r[0]-dr/2),float(r[-1]+dr/2)],'label':f'{dt:g} s × {dr:.2f} km'}
with h5py.File(DENSITY) as h,h5py.File(SOURCE) as raw:
 names=sorted(k for k in h if k.startswith('beam_') and isinstance(h[k],h5py.Group))
 for b in range(1,12):
  g=h[names[b-1]];t=g['time_unix_s'][()];r=g['range_km'][()];a=g['background_equalized_Ne_m3'][()];dt=float(np.median(np.diff(t)));dr=float(np.median(np.diff(r)));q=(t>=manifest['tmin'])&(t<manifest['tmax']);tt=t[q];aa=a[q]
  quiet=(t>=1772362800)&(t<1772364600)
  baseline=np.nanmedian(a[quiet],axis=0);baseline[baseline<=0]=np.nan
  (pub/f'density_ref{b:02}.bin').write_bytes(baseline.astype('<f4').tobytes())
  meta={'power':{},'density':{},'density_prediction':{'nt':len(tt),'nr':len(r),'time0':float(tt[0]),'dt':dt,'r0':float(r[0]),'dr':dr,'reference_file':f'density_ref{b:02}.bin'},'density_calibration_supported':bool(g.attrs['supported'])}
  for level,f in [('coarse',1),('verycoarse',2),('smooth',4)]:
   z=reduce(aa,f,f);tx=tt[:z.shape[0]*f].reshape(-1,f).mean(1);rx=r[:z.shape[1]*f].reshape(-1,f).mean(1);meta['density'][level]=layer('density',names[b-1],level,z,tx,rx,dt*f,dr*f,b)
  bg=raw[f'rtis/beam_{b:02}'];s=bg['signed_signal'][()];r=bg['range_km'][()];te=bg['time_edges_unix_s'][()];tx=(te[:-1]+te[1:])/2;dr=float(np.median(np.diff(r)));dt=float(np.median(np.diff(tx)))
  for level,ft,fr in [('medium',3,4),('coarse',12,16),('verycoarse',24,32),('smooth',48,64)]:
   z=reduce(s,ft,fr);tt=tx[:z.shape[0]*ft].reshape(-1,ft).mean(1);rr=r[:z.shape[1]*fr].reshape(-1,fr).mean(1);power=z*rr[None,:]**2
   meta['power'][level]=layer('power',names[b-1],level,power,tt,rr,dt*ft,dr*fr,b)
  manifest['measurement_levels'][str(b)]=meta;print('LEVELS',b)
science.attrs['method']='Signed linear arithmetic averages before any log/saturation. Density source retains matched-background perbeam calibration. Power starts from 5s pulse-median display cells; averaging these is not a raw pulse-mean estimator.'
science.attrs['density_prediction']='Model ratio times median quiet background-equalized apparent density for each range. Initial morphology uses a different signed-power estimator; prediction is a conditional proxy, not a calibrated ISR spectral fit.'
science.close();(pub/'measurement_levels.json').write_text(json.dumps(manifest['measurement_levels'],separators=(',',':')))
print('EXPORTED_MEASUREMENT_LEVELS')
