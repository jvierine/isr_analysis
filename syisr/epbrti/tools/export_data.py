"""Export signed HDF5 science products to WebGL textures and physical mappings.
The compact R8 display texture is a 0.106dB quantization; native HDF5 retains values.
"""
from pathlib import Path
import json,sys
import numpy as np,h5py
from scipy.interpolate import interp1d
sys.path.insert(0,str(Path(__file__).resolve().parents[2]/'lfm_epb'/'meridians'))
from storm_palette import STORM
root=Path(sys.argv[1]);out=Path(sys.argv[2]);pub=out/'public';pub.mkdir(parents=True,exist_ok=True)
G=['west','central','east'];groups=[[8,7,6],[9,10,11,5,4],[1,2,3]]
manifest={'event':'SYISR · 1 March 2026','tref':1772369100.,'tmin':1772365500.,'tmax':1772372700.,'grid':{'nx':701,'ny':351,'x0':-700.,'y0':-300.,'dx':2.},'groups':[{'name':n,'beams':b} for n,b in zip(G,groups)],'beams':[],'db':[78.,105.],'palette':STORM,'resolution_note':'Measured: 5 s × 0.375 km range sampling. Model grid: 2 km; physical resolution is limited by the beam and inversion averaging.'}
seed=[];support=[]
for n in G:
 with h5py.File(root/f'editor_seed/{n}.h5') as h:
  q=1+h['contrast'][()];s=h['support_weight'][()];assert q.shape==(351,701);assert np.isfinite(q).all();seed.append(q.astype('f4'));support.append((s>.005*s.max()).astype('f4'));print(n,h['diagnostics'][()])
seed=np.array(seed);support=np.array(support);(pub/'seed.bin').write_bytes(seed.astype('<f4').tobytes());(pub/'support.bin').write_bytes(support.astype('<f4').tobytes())
meta={'revision':0,'drift_m_s':120.,'upward_m_s':0.,'nx':701,'ny':351,'updated_unix_s':0}
with h5py.File(out/'seed.h5','w') as h:
 h['metadata_json']=np.frombuffer(json.dumps(meta).encode(),dtype='u1');h.create_dataset('state',data=np.array([seed,np.zeros_like(seed)]),compression='gzip',compression_opts=4)
 h.attrs['science']='Independent meridian robust depletion relative-echo maps, constant drift120m/s; no calibrated density. Finer grid is model parameterization, not claimed independent physical resolution.'
with h5py.File(root/'meridian_cutouts/cutouts.h5') as source,h5py.File(root/'geometry.h5') as geo,h5py.File(root/'depletion_prediction/rti_predictions.h5') as ob:
 names=sorted(k for k in geo if k.startswith('beam_'));obs=ob['observations'][()]
 for b in range(1,12):
  a=source[f'rtis/beam_{b:02}'];ranges=a['range_km'][()];te=a['time_edges_unix_s'][()];ze=a['altitude_edges_km'][()];sig=a['signed_signal'][()];nr=len(ranges);nt=len(te)-1
  g=geo[names[b-1]];r=g['range_km'][()];center=g['field_line_reference_plane_km'][()];quad=g['beam_quadrature_field_line_km'][()][:,:,:2].reshape(len(r),18)
  z=g['geodetic_lat_lon_alt'][()][:,2];geom=interp1d(r,np.column_stack([z,center[:,:2],quad]),axis=0,bounds_error=False,fill_value='extrapolate')(ranges).astype('<f4')
  rr=obs[obs[:,1]==b-1];unique,ix=np.unique(rr[:,2],return_index=True);ref=np.interp(ranges,unique,rr[ix,8],left=np.nan,right=np.nan).astype('<f4')
  # Do not fabricate quiet-reference interpolation across large rejected gaps.
  nearest=np.min(abs(ranges[:,None]-unique[None,:]),axis=1);ref[nearest>7.]=np.nan
  with np.errstate(invalid='ignore',divide='ignore'):db=np.where(sig>0,10*np.log10(sig*ranges[None,:]**2),78.)
  tex=(np.clip(np.rint((db-78)/27*254),0,254)+1).astype('u1');tex[~np.isfinite(sig)]=0
  (pub/f'beam{b:02}.bin').write_bytes(tex.T.copy().tobytes());(pub/f'geom{b:02}.bin').write_bytes(geom.tobytes());(pub/f'ref{b:02}.bin').write_bytes(ref.tobytes())
  az,el=a['az_el_deg'][()];manifest['beams'].append({'id':b,'group':next(i for i,ids in enumerate(groups) if b in ids),'nt':nt,'nr':nr,'time0':float((te[0]+te[1])/2),'dt':float(np.median(np.diff(te))),'time_edges':[float(te[0]),float(te[-1])],'r0':float(ranges[0]),'dr':float(np.median(np.diff(ranges))),'range_edges':[float(ranges[0]-(ranges[1]-ranges[0])/2),float(ranges[-1]+(ranges[1]-ranges[0])/2)],'altitude_edges':[float(ze[0]),float(ze[-1])],'az':float(az),'el':float(el),'training':b!=9})
  print('BEAM',b,nt,nr)
(pub/'manifest.json').write_text(json.dumps(manifest,separators=(',',':')))
print('EXPORTED',out)
