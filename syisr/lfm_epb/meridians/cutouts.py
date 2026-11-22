"""Three independent depletion cutouts and exact field-line/beam marker mappings.
Scientific data stay HDF5. Markers selected geometrically, never shifted onto RTIs.
"""
from pathlib import Path
import sys,os,subprocess
import h5py,numpy as np
from scipy.ndimage import label
from mpi4py import MPI
root=Path(sys.argv[1]);native=Path(sys.argv[2]);out=root/'meridian_cutouts';out.mkdir(exist_ok=True)
comm=MPI.COMM_WORLD
assert comm.size<=16
GROUPS={'west':[8,7,6],'central':[9,10,11,5,4],'east':[1,2,3]}
with h5py.File(root/'cavity/observations.h5') as h:obs=h['observations'][()];eligible=h['eligible'][()];stats=h['beam_statistics'][()]
with h5py.File(root/'three_meridians/drift_profile.h5') as h:vx=float(h['pooled_speed_m_s'][()])/1000
T0=1772369100.;TMIN=T0-3600;TMAX=T0+3600
for j,(name,beams) in enumerate(GROUPS.items()):
 if j%comm.size!=comm.rank:continue
 with h5py.File(out/f'{name}_observations.h5','w') as h:
  h['observations']=obs;h['eligible']=eligible*np.isin(obs[:,1]+1,[b for b in beams if b!=9]);h['beam_statistics']=stats
 env=dict(os.environ,SYISR_CAVITY_DEPLETION='.2',SYISR_CAVITY_ITERATIONS='6000')
 subprocess.run([str(native),'fit',str(out/f'{name}_observations.h5'),str(out/f'{name}_fit.h5'),str(vx),'0','100','-1'],env=env,check=True)
comm.Barrier()
if comm.rank:sys.exit(0)
geo=h5py.File(root/'geometry.h5');names=sorted(k for k in geo if k.startswith('beam_'));weights=np.outer([1/6,2/3,1/6],[1/6,2/3,1/6]).ravel()
def crossing(beam,u,v,quad=None):
 g=geo[names[beam-1]];r=g['range_km'][()];z=g['geodetic_lat_lon_alt'][()][:,2]
 m=g['field_line_reference_plane_km'][()] if quad is None else g['beam_quadrature_field_line_km'][()][:,quad,:]
 ix=(z>=150)&(z<=900);r=r[ix];z=z[ix];m=m[ix]
 if not np.all(np.diff(m[:,1])>0):raise ValueError('Nonmonotonic meridian crossing: solve all roots explicitly')
 if not m[0,1]<=v<=m[-1,1]:return np.full(4,np.nan)
 rr=np.interp(v,m[:,1],r);uu=np.interp(rr,r,m[:,0]);vv=np.interp(rr,r,m[:,1]);tt=T0+(uu-u)/vx;zz=np.interp(rr,r,z)
 if not TMIN<=tt<TMAX:return np.full(4,np.nan)
 return np.array([tt,rr,zz,np.hypot(uu-vx*(tt-T0)-u,vv-v)])
x=np.arange(261)*10-1300;y=np.arange(161)*10-800;xx,yy=np.meshgrid(x,y)
with h5py.File(out/'cutouts.h5','w') as result:
 result['x_km']=x;result['y_km']=y;result['speed_m_s']=vx*1000;result['reference_epoch_unix_s']=T0;result['basis_e1_e2_B_ecef']=geo['basis_e1_e2_B_ecef'][()];result['reference_ecef_km']=geo['reference_ecef_km'][()]
 for name,beams in GROUPS.items():
  with h5py.File(out/f'{name}_fit.h5') as h:q=1+h['contrast'][()];support=h['support_weight'][()];diagn=h['diagnostics'][()]
  assert diagn[2]<1e-5,(name,diagn)
  mask=(support>.005*support.max());candidate=mask&(q<.6)&(xx>=-180)&(xx<=220)&(yy>=-130)&(yy<=180)
  lab,n=label(candidate);sizes=np.bincount(lab.ravel());sizes[0]=0;candidate &=lab==np.argmax(sizes)
  points=np.argwhere(candidate);valid=[]
  for iy,ix in points:
   if all(np.isfinite(crossing(b,x[ix],y[iy])).all() for b in beams):valid.append((iy,ix))
  points=np.array(valid);assert len(points)>=5
  uv=np.column_stack([x[points[:,1]],y[points[:,0]]]);depth=1-q[points[:,0],points[:,1]]
  # Spread over the connected depleted structure; seeded near its central trunk.
  chosen=[int(np.argmin(np.linalg.norm(uv,axis=1)))]
  while len(chosen)<5:
   dist=np.min(np.linalg.norm(uv[:,None,:]-uv[chosen][None,:,:],axis=2),axis=1);score=dist*np.sqrt(depth);score[chosen]=-1;chosen.append(int(np.argmax(score)))
  chosen=sorted(chosen,key=lambda j:uv[j,1]);marks=uv[chosen];mratio=q[points[chosen,0],points[chosen,1]]
  g=result.create_group(name);g['beam_ids']=beams;g['ratio']=q;g['support_weight']=support;g['display_support']=mask;g['markers_uv_km']=marks;g['markers_fitted_ratio']=mratio;g['diagnostics']=diagn
  rows=[]
  for b in beams:
   for k,(u,v) in enumerate(marks):
    cr=crossing(b,u,v);quad=np.array([crossing(b,u,v,j) for j in range(9)])
    assert cr[3]<1e-6
    if np.isfinite(quad).all():
     mean=weights@quad[:,:3];cov=(quad[:,:3]-mean).T@np.diag(weights)@(quad[:,:3]-mean);sd=np.sqrt(np.diag(cov))
    else:sd=np.full(3,np.nan)
    take=(obs[:,1]==b-1)&(abs(obs[:,0]-cr[0])<60)&(abs(obs[:,2]-cr[1])<10);meas=np.median(obs[take,6]) if take.any() else np.nan
    rows.append([b,k+1,*cr,*sd,meas])
  g['marker_crossings']=rows;g['marker_crossings'].attrs['columns']='beam,id,time_unix_s,range_km,altitude_km,mapping_residual_km,beam_rms_time_s,beam_rms_range_km,beam_rms_altitude_km,local_observed_ratio_median'
  g.attrs['marker_selection']='Five spatially separated points of largest connected supported ratio<0.6 component in u[-180,220],v[-130,180]km; finite crossings in every meridian beam. Independent labels per meridian. No RTI alignment tuning.'
  print(name,'MARKERS',marks,'RATIOS',mratio,'CROSSINGS',np.array(rows),flush=True)
 # Retain fine native display measurements separately from coarse inverse samples.
 bg=result.create_group('rtis')
 for b,name in enumerate(names,1):
  with h5py.File(root/'beams'/name/'display.h5') as h:
   te=h['time_edges_unix_s'][()];ze=h['altitude_edges_km'][()];r=h['range_km'][()]
   ti=np.flatnonzero((te[:-1]>=TMIN)&(te[1:]<=TMAX));zi=np.flatnonzero((ze[:-1]>=150)&(ze[1:]<=900));a,c=ti[0],ti[-1]+1;d,e=zi[0],zi[-1]+1
   g=bg.create_group(f'beam_{b:02}');g['time_edges_unix_s']=te[a:c+1];g['altitude_edges_km']=ze[d:e+1];g['range_km']=r[d:e];g['az_el_deg']=geo[name]['az_el_deg'][()]
   signal=h['median_signal'][a:c,d:e];g.create_dataset('signed_signal',data=signal.astype('f4'),compression='gzip',compression_opts=6,shuffle=True)
   g.attrs['display_units']='Signed noise-subtracted signal; range squared dB calculated only for display. Nonpositive samples shown at lowest color, not NaN.'
 result.attrs['method']='Three independent finite-beam field-aligned depletion fits: common120m/s,lambda100,delta0.2 asymmetric Huber,relativeecho0..1. Beam9 excluded fit, included marker geometry and RTIs. Reference plane is perpendicular to IGRF at400km above site. No imposed shared morphology across meridians.'
 result.attrs['mapping']='For marker(u0,v0): solve v_beam(r)=v0; t=T0+(u_beam(r)-u0)/vx. Each point denotes a co-moving field-line label. Finitebeam RMS spreads use nine angular quadrature rays and are not confidence intervals. Geometry from dated IGRF; altitude separate from range.'
 result.attrs['limitations']='Fieldalignment assumed within each group. Frozen morphology within crossing; distinct groups sample different times. Marker coincidence is geometric prediction, not proof of alignment. Depth is relative echo and contaminated echoes may hide a depletion.'
print('CUTOUTS_COMPLETE',flush=True)
