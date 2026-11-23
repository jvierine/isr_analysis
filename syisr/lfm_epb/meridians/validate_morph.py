"""Central-meridian prediction from outer sections: temporal blending vs frozen.
No central event samples enter these field fits. Drift comes from W/E pairs;
smoothing comes from W/E whole-beam CV only. Error is conditional on reference,
constant drift, empirical weights and the common support selection.
"""
from pathlib import Path
import sys,subprocess
import h5py,numpy as np
ROOT=Path(sys.argv[1]);OUT=ROOT/'three_meridians'
sys.path.insert(0,str(Path(__file__).resolve().parents[1]));from symmetry_validation import design
with h5py.File(OUT/'section_selection.h5') as h:
    lam=h['lambda_grid'][()][np.argmin(np.mean(h['whole_beam_holdout_rmse'][()][[0,2]]**2,axis=0))];speed=h['speed_m_s'][()]/1000
with h5py.File(ROOT/'cavity/observations.h5') as h:
    o=h['observations'][()];e=h['eligible'][()];stats=h['beam_statistics'][()]
with h5py.File(OUT/'outer_observations.h5','w') as h:
    h['observations']=o;h['eligible']=e*np.isin(o[:,1]+1,[1,2,3,6,7,8]);h['beam_statistics']=stats
subprocess.run([str(ROOT/'cavity/native'),'fit',str(OUT/'outer_observations.h5'),str(OUT/'outer_frozen.h5'),str(speed),'0',str(lam),'-1'],check=True)
p,arc,A,_,_,_,original=design(ROOT,exclude_beam=9,velocity=(speed,0))
with h5py.File(OUT/'sections.h5') as h:
    cw=h['west/contrast'][()].ravel();ce=h['east/contrast'][()].ravel();tw=h['west/mean_observation_time_unix_s'][()].ravel();te=h['east/mean_observation_time_unix_s'][()].ravel()
    sw=h['west/support_weight'][()].ravel();se=h['east/support_weight'][()].ravel();mask=(sw>.005*sw.max())&(se>.005*se.max())&(te>tw)
with h5py.File(OUT/'outer_frozen.h5') as h:frozen=h['contrast'][()].ravel()
keep=np.isin(p[:,1]+1,[4,5,10,11])&(A@mask.astype(float)>.999)
W=A@cw;E=A@ce;TW=A@tw;TE=A@te;fraction=(p[:,0]-TW)/np.maximum(TE-TW,1)
keep&=(fraction>=0)&(fraction<=1)&(TE-TW>300)
pred_morph=1+(1-fraction)*W+fraction*E;pred_frozen=1+A@frozen
with h5py.File(OUT/'morph_validation.h5','w') as h:
    h['observation_index']=original[keep];h['observed_ratio']=p[keep,6];h['morph_prediction']=pred_morph[keep];h['frozen_prediction']=pred_frozen[keep];h['local_time_fraction']=fraction[keep];h['time_unix_s']=p[keep,0];h['altitude_km']=p[keep,3];h['beam_id']=p[keep,1]+1
    h['rmse']=[np.sqrt(np.mean((pred_frozen[keep]-p[keep,6])**2)),np.sqrt(np.mean((pred_morph[keep]-p[keep,6])**2))];h['count']=keep.sum();h['central_eligible_count']=np.isin(p[:,1]+1,[4,5,10,11]).sum();h['lambda_from_outer_beams']=lam
    h.attrs['scope']='Central meridian held out entirely from field fitting and outer-only smoothing choice. Same observed targets evaluated. Predictions restricted to outer common finite-beam support, local temporal bracketing, and>5min time separation. Drift estimated using W/E pairs. No significance claim.'
    print('CENTRAL_MERIDIAN_TEST',h['rmse'][()],'count',keep.sum(),'of',h['central_eligible_count'][()],flush=True)
