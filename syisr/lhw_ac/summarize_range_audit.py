"""Numerical range/altitude checks and compact plot inputs; no fitted range shift."""
import argparse,importlib.util
from pathlib import Path
import h5py,numpy as np
from pyproj import Transformer
from reduction import range_mean

p=argparse.ArgumentParser();p.add_argument('evidence',type=Path);p.add_argument('output',type=Path);q=p.parse_args()
sp=importlib.util.spec_from_file_location('geo',Path(__file__).parent.parent/'lower_hybrid/index_recording.py');geo=importlib.util.module_from_spec(sp);sp.loader.exec_module(geo)
site=geo.ecef(18.34,109.62,0);basis=geo.basis(18.34,109.62);proj=Transformer.from_crs('EPSG:4978','EPSG:4979',always_xy=True)
els=[61.58935546875,60.897216796875,60.1776123046875,59.425048828125,58.64501953125]
def alt(r,b):
    a,e=np.deg2rad([357.9949951171875,els[b]]);u=np.array([np.sin(a)*np.cos(e),np.cos(a)*np.cos(e),np.sin(e)])@basis
    pts=site+np.asarray(r)[...,None]*u
    _,_,height=proj.transform(pts[...,0]*1000,pts[...,1]*1000,pts[...,2]*1000)
    test=np.array([geo.geod(v)[2] for v in pts.reshape(-1,3)]).reshape(np.shape(r))
    np.testing.assert_allclose(height/1000,test,atol=1e-5,rtol=0)
    return height/1000
def acf(h,name,cut,b):
    g=h[f'{name}/cut{cut}'];v=g[f'beam_{b}_acf'][:];return g['slant_range_km'][:],v[...,0]+1j*v[...,1]
with h5py.File(q.evidence) as h,h5py.File(q.output,'w') as out:
    h.copy('raw_ipp',out);sim=[];sat=[]
    for b in range(5):
        source=h['injected_source_sample_offset'][b];truth=(667+source*10-h.attrs['delay_correction_us'])*.149896229
        for cut in [0,32]:
            r,a=acf(h,'coherent_point',cut,b)
            for lag in [1,2,10]:
                peak=np.nanargmax(abs(a[:,lag]));assert abs(r[peak]-truth)<1e-8,(b,cut,lag,r[peak],truth)
                sim.append([b,cut,lag,truth,r[peak]])
        r,a=acf(h,'coherent_point',32,b);rc,ac=acf(h,'coherent_point_plus_clutter',32,b)
        np.testing.assert_allclose(a[:,1:11],ac[:,1:11],rtol=0,atol=1e-5,equal_nan=True)
        z=alt(r,b);g=out.create_group(f'sim_{b}');g['altitude_km']=z;g.attrs['truth_altitude_km']=alt(np.asarray(truth),b)
        for name,case,cut in [('plain','coherent_point',0),('excluded','coherent_point',32),('clutter_excluded','coherent_point_plus_clutter',32)]:
            rr,aa=acf(h,case,cut,b);g[name]=abs(aa[:,2])/1e8
        g['smoothed_altitude_km']=alt(range_mean(r,4),b);g['four_gate']=abs(range_mean(a[:,2],4))/1e8
        print('POINT',b,'physical altitude km',g.attrs['truth_altitude_km'],'range correct, clutter-veto identical',flush=True)
    for case in range(2):
        for b in range(5):
            raw=h['raw_matched_selected_range_km'][case,b];pred=h['predicted_satellite_range_km'][case,b]
            r,a=acf(h,f'satellite_{case}',32,b);_,un=acf(h,f'satellite_{case}',0,b)
            local=(abs(r-raw)<30)&np.isfinite(a[:,2]);power=abs(a[:,2]);ix=np.flatnonzero(local)[np.argmax(power[local])]
            global_peak=np.nanargmax(power);sat.append([case,150+50*b,raw,r[ix],r[ix]-raw,pred.min(),pred.max(),r[global_peak]])
            assert global_peak==ix
            assert abs(r[ix]-raw)<=1.4991
            np.testing.assert_array_equal(a[local],un[local])
            g=out.create_group(f'sat_{case}_{b}');g['relative_range_km']=r[local]-raw;g['excluded']=power[local]/power[ix];g['unexcluded']=abs(un[local,2])/power[ix]
            g.attrs.update(raw_matched_range_km=raw,decoded_peak_range_km=r[ix],predicted_min_relative_km=pred.min()-raw,predicted_max_relative_km=pred.max()-raw)
            print('SAT',case,150+50*b,'raw',raw,'decoded',r[ix],'difference km',r[ix]-raw,'global decoded',r[global_peak],flush=True)
    out['simulation_checks']=sim;out['satellite_checks']=sat
    out.attrs.update(simulation_result='PASS 30 production point-target lag/range checks; early clutter exclusion gives identical short-lag target response',altitude_result='Independent PROJ ECEF-to-WGS84 geodetic height agrees within 1 cm',delay_correction_us=h.attrs['delay_correction_us'])
