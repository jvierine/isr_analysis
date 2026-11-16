"""Independent NumPy/raw-IQ check of Rust spectra and all normalization rows."""
import argparse
import json
from pathlib import Path

import h5py
import numpy as np
from subtract_background import quiet_average


def main():
    parser=argparse.ArgumentParser();parser.add_argument('index',type=Path);parser.add_argument('spectra',type=Path)
    q=parser.parse_args()
    report={}
    with h5py.File(q.index) as ix,h5py.File(q.spectra,'r+') as h:
        t=ix['time_utc_s'][:];beam=ix['beam_index'][:];starts=h['window_start_utc_s'][:];ends=h['window_end_utc_s'][:]
        assert len(starts)==38 and np.allclose(ends-starts,60.,rtol=0,atol=1e-8)
        assert np.all(np.diff(starts)>0) and np.all(np.diff(starts)<=30.000001)
        assert starts[0]==t[0] and ends[-1]>=t[-1]
        assert np.all(np.diff(t)>0) and len(t)==116509
        mm=np.memmap(ix.attrs['raw_path'],mode='r',dtype=np.uint8)
        stride=int(ix.attrs['record_stride_bytes']);n=int(ix.attrs['sample_count'])
        # Explicit signed int32 layout, independent of the Rust IQ decoder.
        samples=np.ndarray((len(t),n,2),dtype='<i4',buffer=mm,offset=1024,strides=(stride,8,4))
        w=h['fast_window_hann'][:].astype(float);nf=len(h['frequency_hz']);df=h.attrs['frequency_bin_hz'];fs=h.attrs['sample_rate_hz']
        gate=h['range_window_start_sample'][:].astype(int)
        assert len(w)==48 and nf==256 and len(gate)==73 and df==390.625
        expected=(533+(gate+(len(w)-1)/2)*10-240)*0.149896229
        np.testing.assert_array_equal(expected,h['slant_range_km'][:])
        geometry=h['beam_geometry'][:]
        assert list(geometry[:,0])==[250,350,450,550,650]
        assert np.max(abs(geometry[:,-1]))<1e-7
        worst_area=0.
        for b in range(5):
            g=h[f'beam_{b}'];raw=g['psd_adu2_per_hz'][:];norm=g['normalized_psd_per_hz'][:]
            assert np.isfinite(raw).all() and (raw>=0).all() and np.isfinite(norm).all()
            area=raw.sum(axis=-1)*df
            np.testing.assert_allclose(area,g['integrated_power_adu2'][:],rtol=2e-14)
            np.testing.assert_allclose(norm,raw/area[:,:,None],rtol=2e-14)
            deviation=float(np.max(abs(norm.sum(axis=-1)*df-1)))
            worst_area=max(worst_area,deviation);assert deviation<1e-12
            counts=np.array([np.count_nonzero((t>=s)&(t<e)&(beam==b)) for s,e in zip(starts,ends)])
            np.testing.assert_array_equal(counts,g['pulse_count'][:])
            if 'background_subtracted_psd_adu2_per_hz' in g:
                reference=h['background_reference_range_indices'][:]
                used=g['background_reference_gate_used'][:].astype(bool)
                background=g['background_psd_adu2_per_hz'][:]
                residual=g['background_subtracted_psd_adu2_per_hz'][:]
                np.testing.assert_array_equal(residual,raw-background[:,None,:])
                assert (residual<0).any()
                for i in range(len(starts)):
                    np.testing.assert_array_equal(background[i],raw[i,reference[used[i]],:].mean(axis=0))
                power=residual.sum(axis=2)*df
                np.testing.assert_array_equal(power,g['background_subtracted_integrated_power_adu2'][:])
                snorm=g['normalized_signal_psd_per_hz'][:]
                visible=g['normalization_display_valid'][:].astype(bool)
                np.testing.assert_allclose(snorm[visible].sum(axis=-1)*df,1.,rtol=0,atol=1e-10)
                assert np.isnan(snorm[power<=0]).all()
                assert not visible[:,reference].any()
        comparisons=[]
        for b,frame,r in [(0,0,10),(0,19,30),(1,12,16),(2,19,28),(3,30,39),(4,37,45)]:
            p=np.flatnonzero((t>=starts[frame])&(t<ends[frame])&(beam==b))
            block=samples[p,gate[r]:gate[r]+48,:]
            z=block[:,:,1].astype(float)+1j*block[:,:,0].astype(float)
            spectrum=np.mean(abs(np.fft.fftshift(np.fft.fft(z*w[None,:],nf,axis=1),axes=1))**2,axis=0)/(fs*np.sum(w*w))
            stored=h[f'beam_{b}/psd_adu2_per_hz'][frame,r,:]
            difference=float(np.max(abs(stored-spectrum))/np.max(abs(spectrum)))
            assert difference<2e-6,(b,frame,r,difference)
            # Parseval against original signed samples, not another FFT norm.
            integrated=np.mean(np.sum(abs(z*w)**2,axis=1)/np.sum(w*w))
            np.testing.assert_allclose(stored.sum()*df,integrated,rtol=2e-6)
            comparisons.append(dict(beam=b,frame=frame,range_gate=r,max_relative_difference=difference))
        report=dict(records=len(t),beam_count=5,frames_per_beam=len(starts),
                    maximum_unit_integral_error=worst_area,independent_raw_checks=comparisons)
        if 'background_audit_json' in h.attrs:
            audit=json.loads(h.attrs['background_audit_json'])
            assert max(v['maximum_core_sensitivity_to_top4_or_top12'] for v in audit)<0.002
            report['maximum_background_reference_sensitivity']=max(v['maximum_core_sensitivity_to_top4_or_top12'] for v in audit)
            report['signed_background_subtraction_verified']=True
        h.attrs['independent_validation_json']=json.dumps(report)
        h.attrs['independent_validation']='passed: signed raw IQ, NumPy FFT, pulse counts, Parseval, exact unit integral'
    # A range-independent narrow feature belongs in the background, while an
    # isolated contaminating gate must not bias the average reference.
    rng=np.random.default_rng(94)
    common=1+3*np.exp(-((np.arange(256)-122)/3)**2)
    trial=common[None,None,:]*(1+rng.normal(0,.01,(2,8,256)))
    trial[0,0,170:174]+=10
    estimate,used,_=quiet_average(trial,390.625)
    assert not used[0,0] and used[1].all()
    assert estimate[0,122]>3.8 and estimate[0,172]<1.1
    print(json.dumps(report,indent=2))


if __name__=='__main__':main()
