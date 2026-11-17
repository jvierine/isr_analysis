"""Independent NumPy/raw-IQ check of Rust spectra and all normalization rows."""
import argparse
import json
from pathlib import Path

import h5py
import numpy as np
from subtract_background import quiet_average


def main():
    parser=argparse.ArgumentParser();parser.add_argument('index',type=Path);parser.add_argument('spectra',type=Path)
    parser.add_argument('--extra-only',action='store_true',help='check the strongest newly rejected block/range case and retain previous checks')
    q=parser.parse_args()
    report={}
    with h5py.File(q.index) as ix,h5py.File(q.spectra,'r+') as h:
        t=ix['time_utc_s'][:];beam=ix['beam_index'][:];starts=h['window_start_utc_s'][:];ends=h['window_end_utc_s'][:]
        duration=min(float(h.attrs.get("integration_window_s",60.)),t[-1]+0.01-t[0])
        assert len(starts)>=1 and np.allclose(ends-starts,duration,rtol=0,atol=1e-8)
        assert np.all(np.diff(starts)>0) and np.all(np.diff(starts)<=float(h.attrs.get("integration_step_s",30.))+1e-6)
        assert starts[0]==t[0]
        if 'aggregation_method' in h.attrs:
            assert 0<=t[-1]+0.01-ends[-1]<float(h.attrs['integration_step_s'])+1e-6
        else:assert ends[-1]>=t[-1]
        assert np.all(np.diff(t)>0)
        paths=ix['compute_cache_paths'].asstr()[:] if 'compute_cache_paths' in ix else ix['source_paths'].asstr()[:] if 'source_paths' in ix else [ix.attrs['raw_path']]
        maps=[np.memmap(path,mode='r',dtype=np.uint8) for path in paths]
        file_id=ix['source_file_index'][:] if 'source_file_index' in ix else np.zeros(len(t),dtype=np.uint16)
        local_id=ix['source_record_index'][:] if 'source_record_index' in ix else np.arange(len(t))
        stride=int(ix.attrs['record_stride_bytes']);n=int(ix.attrs['sample_count'])
        # Explicit signed int32 layout, independent of the Rust IQ decoder.
        samples=[np.ndarray((len(mm)//stride,n,2),dtype='<i4',buffer=mm,offset=1024,strides=(stride,8,4)) for mm in maps]
        w=h['fast_window_hann'][:].astype(float);nf=len(h['frequency_hz']);df=h.attrs['frequency_bin_hz'];fs=h.attrs['sample_rate_hz']
        gate=h['range_window_start_sample'][:].astype(int)
        assert len(w)==48 and nf==256 and len(gate)==73 and df==390.625
        expected=(533+(gate+(len(w)-1)/2)*10-240)*0.149896229
        expected-=float(h.attrs.get('range_delay_correction_us',0))*0.149896229
        np.testing.assert_array_equal(expected,h['slant_range_km'][:])
        geometry=h['beam_geometry'][:]
        assert list(geometry[:,0])==[250,350,450,550,650]
        assert np.max(abs(geometry[:,-1]))<1e-7
        worst_area=0.
        selected=[b for b in range(5) if f'beam_{b}' in h]
        assert selected
        for b in selected:
            g=h[f'beam_{b}'];raw=g['psd_adu2_per_hz'][:];norm=g['normalized_psd_per_hz'][:]
            assert np.isfinite(raw).all() and (raw>=0).all() and np.isfinite(norm).all()
            area=raw.sum(axis=-1)*df
            np.testing.assert_allclose(area,g['integrated_power_adu2'][:],rtol=2e-14)
            np.testing.assert_allclose(norm,raw/area[:,:,None],rtol=2e-14)
            deviation=float(np.max(abs(norm.sum(axis=-1)*df-1)))
            worst_area=max(worst_area,deviation);assert deviation<1e-12
            prefix=np.r_[0,np.cumsum(beam==b)]
            counts=prefix[np.searchsorted(t,ends)]-prefix[np.searchsorted(t,starts)]
            np.testing.assert_array_equal(counts,g['pulse_count'][:])
            if 'accepted_pulse_count' in g:
                mask=g['transient_rejected_time_range'][:].astype(bool);sec_count=g['one_second_pulse_count'][:]
                np.testing.assert_array_equal(sec_count.sum(axis=1),counts)
                expected_count=np.sum(np.where(mask,0,sec_count[:,:,None]),axis=1)
                np.testing.assert_array_equal(expected_count,g['accepted_pulse_count'][:])
                assert np.all(expected_count>0) and np.all(expected_count<=counts[:,None])
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
                if 'background_control_difference_envelope_adu2_per_hz' in g:
                    envelope=np.maximum(abs(background-g['background_top4_screened_psd_adu2_per_hz'][:]),abs(background-g['background_top12_screened_psd_adu2_per_hz'][:]))
                    np.testing.assert_array_equal(envelope,g['background_control_difference_envelope_adu2_per_hz'][:])
                    assert np.all(residual.max(axis=2)[visible]>np.broadcast_to(3*envelope.max(axis=1)[:,None],visible.shape)[visible])
                np.testing.assert_allclose(snorm[visible].sum(axis=-1)*df,1.,rtol=0,atol=1e-10)
                assert np.isnan(snorm[power<=0]).all()
                assert not visible[:,reference].any()
        comparisons=[]
        checks=[] if q.extra_only else [(b,frame,gate) for b in selected for frame,gate in [(0,10),(len(starts)//2,30),(len(starts)-1,45)]]
        for b in selected:
            g=h[f'beam_{b}']
            if 'additional_rejected_block_range' in g:
                rejected=g['additional_rejected_block_range'][:].sum(axis=1)
                if rejected.max()>0:
                    frame,selected_gate=np.unravel_index(np.argmax(rejected),rejected.shape)
                    checks.append((b,int(frame),int(selected_gate)))
        assert checks,'No additional rejected exposure to check'
        for boundary in ([] if q.extra_only else np.flatnonzero(np.diff(file_id)!=0)+1):
            frame=int(np.searchsorted(starts,t[boundary],side='right')-1)
            if starts[0]<=t[boundary]<ends[-1]:checks.append((int(beam[boundary]) if int(beam[boundary]) in selected else selected[0],max(0,frame),25))
        for b,frame,r in checks:
            p=np.flatnonzero((t>=starts[frame])&(t<ends[frame])&(beam==b))
            block=np.stack([samples[int(file_id[j])][int(local_id[j]),gate[r]:gate[r]+48,:] for j in p])
            z=block[:,:,1].astype(float)+1j*block[:,:,0].astype(float)
            spectrum=np.mean(abs(np.fft.fftshift(np.fft.fft(z*w[None,:],nf,axis=1),axes=1))**2,axis=0)/(fs*np.sum(w*w))
            g=h[f'beam_{b}'];stored=g['psd_adu2_per_hz'][frame,r,:]
            if 'transient_rejected_time_range' in g:
                np.testing.assert_allclose(g['unscreened_psd_adu2_per_hz'][frame,r,:],spectrum,rtol=2e-6,atol=1e-12)
                seconds=np.minimum(np.floor(t[p]-starts[frame]).astype(int),g['transient_rejected_time_range'].shape[1]-1)
                keep=g['transient_rejected_time_range'][frame,:,r][seconds]==0
                assert keep.sum()==g['accepted_pulse_count'][frame,r]
                z=z[keep]
                spectrum=np.mean(abs(np.fft.fftshift(np.fft.fft(z*w[None,:],nf,axis=1),axes=1))**2,axis=0)/(fs*np.sum(w*w))
            difference=float(np.max(abs(stored-spectrum))/np.max(abs(spectrum)))
            assert difference<2e-6,(b,frame,r,difference)
            # Parseval against original signed samples, not another FFT norm.
            integrated=np.mean(np.sum(abs(z*w)**2,axis=1)/np.sum(w*w))
            np.testing.assert_allclose(stored.sum()*df,integrated,rtol=2e-6)
            comparisons.append(dict(beam=b,frame=frame,range_gate=r,max_relative_difference=difference))
        report=dict(records=len(t),beam_count=len(selected),frames_per_beam=len(starts),
                    maximum_unit_integral_error=worst_area,independent_raw_checks=comparisons)
        if q.extra_only and 'independent_validation_json' in h.attrs:
            previous=json.loads(h.attrs['independent_validation_json'])
            report['independent_raw_checks']=previous['independent_raw_checks']+comparisons
            report['additional_rejection_raw_iq_check']=comparisons
        if 'background_audit_json' in h.attrs:
            audit=json.loads(h.attrs['background_audit_json'])
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
