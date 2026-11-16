"""Independent range-origin/ambiguity audit; never align data by assumption.

The median spectral-narrowing location is an empirical diagnostic. It is not
an independent calibration of transmitter timing, receiver group delay or
beam pointing. Raw voltages and the original movie products are untouched.
"""
import argparse
from datetime import datetime,timezone
import hashlib
from pathlib import Path

import h5py
import numpy as np

from index_recording import basis,ecef,geod,field


def main():
    p=argparse.ArgumentParser();p.add_argument('index',type=Path);p.add_argument('spectra',type=Path);p.add_argument('output',type=Path)
    q=p.parse_args()
    with h5py.File(q.index) as ix,h5py.File(q.spectra) as h,h5py.File(q.output,'w') as out:
        t=ix['time_utc_s'][:];r=h['slant_range_km'][:];freq=h['frequency_hz'][:]
        starts=h['range_window_start_sample'][:].astype(int);weights=h['fast_window_hann'][:].astype(float)
        fs=float(ix.attrs['sample_rate_hz']);dt_us=1e6/fs;tp=float(ix.attrs['pulse_width_us'])
        raw=Path(ix.attrs['raw_path']);header=raw.open('rb').read(1024)
        u16=lambda j:int.from_bytes(header[j:j+2],'little')
        gate_front=u16(766);tx_front=u16(872);tx_end=u16(874)
        recorded_complex=(int(ix.attrs['record_stride_bytes'])-1024)//8
        assert gate_front==tx_front
        assert abs((tx_end-tx_front)-recorded_complex*dt_us)<1e-12
        mu=np.dot(np.arange(len(weights))*dt_us,weights**2)/np.sum(weights**2)
        calculated=(gate_front+starts*dt_us+mu-tp/2)*0.149896229
        assert np.max(abs(calculated-r))<1e-10
        out['slant_range_km']=r;out['window_start_sample']=starts;out['beam_geometry']=h['beam_geometry'][:]
        out['sample_delay_us']=gate_front+np.arange(int(ix.attrs['sample_count']))*dt_us
        out.attrs['header_gate_front_us']=gate_front;out.attrs['header_transmit_site_receive_gate_front_us']=tx_front
        out.attrs['header_transmit_site_receive_gate_end_us']=tx_end;out.attrs['recorded_complex_samples_including_extra']=recorded_complex
        out.attrs['gate_span_us']=tx_end-tx_front;out.attrs['hann_power_centroid_us']=mu
        out.attrs['transmit_pulse_centroid_us']=tp/2;out.attrs['range_grid_step_km']=r[1]-r[0]
        out.attrs['header_assumption']='receive gate delay is measured from transmitted pulse leading edge; unverified absolute receiver group delay'
        # Continuous rectangular-pulse point response. This independently
        # integrates the window weights over a dense physical-range grid.
        dense=np.linspace(0,1500,15001)
        tau=2*dense/299792.458*1e6
        time=gate_front+starts[:,None]*dt_us+np.arange(len(weights))[None,:]*dt_us
        kernel=np.zeros((len(r),len(dense)))
        for j in range(len(r)):
            inside=(time[j,:,None]>=tau[None,:])&(time[j,:,None]<tau[None,:]+tp)
            kernel[j]=(weights[:,None]**2*inside).sum(axis=0)
        areas=np.trapezoid(kernel,dense,axis=1)
        centroids=np.trapezoid(kernel*dense[None,:],dense,axis=1)/areas
        assert np.max(abs(centroids-r))<0.11
        out['physical_range_km']=dense
        out['range_power_kernels']=kernel/areas[:,None]
        out['kernel_centroid_range_km']=centroids
        # Inject a point scatterer at each nominal perpendicular location.
        # Sum every frequency via Parseval, and also inspect a coherent tone
        # at its exact FFT frequency. Neither test uses the output range label
        # to place the synthetic scatterer.
        expected=h['beam_geometry'][:,3]
        point=[];coherent=[];summaries=[]
        for truth in expected:
            delay=2*truth/299792.458*1e6
            mask=(time>=delay)&(time<delay+tp)
            power=np.sum(mask*weights[None,:]**2,axis=1)
            tone=np.sum(mask*weights[None,:],axis=1)**2
            peak=r[np.argmax(power)]
            assert abs(peak-truth)<(r[1]-r[0])/2+1.
            point.append(power/power.max());coherent.append(tone/tone.max())
            summaries.append([truth,peak,np.sum(r*power)/power.sum(),np.sum(r*tone)/tone.sum()])
        out['injected_point_power_response']=point;out['injected_coherent_tone_response']=coherent
        out['injection_summary']=summaries
        out.attrs['injection_columns']='true_range_km,peak_gate_range_km,power_response_centroid_km,coherent_tone_response_centroid_km'
        # Median across windows suppresses localized strong targets. The ratio
        # uses signed background-subtracted estimates; negative bins are not
        # clipped. The interval around each nominal crossing is predeclared.
        near=abs(freq)<=1562.5;wide=abs(freq)<=8000
        profiles=[];summary=[];discrete=[];widths=[]
        for b,target in enumerate(expected):
            g=h[f'beam_{b}'];spectra=g['background_subtracted_psd_adu2_per_hz'][:]
            median=np.median(spectra,axis=0)
            fraction=median[:,near].sum(axis=1)/median[:,wide].sum(axis=1)
            candidates=np.where((abs(r-target)<100)&(r>230))[0]
            j=candidates[np.argmax(fraction[candidates])]
            assert 0<j<len(r)-1
            x=r[j-1:j+2]-r[j];coeff=np.polyfit(x,fraction[j-1:j+2],2)
            assert coeff[0]<0
            estimate=r[j]-coeff[1]/(2*coeff[0])
            profiles.append(fraction);discrete.append(r[j]);summary.append([target,estimate,estimate-target,(estimate-target)/0.149896229])
            widths.append(coeff)
        out['central_fraction_profile']=profiles;out['narrowing_summary']=summary
        out['narrowest_discrete_gate_range_km']=discrete
        out['local_quadratic_coefficients']=widths
        out.attrs['central_fraction_definition']='signed PSD integrated over |f|<=1562.5 Hz divided by |f|<=8000 Hz; median PSD over 38 overlapping windows; local maximum within nominal range +/-100 km and range>230 km; 3-gate quadratic interpolation'
        out.attrs['narrowing_columns']='nominal_perp_range_km,empirical_narrowing_peak_km,range_difference_km,equivalent_two_way_delay_us'
        out.attrs['empirical_offset_status']='diagnostic association only; not a hardware/pointing calibration; do not shift raw ranges to force alignment'
        offsets=np.asarray(summary)[:,2]
        out.attrs['median_empirical_offset_km']=np.median(offsets)
        out.attrs['median_equivalent_delay_us']=np.median(offsets)/0.149896229
        # Prepare independently evaluated WGS84 targets for native Rust IGRF.
        site_lat,site_lon=ix.attrs['site_lat_lon_deg'];site=ecef(site_lat,site_lon,0);enu=basis(site_lat,site_lon)
        date=datetime.fromtimestamp(t[0],timezone.utc).replace(tzinfo=None)
        independent=[]
        for row in h['beam_geometry'][:]:
            az,el=np.deg2rad(row[1:3]);u=np.array([np.sin(az)*np.cos(el),np.cos(az)*np.cos(el),np.sin(el)])@enu
            x=site+row[3]*u;lat,lon,height=geod(x)
            independent.append([lat,lon,height,*field(x,date)])
        out['target_geodetic_and_field_ecef_nT']=independent
        out['site_lat_lon_deg']=[site_lat,site_lon]
        out.attrs['geometry_epoch_utc']=date.isoformat()+'Z'
        out.attrs['index_path']=str(q.index.resolve());out.attrs['spectra_path']=str(q.spectra.resolve())
        out.attrs['audit_script_sha256']=hashlib.sha256(Path(__file__).read_bytes()).hexdigest()
        print('Gate span',tx_end-tx_front,'us =',recorded_complex,'samples x',dt_us,'us')
        print('Receiver Hann centroid',mu,'us, transmit centroid',tp/2,'us; net',mu-tp/2,'us')
        print('Maximum numerical range-grid discrepancy km',np.max(abs(calculated-r)))
        print('Maximum impulse-kernel centroid discrepancy km',np.max(abs(centroids-r)))
        print('Injection columns:',out.attrs['injection_columns']);print(np.asarray(summaries))
        print('Narrowing columns:',out.attrs['narrowing_columns']);print(np.asarray(summary))
        print('Median diagnostic offset km/us',np.median(offsets),np.median(offsets)/0.149896229)


if __name__=='__main__':main()
