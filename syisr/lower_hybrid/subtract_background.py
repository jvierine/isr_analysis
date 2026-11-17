"""Subtract a same-beam, same-window far-range spectrum without clipping.

The eight highest complete gates are averaged, including any common narrow
receiver/interference feature. Their correlations are not treated as eight
independent measurements. A separate empirical power-scatter guard controls
only the normalized display; all signed spectra remain in the HDF5 product.
"""
import argparse
import json
from pathlib import Path

import h5py
import numpy as np


def put(group,name,value):
    if name in group: del group[name]
    group.create_dataset(name,data=value,compression='gzip',compression_opts=3)


def quiet_average(spectra,df):
    """Average clean reference gates, rejecting localized positive excess.

    A common feature shared by all gates is deliberately retained as part of
    the background. Guards are conservative relative-excess diagnostics, not
    calibrated detection probabilities and not edits to the raw spectra.
    """
    nframe,ngate,nfreq=spectra.shape
    average=[];used=[];scatter=[]
    for frame,values in enumerate(spectra):
        power=values.sum(axis=1)*df
        median=np.median(power);mad=1.4826*np.median(abs(power-median))
        typical=np.median(values,axis=0)
        fractional=values/typical[None,:]-1
        keep=(power<=median+max(6*mad,0.03*median)) & (fractional.max(axis=1)<=0.30)
        required=min(4,max(2,ngate//2))
        assert keep.sum()>=required,('too few clean background gates',frame,ngate,int(keep.sum()),required)
        average.append(values[keep].mean(axis=0));used.append(keep)
        scatter.append(power[keep].std(ddof=1))
    return np.asarray(average),np.asarray(used),np.asarray(scatter)


def main():
    p=argparse.ArgumentParser();p.add_argument('spectra',type=Path);q=p.parse_args()
    with h5py.File(q.spectra,'r+') as h:
        r=h['slant_range_km'][:];f=h['frequency_hz'][:];df=f[1]-f[0]
        reference=np.arange(len(r)-8,len(r))
        report=[]
        for b in [b for b in range(5) if f'beam_{b}' in h]:
            g=h[f'beam_{b}'];raw=g['psd_adu2_per_hz'][:]
            unvetted=raw[:,reference,:].mean(axis=1)
            bg,used,scatter=quiet_average(raw[:,reference,:],df)
            signal=raw-bg[:,None,:]
            net=signal.sum(axis=2)*df
            scatter=np.maximum(scatter,np.median(scatter))
            threshold=3*scatter
            valid=net>threshold[:,None]
            valid[:,reference]=False
            normalized=np.full_like(signal,np.nan)
            positive=net>0
            normalized[positive]=signal[positive]/net[positive,None]
            bg4=raw[:,-4:,:].mean(axis=1)
            bg12=raw[:,-12:,:].mean(axis=1)
            bg4_clean,_,_=quiet_average(raw[:,-4:,:],df)
            bg12_clean,_,_=quiet_average(raw[:,-12:,:],df)
            reference_envelope=np.maximum(abs(bg-bg4_clean),abs(bg-bg12_clean))
            # Do not turn uncertain weak residuals into apparently equal-power
            # plasma features. This is an empirical display guard, not a
            # probabilistic error bar; every signed residual remains stored.
            valid&=signal.max(axis=2)>3*reference_envelope.max(axis=1)[:,None]
            put(g,'background_psd_adu2_per_hz',bg)
            put(g,'background_unvetted_top8_psd_adu2_per_hz',unvetted)
            put(g,'background_reference_gate_used',used.astype(np.uint8))
            put(g,'background_subtracted_psd_adu2_per_hz',signal)
            put(g,'background_subtracted_integrated_power_adu2',net)
            put(g,'normalized_signal_psd_per_hz',normalized)
            put(g,'normalization_display_valid',valid.astype(np.uint8))
            put(g,'background_power_scatter_adu2',scatter)
            put(g,'normalization_min_power_adu2',threshold)
            put(g,'background_control_difference_envelope_adu2_per_hz',reference_envelope)
            put(g,'background_top4_psd_adu2_per_hz',bg4)
            put(g,'background_top12_psd_adu2_per_hz',bg12)
            put(g,'background_top4_screened_psd_adu2_per_hz',bg4_clean)
            put(g,'background_top12_screened_psd_adu2_per_hz',bg12_clean)
            peak=int(np.argmax(bg.mean(axis=0)))
            stable=(r>=250)&(r<=800)
            peak_sig=np.max(np.abs(signal[:,stable,:]),axis=(1,2))
            sensitivity=np.maximum(np.max(abs(bg-bg4_clean),axis=1),np.max(abs(bg-bg12_clean),axis=1))/peak_sig
            row=dict(beam=b,background_peak_frequency_hz=float(f[peak]),
                     background_peak_to_median=float(bg.mean(axis=0)[peak]/np.median(bg)),
                     median_integrated_background_adu2=float(np.median(bg.sum(axis=1)*df)),
                     normalization_valid_fraction=float(valid.mean()),
                     background_reference_excluded_gates=int((~used).sum()),
                     maximum_core_sensitivity_to_top4_or_top12=float(sensitivity.max()))
            report.append(row)
            for i in range(len(signal)):
                assert np.allclose(signal[i,reference[used[i]],:].mean(axis=0),0,rtol=0,atol=1e-12)
            assert np.any(signal<0),'signed negative residuals must be retained'
            assert np.max(abs(normalized[valid].sum(axis=1)*df-1))<1e-10
        put(h,'background_reference_range_indices',reference)
        h.attrs['background_range_km']=[r[reference[0]],r[reference[-1]]]
        h.attrs['background_subtraction']=f"same-beam same-{float(h.attrs.get('integration_window_s',60.)):g}-s-window arithmetic mean of clean candidates among eight highest complete range spectra; frequency-dependent subtraction; signed residuals retained"
        h.attrs['background_reference_guard']='exclude candidate gate with integrated power > median + max(6 robust MAD, 3 percent of median), or any frequency bin > 1.30 times the per-frequency median across candidate gates; require at least four clean gates; common features retained as background'
        h.attrs['background_reference_correlation']='overlapping range windows; no division of scatter by sqrt(number of gates)'
        h.attrs['signal_normalization']='divide signed residual by its full-band signed frequency integral when positive; negative frequency bins retained'
        h.attrs['normalization_display_guard']='integrated residual > 3 * far-range integrated-power standard deviation, floored at its same-beam temporal median; positive spectral peak > 3 * maximum top4/top12 screened-reference difference from top8; reference gates masked; empirical display guards, not calibrated detection probabilities'
        h.attrs['background_audit_json']=json.dumps(report)
        print('Background reference centres',r[reference])
        print(json.dumps(report,indent=2))


if __name__=='__main__':main()
