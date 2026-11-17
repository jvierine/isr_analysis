"""Common signed colour scales and separately labelled provisional range copies."""
import argparse
import hashlib
from pathlib import Path
import shutil
import h5py
import numpy as np


def main():
    p=argparse.ArgumentParser();p.add_argument('root',type=Path);q=p.parse_args()
    with h5py.File(q.root/'pulse-edge-audit.h5') as a:
        bounds=a['edge_summary'][:,-2:]
        assert np.all((bounds[:,0]>=143)&(bounds[:,1]<=173)),np.unique(bounds,axis=0)
    raw=[];norm=[];products=sorted(q.root.glob('session_*/range-doppler.h5'))
    for path in products:
        with h5py.File(path) as h:
            for b in range(5):
                g=h[f'beam_{b}'];n=g['pulse_count'].shape[0];ids=np.unique(np.linspace(0,n-1,min(n,80),dtype=int))
                r=h['slant_range_km'][:];keep=(r>=150)&(r<=1250)
                raw.append(abs(g['background_subtracted_psd_adu2_per_hz'][ids][:,keep,:]).ravel())
                values=g['normalized_signal_psd_per_hz'][ids][:,keep,:]*1e4
                mask=g['normalization_display_valid'][ids][:,keep].astype(bool)
                norm.append(abs(values[mask]).ravel())
    absolute=max(5.,np.ceil(np.quantile(np.concatenate(raw),.9975)/5)*5)
    normal=max(.5,np.ceil(np.quantile(np.concatenate(norm),.9975)*2)/2)
    for path in products:
        with h5py.File(path,'r+') as h:
            h.attrs['fixed_signed_absolute_limit']=absolute;h.attrs['fixed_signed_normalized_limit']=normal
            h.attrs['display_scale_method']='global 99.75 percentile magnitude of up to 80 uniformly sampled frames per session and all five beams; same limits across sessions'
        out=path.parent/'provisional_delay160us';out.mkdir(exist_ok=True);target=out/path.name
        shutil.copy2(path,target)
        with h5py.File(target,'r+') as h:
            r=h['slant_range_km'][:];h['nominal_uncorrected_slant_range_km']=r;h['slant_range_km'][:]=r-160*.149896229
            h.attrs['background_range_km']=np.asarray(h.attrs['background_range_km'])-160*.149896229
            h.attrs['range_delay_correction_us']=160.;h.attrs['range_delay_correction_km']=160*.149896229
            h.attrs['range_delay_status']='PROVISIONAL: 160 us from coherent pulse edge; assumes zero-range transmitter leakage; hardware timing reference unverified; not calibrated from ion-line narrowing'
            h.attrs['range_reference']='pulse-centred monostatic slant range, with provisional 160 us common timing-delay subtraction'
            h.attrs['parent_product']=str(path.resolve())
            h.attrs['timing_audit_sha256']=hashlib.sha256((q.root/'pulse-edge-audit.h5').read_bytes()).hexdigest()
    print('Common fixed signed display limits:',absolute,normal,flush=True)


if __name__=='__main__':main()
