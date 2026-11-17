"""Select the strongest screened frame and retain before/after review data."""
import argparse
from pathlib import Path
import h5py
import numpy as np


def main():
    p=argparse.ArgumentParser();p.add_argument('product',type=Path);p.add_argument('output',type=Path);q=p.parse_args()
    best=None;report=[]
    with h5py.File(q.product) as h:
        r=h['slant_range_km'][:];core=(r>=150)&(r<=1250)
        for b in range(5):
            g=h[f'beam_{b}'];accepted=g['accepted_pulse_count'][:];counts=g['pulse_count'][:]
            fraction=1-accepted/counts[:,None]
            report.append([b,float(fraction[:,core].mean()),float(fraction[:,core].max())])
            # Exposure-weighted integrated removed positive power is a review
            # selector, not a calibrated target statistic or detection claim.
            for frame in range(len(counts)):
                a=g['unscreened_psd_adu2_per_hz'][frame];z=g['psd_adu2_per_hz'][frame]
                score=float(np.maximum(a[core]-z[core],0).sum())
                if best is None or score>best[0]:best=(score,b,frame,a,z)
        score,b,frame,before,after=best;g=h[f'beam_{b}'];background=g['background_psd_adu2_per_hz'][frame]
        with h5py.File(q.output,'w') as out:
            out['before']=before-background;out['after']=after-background;out['difference']=before-after
            out['rejection_mask']=g['transient_rejected_time_range'][frame].T
            out['slant_range_km']=r;out['frequency_hz']=h['frequency_hz'][:];out['beam_geometry']=h['beam_geometry'][:]
            out['rejection_summary']=report
            out.attrs['beam']=b;out.attrs['frame']=frame;out.attrs['window_start_utc_s']=h['window_start_utc_s'][frame]
            out.attrs['window_end_utc_s']=h['window_end_utc_s'][frame]
            out.attrs['range_delay_correction_us']=h.attrs.get('range_delay_correction_us',0.)
            out.attrs['screen_status']='candidate satellite morphology; thermal estimates can be biased by censoring; unscreened spectra and full masks retained'
    print('Per-beam mean/max rejected exposure:',report,flush=True)
    print('Review selected beam/frame:',b,frame,flush=True)


if __name__=='__main__':main()
