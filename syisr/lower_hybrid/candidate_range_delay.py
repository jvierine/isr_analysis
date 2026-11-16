"""Create a separate, explicitly provisional leakage-referenced range product."""
import argparse
import hashlib
from pathlib import Path
import shutil

import h5py
import numpy as np


def main():
    p=argparse.ArgumentParser();p.add_argument('original',type=Path);p.add_argument('audit',type=Path);p.add_argument('output',type=Path);q=p.parse_args()
    assert q.output.resolve()!=q.original.resolve()
    with h5py.File(q.audit) as audit:
        delay=float(audit['pulse_edge'].attrs['candidate_correction_us'])
    assert delay==160.
    q.output.parent.mkdir(parents=True,exist_ok=True);shutil.copy2(q.original,q.output)
    distance=delay*0.149896229
    with h5py.File(q.output,'r+') as h,h5py.File(q.original) as old:
        nominal=h['slant_range_km'][:]
        h['nominal_uncorrected_slant_range_km']=nominal
        h['slant_range_km'][:]=nominal-distance
        h.attrs['range_delay_correction_us']=delay
        h.attrs['range_delay_correction_km']=distance
        h.attrs['range_delay_status']='PROVISIONAL: 160 us from coherent pulse falling-edge relative to commanded pulse end; assumes zero-range transmitter leakage; not fitted to ion-line narrowing; hardware timing/pointing calibration pending'
        h.attrs['range_reference']='pulse-centred monostatic slant range with additional provisional 160 us common timing-delay subtraction'
        h.attrs['background_range_km']=np.asarray(h.attrs['background_range_km'])-distance
        h.attrs['parent_product']=str(q.original.resolve());h.attrs['parent_product_sha256']=hashlib.sha256(q.original.read_bytes()).hexdigest()
        h.attrs['timing_audit_path']=str(q.audit.resolve())
        h.attrs['fixed_signed_absolute_limit']=15.
        h.attrs['fixed_signed_normalized_limit']=1.5
        for b in range(5):
            for name in ['psd_adu2_per_hz','background_subtracted_psd_adu2_per_hz','normalized_signal_psd_per_hz','normalization_display_valid']:
                np.testing.assert_array_equal(h[f'beam_{b}/{name}'][:],old[f'beam_{b}/{name}'][:])
        assert np.allclose(h['slant_range_km'][:]+distance,old['slant_range_km'][:],rtol=0,atol=1e-12)
        h.attrs['range_correction_source_sha256']=hashlib.sha256(Path(__file__).read_bytes()).hexdigest()
    print('Candidate range axis shifted down by',distance,'km; all spectral samples and masks unchanged')


if __name__=='__main__':main()
