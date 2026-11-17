"""Exposure-weighted long integrations from non-overlapping measured PSDs.

No averaging of normalized displays or already background-subtracted spectra.
Keep the original one-second masks. An additional conservative block-level
ambiguity screen can reject residual localized echoes before integration.
"""
import argparse
import hashlib
import json
from pathlib import Path
import subprocess
import h5py
import numpy as np
from scipy.ndimage import label


def ambiguity_template():
    w=np.hanning(48).astype(np.float32).astype(float)
    j=np.arange(48);dr=np.arange(-8,9);df=np.arange(-32,33)
    overlap=(j[None,:]*10>=(-5-dr[:,None]*120))&(j[None,:]*10<(-5-dr[:,None]*120+480))
    kernel=np.exp(2j*np.pi*df[:,None]*j[None,:]/256)
    return abs((overlap*w[None,:])@kernel.T)**2


def residual_echo_mask(psd, ranges, frequency, perpendicular_range):
    """Whole block/range exclusions, never selected Doppler bins.

    Conservative additional check of already screened five-minute spectra.
    Require temporal excess AND localized rectangular-Tx/Hann ambiguity shape.
    Protect the central perpendicular ion line and retain broad common features.
    """
    nt,nr,nf=psd.shape;assert nf==256
    reference=np.median(psd,axis=0);template=ambiguity_template()
    mask=np.zeros((nt,nr),dtype=bool);candidates=[]
    for t in range(nt):
        hot=psd[t]>8*np.maximum(reference,1e-30)
        components,n=label(hot)
        excess=np.maximum(psd[t]-reference,0)
        for number in range(1,n+1):
            rr,ff=np.where(components==number)
            if len(rr)<3 or rr.max()-rr.min()+1>16:continue
            peak=int(np.argmax(excess[rr,ff]));rp=int(rr[peak]);fp=int(ff[peak])
            if abs(ranges[rp]-perpendicular_range)<=8*abs(ranges[1]-ranges[0]) and abs(frequency[fp])<=5078.125:continue
            best=0.
            for centre in range(max(0,rp-2),min(nr,rp+3)):
                r0=max(0,centre-8);r1=min(nr,centre+9);f0=max(0,fp-32);f1=min(nf,fp+33)
                x=excess[r0:r1,f0:f1];y=template[r0-centre+8:r1-centre+8,f0-fp+32:f1-fp+32]
                denominator=np.linalg.norm(x)*np.linalg.norm(y)
                if denominator>0:best=max(best,float(np.sum(x*y)/denominator))
            if best<.7:continue
            # Block averages do not locate an echo within a 300 s block.
            # Reject that block's complete ambiguity footprint, not neighbours.
            lo=max(0,int(rr.min())-4);hi=min(nr,int(rr.max())+5)
            mask[t,lo:hi]=True
            candidates.append([t,lo,hi,float(frequency[fp]),best])
    return mask,np.asarray(candidates,dtype=float).reshape(-1,5)


def main():
    p=argparse.ArgumentParser();p.add_argument('source',type=Path);p.add_argument('output',type=Path)
    p.add_argument('--beam',type=int,default=4);p.add_argument('--window-s',type=float,default=1800.);p.add_argument('--step-s',type=float,default=300.);q=p.parse_args()
    q.output.parent.mkdir(parents=True,exist_ok=True)
    with h5py.File(q.source) as src,h5py.File(q.output,'w') as h:
        parent=float(src.attrs['integration_window_s']);assert q.window_s>=parent and q.step_s>=parent
        blocks=round(q.window_s/parent);assert abs(blocks*parent-q.window_s)<1e-6
        assert abs(round(q.step_s/parent)*parent-q.step_s)<1e-6
        starts=src['window_start_utc_s'][:];ends=src['window_end_utc_s'][:]
        new_starts=np.arange(starts[0],ends[-1]-q.window_s+1e-6,q.step_s);assert len(new_starts)
        indices=[]
        for start in new_starts:
            wanted=start+np.arange(blocks)*parent
            ids=np.searchsorted(starts,wanted-1e-5)
            assert np.all(ids<len(starts)) and np.allclose(starts[ids],wanted,rtol=0,atol=1e-5)
            assert np.allclose(ends[ids],wanted+parent,rtol=0,atol=1e-5)
            assert len(np.unique(ids))==blocks
            indices.append(ids)
        indices=np.asarray(indices)
        for k,v in src.attrs.items():h.attrs[k]=v
        for key in ['frequency_hz','slant_range_km','nominal_uncorrected_slant_range_km','range_window_start_sample','fast_window_hann','beam_geometry','source_paths','source_sha256']:
            if key in src:src.copy(key,h)
        h['window_start_utc_s']=new_starts;h['window_end_utc_s']=new_starts+q.window_s
        h['parent_frame_indices']=indices
        h.attrs['integration_window_s']=q.window_s;h.attrs['integration_step_s']=q.step_s
        h.attrs['aggregation_method']='Exposure-weighted mean of non-overlapping measured parent PSDs before background subtraction or display normalization; original one-second masks retained; additional block/range ambiguity-screen exclusions'
        h.attrs['time_averaging']=f'{q.window_s:g} s wall-clock windows, {q.step_s:g} s steps; incomplete trailing window omitted; no windows cross source-session boundaries'
        h.attrs['additional_screen']='Parent block PSD >8 times same-bin temporal median across integration; >=3 connected bins, <=16 range gates; excess cosine similarity >=0.7 to 480 us rectangular-Tx/48-sample Hann range-Doppler ambiguity; range footprint padded by four gates; only candidate block excluded across all frequencies; protect central |f|<=5078.125 Hz within eight gates of perpendicular crossing'
        h.attrs['screen_limitations']='Morphology candidates are not confirmed satellites; weak or persistent echoes may survive. Whole-block rejection sacrifices exposure and can reject matching physical transients; counts and original products retained.'
        h.attrs['parent_product']=str(q.source.resolve());h.attrs['parent_product_sha256']=hashlib.sha256(q.source.read_bytes()).hexdigest()
        h.attrs['aggregation_program']='syisr-rd-integrate';h.attrs['aggregation_version']='0.1.0'
        h.attrs['aggregation_git_commit']=subprocess.check_output(['git','rev-parse','HEAD'],cwd=Path(__file__).parent,text=True).strip()
        h.attrs['aggregation_source_sha256']=hashlib.sha256(Path(__file__).read_bytes()).hexdigest()
        chain=json.loads(src.attrs.get('processing_chain_json','[]'))
        chain.append(dict(program='syisr-rd-integrate',version='0.1.0',git_commit=h.attrs['aggregation_git_commit']))
        h.attrs['processing_chain_json']=json.dumps(chain)
        # Do not inherit validation of five-minute arrays as validation of these.
        for key in ['independent_validation','independent_validation_json','background_audit_json','background_subtraction']:
            if key in h.attrs:del h.attrs[key]
        g=src[f'beam_{q.beam}'];out=h.create_group(f'beam_{q.beam}')
        ranges=h['slant_range_km'][:];freq=h['frequency_hz'][:];df=freq[1]-freq[0]
        perpendicular=h['beam_geometry'][q.beam,3]
        bank={k:[] for k in ['psd_adu2_per_hz','unscreened_psd_adu2_per_hz','normalized_psd_per_hz','integrated_power_adu2','accepted_pulse_count','pulse_count','first_pulse_utc_s','last_pulse_utc_s','one_second_pulse_count','transient_rejected_time_range','additional_rejected_block_range','parent_accepted_pulse_count']}
        audit=out.create_group('additional_echo_candidates')
        for frame,ids in enumerate(indices):
            psd=g['psd_adu2_per_hz'][ids];counts=g['accepted_pulse_count'][ids];total=g['pulse_count'][ids]
            extra,candidates=residual_echo_mask(psd,ranges,freq,perpendicular)
            weights=np.where(extra,0,counts);accepted=weights.sum(axis=0)
            assert np.all(accepted>0),('all exposure rejected',frame)
            raw=(psd*weights[:,:,None]).sum(axis=0)/accepted[:,None]
            unscreened=(g['unscreened_psd_adu2_per_hz'][ids]*total[:,None,None]).sum(axis=0)/total.sum()
            mask=g['transient_rejected_time_range'][ids].astype(bool)|extra[:,None,:]
            mask=mask.reshape(-1,len(ranges));sec=g['one_second_pulse_count'][ids].reshape(-1)
            assert len(sec)==round(q.window_s)
            np.testing.assert_array_equal(np.where(mask,0,sec[:,None]).sum(axis=0),accepted)
            power=raw.sum(axis=1)*df
            values=dict(psd_adu2_per_hz=raw,unscreened_psd_adu2_per_hz=unscreened,normalized_psd_per_hz=raw/power[:,None],integrated_power_adu2=power,accepted_pulse_count=accepted,pulse_count=total.sum(),first_pulse_utc_s=g['first_pulse_utc_s'][ids[0]],last_pulse_utc_s=g['last_pulse_utc_s'][ids[-1]],one_second_pulse_count=sec,transient_rejected_time_range=mask.astype(np.uint8),additional_rejected_block_range=extra.astype(np.uint8),parent_accepted_pulse_count=counts.sum(axis=0))
            for k,v in values.items():bank[k].append(v)
            audit[f'frame_{frame:04}']=candidates
        for k,v in bank.items():out.create_dataset(k,data=np.asarray(v),compression='gzip',compression_opts=3)
        out['scheduled_beam_time_s']=np.asarray(bank['pulse_count'])*.01
        rejected=out['parent_accepted_pulse_count'][:]-out['accepted_pulse_count'][:]
        h.attrs['additional_rejected_range_gate_pulse_exposures']=int(rejected.sum())
        h.attrs['additional_rejected_fraction']=float(rejected.sum()/out['parent_accepted_pulse_count'][:].sum())
        print(q.output,'frames',len(indices),'additional range-pulse rejection fraction',h.attrs['additional_rejected_fraction'],flush=True)


if __name__=='__main__':main()
