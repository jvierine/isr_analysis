"""Save exact pulse exclusions and surviving code windows for raw RFI examples."""
import pathlib,json
import h5py,numpy as np
from pulse_mask import detect,accepted_windows,guarded_clean_runs,intervals
root=pathlib.Path('/gpfs/pub/data_other/juha/plasmaline-20261125/rfi-diagnostics');dest=root/'masks';dest.mkdir(exist_ok=True)
for path in root.glob('*.h5'):
    with h5py.File(path) as f:
        if 'power_2us_adu2' not in f:continue
        powers=f['power_2us_adu2'][:];rx=f['receiver_center_hz'][:];selected=f['selected_record_indices'][:]
        info=json.loads(f['ipp0'].attrs['header_json']);ne=info['n'];nt=round(info['pulse']*16);step=32
        ranges=(info['front']+np.arange((ne-nt)//step+1)*2)*.149896229
        intrinsic=accepted_windows(np.zeros(powers.shape[1],bool),ne,nt,step)
        view=intrinsic&(ranges<=500);masks=[];bad_masks=[];floors=[];raw_runs=[]
        for i,p in enumerate(powers):
            bad,floor=detect(p);good=accepted_windows(bad,ne,nt,step)
            masks.append(good);bad_masks.append(bad);floors.append(floor)
            for l,r in guarded_clean_runs(bad,ne):raw_runs.append((i,int(l),int(r)))
        masks=np.asarray(masks);bad_masks=np.asarray(bad_masks)
        out=dest/(path.stem+'-mask.h5')
        with h5py.File(out,'w') as g:
            g['pulse_time_utc_s']=f['pulse_time_utc_s'][:];g['receiver_center_hz']=rx;g['slant_range_km']=ranges;g['noise_floor_2us_adu2']=floors
            g.create_dataset('bad_raw_bins_packed',data=np.packbits(bad_masks,axis=1),compression='gzip')
            g.create_dataset('accepted_range_mask_packed',data=np.packbits(masks,axis=1),compression='gzip')
            g['clean_raw_segments_record_start_stop_sample']=np.asarray(raw_runs,dtype='i8').reshape(-1,3)
            for j,i in enumerate(selected):
                h=g.create_group('ipp%d'%j);h['accepted_range_mask']=masks[i];h['detected_raw_pulse_bins']=bad_masks[i]
                h['clean_raw_segments_sample']=guarded_clean_runs(bad_masks[i],ne)
                h['original_voltage']=h5py.ExternalLink(str(path),'/ipp%d/qi_int32'%j)
            for center in (436e6,444e6):
                keep=masks[rx==center].sum(axis=0);g['accepted_counts_rx%d'%(center/1e6)]=keep
                print(path.stem,'rx',center/1e6,'clean decode-window fraction <=500 km',float(np.mean(masks[rx==center][:,view])),'records',int(np.sum(rx==center)),flush=True)
            g.attrs.update(raw_source=str(path),threshold_power_ratio=4.,pulse_guard_us=16.,code_support_us=info['pulse'],range_step_us=2.,raw_bin_samples=32,raw_samples=ne,definition='Raw-time masks and full-code support only; no whitening is applied in this diagnostic. Original voltage is unchanged. Processing filters require their own additional support guard.')
