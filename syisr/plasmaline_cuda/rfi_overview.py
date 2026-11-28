"""Minute-by-minute raw broadband RFI indicators; no CUDA or plotting."""
import pathlib,h5py,numpy as np
base=pathlib.Path('/gpfs/pub/data_other/juha/plasmaline-20261125')
root=base/'dual-channel-hour';out=base/'rfi-diagnostics'/'overview.h5'
with h5py.File(out.with_suffix('.partial.h5'),'w') as dst:
    for center in (436000000,444000000):
        with h5py.File(root/'1795577400'/str(center)/'diagnostic.h5') as f:ref=f['passband_reference_adu2'][:];freq=f['frequency_offset_hz'][:]
        times=[];spectra=[];accepted=[];count=[]
        for d in sorted(root.iterdir()):
            if not d.name.isdigit():continue
            path=d/str(center)/'diagnostic.h5'
            if not path.exists():continue
            with h5py.File(path) as f:
                assert f.attrs['receiver_center_hz']==center
                p=f['raw_mean_power_adu2'][:];assert p.shape[1]==1024
                spectra.append(10*np.log10(np.maximum(p.mean(axis=0)/ref,1e-30)))
                times.append(float(f.attrs['integration_start_utc_s']));accepted.append(float(np.mean(f['pulse_accepted'][:])));count.append(int(f.attrs['pulses']))
        g=dst.create_group('rx%d'%(center//1000000));g['time_utc_s']=times;g['frequency_relative_430_hz']=center-430e6+freq
        g.create_dataset('raw_mean_spectrum_relative_quiet_db',data=np.asarray(spectra,dtype='f4'),compression='gzip',compression_opts=1)
        level=np.median(spectra,axis=1);g['broadband_level_db']=level;g['accepted_pulse_fraction']=accepted;g['recorded_pulses']=count
        g.attrs.update(receiver_center_hz=center,reference_start_utc_s=1795577400,definition='Mean raw STFT power over receive gates, divided by quiet-reference PSD; broadband level is median across frequency bins. Before decoding and RFI censoring.')
        t=np.asarray(times);pre=(t>=1795575600)&(t<1795579200);post=(t>=1795579200)&(t<1795582800)
        print('OVERVIEW',center,'minutes',len(times),'pre-noon median dB',float(np.median(level[pre])),'12-13 local median dB',float(np.median(level[post])),'maximum dB',float(np.max(level)),flush=True)
out.with_suffix('.partial.h5').replace(out)
