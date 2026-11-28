"""CPU-only raw-IPP selection and exact source ledgers; plotting/FFTs use Rust."""
import pathlib,json
import h5py,numpy as np
from process import catalogue,layout,frames
base=pathlib.Path('/gpfs/pub/data_other/juha/plasmaline-20261125')
raw=pathlib.Path('/gpfs/pub/data/L0/2026/plasmaline/20261125')
out=base/'rfi-diagnostics';out.mkdir(exist_ok=True)
index=catalogue(raw,base/'catalogue.h5');ref=layout(index[0][1]);ref.pop('receiver_center_hz',None)
reference={}
for center in (436000000,444000000):
    with h5py.File(base/'dual-channel-hour'/'1795577400'/str(center)/'diagnostic.h5') as f:
        reference[center]=(f['frequency_offset_hz'][:],f['passband_reference_adu2'][:])
        # 03:30 UTC reference, before the reported local-noon onset.
cases=[('before_noon',1795578600),('after_noon',1795579800),('early_0619',1795587542),('strong_0619',1795587582),('late_0900',1795597200)]
for tag,start in cases:
    path=out/(tag+'.h5')
    if path.exists():continue
    for shift in (0,10,20,30,40,50):
        t=start+shift;stop=t+1
        paths=[p for ft,p in index if ft<stop and ft+(p.stat().st_size//ref['stride'])*.01>=t]
        records=list(frames(paths,t,stop,ref,[]))
        if len(records)>=10:break
    if not records:print('NO VALID RAW IPPS',tag,start,flush=True);continue
    energies=[];envelopes=[]
    for h,qi,*_ in records:
        power=np.square(qi.astype('f8')).sum(axis=1)
        energies.append(power.mean());envelopes.append(power[:len(power)//32*32].reshape(-1,32).mean(axis=1).astype('f4'))
    selected=[]
    ch444=[i for i,r in enumerate(records) if r[0]['receiver_center_hz']==444e6]
    if not ch444:raise ValueError('missing 444 MHz channel')
    if tag=='before_noon':chosen=sorted(ch444,key=lambda i:energies[i])[len(ch444)//2]
    else:chosen=max(ch444,key=lambda i:float(np.max(envelopes[i])))
    selected.append(chosen)
    ch436=[i for i,r in enumerate(records) if r[0]['receiver_center_hz']==436e6]
    if ch436:selected.append(min(ch436,key=lambda i:abs(records[i][0]['time']-records[chosen][0]['time'])))
    tmp=path.with_suffix('.partial.h5')
    with h5py.File(tmp,'w') as f:
        f['pulse_time_utc_s']=[r[0]['time'] for r in records];f['receiver_center_hz']=[r[0]['receiver_center_hz'] for r in records]
        f['mean_adc_power_adu2']=energies;f.create_dataset('power_2us_adu2',data=np.asarray(envelopes),compression='gzip',compression_opts=1,shuffle=True)
        f['receive_delay_us']=ref['front']+(np.arange(len(envelopes[0]))*32+15.5)/16
        f['selected_record_indices']=selected
        for j,i in enumerate(selected):
            h,qi,name,offset,header=records[i];g=f.create_group('ipp%d'%j)
            g.create_dataset('qi_int32',data=qi,compression='gzip',compression_opts=1,shuffle=True)
            g['raw_header']=np.frombuffer(header,dtype='u1');ff,pp=reference[int(h['receiver_center_hz'])];g['reference_frequency_hz']=ff;g['reference_passband_adu2']=pp
            g.attrs.update(header_json=json.dumps(h),source_filename=name,source_byte_offset=offset,pulse_time_utc_s=h['time'],receiver_center_hz=h['receiver_center_hz'],front_us=h['front'],sample_rate_hz=h['fs'],phase=h['phase'])
        f.attrs.update(case=tag,requested_start_utc_s=start,actual_start_utc_s=t,reference_start_utc_s=1795577400,decoded=False,filtered=False,header_clock_offset_to_utc_s=-28800)
    tmp.replace(path);print('EXTRACTED',tag,'records',len(records),'selected',[(records[i][0]['time'],records[i][0]['receiver_center_hz'],float(np.max(envelopes[i])))for i in selected],flush=True)
