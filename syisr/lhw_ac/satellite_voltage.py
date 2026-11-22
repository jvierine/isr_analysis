"""Doppler-compensated, code-matched raw-voltage satellite delay audit.

No time shift is assumed. Search +/-2 ms about predicted light travel time.
Both complex Doppler signs are searched; detections need repeated track support.
Range correlation uses a full 64-sample pulse, labelled by its leading edge.
"""
import argparse,json,os
from pathlib import Path
from datetime import datetime,timezone
import numpy as np
import h5py
from scipy.signal import correlate
from sgp4.api import Satrec
from satellite_crossings import observe

p=argparse.ArgumentParser();p.add_argument('root');p.add_argument('--beam',type=int,required=True);p.add_argument('--count',type=int,default=30);p.add_argument('--step',type=float,default=.4);p.add_argument('--half-window',type=float,default=20.);p.add_argument('--output-prefix',default='voltage-beam');q=p.parse_args()
root=Path(q.root).resolve();b=q.beam;outdir=root/'satellite-calibration';outdir.mkdir(exist_ok=True)
index=root.parent/'syisr-lhw-ac-20261119-15m-step5m-v01/index.h5'
with h5py.File(index) as h:
    bi=h['beam_index'][:];sel=np.flatnonzero(bi==b);bounds=h['cycle_time_utc_s'][:][sel]
    times=(bounds[:,0,None]+(bounds[:,1]-bounds[:,0])[:,None]*np.arange(64)[None,:]/63).reshape(-1)
    offsets=h['frame_offset'][:][sel].reshape(-1);files=h['file_index'][:][sel].reshape(-1)
    paths=json.loads(bytes(h['files_json_utf8'][:]))
codes=np.array([[1 if c=='+' else -1 for c in line.strip()] for line in (root/'source/syisr/m32-reference.bc').read_text().splitlines() if line.strip()])
with h5py.File(outdir/'crossings.h5') as h:
    rows=h['crossings'][:];idx=np.flatnonzero(rows['beam_index']==b)[:q.count]
    rows=rows[idx];lines1=h['tle_line1'].asstr()[:][idx];lines2=h['tle_line2'].asstr()[:][idx]
maps={};summary=[]
with h5py.File(outdir/f'{q.output_prefix}-{150+50*b}.h5','w') as out:
    out.attrs.update(beam_index=b,search_delay_us=[-2000,2000],method='Full-pulse leading-edge matched filter; predicted monostatic Doppler both signs plus -2,-1,0,1,2 kHz; raw I+iQ; no range correction',git_commit=os.environ.get('SOURCE_GIT_COMMIT','working'))
    for rank,(row,l1,l2) in enumerate(zip(rows,lines1,lines2)):
        t0=row['utc_s'];sat=Satrec.twoline2rv(l1,l2)
        wanted=np.arange(t0-q.half_window,t0+q.half_window+q.step*.01,q.step);wanted=wanted[(wanted>=times[0])&(wanted<=times[-1])]
        ii=np.searchsorted(times,wanted);ii=np.minimum(ii,len(times)-1)
        prev=np.maximum(ii-1,0);ii=np.where(abs(times[prev]-wanted)<abs(times[ii]-wanted),prev,ii);ii=np.unique(ii)
        t=times[ii];angle,range_km,rate,az,el=observe(sat,t)
        predicted=range_km/.149896229
        peakdelay=[];peakratio=[];peakfreq=[];planes=[];lags=None;measured=[];residual=[];coherence=[]
        for k,j in enumerate(ii):
            fi=int(files[j]);offset=int(offsets[j])
            if fi not in maps:maps[fi]=np.memmap(paths[fi],mode='r',dtype='u1')
            head=bytes(maps[fi][offset:offset+1024]);n=int.from_bytes(head[768:772],'little')+1
            clock=datetime(2000+head[13],head[12],head[15],head[14],head[17],head[16],tzinfo=timezone.utc).timestamp()+int.from_bytes(head[18:20],'little')*25e-6-28800
            assert abs(clock-t[k])<3e-5, (clock,t[k])
            assert int.from_bytes(head[856:858],'little')*360/65536==357.9949951171875
            codeid=int.from_bytes(head[942:944],'little');gate=int.from_bytes(head[766:768],'little')
            v=np.frombuffer(maps[fi][offset+1024:offset+1024+n*8],'<i4').reshape(n,2)
            x=v[:,1].astype(float)+1j*v[:,0];x[:32]=0 # omit known early clutter, no clock inference
            template=np.repeat(codes[codeid],2)
            frequency=(430+head[32]*.2)*1e6;fd=-2*rate[k]*1000*frequency/299792458.
            fbank=np.unique(np.r_[fd+np.arange(-2000,2001,1000),-fd+np.arange(-2000,2001,1000)])
            lags=gate+10*np.arange(n-len(template)+1)
            plane=np.zeros(len(lags));freqbest=np.zeros(len(lags))
            for f in fbank:
                c=correlate(x*np.exp(-2j*np.pi*f*np.arange(n)/1e5),template,mode='valid',method='fft')
                power=abs(c)**2;floor=np.median(power[64:]);score=power/max(floor,1e-30)
                new=score>plane;plane[new]=score[new];freqbest[new]=f
            allowed=(abs(lags-predicted[k])<=2000)&(np.arange(len(lags))>=32)
            pick=np.argmax(np.where(allowed,plane,-np.inf));peakdelay.append(lags[pick]-predicted[k]);peakratio.append(plane[pick]);peakfreq.append(freqbest[pick]);planes.append(plane)
            segment=x[pick:pick+64]*template
            spec=abs(np.fft.fft(segment,n=4096))**2;ff=np.fft.fftfreq(4096,1e-5)
            imax=np.argmax(spec);fm=ff[imax];measured.append(fm)
            residual.append(min(abs(fm-fd),abs(fm+fd)))
            coherence.append(spec[imax]/max(64*np.sum(abs(segment)**2),1e-30))
        d=np.array(peakdelay);s=np.array(peakratio);angle=angle[:,b]
        good=(s>30)&(angle<2.)&(np.array(residual)<500)&(np.array(coherence)>.15)
        # Find a repeated delay mode before quoting its median; no fixed 640 us prior.
        centre=np.nan;scatter=np.nan;accepted=np.zeros(len(d),bool)
        if good.sum()>=3:
            support=np.array([np.sum(good&(abs(d-v)<30)) for v in d]);best=np.argmax(support)
            accepted=good&(abs(d-d[best])<30)
            if accepted.sum()>=3:centre=np.median(d[accepted]);scatter=1.4826*np.median(abs(d[accepted]-centre))
        summary.append([rank,row['norad_id'],row['utc_s'],row['separation_deg'],centre,scatter,accepted.sum(),s.max()])
        g=out.create_group(f'pass_{rank:03d}');g.attrs.update(norad_id=row['norad_id'],closest_utc_s=t0,closest_angle_deg=row['separation_deg'])
        for key,value in [('utc_s',t),('predicted_range_km',range_km),('predicted_range_rate_km_s',rate),('separation_deg',angle),('recorded_leading_edge_delay_us',lags),('peak_excess_delay_us',d),('peak_ratio',s),('peak_doppler_hz',peakfreq),('measured_doppler_hz',measured),('doppler_absolute_sign_residual_hz',residual),('despread_coherent_fraction',coherence),('accepted_pulses',accepted.astype('u1'))]:g[key]=value
        g.create_dataset('matched_filter_power_to_median',data=np.array(planes,dtype='f4'),compression='gzip')
        print('BEAM',150+50*b,'rank',rank,'NORAD',int(row['norad_id']),'angle',round(row['separation_deg'],4),'delay',centre,'MAD',scatter,'pulses',accepted.sum(),'max ratio',round(s.max(),1),flush=True)
        out.flush()
    out['summary']=summary
    out.attrs['summary_columns']='rank,norad_id,utc_s,separation_deg,excess_delay_us,delay_MAD_us,accepted_pulses,max_power_to_median'
