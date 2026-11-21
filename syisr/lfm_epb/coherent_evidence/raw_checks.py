"""Audit digital scaling and event/quiet raw-voltage diagnostics.
No ADC rail specification is supplied, so absence of plateaus is not a hardware
saturation certification. Wrong-code comparison is a waveform diagnostic.
"""
import argparse,re
from pathlib import Path
from datetime import datetime,timezone,timedelta
import h5py,numpy as np
p=argparse.ArgumentParser();p.add_argument('parts');p.add_argument('output');a=p.parse_args()
beams=sorted(Path(a.parts).glob('beam_*'));targets=[1772363700.,1772369310.]
with h5py.File(a.output,'w') as out:
 out.attrs['targets_UTC']='2026-03-01T11:15:00Z quiet;2026-03-01T12:48:30Z event'
 out.attrs['ADC_limitation']='Input is already beamformed complex floating-point voltage. ADC rail/fullscale or per-element diagnostics unavailable. Exact extrema counts and quantization are descriptive, not a saturation test.'
 for beam in beams:
  g=out.create_group(beam.name)
  candidates=[]
  for file in beam.glob('*.h5'):
   match=re.search(r'raw(\d{12})_',file.name)
   if match:
    t=datetime.strptime(match[1],'%y%m%d%H%M%S').replace(tzinfo=timezone(timedelta(hours=8))).timestamp();candidates.append((t,file))
  for label,target in zip(['quiet','event'],targets):
   eligible=[(t,f) for t,f in candidates if t<=target]
   _,file=max(eligible,key=lambda x:x[0])
   with h5py.File(file) as h:
    ts=h['times_unix_s'][()];i=np.argmin(abs(ts-target));r=h['range_km'][()];z=h['altitude_km'][()];noise=h['noise_power'][()];rawtop=h['raw_topside_mean_power'][()]
    k=np.argmin(abs(z-323.6296100373802));correct=h['downchirp_power'][:,k];wrong=h['upchirp_power'][:,k]
    q=g.create_group(label);q.attrs['part_file']=str(file);src=h.attrs['source_file'];q.attrs['source_file']=src;q['para']=h['para'][()]
    q['pulse_time_unix_s']=ts;q['noise_power']=noise;q['raw_topside_mean_power']=rawtop;q['correct_code_power_at323km']=correct;q['wrong_code_power_at323km']=wrong
    q['altitude_km']=z[k];q['range_km']=r[k]
   with h5py.File(src) as h:
    d=h['data_raw'][()];real=d['real'];imag=d['imag'];x=real+1j*imag;pp=abs(x)**2
    maximum=max(abs(real).max(),abs(imag).max());extrema=[real.min(),real.max(),imag.min(),imag.max()]
    counts=[np.sum(real==extrema[0]),np.sum(real==extrema[1]),np.sum(imag==extrema[2]),np.sum(imag==extrema[3])]
    q['voltage_component_extrema']=extrema;q['exact_extrema_sample_counts']=counts;q['voltage_abs_max_per_pulse']=abs(x).max(axis=0);q['voltage_rms_per_pulse']=np.sqrt(pp.mean(axis=0));q['raw_sample_count']=real.size
    q['zero_fraction']=[np.mean(real==0),np.mean(imag==0)];q['integer_fraction']=[np.mean(real==np.round(real)),np.mean(imag==np.round(imag))]
   print('RAW',beam.name,label,'maxcomponent',maximum,'extrema_counts',counts,'noise',np.median(noise),'correct/wrong',np.median(correct)/np.median(wrong),flush=True)
print('RAW_CHECKS_COMPLETE',flush=True)
