"""Check all recorded beam identifiers/angles and quantify a run-level change.

Commanded/header angles are not a surveyed antenna-pattern calibration.
Observed spectral narrowing is a diagnostic, never an imposed range correction.
"""
import argparse
import itertools
from pathlib import Path
import h5py
import numpy as np


def narrowing(h,frames):
 r=h['slant_range_km'][:];f=h['frequency_hz'][:];geometry=h['beam_geometry'][:];profiles=[];peaks=[]
 for b in range(5):
  signal=np.median(h[f'beam_{b}/background_subtracted_psd_adu2_per_hz'][frames],axis=0)
  ratio=signal[:,abs(f)<=1562.5].sum(axis=1)/signal[:,abs(f)<=8000].sum(axis=1)
  keep=(r>220)&(abs(r-geometry[b,3])<130);j=np.flatnonzero(keep)[np.argmax(ratio[keep])]
  if 0<j<len(r)-1:
   x=r[j-1:j+2];q=np.polyfit(x-r[j],ratio[j-1:j+2],2);delta=np.clip(-q[1]/(2*q[0]),-18,18) if q[0]<0 else 0.
  else:delta=0.
  profiles.append(ratio);peaks.append(r[j]+delta)
 return np.asarray(profiles),np.asarray(peaks)


def main():
 p=argparse.ArgumentParser();p.add_argument('products',type=Path);p.add_argument('output',type=Path);q=p.parse_args()
 with h5py.File(q.output,'w') as out:
  for session in ['190252','205054','205701']:
   with h5py.File(q.products/f'session_{session}/recording-index.h5') as ix:
    groups=out.create_group(f'session_{session}');beam=ix['beam_index'][:];expected=ix['beam_raw_angles'][:];time=ix['time_utc_s'][:];cursor=0;events=[];last=None;bad_id=0;bad_rx=0;bad_tx=0;headers=[]
    for path in ix['compute_cache_paths'].asstr()[:]:
     mm=np.memmap(path,mode='r',dtype='u1');n=len(mm)//8336
     word=lambda offset:np.ndarray((n,),dtype='<u2',buffer=mm,offset=offset,strides=(8336,))
     assignment=beam[cursor:cursor+n];bad_id+=int(np.count_nonzero(word(772)!=assignment));bad_rx+=int(np.count_nonzero((word(856)!=expected[assignment,0])|(word(858)!=expected[assignment,1])));bad_tx+=int(np.count_nonzero((word(868)!=expected[assignment,0])|(word(870)!=expected[assignment,1])))
     cfg=np.stack([word(j) for j in [664,732,740]],axis=1);changes=np.r_[0,np.flatnonzero(np.any(np.diff(cfg.astype(int),axis=0)!=0,axis=1))+1]
     for j in changes:
      if last is None or np.any(cfg[j]!=last):events.append([time[cursor+j],*cfg[j]]);last=cfg[j].copy()
     headers.append(np.frombuffer(mm[:1024],dtype='<u2').copy());cursor+=n
    groups['configuration_events']=events;groups['first_header_u16_by_file']=headers
    groups.attrs['records_checked']=cursor;groups.attrs['raw_word772_mismatches']=bad_id;groups.attrs['receive_angle_mismatches']=bad_rx;groups.attrs['transmit_angle_mismatches']=bad_tx
    print(session,'records',cursor,'mismatches id/rx/tx',bad_id,bad_rx,bad_tx,flush=True);assert bad_id==bad_rx==bad_tx==0
  with h5py.File(q.products/'session_190252/provisional_delay160us/range-doppler.h5') as old,h5py.File(q.products/'session_205701/provisional_delay160us/range-doppler.h5') as new:
   a,pa=narrowing(old,np.arange(0,7));b,pb=narrowing(new,np.arange(27,41));nominal=old['beam_geometry'][:,3];out['slant_range_km']=old['slant_range_km'][:];out['beam_geometry']=old['beam_geometry'][:];out['early_narrowing_profile']=a;out['later_narrowing_profile']=b;out['early_peak_range_km']=pa;out['later_peak_range_km']=pb
   ranking=[]
   for order in itertools.permutations(range(5)):
    residual=pb-nominal[list(order)];offset=float(np.mean(residual));rms=float(np.sqrt(np.mean((residual-offset)**2)));ranking.append((rms,order,offset))
   ranking.sort();out['best_beam_permutation']=ranking[0][1];out.attrs['common_offset_best_fit_km']=ranking[0][2];out.attrs['best_permutation_residual_rms_km']=ranking[0][0];out.attrs['next_best_permutation_rms_km']=ranking[1][0]
   print('Early offsets',pa-nominal,'later offsets',pb-nominal,'best assignments',ranking[:2],flush=True)
  out.attrs['configuration_columns']='UTC_epoch_s,raw_u16_byte664,raw_u16_byte732,raw_u16_byte740'
  out.attrs['configuration_semantics']='Raw word meanings unavailable; no unsupported filtering/pulse-compression interpretation is assigned'
  out.attrs['conclusion']='Recorded beam id, RX angle and TX angle associations agree. This checks software grouping, not physical pointing. Later narrowing shifts after a configuration change; a unique extra timing or steering correction is not established.'


if __name__=='__main__':main()
