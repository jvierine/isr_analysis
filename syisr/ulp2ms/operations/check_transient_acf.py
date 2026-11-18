"""Inspect temporal concentration of suspected coherent echoes, not identity."""
from pathlib import Path
import h5py,numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
root=Path('/Volumes/LaCie/SYISR_analysis/ulp2ms-20261118/session-00');out=root/'range-doppler-subtracted'
with h5py.File(root/'windows/016/lpm.h5') as f:
 r=f['slant_range_km'][:];a=f['block_acf_real_adu2'][:]+1j*f['block_acf_imag_adu2'][:];cnt=f['block_retained_pulses'][:];tt=f['time_utc_s'][:];lag=f['lag_us'][:]
bt=np.array([np.mean(tt[i*200:min((i+1)*200,len(tt))]) for i in range(len(cnt))]);minutes=(bt-tt[0])/60
fig,axes=plt.subplots(2,2,figsize=(12,7),layout='constrained');summary=[]
for row,target in enumerate([603,753]):
 g=np.argmin(abs(r-target));valid=cnt>0;x=a[:,g];mean=np.nansum(x*cnt[:,None],axis=0)/cnt.sum();norm=mean[1].real
 med=np.nanmedian(np.where(valid[:,None],x,np.nan),axis=0);delta=x-med
 axes[row,0].pcolormesh(minutes,lag,x.imag.T/norm,shading='nearest',cmap='RdBu_r',vmin=-.05,vmax=.05)
 axes[row,0].set_ylabel(f'{r[g]:.0f} km slant range\nLag (µs)');axes[row,0].set_title('Imaginary ACF / mean real 10 µs lag')
 # Scale-independent late-lag amplitude; no fit residual is used in this statistic.
 score=np.sqrt(np.mean(abs(delta[:,30:])**2,axis=1))/norm
 axes[row,1].plot(minutes,score);axes[row,1].set_ylabel('Late-lag deviation / mean short lag');axes[row,1].set_title('Deviation from temporal median')
 for ax in axes[row]:ax.set_xlabel('Minutes after 22:00 UTC')
 order=np.argsort(np.where(valid,score,-np.inf))[-5:][::-1]
 print(target,[(round(float(minutes[i]),2),round(float(score[i]),4),int(cnt[i])) for i in order],flush=True)
 summary.append(score)
fig.suptitle('21 Aug 2023, 22:00–22:30 UTC | retained pulses, top-gate subtraction')
fig.savefig(out/'suspected-echo-time-check.png',dpi=150)
with h5py.File(out/'suspected-echo-time-check.h5','w') as f:
 f.attrs['method']='Saved screened 10-second block ACFs; complex deviation from time median at lags >=300 microseconds. Diagnostic only; not satellite identification or a new rejection mask.'
 f['minutes_after_2200_utc']=minutes;f['late_lag_deviation_fraction']=summary;f['retained_pulses']=cnt
