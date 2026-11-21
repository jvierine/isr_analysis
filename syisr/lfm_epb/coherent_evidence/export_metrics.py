"""Export retained HDF5 evidence to LaTeX values; provenance is comments only."""
from pathlib import Path
import sys,h5py,numpy as np
from datetime import datetime,timezone
root=Path(sys.argv[1]);out=Path(sys.argv[2]);values={}
def number(k,x,fmt='.3f'):values[k]=format(float(x),fmt)
def sci(k,x):
 m,e=f'{float(x):.2e}'.split('e');values[k]=m+r'\times10^{'+str(int(e))+'}'
with h5py.File(root/'apparent_density.h5') as h:
 s=h['beam_358.0_60.0/persistent_summary'][()];sci('peakNe',s[4]);sci('lowerNe',s[4]/10);number('peakHeight',s[5],'.1f');number('peakAspect',abs(s[6]),'.2f');number('peakPersistence',s[7],'.2f');number('iriExcess',s[8],'.0f');number('strongCells',s[0],'.0f');number('strongMinutes',s[1],'.0f');values['firstStrong']=datetime.fromtimestamp(s[2],timezone.utc).strftime('%H:%M:%S');values['lastStrong']=datetime.fromtimestamp(s[3],timezone.utc).strftime('%H:%M:%S')
 c=h['zenith_calibration_samples'][()];number('calibrationCount',len(c),'.0f');number('coefficient',np.median(c[:,-1]),'.2f');number('coefficientLow',np.percentile(c[:,-1],10),'.2f');number('coefficientHigh',np.percentile(c[:,-1],90),'.2f');sci('criticalNe',h['critical_plasma_density_at432MHz_m3'][()]);number('cutoffFactor',s[4]/h['critical_plasma_density_at432MHz_m3'][()],'.1f');number('lowerPlasmaMHz',h['peak_plasma_frequency_scenarios_Hz'][0]/1e6,'.0f');number('controlMax',h['nonperpendicular_relative_calibration_coefficients'][()].max(),'.3f')
with h5py.File(root/'raw_checks.h5') as h:
 g=h['beam_358.0_60.0'];number('codeQuiet',np.median(g['quiet/correct_code_power_at323km'][()])/np.median(g['quiet/wrong_code_power_at323km'][()]),'.2f');number('codeEvent',np.median(g['event/correct_code_power_at323km'][()])/np.median(g['event/wrong_code_power_at323km'][()]),'.1f')
with h5py.File(root/'thermal_null_angles33.h5') as h:b=h['cases_aspect_width_nu_integral'][()]
with h5py.File(root/'thermal_null_angles09.h5') as h:a=h['cases_aspect_width_nu_integral'][()]
r=b[::2,3]/b[1::2,3];number('thermalLow',r.min(),'.3f');number('thermalHigh',r.max(),'.3f');number('angleRefinement',100*np.max(abs(a[:,3]-b[:,3])/b[:,3]),'.1f')
for name,key in [('beam_358.0_60.0','beamNoise'),('beam_358.0_90.0','zenithNoise')]:
 with h5py.File(root/'mean_power'/f'{name}.h5') as h:
  t=h['time_unix_s'][()];n=h['noise_power'][()];number(key,np.median(n[(t>=1772368170)&(t<=1772369670)])/np.median(n[t<1772364600]),'.3f')
out.write_text('% Values exported from retained HDF5 evidence products.\n'+''.join('\\newcommand{\\'+k+'}{'+v+'}\n' for k,v in values.items()))
print(out)
