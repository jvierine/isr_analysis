"""Copy the small full-band case table without transferring large spectra."""
import sys,h5py,numpy as np
with h5py.File(sys.argv[1]) as source,h5py.File(sys.argv[2],'w') as out:
 for key in ['cases_aspect_width_nu_integral','B_height_Ne_Ti_Te_reference_RF_actual_RF_k']:out[key]=source[key][()]
 f=source['frequency_Hz'][()];out['frequency_bounds_Hz']=[f.min(),f.max()];out['frequency_node_count']=len(f)
 for key,value in source.attrs.items():out.attrs[key]=value
 out.attrs['source_full_spectrum']=sys.argv[1]
 a=out['cases_aspect_width_nu_integral'][()];print('FULLBAND_RATIO',a[::2,3]/a[1::2,3],flush=True)
