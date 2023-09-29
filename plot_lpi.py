import numpy as n
import glob
import matplotlib.pyplot as plt
import h5py

import scipy.signal as ss

import tx_power as txp


#zpm,mpm=txp.get_tx_power_model(dirn="/media/j/fee7388b-a51d-4e10-86e3-5cabb0e1bc13/isr/2023-09-05/usrp-rx0-r_20230905T214448_20230906T040054/metadata/powermeter")    
import sys
dirname=sys.argv[1]
fl=glob.glob("%s/lpi*.h5"%(dirname))
fl.sort()

N=1
acf_key="acfs_e"

h=h5py.File(fl[0],"r")
a=h["acfs_e"][()]
rmax=a.shape[0]
h.close()
lag=0
nt=len(fl)
nlags=a.shape[1]
A=n.zeros([nt,rmax,nlags],dtype=n.complex64)
ts=n.zeros(nt)
tv=n.zeros(nt)

for fi,f in enumerate(fl):
    h=h5py.File(f,"r")
    print(f)
    a=h[acf_key][()]
    ts[fi]=h["T_sys"][()]
    A[fi,:,:]=a
    rgs_km=h["rgs_km"][()]
    lags=h["lags"][()]    
    tv[fi]=(h["i0"][()])
    h.close()

for ri in range(len(rgs_km)):
    A[:,ri,:]=A[:,ri,:]*rgs_km[ri]**2.0
dB=10.0*n.log10(A[:,:,lag].real.T)
nf=n.nanmedian(dB)
plt.pcolormesh(tv,rgs_km,dB,vmin=nf-10,vmax=nf+10,cmap="plasma")
cb=plt.colorbar()
cb.set_label("Range corrected volume scatter power (dB)")
plt.xlabel("Time (unix)")
plt.ylabel("Range (km)")
plt.show()
