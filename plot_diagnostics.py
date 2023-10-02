import numpy as n
import matplotlib.pyplot as plt
import h5py

import sys

import glob
import scipy.signal as ss



fl=glob.glob("%s/lpi-*.h5"%(sys.argv[1]))
fl.sort()
n_t=len(fl)
h=h5py.File(fl[0],"r")
S=h["diagnostic_pwr_spec"][()]

R=h["retained_measurement_fraction"][()]
meas_delay=h["meas_delays_us"][()]

#plt.plot(meas_delay,R)
#plt.show()

dop=n.fft.fftshift(n.fft.fftfreq(1024,d=1/1e6))
dop_idx=n.where(n.abs(dop)<100e3)[0]
#plt.plot(dop,10.0*n.log10(S))
#plt.show()

n_fft=len(dop_idx)
h.close()

PS=n.zeros([n_t,n_fft],dtype=n.float32)
PS[:,:]=n.nan
RF=n.zeros([n_t,len(R)])
tv=n.zeros(n_t)
alpha=n.zeros(n_t)
T_sys=n.zeros(n_t)
for fi,f in enumerate(fl):
    h=h5py.File(f,"r")
    if "retained_measurement_fraction" in h.keys():
        RF[fi,:]=h["retained_measurement_fraction"][()]
    if "T_sys" in h.keys():
        T_sys[fi]=h["T_sys"][()]
    if "alpha" in h.keys():
        alpha[fi]=h["alpha"][()]
    if "diagnostic_pwr_spec" in h.keys():
#        PS[fi,:]=n.convolve(h["diagnostic_pwr_spec"][()],n.repeat(1/5,5),mode="same")[dop_idx]
        PS[fi,:]=h["diagnostic_pwr_spec"][()][dop_idx]/alpha[fi]
        PS[fi,:]=PS[fi,:]/n.nanmedian(PS[fi,:])
        
        
    tv[fi]=h["i0"][()]
    h.close()


plt.subplot(311)
dB=10.0*n.log10(PS.T)
nf=n.nanmedian(dB)
print(nf)
plt.pcolormesh(tv,dop[dop_idx]/1e3,dB,vmin=nf-1,vmax=nf+1,cmap="plasma")
plt.title(sys.argv[1])
plt.xlabel("Time (unix)")
plt.ylabel("Frequency offset from 440.2 MHz (kHz)")
cb=plt.colorbar()
cb.set_label("Power at 1000 km (dB)")

plt.subplot(312)
plt.pcolormesh(tv,meas_delay*0.15,RF.T,vmin=0.7,vmax=1,cmap="jet")
plt.ylabel("Delay (km)")
plt.xlabel("Time (unix)")
cb=plt.colorbar()
cb.set_label("Retained measurement fraction")

plt.subplot(313)
plt.plot(tv,T_sys,label="T_sys")
plt.plot(tv,1000.0*(alpha-n.median(alpha))/n.median(alpha)+500,label="Receiver gain x 100")
plt.legend()
plt.ylim([0,1900])
plt.xlabel("Time (unix)")
#plt.ylabel("T$_{\\mathrm{sys}}$ (K)")

plt.tight_layout()
plt.show()
    
