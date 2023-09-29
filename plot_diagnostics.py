import numpy as n
import matplotlib.pyplot as plt
import h5py



import glob



fl=glob.glob("lpi_240/lpi-*.h5")
fl.sort()
n_t=len(fl)
h=h5py.File(fl[0],"r")
S=h["diagnostic_pwr_spec"][()]

R=h["retained_measurement_fraction"][()]
meas_delay=h["meas_delays_us"][()]

#plt.plot(meas_delay,R)
#plt.show()

dop=n.fft.fftshift(n.fft.fftfreq(1024,d=1/1e6))
dop_idx=n.where(n.abs(dop)<500e3)[0]
#plt.plot(dop,10.0*n.log10(S))
#plt.show()

n_fft=len(dop_idx)
h.close()

PS=n.zeros([n_t,n_fft],dtype=n.float32)
PS[:,:]=n.nan
RF=n.zeros([n_t,len(R)])
tv=n.zeros(n_t)
T_sys=n.zeros(n_t)
for fi,f in enumerate(fl):
    h=h5py.File(f,"r")
    if "diagnostic_pwr_spec" in h.keys():
        PS[fi,:]=h["diagnostic_pwr_spec"][()][dop_idx]
    if "retained_measurement_fraction" in h.keys():
        RF[fi,:]=h["retained_measurement_fraction"][()]
    if "T_sys" in h.keys():
        T_sys[fi]=h["T_sys"][()]
    tv[fi]=h["i0"][()]
    h.close()
    
#fig = plt.figure(layout='constrained', figsize=(10, 4))
#subfigs = fig.subfigures(1, 2, wspace=0.07)

plt.subplot(311)
dB=10.0*n.log10(PS.T)
nf=n.nanmedian(dB)
print(nf)
plt.pcolormesh(tv,dop[dop_idx]/1e3,dB,vmin=nf-0.5,vmax=nf+6,cmap="nipy_spectral")
plt.xlabel("Time (unix)")
plt.ylabel("Doppler (kHz)")
cb=plt.colorbar()
cb.set_label("Power spectral density (dB)")

plt.subplot(312)
plt.pcolormesh(tv,meas_delay*0.15,RF.T,vmin=0.8,vmax=1.0)
plt.ylabel("Delay (km)")
plt.xlabel("Time (unix)")
cb=plt.colorbar()
cb.set_label("Retained measurement fraction")

plt.subplot(313)
plt.plot(tv,T_sys)
plt.xlabel("Time (unix)")
plt.ylabel("T$_{\\mathrm{sys}}$ (K)")

plt.tight_layout()
plt.show()
    
