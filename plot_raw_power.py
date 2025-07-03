import numpy as n
import glob
import matplotlib.pyplot as plt
import h5py

import scipy.signal as ss
import stuffr

import tx_power as txp
plt.rcParams["date.autoformatter.minute"] = "%H:%M:%S"
plt.rcParams["date.autoformatter.hour"] = "%H:%M"


import sys
dirname=sys.argv[1]
    
fl=glob.glob("%s/lpi*.h5"%(dirname))
fl.sort()

tv=n.zeros(len(fl))
for fi,f in enumerate(fl):
    h=h5py.File(f,"r")
    tv[fi]=(h["i0"][()])

# acf estimate with ground clutter removal
acf_key="acfs_g"

h=h5py.File(fl[0],"r")
a=h[acf_key][()]
rgs_km=h["rgs_km"][()]
lags=h["lags"][()]

rmax=a.shape[0]
nlags=a.shape[1]
nt=len(fl)
h.close()

tdec=5
wgt_sum=n.zeros(rmax)
nout=int(n.floor(len(fl)/tdec))
A=n.zeros([nout,rmax],dtype=n.complex64)
tms=n.zeros(nout)

for ti in range(nout):
    As=[]
    tvs=[]
    for di in range(tdec):
        fi=ti*tdec + di
        f=fl[fi]
        h=h5py.File(f,"r")
        ptx=h["P_tx"][()]
        # scale with tx power and magic constant
        a=h[acf_key][()]/h["alpha"][()]/ptx
        # three first largs
        As.append(a[:,1]+a[:,2]+a[:,3])
        tvs.append(tv[fi])
    As=n.array(As,dtype=n.complex64)
    A[ti,:]=n.nanmean(As,axis=0)#a[:,0:A.shape[2]]
    tms[ti]=n.nanmean(tvs)
    
for ri in range(A.shape[1]):
    A[:,ri]=A[:,ri]*rgs_km[ri]**2

dt=n.median(n.diff(tms))
dr=n.median(n.diff(rgs_km))
tms=tms.astype('datetime64[s]')
plt.figure(figsize=(10,6))
plt.pcolormesh(tms,rgs_km,n.real(A).T,cmap="turbo",vmin=1e1,vmax=6e3)
plt.title(r"$\Delta t$=%1.0f s $\Delta r$=%1.1f km"%(dt,dr))
cb=plt.colorbar()
cb.set_label("Range corrected volume scatter power (linear)")
plt.xlabel("Time since %s"%(tms[0]))
plt.ylabel("Range (km)")
plt.tight_layout()
#plt.savefig("depletion/zenith_ac30_raw_power.png",dpi=300)
plt.savefig("depletion/misa_ac30_raw_power.png",dpi=300)
plt.show()
