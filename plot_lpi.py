import numpy as n
import glob
import matplotlib.pyplot as plt
import h5py


fl=glob.glob("lpi/lpi*.h5")
fl.sort()


rmax=200
nt=len(fl)
nlags=15
A=n.zeros([nt,rmax,nlags],dtype=n.complex64)
tv=[]
acfs=n.zeros([200,15],dtype=n.complex64)
for fi,f in enumerate(fl):
    h=h5py.File(f,"r")
    print(f)
    A[fi,:,:]=h["acfs"][()]
    acfs+=h["acfs"][()]
    
    rgs_km=h["rgs_km"][()]
    lags=h["lags"][()]    
    tv.append(h["i0"][()])
    h.close()

    
for ri in range(rmax):
    acfs[ri,:]=acfs[ri,:]/acfs[ri,0].real
    
plt.pcolormesh(lags*1e6,rgs_km,acfs.real,vmin=-0.5,vmax=1.0)
plt.xlabel("Lag ($\mu$s)")
plt.ylabel("Range (km)")
plt.colorbar()
plt.show()


neraw=n.copy(A[:,:,0].real)

for ri in range(rmax):
    neraw[:,ri]=neraw[:,ri]*rgs_km[ri]**2.0


plt.pcolormesh(tv,rgs_km,n.log10(neraw.T),cmap="jet",vmin=6.5,vmax=9)
plt.colorbar()
plt.show()


plt.pcolormesh(tv,rgs_km,A[:,:,0].real.T,cmap="jet",vmin=-1e3,vmax=1e4)
plt.colorbar()
plt.show()
