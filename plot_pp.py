import numpy as n
import matplotlib.pyplot as plt
import h5py
import glob


fl=glob.glob("lpi_f/pp*.h5")
fl.sort()


nt=len(fl)
h=h5py.File(fl[0],"r")
nr=len(h["rgs"][()])
rgs=h["rgs"][()]
h.close()

P=n.zeros([nt,nr,4])
tv=n.zeros(nt)
P[:,:]=n.nan
for i in range(nt):
    print(i)
    h=h5py.File(fl[i],"r")
    P[i,:,0]=h["Te"][()]
    P[i,:,1]=h["Ti"][()]
    P[i,:,2]=h["vi"][()]
    P[i,:,3]=h["zl"][()]
    tv[i]=0.5*(h["t0"][()]+h["t1"][()])
fig=plt.figure(figsize=(12,6))
plt.subplot(411)
plt.pcolormesh(tv,rgs,P[:,:,0].T,vmin=300,vmax=5000,cmap="jet")
plt.colorbar()
plt.subplot(412)
plt.pcolormesh(tv,rgs,P[:,:,1].T,vmin=300,vmax=3000,cmap="jet")
plt.colorbar()
plt.subplot(413)
plt.pcolormesh(tv,rgs,P[:,:,2].T,vmin=-100,vmax=100,cmap="jet")
plt.colorbar()
plt.subplot(414)
plt.pcolormesh(tv,rgs,P[:,:,3].T,vmin=0.5,vmax=2,cmap="jet")
plt.colorbar()
plt.tight_layout()
plt.show()
