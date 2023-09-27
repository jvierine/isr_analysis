import numpy as n
import matplotlib.pyplot as plt
import h5py
import glob
import sys


fl=glob.glob("%s/pp*.h5"%(sys.argv[1]))
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
    P[i,:,3]=h["neraw"][()]
    tv[i]=0.5*(h["t0"][()]+h["t1"][()])
    h.close()
#fig=plt.figure(figsize=(12,6))
plt.subplot(221)
plt.pcolormesh(tv,rgs,P[:,:,0].T,vmin=300,vmax=6000,cmap="jet")
plt.colorbar()
plt.subplot(222)
plt.pcolormesh(tv,rgs,P[:,:,1].T,vmin=300,vmax=4000,cmap="jet")
plt.colorbar()
plt.subplot(223)
plt.pcolormesh(tv,rgs,P[:,:,2].T,vmin=-500,vmax=500,cmap="seismic")
plt.colorbar()
plt.subplot(224)
plt.pcolormesh(tv,rgs,n.log10(1e2*P[:,:,3].T),cmap="jet")
plt.colorbar()
plt.tight_layout()
plt.show()
