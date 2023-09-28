import numpy as n
import matplotlib.pyplot as plt
import h5py
import glob
import sys


fl=glob.glob("%s/pp*.h5"%(sys.argv[1]))
fl.sort()

magic_constant=5e11 * 0.001

nt=len(fl)
h=h5py.File(fl[0],"r")
nr=len(h["rgs"][()])
rgs=h["rgs"][()]
h.close()

P=n.zeros([nt,nr,4])
tv=n.zeros(nt)
P[:,:]=n.nan
DP=n.zeros([nt,nr,4])
P[:,:]=n.nan
for i in range(nt):
    print(i)
    h=h5py.File(fl[i],"r")
    try:    
        P[i,:,0]=h["Te"][()]
        P[i,:,1]=h["Ti"][()]
        P[i,:,2]=h["vi"][()]
        P[i,:,3]=h["ne"][()]

        DP[i,:,0]=h["dTe/Ti"][()]
        DP[i,:,1]=h["dTi"][()]
        DP[i,:,2]=h["dvi"][()]
        DP[i,:,3]=h["dne"][()]/h["ne"][()]
    except:
        print("missing key in hdf5 file. probably incompable data.")
    tv[i]=0.5*(h["t0"][()]+h["t1"][()])

    
#    P[i,DP[i,:,0]>4,:]=n.nan
#    P[i,DP[i,:,3]>1,:]=n.nan    
#    P[i,DP[i,:,1]>2000,:]=n.nan
#    P[i,DP[i,:,2]>400,:]=n.nan    
    P[i,n.isnan(DP[i,:,0]),:]=n.nan
    P[i,n.isnan(DP[i,:,1]),:]=n.nan
    P[i,n.isnan(DP[i,:,2]),:]=n.nan
    P[i,n.isnan(DP[i,:,3]),:]=n.nan        
    
    h.close()


    
#fig=plt.figure(figsize=(12,6))
plt.subplot(221)
plt.pcolormesh(tv,rgs,P[:,:,0].T,vmin=500,vmax=4000,cmap="jet")
#plt.ylim([100,900])
plt.xlabel("Time (unix seconds)")
plt.ylabel("Range (km)")
cb=plt.colorbar()
cb.set_label("$T_e$ (K)")
plt.subplot(222)
plt.pcolormesh(tv,rgs,P[:,:,1].T,vmin=500,vmax=3000,cmap="jet")
plt.xlabel("Time (unix seconds)")
plt.ylabel("Range (km)")

#plt.ylim([100,900])
cb=plt.colorbar()
cb.set_label("$T_i$ (K)")

plt.subplot(223)
plt.pcolormesh(tv,rgs,P[:,:,2].T,vmin=-100,vmax=100,cmap="seismic")
plt.xlabel("Time (unix seconds)")
plt.ylabel("Range (km)")

#plt.ylim([100,900])
cb=plt.colorbar()
cb.set_label("$v_i$ (m/s)")

plt.subplot(224)
plt.pcolormesh(tv,rgs,n.log10(magic_constant*P[:,:,3].T),cmap="jet")
plt.xlabel("Time (unix seconds)")
plt.ylabel("Range (km)")

#plt.ylim([100,900])
cb=plt.colorbar()
cb.set_label("$N_e$ (m$^{-1}$)")
plt.tight_layout()
plt.show()

plt.subplot(221)
plt.pcolormesh(tv,rgs,DP[:,:,0].T,vmin=0,vmax=5,cmap="plasma")
cb=plt.colorbar()
cb.set_label("$dT_e/T_i$ (K)")
plt.subplot(222)
plt.pcolormesh(tv,rgs,DP[:,:,1].T,vmin=0,vmax=4000,cmap="plasma")
cb=plt.colorbar()
cb.set_label("$dT_i$ (K)")

plt.subplot(223)
plt.pcolormesh(tv,rgs,DP[:,:,2].T,vmin=0,vmax=2000,cmap="plasma")
cb=plt.colorbar()
cb.set_label("$dv_i$ (m/s)")

plt.subplot(224)
plt.pcolormesh(tv,rgs,DP[:,:,3].T/P[:,:,3].T,vmin=0,vmax=0.001,cmap="plasma")
cb=plt.colorbar()
cb.set_label("dNe/Ne")
plt.tight_layout()
plt.show()


ho=h5py.File("plasma_par.h5","w")
ho["range"]=rgs
ho["Te"]=P[:,:,0]
ho["Ti"]=P[:,:,1]
ho["vi"]=P[:,:,2]
ho["ne"]=P[:,:,3]*magic_constant
ho["time_unix"]=tv
ho["dTe/Ti"]=DP[:,:,0]
ho["dTi"]=DP[:,:,1]
ho["dvi"]=DP[:,:,2]
ho["dne/ne"]=DP[:,:,3]
ho.close()
