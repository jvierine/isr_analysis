import numpy as n
import matplotlib.pyplot as plt
import h5py
import glob
import sys
import stuffr
import os

import matplotlib.dates as mdates

plt.rcParams["date.autoformatter.minute"] = "%H:%M:%S"
plt.rcParams["date.autoformatter.hour"] = "%H:%M"

dirname=sys.argv[1]
fl=glob.glob("%s/pp*.h5"%(dirname))
fl.sort()

# this can be derived with the help of
# two helpers functions: plasma_line_clicker.py and estimate_magic_constant.py
#
magic_constant=75800958.63

mc_file="%s/magic_const.h5"%(dirname)
if os.path.exists(mc_file):
    
    mch=h5py.File(mc_file,"r")
    magic_constant=mch["magic_constant"][()]
    print("reading magic constant from file %s\ngot mc=%1.0f"%(mc_file,magic_constant))
    mch.close()

minimum_tx_pwr=400e3
maximum_tsys=2e3
show_space_objects=False
nan_space_objects=6
nan_noisy_estimates=True

nt=len(fl)
h=h5py.File(fl[0],"r")
nr=len(h["rgs"][()])
rgs=h["rgs"][()]
range_avg_limits_km=h["range_avg_limits_km"][()]
range_avg_window_km=h["range_avg_window_km"][()]

h.close()

P=n.zeros([nt,nr,4])
tv=n.zeros(nt)
P[:,:]=n.nan
DP=n.zeros([nt,nr,4])
P[:,:]=n.nan

so_t=n.array([])
so_r=n.array([])
tv_dt=[]
tx_pwr=n.zeros(nt)
so_count=n.zeros([nt,nr],dtype=int)
tsys=n.zeros(nt)
for i in range(nt):
    print(i)
    h=h5py.File(fl[i],"r")
    try:    
        P[i,:,0]=h["Te"][()]
        P[i,:,1]=h["Ti"][()]
        P[i,:,2]=h["vi"][()]
        P[i,:,3]=h["ne"][()]

        if "P_tx" in h.keys():
            tx_pwr[i]=h["P_tx"][()]
        if "T_sys" in h.keys():
            tsys[i]=h["T_sys"][()]
            
        
        if "space_object_times" in h.keys():
            so_t=n.concatenate((so_t,h["space_object_times"][()]))
            so_r=n.concatenate((so_r,h["space_object_rgs"][()]))            
        if "space_object_count" in h.keys():
            so_count[i,:]=h["space_object_count"][()]


        DP[i,:,0]=h["dTe/Ti"][()]
        DP[i,:,1]=h["dTi"][()]
        DP[i,:,2]=h["dvi"][()]
        DP[i,:,3]=h["dne"][()]/h["ne"][()]

        if tx_pwr[i] < minimum_tx_pwr:
            P[i,:,:]=n.nan
            DP[i,:,:]=n.nan
        if tsys[i] > maximum_tsys:
            P[i,:,:]=n.nan
            DP[i,:,:]=n.nan
        
    except:
        print("missing key in hdf5 file. probably incompable data.")
    tv[i]=0.5*(h["t0"][()]+h["t1"][()])
    tv_dt.append(stuffr.unix2date(tv[i]))

    P_orig=n.copy(P)
    if nan_noisy_estimates:
        P[i,DP[i,:,0]>4,:]=n.nan
        P[i,DP[i,:,3]>0.2,:]=n.nan    
#    P[i,DP[i,:,3]>1,:]=n.nan    
#    P[i,DP[i,:,1]>2000,:]=n.nan

    P[i,n.isnan(DP[i,:,0]),:]=n.nan
    P[i,n.isnan(DP[i,:,1]),:]=n.nan
    P[i,n.isnan(DP[i,:,2]),:]=n.nan
    P[i,n.isnan(DP[i,:,3]),:]=n.nan

    # space object filter
    P[i,so_count[i,:]>nan_space_objects,:]=n.nan
    
    h.close()

so_t_dt=[]
for sot in so_t:
    so_t_dt.append(stuffr.unix2date(sot))


fig,((ax00,ax01),(ax10,ax11))=plt.subplots(2,2,figsize=(16,9))

#plt.subplot(221)
p=ax00.pcolormesh(tv_dt,rgs,P[:,:,0].T,vmin=500,vmax=4000,cmap="jet")
if show_space_objects:
    ax00.plot(so_t_dt,so_r,"x",color="black",alpha=0.1)
#ax00.set_ylim([100,900])
ax00.set_xlabel("Time (UT)\n(since %s)"%(stuffr.unix2datestr(tv[0])))
ax00.set_ylabel("Range (km)")
cb=fig.colorbar(p,ax=ax00)
cb.set_label("$T_e$ (K)")
#plt.subplot(222)
p=ax01.pcolormesh(tv_dt,rgs,P[:,:,1].T,vmin=500,vmax=3000,cmap="jet")
ax01.set_xlabel("Time (UT)")
ax01.set_ylabel("Range (km)")

#plt.ylim([100,900])
cb=fig.colorbar(p,ax=ax01)
cb.set_label("$T_i$ (K)")

#plt.subplot(223)
p=ax10.pcolormesh(tv_dt,rgs,P[:,:,2].T,vmin=-200,vmax=200,cmap="seismic")
ax10.set_xlabel("Time (UT)")
ax10.set_ylabel("Range (km)")

#plt.ylim([100,900])
cb=fig.colorbar(p,ax=ax10)
cb.set_label("$v_i$ (m/s)")

#plt.subplot(224)
p=ax11.pcolormesh(tv_dt,rgs,n.log10(magic_constant*P[:,:,3].T),vmin=9,vmax=12.2,cmap="jet")
ax11.set_xlabel("Time (UT)")
ax11.set_ylabel("Range (km)")

#plt.ylim([100,900])
cb=fig.colorbar(p,ax=ax11)
cb.set_label("$N_e$ (m$^{-1}$)")
plt.tight_layout()
plt.show()

fig,((ax00,ax01),(ax10,ax11))=plt.subplots(2,2,figsize=(16,9))
p=ax00.pcolormesh(tv_dt,rgs,DP[:,:,0].T,vmin=0,vmax=5,cmap="plasma")
cb=fig.colorbar(p,ax=ax00)
cb.set_label("$\Delta T_e/T_i$ (K)")
ax00.set_xlabel("Time (UT)\n(since %s)"%(stuffr.unix2datestr(tv[0])))
ax00.set_ylabel("Range (km)")

p=ax01.pcolormesh(tv_dt,rgs,DP[:,:,1].T,vmin=0,vmax=4000,cmap="plasma")
cb=fig.colorbar(p,ax=ax01)
cb.set_label("$\Delta T_i$ (K)")
ax01.set_xlabel("Time (UT)")
ax01.set_ylabel("Range (km)")

p=ax10.pcolormesh(tv_dt,rgs,DP[:,:,2].T,vmin=0,vmax=200,cmap="plasma")
ax10.set_xlabel("Time (UT)")
ax10.set_ylabel("Range (km)")

cb=fig.colorbar(p,ax=ax10)
cb.set_label("$\Delta v_i$ (m/s)")


p=ax11.pcolormesh(tv_dt,rgs,DP[:,:,3].T,vmin=0,vmax=1.0,cmap="plasma")
cb=fig.colorbar(p,ax=ax11)
cb.set_label("$\Delta N_e/N_e$")
ax11.set_xlabel("Time (UT)")
ax11.set_ylabel("Range (km)")

plt.tight_layout()
plt.show()


ho=h5py.File("plasma_par.h5","w")
ho["range"]=rgs
ho["Te"]=P_orig[:,:,0]
ho["Ti"]=P_orig[:,:,1]
ho["vi"]=P_orig[:,:,2]
ho["ne"]=P_orig[:,:,3]*magic_constant
ho["time_unix"]=tv
ho["dTe/Ti"]=DP[:,:,0]
ho["dTi"]=DP[:,:,1]
ho["dvi"]=DP[:,:,2]
ho["dne/ne"]=DP[:,:,3]
ho["space_object_count"]=so_count
ho["space_object_ts"]=so_t
ho["space_object_range"]=so_r
ho["magic_constant"]=magic_constant

ho["range_avg_limits_km"]=range_avg_limits_km
ho["range_avg_window_km"]=range_avg_window_km

ho.close()
