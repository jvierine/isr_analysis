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
if len(sys.argv) == 3:
    channel=sys.argv[2]
else:
    channel="none"
print(dirname)
print(channel)
fl=glob.glob("%s/pp*.h5"%(dirname))
fl.sort()


# this can be derived with the help of
# two helpers functions: plasma_line_clicker.py and estimate_magic_constant.py
#
magic_constant=75800958.63*10

mc_file="%s/magic_const.h5"%(dirname)
if os.path.exists(mc_file):
    
    mch=h5py.File(mc_file,"r")
    magic_constant=mch["magic_constant"][()]
    print("reading magic constant from file %s\ngot mc=%1.0f"%(mc_file,magic_constant))
    mch.close()

minimum_tx_pwr=400e3
maximum_tsys=2e3
show_space_objects=False
nan_space_objects=1
nan_noisy_estimates=True

nt=len(fl)
h=h5py.File(fl[0],"r")
nr=len(h["rgs"][()])
rgs=h["rgs"][()]
d_rg=rgs[1]-rgs[0]
if "range_avg_limits_km" in h.keys():
    range_avg_limits_km=h["range_avg_limits_km"][()]
    range_avg_window_km=h["range_avg_window_km"][()]
else:
    range_avg_limits_km=0
    range_avg_window_km=0

h.close()

P=n.zeros([nt,nr,5])
tv=n.zeros(nt)
P[:,:,:]=n.nan
DP=n.zeros([nt,nr,5])
P[:,:,:]=n.nan

so_t=n.array([])
so_r=n.array([])
tv_dt=[]
tx_pwr=n.zeros(nt)
so_count=n.zeros([nt,nr],dtype=int)
tsys=n.zeros(nt)

t_mat=n.zeros([nt+1,nr+1])
r_mat=n.zeros([nt+1,nr+1])
rgs_limits=n.concatenate((rgs,[rgs[-1]+d_rg]))

az=n.zeros(nt)
el=n.zeros(nt)

for i in range(nt):
    h=h5py.File(fl[i],"r")
    try:    
        P[i,:,0]=h["Te"][()]
        P[i,:,1]=h["Ti"][()]
        P[i,:,2]=h["vi"][()]
        P[i,:,3]=h["ne"][()]
        if "az" in h.keys():
            az[i]=h["az"][()]
            print(az[i])
        if "el" in h.keys():
            el[i]=h["el"][()]
            print(el[i])            
        
        if "heavy_ion_frac" in h.keys():
            P[i,:,4]=h["heavy_ion_frac"][()]        

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
    t_mat[i,:]=h["t0"][()]
    t_mat[i+1,:]=h["t1"][()]
    r_mat[i,:]=rgs_limits
    r_mat[i+1,:]=rgs_limits
    
    tv_dt.append(stuffr.unix2date(tv[i]))

    P_orig=n.copy(P)
    if nan_noisy_estimates:
        P[i,DP[i,:,0]>10,:]=n.nan
        P[i,DP[i,:,3]>0.8,:]=n.nan    
#    P[i,DP[i,:,3]>1,:]=n.nan    
#    P[i,DP[i,:,1]>2000,:]=n.nan

    P[i,n.isnan(DP[i,:,0]),:]=n.nan
    P[i,n.isnan(DP[i,:,1]),:]=n.nan
    P[i,n.isnan(DP[i,:,2]),:]=n.nan
    P[i,n.isnan(DP[i,:,3]),:]=n.nan

    # space object filter
    P[i,so_count[i,:]>nan_space_objects,:]=n.nan
    
    h.close()

#plt.plot(tv,az,".")
#plt.show()
#r_mat[:,]

    
so_t_dt=[]
for sot in so_t:
    try:
        so_t_dt.append(stuffr.unix2date(sot))
    except:
        so_t_dt.append(stuffr.unix2date(sot/1e6))        


#plt.plot(tv_dt,tx_pwr)
#plt.show()
#print(P.shape)
#print(t_mat.shape)
#print(r_mat.shape)

# what a mess to date dates in a matrix
t_mat=n.array(t_mat,dtype=n.timedelta64) + n.datetime64(0,"s")

if False:
    plt.pcolormesh(t_mat,r_mat,P[:,:,4])
    cb=plt.colorbar()
    cb.set_label("Heavy ion fraction")
    plt.show()
        

        
fig,((ax00,ax01),(ax10,ax11))=plt.subplots(2,2,figsize=(16,9))

#plt.subplot(221)
p=ax00.pcolormesh(t_mat,r_mat, P[:,:,0],vmin=500,vmax=4000,cmap="turbo")
#ax00.set_ylim([70,1000])
if show_space_objects:
    ax00.plot(so_t_dt,so_r,"x",color="black",alpha=0.1)
#ax00.set_ylim([100,900])
ax00.set_xlabel("Time (UT)\n(since %s)"%(stuffr.unix2datestr(tv[0])))
ax00.set_ylabel("Range (km)")
cb=fig.colorbar(p,ax=ax00)
cb.set_label("$T_e$ (K)")
#plt.subplot(222)
p=ax01.pcolormesh(t_mat,r_mat,P[:,:,1],vmin=500,vmax=2000,cmap="turbo")
#ax01.set_ylim([70,1000])
ax01.set_xlabel("Time (UT)")
ax01.set_ylabel("Range (km)")

#plt.ylim([100,900])
cb=fig.colorbar(p,ax=ax01)
cb.set_label("$T_i$ (K)")

#plt.subplot(223)
p=ax10.pcolormesh(t_mat,r_mat,P[:,:,2],vmin=-100,vmax=100,cmap="seismic")
print(n.nanmedian(P[:,:,2]))
#ax10.set_ylim([70,1000])
#p=ax10.pcolormesh(t_mat,r_mat,(P[:,:,2]-n.nanmedian(P[:,:,2])),vmin=-100,vmax=100,cmap="seismic")
ax10.set_xlabel("Time (UT)")
ax10.set_ylabel("Range (km)")

#plt.ylim([100,900])
cb=fig.colorbar(p,ax=ax10)
cb.set_label("$v_i$ (m/s)")

#plt.subplot(224)
p=ax11.pcolormesh(t_mat,r_mat,n.log10(magic_constant*P[:,:,3]),vmin=9,vmax=12.0,cmap="turbo")
#ax11.set_ylim([70,1000])
ax11.set_xlabel("Time (UT)")
ax11.set_ylabel("Range (km)")

#plt.ylim([100,900])
cb=fig.colorbar(p,ax=ax11)
cb.set_label("$N_e$ (m$^{-1}$)")
plt.tight_layout()
plt.savefig("ppar-%s-%s.png"%(channel,stuffr.unix2datestr(tv[0])),dpi=100)
plt.show()

if False:
    fig,((ax00,ax01),(ax10,ax11))=plt.subplots(2,2,figsize=(16,9))
    p=ax00.pcolormesh(t_mat,r_mat,DP[:,:,0],vmin=0,vmax=5,cmap="plasma")
    cb=fig.colorbar(p,ax=ax00)
    cb.set_label("$\Delta T_e/T_i$ (K)")
    ax00.set_xlabel("Time (UT)\n(since %s)"%(stuffr.unix2datestr(tv[0])))
    ax00.set_ylabel("Range (km)")

    p=ax01.pcolormesh(t_mat,r_mat,DP[:,:,1],vmin=0,vmax=4000,cmap="plasma")
    cb=fig.colorbar(p,ax=ax01)
    cb.set_label("$\Delta T_i$ (K)")
    ax01.set_xlabel("Time (UT)")
    ax01.set_ylabel("Range (km)")

    p=ax10.pcolormesh(t_mat,r_mat,DP[:,:,2],vmin=0,vmax=200,cmap="plasma")
    ax10.set_xlabel("Time (UT)")
    ax10.set_ylabel("Range (km)")

    cb=fig.colorbar(p,ax=ax10)
    cb.set_label("$\Delta v_i$ (m/s)")


    p=ax11.pcolormesh(t_mat,r_mat,DP[:,:,3],vmin=0,vmax=1.0,cmap="plasma")
    cb=fig.colorbar(p,ax=ax11)
    cb.set_label("$\Delta N_e/N_e$")
    ax11.set_xlabel("Time (UT)")
    ax11.set_ylabel("Range (km)")

    plt.tight_layout()
    plt.show()


ofile="ppar-%s-%s.h5"%(channel,stuffr.unix2datestr(tv[0]))
print("writing %s"%(ofile))
ho=h5py.File(ofile,"w")
ho["range"]=rgs
ho["Te"]=P_orig[:,:,0]
ho["Ti"]=P_orig[:,:,1]
ho["vi"]=P_orig[:,:,2]
ho["ne"]=P_orig[:,:,3]*magic_constant
ho["time_unix"]=tv
ho["dTe/Ti"]=DP[:,:,0]
ho["az"]=az
ho["el"]=el
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
