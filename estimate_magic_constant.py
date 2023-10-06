import numpy as n
import matplotlib.pyplot as plt
import h5py
import glob
import sys
import stuffr


dirname=sys.argv[1]
calfile=sys.argv[2]

# read calibration info
cal=h5py.File(calfile,"r")
# can be positive or negative
pf=n.abs(cal["plasma_frequency"][()])
#print(pf.shape)

pf_ts=pf[:,0]
pf_ne = (1e6*pf[:,1]/8.98)**2.0
    
fl=glob.glob("%s/pp*.h5"%(dirname))
fl.sort()
#print(fl)

minimum_tx_pwr=400e3
maximum_tsys=2e3
maximum_time_offset = 60.0
debug_cal=False

nan_space_objects=0
nan_noisy_estimates=True

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

so_t=n.array([])
so_r=n.array([])
tv_dt=[]
tx_pwr=n.zeros(nt)
so_count=n.zeros([nt,nr],dtype=int)
tsys=n.zeros(nt)
for i in range(nt):
#    print(i)
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
    
    if nan_noisy_estimates:
        P[i,DP[i,:,0]>5,:]=n.nan
    P[i,n.isnan(DP[i,:,0]),:]=n.nan
    P[i,n.isnan(DP[i,:,1]),:]=n.nan
    P[i,n.isnan(DP[i,:,2]),:]=n.nan
    P[i,n.isnan(DP[i,:,3]),:]=n.nan

    # space object filter
    P[i,so_count[i,:]>nan_space_objects,:]=n.nan
    h.close()



uncal_ne = []
cal_ne = []
good_cal_idx=[]
for cal_idx in range(len(pf_ts)):
    bi=n.argmin(n.abs(tv-pf_ts[cal_idx]))
    dt=tv[bi]-pf_ts[cal_idx]
#    print(dt)

    if n.abs(dt) < maximum_time_offset:
        cal_rgi=n.where( (rgs >= 200) & (rgs < 450) & (n.isnan(P[bi,:,3]) != True)  )[0]
        if len(cal_rgi) < 3:
            print("not enough ion-line measurements")
            continue
        good_cal_idx.append(cal_idx)
        log_rg=n.log10(rgs[cal_rgi])
        log_ne=n.log10(P[bi,cal_rgi,3])
        
        n_meas=len(cal_rgi)
        A=n.zeros([n_meas,3])
        A[:,0]=1
        A[:,1]=log_rg
        A[:,2]=log_rg**2.0
        xhat=n.linalg.lstsq(A,log_ne)[0]

        # logne = xhat[0] + xhat[1]*log_rg + xhat[2]*log_rg**2.0
        # peak is at
        # dlogne/dlogrg = 0 =>  log_rg = -xhat[1]/(2*xhat[2])
        peak_rg=-xhat[1]/2/xhat[2]        
        peak_ne = xhat[0] + xhat[1]*peak_rg + xhat[2]*peak_rg**2.0
        rgsweep=n.linspace(n.min(log_rg),n.max(log_rg),num=100)

        uncal_ne.append(10**peak_ne)
        cal_ne.append(pf_ne[cal_idx])
        
        if debug_cal:
            plt.plot(log_rg,log_ne,".")
            plt.plot(rgsweep,xhat[0]+xhat[1]*rgsweep+xhat[2]*rgsweep**2.0)
            plt.axvline(peak_rg)
            plt.axhline(peak_ne)        
            plt.show()
    else:
        print("no measurement found within reasonable offset from plasma lien cal (best is %1.2f s)"%(dt))

magic_consts=n.array(cal_ne)/n.array(uncal_ne)
if len(magic_consts)>2:
#    print(magic_consts)
    mc_mean=n.mean(magic_consts)
 #   print(mc_mean)
    mc_std=n.std(magic_consts)
    print("the magic constant is %1.2f +/- %1.2f (%1.2f %%)"%(mc_mean,mc_std,100.0*mc_std/mc_mean))
    if debug_cal:
        plt.plot(magic_consts,".")
        plt.axhline(mc_mean)
        plt.show()

    good_cal_idx=n.array(good_cal_idx,dtype=int)
    plt.semilogy(pf_ts[good_cal_idx],mc_mean*n.array(uncal_ne),".",label="Ion-line ne")
    plt.semilogy(pf_ts[good_cal_idx],n.array(cal_ne),".",label="Plasma-line ne")
    plt.xlabel("Time (unix)")
    plt.ylabel("Ne")
    plt.legend()
    plt.show()

    ofile="%s/magic_const.h5"%(dirname)
    print("saving %s"%(ofile))
    ho=h5py.File(ofile,"w")
    ho["t_min"]=n.min(pf_ts)
    ho["t_max"]=n.max(pf_ts)
    ho["magic_constant"]=mc_mean
    ho["magic_std"]=mc_std
    ho["pl_time"]=pf_ts[good_cal_idx]
    ho["pl_freq"]=pf[good_cal_idx,:1]
    ho.close()
else:
    print("didn't find enough measurements to derive value")
    
