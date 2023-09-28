import glob
import matplotlib.pyplot as plt
import h5py
import numpy as n
import stuffr

import isr_spec.il_interp as il
import fit_ionline
import isr_spec
import scipy.optimize as so
import scipy.interpolate as si
import scipy.constants as c
import tx_power as txp

from mpi4py import MPI

comm=MPI.COMM_WORLD
size=comm.Get_size()
rank=comm.Get_rank()

zpm,mpm=txp.get_tx_power_model(dirn="/media/j/fee7388b-a51d-4e10-86e3-5cabb0e1bc13/isr/2023-09-05/usrp-rx0-r_20230905T214448_20230906T040054/metadata/powermeter")    
ilf=il.ilint(fname="isr_spec/ion_line_interpolate.h5")

def model_spec(te,ti,mol_frac,vi,dop):
    dop_shift=2*n.pi*2*440.2e6*vi/c.c

    model=ilf.getspec(ne=n.array([1e11]),
                      te=n.array([te]),
                      ti=n.array([ti]),
                      mol_frac=n.array([mol_frac]),
                      vi=n.array([0.0]),
                      acf=False,
                      normalize=True,
                      )[0,:]
    
    dhz=n.copy(ilf.doppler_hz)
    dhz[0]=-1e6
    dhz[len(dhz)-1]=1e6
    specf=si.interp1d(dhz, model)
    return(specf(dop+dop_shift))

#spec=model_spec(2000,1500,0.0,100,n.linspace(-20e3,20e3,num=1000))
#plt.plot(n.linspace(-20e3,20e3,num=1000),spec)
#plt.show()

def fit_spec(meas,dop_amb,dop_hz,hgt,fit_idx,plot=False):
    mol_frac=fit_ionline.mh_molecular_ion_fraction(n.array([hgt]))[0]
    DA=n.fft.fft(n.fft.fftshift(dop_amb))
    
    peak=n.max(meas)
    noise_floor_est=n.nanmedian(meas[n.abs(dop_hz)>30e3])
    pwr_est=peak-noise_floor_est
    
#    plt.plot(meas)
 #   plt.axhline(peak)
  #  plt.axhline(noise_floor_est)    
   # plt.show()
    def ss(x):
        tr=x[0]
        ti=x[1]
        vi=x[2]
        pwr=x[3]
        noise_floor=x[4]        
        te=ti*tr
        spec=pwr*model_spec(te,ti,mol_frac,vi,dop_hz)
#        plt.plot(spec)sim
        model=n.fft.ifft(n.fft.fft(spec)*DA)+noise_floor
  #      plt.plot(model)
 #       plt.plot(meas,".")
#        plt.show()
#        plt.show()
        ss=n.nansum(n.abs(meas[fit_idx]-model[fit_idx])**2.0)
#        print(ss)
        return(ss)

    xhat=so.minimize(ss,[1.5,1000,142,pwr_est,noise_floor_est],method="Nelder-Mead",bounds=((0.99,5),(150,4000),(-1500,1500),(0.5*pwr_est,2*pwr_est),(0.5*noise_floor_est,1.5*noise_floor_est))).x
    xhat2=so.minimize(ss,[1.5,2000,-142,pwr_est,noise_floor_est],method="Nelder-Mead",bounds=((0.99,5),(150,4000),(-1500,1500),(0.5*pwr_est,2*pwr_est),(0.5*noise_floor_est,1.5*noise_floor_est))).x
    if ss(xhat2) < ss(xhat):
        xhat=xhat2
    
    spec=xhat[3]*model_spec(xhat[0]*xhat[1],xhat[1],mol_frac,xhat[2],dop_hz)
    snr=n.sum(spec)/(len(spec)*xhat[4])

    model=n.fft.ifft(n.fft.fft(spec)*DA) + xhat[4]
    if plot:
        plt.plot(dop_hz,model,label="Best fit")
        plt.plot(dop_hz,meas,".",label="Measurements")
        plt.plot(dop_hz[fit_idx],meas[fit_idx],".",label="Fit measurements")        
        plt.title("Millstone Hill\nRange %1.0f (km)"%(hgt))
        plt.xlabel("Doppler (Hz)")
        plt.ylabel("Power (arb)")
        plt.legend()
        plt.show()

    return(xhat,model,snr)
    

fl=glob.glob("tmpe/il*.h5")
fl.sort()


n_avg=6
sr=1e6
h=h5py.File(fl[0],"r")
LP=n.copy(h["E_RDS_LP"][()])
WLP=n.copy(h["E_RDS_LP"][()])
TX=n.copy(h["TX_LP"][()])


dop_hz=n.copy(h["dop_hz"][()])
rgs_km=n.copy(h["rgs_km"][()])
rgs_km=rgs_km

ridx=[35,230]

LP=n.zeros(LP.shape,dtype=n.float128)
WLP=n.zeros(LP.shape,dtype=n.float128)

h.close()

n_freq=LP.shape[1]
n_t=len(fl)
overview_rg=[80,170]
n_overview=len(overview_rg)
S=n.zeros([n_overview,n_t,n_freq])
n_r=LP.shape[0]
fit_idx=n.where(n.abs(dop_hz)<20e3)[0]

tsys=n.zeros(n_t)

tv=n.zeros(n_t)
pp=n.zeros([n_t,n_r,5])
n_avg=5
n_ints=int(n.floor(len(fl)/n_avg))
for fi in range(rank,n_ints,size):
    LPA=n.zeros([n_avg,n_r,n_freq])
    TX[:,:]=0.0
    i0=0
    for ai in range(n_avg):
        f=fl[fi*n_avg+ai]
        h=h5py.File(f,"r")
        if ai==0:
            i0=h["i0"][()]
            tv[fi]=i0
        lpvar=(h["RDS_LP_var"][()])
        LPA[ai,:,:]=h["E_RDS_LP"][()]
        TX+=h["TX_LP"][()]
        tsys[fi]+=h["T_sys"][()]
        print(h["T_sys"][()])
        h.close()
    tsys[fi]=tsys[fi]/n_avg

    LP=n.nanmean(LPA,axis=0)
    
    ATX=TX[5:36,:]
    ATX=ATX/n.max(TX)
    dop_amb=n.sum(ATX,axis=0)
    dop_amb=dop_amb/n.sum(dop_amb)

#    plt.plot(dop_hz,10.0*n.log10(dop_amb),".")
 #   plt.show()

    

    
    
    #nf=n.nanmedian(LP[130:230,n.abs(dop_hz)>30e3])
    
    # scale power to noise temperature
    nf=n.nanmedian(LP)
        
    for oi in range(n_overview):
        S[oi,fi,:]=(LP[overview_rg[oi],:]-nf)/nf

    # remove range dependence
    #for ri in range(LP.shape[0]):
    #    LP[ri,:]=LP[ri,:]*rgs_km[ri]**2.0

    avg_window=31
    for freqi in range(LP.shape[1]):
        LP[:,freqi]=n.convolve(LP[:,freqi],n.repeat(1/avg_window,avg_window),mode="same")

    if False:
        plt.pcolormesh(10.0*n.log10(LP))
        plt.colorbar()
        plt.show()

    LP2=n.copy(LP)
    LP2[:,:]=0.0
    
    for ri in range(ridx[0],ridx[1]):#LP.shape[0]):
        if rgs_km[ri]>600:
            xhat,model,snr=fit_spec(LP[ri,:],dop_amb,dop_hz,rgs_km[ri],fit_idx,plot=False)
        else:
            xhat,model,snr=fit_spec(LP[ri,:],dop_amb,dop_hz,rgs_km[ri],fit_idx,plot=False)            
        pp[fi,ri,:]=xhat

        # get electron density from snr
        # snr = s/n = T_echo/T_sys
        # T_sys*snr = T_echo
        # T_echo = (1/magic_const) * Ptx * ne / (1+Te/Ti) / R**2.0 = T_sys*snr
        # ne = magic_const*T_sys*snr*(1+Te/Ti)/Ptx
        pp[fi,ri,3]=(1+xhat[0])*snr*tsys[fi]*rgs_km[ri]**2.0/zpm(i0/1e6)
        
        LP2[ri,:]=model

    plt.subplot(231)
    dB=10.0*n.log10(LP2[:,fit_idx])

    dB0=10.0*n.log10(LP2[:,:])
    nfl=n.nanmedian(dB0)
    
    plt.pcolormesh(dop_hz/1e3,rgs_km,10.0*n.log10(LP2[:,:]),vmin=nfl-6,vmax=nfl+10)
    plt.title("Model")    
    plt.xlabel("Doppler (kHz)")
    plt.ylabel("Range (km)")    
    plt.colorbar()
    plt.subplot(232)
    plt.pcolormesh(dop_hz/1e3,rgs_km,10.0*n.log10(LP[:,]),vmin=nfl-6,vmax=nfl+10)
    plt.title("Measurement")
    plt.xlabel("Doppler (kHz)")
    plt.ylabel("Range (km)")    
    
    plt.colorbar()
    plt.subplot(233)
    plt.plot(pp[fi,:,0]*pp[fi,:,1],rgs_km,label="Te")
    plt.plot(pp[fi,:,1],rgs_km,label="Ti")
    plt.plot(pp[fi,:,2]*10,rgs_km,label="vi*10")
    plt.xlim([-1e3,5e3])
    plt.ylabel("Range (km)")    

    plt.legend()
    plt.subplot(234)
    plt.pcolormesh(dop_hz[fit_idx]/1e3,rgs_km,10.0*n.log10(LP2[:,fit_idx]),vmin=nfl-1,vmax=nfl+2)
    plt.title("Model")    
    plt.xlabel("Doppler (kHz)")
    plt.ylabel("Range (km)")    
    plt.colorbar()
    plt.subplot(235)
    plt.pcolormesh(dop_hz[fit_idx]/1e3,rgs_km,10.0*n.log10(LP[:,fit_idx]),vmin=nfl-1,vmax=nfl+2)
    plt.title("Measurement")
    plt.xlabel("Doppler (kHz)")
    plt.ylabel("Range (km)")    
    
    plt.colorbar()
    plt.subplot(236)
    plt.pcolormesh(dop_hz[fit_idx]/1e3,rgs_km,LP[:,fit_idx]-LP2[:,fit_idx],vmin=-1e2,vmax=1e2)
    plt.title("Meas-Model")
    plt.xlabel("Doppler (kHz)")
    plt.ylabel("Range (km)")    
    
    plt.colorbar()
    
    plt.tight_layout()
#    plt.show()
    plt.savefig("tmp/pp_lp_%d.png"%(i0/1e6))
    plt.close()

    ho=h5py.File("%s/pp-%d.h5"%("tmp",i0/1e6),"w")
    ho["Te"]=pp[fi,:,0]*pp[fi,:,1]
    ho["Ti"]=pp[fi,:,1]
    ho["vi"]=pp[fi,:,2]
    ho["neraw"]=pp[fi,:,3]
    
    ho["dTe/Ti"]=n.zeros(len(rgs_km))  # tbd fix this
    ho["dTi"]=n.zeros(len(rgs_km))
    ho["dvi"]=n.zeros(len(rgs_km))
    ho["dzl"]=n.zeros(len(rgs_km))
    
    ho["rgs"]=rgs_km
    ho["t0"]=i0/1e6
    ho["t1"]=i0/1e6+10        
    ho.close()


    h.close()

if False:
    #plt.plot(tv,tsys)
    #plt.show()
    for oi in range(n_overview):    
        plt.pcolormesh(dop_hz/1e3,tv,n.log10(S[oi,:,:]),vmin=-1,vmax=4)
        plt.title("Range gate %1.0f km"%(rgs_km[overview_rg[oi]]))
        plt.xlabel("Doppler (kHz)")
        plt.ylabel("Time (unix)")
        cb=plt.colorbar()
        cb.set_label("Signal temperature (K,log$_{10}$)")
        plt.show()



