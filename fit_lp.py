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
import scipy.signal as s
import traceback

import jcoord
import millstone_radar_state as mrs

from mpi4py import MPI

comm=MPI.COMM_WORLD
size=comm.Get_size()
rank=comm.Get_rank()


ilf=il.ilint(fname="isr_spec/ion_line_interpolate.h5")
ilf_ho=il.ilint(fname="isr_spec/ion_line_interpolate_h_o.h5")


def model_spec(te,ti,mol_frac,vi,dop,topside=False):
    dop_shift=2*n.pi*2*440.2e6*vi/c.c

    if topside:
        model=ilf_ho.getspec(ne=n.array([1e11]),
                             te=n.array([te]),
                             ti=n.array([ti]),
                             mol_frac=n.array([mol_frac]),
                             vi=n.array([0.0]),
                             acf=False,
                             normalize=True,
                             )[0,:]
    else:
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

def fit_spec(meas,dop_amb,dop_hz,hgt,fit_idx,plot=True):
    
    mol_frac=fit_ionline.mh_molecular_ion_fraction(n.array([hgt]))[0]
    DA=n.fft.fft(n.fft.fftshift(dop_amb))
    
    peak=n.max(meas)
    noise_floor_est=n.nanmin(meas[n.abs(dop_hz)>30e3])
    pwr_est=peak-noise_floor_est

    topside=False
    if hgt>500:
        topside=True
    
    def ss(x):
        tr=x[0]
        ti=x[1]
        vi=x[2]
        pwr=x[3]
        noise_floor=x[4]
        heavy_frac=x[5]
        
        te=ti*tr
        spec=model_spec(te,ti,heavy_frac,vi,dop_hz,topside=topside)
        spec=n.fft.ifft(n.fft.fft(spec)*DA)
        spec=spec/n.max(spec)
        spec=pwr*spec
        model=spec+noise_floor
        ss=n.nansum(n.abs(meas[fit_idx]-model[fit_idx])**2.0)
        # add strong prior assumption
        if topside==False:
            ss+= 1e6*(heavy_frac-mol_frac)**2.0
        return(ss)

    if topside:
        heavy_frac_guess=0.9
    else:
        heavy_frac_guess=mol_frac
    
    
    xhat=so.minimize(ss,[1.5,1000,142,pwr_est,noise_floor_est,heavy_frac_guess],method="Nelder-Mead",bounds=((0.99,5),(150,4000),(-1500,1500),(0.2*pwr_est,3*pwr_est),(0.5*noise_floor_est,3*noise_floor_est),(0,1))).x
    xhat2=so.minimize(ss,[1.5,2000,-142,pwr_est,noise_floor_est,heavy_frac_guess],method="Nelder-Mead",bounds=((0.99,5),(150,4000),(-1500,1500),(0.2*pwr_est,3*pwr_est),(0.5*noise_floor_est,3*noise_floor_est),(0,1))).x
    
    if ss(xhat2) < ss(xhat):
        xhat=xhat2
        
    spec=model_spec(xhat[0]*xhat[1],xhat[1],xhat[5],xhat[2],dop_hz,topside=topside)
    spec=n.real(n.fft.ifft(n.fft.fft(spec)*DA))
    spec=xhat[3]*spec/n.max(spec)
    model=spec+xhat[4]
    
    sigmas=[n.nan,n.nan,n.nan,n.nan,n.nan,n.nan]
    try:
        # how much do we oversample the spectrum. this reduces the number of independent measurements
        oversampling_factor=len(meas)/480
        sigma=n.sqrt(oversampling_factor*n.mean(n.abs(model[fit_idx]-meas[fit_idx])**2.0))
        n_par=5
        J=n.zeros([len(fit_idx),5])
        
        if topside:
            J=n.zeros([len(fit_idx),6])
            n_par=6

        for i in range(n_par):
            xhat1=n.copy(xhat)
            dx=0.05*xhat1[i]+0.001
            xhat1[i]=xhat1[i]+dx
            spec1=model_spec(xhat1[0]*xhat1[1],xhat1[1],xhat1[5],xhat1[2],dop_hz,topside=topside)
            spec1=n.real(n.fft.ifft(n.fft.fft(spec1)*DA))
            spec1=xhat1[3]*spec1/n.max(spec1)
            model1=spec1+xhat1[4]
            J[:,i]=(model1[fit_idx]-model[fit_idx])/dx/sigma

        sigmas=n.sqrt(n.diag(n.linalg.inv(n.dot(n.transpose(J),J))))
        
        if topside==False:
            # molecular fraction has no uncertainty
            sigmas=n.concatenate((sigmas,[n.nan]))
    except:
        traceback.print_exc()
        print("singular matrix at %1.0f km. no uncertainty determined"%(hgt))
        

    # ion line power for raw ne
    snr=n.sum(spec)/(len(spec)*xhat[4])
 #   print(xhat)
#    model=n.fft.ifft(n.fft.fft(spec)*DA) + xhat[4]
            
    if plot:
        plt.plot(dop_hz,model,label="Best fit")
        plt.plot(dop_hz,meas,".",label="Measurements")
        plt.plot(dop_hz[fit_idx],meas[fit_idx],".",label="Fit measurements")        
        plt.title("Millstone Hill\nRange %1.0f (km)\n heavy_frac=%1.2f (%1.2f)"%(hgt,xhat[5],mol_frac))
        plt.xlabel("Doppler (Hz)")
        plt.ylabel("Power (arb)")
        plt.legend()
        plt.show()

    return(xhat,model,snr,sigmas)


def fit_gaussian(meas,dop_amb,dop_hz,hgt,fit_idx,plot=True,frad=440.2e6):
    
    DA=n.fft.fft(n.fft.fftshift(dop_amb))
    peak=n.max(meas)
    noise_floor_est=n.nanmin(meas[n.abs(dop_hz)>30e3])
    pwr_est=peak-noise_floor_est

    def ss(xhat):
        vel=xhat[0]
        width=xhat[1]
        pwr=xhat[2]
        noise_floor=xhat[3]
        dopfreq=2.0*n.pi*2.0*frad*vel/c.c
        dopwidth=2.0*n.pi*2.0*frad*width/c.c        
        
        spec=(1/n.sqrt(2.0*n.pi*dopwidth**2.0))*n.exp(-0.5*(dop_hz-dopfreq)**2.0/(dopwidth**2.0))
        spec=n.fft.ifft(n.fft.fft(spec)*DA)
        spec=pwr*spec/n.max(spec)
        model=spec+noise_floor
        ss=n.nansum(n.abs(meas[fit_idx]-model[fit_idx])**2.0)
        return(ss)

    xhat=so.minimize(ss,[0.0, 100, pwr_est, noise_floor_est],method="Nelder-Mead",bounds=((-5e3,5e3),(1,3e3),(0.5*pwr_est,1.5*pwr_est),(0.5*noise_floor_est,1.5*noise_floor_est))).x
    vel=xhat[0]
    width=xhat[1]
    pwr=xhat[2]
    noise_floor=xhat[3]
    dopfreq=2.0*n.pi*2.0*frad*vel/c.c
    dopwidth=2.0*n.pi*2.0*frad*width/c.c        

    spec=(1/n.sqrt(2.0*n.pi*dopwidth**2.0))*n.exp(-0.5*(dop_hz-dopfreq)**2.0/(dopwidth**2.0))
    spec=n.fft.ifft(n.fft.fft(spec)*DA)
    spec=pwr*spec/n.max(spec)
    model=spec+noise_floor
    
    if plot:
        plt.plot(dop_hz,model,label="Best fit")
        plt.plot(dop_hz,meas,".",label="Measurements")
        plt.xlabel("Doppler (Hz)")
        plt.ylabel("Power (arb)")
        plt.legend()
        plt.show()

    return(xhat)


def fit_spectra(dirname="/media/j/fee7388b-a51d-4e10-86e3-5cabb0e1bc13/isr/2023-09-05/usrp-rx0-r_20230905T214448_20230906T040054/",n_avg=6):
    
    zpm,mpm=mrs.get_tx_power_model(dirn="%s/metadata/powermeter"%(dirname))
    azf,elf,azelb=mrs.get_misa_az_el_model(dirn="%s/metadata/antenna_conrol_metadata"%(dirname))


    fl=glob.glob("%s/range_doppler/il*.h5"%(dirname))
    fl.sort()

    sr=1e6
    h=h5py.File(fl[0],"r")
    LP=n.copy(h["E_RDS_LP"][()])
    WLP=n.copy(h["E_RDS_LP"][()])
    TX=n.copy(h["TX_LP"][()])

    dop_hz=n.copy(h["dop_hz"][()])
    rgs_km=n.copy(h["rgs_km"][()])

    ridx=[35,230]

    LP=n.zeros(LP.shape,dtype=n.float128)
    WLP=n.zeros(LP.shape,dtype=n.float128)

    h.close()

    n_freq=LP.shape[1]
    n_t=len(fl)
    overview_rg=[80,170]
    n_overview=len(overview_rg)
#    S=n.zeros([n_t,n_r,n_freq])
    n_r=LP.shape[0]
    fit_idx=n.where( (dop_hz>-30e3) & (dop_hz<20e3) )[0]

    tsys=n.zeros(n_t)

    tv=n.zeros(n_t)
    pp=n.zeros([n_t,n_r,6])
    pp_sigma=n.zeros([n_t,n_r,6])    

    pp[:,:,:]=n.nan
    pp_sigma[:,:,:]=n.nan    
    
    n_ints=int(n.floor(len(fl)/n_avg))
    for fi in range(rank,n_ints,size):
        LPA=n.zeros([n_avg,n_r,n_freq])
        LPV=n.zeros([n_avg,n_r,n_freq])


        space_object_count=n.zeros(n_r,dtype=int)
        space_object_times=[]
        space_object_rgs=[]   

        
        TX[:,:]=0.0
        i0=0
        tall=n.zeros(n_avg)
        for ai in range(n_avg):
            f=fl[fi*n_avg+ai]
            h=h5py.File(f,"r")
            if ai==0:
                i0=h["i0"][()]
                tv[fi]=i0
            tall[ai]=h["i0"][()]/1e6
            LPV[ai,:,:]=(h["RDS_LP_var"][()])
            LPA[ai,:,:]=h["RDS_LP"][()]
            
            TX+=h["TX_LP"][()]
            tsys[fi]+=h["T_sys"][()]
            h.close()
        tsys[fi]=tsys[fi]/n_avg


        ATX=TX[5:36,:]


        ATX=ATX/n.max(TX)
        dop_amb=n.sum(ATX,axis=0)
        dop_amb=dop_amb/n.sum(dop_amb)

        # check for debris by fitting gaussian spectrum
        for ai in range(n_avg):
            for ri in range(LPA.shape[1]):
                if rgs_km[ri]>300:
                    if n.sum(n.isnan(LPA[ai,ri,:])) == 0:
                        xhat=fit_gaussian(LPA[ai,ri,:],dop_amb,dop_hz,rgs_km[ri],fit_idx,plot=False)
                        if xhat[1]<100 and xhat[2]/xhat[3] > 0.5:
                            print("--->   debris %1.0f km dopwidth %1.0f m/s"%(rgs_km[ri],xhat[1]))
                            space_object_rgs.append(rgs_km[ri])
                            space_object_times.append(tall[ai])

                            # one pulse length in each direction
                            for j in range(-14,14):
                                if (j+ri > 0) and (j+ri)<LPA.shape[1]:
                                    LPA[ai,j+ri,:]=n.nan
                                    space_object_count[ri+j]+=1

        LP=n.nanmean(LPA,axis=0)

        # scale power to noise temperature
#        nf=n.nanmedian(LP[ridx[0]:ridx[1],:])

#        for ri in range(n_r):
 #           S[ri,fi,:]=(LP[ri,:]-nf)*rgs_km[ri]**2.0

  #      if True:
   #         plt.pcolormesh(10.0*n.log10(S))
    #        plt.colorbar()
     #       plt.show()

        LP2=n.copy(LP)
        LP2[:,:]=0.0

        LPM=n.copy(LP)
        LPM[:,:]=0.0

        for ri in range(ridx[0],ridx[1]):#LP.shape[0]):
            sigmas=n.array([n.nan,n.nan,n.nan,n.nan,n.nan,n.nan])
            if n.sum(n.isnan(LP[ri,:])) == 0:
                if rgs_km[ri]>400:
                    xhat,model,snr,sigmas=fit_spec(LP[ri,:],dop_amb,dop_hz,rgs_km[ri],fit_idx,plot=False)
                else:
                    xhat,model,snr,sigmas=fit_spec(LP[ri,:],dop_amb,dop_hz,rgs_km[ri],fit_idx,plot=False)            
                pp[fi,ri,:]=xhat
                pp_sigma[fi,ri,:]=sigmas
            else:
                pp[fi,ri,:]=n.nan
                pp_sigma[fi,ri,:]=n.nan                

            # get electron density from snr
            # snr = s/n = T_echo/T_sys
            # T_sys*snr = T_echo
            # T_echo = (1/magic_const) * Ptx * ne / (1+Te/Ti) / R**2.0 = T_sys*snr
            # ne = magic_const*T_sys*snr*(1+Te/Ti)/Ptx
            
            pp[fi,ri,3]=(1+xhat[0])*snr*tsys[fi]*rgs_km[ri]**2.0/zpm(i0/1e6)
            
            # approximately no error contribution from noise floor estimate on the denominator
            snr_sigma=(n.sqrt(sigmas[3]**2.0+sigmas[4]**2.0))/xhat[4]            
            pp_sigma[fi,ri,3]=(1+xhat[0])*snr_sigma*tsys[fi]*rgs_km[ri]**2.0/zpm(i0/1e6)            

            LP2[ri,:]=(model-xhat[4])/xhat[3]
            # scaled measurement
            LPM[ri,:]=(LP[ri,:]-xhat[4])/xhat[3]       

        plt.subplot(231)
        dB=10.0*n.log10(LP2[:,fit_idx])

        dB0=10.0*n.log10(LP2[:,:])
        nfl=n.nanmedian(dB0)

        plt.pcolormesh(dop_hz/1e3,rgs_km,10.0*n.log10(LP2[:,:]),vmin=-20,vmax=0,cmap="plasma")
        plt.title("Model")    
        plt.xlabel("Doppler (kHz)")
        plt.ylabel("Range (km)")    
 #       plt.colorbar()
        plt.subplot(232)
        plt.pcolormesh(dop_hz/1e3,rgs_km,10.0*n.log10(LPM[:,]),vmin=-20,vmax=0,cmap="plasma")
        plt.title("Measurement")
        plt.xlabel("Doppler (kHz)")
        plt.ylabel("Range (km)")    

  #      plt.colorbar()
        plt.subplot(233)
        plt.plot(pp[fi,:,0]*pp[fi,:,1],rgs_km,label="Te")
        plt.plot(pp[fi,:,1],rgs_km,label="Ti")
        plt.plot(pp[fi,:,2]*10,rgs_km,label="vi*10")
        plt.plot(pp[fi,:,5]*1000,rgs_km,label="rho")    
        plt.xlim([-1e3,5e3])
        plt.ylabel("Range (km)")    

#        plt.legend()
        plt.subplot(234)
        plt.pcolormesh(dop_hz[fit_idx]/1e3,rgs_km,10.0*n.log10(LP2[:,fit_idx]),vmin=-20,vmax=0,cmap="plasma")
        plt.title("Model")    
        plt.xlabel("Doppler (kHz)")
        plt.ylabel("Range (km)")    
#        plt.colorbar()
        plt.subplot(235)
        plt.pcolormesh(dop_hz[fit_idx]/1e3,rgs_km,10.0*n.log10(LPM[:,fit_idx]),vmin=-20,vmax=0,cmap="plasma")
        plt.title("Measurement")
        plt.xlabel("Doppler (kHz)")
        plt.ylabel("Range (km)")    

   #     plt.colorbar()
        plt.subplot(236)
        plt.pcolormesh(dop_hz[fit_idx]/1e3,rgs_km,LPM[:,fit_idx]-LP2[:,fit_idx],vmin=-0.5,vmax=0.5)
        plt.title("Meas-Model")
        plt.xlabel("Doppler (kHz)")
        plt.ylabel("Range (km)")    

        plt.colorbar()

        plt.tight_layout()
        plt.savefig("%s/range_doppler/pp_lp_%d.png"%(dirname,i0/1e6))
        plt.close()

        ho=h5py.File("%s/range_doppler/pp-%d.h5"%(dirname,i0/1e6),"w")
        ho["Te"]=pp[fi,:,0]*pp[fi,:,1]
        ho["Ti"]=pp[fi,:,1]
        ho["vi"]=pp[fi,:,2]
        ho["ne"]=pp[fi,:,3]
        ho["heavy_ion_frac"]=pp[fi,:,5]    

        ho["dTe/Ti"]=pp_sigma[fi,:,0]
        ho["dTi"]=pp_sigma[fi,:,1]
        ho["dvi"]=pp_sigma[fi,:,2]
        ho["dne"]=pp_sigma[fi,:,3]
        ho["dfrac"]=pp_sigma[fi,:,5]        

        ho["P_tx"]=zpm(i0/1e6)
        ho["T_sys"]=tsys[fi]

        ho["rgs"]=rgs_km
        ho["t0"]=i0/1e6
        ho["t1"]=i0/1e6+10

        ho["space_object_count"]=space_object_count
        ho["space_object_times"]=space_object_times
        ho["space_object_rgs"]=space_object_rgs

        ho["range_avg_limits_km"]=[0,1500]
        ho["range_avg_window_km"]=[480e-6*c.c/2/1e3]

        
        ho.close()

        h.close()



fit_spectra(dirname="/media/j/fee7388b-a51d-4e10-86e3-5cabb0e1bc13/isr/2023-09-28/usrp-rx0-r_20230928T211929_20230929T040533/",
            n_avg=30)
