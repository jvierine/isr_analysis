
import numpy as n
import matplotlib.pyplot as plt
import pyfftw
import stuffr
import scipy.signal as s
from digital_rf import DigitalRFReader, DigitalMetadataReader, DigitalMetadataWriter
import os
import h5py
from scipy.ndimage import median_filter

from mpi4py import MPI

comm=MPI.COMM_WORLD
size=comm.Get_size()
rank=comm.Get_rank()

# we really need to be narrow band to avoid interference. there is plenty of it.
pass_band=0.05e6

def ideal_lpf(z,sr=1e6,f0=1.2*pass_band,L=200):
    m=n.arange(-L,L)+1e-6
    om0=n.pi*f0/(0.5*sr)
    h=s.hann(len(m))*n.sin(om0*m)/(n.pi*m)
    Z=n.fft.fft(z)
    H=n.fft.fft(h,len(Z))
    z_filtered=n.roll(n.fft.ifft(Z*H),-L)
    return(z_filtered)

def estimate_dc(d_il,tmm,sid,channel):
    # estimate dc offset first
    z_dc=n.zeros(10000,dtype=n.complex64)
    n_dc=0.0
    for keyi,key in enumerate(sid.keys()):
        if sid[key] not in tmm.keys():
            print("unknown pulse, ignoring")
        # fftw "allocated vector"
        z_echo = d_il.read_vector_c81d(key, 10000, channel)
        last_echo=tmm[sid[key]]["last_echo"]        
        gc=tmm[sid[key]]["gc"]
        
        z_echo[0:(gc+4000)]=n.nan
        z_echo[last_echo:10000]=n.nan
        z_dc+=z_echo
        n_dc+=1.0
    z_dc=z_dc/n_dc

    if False:
        # plot the dc offset estimate
        plt.plot(z_dc.real)
        plt.plot(z_dc.imag)
        plt.axhline(n.nanmedian(z_dc.real))
        plt.axhline(n.nanmedian(z_dc.imag))
        print(n.nanmedian(z_dc.real))
        print(n.nanmedian(z_dc.imag))        
        plt.show()
    
    z_dc=n.complex64(n.nanmedian(z_dc.real)+n.nanmedian(z_dc.imag)*1j)
    return(z_dc)
    

def estimate_tsys(tmm,sid,key,d_il,z_echo):
    noise0=tmm[sid[key]]["noise0"]
    noise1=tmm[sid[key]]["noise1"]
    last_echo=tmm[sid[key]]["last_echo"]        
    tx0=tmm[sid[key]]["tx0"]
    tx1=tmm[sid[key]]["tx1"]
    gc=tmm[sid[key]]["gc"]                

    inj_plus_noise=n.mean(n.abs(z_echo[noise0:noise1])**2.0)
    noise=n.mean(n.abs(z_echo[(last_echo-500):last_echo])**2.0)
    # https://en.wikipedia.org/wiki/Median
    inj_plus_noise2=(n.median(n.abs(z_echo[noise0:noise1]))/0.7746)**2.0
    noise2=(n.median(n.abs(z_echo[(last_echo-500):last_echo]))/0.7746)**2.0

    # (tsys+tinj)*alpha = inj_plus_noise
    # tsys*alpha = noise
    # inj_plus_noise - noise = alpha*tinj
    # (inj_plus_noise - noise)/tinj = alpha
    # noise/alpha = tsys
    alpha=(inj_plus_noise-noise)/T_injection 
    T_sys=noise/alpha

    alpha2=(inj_plus_noise2-noise2)/T_injection
    T_sys2=noise2/alpha
    return(T_sys,T_sys2)

def range_dop_spec(z_echo,z_tx,rgs,tx0,tx1,fftlen):
    n_rg=len(rgs)
    RDS=n.zeros([n_rg,fftlen],dtype=n.float32)
    z_tx=n.copy(n.conj(z_tx))
    txlen=tx1
    # try to reduce the doppler spread of the ambiguity function
    wf=1.0#s.hann(txlen)
    for ri in range(n_rg):
        RDS[ri,:]=n.abs(n.fft.fftshift(n.fft.fft(wf*z_tx[0:tx1]*z_echo[ (rgs[ri]):(rgs[ri]+txlen) ],fftlen)))**2.0
    return(RDS)

tmm = {}
T_injection=1172.0 # May 24th 2022 value
tmm[300]={"noise0":7800,"noise1":8371,"tx0":76,"tx1":645,"gc":1000,"last_echo":7700,"e_gc":800}
for i in range(1,33):
    tmm[i]={"noise0":8400,"noise1":8850,"tx0":76,"tx1":624,"gc":1000,"last_echo":8200,"e_gc":800}

# one minute averaging time
avg_dur=60.0

id_read = DigitalMetadataReader("/media/j/fee7388b-a51d-4e10-86e3-5cabb0e1bc13/isr/2023-09-05/usrp-rx0-r_20230905T214448_20230906T040054/metadata/id_metadata")
d_il = DigitalRFReader("/media/j/fee7388b-a51d-4e10-86e3-5cabb0e1bc13/isr/2023-09-05/usrp-rx0-r_20230905T214448_20230906T040054/rf_data/")
channel="zenith-l"

idb=id_read.get_bounds()
# sample rate for metadata
idsr=1000000
sr=1000000

output_prefix="tmpe"
plot_voltage=False
use_ideal_filter=True
debug_gc_rem=False
reanalyze=True
n_times = int(n.floor((idb[1]-idb[0])/idsr/avg_dur))

range_shift=600
# 30 microsecond range gates
fftlen=1024
rg=30
n_rg=int(n.floor(8000/rg))
rgs=n.arange(n_rg)*rg
rgs_km=rgs*0.15
dop_hz=n.fft.fftshift(n.fft.fftfreq(fftlen,d=1/sr))
freq_idx = n.where( n.abs(dop_hz) < pass_band )[0]
n_freq=len(freq_idx)
fi0=n.min(freq_idx)
fi1=n.max(freq_idx)+1

i0=idb[0]
# go through one integration window
for ai in range(rank,n_times,size):
    i0 = ai*int(avg_dur*idsr) + idb[0]

    if os.path.exists("%s/il_%d.png"%(output_prefix,int(i0/1e6))) and reanalyze==False:
        print("already analyzed %d"%(i0/1e6))
        continue

    
    # get info on all the pulses transmitted during this averaging interval
    # get some extra for gc
    sid = id_read.read(i0,i0+int(avg_dur*idsr)+40000,"sweepid")

    n_pulses=len(sid.keys())

    RDS_LP=n.zeros([n_pulses,n_rg,n_freq],dtype=n.float32)
    RDS_AC=n.zeros([n_pulses,n_rg,n_freq],dtype=n.float32)
    
    E_RDS_LP=n.zeros([n_pulses,n_rg,n_freq],dtype=n.float32)
    E_RDS_AC=n.zeros([n_pulses,n_rg,n_freq],dtype=n.float32)

    RDS_LP_var=n.zeros([n_rg,n_freq],dtype=n.float32)
    RDS_AC_var=n.zeros([n_rg,n_freq],dtype=n.float32)

    E_RDS_LP_var=n.zeros([n_rg,n_freq],dtype=n.float32)
    E_RDS_AC_var=n.zeros([n_rg,n_freq],dtype=n.float32)
    
    # averaging weights
    W_LP=n.zeros(n_pulses,dtype=n.float32)
    W_AC=n.zeros(n_pulses,dtype=n.float32)

    E_W_LP=n.zeros(n_pulses,dtype=n.float32)
    E_W_AC=n.zeros(n_pulses,dtype=n.float32)
    
    TX_RDS_LP=n.zeros([n_rg,n_freq],dtype=n.float32)
    TX_RDS_AC=n.zeros([n_rg,n_freq],dtype=n.float32)

    # sum of weights
    WS_LP=0.0
    WS_AC=0.0

    E_WS_LP=0.0
    E_WS_AC=0.0

    ac_idx=0
    lp_idx=0    
    

    T_sys_a=[]
    T_sys_m=[]

    # estimate dc offset. slow but necessary if not known.
    #    z_dc=estimate_dc(d_il,tmm,sid,channel)
    #    z_dc=estimate_dc(d_il,tmm,sid,channel)
    
    # USRP DC offset bug due to truncation instead of rounding.
    # Ryan Volz has a fix for firmware in USRPs. 
    z_dc=n.complex64(-0.212-0.221j)

    
    bg_samples=[]
    bg_plus_inj_samples=[]

    sidkeys=list(sid.keys())
#    n_pulses=len(sidkeys)

    # start at 3, because we may need to look back for GC
    for keyi in range(3,n_pulses-3):
        key=sidkeys[keyi]

        if sid[key] not in tmm.keys():
            print("unknown pulse code %d encountered, halting."%(sid[key]))
            exit(0)
        
        z_echo=None
        zd=None
        if sid[key] == 300:
            next_key = sidkeys[keyi+3]
            z_echo = d_il.read_vector_c81d(key, 10000, channel) - z_dc
            z_echo=ideal_lpf(z_echo)            
            z_echo1 = d_il.read_vector_c81d(next_key, 10000, channel) - z_dc
            z_echo1=ideal_lpf(z_echo1)                        

            zd=z_echo-z_echo1
            if debug_gc_rem:
                plt.plot(zd.real+2000)
                plt.plot(zd.imag+2000)
                plt.plot(z_echo.real)
                plt.plot(z_echo.imag)
                
                plt.title(sid[key])
                plt.show()
        elif sid[key] == sid[sidkeys[keyi+1]]:
            next_key = sidkeys[keyi+1]
            z_echo = d_il.read_vector_c81d(key, 10000, channel) - z_dc
            z_echo=ideal_lpf(z_echo)
            z_echo1 = d_il.read_vector_c81d(next_key, 10000, channel) - z_dc
            z_echo1=ideal_lpf(z_echo1)            

            zd=z_echo-z_echo1
            if debug_gc_rem:
                plt.plot(zd.real+2000)
                plt.plot(zd.imag+2000)
                plt.plot(z_echo.real)
                plt.plot(z_echo.imag)            
                plt.title(sid[key])            
                plt.show()
        elif sid[key] == sid[sidkeys[keyi-1]]:
            next_key = sidkeys[keyi-1]
            z_echo = d_il.read_vector_c81d(key, 10000, channel) - z_dc
            z_echo=ideal_lpf(z_echo)
            z_echo1 = d_il.read_vector_c81d(next_key, 10000, channel) - z_dc
            z_echo1=ideal_lpf(z_echo1)            

            zd=z_echo-z_echo1
            if debug_gc_rem:
                plt.plot(zd.real+2000)
                plt.plot(zd.imag+2000)
                plt.plot(z_echo.real)
                plt.plot(z_echo.imag)            
                plt.title(sid[key])            
                plt.show()
                    
        print("%d/%d"%(keyi,n_pulses))


        T_sys,T_sys2=estimate_tsys(tmm,sid,key,d_il,z_echo)
        print("%d/%d T_sys %1.0f K"%(keyi,n_pulses,T_sys))

#        if use_ideal_filter:
 #           z_echo=ideal_lpf(z_echo)

        noise0=tmm[sid[key]]["noise0"]
        noise1=tmm[sid[key]]["noise1"]
        last_echo=tmm[sid[key]]["last_echo"]        
        tx0=tmm[sid[key]]["tx0"]
        tx1=tmm[sid[key]]["tx1"]
        gc=tmm[sid[key]]["gc"]
        e_gc=tmm[sid[key]]["e_gc"]        

        bg_samples.append( n.mean(n.abs(z_echo[(last_echo-500):last_echo])**2.0) )
        bg_plus_inj_samples.append( n.mean(n.abs(z_echo[(noise0):noise1])**2.0) )
  
        if False:
            plt.plot(z_echo.real)
            plt.plot(z_echo.imag)
            plt.axvline(noise0)
            plt.axvline(noise1)
            plt.axvline(gc)
            plt.axvline(tx0)
            plt.axvline(tx1)
            plt.show()
        
        z_tx=n.copy(z_echo)
        z_tx[0:tx0]=0.0
        z_tx[tx1:10000]=0.0
        # normalize tx pwr
        z_tx=z_tx/n.sqrt(n.sum(n.abs(z_tx)**2.0))
        z_echo[0:gc]=0.0
        # e-region
        zd[0:e_gc]=0.0
        z_echo[last_echo:10000]=0.0

        # use this to estimate the range-Doppler ambiguity function
        z_tx2=n.copy(z_tx)
        z_tx2=n.roll(z_tx2,range_shift)

        RDS=range_dop_spec(z_echo,z_tx,rgs,tx0,tx1,fftlen)
        TX_RDS=range_dop_spec(z_tx2,z_tx,rgs,tx0,tx1,fftlen)

        if False:
            plt.plot(zd.real)
            plt.plot(zd.imag)
            plt.show()

        
        E_RDS=range_dop_spec(zd,z_tx,rgs,tx0,tx1,fftlen)       
        
        if sid[key] == 300:
            # uncoded long pulse
            RDS_LP[lp_idx,:,:]=RDS[:,fi0:fi1]
            E_RDS_LP[lp_idx,:,:]=E_RDS[:,fi0:fi1]


            
#            W_LP[lp_idx]=1/( (n.max(RDS_LP[lp_idx,:,:],axis=1)/1e4)**2.0)

            
            if False:
#                plt.subplot(121)
 #               plt.pcolormesh(W_LP[lp_idx,:,:])
  #              plt.colorbar()
   #             plt.subplot(122)
                plt.pcolormesh(RDS_LP[lp_idx,:,:])            
                plt.show()
                
            TX_RDS_LP[:,:]+=TX_RDS[:,fi0:fi1]
            lp_idx+=1
        else:

#            bg_samples.append( n.mean(n.abs(z_echo_o[(last_echo-500):last_echo])**2.0) )
 #           bg_plus_inj_samples.append( n.mean(n.abs(z_echo_o[(noise0):noise1])**2.0) )
            
            # coded long pulse
            RDS_AC[ac_idx,:,:]=RDS[:,fi0:fi1]
            E_RDS_AC[lp_idx,:,:]=E_RDS[:,fi0:fi1]
            TX_RDS_AC[:,:]+=TX_RDS[:,fi0:fi1]
            
#            W_AC[ac_idx]=1/( (n.max(RDS_AC[ac_idx,:,:],axis=1)/1e4)**2.0)

            if False:
#                plt.subplot(121)
 #               plt.pcolormesh(W_AC[ac_idx,:,:])
  #              plt.colorbar()
   #             plt.subplot(122)
                plt.pcolormesh(RDS_AC[ac_idx,:,:])            
                plt.show()
            
            
            ac_idx+=1


        if plot_voltage:
            plt.subplot(311)
            plt.plot(z_echo.real)
            plt.plot(z_echo.imag)
            plt.title("tsys %1.0f,%1.0f K"%(T_sys,T_sys2))
            plt.subplot(312)
            plt.plot(z_tx.real)
            plt.plot(z_tx.imag)
            plt.title(sid[key])
            plt.subplot(313)
            plt.plot(z_tx2.real)
            plt.plot(z_tx2.imag)
            plt.show()

    # distribution statistics
    # there can be a lot of outliers, so we estimate 0.5 sigma
    # from the distribution


    noise=n.median(bg_samples)    
    alpha=(n.median(bg_plus_inj_samples)-n.median(bg_samples))/T_injection 
    T_sys=noise/alpha

    if False:
        # debug noise cal
        plt.plot(bg_samples,".")
        plt.axhline(n.median(bg_samples))
        plt.plot(bg_plus_inj_samples,".")
        plt.axhline(n.median(bg_plus_inj_samples))    
        plt.title(T_sys)
        plt.show()
    
    
    # this is the most aggressive method I know of doing an average, but it might 
    # be a bit too aggressive. I don't even know if this is an unbiased estimator
    use_median_mean=False
    if use_median_mean:
        RDS_LP=n.median(RDS_LP[0:lp_idx,:,:],axis=0)
        RDS_AC=n.median(RDS_AC[0:ac_idx,:,:],axis=0)
    else:

        lp_sigma_est=n.percentile(RDS_LP[0:lp_idx,:,:],34,axis=0)*2
        ac_sigma_est=n.percentile(RDS_AC[0:ac_idx,:,:],34,axis=0)*2

        # smooth a bit
        lp_sigma_est=median_filter(lp_sigma_est,11)
        ac_sigma_est=median_filter(ac_sigma_est,11)

        if False:
            plt.pcolormesh(ac_sigma_est)
            plt.colorbar()
            plt.show()


        RDS_LP_var=n.var(RDS_LP[0:lp_idx,:,:],axis=0)
        RDS_AC_var=n.var(RDS_AC[0:ac_idx,:,:],axis=0) 


        n_avg0=6
        lp_idx=int(n.floor(lp_idx/n_avg0)*n_avg0)
        for i in range(0,lp_idx,n_avg0):
            RDS_THIS=n.mean(RDS_LP[i:(i+n_avg0),:,:],axis=0)
            ratio_test=RDS_THIS/lp_sigma_est

            
            if False:
                plt.subplot(121)
                plt.pcolormesh(ratio_test)
                plt.colorbar()

            # remove range ambiguity length around from every detected hard target echo
            # outlier reject 5-sigma            
            bidx=n.where(ratio_test > 5)[0]
            for bi in bidx:
                print("Hard target at %1.0f km"%(rgs_km[bi]))
                # take a bit extra after
                for j in range(n_avg0*2):
                    RDS_LP[i+j,(n.max([0,bi-16])):(n.min([n_rg,bi+16])),:]=n.nan


            RDS_THIS=n.mean(RDS_LP[i:(i+n_avg0),:,:],axis=0) 
 #           W_LP[i]=1.0/n.nanmax(RDS_THIS)
  #          WS_LP+=W_LP[i]
#            RDS_LP[i:(i+n_avg0),:,:]=RDS_LP[i:(i+n_avg0),:,:]*W_LP[i]
   #         WS_LP+=W_LP[i]*n_avg0
                
            if False:
                plt.subplot(122)
                plt.pcolormesh(RDS_THIS)
                plt.colorbar()
                plt.show()


        RDS_LP=n.nanmean(RDS_LP[0:lp_idx,:,:],axis=0)#/WS_LP

        E_RDS_LP=n.nanmean(E_RDS_LP[0:lp_idx,:,:],axis=0)#/WS_LP        

        ac_idx=int(n.floor(ac_idx/n_avg0))*n_avg0
        for i in range(0,ac_idx,n_avg0):
            RDS_THIS=n.mean(RDS_AC[i:(i+n_avg0),:,:],axis=0)
            ratio_test=RDS_THIS/ac_sigma_est
            if False:
                plt.subplot(121)
                plt.pcolormesh(ratio_test)
                plt.colorbar()

            # remove range ambiguity length around from every detected hard target echo
            # reject 5-sigma outliers. might be a bit too aggressive...
            bidx=n.where(ratio_test > 5)[0]
            for bi in bidx:
                print("Hard target at %1.0f km"%(rgs_km[bi]))
                # remove some extra ipps, where the detection might be weaker
                for j in range(n_avg0*2):
                    RDS_AC[i+j,(n.max([0,bi-16])):(n.min([n_rg,bi+16])),:]=n.nan
#                    RDS_AC[i+1,(n.max([0,bi-16])):(n.min([n_rg,bi+16])),:]=n.nan
 #                   RDS_AC[i+2,(n.max([0,bi-16])):(n.min([n_rg,bi+16])),:]=n.nan            

            RDS_THIS=n.mean(RDS_AC[i:(i+n_avg0),:,:],axis=0)
      #      W_AC[i]=1.0/n.nanmax(RDS_THIS)
       #     WS_AC+=W_AC[i]
        #    RDS_AC[i:(i+n_avg0),:,:]=RDS_AC[i:(i+n_avg0),:,:]*W_AC[i]
         #   WS_AC+=W_AC[i]*n_avg0
            
            if False:
                plt.subplot(122)
                plt.pcolormesh(RDS_THIS)
                plt.colorbar()
                plt.show()

        RDS_AC=n.nanmean(RDS_AC[0:ac_idx,:,:],axis=0)#/WS_AC
        E_RDS_AC=n.nanmean(E_RDS_AC[0:ac_idx,:,:],axis=0)#/WS_AC        

    # noise floor only in filter pass band
    range_idx = n.where( (rgs_km > 700) & (rgs_km < 1200) )[0]    
    
    noise_lp=n.median(RDS_LP[n.min(range_idx):n.max(range_idx),:])
    noise_ac=n.median(RDS_AC[n.min(range_idx):n.max(range_idx),:])

    e_noise_lp=n.median(E_RDS_LP[n.min(range_idx):n.max(range_idx),:])
    e_noise_ac=n.median(E_RDS_AC[n.min(range_idx):n.max(range_idx),:])
    
    plot_amb=False
    plt.figure(figsize=(12,6))
    if plot_amb:
        plt.subplot(221)
    else:
        plt.subplot(121)
        
    snr_ac=(RDS_AC-noise_ac)/noise_ac
    
    plt.pcolormesh(dop_hz[fi0:fi1]/1e3,rgs_km,snr_ac,vmin=-0.5,vmax=5,cmap="plasma")
    cb=plt.colorbar()
    cb.set_label("AC SNR")
    plt.xlim([-50,50])        
    plt.title("%s"%(stuffr.unix2datestr(i0/sr)))
    plt.ylabel("Range (km)")        
    plt.xlabel("Doppler (kHz)")
    if plot_amb:
        plt.subplot(222)
    else:
        plt.subplot(122)        
    
#    plt.subplot(222)
    snr_lp=(RDS_LP-noise_lp)/noise_lp
    
    plt.pcolormesh(dop_hz[fi0:fi1]/1e3,rgs_km,snr_lp,vmin=-0.5,vmax=20,cmap="plasma")
    cb=plt.colorbar()
    cb.set_label("LP SNR")
    plt.xlim([-50,50])        
    plt.title("Millstone Hill LO T_sys=%1.0f K"%(T_sys))
    plt.ylabel("Range (km)")
    plt.xlabel("Doppler (kHz)")
    if plot_amb:
        plt.subplot(223)
        TX_RDS_AC=TX_RDS_AC/n.max(TX_RDS_AC)
        plt.pcolormesh(dop_hz/1e6,rgs_km-range_shift*0.15,TX_RDS_AC)
        plt.colorbar()    
        plt.ylim([-100,100])
        plt.xlim([-0.1,0.1])
        plt.title("AC 16x30")
        plt.ylabel("Range (km)")        
        plt.xlabel("Doppler (MHz)")
        plt.subplot(224)
        TX_RDS_LP=TX_RDS_LP/n.max(TX_RDS_LP)    
        plt.pcolormesh(dop_hz[fi0:fi1]/1e6,rgs_km-range_shift*0.15,TX_RDS_LP)
        plt.colorbar()    
        plt.ylim([-100,100])
        plt.xlim([-0.05,0.05])
        plt.title("LP 480")
        plt.ylabel("Range (km)")
        plt.xlabel("Doppler (MHz)")
    plt.tight_layout()
    plt.savefig("%s/il_%d.png"%(output_prefix,int(i0/1e6)))
    plt.close()
    plt.clf()


    plt.figure(figsize=(12,6))
    plt.subplot(121)
        
    snr_e_ac=(E_RDS_AC-e_noise_ac)/e_noise_ac    
    
    plt.pcolormesh(dop_hz[fi0:fi1]/1e3,rgs_km,snr_e_ac,vmin=-0.5,vmax=5,cmap="plasma")
    cb=plt.colorbar()
    cb.set_label("AC SNR")
    plt.xlim([-50,50])        
    plt.title("%s"%(stuffr.unix2datestr(i0/sr)))
    plt.ylabel("Range (km)")        
    plt.xlabel("Doppler (kHz)")

    plt.subplot(122)        

    snr_e_lp=(E_RDS_LP-e_noise_lp)/e_noise_lp
    
    plt.pcolormesh(dop_hz[fi0:fi1]/1e3,rgs_km,snr_e_lp,vmin=-0.5,vmax=20,cmap="plasma")
    cb=plt.colorbar()
    cb.set_label("LP SNR")
    plt.xlim([-50,50])        
    plt.title("Millstone Hill LO T_sys=%1.0f K"%(T_sys))
    plt.ylabel("Range (km)")
    plt.xlabel("Doppler (kHz)")
    plt.tight_layout()
    plt.savefig("%s/eil_%d.png"%(output_prefix,int(i0/1e6)))
    plt.close()
    plt.clf()
    

    ho=h5py.File("tmpe/il_%d.h5"%(int(i0/1e6)),"w")
    ho["RDS_LP"]=RDS_LP
    ho["RDS_AC"]=RDS_AC    
    ho["E_RDS_LP"]=E_RDS_LP
    ho["E_RDS_AC"]=E_RDS_AC    

    ho["RDS_LP_var"]=RDS_LP_var
    ho["RDS_AC_var"]=RDS_AC_var
    ho["i0"]=i0
    ho["dop_hz"]=dop_hz[fi0:fi1]
    ho["rgs_km"]=rgs_km
    ho["TX_LP"]=TX_RDS_LP
    ho["TX_AC"]=TX_RDS_AC
    ho["range_shift"]=range_shift
    ho["rg"]=rg
    ho["T_sys"]=T_sys
    ho.close()
    
        # Range-Doppler ambiguity function at 600 samples offset
        #TXRDS=range_dop_spec(z_echo,z_tx,rgs,tx0,tx1)        
        
    

        
#        if sid[key] not in pl_dict.keys():
 #           sweep = tmm.getTimingSweepByID(sid[key])
  #          code = sweep.getCodeObject()
   #         pl = long(code.getPulseLength()/1000) # convert from nanoseconds to microseconds
    #        pl_dict[sid[key]] = pl # pl = pulse length
     #   else:
      #      # already know id
       #     pl = pl_dict[sid[key]]
#        # keep count
 #       if pl_count.has_key(pl):99
  #          pl_count[pl] += 1
   #     else:
    #        pl_count[pl] = 1
#
print(idb)
