
import numpy as n
import matplotlib.pyplot as plt
import pyfftw
import stuffr
import scipy.signal as s
from digital_rf import DigitalRFReader, DigitalMetadataReader, DigitalMetadataWriter
import os
import h5py
from scipy.ndimage import median_filter
import traceback

import millstone_radar_state as mrs

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
    wf=s.hann(txlen)
    for ri in range(n_rg):
        RDS[ri,:]=n.abs(n.fft.fftshift(n.fft.fft(wf*z_tx[0:tx1]*z_echo[ (rgs[ri]):(rgs[ri]+txlen) ],fftlen)))**2.0
    return(RDS)

tmm = {}
T_injection=1172.0 # May 24th 2022 value
tmm[300]={"noise0":7800,"noise1":8371,"tx0":76,"tx1":645,"gc":1000,"last_echo":7700,"e_gc":800}
for i in range(1,33):
    tmm[i]={"noise0":8400,"noise1":8850,"tx0":76,"tx1":624,"gc":1000,"last_echo":8200,"e_gc":800}


def avg_range_doppler_spectra(dirname="/media/j/fee7388b-a51d-4e10-86e3-5cabb0e1bc13/isr/2023-09-05/usrp-rx0-r_20230905T214448_20230906T040054/",
                              save_png=True,
                              avg_dur=10,
                              step=10,
                              min_tx_pulses=100,
                              reanalyze=False,
                              channel="zenith-l"
                              ):


    id_read = DigitalMetadataReader("%s/metadata/id_metadata"%(dirname))
    d_il = DigitalRFReader("%s/rf_data/"%(dirname))

    zpm,mpm=mrs.get_tx_power_model("%s/metadata/powermeter"%(dirname))
    tx_ant,rx_ant=mrs.get_antenna_select("%s/metadata/antenna_control_metadata"%(dirname))    
    
    os.system("mkdir -p %s/range_doppler/%s"%(dirname,channel))
    idb=id_read.get_bounds()
    # sample rate for metadata
    idsr=1000000
    sr=1000000
    # min transmit power required to produce an estimate of range-Doppler spectra. lower powers ignored.
    min_tx_pwr=400e3

    plot_voltage=False
    use_ideal_filter=True
    debug_gc_rem=False

    n_times = int(n.floor(((idb[1]-idb[0])-avg_dur)/idsr/step))

    # range shift needed for estimating the range-Doppler ambiguity function
    # this needs longer than the transmit pulse length
    range_shift=600
    # 30 microsecond range gates
    fftlen=4096
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
        i0 = ai*int(step*idsr) + idb[0]

        if os.path.exists("%s/range_doppler/%s/il_%d.png"%(dirname,channel,int(i0/1e6))) and reanalyze==False:
            print("already analyzed %d"%(i0/1e6))
            continue


        # get info on all the pulses transmitted during this averaging interval
        # get some extra for gc
        sid = id_read.read(i0,i0+int(avg_dur*idsr)+40000,"sweepid")

        n_pulses=len(sid.keys())

        RDS_LP=n.zeros([n_pulses,n_rg,n_freq],dtype=n.float32)
        RDS_LP_var=n.zeros([n_rg,n_freq],dtype=n.float32)

        # averaging weights
        W_LP=n.zeros(n_pulses,dtype=n.float32)

        # range-doppler ambiguity function
        TX_RDS_LP=n.zeros([n_rg,n_freq],dtype=n.float32)

        # sum of weights
        WS_LP=0.0

        lp_idx=0    

        T_sys_a=[]
        T_sys_m=[]

        # USRP DC offset bug due to truncation instead of rounding.
        # Ryan Volz has a fix for firmware in USRPs.
        
        z_dc=n.complex64(-0.212-0.221j)
        if channel=="zenith-l2":
            z_dc=0.0

        bg_samples=[]
        bg_plus_inj_samples=[]

        avg_tx_pwr=0.0
        avg_tx_pwr_samples=0

        sidkeys=list(sid.keys())
    #    n_pulses=len(sidkeys)

        for keyi in range(0,n_pulses):
            key=sidkeys[keyi]

            if sid[key] not in tmm.keys():
                print("unknown pulse code %d encountered, skipping."%(sid[key]))
                continue

            z_echo=None
            z_tx=None
            print(sid[key])
            zenith_pwr=zpm(key/1e6)
            misa_pwr=mpm(key/1e6)            
            if sid[key] == 300:

                if (tx_ant(key) < -0.99) and (rx_ant(key) < -0.99) and (channel == "zenith-l") and (zenith_pwr > min_tx_pwr):
                    try:
                        z_echo = d_il.read_vector_c81d(key, 10000, "zenith-l") - z_dc
                        z_tx = d_il.read_vector_c81d(key, 10000, "tx-h")# - z_dc                        
                        avg_tx_pwr+=zenith_pwr
                        avg_tx_pwr_samples+=1
                    except:
                        traceback.print_exc()                        
                        print("couldn't read %d %s"%(key,"zenith-l"))
                        continue
                elif (tx_ant(key) > 0.99) and (rx_ant(key) > 0.99) and (channel == "misa-l") and (misa_pwr > min_tx_pwr):
                    try:
                        z_echo = d_il.read_vector_c81d(key, 10000, "misa-l") - z_dc
                        z_tx = d_il.read_vector_c81d(key, 10000, "tx-h")# - z_dc                                                
                        avg_tx_pwr+=misa_pwr
                        avg_tx_pwr_samples+=1
                    except:
                        traceback.print_exc()
                        print("couldn't read %d %s %s"%(key,"misa-l",stuffr.unix2datestr(key/1e6)))
                        continue
                else:
                    print("not enough tx power (%1.0f,%1.0f MW) on %s skipping this pulse"%(zpm(key/1e6)/1e6,mpm(key/1e6)/1e6,channel))
                    continue
                    
                
            else:
                continue

            T_sys,T_sys2=estimate_tsys(tmm,sid,key,d_il,z_echo)
            print("%d found %d pulses in %s T_sys %1.0f K"%(rank,lp_idx,channel,T_sys))

            noise0=tmm[sid[key]]["noise0"]
            noise1=tmm[sid[key]]["noise1"]
            last_echo=tmm[sid[key]]["last_echo"]        
            tx0=tmm[sid[key]]["tx0"]
            tx1=tmm[sid[key]]["tx1"]
            gc=tmm[sid[key]]["gc"]

            z_tx[0:tx0]=0.0
            z_tx[tx1:10000]=0.0

            # tbd:
            # this is eyeballed by comparing with leakthrough tx in echo
            # this is _not_ a good way to get absolute altitudes. we really should use an analog switch
            # to interleave the 
            # tx sample into the echo channel to get the correct relative delay
            # now we only get 1 us accuracy.
            # interpolate this by a lot (e.g., 100) and XC with tx-h and leakthrough to determine the signal processing channel delay
            z_tx=n.roll(z_tx,11)
            
            if False:
                plt.plot(4*n.abs(z_tx)/100)
                plt.plot(n.abs(z_echo))
                plt.show()

            z_echo=ideal_lpf(z_echo)
            z_tx=ideal_lpf(z_tx)
            bg_samples.append( n.mean(n.abs(z_echo[(last_echo-500):last_echo])**2.0) )
            bg_plus_inj_samples.append( n.mean(n.abs(z_echo[(noise0):noise1])**2.0) )

            
            # normalize tx pwr
            z_tx=z_tx/n.sqrt(n.sum(n.abs(z_tx)**2.0))
            z_echo[0:gc]=0.0
            z_echo[last_echo:10000]=0.0

            # use this to estimate the range-Doppler ambiguity function
            z_tx2=n.copy(z_tx)
            z_tx2=n.roll(z_tx2,range_shift)

            RDS=range_dop_spec(z_echo,z_tx,rgs,tx0,tx1,fftlen)
            TX_RDS=range_dop_spec(z_tx2,z_tx,rgs,tx0,tx1,fftlen)

            # this could be the error
            RDS_LP[lp_idx,:,:]=RDS[:,fi0:fi1]
            TX_RDS_LP[:,:]+=TX_RDS[:,fi0:fi1]
            lp_idx+=1
            
        if lp_idx < min_tx_pulses:
            print("less than %d pulses found. skipping this integration period"%(min_tx_pulses))
            continue
        
        noise=n.median(bg_samples)
        alpha=(n.median(bg_plus_inj_samples)-n.median(bg_samples))/T_injection 
        T_sys=noise/alpha

        # this is the most aggressive method I know of doing an average, but it might 
        # be a bit too aggressive. I don't even know if this is an unbiased estimator
        use_median_mean=False
        if use_median_mean:
            RDS_LP=n.median(RDS_LP[0:lp_idx,:,:],axis=0)
        else:
            # distribution statistics
            # there can be a lot of outliers, so we estimate 0.5 sigma
            # from the distribution

            lp_sigma_est=n.percentile(RDS_LP[0:lp_idx,:,:],34,axis=0)*2

            # smooth a bit
            lp_sigma_est=median_filter(lp_sigma_est,11)

            RDS_LP_var=n.var(RDS_LP[0:lp_idx,:,:],axis=0)

            n_avg0=6
            lp_idx=int(n.floor(lp_idx/n_avg0)*n_avg0)
            for i in range(0,lp_idx,n_avg0):
                RDS_THIS=n.mean(RDS_LP[i:(i+n_avg0),:,:],axis=0)
                ratio_test=RDS_THIS/lp_sigma_est
                
                # remove range ambiguity length around from every detected hard target echo
                # outlier reject 5-sigma            
                bidx=n.where(ratio_test > 7)[0]
                for bi in bidx:
#                    print("Hard target at %1.0f km"%(rgs_km[bi]))
                    # take a bit extra after
                    for j in range(n_avg0*2):
                        RDS_LP[i+j,(n.max([0,bi-16])):(n.min([n_rg,bi+16])),:]=n.nan

            RDS_LP=n.nanmean(RDS_LP[0:lp_idx,:,:],axis=0)


        # scale to kelvins
        # tbd, evaluate if this should be done in fit_lp.py on a longer cadence to
        # avoid outliers in gain
        RDS_LP=RDS_LP/alpha
        # we need to scale, as counts can vary a lot, the receiver gain also changes quite a lot
        RDS_LP_var=RDS_LP_var/alpha/lp_idx
            
        # noise floor only in filter pass band
        range_idx = n.where( (rgs_km > 700) & (rgs_km < 1200) )[0]    

        noise_lp=n.median(RDS_LP[n.min(range_idx):n.max(range_idx),:])

        plot_amb=False
        plt.figure(figsize=(12,6))

        gc_rg0=n.where(rgs_km > 200)[0][0]
        snr_lp=(RDS_LP-noise_lp)/noise_lp

        if lp_idx > min_tx_pulses:

            plt.pcolormesh(dop_hz[fi0:fi1]/1e3,rgs_km,snr_lp,vmin=-0.5,vmax=20,cmap="plasma")
            cb=plt.colorbar()
            cb.set_label("LP SNR")
            plt.xlim([-50,50])        
            plt.title("Millstone Hill T$_{\\mathrm{sys}}$=%1.0f K %s"%(T_sys,stuffr.unix2datestr(i0/1e6)))
            plt.ylabel("Range (km)")
            plt.xlabel("Doppler (kHz)")
            plt.tight_layout()
            plt.savefig("%s/range_doppler/%s/il_%d.png"%(dirname,channel,int(i0/1e6)))
            plt.close()
            plt.clf()

            ho=h5py.File("%s/range_doppler/%s/il_%d.h5"%(dirname,channel,int(i0/1e6)),"w")
            # long 
            ho["RDS_LP"]=RDS_LP
            ho["RDS_LP_var"]=RDS_LP_var
            ho["n_pulses"]=lp_idx
            ho["i0"]=i0
            ho["i1"]=i0+avg_dur*idsr
            ho["dop_hz"]=dop_hz[fi0:fi1]
            ho["rgs_km"]=rgs_km
            ho["TX_LP"]=TX_RDS_LP
            ho["range_shift"]=range_shift
            ho["rg"]=rg
            ho["T_sys"]=T_sys
            ho["alpha"]=alpha
            ho["channel"]=channel
            ho["P_tx"]=avg_tx_pwr/avg_tx_pwr_samples
            ho.close()

if __name__ == "__main__":

    dirs=["/media/j/fee7388b-a51d-4e10-86e3-5cabb0e1bc13/isr/2021-12-03a/usrp-rx0-r_20211203T224500_20211204T160000/",
          "/media/j/fee7388b-a51d-4e10-86e3-5cabb0e1bc13/isr/2021-12-01/usrp-rx0-r_20211201T230000_20211202T160100/",
          "/media/j/fee7388b-a51d-4e10-86e3-5cabb0e1bc13/isr/2023-09-05/usrp-rx0-r_20230905T214448_20230906T040054",
          "/media/j/fee7388b-a51d-4e10-86e3-5cabb0e1bc13/isr/2023-09-24/usrp-rx0-r_20230924T200050_20230925T041059/",
          "/media/j/fee7388b-a51d-4e10-86e3-5cabb0e1bc13/isr/2023-09-28/usrp-rx0-r_20230928T211929_20230929T040533/",
          "/media/j/fee7388b-a51d-4e10-86e3-5cabb0e1bc13/isr/2021-12-03/usrp-rx0-r_20211203T000000_20211203T033600/",
          "/media/j/fee7388b-a51d-4e10-86e3-5cabb0e1bc13/isr/2021-12-05/usrp-rx0-r_20211205T000000_20211205T160100/",
          "/media/j/fee7388b-a51d-4e10-86e3-5cabb0e1bc13/isr/2021-12-06/usrp-rx0-r_20211206T000000_20211206T132500/",
          "/media/j/fee7388b-a51d-4e10-86e3-5cabb0e1bc13/isr/2021-12-21/usrp-rx0-r_20211221T125500_20211221T220000/"]

    for d in dirs:
        try:
            avg_range_doppler_spectra(dirname=d,
                                      channel="zenith-l",
                                      save_png=True,
                                      avg_dur=10,
                                      reanalyze=True
                                      )
        except:
            traceback.print_exc()
        try:
            avg_range_doppler_spectra(dirname=d,
                                      channel="misa-l",
                                      save_png=True,
                                      avg_dur=10,
                                      reanalyze=True
                                      )
        except:
            traceback.print_exc()


