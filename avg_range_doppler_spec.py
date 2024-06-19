import scipy.constants as c
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
    h=s.windows.hann(len(m))*n.sin(om0*m)/(n.pi*m)
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
    wf=s.windows.hann(txlen)
    for ri in range(n_rg):
        RDS[ri,:]=n.abs(n.fft.fftshift(n.fft.fft(wf*z_tx[0:tx1]*z_echo[ (rgs[ri]):(rgs[ri]+txlen) ],fftlen)))**2.0
    return(RDS)


# sample rate for metadata
idsr=1000000
sr=1000000


tmm = {}
T_injection=1172.0 # May 24th 2022 value
tmm[300]={"noise0":7800,"noise1":8371,"tx0":76,"tx1":645,"gc":1000,"last_echo":7700,"e_gc":800,
          "read_length":10000
          }

# this is the horizon scanning mode
tmm[800]={"noise0":30176,"noise1":32033,"tx0":69,"tx1":2171,"gc":3721,"last_echo":30000,"e_gc":3721,
          "read_length":40000
          }

# create data structures for all supported modes
lp_data={
    300:{
        "LP":None,
        "range_shift":600,
        "fft_length":4096,
        "rg":30,
        "noise_r0_km":700,
        "noise_r1_km":1200
        
         },
    800:{
        "LP":None,
        "range_shift":2171, # this is how much we range shift transmit pulse when calculating the range-Doppler ambiguity function
        "fft_length":4096,
        "rg":120,
        "noise_r0_km":3600,
        "noise_r1_km":4500
         }
}

# how many range gates
lp_data[300]["n_rg"]=int(n.floor(8000/lp_data[300]["rg"]))
lp_data[800]["n_rg"]=int(n.floor(28000/lp_data[800]["rg"]))

# the range gates
lp_data[300]["rgs"]=n.arange(lp_data[300]["n_rg"])*lp_data[300]["rg"]
lp_data[800]["rgs"]=n.arange(lp_data[800]["n_rg"])*lp_data[800]["rg"]

# range gates in km
lp_data[300]["rgs_km"]=lp_data[300]["rgs"]*1e-6*c.c/2.0/1e3
lp_data[800]["rgs_km"]=lp_data[800]["rgs"]*1e-6*c.c/2.0/1e3

# doppler shifts
lp_data[300]["dop_hz"]=n.fft.fftshift(n.fft.fftfreq(lp_data[300]["fft_length"],d=1/sr))
lp_data[800]["dop_hz"]=n.fft.fftshift(n.fft.fftfreq(lp_data[800]["fft_length"],d=1/sr))

# which frequency bins do we store for the ion-line
lp_data[300]["freq_idx"]=n.where( n.abs(lp_data[300]["dop_hz"]) < pass_band )[0]
lp_data[800]["freq_idx"]=n.where( n.abs(lp_data[800]["dop_hz"]) < pass_band )[0]

# how many frequency bins are stored for the ion-line spectrum
lp_data[300]["n_freq"]=len(lp_data[300]["freq_idx"])
lp_data[800]["n_freq"]=len(lp_data[800]["freq_idx"])

# these are the min and max frequency indices for the ion-line spectrum bins
lp_data[300]["fi0"]=n.min(lp_data[300]["freq_idx"])
lp_data[300]["fi1"]=n.max(lp_data[300]["freq_idx"])+1
lp_data[800]["fi0"]=n.min(lp_data[800]["freq_idx"])
lp_data[800]["fi1"]=n.max(lp_data[800]["freq_idx"])+1


def avg_range_doppler_spectra(dirname="/media/j/fee7388b-a51d-4e10-86e3-5cabb0e1bc13/isr/2023-09-05/usrp-rx0-r_20230905T214448_20230906T040054/",
                              save_png=True,
                              avg_dur=10,
                              step=10,
                              min_tx_pulses=100,
                              reanalyze=False,
                              channel="zenith-l",
                              avg_type="outlier_removal",
                              postfix="_outlier",
                              mode=300
                              ):


    id_read = DigitalMetadataReader("%s/metadata/id_metadata"%(dirname))
    d_il = DigitalRFReader("%s/rf_data/"%(dirname))

    zpm,mpm=mrs.get_tx_power_model("%s/metadata/powermeter"%(dirname))
    tx_ant,rx_ant=mrs.get_antenna_select("%s/metadata/antenna_control_metadata"%(dirname))    
    
    os.system("mkdir -p %s/range_doppler_%d%s/%s"%(dirname,mode,postfix,channel))
    idb=id_read.get_bounds()
    # min transmit power required to produce an estimate of range-Doppler spectra. lower powers ignored.
    min_tx_pwr=400e3

    plot_voltage=False
    use_ideal_filter=True
    debug_gc_rem=False

    # how many integration periods do we have
    n_times = int(n.floor(((idb[1]-idb[0])-avg_dur)/idsr/step))

    i0=idb[0]

    n_rg=lp_data[mode]["n_rg"]
    n_freq=lp_data[mode]["n_freq"]
    
    rgs=lp_data[mode]["rgs"]
    rgs_km=lp_data[mode]["rgs_km"]
    dop_hz=lp_data[mode]["dop_hz"]        
    freq_idx=lp_data[mode]["freq_idx"]
    fi0=lp_data[mode]["fi0"]
    fi1=lp_data[mode]["fi1"]

    range_shift=lp_data[mode]["range_shift"]    

    fft_length=lp_data[mode]["fft_length"]

    noise0=tmm[mode]["noise0"]
    noise1=tmm[mode]["noise1"]
    last_echo=tmm[mode]["last_echo"]        
    tx0=tmm[mode]["tx0"]
    tx1=tmm[mode]["tx1"]
    noise_r0_km=lp_data[mode]["noise_r0_km"]
    noise_r1_km=lp_data[mode]["noise_r1_km"]
    gc=tmm[mode]["gc"]

    # go through one integration window
    for ai in range(rank,n_times,size):
        i0 = ai*int(step*idsr) + idb[0]
        print(stuffr.unix2datestr(i0/1e6))
        
        if os.path.exists("%s/range_doppler_%d%s/%s/il_%d.png"%(dirname,mode,postfix,channel,int(i0/1e6))) and reanalyze==False:
            print("already analyzed %d"%(i0/1e6))
            continue


        try:


            # get info on all the pulses transmitted during this averaging interval
            # get some extra for gc
            sid = id_read.read(i0,i0+int(avg_dur*idsr)+40000,"sweepid")

            n_pulses=len(sid.keys())

            RDS_LP=n.zeros([n_pulses, n_rg, n_freq],dtype=n.float32)
            RDS_LP_var=n.zeros([n_rg, n_freq],dtype=n.float32)
            W_LP=n.zeros(n_pulses,dtype=n.float32)
            # range-doppler ambiguity function
            TX_RDS_LP=n.zeros([n_rg,n_freq],dtype=n.float32)
            # sum of weights
            WS_LP=0.0
            # running index for pulses added to array
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

            for keyi in range(0,n_pulses):
                key=sidkeys[keyi]

                if sid[key] != mode:
#                    print("wrong mode %d != %d. ignoring."%(sid[key],mode))
                    continue

                if sid[key] not in tmm.keys():
                    #print("unknown pulse code %d encountered, skipping."%(sid[key]))
                    continue

#                print("pulse id %d"%(sid[key]))

                # how many samples do we read
                read_length=tmm[sid[key]]["read_length"]

                z_echo=None
                z_tx=None

                zenith_pwr=zpm(key/1e6)
                misa_pwr=mpm(key/1e6)            
                if sid[key] == 300:

                    if (tx_ant(key) < -0.99) and (rx_ant(key) < -0.99) and (channel == "zenith-l") and (zenith_pwr > min_tx_pwr):
                        try:
                            z_echo = d_il.read_vector_c81d(key, read_length, "zenith-l") - z_dc
                            z_tx = d_il.read_vector_c81d(key, read_length, "tx-h")# - z_dc                        
                            avg_tx_pwr+=zenith_pwr
                            avg_tx_pwr_samples+=1
                        except:
                            traceback.print_exc()                        
                            print("couldn't read %d %s"%(key,"zenith-l"))
                            continue
                    elif (tx_ant(key) > 0.99) and (rx_ant(key) > 0.99) and (channel == "misa-l") and (misa_pwr > min_tx_pwr):
                        try:
                            z_echo = d_il.read_vector_c81d(key, read_length, "misa-l") - z_dc
                            z_tx = d_il.read_vector_c81d(key, read_length, "tx-h")# - z_dc

                            avg_tx_pwr+=misa_pwr
                            avg_tx_pwr_samples+=1
                        except:
                            traceback.print_exc()
                            print("couldn't read %d %s %s"%(key,"misa-l",stuffr.unix2datestr(key/1e6)))
                            continue
                    else:
#                        print("not enough tx power (%1.0f,%1.0f MW) on %s skipping this pulse"%(zpm(key/1e6)/1e6,mpm(key/1e6)/1e6,channel))
                        continue

                elif sid[key]==800:
                    # if there is enough power on misa, the tx and rx are switched to misa, and we are
                    # analyzing misa
                    if (tx_ant(key) > 0.99) and (rx_ant(key) > 0.99) and (channel == "misa-l") and (misa_pwr > min_tx_pwr):
                        z_echo = d_il.read_vector_c81d(key, read_length, "misa-l") - z_dc
                        z_tx = d_il.read_vector_c81d(key, read_length, "tx-h")# - z_dc

                        avg_tx_pwr+=misa_pwr
                        avg_tx_pwr_samples+=1

                    else:
#                        print("not enough power")
                        continue

                else:
                    continue

                # we only procede here if we have enough power and we have a known mode

                T_sys,T_sys2=estimate_tsys(tmm,sid,key,d_il,z_echo)
                print("%d found %d pulses in %s T_sys %1.0f K"%(rank,lp_idx,channel,T_sys))

                z_tx[0:tx0]=0.0
                z_tx[tx1:read_length]=0.0

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
                z_echo[last_echo:read_length]=0.0

                # use this to estimate the range-Doppler ambiguity function
                z_tx2=n.copy(z_tx)
                z_tx2=n.roll(z_tx2,range_shift)

                RDS=range_dop_spec(z_echo,z_tx,rgs,tx0,tx1,fft_length)
                TX_RDS=range_dop_spec(z_tx2,z_tx,rgs,tx0,tx1,fft_length)

                RDS_LP[lp_idx,:,:]=RDS[:,fi0:fi1]
                TX_RDS_LP[:,:]+=TX_RDS[:,fi0:fi1]
                lp_idx+=1

            # go through all supported modes
            if lp_idx < min_tx_pulses:
#                print("less than %d pulses found. skipping this integration period"%(min_tx_pulses))
                continue

            noise=n.median(bg_samples)
            alpha=(n.median(bg_plus_inj_samples)-n.median(bg_samples))/T_injection 
            T_sys=noise/alpha

            # this is the most aggressive method I know of doing an average, but it might 
            # be a bit too aggressive. I don't even know if this is an unbiased estimator
            use_median_mean=False
            if avg_type=="median":
                RDS_LP_var=n.var(RDS_LP[0:lp_idx,:,:],axis=0)                        
                RDS_LP=n.median(RDS_LP[0:lp_idx,:,:],axis=0)
            if avg_type=="mean":
                RDS_LP_var=n.var(RDS_LP[0:lp_idx,:,:],axis=0)            
                RDS_LP=n.mean(RDS_LP[0:lp_idx,:,:],axis=0)

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
                            if (i+j) < RDS_LP.shape[0]:
                                RDS_LP[i+j,(n.max([0,bi-16])):(n.min([n_rg-1,bi+16])),:]=n.nan

                RDS_LP=n.nanmean(RDS_LP[0:lp_idx,:,:],axis=0)


            # scale to kelvins
            # note that fit_lp.py undoes the 1/alpha scaling, calculates a median alpha
            # and reapplies it to ensure outliers are removed from the receiver gain
            RDS_LP=RDS_LP/alpha
            # we need to scale, as counts can vary a lot, the receiver gain also changes quite a lot
            RDS_LP_var=RDS_LP_var/alpha/lp_idx

            # get noise floor for far enough ranges outside the ionosphere, and within pass band
            # noise floor only in filter pass band
            range_idx = n.where( (rgs_km > noise_r0_km) & (rgs_km < noise_r1_km) )[0]    
            noise_lp=n.median(RDS_LP[n.min(range_idx):n.max(range_idx),:])

            plot_amb=False
            plt.figure(figsize=(12,6))

    #        gc_rg0=n.where(rgs_km > 200)[0][0]
            snr_lp=(RDS_LP-noise_lp)/noise_lp

            if lp_idx > min_tx_pulses:

                plt.pcolormesh(dop_hz[fi0:fi1]/1e3,rgs_km,snr_lp,vmin=-0.5,vmax=2,cmap="plasma")
                cb=plt.colorbar()
                cb.set_label("LP SNR")
                plt.xlim([-50,50])        
                plt.title("Millstone Hill T$_{\\mathrm{sys}}$=%1.0f K %s"%(T_sys,stuffr.unix2datestr(i0/1e6)))
                plt.ylabel("Range (km)")
                plt.xlabel("Doppler (kHz)")
                plt.tight_layout()
                plt.savefig("%s/range_doppler_%d%s/%s/il_%d.png"%(dirname,mode,postfix,channel,int(i0/1e6)))
                plt.close()
                plt.clf()

                ho=h5py.File("%s/range_doppler_%d%s/%s/il_%d.h5"%(dirname,mode,postfix,channel,int(i0/1e6)),"w")
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
                ho["rg"]=rgs
                ho["T_sys"]=T_sys
                ho["alpha"]=alpha
                ho["channel"]=channel
                ho["P_tx"]=avg_tx_pwr/avg_tx_pwr_samples
                ho["mode"]=mode
                ho.close()
        except:
            traceback.print_exc()
            print("problem with integration period")

if __name__ == "__main__":

#    dirs=["/media/j/fee7388b-a51d-4e10-86e3-5cabb0e1bc13/isr/2021-12-03a/usrp-rx0-r_20211203T224500_20211204T160000/",
#          "/media/j/fee7388b-a51d-4e10-86e3-5cabb0e1bc13/isr/2021-12-01/usrp-rx0-r_20211201T230000_20211202T160100/",
#          "/media/j/fee7388b-a51d-4e10-86e3-5cabb0e1bc13/isr/2023-09-05/usrp-rx0-r_20230905T214448_20230906T040054",
#          "/media/j/fee7388b-a51d-4e10-86e3-5cabb0e1bc13/isr/2023-09-24/usrp-rx0-r_20230924T200050_20230925T041059/",
#          "/media/j/fee7388b-a51d-4e10-86e3-5cabb0e1bc13/isr/2023-09-28/usrp-rx0-r_20230928T211929_20230929T040533/",
#          "/media/j/fee7388b-a51d-4e10-86e3-5cabb0e1bc13/isr/2021-12-03/usrp-rx0-r_20211203T000000_20211203T033600/",
#          "/media/j/fee7388b-a51d-4e10-86e3-5cabb0e1bc13/isr/2021-12-05/usrp-rx0-r_20211205T000000_20211205T160100/",
#          "/media/j/fee7388b-a51d-4e10-86e3-5cabb0e1bc13/isr/2021-12-06/usrp-rx0-r_20211206T000000_20211206T132500/",
#          "/media/j/fee7388b-a51d-4e10-86e3-5cabb0e1bc13/isr/2021-12-21/usrp-rx0-r_20211221T125500_20211221T220000/"]


    dirs=["/media/j/4df2b77b-d2db-4dfa-8b39-7a6bece677ca/eclipse2024/usrp-rx0-r_20240407T100000_20240409T110000/"]
    
    for d in dirs:
        try:
            avg_range_doppler_spectra(dirname=d,
                                      channel="misa-l",
                                      save_png=True,
                                      avg_dur=10,
                                      reanalyze=False,
                                      avg_type="outlier_mean",
                                      postfix="_outlier",
                                      mode=300
                                      )
        except:
            traceback.print_exc()
        try:
            avg_range_doppler_spectra(dirname=d,
                                      channel="zenith-l",
                                      save_png=True,
                                      avg_dur=10,
                                      reanalyze=False,
                                      avg_type="outlier_mean",
                                      postfix="_outlier",
                                      mode=300
                                      )
        except:
            traceback.print_exc()
        if False:
            try:
                avg_range_doppler_spectra(dirname=d,
                                          channel="misa-l",
                                          save_png=True,
                                          avg_dur=10,
                                          reanalyze=False,
                                          avg_type="outlier_mean",
                                          postfix="_outlier",
                                          mode=800
                                          )
            except:
                traceback.print_exc()
        
#        try:
 #           avg_range_doppler_spectra(dirname=d,
  #                                    channel="zenith-l",
   #                                   save_png=True,
     #                                 avg_dur=10,
    #                                  reanalyze=True
      #                                )
       # except:
        #    traceback.print_exc()


