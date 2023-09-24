
import numpy as n
import matplotlib.pyplot as plt
import pyfftw
import stuffr
import scipy.signal as s
from digital_rf import DigitalRFReader, DigitalMetadataReader, DigitalMetadataWriter
import os
import h5py
from scipy.ndimage import median_filter
from scipy import sparse
from mpi4py import MPI
import traceback
#from scipy.ndimage import median_filter

comm=MPI.COMM_WORLD
size=comm.Get_size()
rank=comm.Get_rank()

# we really need to be narrow band to avoid interference. there is plenty of it.
# but we can't go too narrow, because then we lose the ion-line.
# 1000 m/s is 3 kHz and the code itself is 66.6 kHz
# 66.6/2 + 3 = 70 kHz, and thus the maximum frequency offset will be 35 kHz.
# that is tight, but hopefully enough to filter out the interference.
pass_band=0.1e6

def ideal_lpf(z,sr=1e6,f0=1.2*pass_band,L=200):
    m=n.arange(-L,L)+1e-6
    om0=n.pi*f0/(0.5*sr)
    h=s.hann(len(m))*n.sin(om0*m)/(n.pi*m)
#    plt.plot(h)
 #   plt.show()
    Z=n.fft.fft(z)
    H=n.fft.fft(h,len(Z))
    z_filtered=n.roll(n.fft.ifft(Z*H),-L)
#    plt.plot(z_filtered.real)
 #   plt.plot(z_filtered.imag)
  #  plt.show()
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


def convolution_matrix(envelope, rmin=0, rmax=100):
    """
    we imply that the number of measurements is equal to the number of elements
    in code

    """
    L = len(envelope)
    ridx = n.arange(rmin, rmax,dtype=int)
    A = n.zeros([L, rmax - rmin], dtype=n.complex64)
    idxm = n.zeros([L, rmax - rmin], dtype=int)    
    for i in n.arange(L):
        A[i, :] = envelope[(i - ridx) % L]
        idxm[i,:] = n.array(n.mod( i-ridx, L ),dtype=int)
    result = {}
    result["A"] = A
    result["ridx"] = ridx
    result["idxm"] = idxm
    return(result)

        

tmm = {}
T_injection=1172.0 # May 24th 2022 value
tmm[300]={"noise0":7800,"noise1":8371,"tx0":76,"tx1":645,"gc":1000,"last_echo":7700,"e_gc":800}
for i in range(1,33):
    tmm[i]={"noise0":8400,"noise1":8850,"tx0":76,"tx1":624,"gc":1000,"last_echo":8200,"e_gc":800}

# n seconds to average
avg_dur=10.0

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

lags=n.arange(1,48,dtype=int)*10
lag_avg=3

n_lags=len(lags)-lag_avg
mean_lags=n.zeros(n_lags)
for i in range(n_lags):
    mean_lags[i]=n.mean(lags[i:(i+lag_avg)])

range_shift=600
# 30 microsecond range gates
fftlen=1024

rg=60
# A

n_rg=int(n.floor(7000/rg))
rgs=n.arange(n_rg)*rg
rgs_km=rgs*0.15
dop_hz=n.fft.fftshift(n.fft.fftfreq(fftlen,d=1/sr))
freq_idx = n.where( n.abs(dop_hz) < pass_band )[0]
n_freq=len(freq_idx)
fi0=n.min(freq_idx)
fi1=n.max(freq_idx)+1

rmax=n_rg#*rg#200

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

    acf=n.zeros([n_pulses,n_rg,n_freq],dtype=n.float32)
    
    ac_idx=0
    lp_idx=0    
    
    # USRP DC offset bug due to truncation instead of rounding.
    # Ryan Volz has a fix for firmware in USRPs. 
    z_dc=n.complex64(-0.212-0.221j)
    
    bg_samples=[]
    bg_plus_inj_samples=[]

    sidkeys=list(sid.keys())

    A=[]
    mgs=[]
    mes=[]
    sigmas=[]
    idxms=[]
    rmins=[]
    
    sample0=800
    sample1=7750
    rdec=rg
    m0=int(n.round(sample0/rdec))
    m1=int(n.round(sample1/rdec))

    n_meas=m1-m0
    
    for li in range(n_lags):
        # determine what is the lowest range that can be estimated
        # rg0=(gc - txstart - 0.6*pulse_length + lag)/range_decimation
        rmin=int(n.round((sample0-111-480*0.5+lags[li])/rdec))
#        rmin=int(n.round(lags[li]/rdec+sample0/rdec+480/rdec/2))# half a pulse length needed
        cm=convolution_matrix(n.zeros(m1),rmin=rmin,rmax=rmax)
        rmins.append(rmin)
        idxms.append(cm["idxm"])
        A.append([])
        mgs.append([])
        mes.append([])
        sigmas.append([])                
    import stuffr

    varsum=0.0

    # start at 3, because we may need to look back for GC
    for keyi in range(3,n_pulses-3):
        key=sidkeys[keyi]

        if sid[key] not in tmm.keys():
            print("unknown pulse code %d encountered, halting."%(sid[key]))
            exit(0)
        
        z_echo=None
        zd=None

        z_echo = d_il.read_vector_c81d(key, 10000, channel) - z_dc
        # no filtering of tx
        z_tx=n.copy(z_echo)
#        z_echo=z_echo
        
        if sid[key] == 300:
            next_key = sidkeys[keyi+3]
            z_echo1 = d_il.read_vector_c81d(next_key, 10000, channel) - z_dc
 #           z_echo1=z_echo1

#            zd=z_echo-z_echo1
            if debug_gc_rem:
                plt.plot(zd.real+2000)
                plt.plot(zd.imag+2000)
                plt.plot(z_echo.real)
                plt.plot(z_echo.imag)
                
                plt.title(sid[key])
                plt.show()
#            print("ignoring long pulse")
 #           continue
        elif sid[key] == sid[sidkeys[keyi+1]]:
            next_key = sidkeys[keyi+1]
#            z_echo = d_il.read_vector_c81d(key, 10000, channel) - z_dc
 #           z_echo=ideal_lpf(z_echo)
            z_echo1 = d_il.read_vector_c81d(next_key, 10000, channel) - z_dc
#            z_echo1=z_echo1

#            zd=z_echo-z_echo1
            if debug_gc_rem:
                plt.plot(zd.real+2000)
                plt.plot(zd.imag+2000)
                plt.plot(z_echo.real)
                plt.plot(z_echo.imag)            
                plt.title(sid[key])            
                plt.show()
        elif sid[key] == sid[sidkeys[keyi-1]]:
            next_key = sidkeys[keyi-1]
#            z_echo = d_il.read_vector_c81d(key, 10000, channel) - z_dc
 #           z_echo=ideal_lpf(z_echo)
            z_echo1 = d_il.read_vector_c81d(next_key, 10000, channel) - z_dc
#            z_echo1=z_echo1

            
            if debug_gc_rem:
                plt.plot(zd.real+2000)
                plt.plot(zd.imag+2000)
                plt.plot(z_echo.real)
                plt.plot(z_echo.imag)            
                plt.title(sid[key])            
                plt.show()
                    
        print("%d/%d"%(keyi,n_pulses))

        noise0=tmm[sid[key]]["noise0"]
        noise1=tmm[sid[key]]["noise1"]
        last_echo=tmm[sid[key]]["last_echo"]        
        tx0=tmm[sid[key]]["tx0"]
        tx1=tmm[sid[key]]["tx1"]
        gc=tmm[sid[key]]["gc"]
        e_gc=tmm[sid[key]]["e_gc"]        



        # filter noise injection.
        z_noise=n.copy(z_echo)
        z_noise=ideal_lpf(z_noise)

        
        bg_samples.append( n.mean(n.abs(z_noise[(last_echo-500):last_echo])**2.0) )
        bg_plus_inj_samples.append( n.mean(n.abs(z_noise[(noise0):noise1])**2.0) )


        z_tx[0:tx0]=0.0
        z_tx[tx1:10000]=0.0

        # normalize tx pwr
        z_tx=z_tx/n.sqrt(n.sum(n.abs(z_tx)**2.0))
        z_echo[last_echo:10000]=0.0
        z_echo1[last_echo:10000]=0.0
        
        z_echo[0:gc]=0.0#0.0
        z_echo1[0:gc]=0.0
        

        #zd=z_echo-z_echo1
        #zd[last_echo:10000]=0.0

        z_echo=ideal_lpf(z_echo)
        z_echo1=ideal_lpf(z_echo1)
        
        zd=z_echo-z_echo1

        zd[0:gc]=n.nan
        z_echo[0:gc]=n.nan
        z_echo[last_echo:10000]=n.nan
        zd[last_echo:10000]=n.nan        
        
        #A=n.zeros([n_meas,n_rg],dtype=n.complex64)


    #    col_merge=[]
   #     for ci in range(100):
  #          col_merge.append([ci])
 #       for ci in range(50):
#            col_merge.append([ci+100,ci+100+1])
        
        for li in range(n_lags):
            for ai in range(lag_avg):
                amb=stuffr.decimate(z_tx[0:(len(z_tx)-lags[li+ai])]*n.conj(z_tx[lags[li+ai]:len(z_tx)]),dec=rdec)

                # gc removal by the T. Turunen subtraction of two pulses with the same code, transmitted in
                # close proximity to one another.
                measg=stuffr.decimate(zd[0:(len(z_echo)-lags[li+ai])]*n.conj(zd[lags[li+ai]:len(z_echo)]),dec=rdec)

                # no gc removal
                mease=stuffr.decimate(z_echo[0:(len(z_echo)-lags[li+ai])]*n.conj(z_echo[lags[li+ai]:len(z_echo)]),dec=rdec)

    #            plt.plot(mease.real)
     #           plt.plot(mease.imag)
      #          plt.show()

                #cm=convolution_matrix(amb,rmin=li+15,rmax=rmax)

                TM=sparse.csc_matrix(amb[idxms[li]])
    #            TM=amb[idxms[li]]
    #            print(TM.shape)

                # each ipp has its own standard deviation estimate
                sigma_est = n.nanstd(mease)
                varsum+=sigma_est**2.0

                sigmas[li].append(sigma_est)
                mgs[li].append(measg[m0:m1])
                mes[li].append(mease[m0:m1])
                A[li].append(TM[m0:m1,:])

    acfs_g=n.zeros([rmax,n_lags],dtype=n.complex64)
    acfs_e=n.zeros([rmax,n_lags],dtype=n.complex64)    
    acfs_g[:,:]=n.nan
    acfs_e[:,:]=n.nan    
    
    acfs_var=n.zeros([rmax,n_lags],dtype=n.float32)
    acfs_var[:,:]=n.nan

    noise=n.median(bg_samples)    
    alpha=(n.median(bg_plus_inj_samples)-n.median(bg_samples))/T_injection 
    T_sys=noise/alpha

    import time
    for li in range(n_lags):
        print(li)
        
        AA=sparse.vstack(A[li])
        #print(AA.shape)
        mm_g=n.concatenate(mgs[li])
        mm_e=n.concatenate(mes[li])
        sigma_lp_est=n.zeros(len(mm_g))
        sigma_lp_est[:]=1.0

        # remove outliers and estimate standard deviation 
        if True:
            print("ratio test")
            mm_gm=n.copy(mm_g)
            mm_em=n.copy(mm_e)
            n_ipp=int(len(mm_gm)/n_meas)
            mm_gm.shape=(n_ipp,n_meas)
            mm_em.shape=(n_ipp,n_meas)

            sigma_lp_est=n.sqrt(n.percentile(n.abs(mm_em[:,:])**2.0,34,axis=0)*2.0)
            sigma_lp_est_g=n.sqrt(n.percentile(n.abs(mm_gm[:,:])**2.0,34,axis=0)*2.0)            

            ratio_test=n.abs(mm_em)/sigma_lp_est
            ratio_test_g=n.abs(mm_gm)/sigma_lp_est_g

            localized_sigma=n.abs(n.copy(mm_em))**2.0
            wf=n.repeat(1/10,10)
            WF=n.fft.fft(wf,localized_sigma.shape[0])
            for ri in range(mm_em.shape[1]):
                # we need to wrap around, to avoid too low values.
                localized_sigma[:,ri]=n.roll(n.sqrt(n.fft.ifft(WF*n.fft.fft(localized_sigma[:,ri])).real),-5)
#                localized_sigma[:,ri]=n.sqrt(n.convolve(localized_sigma[:,ri],n.repeat(1.0/50,50),mode="same"))

            # make sure we don't have a division by zero
            msig=n.nanmedian(localized_sigma)
            if msig<0:
                msig=1.0
            localized_sigma[localized_sigma<msig]=msig


            if False:
                plt.pcolormesh(localized_sigma.T)
                plt.colorbar()
                plt.show()
                
                plt.pcolormesh(ratio_test.T)
                plt.colorbar()
                plt.show()
                plt.pcolormesh(ratio_test_g.T)
                plt.colorbar()
                plt.show()
            
            
            #            plt.pcolormesh(localized_sigma)
            #           plt.colorbar()
            #          plt.show()
            
            debug_outlier_test=False
            if debug_outlier_test:
                plt.pcolormesh(mm_em.real.T)
                plt.colorbar()
                plt.show()
                

            # is this threshold too high?
            # maybe 10
            mm_em[ratio_test > 10]=n.nan
            mm_gm[ratio_test_g > 10]=n.nan

            # these will be shit no matter what
            mm_em[localized_sigma > 100*msig]=n.nan
            mm_gm[localized_sigma > 100*msig]=n.nan            
            
            if debug_outlier_test:            
                plt.pcolormesh(mm_em.real.T)
                plt.colorbar()
                plt.show()

                plt.pcolormesh(localized_sigma.T)
                plt.colorbar()
                plt.show()
                
            #sigma_lp_est=n.tile(sigma_lp_est,n_ipp)
            #print(sigma_lp_est.shape)
            sigma_lp_est=localized_sigma
            sigma_lp_est.shape=(len(mm_g),)

            mm_gm.shape=(len(mm_g),)
            mm_em.shape=(len(mm_e),)
            mm_g=mm_gm
            mm_e=mm_em
            
        mm_g=mm_g/sigma_lp_est
        mm_e=mm_e/sigma_lp_est

        gidx = n.where( (n.isnan(mm_e)==False) & (n.isnan(mm_g)==False) & (n.isnan(sigma_lp_est) == False) )[0]# & (n.isinf(mm_e)==False) & (n.isneginf(mm_e)==False) & (n.isnan(mm_g)==False) & (n.isinf(mm_g)==False) & (n.isneginf(mm_g)==False)  )[0]
        print("%d/%d measurements good"%(len(gidx),len(mm_g)))

        srow=n.arange(len(gidx),dtype=int)
        scol=n.arange(len(gidx),dtype=int)
        sdata=1/sigma_lp_est[gidx]#.flatten#sigma_lp_est
        Sinv = sparse.csc_matrix( (sdata, (srow,scol)) ,shape=(len(gidx),len(gidx)))
        #print("scaling matrix test")
        #AA = AA.dot(S_mat)
        
#        for ri in range(AA.shape[0]):
 #           AA[ri,:]=AA[ri,:]/sigma_lp_est[ri]

        # take out bad measurements
        AA=AA[gidx,:]
        mm_g=mm_g[gidx]
        mm_e=mm_e[gidx]        

        # do some distribution statistics here to weed out bad measurements

        try:
            t0=time.time()
            #ATA=n.zeros([AA.shape[1],AA.shape[1]],dtype=n.complex64)
            #for ri in range(AA.shape[1]):
            #    for ci in range(AA.shape[1]):
            #        ATA[ci,ri]=n.dot(n.conj(AA[ri,:]),AA[ci,:])
            AT=n.conj(AA.T).dot(Sinv)
            ATA=AT.dot(n.dot(Sinv,AA)).toarray()

            # we get to calculate gc and no gc practically free
            # with the same precalculated matrix
            ATm_g=AT.dot(mm_g)
            ATm_e=AT.dot(mm_e)

            print(ATA.shape)
            Sigma=n.linalg.inv(ATA)
            xhat_e=n.dot(Sigma,ATm_e)
            xhat_g=n.dot(Sigma,ATm_g)        
            t1=time.time()
            t_simple=t1-t0        
            print("simple %1.2f"%(t_simple))
            acfs_e[ rmins[li]:rmax, li ]=xhat_e
            acfs_g[ rmins[li]:rmax, li ]=xhat_g

            acfs_var[ rmins[li]:rmax, li ] = n.diag(Sigma.real)
        except:
            traceback.print_exc()
            print("something went wrong.")
            

    if False:
        plt.pcolormesh(mean_lags,rgs_km[0:rmax],acfs_g.real,vmin=-200,vmax=400)
        plt.xlabel("Lag ($\mu$s)")
        plt.ylabel("Range (km)")
        plt.colorbar()
        plt.title("%s T_sys=%1.0f K"%(stuffr.unix2datestr(i0/sr),T_sys))
        plt.tight_layout()
        plt.show()
    plt.pcolormesh(mean_lags,rgs_km[0:rmax],acfs_e.real,vmin=-2e3,vmax=5e3)
    plt.xlabel("Lag ($\mu$s)")
    plt.ylabel("Range (km)")
    plt.colorbar()
    plt.title("%s T_sys=%1.0f K"%(stuffr.unix2datestr(i0/sr),T_sys))
    plt.tight_layout()
#    plt.show()
 #   exit(0)
    plt.savefig("lpi2/lpi-%d.png"%(i0/sr))
    plt.close()
    plt.clf()
        
    ho=h5py.File("lpi2/lpi-%d.h5"%(i0/sr),"w")
    ho["acfs_g"]=acfs_g
    ho["acfs_e"]=acfs_e    
    ho["acfs_var"]=acfs_var
    ho["rgs_km"]=rgs_km[0:rmax]
    ho["lags"]=mean_lags/sr
    ho["i0"]=i0/sr
    ho["T_sys"]=T_sys
    ho["var_est"]=varsum
    ho.close()
#    plt.show()

            
            
#            plt.plot(amb.real)
 #           plt.plot(amb.imag)
  #          plt.title(lags[li])
   #         plt.show()
