
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
avg_dur=5.0

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

lags=n.arange(1,16,dtype=int)*30

range_shift=600
# 30 microsecond range gates
fftlen=1024
rg=30
n_rg=int(n.floor(6000/rg))
rgs=n.arange(n_rg)*rg
rgs_km=rgs*0.15
dop_hz=n.fft.fftshift(n.fft.fftfreq(fftlen,d=1/sr))
freq_idx = n.where( n.abs(dop_hz) < pass_band )[0]
n_freq=len(freq_idx)
fi0=n.min(freq_idx)
fi1=n.max(freq_idx)+1

rmax=200

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
    ms=[]
    idxms=[]
    for li in range(len(lags)):
        cm=convolution_matrix(n.zeros(265),rmin=li+15,rmax=rmax)
        idxms.append(cm["idxm"])
        A.append([])
        ms.append([])
    import stuffr
    

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
        z_echo=ideal_lpf(z_echo)            
        
        if sid[key] == 300:
            next_key = sidkeys[keyi+3]
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
#            z_echo = d_il.read_vector_c81d(key, 10000, channel) - z_dc
 #           z_echo=ideal_lpf(z_echo)
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
#            z_echo = d_il.read_vector_c81d(key, 10000, channel) - z_dc
 #           z_echo=ideal_lpf(z_echo)
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

        noise0=tmm[sid[key]]["noise0"]
        noise1=tmm[sid[key]]["noise1"]
        last_echo=tmm[sid[key]]["last_echo"]        
        tx0=tmm[sid[key]]["tx0"]
        tx1=tmm[sid[key]]["tx1"]
        gc=tmm[sid[key]]["gc"]
        e_gc=tmm[sid[key]]["e_gc"]        

        bg_samples.append( n.mean(n.abs(z_echo[(last_echo-500):last_echo])**2.0) )
        bg_plus_inj_samples.append( n.mean(n.abs(z_echo[(noise0):noise1])**2.0) )


        z_tx[0:tx0]=0.0
        z_tx[tx1:10000]=0.0

        # normalize tx pwr
        z_tx=z_tx/n.sqrt(n.sum(n.abs(z_tx)**2.0))
        z_echo[0:gc]=0.0
        # e-region
        zd[0:e_gc]=0.0
        z_echo[last_echo:10000]=0.0
        zd[last_echo:10000]=0.0

        #A=n.zeros([n_meas,n_rg],dtype=n.complex64)
        m0=26
        m1=265
        for li in range(len(lags)):
            amb=stuffr.decimate(z_tx[0:(len(z_tx)-lags[li])]*n.conj(z_tx[lags[li]:len(z_tx)]),dec=30)
            meas=stuffr.decimate(zd[0:(len(z_echo)-lags[li])]*n.conj(zd[lags[li]:len(z_echo)]),dec=30)
            #cm=convolution_matrix(amb,rmin=li+15,rmax=rmax)
            TM=amb[idxms[li]]

            # each ipp has its own standard deviation estimate
            sigma_est = n.std(meas)
            
            ms[li].append(meas[m0:m1]/sigma_est)
            A[li].append(TM[m0:m1,:]/sigma_est)

    acfs=n.zeros([rmax,len(lags)],dtype=n.complex64)
    acfs[:,:]=n.nan

    noise=n.median(bg_samples)    
    alpha=(n.median(bg_plus_inj_samples)-n.median(bg_samples))/T_injection 
    T_sys=noise/alpha
    
    for li in range(len(lags)):
        print(li)
        AA=n.row_stack(A[li])
        mm=n.concatenate(ms[li])
        xhat=n.linalg.lstsq(AA,mm)[0]
        acfs[ (li+15):rmax,li ]=xhat

    ho=h5py.File("lpi/lpi-%d.h5"%(i0/sr),"w")
    ho["acfs"]=acfs
    ho["rgs_km"]=rgs_km[0:rmax]
    ho["lags"]=lags/sr
    ho["i0"]=i0/sr
    ho["T_sys"]=T_sys
    ho.close()
    plt.pcolormesh(lags,rgs_km[0:rmax],acfs.real,vmin=-3e3,vmax=3e3)
    plt.xlabel("Lag ($\mu$s)")
    plt.ylabel("Range (km)")
    plt.colorbar()
    plt.title("%s T_sys=%1.0f K"%(stuffr.unix2datestr(i0/sr),T_sys))
    plt.tight_layout()
    plt.savefig("lpi/lpi-%d.png"%(i0/sr))
    plt.close()
    plt.clf()
#    plt.show()

            
            
#            plt.plot(amb.real)
 #           plt.plot(amb.imag)
  #          plt.title(lags[li])
   #         plt.show()
