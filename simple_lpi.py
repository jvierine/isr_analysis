import os
# mpi does paralelization, both multithread matrix operations
# just one per process to avoid cache trashing. otherwise each mpi process will
# try to use all cpus for the linear algebra, which slows things to a halt.
os.system("export OMP_NUM_THREADS=1")
os.environ["OMP_NUM_THREADS"] = "1"

import numpy as n
import matplotlib.pyplot as plt
import pyfftw
import stuffr
import scipy.signal as s
from digital_rf import DigitalRFReader, DigitalMetadataReader, DigitalMetadataWriter

import h5py
from scipy.ndimage import median_filter
from scipy import sparse
from mpi4py import MPI
import scipy.signal as ss
import scipy.constants as c
import traceback
import time

comm=MPI.COMM_WORLD
size=comm.Get_size()
rank=comm.Get_rank()

fft=pyfftw.interfaces.numpy_fft.fft
ifft=pyfftw.interfaces.numpy_fft.ifft

# we really need to be narrow band to avoid interference. there is plenty of it.
# but we can't go too narrow, because then we lose the ion-line.
# 1000 m/s is 3 kHz and the code itself is 66.6 kHz
# 66.6/2 + 3 = 70 kHz, and thus the maximum frequency offset will be 35 kHz.
# that is tight, but hopefully enough to filter out the interference.
#pass_band=0.1e6

def ideal_lpf(z,sr=1e6,f0=1.2*0.1e6,L=200):
    m=n.arange(-L,L)+1e-6
    om0=n.pi*f0/(0.5*sr)
    h=s.hann(len(m))*n.sin(om0*m)/(n.pi*m)

    Z=n.fft.fft(z)
    H=n.fft.fft(h,len(Z))
    z_filtered=n.roll(n.fft.ifft(Z*H),-L)

    return(z_filtered)

class simple_decimator:
    def __init__(self,L=10000,dec=10):
        self.L=L
        self.dec=dec
        decL=int(n.floor(L/dec))
        self.idxm=n.zeros([decL,dec],dtype=int)
        for ti in range(decL):
            self.idxm[ti,:]=n.arange(dec,dtype=int) + ti*dec
            
    def decimate(self,z):
        """ 
        this decimate has to be a sum to ensure all range resolutions have the same 
        magic constant.
        """
        decL=int(n.floor(len(z)/self.dec))
        return(n.sum(z[self.idxm[0:decL,:]],axis=1))

class fft_lpf:
    def __init__(self,z_len=10000,sr=1e6,f0=1.2*0.1e6,L=20):
        m=n.arange(-L,L)+1e-6
        om0=n.pi*f0/(0.5*sr)
        h=s.hann(len(m))*n.sin(om0*m)/(n.pi*m)
        # normalize to impulse response to unity.
        #h=n.array(h/n.sum(n.abs(h)**2.0),dtype=n.complex64)
        # unity gain at DC
        h=n.array(h/n.sum(h),dtype=n.complex64)
        self.h=h
        #pyfftw.interfaces.numpy_fft.fft()
        self.H=fft(h,z_len)
        self.L=L
        
    def lpf(self,z):
        return(n.roll(ifft(self.H*fft(z)),-self.L))
        
        


def ideal_lpf_h(sr=1e6,f0=1.2*0.1e6,L=200):
    m=n.arange(-L,L)+1e-6
    om0=n.pi*f0/(0.5*sr)
    h=s.hann(len(m))*n.sin(om0*m)/(n.pi*m)
    return(h)

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
    

def convolution_matrix(envelope, rmin=0, rmax=100):
    """
    we imply that the number of measurements is equal to the number of elements
    in code

    Use the index matrix (idxm) to efficiently grab the numbers from a 1d array to build a matrix
    A = z_tx_envelope[idxm]
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

#
# tbd: add range gates of different sizes
#
def lpi_files(dirname="/media/j/fee7388b-a51d-4e10-86e3-5cabb0e1bc13/isr/2023-09-05/usrp-rx0-r_20230905T214448_20230906T040054",
              avg_dur=10,  # n seconds to average
              channel="zenith-l",
              rg=60,       # how many microseconds is one range gate
              output_prefix="lpi_f",
              min_tx_frac=0.5,  # how much of the pulse can be missing due to ground clutter clipping, defines the minimum range gate
              reanalyze=False,
              pass_band=0.1e6,
              filter_len=20,
              use_long_pulse=True,
              maximum_range_delay=7000,    # microseconds. defines the highest range to analyze
              save_acf_images=True,
              fft_len=1024,                 # store diagnostic spectrum for RFI identification
              lags=n.arange(1,46,dtype=int)*10,
              lag_avg=1
              ):

    os.system("mkdir -p %s"%(output_prefix))
    
    id_read = DigitalMetadataReader("%s/metadata/id_metadata"%(dirname))
    d_il = DigitalRFReader("%s/rf_data/"%(dirname))

    idb=id_read.get_bounds()
    # sample rate for metadata
    idsr=1000000
    # sample rate for ion line channel
    sr=1000000

    plot_voltage=False
    use_ideal_filter=True
    debug_gc_rem=False

    # how many integration cycles do we have
    n_times = int(n.floor((idb[1]-idb[0])/idsr/avg_dur))

    # which lags to calculate
    
    # how many lags do we average together?
    

    # calculate the average lag value
    n_lags=len(lags)-lag_avg+1
    mean_lags=n.zeros(n_lags)
    for i in range(n_lags):
        mean_lags[i]=n.mean(lags[i:(i+lag_avg)])

    # maximum number of microseconds of delay, which we analyze
    # this is experiment specific. need to read from configuration eventually
    
    n_rg=int(n.floor(maximum_range_delay/rg))
    rgs=n.arange(n_rg)*rg
    rmax=n_rg

    # round trip speed of light in vacuum propagation, one microsecond
    rg_1us=c.c/1e6/2.0/1e3

    # range gates
    rgs_km=rgs*rg_1us

    # first entry in tx pulse metadata
    i0=idb[0]

    lpf=fft_lpf(10000,f0=1.2*pass_band,L=filter_len)

    decim=simple_decimator(L=10000,dec=rg)

    pwr_spec=n.zeros(fft_len,dtype=n.float32)
    spec_window=ss.hann(fft_len)

    # go through one integration window at a time
    for ai in range(rank,n_times,size):
        
        i0 = ai*int(avg_dur*idsr) + idb[0]

        if os.path.exists("%s/lpi-%d.png"%(output_prefix,int(i0/1e6))) and reanalyze==False:
            print("already analyzed %d"%(i0/1e6))
            continue

        # get info on all the pulses transmitted during this averaging interval
        # get some extra for gc
        sid = id_read.read(i0,i0+int(avg_dur*idsr)+40000,"sweepid")

        n_pulses=len(sid.keys())

        # USRP DC offset bug due to truncation instead of rounding.
        # Ryan Volz has a fix for firmware in USRPs.
        # note that this appears to change as a function of time
        # we can probably only estimate this from the estimated autocorrelation functions 
        z_dc=n.complex64(-0.212-0.221j)

        bg_samples=[]
        bg_plus_inj_samples=[]
        z_dc_samples=[]

        sidkeys=list(sid.keys())

        A=[]
        mgs=[]
        mes=[]
#        sigmas=[]
        idxms=[]
        rmins=[]

        sample0=800
        sample1=8200
        rdec=rg
        m0=int(n.round(sample0/rdec))
        m1=int(n.round(sample1/rdec))

        n_meas=m1-m0

        # count the number of good measurements encountered as a function of delay
        # in lagged products
        ok_count = n.zeros(n_meas,dtype=int)
        meas_count = n.zeros(n_meas,dtype=int)
        meas_delays_us = n.arange(m0,m1)*rdec
        
        pwr_spec[:]=0.0

        for li in range(n_lags):
            # determine what is the lowest range that can be estimated
            # rg0=(gc - txstart - 0.6*pulse_length + lag)/range_decimation
            rmin=int(n.round((sample0-111-480*min_tx_frac+lags[li])/rdec))
            cm=convolution_matrix(n.zeros(m1),rmin=rmin,rmax=rmax)
            rmins.append(rmin)
            idxms.append(cm["idxm"])
            A.append([])
            mgs.append([])
            mes.append([])


        # start at 3, because we may need to look back for GC
        for keyi in range(3,n_pulses-3):

            t0=time.time()
            key=sidkeys[keyi]

            if sid[key] not in tmm.keys():
                print("unknown pulse code %d encountered, halting."%(sid[key]))
                exit(0)

            z_echo=None
            zd=None

            z_echo = d_il.read_vector_c81d(key, 10000, channel) - z_dc

            # no filtering of tx to get better ambiguity function
            z_tx=n.copy(z_echo)

            if sid[key] == 300:
                if use_long_pulse == False:
                    # ignore long pulse
                    continue
                # if long pulse, then take the next long pulse
                next_key = sidkeys[keyi+3]
                z_echo1 = d_il.read_vector_c81d(next_key, 10000, channel) - z_dc
            elif sid[key] == sid[sidkeys[keyi+1]]:
                # if first AC, subtract next one
                next_key = sidkeys[keyi+1]
                z_echo1 = d_il.read_vector_c81d(next_key, 10000, channel) - z_dc

            elif sid[key] == sid[sidkeys[keyi-1]]:
                # if second AC, subtract previous one.
                next_key = sidkeys[keyi-1]
                z_echo1 = d_il.read_vector_c81d(next_key, 10000, channel) - z_dc

                if debug_gc_rem:
                    plt.plot(zd.real+2000)
                    plt.plot(zd.imag+2000)
                    plt.plot(z_echo.real)
                    plt.plot(z_echo.imag)            
                    plt.title(sid[key])            
                    plt.show()



            noise0=tmm[sid[key]]["noise0"]
            noise1=tmm[sid[key]]["noise1"]
            last_echo=tmm[sid[key]]["last_echo"]
            tx0=tmm[sid[key]]["tx0"]
            tx1=tmm[sid[key]]["tx1"]
            gc=tmm[sid[key]]["gc"]
            e_gc=tmm[sid[key]]["e_gc"]



            # filter noise injection.
            z_noise=n.copy(z_echo)
            z_noise=lpf.lpf(z_noise)

            # the dc offset changes
            z_dc_noise=n.mean(z_noise[(last_echo-500):last_echo])
            z_dc_samples.append(z_dc_noise)
            bg_samples.append( n.mean(n.abs(z_noise[(last_echo-500):last_echo]-z_dc_noise)**2.0) )
            bg_plus_inj_samples.append( n.mean(n.abs(z_noise[(noise0):noise1]-z_dc_noise)**2.0) )

            z_tx[0:tx0]=0.0
            z_tx[tx1:10000]=0.0

            # normalize tx pwr
            z_tx=z_tx/n.sqrt(n.sum(n.real(z_tx*n.conj(z_tx))))
            z_echo[last_echo:10000]=0.0
            z_echo1[last_echo:10000]=0.0

            z_echo[0:gc]=0.0
            z_echo1[0:gc]=0.0

            if False:
                # testing notching of frequencies.
                ZE=fft(z_echo)
                ZE1=fft(z_echo1)
                z_fftfreq=n.fft.fftfreq(len(z_echo),d=1/sr)

                if False:
                    plt.plot(n.fft.fftshift(z_fftfreq),n.fft.fftshift(10.0*n.log10(n.abs(ZE))**2.0))
                    plt.show()

                for freq_range in notch_freq_range:
                    fridx0=n.argmin(n.abs(z_fftfreq-freq_range[0]))
                    fridx1=n.argmin(n.abs(z_fftfreq-freq_range[1]))
                    noise_std=n.sqrt(0.25*(n.mean(n.abs(ZE[(fridx0-200):(fridx0-100)])**2.0)+n.mean(n.abs(ZE[(fridx1+100):(fridx1+200)])**2.0)+n.mean(n.abs(ZE1[(fridx0-200):(fridx0-100)])**2.0)+n.mean(n.abs(ZE1[(fridx1+100):(fridx1+200)])**2.0)))
                    nrand=fridx1-fridx0

                    ZE[fridx0:fridx1]=noise_std*(n.random.randn(nrand)+n.random.randn(nrand)*1j)/n.sqrt(2.0)
                    ZE1[fridx0:fridx1]=noise_std*(n.random.randn(nrand)+n.random.randn(nrand)*1j)/n.sqrt(2.0)

                if False:
                    plt.plot(n.fft.fftshift(z_fftfreq),n.fft.fftshift(10.0*n.log10(n.abs(ZE))**2.0))                
                    plt.show()

                z_echo=ifft(ZE)
                z_echo1=ifft(ZE1)

            

            # calculate power spectrum after notch
            Z=n.fft.fftshift(fft(spec_window*z_echo[(last_echo-fft_len):(last_echo)]))
            pwr_spec+=n.real(Z*n.conj(Z))

            z_echo=lpf.lpf(z_echo)
            z_echo1=lpf.lpf(z_echo1)

            zd=z_echo-z_echo1

            zd[0:gc]=n.nan
            z_echo[0:gc]=n.nan
            z_echo[last_echo:10000]=n.nan
            zd[last_echo:10000]=n.nan        
            t1=time.time()
            read_time=t1-t0
            t0=time.time()
            for li in range(n_lags):
                for ai in range(lag_avg):
                    amb=decim.decimate(z_tx[0:(len(z_tx)-lags[li+ai])]*n.conj(z_tx[lags[li+ai]:len(z_tx)]))

                    # gc removal by the T. Turunen subtraction of two pulses with the same code, transmitted in
                    # close proximity to one another.
                    measg=decim.decimate(zd[0:(len(z_echo)-lags[li+ai])]*n.conj(zd[lags[li+ai]:len(z_echo)]))

                    # no gc removal
                    mease=decim.decimate(z_echo[0:(len(z_echo)-lags[li+ai])]*n.conj(z_echo[lags[li+ai]:len(z_echo)]))

                    #TM=amb[idxms[li]]
                    # add a column of ones to allow an additional noise process that is independent of range
                    O=n.ones(m1,dtype=n.complex64)
                    O.shape=(m1,1)
                    TM=n.hstack([amb[idxms[li]],O])
                    TM=sparse.csc_matrix(TM[m0:m1,:])

                    mgs[li].append(measg[m0:m1])
                    mes[li].append(mease[m0:m1])
                    A[li].append(TM)
            t1=time.time()
            ambiguity_time=t1-t0
            print("prep %d/%d ambiguity time %1.2f read time %1.2f (s)"%(keyi,n_pulses,ambiguity_time,read_time))            

        acfs_g=n.zeros([rmax,n_lags],dtype=n.complex64)
        acfs_e=n.zeros([rmax,n_lags],dtype=n.complex64)
        
        # store noise autocorrelation function
        noise_e=n.zeros(n_lags,dtype=n.complex64)
        noise_g=n.zeros(n_lags,dtype=n.complex64)
        
        acfs_g[:,:]=n.nan
        acfs_e[:,:]=n.nan    

        acfs_var=n.zeros([rmax,n_lags],dtype=n.float32)
        acfs_var[:,:]=n.nan

        noise=n.median(bg_samples)    
        alpha=(n.median(bg_plus_inj_samples)-n.median(bg_samples))/T_injection 
        T_sys=noise/alpha

        for li in range(n_lags):
            print(li)
            
            AA=sparse.vstack(A[li])
            #print(AA.shape)
            mm_g=n.concatenate(mgs[li])
            mm_e=n.concatenate(mes[li])
            sigma_lp_est=n.zeros(len(mm_g))
            sigma_lp_est[:]=1.0
            
            n_ipp=0
            # remove outliers and estimate standard deviation 
            if True:
                print("ratio test")
                # tbd: estimate the fourth moments for lagged products
                # 
                # <(m_t m_{t+\tau}^*) (m_t^* m_{t+\tau})>
                # but also for this one:
                # <(m_t m_{t+\tau}^*) (m_t m_{t+\tau}^*)>
                # as it might not be zero when snr is high!!!
                # this would require doing the least-squares with
                # a slightly different method
                # 
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
                WF=fft(wf,localized_sigma.shape[0])
                for ri in range(mm_em.shape[1]):
                    # we need to wrap around, to avoid too low values.
                    localized_sigma[:,ri]=n.roll(n.sqrt(ifft(WF*fft(localized_sigma[:,ri])).real),-5)

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

                debug_outlier_test=False
                if debug_outlier_test:
                    plt.pcolormesh(mm_em.real.T)
                    plt.colorbar()
                    plt.show()


                # is this threshold too high?
                # maybe 6-7 might still be possible.
                mm_em[ratio_test > 10]=n.nan
                mm_gm[ratio_test_g > 10]=n.nan

                # these will be shit no matter what
                mm_em[localized_sigma > 100*msig]=n.nan
                mm_gm[localized_sigma > 100*msig]=n.nan

                ok_count+=n.sum((n.isnan(mm_em)!=True)*(n.isnan(mm_gm)!=True),axis=0)
                meas_count+=n_ipp
                
                if debug_outlier_test:            
                    plt.pcolormesh(mm_em.real.T)
                    plt.colorbar()
                    plt.show()

                    plt.pcolormesh(localized_sigma.T)
                    plt.colorbar()
                    plt.show()

                sigma_lp_est=localized_sigma
                sigma_lp_est.shape=(len(mm_g),)

                mm_gm.shape=(len(mm_g),)
                mm_em.shape=(len(mm_e),)
                mm_g=mm_gm
                mm_e=mm_em

            mm_g=mm_g/sigma_lp_est
            mm_e=mm_e/sigma_lp_est

            gidx = n.where( (n.isnan(mm_e)==False) & (n.isnan(mm_g)==False) & (n.isnan(sigma_lp_est) == False) )[0]
            print("%d/%d measurements good"%(len(gidx),len(mm_g)))

            # take outliers and bad measurements
            AA=AA[gidx,:]
            mm_g=mm_g[gidx]
            mm_e=mm_e[gidx]
            
            # at this point, we could add regularization to reduce range resolution on the top-side
            #
            # acf(rg[i])**rg[i]**2.0 = acf(rg[i+1])**rg[i+1]**2.0
            #
            # Something like this:
            # acf(rg[i]) - acf(rg[i+1])*(rg[i+1]**2.0/rg[i]**2.0) = 0
            #
            # n_rgs_this_lag = rmax-rmins[li]
            #

            
            srow=n.arange(len(gidx),dtype=int)
            scol=n.arange(len(gidx),dtype=int)
            sdata=1/sigma_lp_est[gidx]

            Sinv = sparse.csc_matrix( (sdata, (srow,scol)) ,shape=(len(gidx),len(gidx)))


            

            try:
                t0=time.time()
                # we should probably do a
                # AA=n.dot(AA,Sinv)
                # first. this would save all the Sinv dot products. no time to test and validate this now
                # 
                # A^H diag(1/sigma)
                AT=n.conj(AA.T).dot(Sinv)
                # A^H S^{-1} A (Fisher information matrix)
                ATA=AT.dot(n.dot(Sinv,AA)).toarray()

                # A^H \Sigma^{-1} m_g with ground clutter mitigation
                # note that 1/sigma is taken earlier when forming mm_g and mm_e
                # here we add a 1/sigma to get 1/sigma^2 on the diagonal of Sigma^{-1}
                ATm_g=AT.dot(mm_g)
                # A^H \Sigma^{-1} m_e no ground clutter mitigation
                # note that 1/sigma is taken earlier when forming mm_g and mm_e
                ATm_e=AT.dot(mm_e)

                # error covariance
                Sigma=n.linalg.inv(ATA)

                # ML estimate for ACF lag without ground clutter mitigation measures in place
                xhat_e=n.dot(Sigma,ATm_e)

                # ML estimate for ACF lag with ground clutter mitigation measures            
                xhat_g=n.dot(Sigma,ATm_g)

                t1=time.time()
                t_simple=t1-t0        
                print("simple %1.2f"%(t_simple))
                acfs_e[ rmins[li]:rmax, li ]=xhat_e[0:(rmax-rmins[li])]
                noise_e[li]=xhat_e[len(xhat_e)-1]
                acfs_g[ rmins[li]:rmax, li ]=xhat_g[0:(rmax-rmins[li])]
                noise_g[li]=xhat_g[len(xhat_g)-1]                

                acfs_var[ rmins[li]:rmax, li ] = n.diag(Sigma.real)[0:(rmax-rmins[li])]
            except:
                traceback.print_exc()
                print("something went wrong.")

        if save_acf_images:
            # plot real part of acf
            acf_std=1.77*n.nanmedian(n.abs(acfs_e.real))
            plt.pcolormesh(mean_lags,rgs_km[0:rmax],acfs_e.real,vmin=-acf_std,vmax=2*acf_std)
            plt.xlabel("Lag ($\mu$s)")
            plt.ylabel("Range (km)")
            plt.colorbar()
            plt.title("%s T_sys=%1.0f K"%(stuffr.unix2datestr(i0/sr),T_sys))
            plt.tight_layout()
            plt.savefig("%s/lpi-%d.png"%(output_prefix,i0/sr))
            plt.close()
            plt.clf()

        ho=h5py.File("%s/lpi-%d.h5"%(output_prefix,i0/sr),"w")
        ho["acfs_g"]=acfs_g       # pulse to pulse ground clutter removal
        ho["acfs_e"]=acfs_e       # no ground clutter removal
        ho["noise_e"]=noise_e     # store estimated noise ACF
        ho["noise_g"]=noise_g     # store estimated noise ACF   
        ho["acfs_var"]=acfs_var   # variance of the acf estimate
        ho["rgs_km"]=rgs_km[0:rmax]
        ho["lags"]=mean_lags/sr
        ho["i0"]=i0/sr
        ho["T_sys"]=T_sys     # T_sys = alpha*noise_power
        ho["alpha"]=alpha     # This can scale power to T_sys (e.g., noise_power = T_sys/alpha)   T_sys * power/noise_pwr = T_pwr
        ho["z_dc"]=n.median(z_dc_samples)
        ho["pass_band"]=pass_band        # sort of important to store this, as this defines the low pass filter  
        ho["filter_len"]=filter_len      #
        # keep track of how many lagged products are rejected as bad as a function of time delay
        ho["retained_measurement_fraction"]=n.array(ok_count/meas_count,dtype=n.float32)
        ho["meas_delays_us"]=meas_delays_us
        ho["diagnostic_pwr_spec"]=pwr_spec/n_pulses
        ho.close()

if __name__ == "__main__":
#    datadir="/mnt/data/juha/millstone_hill/isr/2023-09-05/usrp-rx0-r_20230905T214448_20230906T040054/"

    if False:
        datadir="/media/j/fee7388b-a51d-4e10-86e3-5cabb0e1bc13/isr/2023-09-05/usrp-rx0-r_20230905T214448_20230906T040054"    
        # Top-side
        lpi_files(dirname=datadir,
                  avg_dur=10,  # n seconds to average
                  channel="zenith-l",
                  rg=240,       # how many microseconds is one range gate
                  output_prefix="%s/lpi_240"%(datadir),
                  min_tx_frac=0.0, # of the pulse can be missing
                  pass_band=0.05e6, # +/- 50 kHz 
                  filter_len=10,    # short filter, less problems with correlated noise, more problems with RFI
                  maximum_range_delay=7200,
                  save_acf_images=True,
                  lag_avg=1,
                  reanalyze=False)
        exit(0)
    if True:
        dirname="/media/j/fee7388b-a51d-4e10-86e3-5cabb0e1bc13/isr/2023-09-28/usrp-rx0-r_20230928T211929_20230929T040533"
        lpi_files(dirname=dirname,
                  avg_dur=10,  # n seconds to average
                  channel="zenith-l",
                  rg=240,       # how many microseconds is one range gate
                  output_prefix="%s/lpi_240"%(dirname),
                  min_tx_frac=0.0, # how much of the pulse can be missing
                  filter_len=20,
                  pass_band=0.05e6,
                  maximum_range_delay=7200,
                  save_acf_images=False,
                  reanalyze=False,
                  lag_avg=1
                  )
        lpi_files(dirname=dirname,
                  avg_dur=10,  # n seconds to average
                  channel="zenith-l",
                  rg=120,       # how many microseconds is one range gate
                  output_prefix="%s/lpi_120"%(dirname),
                  min_tx_frac=0.0, # how much of the pulse can be missing
                  filter_len=20,
                  pass_band=0.05e6,
                  maximum_range_delay=7200,
                  save_acf_images=False,
                  reanalyze=False,
                  lag_avg=1
                  )
        lpi_files(dirname=dirname,
                  avg_dur=10,  # n seconds to average
                  channel="zenith-l",
                  rg=60,       # how many microseconds is one range gate
                  output_prefix="%s/lpi_60"%(dirname),
                  min_tx_frac=0.0, # how much of the pulse can be missing
                  filter_len=20,
                  pass_band=0.05e6,
                  maximum_range_delay=7200,
                  save_acf_images=False,
                  reanalyze=False,
                  lag_avg=1
                  )
        lpi_files(dirname=dirname,
                  avg_dur=10,  # n seconds to average
                  channel="zenith-l",
                  rg=30,       # how many microseconds is one range gate
                  output_prefix="%s/lpi_30"%(dirname),
                  min_tx_frac=0.5, # how much of the pulse can be missing
                  filter_len=20,
                  pass_band=0.05e6,
                  maximum_range_delay=7200,
                  save_acf_images=False,
                  reanalyze=False,
                  lag_avg=1
                  )

        
        exit(0)



    if False:
        dirname="/media/j/fee7388b-a51d-4e10-86e3-5cabb0e1bc13/isr/2023-09-24/usrp-rx0-r_20230924T200050_20230925T041059"
        lpi_files(dirname=dirname,
                  avg_dur=10,  # n seconds to average
                  channel="zenith-l",
                  rg=240,       # how many microseconds is one range gate
                  output_prefix="%s/lpi_240"%(dirname),
                  min_tx_frac=0.0, # how much of the pulse can be missing
                  filter_len=20,
                  pass_band=0.05e6,
                  maximum_range_delay=7200,
                  save_acf_images=True,
                  reanalyze=False,
                  lag_avg=1
                  )
        exit(0)        
        # E-region analysis
        # newly acquired fact: spectrally wide ambiguity functions mix more with out of band interference.
        # lower range resolutions works better in the presence of noise.
        # 30 microsecond gating is worse than 60 us or 120 us gating!
        lpi_files(dirname=dirname,
                  avg_dur=10,  # n seconds to average
                  channel="zenith-l",
                  rg=30,       # how many microseconds is one range gate
                  output_prefix="lpi_2023-09-24_30",
                  min_tx_frac=0.5, # how much of the pulse can be missing
                  filter_len=10,
                  pass_band=0.1e6,
                  maximum_range_delay=5000,
                  save_acf_images=False,
                  reanalyze=False,
                  lag_avg=3
                  )
        exit(0)
    # F1-region analysis
    lpi_files(dirname=dirname,
              avg_dur=10,  # n seconds to average
              channel="zenith-l",
              rg=60,       # how many microseconds is one range gate
              output_prefix="lpi_2023-09-24_60",
              min_tx_frac=0.3, # how much of the pulse can be missing
              filter_len=10,
              pass_band=0.1e6,
              maximum_range_delay=7200,
              save_acf_images=False,
              reanalyze=True
              )
    exit(0)
    
    if True:
        # F-region analysis
        lpi_files(dirname=datadir,
                  avg_dur=10,  # n seconds to average
                  channel="zenith-l",
                  rg=120,       # how many microseconds is one range gate
                  output_prefix="lpi_120",
                  min_tx_frac=0.0, # of the pulse can be missing
                  pass_band=0.1e6, # +/- 100 kHz 
                  filter_len=10,    # short filter, less problems with correlated noise, more problems with RFI
                  maximum_range_delay=7200,
                  save_acf_images=False,                  
                  reanalyze=True)
        
    
    if True:
        # F1-region analysis
        lpi_files(dirname=datadir,
                  avg_dur=10,  # n seconds to average
                  channel="zenith-l",
                  rg=60,       # how many microseconds is one range gate
                  output_prefix="lpi_60",
                  min_tx_frac=0.3, # how much of the pulse can be missing
                  filter_len=10,
                  pass_band=0.1e6,
                  maximum_range_delay=7200,
                  save_acf_images=False,
                  reanalyze=True
                  )

    if True:
        # E-region analysis
        # newly acquired fact: spectrally wide ambiguity functions mix more with out of band interference.
        # lower range resolutions works better in the presence of noise.
        # 30 microsecond gating is worse than 60 us or 120 us gating!
        lpi_files(dirname=datadir,
                  avg_dur=10,  # n seconds to average
                  channel="zenith-l",
                  rg=30,       # how many microseconds is one range gate
                  output_prefix="lpi_30",
                  min_tx_frac=0.5, # how much of the pulse can be missing
                  filter_len=10,
                  pass_band=0.1e6,
                  maximum_range_delay=4000,
                  save_acf_images=False,
                  reanalyze=True
                  )



