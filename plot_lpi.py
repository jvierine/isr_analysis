import numpy as n
import glob
import matplotlib.pyplot as plt
import h5py

import scipy.signal as ss
import stuffr

import tx_power as txp
plt.rcParams["date.autoformatter.minute"] = "%H:%M:%S"
plt.rcParams["date.autoformatter.hour"] = "%H:%M"


#zpm,mpm=txp.get_tx_power_model(dirn="/media/j/fee7388b-a51d-4e10-86e3-5cabb0e1bc13/isr/2023-09-05/usrp-rx0-r_20230905T214448_20230906T040054/metadata/powermeter")    
import sys
dirname=sys.argv[1]

    
fl=glob.glob("%s/lpi*.h5"%(dirname))
fl.sort()

tv=n.zeros(len(fl))
for fi,f in enumerate(fl):
    h=h5py.File(f,"r")
    tv[fi]=(h["i0"][()])

min_dt=n.min(n.diff(tv))
nt=int(n.ceil((n.max(tv)-n.min(tv))/min_dt))+1
t0=n.min(tv)
tv=n.zeros(nt)
tv_dt=n.zeros(nt)

tv_dt=[]
for i in range(nt):
    tv[i]=t0+min_dt*i
    print(tv[i])
    tv_dt.append(stuffr.unix2date(tv[i]))

N=1
acf_key="acfs_g"

h=h5py.File(fl[0],"r")
a=h[acf_key][()]

have_noise_est=False
n_noise=0
if "noise_e" in h.keys():
    noise=h["noise_e"][()]
    have_noise_est=True
    noise_acf=h["noise_e"][()]
    noise_acf=n.concatenate((noise_acf,n.conjugate(n.flip(noise_acf[1:len(noise_acf)]))))
    n_noise=len(noise_acf)
    
rmax=a.shape[0]

lag=1
#nt=len(fl)
nlags=a.shape[1]
dt=n.diff(h["lags"][()])[0]
sf=n.fft.fftshift(n.fft.fftfreq(n_noise,d=dt))
A=n.zeros([nt,rmax,nlags],dtype=n.complex64)
A[:,:,:]=n.nan

if have_noise_est:
    NS=n.zeros([nt,n_noise],dtype=n.float32)

ts=n.zeros(nt)
tv=n.zeros(nt)
ptx=n.zeros(nt)

h.close()
wf=ss.windows.hann(n_noise)

noise0=n.zeros(nt)

for fi,f in enumerate(fl):
    h=h5py.File(f,"r")
    print(f)
    tidx=int(n.floor((h["i0"][()]-t0)/min_dt))
    ptx[tidx]=h["P_tx"][()]    
    a=h[acf_key][()]/h["alpha"][()]/ptx[tidx]
    ts[tidx]=h["T_sys"][()]

    A[tidx,:,:]=a[:,0:A.shape[2]]
    rgs_km=h["rgs_km"][()]
    lags=h["lags"][()]    

    # estimate the spectrum of the interference component of the received signal
    if have_noise_est:

        noise_acf=h["noise_e"][()]
        noise0[tidx]=noise_acf[0].real
        #plt.plot(noise_acf.real)
        #plt.plot(noise_acf.imag)
        #plt.show()
        
        noise_acf=n.conjugate(n.fft.fftshift(wf*(n.fft.fftshift(n.concatenate((noise_acf,n.conjugate(n.flip(noise_acf[1:len(noise_acf)]))))))))
        NS[tidx,:]=n.fft.fftshift(n.abs(n.fft.fft(noise_acf)))
    
    h.close()
    if len(sys.argv) > 2:
        avg=int(sys.argv[2])
        A[tidx,:,lag]=n.convolve(A[tidx,:,lag],n.repeat(1.0/avg,avg),mode="same")

if have_noise_est and False:
    plt.plot(tv_dt,noise0)
    plt.xlabel("Time (unix)")
    plt.ylabel("Noise acf zero lag")
    plt.show()
    plt.pcolormesh(tv_dt,sf/1e3,10.0*n.log10(NS.T),vmin=0)
    plt.title("RFI spectrum")
    plt.ylabel("Frequency offset (kHz)")
    plt.xlabel("Time (unix)")
    plt.colorbar()
    plt.show()

if False:
    plt.plot(tv_dt,ptx)
    plt.xlabel("Time (unix)")
    plt.ylabel("TX Power")
    plt.show()
#for ti in ra

if True:
    avg=4
    for ri in range(len(rgs_km)):
        #  if len(sys.argv) > 3:
        #     avg=int(sys.argv[3])
        A[:,ri,lag]=n.convolve(A[:,ri,lag],n.repeat(1.0/avg,avg),mode="same")
        
        A[:,ri,:]=A[:,ri,:]*rgs_km[ri]**2.0


def prune_nan_col(t,A):
    AA=[]
    tt=[]
    print(A.shape)
    for i in range(len(t)):
        if n.sum(n.isnan(A[:,i]))!=A.shape[0]:
            tt.append(t[i])
            AA.append(A[:,i])

    return(n.array(tt),n.array(AA))
            
    

dB=10.0*n.log10(A[:,:,lag].real.T)
print(dB.shape)
nf=n.nanmedian(dB)
tv_dt,dB=prune_nan_col(tv_dt,dB)
print(len(tv_dt))
print(dB.shape)
plt.pcolormesh(tv_dt,rgs_km,dB.T,vmin=nf-10,vmax=nf+10,cmap="plasma")
cb=plt.colorbar()
cb.set_label("Range corrected volume scatter power (dB)")
plt.xlabel("Time since %s"%(tv_dt[0]))
plt.ylabel("Range (km)")
plt.show()
