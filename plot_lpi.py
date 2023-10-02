import numpy as n
import glob
import matplotlib.pyplot as plt
import h5py

import scipy.signal as ss

import tx_power as txp


#zpm,mpm=txp.get_tx_power_model(dirn="/media/j/fee7388b-a51d-4e10-86e3-5cabb0e1bc13/isr/2023-09-05/usrp-rx0-r_20230905T214448_20230906T040054/metadata/powermeter")    
import sys
dirname=sys.argv[1]

    
fl=glob.glob("%s/lpi*.h5"%(dirname))
fl.sort()

N=1
acf_key="acfs_e"

h=h5py.File(fl[0],"r")
a=h["acfs_e"][()]

have_noise_est=False
n_noise=0
if "noise_e" in h.keys():
    noise=h["noise_e"][()]
    have_noise_est=True
    noise_acf=h["noise_e"][()]
    noise_acf=n.concatenate((noise_acf,n.conjugate(n.flip(noise_acf[1:len(noise_acf)]))))
    n_noise=len(noise_acf)
    
rmax=a.shape[0]

lag=0
nt=len(fl)
nlags=a.shape[1]
dt=n.diff(h["lags"][()])[0]
sf=n.fft.fftshift(n.fft.fftfreq(n_noise,d=dt))
A=n.zeros([nt,rmax,nlags],dtype=n.complex64)

if have_noise_est:
    NS=n.zeros([nt,n_noise],dtype=n.float32)

ts=n.zeros(nt)
tv=n.zeros(nt)

h.close()
wf=ss.hann(n_noise)

for fi,f in enumerate(fl):
    h=h5py.File(f,"r")
    print(f)
    a=h[acf_key][()]/h["alpha"][()]
    ts[fi]=h["T_sys"][()]
    A[fi,:,:]=a[:,0:A.shape[2]]
    rgs_km=h["rgs_km"][()]
    lags=h["lags"][()]    
    tv[fi]=(h["i0"][()])

    # estimate the spectrum of the interference component of the received signal
    if have_noise_est:
        noise_acf=h["noise_e"][()]
        noise_acf=n.conjugate(n.fft.fftshift(wf*(n.fft.fftshift(n.concatenate((noise_acf,n.conjugate(n.flip(noise_acf[1:len(noise_acf)]))))))))
        NS[fi,:]=n.fft.fftshift(n.abs(n.fft.fft(noise_acf)))

    
    h.close()
    if len(sys.argv) > 2:
        avg=int(sys.argv[2])
        A[fi,:,lag]=n.convolve(A[fi,:,lag],n.repeat(1.0/avg,avg),mode="same")

if have_noise_est:
    plt.pcolormesh(tv,sf/1e3,10.0*n.log10(NS.T),vmin=0,vmax=50)
    plt.title("RFI spectrum")
    plt.ylabel("Frequency offset (kHz)")
    plt.xlabel("Time (unix)")
    plt.colorbar()
    plt.show()


#for ti in ra
for ri in range(len(rgs_km)):
    if len(sys.argv) > 3:
        avg=int(sys.argv[3])
        A[:,ri,lag]=n.convolve(A[:,ri,lag],n.repeat(1.0/avg,avg),mode="same")


    
    A[:,ri,:]=A[:,ri,:]*rgs_km[ri]**2.0
dB=10.0*n.log10(A[:,:,lag].real.T)
nf=n.nanmedian(dB)
plt.pcolormesh(tv,rgs_km,dB,vmin=nf-10,vmax=nf+10,cmap="plasma")
cb=plt.colorbar()
cb.set_label("Range corrected volume scatter power (dB)")
plt.xlabel("Time (unix)")
plt.ylabel("Range (km)")
plt.show()
