import numpy as n
import glob
import matplotlib.pyplot as plt
import h5py

import scipy.signal as ss

fl=glob.glob("lpi2/lpi*.h5")
fl.sort()
h=h5py.File(fl[0],"r")
a=h["acfs_e"][()]
rmax=a.shape[0]
h.close()
lag=0
#rmax=200
nt=len(fl)
nlags=a.shape[1]
A=n.zeros([nt,rmax,nlags],dtype=n.complex64)

AV=n.zeros([nt,rmax,nlags],dtype=n.complex64)
tv=[]
acfs=n.zeros([rmax,nlags],dtype=n.complex64)
ts=[]
for fi,f in enumerate(fl):
    h=h5py.File(f,"r")
    print(f)
    a=h["acfs_g"][()]
    # remove DC offset and filter impulse response from highest altitudes
    #zacf=n.mean(a[(a.shape[0]-10):a.shape[0],:],axis=0)
    zacf=n.mean(a[100:111,:],axis=0)
    a=a-zacf
    
    v=h["acfs_var"][()]
    ts.append(h["T_sys"][()])
    if False:
        plt.subplot(121)
        plt.pcolormesh(n.sqrt(v.real))
        plt.colorbar()
        plt.subplot(122)
        plt.pcolormesh(a.real)
        plt.colorbar()
        plt.show()
   # sigma14=n.nanstd(n.abs(a[:,14]))
  #  sigma0=n.nanstd(n.abs(a[:,0]))    
#    bad_idx=n.where( (n.abs(a[:,0]) > 3*sigma0) & (n.abs(a[:,0])> *sigma0)  )[0]
 #   a[bad_idx,:]=n.nan
#    mi=n.nanargmax(a[:,0].real)
 #   print(mi)
  #  plt.plot(a[mi,:].real)
   # plt.plot(a[mi,:].imag)    
    #plt.plot(n.abs(a[mi,:]))
    #    plt.show()
    A[fi,:,:]=a
    AV[fi,:,:]=v

    acfs+=h["acfs_e"][()]
    
    rgs_km=h["rgs_km"][()]
    lags=h["lags"][()]    
    tv.append(h["i0"][()])


    
    
    h.close()

plt.subplot(211)    
plt.plot(ts)
plt.subplot(212)
#z_acf=
plt.pcolormesh(10.0*n.log10(AV[:,:,lag].real.T))
plt.colorbar()
plt.show()
    
#acfsn=n.copy(acfs)
AO=n.copy(A)
AOO=n.copy(A)
for ri in range(rmax):
#    acfsn[ri,:]=acfs[ri,:]/acfs[ri,0].real

    
#    med=n.nanmedian(A[:,ri,1].real)
    med=ss.medfilt(A[:,ri,lag].real,51)
    AO[:,ri,lag]=med
    std=1.77*n.nanmedian(n.abs(A[:,ri,lag].real-med))
    
#    bidx=n.where( n.abs(A[:,ri,lag].real-med) > 3*std)[0]
 #   A[bidx,ri,:]=n.nan

acfs=n.nanmean(A,axis=0)
# determine the filter impulse response effect from top range gates...
#zacf=n.mean(acfs[100:111,:],axis=0)
#acfs=acfs-zacf
#for i in range(A.shape[0]):
#    A[i,:,:]=A[i,:,:]-zacf

acfso=n.nanmean(AO,axis=0)
acfsn=n.copy(acfs)
for ri in range(rmax):
    acfsn[ri,:]=acfsn[ri,:]/acfsn[ri,lag].real


AA=n.copy(A)
N=0
for i in range(A.shape[0]-N):
    AA[i:(i+N),:]=n.nanmean(A[i:(i+N),:],axis=0)

    
    
plt.pcolormesh(lags*1e6,rgs_km,acfsn.real,vmin=-0.5,vmax=1.0)
plt.xlabel("Lag ($\mu$s)")
plt.ylabel("Range (km)")
plt.colorbar()
plt.show()

plt.pcolormesh(lags*1e6,rgs_km,acfs.real,vmin=-3e3,vmax=5e3)
plt.xlabel("Lag ($\mu$s)")
plt.ylabel("Range (km)")
plt.colorbar()
plt.show()


neraw=n.copy(AA[:,:,lag].real)
neraw[neraw<0]=n.nan#1e-9
for ri in range(rmax):
    # zero-lag is already ion-line power, without noise contributions
    # we'd want to divide by transmit power
    neraw[:,ri]=1e3*neraw[:,ri]*rgs_km[ri]**2.0

plt.semilogx(1e3*acfs[:,lag].real*rgs_km**2.0,rgs_km,".")
plt.show()


plt.pcolormesh(tv,rgs_km,n.log10(neraw.T),cmap="plasma",vmin=8.0,vmax=12)
plt.colorbar()
plt.show()


plt.pcolormesh(tv,rgs_km,AOO[:,:,lag].real.T,cmap="jet",vmin=-1e3,vmax=1e4)
plt.colorbar()
plt.show()

plt.pcolormesh(tv,rgs_km,A[:,:,lag].real.T,cmap="jet",vmin=-1e3,vmax=1e4)
plt.colorbar()
plt.show()
plt.pcolormesh(tv,rgs_km,AA[:,:,lag].real.T,cmap="jet",vmin=-1e3,vmax=1e4)
plt.colorbar()
plt.show()
