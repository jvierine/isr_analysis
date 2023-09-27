import numpy as n
import glob
import matplotlib.pyplot as plt
import h5py

import scipy.signal as ss

import tx_power as txp


zpm,mpm=txp.get_tx_power_model(dirn="/media/j/fee7388b-a51d-4e10-86e3-5cabb0e1bc13/isr/2023-09-05/usrp-rx0-r_20230905T214448_20230906T040054/metadata/powermeter")    

dirname="lpi_f2"
fl=glob.glob("%s/lpi*.h5"%(dirname))
fl.sort()

if dirname=="lpi_ts":
    filter_impulse_rem=False
    # number of files to post average
    N=1
    acf_key="acfs_e"

if dirname=="lpi_f2":
    filter_impulse_rem=False
    # number of files to post average
    N=6
    acf_key="acfs_g"
if dirname=="lpi2":    
    filter_impulse_rem=True
    zidx0=100
    zidx1=111    
    # number of files to post average
    N=6
    acf_key="acfs_g"
if dirname=="lpi_e":
    filter_impulse_rem=True    
#    filter_impulse_rem=False
    # number of files to post average
    N=10
    zidx0=180
    zidx1=222    
    
    acf_key="acfs_g"



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

ws=n.zeros([rmax,nlags])
for fi,f in enumerate(fl):
    h=h5py.File(f,"r")
    print(f)
    a=h[acf_key][()]
    # remove DC offset and filter impulse response from highest altitudes
    #zacf=n.mean(a[(a.shape[0]-10):a.shape[0],:],axis=0)
#    plt.pcolormesh(a.real)
 #   plt.show()
    # we used a very long filter impulse response, and we need to remove the correlated noise contribution
    # should no longer be a problem with L=20 filter length...
    if filter_impulse_rem:
        zacf=n.mean(a[zidx0:zidx1,:],axis=0)
        a=a-zacf
    
    v=h["acfs_var"][()]
    ts.append(h["T_sys"][()])
    if False:
        plt.pcolormesh(n.abs(a),vmin=0,vmax=2e3)
        plt.colorbar()
        plt.show()
        
    A[fi,:,:]=a
    AV[fi,:,:]=v

    acfs+=a/v
    ws+=1/v
    rgs_km=h["rgs_km"][()]
    lags=h["lags"][()]    
    tv.append(h["i0"][()])
    if "alpha" in h.keys():
        alpha=h["alpha"][()]
        print(alpha)
    
    h.close()
    
plt.subplot(211)    
plt.plot(tv,ts)
plt.subplot(212)
#z_acf=
plt.pcolormesh(tv,rgs_km,10.0*n.log10(AV[:,:,lag].real.T))
plt.show()
    
#acfsn=n.copy(acfs)
AO=n.copy(A)
AOO=n.copy(A)
r0=n.where(rgs_km>250)[0][0]
for ri in range(r0,rmax):
#    acfsn[ri,:]=acfs[ri,:]/acfs[ri,0].real

    
#    med=n.nanmedian(A[:,ri,1].real)
    med=ss.medfilt(A[:,ri,lag].real,51)
    AO[:,ri,lag]=med
    std=1.77*n.nanmedian(n.abs(A[:,ri,lag].real-med))
    
    bidx=n.where( n.abs(A[:,ri,lag].real-med) > 3.0*std)[0]
    A[bidx,ri,:]=n.nan

    
acfs=acfs/ws
# determine the filter impulse response effect from top range gates...
#zacf=n.mean(acfs[100:111,:],axis=0)
#acfs=acfs-zacf
#for i in range(A.shape[0]):
#    A[i,:,:]=A[i,:,:]-zacf

acfso=n.nanmean(AO,axis=0)
acfsn=n.copy(acfs)
for ri in range(rmax):
    acfsn[ri,:]=acfsn[ri,:]/acfsn[ri,lag].real


n_dec=int(n.floor(A.shape[0]/N))
A_avg=n.zeros([A.shape[1],A.shape[2]],dtype=n.complex64)
WS=n.zeros([A.shape[1],A.shape[2]],dtype=n.complex64)
AA=n.copy(A)
AA[:,:]=0.0
for i in range(A.shape[0]-N):
    A_avg[:,:]=0.0
    WS[:,:]=0.0    
    for j in range(N):
        w=1/AV[i+j,:,:]
        sel=n.isnan(A[i+j,:,:])!=True
        A_avg[sel]+=A[i+j,sel]*w[sel]
        WS[sel]+=1/AV[i+j,sel]
    AA[i,:,:]=A_avg/WS

    if False:
        B=n.copy(AA[i,:,:])
#        for ri in range(AA.shape[1]):
 #           B[ri,:]=B[ri,:]/B[ri,0].real

        ho=h5py.File("avg.h5","w")
        ho["acf"]=B
        ho["lag"]=lags
        ho["rgs"]=rgs_km
        ho["var"]=1/WS
        ho.close()
        
        plt.subplot(121)
        plt.pcolormesh(lags*1e6,rgs_km,B.real,vmin=-0.5,vmax=1.0)
        plt.xlabel("Lag (us)")
        plt.ylabel("Range (km)")
        plt.title("Real")
        plt.colorbar()
        plt.subplot(122)
        plt.pcolormesh(lags*1e6,rgs_km,B.imag,vmin=-0.1,vmax=0.1)
        plt.title("Imaginary")    
        plt.xlabel("Lag (us)")
        plt.ylabel("Range (km)")
        plt.colorbar()
        plt.tight_layout()
        plt.show()

    
    
plt.pcolormesh(lags*1e6,rgs_km,acfsn.real,vmin=-0.5,vmax=1.0)
plt.xlabel("Lag ($\mu$s)")
plt.ylabel("Range (km)")
plt.colorbar()
plt.show()

plt.pcolormesh(lags*1e6,rgs_km,acfs.real,vmin=-2e3,vmax=5e3)
plt.xlabel("Lag ($\mu$s)")
plt.ylabel("Range (km)")
plt.colorbar()
plt.show()


neraw=n.copy(AA[:,:,lag].real)
neraw[neraw<0]=n.nan#1e-9

for ri in range(rmax):
    # zero-lag is already ion-line power, without noise contributions
    # we'd want to divide by transmit power
    neraw[:,ri]=1e9*neraw[:,ri]*rgs_km[ri]**2.0/zpm(tv)

peak_hgts=[]
peak_nes=[]    
for ti in range(neraw.shape[0]):
    # f region between 25 and 50
    # a*x**2 + bx +c = 0
    # 2*a*x + b = 0
    # x = -b/2a
    # 

    midx=n.where( (rgs_km > 200) & (rgs_km < 450) & (n.isnan(neraw[ti,:])==False) )[0]
    n_m=len(midx)
    print(rgs_km[midx])

    x0=n.mean(rgs_km[midx])
    print(x0)
    xp=rgs_km[midx]-x0
    AM=n.zeros([n_m,3])
    AM[:,0]=1
    AM[:,1]= xp
    AM[:,2]= xp**2.0
    xhat=n.linalg.lstsq(AM,neraw[ti,midx]/1e11)[0]
    print(xhat)
    x_peak=(-xhat[1]/(2*xhat[2]))+x0
    peak_nes.append( 1e11*(xhat[0]+xhat[1]*(x_peak-x0)+xhat[2]*(x_peak-x0)**2.0) )
    print(x_peak)
    peak_hgts.append(x_peak)
  #  plt.plot(rgs_km[midx],neraw[ti,midx],".")
 #   plt.plot(rgs_km[midx],1e11*n.dot(A,xhat))
#    plt.show()


plt.plot(tv,peak_nes,".")
plt.show()

plt.semilogx(1e3*acfs[:,lag].real*rgs_km**2.0,rgs_km,".")
plt.show()


plt.pcolormesh(tv,rgs_km,n.log10(neraw.T),cmap="plasma",vmin=9.0,vmax=12.5)
plt.plot(tv,peak_hgts,".")
plt.colorbar()
plt.show()


plt.pcolormesh(tv,rgs_km,AOO[:,:,lag].real.T,cmap="plasma",vmin=0,vmax=1e4)
plt.title("The Starlink-layer of the atmosphere")
plt.xlabel("Time (unix)")
plt.ylabel("Altitude (km)")
cb=plt.colorbar()
cb.set_label("Power (arb)")
plt.show()

plt.pcolormesh(tv,rgs_km,A[:,:,lag].real.T,cmap="plasma",vmin=-1e3,vmax=1e4)
plt.colorbar()
plt.show()
plt.pcolormesh(tv,rgs_km,AA[:,:,lag].real.T,cmap="plasma",vmin=-1e3,vmax=1e4)
plt.colorbar()
plt.show()
