import numpy as n
import matplotlib.pyplot as plt
import h5py
import scipy.signal as s
import glob
import tx_power as txp
import scipy.signal as ss

def forward(pars):
    fitpar={"rgs":n.array([200,250,300,350,400,450,500,650,800,900,1000])
            }


def fit(zpm,mpm,fname="il_1693950288.h5"):
    print(fname)
    h=h5py.File(fname,"r")
    T_sys=h["T_sys"][()]
    use_ac=True
    if use_ac:
        LP=n.copy(h["RDS_AC"][()])
        TXLP=n.copy(h["TX_AC"][()])
    else:
        LP=n.copy(h["RDS_LP"][()])
        TXLP=n.copy(h["TX_LP"][()])

    LP[n.isnan(LP)]=0.0

    # simple deconvolution
#    LPM=ss.medfilt2d(LP,7)
 #   plt.pcolormesh(LP-LPM)
  #  plt.show()
    

    TXLP=TXLP/n.max(TXLP)
    
    I=n.copy(LP)
    I[:,:]=0.0
    I[133,55]=1.0

    # This works!
    #    C=s.convolve2d(I,TXLP,boundary="wrap",mode="same")
    #    plt.pcolormesh(C)
    #    plt.colorbar()
    #    plt.show()
    
    dop_hz=h["dop_hz"][()]
    rgs_km=h["rgs_km"][()]
    t0=h["i0"][()]/1e6
    
    dop_idx0=n.where(dop_hz < -0.08e6)[0]
    dop_idx1=n.where(dop_hz > 0.04e6)[0]    
    rg_idx= n.where((rgs_km > 800) & (rgs_km < 1000) )[0]

    il_idx=n.where((dop_hz > -18e3) & (dop_hz < 18e3))[0]
    print(il_idx)

#    nf=n.median(LP[rg_idx[0]:rg_idx[1],0:dop_idx0[-1]])

  #  plt.pcolormesh(10.0*n.log10(LP))
 #   plt.colorbar()
#    plt.show()

  
    if use_ac:
        nf=n.mean(LP[226:234,:],axis=0)
    else:
        nf=n.median(LP[220:230,:])
        
    s_temp=(LP-nf)/nf
    
#    s_temp[250:s_temp.shape[0],:]=0.0

 #   plt.pcolormesh(s_temp,vmin=0)
#    plt.colorbar()
  #  plt.show()

 #   FT=n.fft.fft2(TXLP)
#    plt.pcolormesh(n.abs(FT))
 #   plt.colorbar()
   # plt.show()
    
  #  plt.pcolormesh(n.abs(TXLP))
 #   plt.colorbar()
#    plt.show()
#    amp=n.mean(n.abs(FT))
    
#    DE=n.fft.ifft2(n.fft.fft2(s_temp)/(FT+3)).real
 #   plt.pcolormesh(DE)
  #  plt.colorbar()
   # plt.show()
    
    
#    plt.pcolormesh(10.0*n.log10(s_temp),vmin=-20,vmax=13)
 #   plt.colorbar()
  #  plt.show()
    s_temp=s_temp/zpm(t0)
    #s_temp[s_temp<0]=n.nan
    zi=n.argmin(n.abs(dop_hz))
    if use_ac:
        raw_pow=n.max(s_temp[:,33:69],axis=1)
    else:
        raw_pow=n.nansum(s_temp[:,33:69],axis=1)
        
#    raw_pow=n.mean(s_temp[:,33:69],axis=1)    
#    plt.plot(raw_pow)
 #   plt.show()
    if False:
        plt.plot(raw_pow,rgs_km)
        plt.show()
        plt.pcolormesh(dop_hz/1e3,rgs_km,s_temp,vmin=0)
        plt.title(T_sys)
        plt.colorbar()
        plt.show()
        h.close()
   
    return(raw_pow,rgs_km,T_sys)




if __name__ == "__main__":
    fl=glob.glob("tmpe/il_*.h5")
    fl.sort()
    zpm,mpm=txp.get_tx_power_model(dirn="/media/j/fee7388b-a51d-4e10-86e3-5cabb0e1bc13/isr/2023-09-05/usrp-rx0-r_20230905T214448_20230906T040054/metadata/powermeter")    
    pwr,rgs,ts=fit(zpm,mpm,fl[0])
    S=n.zeros([len(fl),len(pwr)])


    tsys=n.zeros(len(fl))
    for fi,f in enumerate(fl):
        pwr,rgs,ts=fit(zpm,mpm,f)
        tsys[fi]=ts
        # raw ne   N_e = alpha * T_n * snr * R**2.0 / (1+te/ti)
        S[fi,:]=ts*pwr*(rgs**2.0)

#    plt.plot(tsys)
 #   plt.show()
    plt.subplot(211)
    plt.pcolormesh(n.arange(S.shape[0]),rgs,n.log10(S.T),cmap="jet",vmin=-1,vmax=2)
#    plt.pcolormesh(n.arange(S.shape[0]),rgs,n.log10(S.T),cmap="plasma",vmin=2,vmax=4)    
#    plt.ylim([100,800])
    plt.colorbar()
    plt.subplot(212)
    plt.plot(tsys)
    plt.show()

    exit(0)
    SD=n.copy(S)
    SD[:,:]=0.0

    for ri in range(S.shape[1]):
        nf=ss.medfilt(S[:,ri],21)
        SD[:,ri]=(S[:,ri]-nf)/nf
    plt.pcolormesh(SD.T,vmin=-0.2,vmax=0.2)
    plt.colorbar()
    plt.show()
