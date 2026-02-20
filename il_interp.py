import numpy as n
import matplotlib.pyplot as plt
import h5py
import scipy.interpolate as sint
import isr_spec

class ilint:
    def __init__(self,fname="ion_line_interpolate.h5"):
        h=h5py.File(fname,"r")
        self.S=h["S"][()]   # 3d array molecular to atomic fraction x te/ti x ti

        
        self.te_ti_ratio=h["te_ti_ratios"][()]
        self.dte_ti_ratio=n.diff(self.te_ti_ratio)[0]
        self.te_ti_ratio0=n.min(self.te_ti_ratio)
        self.te_ti_ratio1=n.max(self.te_ti_ratio)
        self.te_ti_ratioN=len(self.te_ti_ratio)

        self.mass0=h["mass0"][()]
        self.mass1=h["mass1"][()]
        
        self.mol_fracs=h["mol_fracs"][()]
        self.dmol_fracs=n.diff(self.mol_fracs)[0]
        self.mol_fracs0=n.min(self.mol_fracs)
        self.mol_fracs1=n.max(self.mol_fracs)
        self.mol_fracsN=len(self.mol_fracs)        
        
        self.tis=h["tis"][()]
        self.dtis=n.diff(self.tis)[0]
        self.tis0=n.min(self.tis)
        self.tis1=n.max(self.tis)
        self.tisN=len(self.tis)        
        
        self.radar_freq=h["freq"][()]
        self.doppler_hz=h["om"][()]/2/n.pi
        self.ne0=h["ne"][()]
        self.sr=h["sample_rate"][()]

        self.n_fft=self.S.shape[3]
        self.n_lags=int(self.S.shape[3]/2)
        self.lag=n.arange(self.n_lags)/self.sr

        print("Read interpolation table for radar frequency %1.0f MHz\nmass1=%1.1f amu, mass2=%1.1f amu"%(self.radar_freq/1e6,self.mass0,self.mass1))


        self.S[:,:,:,n.abs(self.doppler_hz)>40e3]=0.0
        
        create_acfs=False
        if "A" in h.keys():
            self.A=h["A"][()]
        else:
            create_acfs=True
        h.close()

        if create_acfs:

            print("Creating acfs")
            self.A=n.zeros([self.S.shape[0],self.S.shape[1],self.S.shape[2],self.n_lags],dtype=n.complex64)
            # calculate acfs
            for fr_i in range(self.mol_fracsN):
                print("%d"%(fr_i))
                for teti_i in range(self.te_ti_ratioN):
                    for ti_i in range(self.tisN):
                        spec=self.S[fr_i,teti_i,ti_i,:]
                        spec[n.abs(self.doppler_hz)>40e3]=0.0
                        nan_idx=n.where(n.isnan(spec))[0]
                        if len(nan_idx) > 0:
                            print("nans in")
                            print(self.tis[ti_i])
                            continue
                            spec[nan_idx]=0.5*(spec[nan_idx-1]+spec[nan_idx+1])

                        acf=n.fft.ifft(n.fft.fftshift(spec))
                        self.A[fr_i,teti_i,ti_i,:]=acf[0:self.n_lags]
            h=h5py.File(fname,"a")
            print("storing acfs")
            if "A" in h.keys():
                del(h["A"])
            h["A"]=self.A
            h.close()

        


    def getspec(self,
                ne=n.array([1e11]),
                te=n.array([600]),
                ti=n.array([300]),
                mol_frac=n.array([1.0]),
                vi=n.array([0.0]),
                acf=False,
                normalize=True,
                interpf=False,
                debug=False):
        
        te_ti_ratio_idx=(te/ti - self.te_ti_ratio0)/self.dte_ti_ratio
        # edge cases
        te_ti_ratio_idx[te_ti_ratio_idx<=0]=1e-4
        te_ti_ratio_idx[te_ti_ratio_idx>=(self.te_ti_ratioN-1) ]=self.te_ti_ratioN-1-1e-4

        mol_frac_idx=(mol_frac - self.mol_fracs0)/self.dmol_fracs
        # edge cases
        mol_frac_idx[mol_frac_idx<=0]=1e-4
        mol_frac_idx[mol_frac_idx>=(self.mol_fracsN-1) ]=self.mol_fracsN-1-1e-4

        ti_idx=(ti - self.tis0)/self.dtis
        # edge cases
        ti_idx[ti_idx<=0]=1e-4
        ti_idx[ti_idx>=(self.tisN-1) ]=self.tisN-1-1e-4


        # p_rx(ne,te,ti) = ne/(1+te/ti)
        # p_rx(ne,te,ti)/p_rx(ne0,te,ti) = ne/ne0
        pwr_scaling_factor=ne/self.ne0

        # and now linearly interpolate this thing.
        # there are three dimensions, so we need to look at eight corners

        #specs=[]
        if acf:
            S=n.zeros([len(ne),self.A.shape[3]],dtype=n.complex64)
            L=self.A
        else:
            S=n.zeros([len(ne),self.S.shape[3]],dtype=n.float32)
            L=self.S            

        # for all parameter triplets
        for i in range(len(ne)):
            w00=1.0-(mol_frac_idx[i]-n.floor(mol_frac_idx[i]))  # how close to floor [0,1]
            w01=1.0-(n.ceil(mol_frac_idx[i])-mol_frac_idx[i])   # how close to ceil

            w10=1.0-(te_ti_ratio_idx[i]-n.floor(te_ti_ratio_idx[i]))
            w11=1.0-(n.ceil(te_ti_ratio_idx[i])-te_ti_ratio_idx[i])

            w20=1.0-(ti_idx[i]-n.floor(ti_idx[i]))
            w21=1.0-(n.ceil(ti_idx[i])-ti_idx[i])

            if debug:
                print("weights %1.2f,%1.2f-%d,%d %1.2f,%1.2f-%d,%d %1.2f,%1.2f - %d,%d"%(w00,w01,n.floor(mol_frac_idx[i]),n.ceil(mol_frac_idx[i]),
                                                                                         w10,w11,n.floor(te_ti_ratio_idx[i]),n.ceil(te_ti_ratio_idx[i]),
                                                                                         w20,w21,n.floor(ti_idx[i]),n.ceil(ti_idx[i])))
            # 000
            w0=w00*w10*w20
            S[i,:] += w0*L[int(n.floor(mol_frac_idx[i])), int(n.floor(te_ti_ratio_idx[i])), int(n.floor(ti_idx[i])), :]

            # 001
            w1=w00*w10*w21
            S[i,:] += w1*L[int(n.floor(mol_frac_idx[i])), int(n.floor(te_ti_ratio_idx[i])), int(n.ceil(ti_idx[i])), :]

            # 010
            w2=w00*w11*w20
            S[i,:] += w2*L[int(n.floor(mol_frac_idx[i])), int(n.ceil(te_ti_ratio_idx[i])), int(n.floor(ti_idx[i])), :]

            # 011
            w3=w00*w11*w21
            S[i,:] += w3*L[int(n.floor(mol_frac_idx[i])), int(n.ceil(te_ti_ratio_idx[i])), int(n.ceil(ti_idx[i])), :]
            
            # 100
            w4=w01*w10*w20
            S[i,:] += w4*L[int(n.ceil(mol_frac_idx[i])), int(n.floor(te_ti_ratio_idx[i])), int(n.floor(ti_idx[i])), :]

            # 101
            w5=w01*w10*w21
            S[i,:] += w5*L[int(n.ceil(mol_frac_idx[i])), int(n.floor(te_ti_ratio_idx[i])), int(n.ceil(ti_idx[i])), :]

            # 110
            w6=w01*w11*w20
            S[i,:] += w6*L[int(n.ceil(mol_frac_idx[i])), int(n.ceil(te_ti_ratio_idx[i])), int(n.floor(ti_idx[i])), :]
            
            # 111
            w7=w01*w11*w21
            S[i,:] += w7*L[int(n.ceil(mol_frac_idx[i])), int(n.ceil(te_ti_ratio_idx[i])), int(n.ceil(ti_idx[i])), :]
            
            S[i,:]=S[i,:]/(w0+w1+w2+w3+w4+w5+w6+w7)
            
            if normalize:
                if acf:
                    S[i,:]=S[i,:]/S[i,0].real
                else:
                    S[i,:]=S[i,:]/n.max(S[i,])
            else:
                S[i,:]=pwr_scaling_factor[i]*S[i,:]

        if interpf:
            funs=[]
            for i in range(S.shape[0]):
                funs.append(sint.interp1d(self.doppler_hz,S[i,:],kind="cubic"))
            return(funs)
        else:
            return(S)
        

def testne():
    il=ilint()
    ne=n.array([1e10,1e11,1e12])
    S=il.getspec(ne=ne,
                 te=n.array([1000,1000,1000]),
                 ti=n.array([900,900,900]),
                 mol_frac=n.array([1,1,1]),
                 vi=n.array([0,0,0]),
                 acf=False
                 )

    for i in range(S.shape[0]):
        plt.semilogy(il.doppler_hz,S[i,:],label="ne=%1.2g"%(ne[i]),color="black",alpha=0.3)
#    plt.legend()
    plt.show()

def testte():
    il=ilint()
    ne=n.repeat(1e12,400)
    te=n.linspace(500,4000,num=400)
    ti=n.repeat(500,400)
    mol_frac=n.repeat(0.1,400)
    vi=n.repeat(0,400)
    A=il.getspec(ne=ne,
                 te=te,
                 ti=ti,
                 mol_frac=mol_frac,
                 vi=vi,
                 acf=True
                 )
    S=il.getspec(ne=ne,
                 te=te,
                 ti=ti,
                 mol_frac=mol_frac,
                 vi=vi,
                 acf=False
                 )


    plt.subplot(121)
    plt.pcolormesh(il.lag*1e6,te,A.real)
    plt.xlabel("Lag (us)")
    plt.xlim([0,500])
    plt.ylabel("Te (K)")
    plt.colorbar()
    plt.subplot(122)
    plt.pcolormesh(il.doppler_hz/1e3,te,S)
    plt.xlabel("Doppler (kHz)")
    plt.ylabel("Te (K)")    
    plt.colorbar()
    plt.tight_layout()
    plt.show()
    
#    for i in range(S.shape[0]):
 #       plt.plot(il.doppler_hz,S[i,:],color="black",alpha=0.3)

#    plt.legend()
#    plt.show()


def testti():
    il=ilint()
    ne=n.repeat(1e12,100)
    ti=n.linspace(100,3000,num=100)
    te=n.copy(ti)
    mol_frac=n.repeat(0.0,100)
    vi=n.repeat(0,100)
    S=il.getspec(ne=ne,
                 te=te,
                 ti=ti,
                 mol_frac=mol_frac,
                 vi=vi,
                 acf=False
                 )
    A=il.getspec(ne=ne,
                 te=te,
                 ti=ti,
                 mol_frac=mol_frac,
                 vi=vi,
                 acf=True
                 )

    plt.subplot(121)
    plt.pcolormesh(il.lag*1e6,ti,A.real)
    plt.xlim([0,1000])
    plt.xlabel("Lag (us)")
    plt.ylabel("T_i (K)")    
    plt.colorbar()
    plt.subplot(122)
    plt.pcolormesh(il.doppler_hz/1e3,ti,S)
    plt.xlabel("Doppler (kHz)")
    plt.ylabel("T_i (K)")
    plt.colorbar()
    plt.tight_layout()    
    plt.show()

def test_h():
    il=ilint(fname="ion_line_interpolate_h_o.h5")
    ne=n.repeat(1e12,100)
    ti=n.repeat(1500,100)
    te=n.repeat(2000,100)    
    mol_frac=n.linspace(0,1,num=100)
    vi=n.repeat(0,100)
    
    S=il.getspec(ne=ne,
                 te=te,
                 ti=ti,
                 mol_frac=mol_frac,
                 vi=vi,
                 acf=False
                 )
    plt.subplot(121)
    plt.pcolormesh(il.doppler_hz/1e3,mol_frac,S)
    plt.ylabel("O+ fraction")
    plt.xlabel("Doppler (kHz)")
    plt.colorbar()

    ne=n.repeat(1e12,100)
    ti=n.repeat(1500,100)
    te=n.linspace(1500,4000,100)    
    mol_frac=n.repeat(1,100)
    vi=n.repeat(0,100)
    
    S=il.getspec(ne=ne,
                 te=te,
                 ti=ti,
                 mol_frac=mol_frac,
                 vi=vi,
                 acf=False
                 )
    plt.subplot(122)
    plt.pcolormesh(il.doppler_hz/1e3,te,S)
    plt.ylabel("Te")
    plt.xlabel("Doppler (kHz)")
    plt.colorbar()
    plt.show()



    ne=n.repeat(1e12,100)
    ti=n.repeat(1500,100)
    te=n.repeat(2000,100)    
    mol_frac=n.linspace(0,1,num=100)
    vi=n.repeat(0,100)
    
    S=il.getspec(ne=ne,
                 te=te,
                 ti=ti,
                 mol_frac=mol_frac,
                 vi=vi,
                 acf=True
                 )
    plt.subplot(121)
    plt.pcolormesh(il.lag*1e6,mol_frac,S.real)
    plt.ylabel("O+ fraction")
    plt.xlabel("Lag (us)")
    plt.colorbar()

    ne=n.repeat(1e12,100)
    ti=n.repeat(1500,100)
    te=n.linspace(1500,4000,100)    
    mol_frac=n.repeat(1,100)
    vi=n.repeat(0,100)
    
    S=il.getspec(ne=ne,
                 te=te,
                 ti=ti,
                 mol_frac=mol_frac,
                 vi=vi,
                 acf=True
                 )
    plt.subplot(122)
    plt.pcolormesh(il.lag*1e6,te,S.real)
    plt.ylabel("Te")
    plt.xlabel("Lag (us)")
    plt.colorbar()
    
    
    plt.show()

def example_spectra():
    
    il=ilint(fname="ion_line_interpolate_32_16_430_00.h5")
    
    ne=n.repeat(1e11,2)
    ti=n.repeat(700,2)
    te=n.repeat(700*1.5,2) 
    mol_frac=n.array([0,1])
    vi=n.repeat(0,2)
    
    S=il.getspec(ne=ne,
                 te=te,
                 ti=ti,
                 mol_frac=mol_frac,
                 vi=vi,
                 acf=False,
                 interpf=True
                 )
    dops=n.linspace(-10e3,10e3,num=1000)

    fig,((ax00,ax01,ax02),(ax10,ax11,ax12))=plt.subplots(2,3,figsize=(16,9),layout="constrained")
    
    ax00.plot(dops/1e3,S[1](dops),label="$m_i$=31 (amu)")
    ax00.plot(dops/1e3,S[0](dops),label="$m_i$=16 (amu)")
    ax00.set_xlabel("Doppler shift (kHz)")
    ax00.set_title("$f=%1.1f$ MHz $T_i=700$ K $T_e/T_i=1.5$"%(il.radar_freq/1e6))
    ax00.set_ylabel("Normalized power spectral density")
    ax00.legend()
    ax00.set_xlim([-10,10])


    ax01.plot(dops/1e3,S[0](dops),label="$n_e=10^{11}$ (el/m$^3)$")
    ax01.plot(dops/1e3,S[0](dops)*10.0,label="$n_e=10^{12}$ (el/m$^3$)")
    ax01.set_xlabel("Doppler shift (kHz)")
    ax01.set_title("$f=%1.1f$ (MHz) $T_i=700$ (K) $T_e=900$ (K) $m_i=16$ (amu)"%(il.radar_freq/1e6))
    ax01.set_ylabel("Power spectral density")
    ax01.legend()
    ax01.set_xlim([-10,10])


    ne=n.repeat(1e11,2)
    ti=n.array([700/2,700])
    te=n.array([700.0*1.5/2,700*1.5])
    mol_frac=n.array([0, 0])
    vi=n.repeat(0,2)
    
    S=il.getspec(ne=ne,
                 te=te,
                 ti=ti,
                 mol_frac=mol_frac,
                 vi=vi,
                 acf=False,
                 interpf=True
                 )
    

    ax10.plot(dops/1e3,S[0](dops),label="$T_i=700$ (K)")
    ax10.plot(dops/1e3,S[1](dops),label="$T_i=350$ (K)")
    ax10.set_xlabel("Doppler shift (kHz)")
    ax10.set_title("$f=%1.1f$ (MHz) $T_e/T_i=1.5$ K $m_i=16$ (amu)"%(il.radar_freq/1e6))
    ax10.set_ylabel("Power spectral density")
    ax10.legend()
    ax10.set_xlim([-10,10])


    ne=n.repeat(1e11,2)
    ti=n.array([700,700])
    te=n.array([700.0*1.5,700*3.0])
    mol_frac=n.array([0, 0])
    vi=n.repeat(0,2)
    
    S=il.getspec(ne=ne,
                 te=te,
                 ti=ti,
                 mol_frac=mol_frac,
                 vi=vi,
                 acf=False,
                 interpf=True
                 )

    s0=n.sum(S[0](dops))
    s1=n.sum(S[1](dops))
    #(a*s1/s0) = ((1+1.5)/(1+3))
    #a = (s0/s1)*((1+1.5)/(1+3))    
    ax11.plot(dops/1e3,S[0](dops),label="$T_e/T_i=1.5$")
    ax11.plot(dops/1e3,S[1](dops)*(s0/s1)*((1+1.5)/(1+3.0)),label="$T_e/T_i=3$")
    ax11.set_xlabel("Doppler shift (kHz)")
    ax11.set_title("$f=%1.1f$ MHz $T_i=700$ K $m_i=16$ amu"%(il.radar_freq/1e6))
    ax11.set_ylabel("Power spectral density")
    ax11.legend()
    ax11.set_xlim([-10,10])
    

    ne=n.repeat(1e11,2)
    ti=n.array([700,700])
    te=n.array([700.0*1.5,700*1.5])
    mol_frac=n.array([0, 0])
    vi=n.array([0,500.0])

    ds=2*il.radar_freq*500.0/3e8
    
    S=il.getspec(ne=ne,
                 te=te,
                 ti=ti,
                 mol_frac=mol_frac,
                 vi=vi,
                 acf=False,
                 interpf=True
                 )
    dops=n.linspace(-40e3,40e3,num=1000)    
    ax02.plot(dops/1e3,S[0](dops),label="$v_i=0$ (m/s)")
    ax02.plot(dops/1e3,S[1](dops-ds),label="$v_i=500$ (m/s)")
    ax02.set_xlabel("Doppler shift (kHz)")
    ax02.set_title("$f=%1.1f$ MHz $T_i=700$ (K) $T_e/T_i=1.5$ $m_i=16$ (amu)"%(il.radar_freq/1e6))
    ax02.set_ylabel("Power spectral density")
    ax02.legend()
    ax02.set_xlim([-10,10])



    il2=ilint(fname="ion_line_interpolate_16_1_430_00.h5")    
    ne=n.repeat(1e11,2)
    ti=n.array([700,700])
    te=n.array([700.0*1.5,700*1.5])
    mol_frac=n.array([1, 0.5])
    vi=n.array([0,0.0])

    S=il2.getspec(ne=ne,
                  te=te,
                  ti=ti,
                  mol_frac=mol_frac,
                  vi=vi,
                  acf=False,
                  interpf=True
                  )
    
    ax12.plot(dops/1e3,S[0](dops),label="O$^{+}=100\%$, H$^{+}=0 \%$")
    ax12.plot(dops/1e3,S[1](dops),label="O$^{+}=50\%$, H$^{+}=50 \%$")
    ax12.set_xlabel("Doppler shift (kHz)")
    ax12.set_title("$f=%1.1f$ MHz $T_i=700$ (K) $T_e/T_i=1.5$"%(il.radar_freq/1e6))
    ax12.set_ylabel("Power spectral density")
    ax12.legend()
    ax12.set_xlim([-40,40])
    plt.savefig("ppar_ex.png",dpi=150)
    plt.show()
    
    
    
if __name__ == "__main__":
    example_spectra()
#    test_h()
 #   testti()    
  #  testte()
   # testne()
