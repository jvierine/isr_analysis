import numpy as n
import matplotlib.pyplot as plt
import h5py
import scipy.interpolate as sint
import isr_spec
import os
import scipy.constants as sc

class ilint:
    def __init__(self,
                 radar_freq=440.2e6,
                 ion_mass1=32,
                 ion_mass2=16,
                 fname="ion_line_interpolate.h5"):
        """
        Simple two-ion interpolation table for ISR ion-line
        remove the "DC" portion that at low ne becomes more
        significant due to decoupling of the electrons and ions

        only monostatic for now. 
        
        This allows the integrated ion-line power to follow
        the 1/[(1+k^2 D^2)(1+k^2 D^2 + Te/Ti)] given by e.g.,
        Evans (1969).
        """
        fname="ion_line_interpolate_%d_%d_%1.1f.h5"%(ion_mass1,ion_mass2,radar_freq/1e6)
        if os.path.exists(fname) != True:
            # regenerate table
            print("regenerating %d amu and %d amu table for %1.1f MHz. this might take a while."%(ion_mass1,ion_mass2,radar_freq/1e6))
            isr_spec.il_table(mass0=ion_mass1, mass1=ion_mass2, radar_freq=radar_freq)

        h=h5py.File(fname,"r")
        self.S=h["S"][()]   # 5d (ne x first ion fraction x te/ti x ti x frequency)
        print(self.S.shape)

        self.ne=h["ne"][()]
        self.ne0=n.min(self.ne)
        self.ne1=n.max(self.ne)
        self.neN=len(self.ne)
        
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
        self.radar_wavelength=sc.c/self.radar_freq
        self.radar_k=4.0*n.pi/self.radar_wavelength
        
        self.doppler_hz=h["om"][()]/2/n.pi
        self.sr=h["sample_rate"][()]

        self.n_fft=self.S.shape[4]
        self.n_lags=int(self.S.shape[4]/2)
        self.lag=n.arange(self.n_lags)/self.sr

        

        print("Read interpolation table for radar frequency %1.0f MHz\nmass1=%1.1f amu, mass2=%1.1f amu"%(self.radar_freq/1e6,self.mass0,self.mass1))

        # remove the "DC" component due that appears for low ne due to
        # Debye length effects.
        print(self.S.shape)
        
        # remove the "DC" component based on the highest doppler shift power spectral density,
        # only leaving the "ion" contribution to the ion line. numerically
        # this corresponds well to the Buneman (1962) ion line power formula:
        # P_r = ne/[(1+k^2 D^2)(1+k^2 D^2+Te/Ti)]
        bg=self.S[:,:,:,:,-1]
        for i in range(self.S.shape[4]):
            self.S[:,:,:,:,i]=self.S[:,:,:,:,i]-bg

        # normalize everything so that the total integral is unity, so that the above
        # ion-line power scaling can be done after the fact
        for i in range(self.S.shape[0]):
            for j in range(self.S.shape[1]):
                for k in range(self.S.shape[2]):
                    for l in range(self.S.shape[3]):
                        self.S[i,j,k,l,:]=self.S[i,j,k,l,:]/n.sum(self.S[i,j,k,l,:])
            
        
        create_acfs=False
        if "A" in h.keys():
            self.A=h["A"][()]
        else:
            create_acfs=True
        h.close()

        if create_acfs:

            print("Creating acfs")
            self.A=n.zeros([self.S.shape[0],self.S.shape[1],self.S.shape[2],self.S.shape[3],self.n_lags],dtype=n.complex64)
        
            # calculate acfs
            for ne_i in range(self.neN):
                print("%d"%(ne_i))
                for fr_i in range(self.mol_fracsN):
                    for teti_i in range(self.te_ti_ratioN):
                        for ti_i in range(self.tisN):
                            spec=self.S[ne_i,fr_i,teti_i,ti_i,:]
                            # normalize integrated power to 1
                            spec=spec/n.sum(spec)
                            nan_idx=n.where(n.isnan(spec))[0]
                            if len(nan_idx) > 0:
                                print("nans in spectra!")
                                print(self.tis[ti_i])
                                spec[nan_idx]=0.5*(spec[nan_idx-1]+spec[nan_idx+1])
                                continue
                            acf=n.fft.ifft(n.fft.fftshift(spec))
                            self.A[ne_i,fr_i,teti_i,ti_i,:]=acf[0:self.n_lags]
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
                ion1_frac=n.array([1.0]),
                vi=n.array([0.0]),
                acf=False,
                normalize=True,
                interpf=False,
                debug=False):
        """
         ion1_frac: fraction of the first ion 1 = all first ion, 0 = all ion 2.
        """

        # find the two spectra that are closest to the ne we evaluate
        ne_idxl=n.zeros(len(ne),dtype=int)
        ne_idxh=n.zeros(len(ne),dtype=int)
        for i in range(len(ne)):
           a,b= n.sort(n.argsort(n.abs(ne[i]-self.ne))[0:2])
           ne_idxl[i]=a
           ne_idxh[i]=b
        
        te_ti_ratio_idx=(te/ti - self.te_ti_ratio0)/self.dte_ti_ratio
        # edge cases
        te_ti_ratio_idx[te_ti_ratio_idx<=0]=1e-4
        te_ti_ratio_idx[te_ti_ratio_idx>=(self.te_ti_ratioN-1) ]=self.te_ti_ratioN-1-1e-4

        mol_frac_idx=(ion1_frac - self.mol_fracs0)/self.dmol_fracs
        # edge cases
        mol_frac_idx[mol_frac_idx<=0]=1e-4
        mol_frac_idx[mol_frac_idx>=(self.mol_fracsN-1) ]=self.mol_fracsN-1-1e-4

        ti_idx=(ti - self.tis0)/self.dtis
        # edge cases
        ti_idx[ti_idx<=0]=1e-4
        ti_idx[ti_idx>=(self.tisN-1) ]=self.tisN-1-1e-4


        # p_rx(ne,te,ti) = ne/(1+te/ti)
        # p_rx(ne,te,ti)/p_rx(ne0,te,ti) = ne/ne0
        debye_length2=sc.epsilon_0*sc.k*te/(ne*sc.e**2.0)
        # debye length * radar bragg wave number squared
        alpha2=debye_length2 * (4.0*n.pi/self.radar_wavelength)**2.0

        # integrated ion-line power, including debye length effects
        # from Evans (1969), originally from Buneman (1962) eq 2
        # verified by numerical integration of the ion-line with isr_spec.py code,
        # with the "DC" component due to electron contributions
        # subtracted (Buneman eq 1).
        pwr_scaling_factor=ne/((1+alpha2)*(1+alpha2+te/ti))

        # and now linearly interpolate this thing.
        # there are three dimensions, so we need to look at eight corners

        # set lookup table to interpolate
        if acf:
            S=n.zeros([len(ne),self.A.shape[4]],dtype=n.complex64)
            L=self.A
        else:
            S=n.zeros([len(ne),self.S.shape[4]],dtype=n.float32)
            L=self.S            

        # for all parameter quadruplets
        for i in range(len(ne)):
            w00=1.0-(mol_frac_idx[i]-n.floor(mol_frac_idx[i]))  # how close to floor [0,1]
            w01=1.0-(n.ceil(mol_frac_idx[i])-mol_frac_idx[i])   # how close to ceil

            w10=1.0-(te_ti_ratio_idx[i]-n.floor(te_ti_ratio_idx[i]))
            w11=1.0-(n.ceil(te_ti_ratio_idx[i])-te_ti_ratio_idx[i])

            w20=1.0-(ti_idx[i]-n.floor(ti_idx[i]))
            w21=1.0-(n.ceil(ti_idx[i])-ti_idx[i])

            # weights are based on distances to two closest grid points
            print(ne_idxl[i],ne_idxh[i])
            print("%g,%g,%g "%(self.ne[ne_idxl[i]],ne[i],self.ne[ne_idxh[i]]))


            
            w30=(1-((ne[i]-self.ne[ne_idxl[i]])/(self.ne[ne_idxh[i]]-self.ne[ne_idxl[i]])))
            w31=(1-((self.ne[ne_idxh[i]]-ne[i])/(self.ne[ne_idxh[i]]-self.ne[ne_idxl[i]])))
            d0=((ne[i]-self.ne[ne_idxl[i]])/(self.ne[ne_idxh[i]]-self.ne[ne_idxl[i]]))
            d1=((self.ne[ne_idxh[i]]-ne[i])/(self.ne[ne_idxh[i]]-self.ne[ne_idxl[i]]))
            print("%1.2f,%1.2f"%(d0,d1))
            
            print(w30,w31)
            if ne[i]<self.ne[ne_idxl[i]]:
                w30=1.0
                w31=0.0
            elif ne[i]>self.ne[ne_idxh[i]]:
                w30=0.0
                w31=1.0

            if debug:
                print("weights %1.2f,%1.2f-%d,%d %1.2f,%1.2f-%d,%d %1.2f,%1.2f - %d,%d"%(w00,w01,n.floor(mol_frac_idx[i]),n.ceil(mol_frac_idx[i]),
                                                                                         w10,w11,n.floor(te_ti_ratio_idx[i]),n.ceil(te_ti_ratio_idx[i]),
                                                                                         w20,w21,n.floor(ti_idx[i]),n.ceil(ti_idx[i])))
            # 0000
            w0=w30*w00*w10*w20
            S[i,:] += w0*L[ne_idxl[i],int(n.floor(mol_frac_idx[i])), int(n.floor(te_ti_ratio_idx[i])), int(n.floor(ti_idx[i])), :]

            # 0001
            w1=w30*w00*w10*w21
            S[i,:] += w1*L[ne_idxl[i],int(n.floor(mol_frac_idx[i])), int(n.floor(te_ti_ratio_idx[i])), int(n.ceil(ti_idx[i])), :]

            # 0010
            w2=w30*w00*w11*w20
            S[i,:] += w2*L[ne_idxl[i],int(n.floor(mol_frac_idx[i])), int(n.ceil(te_ti_ratio_idx[i])), int(n.floor(ti_idx[i])), :]

            # 0011
            w3=w30*w00*w11*w21
            S[i,:] += w3*L[ne_idxl[i],int(n.floor(mol_frac_idx[i])), int(n.ceil(te_ti_ratio_idx[i])), int(n.ceil(ti_idx[i])), :]
            
            # 0100
            w4=w30*w01*w10*w20
            S[i,:] += w4*L[ne_idxl[i],int(n.ceil(mol_frac_idx[i])), int(n.floor(te_ti_ratio_idx[i])), int(n.floor(ti_idx[i])), :]

            # 0101
            w5=w30*w01*w10*w21
            S[i,:] += w5*L[ne_idxl[i],int(n.ceil(mol_frac_idx[i])), int(n.floor(te_ti_ratio_idx[i])), int(n.ceil(ti_idx[i])), :]

            # 0110
            w6=w30*w01*w11*w20
            S[i,:] += w6*L[ne_idxl[i],int(n.ceil(mol_frac_idx[i])), int(n.ceil(te_ti_ratio_idx[i])), int(n.floor(ti_idx[i])), :]
            
            # 0111
            w7=w30*w01*w11*w21
            S[i,:] += w7*L[ne_idxh[i],int(n.ceil(mol_frac_idx[i])), int(n.ceil(te_ti_ratio_idx[i])), int(n.ceil(ti_idx[i])), :]
            

            # 1000
            w8=w31*w00*w10*w20
            S[i,:] += w8*L[ne_idxh[i],int(n.floor(mol_frac_idx[i])), int(n.floor(te_ti_ratio_idx[i])), int(n.floor(ti_idx[i])), :]

            # 1001
            w9=w31*w00*w10*w21
            S[i,:] += w9*L[ne_idxh[i],int(n.floor(mol_frac_idx[i])), int(n.floor(te_ti_ratio_idx[i])), int(n.ceil(ti_idx[i])), :]

            # 1010
            w10=w31*w00*w11*w20
            S[i,:] += w10*L[ne_idxh[i],int(n.floor(mol_frac_idx[i])), int(n.ceil(te_ti_ratio_idx[i])), int(n.floor(ti_idx[i])), :]

            # 1011
            w11=w31*w00*w11*w21
            S[i,:] += w11*L[ne_idxh[i],int(n.floor(mol_frac_idx[i])), int(n.ceil(te_ti_ratio_idx[i])), int(n.ceil(ti_idx[i])), :]
            
            # 1100
            w12=w31*w01*w10*w20
            S[i,:] += w12*L[ne_idxh[i],int(n.ceil(mol_frac_idx[i])), int(n.floor(te_ti_ratio_idx[i])), int(n.floor(ti_idx[i])), :]

            # 1101
            w13=w31*w01*w10*w21
            S[i,:] += w13*L[ne_idxh[i],int(n.ceil(mol_frac_idx[i])), int(n.floor(te_ti_ratio_idx[i])), int(n.ceil(ti_idx[i])), :]

            # 1110
            w14=w31*w01*w11*w20
            S[i,:] += w14*L[ne_idxh[i],int(n.ceil(mol_frac_idx[i])), int(n.ceil(te_ti_ratio_idx[i])), int(n.floor(ti_idx[i])), :]
            
            # 1111
            w15=w31*w01*w11*w21
            S[i,:] += w15*L[ne_idxh[i],int(n.ceil(mol_frac_idx[i])), int(n.ceil(te_ti_ratio_idx[i])), int(n.ceil(ti_idx[i])), :]
            
            S[i,:]=S[i,:]/(w0+w1+w2+w3+w4+w5+w6+w7+w8+w9+w10+w11+w12+w13+w14+w15)
            
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
    plt.figure(figsize=(14,4))
    plt.subplot(121)
    radar_freq=440.2e6
    te=3000
    ion1_frac=0.999
    il=ilint(radar_freq=radar_freq,ion_mass1=32,ion_mass2=16)
    nes=10**n.linspace(8,12,num=10)#n.array([1e9,1e10,1e11,1e12])
    S=il.getspec(ne=nes,
                 te=n.repeat(te,len(nes)),
                 ti=n.repeat(1000,len(nes)),
                 ion1_frac=n.repeat(ion1_frac,len(nes)),
                 vi=n.repeat(0,len(nes)),
                 normalize=False,
                 acf=False)

    for i in range(S.shape[0]):
        plt.semilogy(il.doppler_hz,S[i,:]/n.max(S[S.shape[0]-1,:]),label="ne=%1.2g"%(nes[i]),alpha=1.0)
    plt.legend()
    plt.ylim([1e-8,2])
    plt.xlim([-15e3,15e3])
    plt.title("il_interp")
    plt.xlabel("Doppler (Hz)")
    plt.ylabel("Power (arb)")
    plt.grid()
    om=2*n.pi*il.doppler_hz
    plt.subplot(122)
    specs=[]
    for ne in nes:
        plpar={"t_i":[1000,1000],
               "t_e":te,
               "m_i":[32,16],
               "n_e":ne,
               "freq":radar_freq,
               "B":45000e-9,
               "alpha":45, 
               "ion_fractions":[ion1_frac,1-ion1_frac]}
        il_spec=isr_spec.isr_spectrum(om,plpar=plpar,n_points=2e3,ni_points=2e3,include_electron_component=True)
        il_spec=il_spec-il_spec[-1]
        specs.append(il_spec)
    for i in range(len(specs)):
        plt.semilogy(il.doppler_hz,specs[i]/n.max(specs[len(specs)-1]),label=r"$n_e=%1.2g$"%(nes[i]))
    plt.ylim([1e-8,2])
    plt.xlim([-15e3,15e3])
    plt.title(r"isr_spectrum")
    plt.legend()
    plt.xlabel("Doppler (Hz)")
    plt.ylabel("Power (arb)")
    plt.grid()
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
                 ion1_frac=mol_frac,
                 vi=vi,
                 acf=True
                 )
    S=il.getspec(ne=ne,
                 te=te,
                 ti=ti,
                 ion1_frac=mol_frac,
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
                 ion1_frac=mol_frac,
                 vi=vi,
                 acf=False
                 )
    A=il.getspec(ne=ne,
                 te=te,
                 ti=ti,
                 ion1_frac=mol_frac,
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
                 ion1_frac=mol_frac,
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
                 ion1_frac=mol_frac,
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
                 ion1_frac=mol_frac,
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
                 ion1_frac=mol_frac,
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
    
    il=ilint()
    
    ne=n.repeat(1e11,2)
    ti=n.repeat(700,2)
    te=n.repeat(700*1.5,2) 
    mol_frac=n.array([0,1])
    vi=n.repeat(0,2)
    
    S=il.getspec(ne=ne,
                 te=te,
                 ti=ti,
                 ion1_frac=mol_frac,
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
                 ion1_frac=mol_frac,
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
                 ion1_frac=mol_frac,
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
                 ion1_frac=mol_frac,
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



    il2=ilint(ion_mass1=16,ion_mass2=1)
    ne=n.repeat(1e11,2)
    ti=n.array([700,700])
    te=n.array([700.0*1.5,700*1.5])
    mol_frac=n.array([1, 0.5])
    vi=n.array([0,0.0])

    S=il2.getspec(ne=ne,
                  te=te,
                  ti=ti,
                  ion1_frac=mol_frac,
                  vi=vi,
                  acf=False,
                  interpf=True
                  )
    
    ax12.plot(dops/1e3,S[0](dops),label=r"O$^{+}=100\%$, H$^{+}=0 \%$")
    ax12.plot(dops/1e3,S[1](dops),label=r"O$^{+}=50\%$, H$^{+}=50 \%$")
    ax12.set_xlabel("Doppler shift (kHz)")
    ax12.set_title(r"$f=%1.1f$ MHz $T_i=700$ (K) $T_e/T_i=1.5$"%(il.radar_freq/1e6))
    ax12.set_ylabel("Power spectral density")
    ax12.legend()
    ax12.set_xlim([-40,40])
    plt.savefig("ppar_ex.png",dpi=150)
    plt.show()
    
    
    
if __name__ == "__main__":
    testne()
    example_spectra()
#    test_h()
 #   testti()    
  #  testte()
  
