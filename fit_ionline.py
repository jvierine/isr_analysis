import h5py
import numpy as n
import matplotlib.pyplot as plt
import scipy.constants as c
import isr_spec.il_interp as il
import scipy.interpolate as si
import scipy.optimize as so
import traceback
import glob

ilf=il.ilint(fname="isr_spec/ion_line_interpolate.h5")

def molecular_ion_fraction(h, h0=100e3, H=20e3):
    """
    Ion-fraction for a Chapman-type exponential scale height behaviour
    h0 = transition height
    H = width of the transition
    """
    return(n.tanh( n.exp(-(h-h0)/H) ))

def mh_molecular_ion_fraction(h):
    """
    John Holt's molecular ion fraction function.
    """
    zz1=-(h-120.0)/40.0
    
    zz1[zz1>50.0]=50.0
    
    H=10.0 - 6.0*n.exp(zz1)

    zz2=-(h-180)/H
    zz2[zz2>50]=50

    fr=1-2.0/(1+n.sqrt(1+8.0*n.exp(zz2)))
    fr[h<120]=1.0
    
    return(fr)

def model_acf(te,ti,mol_frac,vi,lags):
    dop_shift=2*n.pi*2*440.2e6*vi/c.c
    csin=n.exp(1j*dop_shift*lags)
    model=ilf.getspec(ne=n.array([1e11]),
                      te=n.array([te]),
                      ti=n.array([ti]),
                      mol_frac=n.array([mol_frac]),
                      vi=n.array([0.0]),
                      acf=True
                      )[0,:]
    model=model/model[0].real
    acff=si.interp1d(ilf.lag,model)
    return(acff(lags)*csin)


def fit_acf(acf,lags,rgs,var,var_scale=6.0,guess=n.array([n.nan,n.nan,n.nan,n.nan])):

    if n.isnan(guess[0]):
        guess[0]=1.1
    if n.isnan(guess[1]):
        guess[1]=500
    if n.isnan(guess[2]):
        guess[2]=100
    if n.isnan(guess[3]):
        guess[3]=1.05

    print(acf)
    print(n.sqrt(var))
    var=n.real(var)
    # 2x for ground clutter 3x for correlated lags
    std=n.sqrt(var_scale*var)/acf[0].real
    #print(std)
    # normalize all range gates to unity
    nacf=acf/acf[0].real

    mol_fr=molecular_ion_fraction(n.array([rgs*1e3]))[0]

    def ss(x):
        te_ti=x[0]
        ti=x[1]
        vi=x[2]
        zl=x[3]
        
        model=zl*model_acf(te_ti*ti,ti,mol_fr,vi,lags)
        ssq=n.nansum(n.abs(model-nacf)**2.0/std**2.0)
        print(ssq)
        return(ssq)
    
    xhat=so.minimize(ss,guess,method="Nelder-Mead",bounds=((0.99,5),(150,3000),(-1500,1500),(0.99,1.5))).x
    sb=ss(xhat)
    bx=xhat
    guess[2]=-1*guess[2]
    xhat=so.minimize(ss,guess,method="Nelder-Mead",bounds=((0.99,5),(150,3000),(-1500,1500),(0.99,1.5))).x
    st=ss(xhat)
    if st<sb:
        bx=xhat

    xhat=bx
    mlags=n.linspace(0,480e-6,num=100)
    if False:
        #print(xhat)
        model=xhat[3]*model_acf(xhat[0]*xhat[1],xhat[1],0,xhat[2],mlags)
        plt.plot(mlags*1e6,model.real)
        plt.plot(mlags*1e6,model.imag)    
        
        plt.errorbar(lags*1e6,nacf.real,yerr=2*std)
        plt.errorbar(lags*1e6,nacf.imag,yerr=2*std)
        plt.xlabel("Lag ($\mu$s)")
        plt.ylabel("Autocorrelation function R($\\tau)$")
        plt.title("%1.0f km\nT$_e$=%1.0f K T$_i$=%1.0f K v$_i$=%1.0f (m/s) $\\rho=$%1.1f"%(rgs,xhat[0]*xhat[1],xhat[1],xhat[2],mol_fr))
        plt.show()
    return(xhat)


def fit_lpifiles(dirn="lpi_f",n_avg=24,acf_key="acfs_e"):
    fl=glob.glob("%s/lpi*.h5"%(dirn))
    fl.sort()


    h=h5py.File(fl[0],"r")
    acf=n.copy(h[acf_key][()])
    lag=n.copy(h["lags"][()])
    rgs=n.copy(h["rgs_km"][()])    
    acf[:,:]=0.0
    ws=n.copy(acf)
    h.close()
    
    n_ints=int(n.floor(len(fl)/n_avg))

    for fi in range(n_ints):
        acf[:,:]=0.0
        ws[:,:]=0.0
        t0=n.nan
        t1=n.nan
        for ai in range(n_avg):
            h=h5py.File(fl[fi*n_avg+ai],"r")
            a=h[acf_key][()]
            v=h["acfs_var"][()]
            acf+=a/v
            ws+=1/v
            if n.isnan(t0):
                t0=h["i0"][()]
            t1=h["i0"][()]                
            h.close()

        acf=acf/ws
        #plt.pcolormesh(acf.real)
        #plt.colorbar()
        #plt.show()
            
        var=1/ws
        pp=[]

        n_lags=acf.shape[1]
        guess=n.array([n.nan,n.nan,n.nan,n.nan])
        for ri in range(acf.shape[0]):
            try:
                res=fit_acf(acf[ri,0:n_lags],lag[0:n_lags],rgs[ri],var[ri,0:n_lags],guess=guess)
                #print(rgs[ri])
                guess=res
                pp.append(res)
            except:
                pp.append([n.nan,n.nan,n.nan,n.nan])
                traceback.print_exc()
                print("err")

        pp=n.array(pp)
        plt.plot(pp[:,0]*pp[:,1],rgs,".",label="Te")
        plt.plot(pp[:,1],rgs,".",label="Ti")
        plt.plot(pp[:,2]*10,rgs,".",label="vi*10")
        plt.plot(pp[:,3]*100,rgs,".",label="zl*100")
        plt.xlim([-1000,3000])
        plt.legend()
        plt.tight_layout()
        plt.savefig("%s/pp-%d.png"%(dirn,t0))
        plt.close()
        plt.clf()

        ho=h5py.File("%s/pp-%d.h5"%(dirn,t0),"w")
        ho["Te"]=pp[:,0]*pp[:,1]
        ho["Ti"]=pp[:,1]
        ho["vi"]=pp[:,2]
        ho["zl"]=pp[:,3]        
        ho["rgs"]=rgs
        ho["t0"]=t0
        ho["t1"]=t1        
        ho.close()
            
            





if __name__ == "__main__":
    fit_lpifiles(dirn="lpi_f")
#    h=h5py.File("avg.h5","r")
#    acf=h["acf"][()]
#    var=n.real(h["var"][()])
#    lag=h["lag"][()]
#    rgs=h["rgs"][()]

#    pp=[]
#    for ri in range(acf.shape[0]):
#        try:
#            res=fit_acf(acf[ri,:],lag,rgs,var[ri,:])
#            pp.append(res)
#        except:
#            pp.append([n.nan,n.nan,n.nan,n.nan])
#            traceback.print_exc()
#            print("err")

#    pp=n.array(pp)
#    plt.plot(pp[:,0]*pp[:,1],rgs,".",label="Te")
#    plt.plot(pp[:,1],rgs,".",label="Ti")
#    plt.plot(pp[:,2]*10,rgs,".",label="vi*10")
#    plt.plot(pp[:,3]*100,rgs,".",label="zl*100")
#    plt.legend()
#    plt.show()

        
        
#    h.close()
