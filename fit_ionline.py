import h5py
import numpy as n
import matplotlib.pyplot as plt
import scipy.constants as c
import isr_spec.il_interp as il
import scipy.interpolate as si
import scipy.optimize as so
import traceback
import glob
import stuffr
# power meter reading
import tx_power as txp
import os

import optuna

from mpi4py import MPI

comm=MPI.COMM_WORLD
size=comm.Get_size()
rank=comm.Get_rank()

zpm,mpm=txp.get_tx_power_model(dirn="/media/j/fee7388b-a51d-4e10-86e3-5cabb0e1bc13/isr/2023-09-05/usrp-rx0-r_20230905T214448_20230906T040054/metadata/powermeter")    

ilf=il.ilint(fname="isr_spec/ion_line_interpolate.h5")

def molecular_ion_fraction(h, h0=120, H=20):
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
#    model=model/model[0].real
    acff=si.interp1d(ilf.lag,model)
    return(acff(lags)*csin)


def model_gaussian(dw,v,lags):
    dop_shift=2*n.pi*2*440.2e6*v/c.c
    csin=n.exp(1j*dop_shift*lags)
    model = csin*n.exp(- (2*n.pi*2*440.2e6/c.c)*dw*lags)
    return(model)



def fit_gaussian(acf,lags,var,var_scale=4.0,guess=n.array([0,10]),plot=False):
    """
    Fit a gaussian to judge if there is a space object
    """
    var=n.real(var)
    # 2x for ground clutter 2x for correlated lags
    std=n.sqrt(var_scale*var)/n.abs(acf[0].real)

    # normalize all range gates to unity
    scaling_const=acf[0].real
    nacf=acf/scaling_const

    def ss(x):
        
        dop_width=x[0]
        v=x[1]
        zl=x[2]
        
        model=zl*model_gaussian(dop_width,v,lags)
        ssq=n.nansum(n.abs(model-nacf)**2.0/std**2.0)
        return(ssq)

    def ss_optuna(trial):
        
        dop_width=trial.suggest_float("dop_width",10,5e3)
        v=trial.suggest_float("velocity",-3e3,3e3)
        zl=trial.suggest_float("zl",0.8,1.5)
        
        model=zl*model_gaussian(dop_width,v,lags)
        ssq=n.nansum(n.abs(model-nacf)**2.0/std**2.0)
#        print(ssq)
        return(ssq)

    guess=[100,50,1.0]
    xhat=so.minimize(ss,guess,method="Nelder-Mead",bounds=((10,5e3),(-3e3,3e3),(0.8,1.5))).x
    
#    optuna.logging.set_verbosity(optuna.logging.ERROR)
 #   opt=optuna.create_study()
  #  opt.optimize(ss,n_trials=50 )
    
#    optres=opt.best_params
 #   xhat=[optres["dop_width"],optres["velocity"],optres["zl"]]
#    print(xhat)
    # lags for ploting the analytic model, also include real zero-lag

    dx=0.1
    midx=n.where(n.isnan(acf)!=True)[0]
    n_m=len(midx)
    J=n.zeros([2*n_m,3])
    model=xhat[2]*model_gaussian(xhat[0],xhat[1],lags[midx])
    model_dx0=xhat[2]*model_gaussian(xhat[0]+dx,xhat[1],lags[midx])
    model_dx1=xhat[2]*model_gaussian(xhat[0],xhat[1]+dx,lags[midx])
    model_dx2=(xhat[2]+dx)*model_gaussian(xhat[0],xhat[1],lags[midx])    
    J[0:n_m,0]=n.real((model-model_dx0)/dx)
    J[0:n_m,1]=n.real((model-model_dx1)/dx)
    J[0:n_m,2]=n.real((model-model_dx2)/dx)
    J[n_m:(2*n_m),0]=n.imag((model-model_dx0)/dx)
    J[n_m:(2*n_m),1]=n.imag((model-model_dx1)/dx)
    J[n_m:(2*n_m),2]=n.imag((model-model_dx2)/dx)
    
    S=n.zeros([2*n_m,2*n_m])
    for mi in range(n_m):
        S[mi,mi]=1/std[midx[mi]]**2.0
        S[2*mi,2*mi]=1/std[midx[mi]]**2.0        
    Sigma=n.linalg.inv(n.dot(n.dot(n.transpose(J),S),J))
    sigmas=n.sqrt(n.real(n.diag(Sigma)))
    
    
    #mlags=n.linspace(0,480e-6,num=100)
    if plot:
        #print(xhat)
        model=xhat[2]*model_gaussian(xhat[0],xhat[1],lags)
        plt.plot(lags*1e6,model.real)
        plt.plot(lags*1e6,model.imag)    
        
        plt.errorbar(lags*1e6,nacf.real,yerr=2*std)
        plt.errorbar(lags*1e6,nacf.imag,yerr=2*std)
        plt.xlabel("Lag ($\mu$s)")
        plt.ylabel("Autocorrelation function R($\\tau)$")
        plt.title("Gaussian fit\ndoppler width %1.0f $\pm$ %1.0f m/s vel %1.0f $\pm$ %1.0f m/s"%(xhat[0],sigmas[0],xhat[1],sigmas[1]))
        plt.show()
    
    return(xhat,sigmas)
    
    
    
    

def fit_acf(acf,lags,rgs,var,var_scale=2.0,guess=n.array([n.nan,n.nan,n.nan,n.nan]),plot=False, scaling_constant=1e4):

#    acf=acf/scaling_constant
 #   var=var/(scaling_constant**2.0)
    
    if n.isnan(guess[0]):
        guess[0]=1.1
    if n.isnan(guess[1]):
        guess[1]=500
    if n.isnan(guess[2]):
        guess[2]=100

#    print(acf)
 #   print(n.sqrt(var))
    var=n.real(var)
    # 2x for ground clutter 3x for correlated lags
    std=n.sqrt(var_scale*var)#/n.abs(acf[0].real)
    #print(std)
    # normalize all range gates to unity
    zl_guess=1.1*n.abs(acf[0].real)

    # override zero-lag guess
    guess[3]=1.1*zl_guess

    mol_fr=mh_molecular_ion_fraction(n.array([rgs]))[0]

    def ss(x):
        te_ti=x[0]
        ti=x[1]
        vi=x[2]
        zl=x[3]

        model=zl*model_acf(te_ti*ti,ti,mol_fr,vi,lags)

        ssq=n.nansum(n.abs(model-acf)**2.0/std**2.0)
        return(ssq)

    xhat=so.minimize(ss,guess,method="Nelder-Mead",bounds=((0.99,5),(150,4000),(-1500,1500),(0.3*zl_guess,3*zl_guess))).x
    sb=ss(xhat)
    bx=xhat
    guess[2]=-1*guess[2]
    xhat=so.minimize(ss,guess,method="Nelder-Mead",bounds=((0.99,5),(150,4000),(-1500,1500),(0.3*zl_guess,3*zl_guess))).x
    st=ss(xhat)
    if st<sb:
        bx=xhat
    xhat=so.minimize(ss,[1.1,500,100,1.05*zl_guess],method="Nelder-Mead",bounds=((0.99,5),(150,4000),(-1500,1500),(0.3*zl_guess,3*zl_guess))).x
    st=ss(xhat)
    if st<sb:
        bx=xhat
    xhat=bx
        

    dx0=0.1
    dx1=30
    dx2=100.0
    dx3=zl_guess*0.01
    midx=n.where(n.isnan(acf)!=True)[0]
    n_m=len(midx)
    J=n.zeros([2*n_m,4])
    model=xhat[3]*model_acf(xhat[0]*xhat[1],xhat[1],mol_fr,xhat[2],lags[midx])

    # is residuals are worse than twice the standard deviation of errors,
    # make the error std estimate worse, as there is probably something
    # unexplained the the measurement that is making things worse than we think
    # this tends to happen on the top-side in the presence of rfi
    sigma_resid=n.sqrt(n.mean(n.abs(model-acf[midx])**2.0))
    std[sigma_resid>2*std]=sigma_resid
    
    model_dx0=xhat[3]*model_acf((xhat[0]+dx0)*xhat[1],xhat[1],mol_fr,xhat[2],lags[midx])
    model_dx1=xhat[3]*model_acf(xhat[0]*(xhat[1]+dx1),(xhat[1]+dx1),mol_fr,xhat[2],lags[midx])
    model_dx2=xhat[3]*model_acf(xhat[0]*xhat[1],xhat[1],mol_fr,(xhat[2]+dx2),lags[midx])
    model_dx3=(xhat[3]+dx3)*model_acf(xhat[0]*xhat[1],xhat[1],mol_fr,xhat[2],lags[midx])
    J[0:n_m,0]=n.real((model-model_dx0)/dx0)
    J[0:n_m,1]=n.real((model-model_dx1)/dx1)
    J[0:n_m,2]=n.real((model-model_dx2)/dx2)
    J[0:n_m,3]=n.real((model-model_dx3)/dx3)    
    J[n_m:(2*n_m),0]=n.imag((model-model_dx0)/dx0)
    J[n_m:(2*n_m),1]=n.imag((model-model_dx1)/dx1)
    J[n_m:(2*n_m),2]=n.imag((model-model_dx2)/dx2)
    J[n_m:(2*n_m),3]=n.imag((model-model_dx2)/dx3)    
    
    S=n.zeros([2*n_m,2*n_m])
    for mi in range(n_m):
        S[mi,mi]=0.5/std[midx[mi]]**2.0
        S[2*mi,2*mi]=0.5/std[midx[mi]]**2.0
        
    Sigma=n.linalg.inv(n.dot(n.dot(n.transpose(J),S),J))
    sigmas=n.sqrt(n.real(n.diag(Sigma)))
    
    # lags for ploting the analytic model, also include real zero-lag
    mlags=n.linspace(0,480e-6,num=100)
    model=xhat[3]*model_acf(xhat[0]*xhat[1],xhat[1],mol_fr,xhat[2],lags)
    model2=xhat[3]*model_acf(xhat[0]*xhat[1],xhat[1],mol_fr,xhat[2],mlags)
    if plot:
        #print(xhat)
        
        plt.plot(mlags*1e6,model2.real)
        plt.plot(mlags*1e6,model2.imag)    
        
        plt.errorbar(lags*1e6,acf.real,yerr=2*std)
        plt.errorbar(lags*1e6,acf.imag,yerr=2*std)
        plt.ylim([-1.0*zl_guess,2*zl_guess])
        plt.xlabel("Lag ($\mu$s)")
        plt.ylabel("Autocorrelation function R($\\tau)$")
        plt.title("%1.0f km\nT$_e$=%1.0f K T$_i$=%1.0f K v$_i$=%1.0f$\pm$%1.0f (m/s) $\\rho=$%1.1f"%(rgs,xhat[0]*xhat[1],xhat[1],xhat[2],sigmas[2],mol_fr))
        plt.show()
    return(xhat,model,sigmas)


# the scaling constant ensures matrix algebra can be done without problems with numerical accuracy
def fit_lpifiles(dirn="lpi_f",n_avg=120,acf_key="acfs_e",plot=False,
                 scaling_constant=1e5,
                 reanalyze=False,
                 first_lag=0):
    fl=glob.glob("%s/lpi*.h5"%(dirn))
    fl.sort()

    h=h5py.File(fl[0],"r")

    acf=n.copy(h[acf_key][()])
    lag=n.copy(h["lags"][()])
    rgs=n.copy(h["rgs_km"][()])
    h.close()
    n_ints=int(n.floor(len(fl)/n_avg))

#    first_lag=1

    # above this, don't use ground clutter removal
    # use removal below this
    rg300=n.where(rgs>300)[0][0]

    n_rg=len(rgs)
    n_l=len(lag)
    acf[:,:]=0.0
    ws=n.copy(acf)    

    for fi in range(rank,n_ints,size):
        
        acf[:,:]=0.0
        ws[:,:]=0.0
        t0=n.nan
        t1=n.nan
        
        acfs=n.zeros([n_avg,n_rg,n_l],dtype=n.complex64)
        wgts=n.zeros([n_avg,n_rg,n_l],dtype=n.float64)


        
        for ai in range(n_avg):
            h=h5py.File(fl[fi*n_avg+ai],"r")
            a=h["acfs_e"][()]
            # ground clutter removed and scaled
            # in amplitude to correct for the pulse to pulse subtraction
            a_g=h["acfs_g"][()]/2
            a[0:rg300,:]=a_g[0:rg300,:]
            
            v=h["acfs_var"][()]

            debris=n.zeros(n_rg,dtype=bool)
            debris[:]=False

            # fit gaussian model to determine if there are space objects at some range gates
            if plot:
                plt.pcolormesh(lag,rgs,a.real)
                plt.colorbar()
                plt.show()
            ao=n.copy(a)
            vo=n.copy(v)            
            for ri in range(acf.shape[0]):
                if n.sum(n.isnan(a[ri,:]))/len(lag) < 0.5 and rgs[ri] > 250.0:
                    #print(rgs[ri])
                    gres,gsigma=fit_gaussian(ao[ri,:],lag,n.real(n.abs(vo[ri,:])),plot=False)
                    #print("possible debris at %1.0f km dopp width %1.0f+/-%1.0f (m/s)"%(rgs[ri],gres[0],gsigma[0]))                    
                    #                    print(gres)

                    
                    #                   print(gsigma)
                    #if gres[0]<300.0 and gsigma[0]<200:                    
                    
                    if gres[0]<300.0 and gsigma[0]<200:
                        print("debris at %1.0f km dopp width %1.0f+/-%1.0f (m/s)"%(rgs[ri],gres[0],gsigma[0]))
                        # make neighbouring range gates contaminated
                        debris[ri]=True
                        a[ri,:]=n.nan
                        v[ri,:]=n.nan
                        debris[ri-1]=True
                        a[ri-1,:]=n.nan
                        v[ri-1,:]=n.nan
                        if ri+1 < n_rg:
                            debris[ri+1]=True
                            a[ri+1,:]=n.nan
                            v[ri+1,:]=n.nan
                            
            
            acfs[ai,:,:]=a/v
            wgts[ai,:,:]=1/v
#            ws+=1/v
            if n.isnan(t0):
                t0=h["i0"][()]
                print("Starting new integration period %s"%(stuffr.unix2datestr(t0)))
            t1=h["i0"][()]                
            h.close()
        if os.path.exists("%s/pp-%d.h5"%(dirn,t0)) and reanalyze==False:
            print("Already exists. Skipping")
            continue


        var=1/n.nansum(wgts,axis=0)
        acf=n.nansum(acfs,axis=0)/n.nansum(wgts,axis=0)

        avg_acf=n.copy(acf)
        avg_var=n.copy(var)        
        range_avg=1
        # do some range averaging
        for ri in range(acf.shape[0]):
            avg_acf[ri,:]=n.nanmean(acf[n.max((0,(ri-range_avg))):n.min((acf.shape[0],(ri+range_avg))),:],axis=0)
            avg_var[ri,:]=1/(n.nansum(1/var[(ri-range_avg):n.min((acf.shape[0],(ri+range_avg))),:],axis=0))
                
        acf=avg_acf
        var=avg_var        

        acf0=n.copy(acf)
        for ri in range(acf0.shape[0]):
            acf0[ri,:]=acf0[ri,:]/acf0[ri,first_lag].real
            
        if plot:
            plt.pcolormesh(acf0.real,vmin=-0.1,vmax=1.1)
            plt.colorbar()
            plt.show()
            
            #        var=1/ws
        pp=[]
        dpp=[]        
        model_acfs=n.copy(acf0)
        model_acfs[:,:]=n.nan
        
        n_lags=acf.shape[1]
        guess=n.array([n.nan,n.nan,n.nan,n.nan])
        for ri in range(acf.shape[0]):
            try:
                if n.sum(n.isnan(acf[ri,first_lag:n_lags]))/(n_lags-first_lag) < 0.8:
                    res,model_acf,dres=fit_acf(acf[ri,first_lag:n_lags],lag[first_lag:n_lags],rgs[ri],var[ri,first_lag:n_lags],guess=guess,plot=plot ,scaling_constant=scaling_constant)
                    model_acfs[ri,first_lag:n_lags]=model_acf/model_acf[0].real
                    guess=res
#                    print(dres)
                else:
                    res=n.array([n.nan,n.nan,n.nan,n.nan])
                    dres=n.array([n.nan,n.nan,n.nan,n.nan])                    
                # ne raw
                res_out=n.copy(res)
                dres_out=n.copy(dres)                
                
                # get electron density from echo power (acf zero lag)
                # acf(0)=(1/magic const)*ne*Ptx/(1+te/ti)
                # ne = magic_const*acf(0)*(1+te/ti)*r**2.0/Ptx 
                # res[3] is zero-lag power (arb scale)
                ne_const=(1+res[0])*rgs[ri]**2.0/zpm(0.5*(t0+t1))
                res_out[3]=res[3]*ne_const
                dres_out[3]=dres_out[3]*ne_const
                pp.append(res_out)
                dpp.append(dres_out)                
            except:
                pp.append([n.nan,n.nan,n.nan,n.nan])
                dpp.append([n.nan,n.nan,n.nan,n.nan])
                traceback.print_exc()
                print("error caught. marching onwards.")
        plt.subplot(121)
        plt.pcolormesh(lag[first_lag:n_lags]*1e6,rgs,model_acfs[:,first_lag:n_lags].real,vmin=-0.2,vmax=1.1)
        plt.title("Best fit")        
        plt.xlabel("Lag ($\mu$s)")
        plt.ylabel("Range (km)")        
        plt.colorbar()
        plt.subplot(122)
        plt.pcolormesh(lag[first_lag:n_lags]*1e6,rgs,acf0[:,first_lag:n_lags].real,vmin=-0.2,vmax=1.1)
        plt.title("Measurement")
        plt.xlabel("Lag ($\mu$s)")
        plt.ylabel("Range (km)")
        plt.colorbar()
        plt.tight_layout()
        plt.savefig("%s/pp_fit_%d.png"%(dirn,t0))
        plt.close()
        plt.clf()

        
        pp=n.array(pp)
        dpp=n.array(dpp)
        plt.plot(pp[:,0]*pp[:,1],rgs,".",label="Te")
        plt.plot(pp[:,1],rgs,".",label="Ti")
        plt.plot(pp[:,2]*10,rgs,".",label="vi*10")
        plt.plot(n.log10(pp[:,3]),rgs,".",label="ne raw")
        plt.xlim([-1000,5000])
        plt.legend()
        plt.tight_layout()
        plt.savefig("%s/pp-%d.png"%(dirn,t0))
        plt.close()
        plt.clf()

        ho=h5py.File("%s/pp-%d.h5"%(dirn,t0),"w")
        ho["Te"]=pp[:,0]*pp[:,1]
        ho["Ti"]=pp[:,1]
        ho["vi"]=pp[:,2]
        ho["ne"]=pp[:,3]
        
        ho["dTe/Ti"]=dpp[:,0]  # tbd fix this
        ho["dTi"]=dpp[:,1]
        ho["dvi"]=dpp[:,2]
        ho["dne"]=dpp[:,3]          # tbd fix this
        
        ho["rgs"]=rgs
        ho["t0"]=t0
        ho["t1"]=t1        
        ho.close()
            


if __name__ == "__main__":
    import sys
    fit_lpifiles(dirn=sys.argv[1],n_avg=12,plot=bool(int(sys.argv[2])),first_lag=1,reanalyze=True)   
#    fit_lpifiles(dirn="lpi_f2",n_avg=12,plot=False,first_lag=1)
#    fit_lpifiles(dirn="lpi_e",n_avg=12,plot=False,first_lag=1,reanalyze=True)   
#    fit_lpifiles(dirn="lpi_ts",n_avg=1,plot=False)        

 #   fit_lpifiles(dirn="lpi_e",n_avg=60)

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
