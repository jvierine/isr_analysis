import os
# mpi does paralelization, both multithread matrix operations
# just one per process to avoid cache trashing.
os.system("export OMP_NUM_THREADS=1")
os.environ["OMP_NUM_THREADS"] = "1"

import h5py
import numpy as n
import matplotlib.pyplot as plt
import scipy.constants as c
import il_interp as il
import scipy.interpolate as si
import scipy.optimize as so
import traceback
import glob
import stuffr
# power meter reading
import tx_power as txp
import os
import millstone_radar_state as mrs

import jcoord

import isr_spec

from mpi4py import MPI

comm=MPI.COMM_WORLD
size=comm.Get_size()
rank=comm.Get_rank()


#il_table(mass0=32.0, mass1=16.0, radar_freq=430.0e6)
#il_table(mass0=16.0, mass1=1.0, radar_freq=430.0e6)    


# TBD: change names of tables to isr_spec/ion_line_interpolate_31_16.h5 and
# isr_spec/ion_line_interpolate_16_1.h5, as the new merge_tables.py and create_interp_tables.py use this convention now
radar_freq=440.2e6/1e6
table_o2_o="ion_line_interpolate_%d_%d_%1.1f.h5"%(32,16,radar_freq)
table_o_h="ion_line_interpolate_%d_%d_%1.1f.h5"%(16,1,radar_freq)

if os.path.exists(table_o2_o) != True:
    # regenerate table
    print("regenerating 32 amu and 16 amu table for %1.1f MHz. this might take a while."%(radar_freq))
    isr_spec.il_table(mass0=32.0, mass1=16.0, radar_freq=radar_freq*1e6)
if os.path.exists(table_o_h) != True:
    # regenerate table
    print("regenerating 16 amu and 1 amu table for %1.1f MHz this might take a while."%(radar_freq))
    isr_spec.il_table(mass0=16.0, mass1=1.0, radar_freq=radar_freq*1e6)

ilf=il.ilint(fname="ion_line_interpolate_32_16_440.2.h5")
ilf_ho=il.ilint(fname="ion_line_interpolate_16_1_440.2.h5")

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

def model_acf(te,ti,heavy_ion_frac,vi,lags,hplus=False):
    """
    heavy_ion_frac is the fraction of the heavier ion 
    (O_2+/N_2+ with O+ or O+ with H+)
    """
    dop_shift=2*n.pi*2*440.2e6*vi/c.c
    csin=n.exp(1j*dop_shift*lags)
    
    if hplus == False:
        model=ilf.getspec(ne=n.array([1e11]),
                          te=n.array([te]),
                          ti=n.array([ti]),
                          mol_frac=n.array([heavy_ion_frac]),
                          vi=n.array([0.0]),
                          acf=True
                          )[0,:]
        acff=si.interp1d(ilf.lag,model)
        
        return(acff(lags)*csin)
        
    else:
        model=ilf_ho.getspec(ne=n.array([1e11]),
                          te=n.array([te]),
                          ti=n.array([ti]),
                          mol_frac=n.array([heavy_ion_frac]),
                          vi=n.array([0.0]),
                          acf=True
                          )[0,:]
        acff=si.interp1d(ilf_ho.lag,model)
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
        plt.xlabel(r"Lag ($\mu$s)")
        plt.ylabel(r"Autocorrelation function R($\tau)$")
        plt.title(r"Gaussian fit\ndoppler width %1.0f $\pm$ %1.0f m/s vel %1.0f $\pm$ %1.0f m/s"%(xhat[0],sigmas[0],xhat[1],sigmas[1]))
        plt.show()
    
    return(xhat,sigmas)
    
    
    
    

def fit_acf(acf,
            lags,
            rgs,
            var,
            var_scale=2.0,
            guess=n.array([n.nan,n.nan,n.nan,n.nan]),
            plot=False,
            use_optuna=False,
            scaling_constant=1e4):

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
    # estimate zero-lag
    zl_guess=1.1*n.abs(acf[0].real)

    # override zero-lag guess
    guess[3]=1.1*zl_guess

    mol_fr=mh_molecular_ion_fraction(n.array([rgs]))[0]


    xhat=n.zeros(4)
    
    if use_optuna:
        def ss_optuna(trial):
            te_ti=trial.suggest_float("te_ti",1,3)
            ti=trial.suggest_float("ti",150,3000)
            vi=trial.suggest_float("vi",-1500,1500)
            zl=trial.suggest_float("zl",0.3*zl_guess,3*zl_guess)

            model=zl*model_acf(te_ti*ti,ti,mol_fr,vi,lags)
            ssq=n.nansum(n.abs(model-acf)**2.0/std**2.0)

            return(ssq)

        optuna.logging.set_verbosity(optuna.logging.ERROR)
        opt=optuna.create_study()
        opt.optimize(ss_optuna,n_trials=100 )

        optres=opt.best_params
        xhat=n.array([optres["te_ti"],optres["ti"],optres["vi"],optres["zl"]])
#        print("optuna found")
 #       print(xhat)

    else:
        def ss(x):
            te_ti=x[0]
            ti=x[1]
            vi=x[2]
            zl=x[3]
            
            model=zl*model_acf(te_ti*ti,ti,mol_fr,vi,lags)
            ssq=n.nansum(n.abs(model-acf)**2.0/std**2.0)
            return(ssq)

        xhat=so.minimize(ss,guess,method="Nelder-Mead",bounds=((0.99,3),(150,3000),(-1500,1500),(0.3*zl_guess,3*zl_guess))).x
        sb=ss(xhat)
        bx=xhat
        guess[2]=-1*guess[2]
        xhat=so.minimize(ss,guess,method="Nelder-Mead",bounds=((0.99,3),(150,3000),(-1500,1500),(0.3*zl_guess,3*zl_guess))).x
        st=ss(xhat)
        if st<sb:
            bx=xhat
        xhat=so.minimize(ss,[1.1,500,100,1.05*zl_guess],method="Nelder-Mead",bounds=((0.99,3),(150,3000),(-1500,1500),(0.3*zl_guess,3*zl_guess))).x
        st=ss(xhat)
        if st<sb:
            bx=xhat

        xhat=bx

#        print("fmin found")
 #       print(xhat)
        

    dx0=0.05*xhat[0]
    dx1=0.05*xhat[1]
    dx2=0.05*n.abs(xhat[2])
    dx3=0.05*xhat[3]
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
    J[0:n_m,0]=n.real((model_dx0-model)/dx0)
    J[0:n_m,1]=n.real((model_dx1-model)/dx1)
    J[0:n_m,2]=n.real((model_dx2-model)/dx2)
    J[0:n_m,3]=n.real((model_dx3-model)/dx3)    
    J[n_m:(2*n_m),0]=n.imag((model_dx0-model)/dx0)
    J[n_m:(2*n_m),1]=n.imag((model_dx1-model)/dx1)
    J[n_m:(2*n_m),2]=n.imag((model_dx2-model)/dx2)
    J[n_m:(2*n_m),3]=n.imag((model_dx2-model)/dx3)    
    
    S=n.zeros([2*n_m,2*n_m])
    for mi in range(n_m):
        S[mi,mi]=1/std[midx[mi]]**2.0
        S[2*mi,2*mi]=1/std[midx[mi]]**2.0
        
    Sigma=n.linalg.inv(n.dot(n.dot(n.transpose(J),S),J))
    sigmas=n.sqrt(n.real(n.diag(Sigma)))
    
    # lags for ploting the analytic model, also include real zero-lag
    mlags=n.linspace(0,480e-6,num=100)
    model=xhat[3]*model_acf(xhat[0]*xhat[1],xhat[1],mol_fr,xhat[2],lags)
    model2=xhat[3]*model_acf(xhat[0]*xhat[1],xhat[1],mol_fr,xhat[2],mlags) 
    if plot:
        print(bx)
        
        plt.plot(mlags*1e6,model2.real)
        plt.plot(mlags*1e6,model2.imag)    
        
        plt.errorbar(lags*1e6,acf.real,yerr=2*std)
        plt.errorbar(lags*1e6,acf.imag,yerr=2*std)
        plt.ylim([-1.0*zl_guess,2*zl_guess])
        plt.xlabel(r"Lag ($\mu$s)")
        plt.ylabel(r"Autocorrelation function R($\tau)$")
        plt.title(r"%1.0f km\nT$_e$=%1.0f K T$_i$=%1.0f K v$_i$=%1.0f$\pm$%1.0f (m/s) $\rho=$%1.1f"%(rgs,xhat[0]*xhat[1],xhat[1],xhat[2],sigmas[2],mol_fr))
        plt.show()
    return(xhat,model,sigmas)


def fit_acf_ts(acf,
               lags,
               rgs,
               var,
               var_scale=2.0,
               guess=n.array([n.nan,n.nan,n.nan,n.nan,n.nan]), # te/ti, ti, vi, zl, heavy_ion_frac
               plot=False,
               scaling_constant=1e4):
    """ 
    topside 
    """
    
    if n.isnan(guess[0]):
        guess[0]=1.5
    if n.isnan(guess[1]):
        guess[1]=1200
    if n.isnan(guess[2]):
        guess[2]=42.0
#    if n.isnan(guess[4]):
 #       guess[4]=1.0  # 100% O+

    var=n.real(var)
    std=n.sqrt(var_scale*var)

    # estimate zero-lag
    zl_guess=n.abs(acf[0].real)

    # override zero-lag guess
    guess[3]=1.1*zl_guess

    def ss(x):
        te_ti=x[0]
        ti=x[1]
        vi=x[2]
        zl=x[3]
        ofrac=1-1/x[4]        
        
        model=zl*model_acf(te_ti*ti,ti,ofrac,vi,lags,hplus=True)
        
        ssq=n.nansum(n.abs(model-acf)**2.0/std**2.0)
 #       print(ssq)
#        print(x)
  #      print(zl_guess)
        return(ssq)
    guess2=n.zeros(5)
    guess2[0:4]=guess
    guess2[4]=1e3
    xhat=so.minimize(ss,guess2,method="Nelder-Mead",bounds=((0.99,3),(150,3000),(-1500,1500),(0.1*zl_guess,5*zl_guess),(2,10000))).x
    sb=ss(xhat)
    bx=xhat
    guess[2]=-1*guess[2]
    xhat=so.minimize(ss,guess2,method="Nelder-Mead",bounds=((0.99,3),(150,3000),(-1500,1500),(0.1*zl_guess,5*zl_guess),(2,10000))).x
    st=ss(xhat)
    if st<sb:
        bx=xhat
    xhat=so.minimize(ss,[1.1,1500,13,1.05*zl_guess,800],method="Nelder-Mead",bounds=((0.99,3),(150,3000),(-1500,1500),(0.1*zl_guess,5*zl_guess),(2,10000))).x
    st=ss(xhat)
    if st<sb:
        bx=xhat

    xhat=bx
        
    dx0=0.05*xhat[0]
    dx1=0.05*xhat[1]
    dx2=0.05*n.abs(xhat[2])
    dx3=0.05*xhat[3]
    midx=n.where(n.isnan(acf)!=True)[0]
    n_m=len(midx)
    J=n.zeros([2*n_m,4])
    
    ofrac=xhat[4]
    
    model=xhat[3]*model_acf(xhat[0]*xhat[1],xhat[1],ofrac,xhat[2],lags[midx],hplus=True)

    # is residuals are worse than twice the standard deviation of errors,
    # make the error std estimate worse, as there is probably something
    # unexplained the the measurement that is making things worse than we think
    # this tends to happen on the top-side in the presence of rfi
    sigma_resid=n.sqrt(n.mean(n.abs(model-acf[midx])**2.0))
    std[sigma_resid>2*std]=sigma_resid
    
    model_dx0=xhat[3]*model_acf((xhat[0]+dx0)*xhat[1],xhat[1],ofrac,xhat[2],lags[midx],hplus=True)
    model_dx1=xhat[3]*model_acf(xhat[0]*(xhat[1]+dx1),(xhat[1]+dx1),ofrac,xhat[2],lags[midx],hplus=True)
    model_dx2=xhat[3]*model_acf(xhat[0]*xhat[1],xhat[1],ofrac,(xhat[2]+dx2),lags[midx],hplus=True)
    model_dx3=(xhat[3]+dx3)*model_acf(xhat[0]*xhat[1],xhat[1],ofrac,xhat[2],lags[midx],hplus=True)
    J[0:n_m,0]=n.real((model_dx0-model)/dx0)
    J[0:n_m,1]=n.real((model_dx1-model)/dx1)
    J[0:n_m,2]=n.real((model_dx2-model)/dx2)
    J[0:n_m,3]=n.real((model_dx3-model)/dx3)    
    J[n_m:(2*n_m),0]=n.imag((model_dx0-model)/dx0)
    J[n_m:(2*n_m),1]=n.imag((model_dx1-model)/dx1)
    J[n_m:(2*n_m),2]=n.imag((model_dx2-model)/dx2)
    J[n_m:(2*n_m),3]=n.imag((model_dx2-model)/dx3)    
    
    S=n.zeros([2*n_m,2*n_m])
    for mi in range(n_m):
        S[mi,mi]=1/std[midx[mi]]**2.0
        S[2*mi,2*mi]=1/std[midx[mi]]**2.0
        
    Sigma=n.linalg.inv(n.dot(n.dot(n.transpose(J),S),J))
    sigmas=n.sqrt(n.real(n.diag(Sigma)))
    
    # lags for ploting the analytic model, also include real zero-lag
    mlags=n.linspace(0,480e-6,num=100)
    model=xhat[3]*model_acf(xhat[0]*xhat[1],xhat[1],ofrac,xhat[2],lags,hplus=True)
    model2=xhat[3]*model_acf(xhat[0]*xhat[1],xhat[1],ofrac,xhat[2],mlags,hplus=True) 
    if plot:
        print(bx)
        
        plt.plot(mlags*1e6,model2.real)
        plt.plot(mlags*1e6,model2.imag)    
        
        plt.errorbar(lags*1e6,acf.real,yerr=2*std)
        plt.errorbar(lags*1e6,acf.imag,yerr=2*std)
        plt.ylim([-1.0*zl_guess,2*zl_guess])
        plt.xlabel(r"Lag ($\mu$s)")
        plt.ylabel(r"Autocorrelation function R($\tau)$")
        plt.title(r"%1.0f km\nT$_e$=%1.0f K T$_i$=%1.0f K v$_i$=%1.0f$\pm$%1.0f (m/s) $\rho=$%1.1f"%(rgs,xhat[0]*xhat[1],xhat[1],xhat[2],sigmas[2],ofrac))
        plt.show()
    return(xhat[0:4],model,sigmas)


# the scaling constant ensures matrix algebra can be done without problems with numerical accuracy
def fit_lpifiles(dirn="lpi_f",
                 channel="misa-l",
                 postfix="_30",
                 acf_key="acfs_e",
                 plot=False,
                 scaling_constant=1e5,
                 reanalyze=False,
                 gc_cancel_all_ranges=False,
                 minimum_tx_pwr=400e3,
                 range_limits=n.array([0,300,700,1500]),  # range averaging boundaries in km # probably should be changed based on elevation angle...
                 range_avg=n.array([0,  1,  2]),          # range averaging window in range gates symmetric windows are used (ri-window):(ri+window) with range**2.0 weighting
                 max_dt=300,
                 first_lag=0):

#    if zpm == None:
 #       def zpm(t):
  #          return(1.2e6)





    use_misa=False
    if channel=="misa-l":
        use_misa=True

    # defaults to zenith pointing
    def azf(t):
        return(0)
    def elf(t):
        return(90)
    
    if use_misa:
        azf,elf,azelb=mrs.get_misa_az_el_model(dirn="%s/metadata/antenna_control_metadata"%(dirn))        
    output_dir="%s/lpi%s/%s"%(dirn,postfix,channel)
    os.system("mkdir -p %s"%(output_dir))
    fl=glob.glob("%s/lpi*.h5"%(output_dir))
    fl.sort()

    h=h5py.File(fl[0],"r")

    acf=n.copy(h[acf_key][()])
    lag=n.copy(h["lags"][()])
    rgs=n.copy(h["rgs_km"][()])
    t0 = h["i0"][()]
    h.close()

    #n_ints=int(n.floor(len(fl)/n_avg))

    # look for sets of files that lie within integration period
    int_files=[]
    t_start=t0
    this_fl=[]
    for fi,f in enumerate(fl):
        h=h5py.File(f,"r")
        ft0=h["i0"][()]

        if (ft0 >= t_start) and (ft0 < (t_start+max_dt)):
            this_fl.append(f)
        else:
            int_files.append(this_fl)
#            print(this_fl)
            t_start=ft0
            this_fl=[]

            this_fl.append(f)
        h.close()
#    print(len(int_files))
#    print(int_files)
    # above this, don't use ground clutter removal
    # use removal below this
    rg_clutter_rem_cutoff=n.where(rgs>300)[0][0]

    n_rg=len(rgs)
    n_l=len(lag)
    acf[:,:]=0.0
    ws=n.copy(acf)

    n_ints=len(int_files)
    for int_idx in range(rank,n_ints,size):
        
        acf[:,:]=0.0
        ws[:,:]=0.0
        t0=n.nan
        t1=n.nan

        tsys=0.0
        ptx=0.0
        
        n_avg=len(int_files[int_idx])
        acfs=n.zeros([n_avg,n_rg,n_l],dtype=n.complex64)
        wgts=n.zeros([n_avg,n_rg,n_l],dtype=n.float64)

        space_object_times=[]
        space_object_rgs=[]
        space_object_count=n.zeros(n_rg,dtype=int)

        n_avged=0

        mean_az=0.0
        mean_el=0.0        

        h=h5py.File(int_files[int_idx][0],"r")
        t0=h["i0"][()]
        print("starting integration period at %s"%(stuffr.unix2datestr(t0)))
        h.close()
        h=h5py.File(int_files[int_idx][-1],"r")
        # tbd: this should be the timestamp of the end of the file, not the beginning. should be added to output of outlier_lpi.py files.
        t1=h["i0"][()]
        h.close()
        if os.path.exists("%s/pp-%d.h5"%(output_dir,t0)) and reanalyze==False:
            print("already exists")
            continue
            
        ampgains=[]
        for ai in range(n_avg):
            h=h5py.File(int_files[int_idx][ai],"r")
            n_avged+=1
            ampgains.append(h["alpha"][()])            
            ptx+=h["P_tx"][()]
            a=h["acfs_e"][()]

            
            mean_az+=azf(h["i0"][()])
            mean_el+=elf(h["i0"][()])
            print("elevation %1.2f"%(elf(h["i0"][()])))
            
            if gc_cancel_all_ranges:
                # factor of 2 due to summing two echoes together.
                # the factor of 2 is verified by performing a magic constant estimation on two independent fits:
                # one with ground clutter cancel on all heights, and one with no ground clutter cancellation on any height
                a=h["acfs_g"][()]/2.0
            else:
                # only populate lower altitude bins
                a[0:rg_clutter_rem_cutoff,:]=h["acfs_g"][()][0:rg_clutter_rem_cutoff,:]/2.0

            # ground clutter removed and scaled
            # in amplitude to correct for the pulse to pulse subtraction
            # the factor 2 might not be correct, as the acf of ionospheric plasma is affected by clutter subtraction
            # probably best to have a separate calibration constant for ground clutter subtracted data
            #a[0:rg_clutter_rem_cutoff,:]=a_g[0:rg_clutter_rem_cutoff,:]
            
            v=h["acfs_var"][()]
            tsys+=h["T_sys"][()]            

            debris=n.zeros(n_rg,dtype=bool)
            debris[:]=False

            # fit gaussian model to determine if there are space objects at some range gates
            if False:
                plt.pcolormesh(lag,rgs,a.real)
                plt.colorbar()
                plt.show()
            ao=n.copy(a)
            vo=n.copy(v)            
            for ri in range(acf.shape[0]):
                if n.sum(n.isnan(a[ri,:]))/len(lag) < 0.5 and rgs[ri] > 250.0:
                    #print(rgs[ri])
                    try:
                        gres,gsigma=fit_gaussian(ao[ri,:],lag,n.real(n.abs(vo[ri,:])),plot=False)
                    
                        if gres[0]<300.0 and gsigma[0]<100:
                            print("debris at %1.0f km dopp width %1.0f+/-%1.0f (m/s)"%(rgs[ri],gres[0],gsigma[0]))
                            # make neighbouring range gates contaminated
                            
                            # store time and range so that a warning label can be attached
                            # to the data regarding potentially corrupted data near the region
                            space_object_times.append(h["i0"][()])
                            space_object_rgs.append(rgs[ri])
                            
                            for rg_inc in range(-2,2):
                                if (rg_inc+ri >= 0) and (rg_inc+ri < n_rg):
                                    debris[ri+rg_inc]=True
                                    a[ri+rg_inc,:]=n.nan
                                    v[ri+rg_inc,:]=n.nan
                                    space_object_count[ri+rg_inc]+=1
                    except:
                        traceback.print_exc()
                        pass
                    
                            
            
            acfs[ai,:,:]=a/v
            wgts[ai,:,:]=1/v
            h.close()

        tsys=tsys/n_avged
        ptx=ptx/n_avged

        mean_az=mean_az/n_avged
        mean_el=mean_el/n_avged

        var=1/n.nansum(wgts,axis=0)
        acf=n.nansum(acfs,axis=0)/n.nansum(wgts,axis=0)

        # be a bit more careful with this, to avoid outliers due to space debris etc influencing this in short timescales
        ampgain=n.nanmedian(n.array(ampgains))

        # apply receiver gain correction to acf 
        acf=acf/ampgain
        var=var/ampgain**2.0

        hgts=n.copy(rgs)
        if use_misa:
            for hi in range(len(hgts)):
                # height in km based on misa antenna
                hgts[hi] = jcoord.az_el_r2geodetic(mrs.radar_lat,
                                                   mrs.radar_lon,
                                                   mrs.radar_hgt,
                                                   mean_az,mean_el,
                                                   1e3*rgs[hi])[2]/1e3
        
        if True:
            range_limit_idx=[]
            # use heights for range limits, not ranges
            for rai,ra in enumerate(range_limits):
                range_limit_idx.append(n.argmin(n.abs(hgts-ra)))

            acf_orig=n.copy(acf)
            var_orig=n.copy(var)
            range_weight=n.copy(acf)
            range_weight[:,:]=0.0
            for ri in range(len(rgs)):
                range_weight[ri,:]=rgs[ri]**2.0
            for rai,ra in enumerate(range_avg):
                avg_acf=n.copy(acf_orig)
                avg_var=n.copy(var_orig)        

                if ra > 0:

                    for ri in range(acf.shape[0]):
                        avg_acf[ri,:]=n.nansum(range_weight[n.max((0,(ri-ra))):n.min((acf.shape[0],(ri+ra))),:]*acf_orig[n.max((0,(ri-ra))):n.min((acf.shape[0],(ri+ra))),:],axis=0)/n.nansum(range_weight[n.max((0,(ri-ra))):n.min((acf.shape[0],(ri+ra))),:],axis=0)
                        avg_var[ri,:]=1/(n.nansum(1/var_orig[(ri-ra):n.min((acf.shape[0],(ri+ra))),:],axis=0))

                acf[range_limit_idx[rai]:range_limit_idx[rai+1],:]=avg_acf[range_limit_idx[rai]:range_limit_idx[rai+1],:]
                var[range_limit_idx[rai]:range_limit_idx[rai+1],:]=avg_var[range_limit_idx[rai]:range_limit_idx[rai+1],:]
        
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
        #        print("tx power %1.2f MW"%(zpm(0.5*(t0+t1))/1e6))

        for ri in range(acf.shape[0]):
            
            hgt=rgs[ri]
            
            if use_misa:
                print("misa has more power. using misa pointing")
                # height in km based on misa antenna
                hgt = jcoord.az_el_r2geodetic(mrs.radar_lat,
                                              mrs.radar_lon,
                                              mrs.radar_hgt,
                                              mean_az,mean_el,1e3*rgs[ri])[2]/1e3
                print("misa range %1.2f height %1.2f"%(rgs[ri],hgt))            
            try:
                if (n.sum(n.isnan(acf[ri,first_lag:n_lags]))/(n_lags-first_lag) < 0.8):
                    if hgt>700:
                        res,model_acf,dres=fit_acf_ts(acf[ri,first_lag:n_lags],lag[first_lag:n_lags],hgt,var[ri,first_lag:n_lags],guess=guess,plot=plot ,scaling_constant=scaling_constant)
                    else:
                        res,model_acf,dres=fit_acf(acf[ri,first_lag:n_lags],lag[first_lag:n_lags],hgt,var[ri,first_lag:n_lags],guess=guess,plot=plot ,scaling_constant=scaling_constant)
                        
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
                
                ne_const=(1+res[0])*rgs[ri]**2.0/ptx#zpm(0.5*(t0+t1))
                res_out[3]=res[3]*ne_const
                dres_out[3]=dres_out[3]*ne_const
                pp.append(res_out)
                dpp.append(dres_out)                
            except:
                pp.append([n.nan,n.nan,n.nan,n.nan])
                dpp.append([n.nan,n.nan,n.nan,n.nan])
                traceback.print_exc()
                nan_frac=n.sum(n.isnan(acf[ri,first_lag:n_lags]))/(n_lags-first_lag)
                print(acf[ri,first_lag:n_lags])
                print("error caught at range %1.0f km (nanfrac=%1.0f). marching onwards."%(rgs[ri],nan_frac))
        plt.subplot(121)
        plt.pcolormesh(lag[first_lag:n_lags]*1e6,rgs,model_acfs[:,first_lag:n_lags].real,vmin=-0.2,vmax=1.1)
        plt.title("Best fit")        
        plt.xlabel(r"Lag ($\mu$s)")
        plt.ylabel(r"Range (km)")        
        plt.colorbar()
        plt.subplot(122)
        plt.pcolormesh(lag[first_lag:n_lags]*1e6,rgs,acf0[:,first_lag:n_lags].real,vmin=-0.2,vmax=1.1)
        plt.title("Measurement")
        plt.xlabel(r"Lag ($\mu$s)")
        plt.ylabel("Range (km)")
        plt.colorbar()
        plt.tight_layout()
        plt.savefig("%s/pp_fit_%d.png"%(output_dir,t0))
        plt.close()
        plt.clf()

        
        pp=n.array(pp)
        dpp=n.array(dpp)
        plt.plot(pp[:,0]*pp[:,1],rgs,".",label=r"$T_e$")
        plt.plot(pp[:,1],rgs,".",label=r"$T_i$")
        plt.plot(pp[:,2]*10,rgs,".",label=r"$v_i\times 10$")
        plt.plot(n.log10(pp[:,3]),rgs,".",label=r"Raw $n_e$")
        plt.xlim([-1000,5000])
        plt.legend()
        plt.tight_layout()
        plt.savefig("%s/pp-%d.png"%(output_dir,t0))
        plt.close()
        plt.clf()

        ho=h5py.File("%s/pp-%d.h5"%(output_dir,t0),"w")
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
        ho["az"]=mean_az
        ho["el"]=mean_el

        ho["range_avg_limits_km"]=range_limits
        ho["range_avg_window_km"]=(rgs[1]-rgs[0])*(2*range_avg+1)
        

        ho["T_sys"]=tsys
        ho["P_tx"]=ptx#zpm(0.5*(t0+t1))

        ho["space_object_count"]=space_object_count
        ho["space_object_times"]=space_object_times
        ho["space_object_rgs"]=space_object_rgs        
        ho.close()
            


if __name__ == "__main__":
    import sys
    #
    # 
    #               "/media/j/fee7388b-a51d-4e10-86e3-5cabb0e1bc13/isr/2023-09-24/usrp-rx0-r_20230924T200050_20230925T041059/",
    #     dirnames=["/media/j/fee7388b-a51d-4e10-86e3-5cabb0e1bc13/isr/2021-12-01/usrp-rx0-r_20211201T230000_20211202T160100/",
    #               "/media/j/fee7388b-a51d-4e10-86e3-5cabb0e1bc13/isr/2021-12-03a/usrp-rx0-r_20211203T224500_20211204T160000/"]
    #
    #
    #
    # dirnames=["/media/j/fee7388b-a51d-4e10-86e3-5cabb0e1bc13/isr/2021-12-03a/usrp-rx0-r_20211203T224500_20211204T160000/",
    #           
    #           
    #           
    # dirnames=["/media/j/fee7388b-a51d-4e10-86e3-5cabb0e1bc13/isr/2023-09-28/usrp-rx0-r_20230928T211929_20230929T040533/",
    #           "/media/j/fee7388b-a51d-4e10-86e3-5cabb0e1bc13/isr/2021-12-03/usrp-rx0-r_20211203T000000_20211203T033600/",
    #           "/media/j/fee7388b-a51d-4e10-86e3-5cabb0e1bc13/isr/2021-12-05/usrp-rx0-r_20211205T000000_20211205T160100/",
    #           "/media/j/fee7388b-a51d-4e10-86e3-5cabb0e1bc13/isr/2021-12-06/usrp-rx0-r_20211206T000000_20211206T132500/",
    #           "/media/j/fee7388b-a51d-4e10-86e3-5cabb0e1bc13/isr/2021-12-21/usrp-rx0-r_20211221T125500_20211221T220000/",
    #           "/media/j/fee7388b-a51d-4e10-86e3-5cabb0e1bc13/isr/2023-10-14/usrp-rx0-r_20231014T130000_20231015T041500",
    #           "/media/j/fee7388b-a51d-4e10-86e3-5cabb0e1bc13/isr/2023-09-24/usrp-rx0-r_20230924T200050_20230925T041059/",
    #           "/media/j/fee7388b-a51d-4e10-86e3-5cabb0e1bc13/isr/2023-09-05/usrp-rx0-r_20230905T214448_20230906T040054",
    #           "/media/j/fee7388b-a51d-4e10-86e3-5cabb0e1bc13/isr/2021-12-01/usrp-rx0-r_20211201T230000_20211202T160100/",
    #           "/media/j/fee7388b-a51d-4e10-86e3-5cabb0e1bc13/isr/2021-12-03a/usrp-rx0-r_20211203T224500_20211204T160000/",
    #           "/media/j/fee7388b-a51d-4e10-86e3-5cabb0e1bc13/isr/2023-09-20/usrp-rx0-r_20230920T202127_20230921T040637/"]
#    dirnames=["/media/j/4df2b77b-d2db-4dfa-8b39-7a6bece677ca/eclipse2024/usrp-rx0-r_20240407T100000_20240409T110000"]

    dirnames=["/media/j/fee7388b-a51d-4e10-86e3-5cabb0e1bc13/isr/2021-12-01/usrp-rx0-r_20211201T230000_20211202T160100"]
    
    for d in dirnames:
        try:
            fit_lpifiles(dirn=d,channel="misa-l",postfix="_30", max_dt=300, plot=0, first_lag=0, reanalyze=True, range_avg=n.array([1,3,5]))
        except:
            traceback.print_exc()            
        exit(0)

        try:
            fit_lpifiles(dirn=d, channel="zenith-l", postfix="_30", max_dt=300, plot=0, first_lag=0, reanalyze=True, range_avg=n.array([2,3,5]))
        except:
        #           print("problem with zenith")
            traceback.print_exc()
        #          pass

        #        zpm,mpm=txp.get_tx_power_model(dirn="%s/metadata/powermeter"%(d))

