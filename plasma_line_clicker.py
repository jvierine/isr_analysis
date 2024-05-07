import numpy as n
import glob
import h5py
import matplotlib.pyplot as plt
from digital_rf import DigitalRFReader, DigitalMetadataReader, DigitalMetadataWriter
# pip install mpi_point_clicker
from mpl_point_clicker import clicker
import sys

def click_spec(misa_t,misa_s,freqs,dirname,calname="cal.h5"):

    fig, ax = plt.subplots(constrained_layout=True)

    c=ax.pcolormesh(misa_t,freqs,misa_s.T,vmin=0,vmax=0.1,cmap="plasma")
#    fig.colorbar(c,ax=ax)
    ax.set_title(calname)
    

    #zoom_factory(ax)
    #ph = panhandler(fig, button=2)
    klicker = clicker(
        ax,
       ["plasma_frequency"],
       markers=["x"]
    )

    from typing import Tuple
    def point_added_cb(position: Tuple[float, float], klass: str):
        x, y = position
        print(f"New point of class {klass} added at {x=}, {y=}")

        kp=klicker.get_positions()
        print(kp)
        print("writing %s/%s.h5"%(dirname,calname))
        ho=h5py.File("%s/%s.h5"%(dirname,calname),"w")
        ho["plasma_frequency"]=kp["plasma_frequency"]
        ho.close()

    def point_removed_cb(position: Tuple[float, float], klass: str, idx):
        x, y = position

        suffix = {'1': 'st', '2': 'nd', '3': 'rd'}.get(str(idx)[-1], 'th')
        print(
            f"The {idx}{suffix} point of class {klass} with position {x=:.2f}, {y=:.2f}  was removed"
        )

    klicker.on_point_added(point_added_cb)
    klicker.on_point_removed(point_removed_cb)
    plt.show()


dirname=sys.argv[1]#"/media/j/fee7388b-a51d-4e10-86e3-5cabb0e1bc13/isr/2023-09-05/usrp-rx0-r_20230905T214448_20230906T040054/"
plmd=DigitalMetadataReader("%s/metadata/integrated_plasma_line_metadata_hires/"%(dirname))
b=plmd.get_bounds()

dt=plmd.get_file_cadence_secs()

n_files=int((b[1]-b[0])/dt)

d = plmd.read_flatdict(b[0],b[0]+60)
#print(d["antenna"])
#exit(0)
spec_shape=(12000, 288)#d["spec"][0].shape
freqs=n.fft.fftshift(n.fft.fftfreq(12000,d=1/25e6))#d["freqs"][0]
ranges=[]#d["ranges"][0]

spec=n.zeros(spec_shape,dtype=n.float32)
S=n.zeros([2,n_files,len(freqs)],dtype=n.float32)
speci=0
tv=n.zeros([2,n_files])

misa_t=[]
zenith_t=[]
misa_s=[]
zenith_s=[]
for fi in range(n_files):
    print("%d/%d"%(fi,n_files))
    spec[:,:]=0.0
    try:
        d = plmd.read_flatdict(b[0]+fi*60,b[0]+fi*60+60)
    except:
        continue
    if "spec" in d.keys():
        n_misa=0
        n_zenith=0
        z_t=0
        m_t=0
        z_s=n.zeros(len(freqs))
        m_s=n.zeros(len(freqs))
        
        for si in range(len(d["spec"])):
            if len(d["ranges"][si] == 288):
                ranges=d["ranges"][si]
                
            if d["antenna"][si] == "MISA":
                # tbd fix hard coded range gates!
                print(d["spec"][si].shape)
 #               print(d["ranges"][si])
#                print(len(d["ranges"][si]))
                if len(d["ranges"][si])>150:

##                    plt.pcolormesh(d["spec"][si],vmin=0,vmax=0.1)
  #                  plt.colorbar()
   #                 plt.show()
                    
                    m_s+=n.nanmax(d["spec"][si][:,50:150]-n.nanmedian(d["spec"][si][:,50:150],axis=0),axis=1)
                    m_t+=d["t1"][si]
                    n_misa+=1
                else:
                    print("ignoring")
                print("misa")
            else:
                if len(d["ranges"][si])>150:                
                    z_s+=n.nanmax(d["spec"][si][:,50:150]-n.nanmedian(d["spec"][si][:,50:150],axis=0),axis=1)
                    tv[1,speci]+=d["t1"][si]
                    z_t+=d["t1"][si]
                    n_zenith+=1.0
                else:
                    print("ignoring")
                print("zenith")                

        if n_misa>0:
            m_s=m_s/n_misa
            m_s=m_s-n.nanmedian(m_s)
            m_t=m_t/n_misa
            misa_s.append(m_s)
            misa_t.append(m_t)
        if n_zenith>0:
            print(n_zenith)
            z_s=z_s/n_zenith
            z_s=z_s-n.nanmedian(z_s)
            z_t=z_t/n_zenith
            print(z_t)
#            plt.plot(z_s)
 #           plt.show()
            zenith_s.append(z_s)
            zenith_t.append(z_t)
            
    else:
        print("no spec key")

zenith_s=n.vstack(zenith_s)
click_spec(zenith_t,zenith_s,freqs,dirname,calname="zenithcal")
if len(misa_s) > 0:
    misa_s=n.vstack(misa_s)
    click_spec(misa_t,misa_s,freqs,dirname,calname="misacal")
    

