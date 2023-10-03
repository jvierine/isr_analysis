import numpy as n
import glob
import h5py
import matplotlib.pyplot as plt
from digital_rf import DigitalRFReader, DigitalMetadataReader, DigitalMetadataWriter
# pip install mpi_point_clicker
from mpl_point_clicker import clicker


dirname="/media/j/fee7388b-a51d-4e10-86e3-5cabb0e1bc13/isr/2023-09-05/usrp-rx0-r_20230905T214448_20230906T040054/"
plmd=DigitalMetadataReader("%s/metadata/integrated_plasma_line_metadata_hires/"%(dirname))
b=plmd.get_bounds()

dt=plmd.get_file_cadence_secs()

n_files=int((b[1]-b[0])/dt)

d = plmd.read_flatdict(b[0],b[0]+60)
spec_shape=d["spec"][0].shape
freqs=d["freqs"][0]
ranges=d["ranges"][0]

spec=n.zeros(spec_shape,dtype=n.float32)
S=n.zeros([n_files,len(freqs)],dtype=n.float32)
speci=0
tv=n.zeros(n_files)
for fi in range(n_files):
    print("%d/%d"%(fi,n_files))
    spec[:,:]=0.0
    d = plmd.read_flatdict(b[0]+fi*60,b[0]+fi*60+60)
    if "spec" in d.keys():
        for si in range(len(d["spec"])):
            S[speci,:]+=n.max(d["spec"][si][:,50:150],axis=1)
            tv[speci]+=d["t1"][si]
        tv[speci]=tv[speci]/len(d["spec"])
        speci+=1
    else:
        print("no spec key")

dB=10.0*n.log10(S[0:speci,:])
nf=n.nanmedian(dB)


fig, ax = plt.subplots(constrained_layout=True)


ax.pcolormesh(tv[0:speci],freqs,dB.T,vmin=nf,vmax=nf+3,cmap="plasma")
#plt.colorbar()

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
    print("writing %s/cal.h5"%(dirname))
    ho=h5py.File("%s/cal.h5"%(dirname),"w")
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
