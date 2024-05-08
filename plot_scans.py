import numpy as n
import h5py
import matplotlib.pyplot as plt
import jcoord
import sys
import cartopy.crs as ccrs
from datetime import datetime
from cartopy.feature.nightshade import Nightshade

h=h5py.File(sys.argv[1],"r")
print(h.keys())




dt=n.diff(h["time_unix"][()])
#plt.plot(dt,".")
#plt.show()

si=n.where(n.diff(h["time_unix"][()]) > 120)[0]
si=n.concatenate(([-1],si))

az=n.copy(h["az"][()])
el=n.copy(h["el"][()])
ne=n.copy(h["ne"][()])
te=n.copy(h["Te"][()])
vi=n.copy(h["vi"][()])
ranges=n.copy(h["range"][()])
ts=n.copy(h["time_unix"][()])
max_range=2500

for i in range(len(si)-1):
    print("%d-%d"%(si[i],si[i+1]))
    lats=[]
    lons=[]
    hs=[]
    nes=[]
    tes=[]
    vis=[]        
    prgs=[]    

    
    this_az=az[(si[i]+1):si[i+1]]
    this_el=el[(si[i]+1):si[i+1]]
    this_ne=ne[(si[i]+1):si[i+1],:]

    this_te=te[(si[i]+1):si[i+1],:]
    this_vi=vi[(si[i]+1):si[i+1],:]
    

    for ti in range(len(this_az)):
        for ri in range(len(ranges)):
            llh=jcoord.az_el_r2geodetic(jcoord.misa_lat, jcoord.misa_lon, jcoord.misa_h, this_az[ti], this_el[ti], ranges[ri]*1e3)
            if ranges[ri] < max_range:
                lats.append(llh[0])
                lons.append(llh[1])
                hs.append(llh[2])
                nes.append(this_ne[ti,ri])

                tes.append(this_te[ti,ri])
                vis.append(this_vi[ti,ri])
                
                prgs.append(ranges[ri])

    prgs=n.array(prgs)
    nes=n.array(nes)
    vis=n.array(vis)
    tes=n.array(tes)        

    fig = plt.figure(figsize=[8, 8])
    ax = fig.add_subplot(1, 1, 1, projection=ccrs.Orthographic(jcoord.misa_lon, jcoord.misa_lat))
    
    data_projection = ccrs.PlateCarree()
    this_frame_date=datetime.utcfromtimestamp(ts[si[i]+1])
    ax.coastlines(zorder=3)
    ax.stock_img()
    ax.gridlines()
        
    ax.add_feature(Nightshade(this_frame_date))

    # make a scatterplot
    mp=ax.scatter(lons,
                  lats,
                  c=vis,
                  s=1*prgs/1e2,
                  vmin=-500,vmax=500,
                  transform=data_projection,
                  zorder=2,
                  cmap="seismic")
    cb=plt.colorbar(mp,ax=ax)
    cb.set_label("Vi (m/s)")
    time_str=this_frame_date.strftime('%Y-%m-%d %H:%M:%S')
    ax.set_title(time_str)
    plt.tight_layout()
    plt.savefig("eclipse/vi_h_map-%d.png"%(ts[si[i]+1]))
    plt.close()
    
    
    fig = plt.figure(figsize=[8, 8])
    ax = fig.add_subplot(1, 1, 1, projection=ccrs.Orthographic(jcoord.misa_lon, jcoord.misa_lat))
    
    data_projection = ccrs.PlateCarree()
    this_frame_date=datetime.utcfromtimestamp(ts[si[i]+1])
    ax.coastlines(zorder=3)
    ax.stock_img()
    ax.gridlines()
        
    ax.add_feature(Nightshade(this_frame_date))

    # make a scatterplot
    mp=ax.scatter(lons,
                  lats,
                  c=n.log10(nes),
                  s=2*prgs/1e2,
                  vmin=10,vmax=12,
                  transform=data_projection,
                  zorder=2,
                  cmap="turbo")
    cb=plt.colorbar(mp,ax=ax)
    cb.set_label("Ne (log10)")
    time_str=this_frame_date.strftime('%Y-%m-%d %H:%M:%S')
    ax.set_title(time_str)
    plt.tight_layout()
    plt.savefig("eclipse/map-%d.png"%(ts[si[i]+1]))
    plt.close()


    fig = plt.figure(figsize=[8, 8])
    ax = fig.add_subplot(1, 1, 1, projection=ccrs.Orthographic(jcoord.misa_lon, jcoord.misa_lat))
    
    data_projection = ccrs.PlateCarree()
    this_frame_date=datetime.utcfromtimestamp(ts[si[i]+1])
    ax.coastlines(zorder=3)
    ax.stock_img()
    ax.gridlines()
        
    ax.add_feature(Nightshade(this_frame_date))

    # make a scatterplot
    mp=ax.scatter(lons,
                  lats,
                  c=tes,
                  s=2*prgs/1e2,
                  vmin=500,vmax=4000,
                  transform=data_projection,
                  zorder=2,
                  cmap="turbo")
    cb=plt.colorbar(mp,ax=ax)
    cb.set_label("Te (K)")
    time_str=this_frame_date.strftime('%Y-%m-%d %H:%M:%S')
    ax.set_title(time_str)
    plt.tight_layout()
    plt.savefig("eclipse/temap-%d.png"%(ts[si[i]+1]))
    plt.close()


    fig = plt.figure(figsize=[8, 8])
    ax = fig.add_subplot(1, 1, 1, projection=ccrs.Orthographic(jcoord.misa_lon, jcoord.misa_lat))
    
    data_projection = ccrs.PlateCarree()
    this_frame_date=datetime.utcfromtimestamp(ts[si[i]+1])
    ax.coastlines(zorder=3)
    ax.stock_img()
    ax.gridlines()
        
    ax.add_feature(Nightshade(this_frame_date))

    # make a scatterplot
    mp=ax.scatter(lons,
                  lats,
                  c=vis,
                  s=1*prgs/1e2,
                  vmin=-100,vmax=100,
                  transform=data_projection,
                  zorder=2,
                  cmap="seismic")
    cb=plt.colorbar(mp,ax=ax)
    cb.set_label("Vi (m/s)")
    time_str=this_frame_date.strftime('%Y-%m-%d %H:%M:%S')
    ax.set_title(time_str)
    plt.tight_layout()
    plt.savefig("eclipse/vimap-%d.png"%(ts[si[i]+1]))
    plt.close()
    
    
#    plt.scatter(lons,lats,c=n.log10(nes),vmin=10,vmax=12,cmap="turbo",s=5*(prgs/1e2))
 #   plt.colorbar()
  #  plt.show()
#    plt.plot(az[(si[i]+1):si[i+1]],".")
 #   plt.show()

print(si)


h.close()
