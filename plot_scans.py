import numpy as n
import h5py
import matplotlib.pyplot as plt
import jcoord
import sys
import cartopy.crs as ccrs
from datetime import datetime
from cartopy.feature.nightshade import Nightshade

h=h5py.File("ppar.h5","r")


# figure out magic constant using zenith data
az=h["az"][()]
cal_az=-90
cal_daz=10
cal_idx=n.where(n.abs(az-cal_az)<cal_daz)[0]

hc=h5py.File("cal_lowel.h5","r")
print(hc.keys())

time_unix_cal=hc["time_unix"][()]
time_unix=h["time_unix"][()][cal_idx]
ne_cal=hc["ne"][()]
ne=h["ne"][()]
ne=ne[cal_idx,:]

mc_factors=[]
for ti in range(len(time_unix)):
    midx=n.argmin(n.abs(time_unix[ti]-time_unix_cal))
    mc_factor=n.nanmax(ne_cal[midx,:])/n.nanmax(ne[ti,:])
    mc_factors.append(mc_factor)

#plt.plot(mc_factors,".")
#plt.show()
    
plt.subplot(121)
mcfactor=n.nanmedian(mc_factors)
print("misa low elevation long pulse to zenith alternating code magic constant factor %1.2f"%(mcfactor))
plt.pcolormesh(n.log10(mcfactor*ne.T),vmin=10,vmax=12,cmap="turbo")
plt.colorbar()
plt.subplot(122)
plt.pcolormesh(n.log10(ne_cal.T),vmin=10,vmax=12,cmap="turbo")
plt.show()
hc.close()

print(h.keys())




dt=n.diff(h["time_unix"][()])
#plt.plot(dt,".")
#plt.show()

si=n.where(n.diff(h["time_unix"][()]) > 120)[0]
si=n.concatenate(([-1],si))

az=n.copy(h["az"][()])
el=n.copy(h["el"][()])
ne=n.copy(h["ne"][()])*mcfactor
te=n.copy(h["Te"][()])
ti=n.copy(h["Ti"][()])
vi=n.copy(h["vi"][()])
dvi=n.copy(h["dvi"][()])
ranges=n.copy(h["range"][()])
ts=n.copy(h["time_unix"][()])
max_range=2500

vi[dvi>200]=n.nan
ne[dvi>200]=n.nan

for i in range(len(si)-1):
    print("%d-%d"%(si[i],si[i+1]))
    lats=[]
    lons=[]
    hs=[]
    nes=[]
    tes=[]
    tis=[]    
    vis=[]        
    prgs=[]    

    
    this_az=az[(si[i]+1):si[i+1]]
    this_el=el[(si[i]+1):si[i+1]]
    this_ne=ne[(si[i]+1):si[i+1],:]

    this_te=te[(si[i]+1):si[i+1],:]
    this_ti=ti[(si[i]+1):si[i+1],:]    
    this_vi=vi[(si[i]+1):si[i+1],:]
    

    for tid in range(len(this_az)):
        for ri in range(len(ranges)):
            llh=jcoord.az_el_r2geodetic(jcoord.misa_lat, jcoord.misa_lon, jcoord.misa_h, this_az[tid], this_el[tid], ranges[ri]*1e3)
            if ranges[ri] < max_range:
                lats.append(llh[0])
                lons.append(llh[1])
                hs.append(llh[2])
                nes.append(this_ne[tid,ri])

                tes.append(this_te[tid,ri])
                tis.append(this_ti[tid,ri])                
                vis.append(this_vi[tid,ri])
                
                prgs.append(ranges[ri])

    prgs=n.array(prgs)
    nes=n.array(nes)
    vis=n.array(vis)
    tes=n.array(tes)        
    tis=n.array(tis)
    psize=4.5*0.5*0.8*0.8
    fig = plt.figure(figsize=[8, 6.4])
    ax = fig.add_subplot(2, 2, 1, projection=ccrs.Orthographic(jcoord.misa_lon, jcoord.misa_lat))
    
    data_projection = ccrs.PlateCarree()
    this_frame_date=datetime.utcfromtimestamp(ts[si[i]+1])
    ax.coastlines(zorder=3)
    ax.stock_img()
    ax.gridlines()
    ax.set_extent((-60,-100,20,60),crs=ccrs.PlateCarree())    
        
    ax.add_feature(Nightshade(this_frame_date))

    # make a scatterplot
    mp=ax.scatter(lons,
                  lats,
                  c=vis,
                  s=psize*prgs/1e2,
                  vmin=-600,vmax=600,
                  transform=data_projection,
                  zorder=2,
                  cmap="seismic")
    cb=plt.colorbar(mp,ax=ax)
    cb.set_label("v$_i$ (m/s) red is away from radar")
    time_str=this_frame_date.strftime('%Y-%m-%d %H:%M:%S')
    ax.set_title(time_str)
#    plt.tight_layout()
 #   plt.savefig("eclipse/vi_h_map-%d.png"%(ts[si[i]+1]))
  #  plt.close()
    
    
   # fig = plt.figure(figsize=[1.5*8, 1.5*8])
    ax = fig.add_subplot(2, 2, 2, projection=ccrs.Orthographic(jcoord.misa_lon, jcoord.misa_lat))
    
    data_projection = ccrs.PlateCarree()
    this_frame_date=datetime.utcfromtimestamp(ts[si[i]+1])
    ax.coastlines(zorder=3)
    ax.stock_img()
    ax.gridlines()
    ax.set_extent((-60,-100,20,60),crs=ccrs.PlateCarree())
        
    ax.add_feature(Nightshade(this_frame_date))

    # make a scatterplot
    #c=n.log10(nes),
    mp=ax.scatter(lons,
                  lats,
#                  c=n.log10(nes),
                  c=nes,
                  s=psize*prgs/1e2,
                  vmin=10**10,vmax=10**12,
                  transform=data_projection,
                  zorder=2,
                  cmap="turbo")
    cb=plt.colorbar(mp,ax=ax)
    cb.set_label("n$_e$ (m$^{-3}$)")
    time_str=this_frame_date.strftime('%Y-%m-%d %H:%M:%S')
    
#    ax.set_title("MIT Haystack Observatory")
#    plt.tight_layout()
 #   plt.savefig("eclipse/map-%d.png"%(ts[si[i]+1]))
  #  plt.close()


   # fig = plt.figure(figsize=[8, 8])
    ax = fig.add_subplot(2, 2, 3, projection=ccrs.Orthographic(jcoord.misa_lon, jcoord.misa_lat))
    
    data_projection = ccrs.PlateCarree()
    this_frame_date=datetime.utcfromtimestamp(ts[si[i]+1])
    ax.coastlines(zorder=3)
    ax.stock_img()
    ax.gridlines()
    ax.set_extent((-60,-100,20,60),crs=ccrs.PlateCarree())    
        
    ax.add_feature(Nightshade(this_frame_date))

    # make a scatterplot
    mp=ax.scatter(lons,
                  lats,
                  c=tes,
                  s=psize*prgs/1e2,
                  vmin=500,vmax=4000,
                  transform=data_projection,
                  zorder=2,
                  cmap="turbo")
    cb=plt.colorbar(mp,ax=ax)
    cb.set_label("T$_e$ (K)")


   # fig = plt.figure(figsize=[8, 8])
    ax = fig.add_subplot(2, 2, 4, projection=ccrs.Orthographic(jcoord.misa_lon, jcoord.misa_lat))
    
    data_projection = ccrs.PlateCarree()
    this_frame_date=datetime.utcfromtimestamp(ts[si[i]+1])
    ax.coastlines(zorder=3)
    ax.stock_img()
    ax.gridlines()
    ax.set_extent((-60,-100,20,60),crs=ccrs.PlateCarree())    
        
    ax.add_feature(Nightshade(this_frame_date))

    # make a scatterplot
    mp=ax.scatter(lons,
                  lats,
                  c=tis,
                  s=psize*prgs/1e2,
                  vmin=500,vmax=3000,
                  transform=data_projection,
                  zorder=2,
                  cmap="turbo")
    cb=plt.colorbar(mp,ax=ax)
    cb.set_label("T$_i$ (K)")

    time_str=this_frame_date.strftime('%Y-%m-%d %H:%M:%S')
    ax.text(0.0, -0.05, "Millstone Hill Incoherent Scatter Radar, MIT Haystack Observatory", fontsize=6, horizontalalignment="left",verticalalignment="bottom",transform=ax.transAxes)    
 #   ax.set_title(time_str)
#    plt.tight_layout()
  #  plt.savefig("eclipse/temap-%d.png"%(ts[si[i]+1]))
   # plt.close()
    plt.tight_layout()
    plt.savefig("eclipse/4map-%d.png"%(ts[si[i]+1]),dpi=200)
    plt.close()


    fig = plt.figure(figsize=[8, 8])
    ax = fig.add_subplot(1, 1, 1, projection=ccrs.Orthographic(jcoord.misa_lon, jcoord.misa_lat))
    
    data_projection = ccrs.PlateCarree()
    this_frame_date=datetime.utcfromtimestamp(ts[si[i]+1])
    ax.coastlines(zorder=3)
    ax.stock_img()
    ax.gridlines()
    ax.set_extent((-60,-100,20,60),crs=ccrs.PlateCarree())    
        
    ax.add_feature(Nightshade(this_frame_date))

    # make a scatterplot
    mp=ax.scatter(lons,
                  lats,
                  c=vis,
                  s=10*prgs/1e2,
                  vmin=-100,vmax=100,
                  transform=data_projection,
                  zorder=2,
                  cmap="seismic")
    cb=plt.colorbar(mp,ax=ax)
    cb.set_label("v$_i$ (m/s) red is away from radar")
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
