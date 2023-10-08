import numpy as n
import matplotlib.pyplot as plt
from digital_rf import DigitalRFReader, DigitalMetadataReader, DigitalMetadataWriter
import scipy.interpolate as sint

radar_lat=42.61932878636544
radar_lon=-71.49124624803031
radar_hgt=146.0

def get_tx_power_model(dirn,plot=False):
    print("Reading transmit power meter metadata. Might take a few seconds")
    dmd=DigitalMetadataReader(dirn)
    b=dmd.get_bounds()

    zenith_t=[]
    zenith_pwr=[]
    misa_t=[]
    misa_pwr=[]
    sid = dmd.read(b[0],b[1],"zenith_power")    
    for keyi,key in enumerate(sid.keys()):
        zenith_t.append(key)
        zenith_pwr.append(sid[key])
#        print("%1.2f %1.2f"%(key,sid[key]))
        
    sid = dmd.read(b[0],b[1],"misa_power")    
    for keyi,key in enumerate(sid.keys()):
        misa_t.append(key)
        misa_pwr.append(sid[key])
#        print("%1.2f %1.2f"%(key,sid[key]))

    if plot:
        plt.plot(zenith_t,zenith_pwr)
        plt.plot(misa_t,misa_pwr)    
        plt.show()

    zenith_t[0]=zenith_t[0]-3600
    zenith_t[-1]=zenith_t[-1]+3600
    misa_t[0]=misa_t[0]-3600
    misa_t[-1]=misa_t[-1]+3600
    zenith_pwrf=sint.interp1d(zenith_t,zenith_pwr)
    misa_pwrf=sint.interp1d(misa_t,misa_pwr)
    
    return(zenith_pwrf,misa_pwrf)


def get_antenna_select(dirn,plot=False):
    """
    1 = misa
    -1 = zenith
    """
    print("Reading transmit power meter metadata. Might take a few seconds")
    dmd=DigitalMetadataReader(dirn)
    b=dmd.get_bounds()

    tx_t=[]
    tx_v=[]
    rx_t=[]
    rx_v=[]

    sid = dmd.read(b[0],b[1],"rx_antenna")    
    for keyi,key in enumerate(sid.keys()):

        rx_t.append(key)                    
        if sid[key]==b'MISA':
            rx_v.append(1.0)
        else:
            rx_v.append(-1.0)

    sid = dmd.read(b[0],b[1],"tx_antenna")    
    for keyi,key in enumerate(sid.keys()):

        tx_t.append(key)                    
        if sid[key]==b'MISA':
            tx_v.append(1.0)
        else:
            tx_v.append(-1.0)
            
    rx_t=n.array(rx_t)
    tx_t=n.array(tx_t)
    rx_t0=n.copy(rx_t)
    tx_t0=n.copy(tx_t)    
    
    rx_t[0]=rx_t[0]-24*3600e3
    rx_t[-1]=rx_t[-1]+24*3600e6
    tx_t[-1]=tx_t[-1]+24*3600e6
    tx_t[0]=tx_t[0]-24*3600e6
    rx_sel=sint.interp1d(rx_t,rx_v)
    tx_sel=sint.interp1d(tx_t,tx_v)

    if plot:
        t=n.linspace(b[0],b[1],num=100000)
        plt.plot(t/1e6,rx_sel(t))
        for rxt in rx_t0:
            plt.axvline(rxt/1e6)
        plt.plot(t/1e6,tx_sel(t))
        plt.show()
    return(rx_sel,tx_sel)



def get_misa_az_el_model(dirn):
    acmd=DigitalMetadataReader(dirn)
    b=acmd.get_bounds()
    azk = acmd.read(b[0],b[1],"misa_azimuth")
    az=[]
    azt=[]
    el=[]
    elt=[]
    for keyi,key in enumerate(azk.keys()):
        az.append(azk[key])
        azt.append(key/1e6)
        
    elk = acmd.read(b[0],b[1],"misa_elevation")        
    for keyi,key in enumerate(elk.keys()):
        el.append(elk[key])
        elt.append(key/1e6)

    azt=n.array(azt)
    elt=n.array(elt)
    az=n.array(az)
    el=n.array(el)
    azt[0]=azt[0]-24*3600.0
    azt[-1]=azt[-1]+24*3600.0    
    elt[0]=elt[0]-24*3600.0
    elt[-1]=elt[-1]+24*3600.0

    azf=sint.interp1d(azt,az)
    elf=sint.interp1d(elt,el)
    return(azf,elf,[b[0]/1e6,b[1]/1e6])
    
        
        
        
    

if __name__ == "__main__":
    import sys

    get_antenna_select(sys.argv[1],plot=True)
#    get_tx_power_model(sys.argv[1],plot=True)    
    
#    az,el,b=get_misa_az_el_model("/media/j/fee7388b-a51d-4e10-86e3-5cabb0e1bc13/isr/2023-09-05/usrp-rx0-r_20230905T214448_20230906T040054/metadata/antenna_control_metadata")
 #   t=n.linspace(b[0],b[1],num=1000)
  #  plt.plot(t,az(t))
#    plt.plot(t,el(t))
 #   plt.show()
  #  import jcoord
   # llh=jcoord.az_el_r2geodetic(radar_lat,radar_lon,146,az(b[0]+3600),el(b[0]+3600),400e3)
    #    print(llh)
    #    get_tx_power_model()
