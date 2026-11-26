"""Adopted steering model for the November SYISR lower-hybrid campaign."""
import numpy as np

VERSION='1.0.0'
def phase_table_pointing(azimuth_deg,elevation_deg,design_mhz=440.,operating_mhz=430.,boresight=(0.,90.)):
    """Fixed phase gradient, scaled in the mechanical array plane, in ENU."""
    if not all(np.isfinite([azimuth_deg,elevation_deg,design_mhz,operating_mhz,*boresight])) or min(design_mhz,operating_mhz)<=0:
        raise ValueError('Finite angles and positive frequencies required')
    def direction(az,el):
        a,e=np.deg2rad([az,el]);return np.array([np.sin(a)*np.cos(e),np.cos(a)*np.cos(e),np.sin(e)])
    b=direction(azimuth_deg,elevation_deg);n=direction(*boresight)
    if b@n<=0:raise ValueError('Pointing is outside the array forward hemisphere')
    t=(b-(b@n)*n)*(design_mhz/operating_mhz);s=t@t
    if s>=1:raise ValueError('Phase gradient has no forward steering solution')
    corrected=t+np.sqrt(1-s)*n
    return float(np.rad2deg(np.arctan2(corrected[0],corrected[1]))%360),float(np.rad2deg(np.arcsin(np.clip(corrected[2],-1,1))))

def add_arguments(parser):
    parser.add_argument('--pointing-model',choices=['phase-table','header','empirical-offset'],default='phase-table')
    parser.add_argument('--design-frequency-mhz',type=float,default=440.)
    parser.add_argument('--operating-frequency-mhz',type=float,default=430.)
    parser.add_argument('--elevation-offset-deg',type=float,default=None)

def physical_pointing(az,el,args):
    if args.pointing_model!='empirical-offset' and args.elevation_offset_deg is not None:
        raise ValueError('An elevation offset requires --pointing-model empirical-offset; it cannot be stacked on the phase-table prediction')
    if args.pointing_model=='phase-table':return phase_table_pointing(az,el,args.design_frequency_mhz,args.operating_frequency_mhz)
    if args.pointing_model=='empirical-offset':
        if args.elevation_offset_deg is None:raise ValueError('Specify --elevation-offset-deg with empirical-offset')
        return float(az),float(el+args.elevation_offset_deg)
    return float(az),float(el)

def metadata(args):
    return dict(pointing_model=args.pointing_model,pointing_model_version=VERSION,
                phase_table_design_frequency_hz=args.design_frequency_mhz*1e6,operating_frequency_hz=args.operating_frequency_mhz*1e6,
                mechanical_boresight_azimuth_deg=0.,mechanical_boresight_elevation_deg=90.,
                pointing_status=(f'Adopted fixed-phase {args.design_frequency_mhz:g}MHz design hypothesis at {args.operating_frequency_mhz:g}MHz; model prediction, not independent calibration; empirical offset replaced' if args.pointing_model=='phase-table' else 'Header direction' if args.pointing_model=='header' else 'Legacy explicit empirical offset'),
                fixed_phase_hardware_status='Assumed; steering tables/controller unavailable')
