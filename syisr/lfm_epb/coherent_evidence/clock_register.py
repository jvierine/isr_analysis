"""Register L2 clocks to the raw L1 LP pulse clock, without fitting a time shift."""
from pathlib import Path
from datetime import datetime,timezone,timedelta
import re,h5py,numpy as np,sys
out=Path(sys.argv[1]);lp=Path('/gpfs/pub/data/L1/matdata/20260301_SY/LP/beam_358.0_90.0/LP')
source='/gpfs/pub/data/L2/SYISR_results/h5/L2/2026/OSAYA_TISR01_IFIT_L2_DAY_20260301000000_V01.00_N_LP_15sec.h5'
files=[]
for f in lp.glob('*.mat'):
 m=re.search(r'raw(\d{12})_',f.name)
 if m:files.append((datetime.strptime(m[1],'%y%m%d%H%M%S').replace(tzinfo=timezone(timedelta(hours=8))).timestamp(),f))
with h5py.File(source) as h:times=h['Time'][0]
rows=[];names=[]
for i,s in enumerate(times):
 t=datetime.strptime(s.decode().strip(),'%d-%b-%Y %H:%M:%S').replace(tzinfo=timezone.utc).timestamp()
 if i!=0 and not 1772362800<=t<1772373600:continue
 ft,f=min(files,key=lambda a:abs(a[0]-t))
 with h5py.File(f) as h:
  a=h['syspara'][()];n=a.shape[1]
  def clock(j):
   v=a[:,j];d=datetime(2000+int(v[0]),int(v[1]),int(v[2]),int(v[3]),int(v[4]),tzinfo=timezone(timedelta(hours=8)));return d.timestamp()+v[5]
  first,last=clock(0),clock(n-1)
 rows.append([t,ft,first,last,ft-t,first-t,n]);names.append(str(f));print('CLOCK',s.decode(),f.name,'filename_delta_s',ft-t,'first_pulse_delta_s',first-t,flush=True)
with h5py.File(out,'w') as h:
 h['registration']=rows;h['L1_LP_sources']=np.array(names,dtype=h5py.string_dtype());h.attrs['columns']='L2_as_UTC,filename_Beijing_convertedUTC,L1_first_pulse_Beijing_convertedUTC,L1_last_pulse_UTC,filename_delta_s,first_pulse_delta_s,pulse_count'
 h.attrs['method']='Direct registration of nearest zenith L1 LP filename and internal pulse clock to recorded L2 Time; no inferred or fitted offset.'
