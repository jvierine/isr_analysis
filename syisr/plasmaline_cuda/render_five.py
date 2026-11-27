"""Full native-grid inferno figures; common Tukey-IQR limits, no data averaging.

Rendering provenance belongs in HDF5 metadata, never in visible plot text.
"""
import datetime as dt
import copy
import pathlib
import sys
import subprocess
import h5py
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from PIL import Image

root=pathlib.Path(sys.argv[1]);folder=root/'five-profiles-step2us'
with h5py.File(folder/'selection.h5') as f:
    paths=f['matched_path'].asstr()[:];times=f['actual_start_utc_s'][:]
values=[]
column_background=[]
for path in paths:
    with h5py.File(path) as f:
        p=f['whitened_matched_power_ratio'][:].astype('f4')
        background=np.nanmedian(p,axis=0)
        if not np.all(np.isfinite(background)&(background>0)): raise ValueError('invalid decoded column background')
        column_background.append(background)
        p/=background[None,:]
        db=10*np.log10(p,where=p>0,out=np.full_like(p,np.nan))
        values.append(db[np.isfinite(db)])
all_values=np.concatenate(values);q1,q3=np.quantile(all_values,[.25,.75]);iqr=q3-q1;vmin=q1-1.5*iqr;vmax=q3+1.5*iqr
del values,all_values
with h5py.File(folder/'display_settings_frequency_normalized.h5','w') as out:
    out['quartiles_db']=[q1,q3];out['color_limits_db']=[vmin,vmax]
    out['decoded_frequency_background_power_ratio']=np.asarray(column_background)
    out['source_matched_path']=np.asarray(paths,dtype=h5py.string_dtype())
    out.attrs.update(palette='inferno',limits='Shared Q1-1.5IQR to Q3+1.5IQR of all finite post-decoding frequency-normalized power values in dB across five profiles',normalization='For each integration and each decoded frequency bin independently: decoded power divided by median decoded power across all valid range gates. Original decoded products unchanged.',native_grid=True,range_step_us=2,renderer='syisr/plasmaline_cuda/render_five.py',source_commit=subprocess.check_output(['git','rev-parse','HEAD'],cwd=pathlib.Path(__file__).resolve().parent,text=True).strip())
for i,(path,t) in enumerate(zip(paths,times)):
    with h5py.File(path) as f:
        p=f['whitened_matched_power_ratio'][:].astype('f4');r=f['slant_range_km'][:];freq=f['frequency_offset_hz'][:]/1e6
        accepted=int(f.attrs['accepted_pulses']);nr,nf=p.shape
    p/=column_background[i][None,:]
    if not np.allclose(np.nanmedian(p,axis=0),1,rtol=2e-6): raise ValueError('decoded frequency normalization failed')
    db=10*np.log10(p,where=p>0,out=np.full_like(p,np.nan));del p
    dr=r[1]-r[0];df=freq[1]-freq[0]
    extent=[freq[0]-df/2,freq[-1]+df/2,r[0]-dr/2,r[-1]+dr/2]
    begin=dt.datetime.fromtimestamp(float(t),dt.timezone.utc);end=begin+dt.timedelta(seconds=60)
    title=f'{begin:%Y-%m-%d  %H:%M}–{end:%H:%M} UTC\n60 s window; {accepted*.01:.2f} s accepted'
    # Native image occupies exactly nf x nr output pixels. Margins and axes
    # add pixels; the scientific array is never binned, smoothed or reduced.
    width=nf+1800;height=nr+1150;dpi=300
    fig=plt.figure(figsize=(width/dpi,height/dpi),dpi=dpi)
    ax=fig.add_axes([800/width,500/height,nf/width,nr/height])
    cmap=copy.copy(plt.get_cmap('inferno'));cmap.set_bad('white')
    im=ax.imshow(db,origin='lower',aspect='auto',extent=extent,cmap=cmap,vmin=vmin,vmax=vmax,interpolation='none',resample=False)
    ax.set_xlabel('Frequency offset (MHz)',fontsize=30,labelpad=20);ax.set_ylabel('Slant range (km)',fontsize=30,labelpad=20)
    ax.tick_params(labelsize=26,width=2,length=18,pad=12);ax.set_title(title,fontsize=34,pad=35)
    cax=fig.add_axes([(800+nf+150)/width,500/height,140/width,nr/height])
    cb=fig.colorbar(im,cax=cax);cb.ax.tick_params(labelsize=24,length=12,pad=12);cb.set_label('Frequency-normalized decoded power (dB)',fontsize=26,labelpad=20)
    target=folder/f'profile-{i+1:02d}-freqnorm-full.png';fig.savefig(target,dpi=dpi);plt.close(fig)
    # Compact previews are separate from the full-resolution export.
    with Image.open(target) as image:
        image.thumbnail((1800,1400),Image.Resampling.LANCZOS if hasattr(Image,'Resampling') else Image.LANCZOS)
        image.convert('RGB').save(folder/f'profile-{i+1:02d}-freqnorm-preview.jpg',quality=95)
    print(f'RENDERED {target}: native {nf}x{nr}, color limits {vmin:g}..{vmax:g} dB; {accepted} accepted IPPs',flush=True)
