"""Curved open isosurfaces and supported morphing, using VTK depth peeling.
Parallel extent +/-300km is a display choice, not inferred structure. Native
HDF5 maps remain untouched; render scalars and interpolation are derived only.
"""
from pathlib import Path
import argparse
import h5py,numpy as np
import pyvista as pv
from scipy.interpolate import RegularGridInterpolator
from matplotlib.colors import ListedColormap
from storm_palette import STORM
STORM_MAP=ListedColormap(STORM)
p=argparse.ArgumentParser();p.add_argument('root',type=Path);p.add_argument('--frames',type=int,default=72);a=p.parse_args();out=a.root/'figures';out.mkdir(exist_ok=True)
with h5py.File(a.root/'field_grid.h5') as h:
    x=h['x_km'][()];y=h['y_km'][()];s=h['along_B_km'][()];xyz=h['enu_km'][()];alt=h['geodetic_altitude_km'][()]
with h5py.File(a.root/'sections.h5') as h:
    xx=h['x_km'][()];yy=h['y_km'][()];xxg,yyg=np.meshgrid(x,y);coords=np.column_stack([yyg.ravel(),xxg.ravel()])
    fields=[];masks=[];times=[];common=RegularGridInterpolator((yy,xx),h['ordered_common_support'][()].astype(float),method='nearest',bounds_error=False,fill_value=0)(coords).reshape(len(y),len(x))>.5
    for name in ['west','central','east']:
        d=h[name]['contrast'][()];w=h[name]['support_weight'][()];t=h[name]['mean_observation_time_unix_s'][()]
        fields.append(1+RegularGridInterpolator((yy,xx),d)(coords).reshape(len(y),len(x)))
        masks.append(RegularGridInterpolator((yy,xx),(w>.005*w.max()).astype(float),method='nearest')(coords).reshape(len(y),len(x))>.5)
        times.append(RegularGridInterpolator((yy,xx),t)(coords).reshape(len(y),len(x)))
    speed=float(h['speed_m_s'][()])
# VTK order: x fastest, then cross-field upward, then along-B.
grid=pv.StructuredGrid();grid.dimensions=(len(x),len(y),len(s));grid.points=xyz.reshape(-1,3)
# Use explicit cell masking: unsupported samples never become closed cavity caps.
physical=(alt>=150)&(alt<=1100)
colors=[STORM[51],STORM[85],STORM[119]];levels=[.3,.5,.7];opacity=[.8,.35,.12]
def meshes(field,mask):
    g=grid.copy();valid=np.broadcast_to(mask,physical.shape)&physical;g['ratio']=np.broadcast_to(field,physical.shape).ravel();g['valid']=valid.ravel().astype(float)
    cells=g.threshold([.999,1.001],scalars='valid',preference='point',all_scalars=True)
    return [cells.contour([v],scalars='ratio').smooth(n_iter=8,relaxation_factor=.08) for v in levels]
def setup(plot):
    plot.set_background('white');plot.enable_depth_peeling(number_of_peels=12,occlusion_ratio=0)
    plot.add_axes(line_width=2,labels_off=False);plot.camera_position=[(1600,-1800,1400),(0,0,450),(0,0,1)]
    plot.enable_anti_aliasing('ssaa')
    plot.show_bounds(bounds=(-800,800,-700,700,150,1100),xtitle='East (km)',ytitle='North (km)',ztitle='Height (km)',font_size=11,color='#333333',grid=False,location='outer')
def add(plot,field,mask):
    for mesh,c,op in zip(meshes(field,mask),colors,opacity):
        if mesh.n_points:plot.add_mesh(mesh,color=c,opacity=op,smooth_shading=True,specular=.25,specular_power=20)
    # The measured cross-field plane provides context and an honest section cut.
    mid=len(s)//2;plane=pv.StructuredGrid();plane.dimensions=(len(x),len(y),1);plane.points=xyz[mid].reshape(-1,3);plane['ratio']=np.where(mask,field,np.nan).ravel()
    plot.add_mesh(plane,scalars='ratio',clim=(0,1.5),cmap=STORM_MAP,opacity=.18,show_scalar_bar=False,nan_opacity=0)
for i,name in enumerate(['west','central','east']):
    plot=pv.Plotter(off_screen=True,window_size=(1400,1100));setup(plot);add(plot,fields[i],masks[i]);plot.add_text(name.capitalize(),position='upper_left',font_size=16,color='#222222')
    plot.add_text('Contours: 0.3 / 0.5 / 0.7 relative echo\nField-aligned extent prescribed',position='lower_left',font_size=10,color='#333333')
    plot.show(screenshot=str(out/f'{name}_volume.png'),auto_close=True)
# Side-by-side common support, same camera and scale.
plot=pv.Plotter(off_screen=True,shape=(1,3),window_size=(2400,900))
for i,name in enumerate(['west','central','east']):
    plot.subplot(0,i);setup(plot);add(plot,fields[i],common);plot.add_text(name.capitalize(),position='upper_left',font_size=16,color='#222222')
plot.link_views();plot.show(screenshot=str(out/'three_volumes.png'),auto_close=True)
# Time at each node is retained; progress is meridian phase, not a fake universal clock.
plot=pv.Plotter(off_screen=True,window_size=(1400,1100));plot.open_movie(str(out/'meridian_morph.mp4'),framerate=18,quality=8)
for frame in range(a.frames):
    phase=2*frame/(a.frames-1);left=min(int(phase),1);alpha=phase-left;field=(1-alpha)*fields[left]+alpha*fields[left+1]
    plot.clear();setup(plot);add(plot,field,common)
    plot.camera.Azimuth(-20+40*frame/(a.frames-1))
    label=['West → central','Central → east'][left]
    plot.add_text(f'{label}   {alpha:.0%}',position='upper_left',font_size=16,color='#222222')
    plot.add_text('Supported interpolation between independently fitted sections\nParallel extent prescribed; not a measured continuous evolution',position='lower_left',font_size=10,color='#333333')
    plot.write_frame()
plot.close()
with h5py.File(a.root/'render_metadata.h5','w') as h:
    h['relative_echo_isosurfaces']=levels;h['opacity']=opacity;h['prescribed_parallel_extent_km']=[-300,300];h['ordered_common_support']=common;h['morph_phase']=np.linspace(0,2,a.frames)
    h.attrs['visualization']='Curved IGRF coordinates; open isosurfaces clipped only by measured support and display extent. VTK depth peeling;8iteration surface-only smoothing. Linear interpolation between maps on common time-ordered support; no temporal velocity inferred from video playback.'
print('VOLUMETRIC_RENDER_COMPLETE',out,flush=True)
