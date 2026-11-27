"""Freeze available channel pairs in HDF5; all plotting remains Rustplotlib."""
import argparse,pathlib,os,shutil,subprocess,time
import h5py,numpy as np
p=argparse.ArgumentParser();p.add_argument('--plan',action='store_true');p.add_argument('--shard',type=int,default=0);p.add_argument('--shards',type=int,default=24);a=p.parse_args()
root=pathlib.Path('/gpfs/pub/data_other/juha/plasmaline-20261125/dual-channel-hour')
out=root/'movie-so-far';out.mkdir(exist_ok=True);manifest=out/'manifest.h5'
here=pathlib.Path(__file__).resolve().parent
if a.plan:
    times=[];counts=[];totals=[]
    for d in sorted(root.iterdir()):
        if not d.name.isdigit():continue
        paths=[d/str(c)/'matched-step2us.h5' for c in (436000000,444000000)]
        if not all(f.is_file() for f in paths):continue
        cs=[];ns=[];t=int(d.name)
        for center,path in zip((436000000,444000000),paths):
            with h5py.File(path) as f:
                assert f.attrs['receiver_center_hz']==center and f.attrs['range_step_us']==2
                assert f.attrs['integration_start_utc_s']==t and f.attrs['integration_stop_utc_s']==t+60
                n=int(f.attrs['accepted_pulses']);nt=int(f.attrs['total_pulses']);assert 0<n<=nt<=6001
                cs.append(n);ns.append(nt)
        times.append(t);counts.append(cs);totals.append(ns)
    assert times[:217]==list(range(1795564800,1795564800+217*60,60))
    tmp=manifest.with_suffix('.partial.h5')
    with h5py.File(tmp,'w') as f:
        f['integration_start_utc_s']=np.asarray(times,dtype='i8');f['accepted_pulses']=np.asarray(counts,dtype='i8');f['total_pulses']=np.asarray(totals,dtype='i8')
        f['gap_before_seconds']=np.r_[0,np.diff(times)-60].astype('i8')
        f.attrs.update(frame_count=len(times),data_minutes=len(times),wall_span_seconds=times[-1]+60-times[0],reused_prefix_frames=217,gaussian_sigma_frequency_bins=10.,gaussian_sigma_range_bins=10.,range_max_km=500.,color_min_db=-.1,color_max_db=.23735279602706477)
    tmp.replace(manifest);print('PLAN',len(times),'minutes',times[0],times[-1]+60,'missing minutes',sum(np.diff(times)-60)//60,flush=True)
    raise SystemExit
with h5py.File(manifest) as f:times=f['integration_start_utc_s'][:]
frames=out/'frames';frames.mkdir(exist_ok=True)
for i in list(range(217,len(times)))+list(range(217)):
    if i%a.shards!=a.shard:continue
    t=int(times[i]);target=frames/f'frame-{i:05}.png'
    if target.is_file():continue
    if i<217:
        source=root/'snapshot-0000-0337-f10-r10-range500-vminm0p1'/'frames'/f'frame-{i:05}.png'
        os.link(source,target);print('FRAME REUSED',i,t,flush=True);continue
    display=root/'gaussian-f10-r10'/f'minute-{t}'/'display';plots=out/f'minute-{t}'/'plots';plots.mkdir(parents=True,exist_ok=True)
    if not (display/'profile-01-native.h5').is_file():
        subprocess.run([str(here/'rust_extract/target/release/splice_channels'),str(root/str(t)),str(display),str(root/'movie/minute-1795564800/display/profile-01-native.h5'),'10','10'],check=True)
    subprocess.run([str(here/'rust_render/target/release/plasma-rustplotlib'),str(display),str(plots),'rf','video','500','-0.1'],check=True)
    tmp=target.with_suffix('.partial');shutil.copyfile(plots/'profile-01-rustplotlib-preview.png',tmp);tmp.replace(target);print('FRAME COMPLETE',i,t,flush=True)
if a.shard==0:
    while len(list(frames.glob('frame-*.png')))<len(times):time.sleep(5)
    # Transfer only the new tail; concatenate with the existing first 217 minutes locally.
    subprocess.run(['/opt/soft/source/ffmpeg-7.0.2-amd64-static/ffmpeg','-hide_banner','-loglevel','warning','-y','-framerate','4','-start_number','217','-i',str(frames/'frame-%05d.png'),'-vf','fps=24','-frames:v',str((len(times)-217)*6),'-c:v','libx264','-threads','8','-preset','fast','-crf','18','-pix_fmt','yuv420p','-movflags','+faststart',str(out/'tail-after-0337.mp4')],check=True)
    print('TAIL MOVIE COMPLETE',len(times),'available minutes',flush=True)
