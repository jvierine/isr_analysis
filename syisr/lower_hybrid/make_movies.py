"""Render fixed scientific frames, encode five H.264 movies, and verify decoding."""
import argparse
from concurrent.futures import ThreadPoolExecutor
from datetime import datetime,timezone
import hashlib
import json
from pathlib import Path
import shutil
import subprocess

import h5py
import numpy as np


def text_attr(group,key,value):group.attrs[key]=str(value)


def main():
    p=argparse.ArgumentParser();p.add_argument('spectra',type=Path);p.add_argument('output',type=Path)
    p.add_argument('--skip-render',action='store_true');p.add_argument('--png-frames',action='store_true');p.add_argument('--beam',type=int,choices=range(5));q=p.parse_args()
    tool=Path(__file__).resolve().parent;renderer=tool/'target/release/render'
    q.output.mkdir(parents=True,exist_ok=True);figures=q.output/'figures'
    with h5py.File(q.spectra) as h:
        assert 'background_subtraction' in h.attrs
        n=len(h['window_start_utc_s']);date=datetime.fromtimestamp(h['window_start_utc_s'][0],timezone.utc).strftime('%Y%m%d')
        geometry=h['beam_geometry'][:]
        window=float(h.attrs.get('integration_window_s',60.));step=float(h.attrs.get('integration_step_s',30.))
        beams=[b for b in range(5) if f'beam_{b}' in h and (q.beam is None or q.beam==b)]
        assert beams,'No selected beam products'
    def render(b):
        log=q.output/f'render-beam-{b}.log'
        with log.open('w') as stream:
            command=[str(renderer),str(q.spectra),str(figures),f'beam={b}']
            if not q.png_frames:command.append(f'movie={q.output/f"syisr_{date}_{int(geometry[b,0])}km.mp4"}')
            subprocess.run(command,stdout=stream,stderr=subprocess.STDOUT,check=True)
        print('Rendered',int(geometry[b,0]),'km',flush=True)
    if not q.skip_render:
        with ThreadPoolExecutor(max_workers=3) as pool:list(pool.map(render,beams))
    records=[]
    for b in beams:
        row=geometry[b]
        altitude=int(row[0]);frames=figures/f'beam_{altitude}km'
        movie=q.output/f'syisr_{date}_{altitude}km.mp4'
        command=['ffmpeg','-y','-v','error','-framerate','8','-i',str(frames/'frame_%04d.png'),
                 '-vf','fps=24,format=yuv420p','-c:v','libx264','-preset','slow','-crf','18',
                 '-movflags','+faststart','-metadata',f'title=SYISR {altitude} km beam, background-subtracted range-Doppler spectra',str(movie)]
        if q.png_frames:
            assert len(list(frames.glob('frame_*.png')))==n
            subprocess.run(command,check=True)
        info=json.loads(subprocess.check_output(['ffprobe','-v','error','-select_streams','v:0',
                        '-show_entries','stream=codec_name,width,height,nb_frames,r_frame_rate,duration',
                        '-of','json',str(movie)],text=True))['streams'][0]
        assert info['codec_name']=='h264' and info['width']==1920 and info['height']==1080
        assert abs(float(info['duration'])-n/8)<0.05
        assert int(info['nb_frames'])==n*3
        subprocess.run(['ffmpeg','-v','error','-i',str(movie),'-f','null','-'],check=True)
        qa=q.output/'qa';qa.mkdir(exist_ok=True)
        for label,second in [('first',0.),('middle',n/16),('last',max(0.,n/8-0.125))]:
            subprocess.run(['ffmpeg','-y','-v','error','-ss',str(second),'-i',str(movie),'-frames:v','1',str(qa/f'{altitude}km-{label}.png')],check=True)
        shutil.copy2(qa/f'{altitude}km-first.png',q.output/f'syisr_{altitude}km_preview.png')
        records.append((altitude,movie,info,hashlib.sha256(movie.read_bytes()).hexdigest()))
        print('Encoded and decoded',movie.name,info['duration'],'s',flush=True)
    with h5py.File(q.output/'movies.h5','w') as out,h5py.File(q.spectra) as data:
        out['beam_geometry']=geometry
        out['window_start_utc_s']=data['window_start_utc_s'][:]
        out['window_end_utc_s']=data['window_end_utc_s'][:]
        out.attrs['source_product']=str(q.spectra.resolve())
        out.attrs['renderer_git_commit']=subprocess.check_output(['git','log','-1','--format=%H','--',':(top)syisr/lower_hybrid/src/bin/render.rs'],cwd=tool,text=True).strip()
        out.attrs['renderer_source_sha256']=hashlib.sha256((tool/'src/bin/render.rs').read_bytes()).hexdigest()
        out.attrs['renderer_binary_sha256']=hashlib.sha256(renderer.read_bytes()).hexdigest()
        out.attrs['source_sha256']=hashlib.sha256(q.spectra.read_bytes()).hexdigest()
        out.attrs['raw_sha256']=data.attrs['raw_sha256']
        out.attrs['background_subtraction']=data.attrs['background_subtraction']
        out.attrs['background_range_km']=data.attrs['background_range_km']
        out.attrs['display_scales_json']=(figures/'display-scales.json').read_text()
        out.attrs['frames_per_beam']=n;out.attrs['window_s']=window;out.attrs['nominal_step_s']=step
        out.attrs['input_frame_rate_fps']=8;out.attrs['encoded_frame_rate_fps']=24
        out.attrs['encoding_mode']='PNG sequence' if q.png_frames else 'raw RGBA frames streamed from Rustplotlib/resvg to ffmpeg stdin; no PNG sequence'
        out.attrs['movie_validation']='ffprobe dimensions, codec, frame count and duration; complete ffmpeg decode; decoded first/middle/last frames retained for review'
        for altitude,movie,info,sha in records:
            g=out.create_group(f'beam_{altitude}km')
            g.attrs['movie_path']=str(movie.resolve());g.attrs['sha256']=sha
            for k,v in info.items():text_attr(g,k,v)
    print('Selected movies and HDF5 manifest complete:',beams,flush=True)


if __name__=='__main__':main()
