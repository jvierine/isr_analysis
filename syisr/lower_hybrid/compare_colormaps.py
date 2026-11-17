"""Matched colour-map movies with measured-power excess normalization.

All scientific spectra, signed residuals, screening and exposure remain intact.
Each variant contains 250--550 km (5 min) then 650 km (30 min), both sessions.
"""
import argparse
from concurrent.futures import ThreadPoolExecutor,as_completed
import hashlib
import json
from pathlib import Path
import shutil
import subprocess
import sys
import tempfile
import h5py
import numpy as np

MAPS=['plasma','inferno','nipy_spectral']

def sha(path):
    h=hashlib.sha256()
    with Path(path).open('rb') as f:
        while chunk:=f.read(8*1024*1024):h.update(chunk)
    return h.hexdigest()


def prepare(source,target,beams):
    target.parent.mkdir(parents=True,exist_ok=True);shutil.copy2(source,target)
    with h5py.File(target,'r+') as h:
        df=float(h.attrs['frequency_bin_hz']);report=[]
        for b in range(5):
            if f'beam_{b}' not in h:continue
            if b not in beams:del h[f'beam_{b}'];continue
            g=h[f'beam_{b}'];received=g['psd_adu2_per_hz'][:];excess=g['background_subtracted_psd_adu2_per_hz'][:]
            power=received.sum(axis=-1)*df;assert np.all(np.isfinite(power)&(power>0))
            normalized=excess/power[:,:,None]
            np.testing.assert_array_equal(np.sign(normalized),np.sign(excess))
            assert np.isfinite(normalized).all()
            np.testing.assert_allclose(normalized.sum(-1)*df,excess.sum(-1)*df/power,rtol=1e-11,atol=1e-15)
            for name,values in [('received_power_normalized_excess_per_hz',normalized),('normalization_received_power_adu2',power)]:
                if name in g:del g[name]
                g.create_dataset(name,data=values,compression='gzip',compression_opts=3)
            # New data are side-by-side with the historical normalizations.
            # The source PSDs and residuals are never clipped or overwritten.
            np.testing.assert_array_equal(received,g['psd_adu2_per_hz'][:]);np.testing.assert_array_equal(excess,g['background_subtracted_psd_adu2_per_hz'][:])
            old_area=abs(excess).sum(-1)*df
            report.append(dict(beam=b,minimum_positive_denominator=float(power.min()),median_new_to_old_L1_amplitude=float(np.median(old_area/power))))
        h.attrs['display_normalization']='signed excess / integrated positive total received PSD before background subtraction; weak/noise-only spectra not forced to unit strength; integral S/(S+B); negative estimates retained'
        h.attrs['display_normalization_dataset']='received_power_normalized_excess_per_hz'
        h.attrs['display_normalization_program']='isr-range-doppler';h.attrs['display_normalization_version']='0.2.0'
        h.attrs['display_normalization_git_commit']=subprocess.check_output(['git','log','-1','--format=%H','--',':(top)crates/range-doppler/src/lib.rs'],cwd=Path(__file__).parent,text=True).strip()
        h.attrs['display_normalization_audit_json']=json.dumps(report)
        h.attrs['display_parent_product']=str(source.resolve());h.attrs['display_parent_sha256']=sha(source)
        render_commit=subprocess.check_output(['git','log','-1','--format=%H','--',':(top)syisr/lower_hybrid/src/bin/render.rs'],cwd=Path(__file__).parent,text=True).strip()
        chain=json.loads(h.attrs.get('processing_chain_json','[]'));chain=[c for c in chain if c['program']!='syisr-rd-render']
        chain.append(dict(program='isr-range-doppler',version='0.2.0',git_commit=h.attrs['display_normalization_git_commit'],stage='display normalization'))
        render_version=subprocess.check_output([str(Path(__file__).parent/'target/release/render'),'--version'],text=True).strip()
        chain.append(dict(program='syisr-rd-render',version=render_version,git_commit=render_commit))
        h.attrs['processing_chain_json']=json.dumps(chain);h.attrs['renderer_version']=render_version;h.attrs['renderer_git_commit']=render_commit
        h.attrs['renderer_source_sha256']=sha(Path(__file__).parent/'src/bin/render.rs');h.attrs['final_renderer_binary_sha256']=sha(Path(__file__).parent/'target/release/render')
        h.attrs['display_colormap']='per-movie manifest: plasma, inferno, nipy_spectral'
        # Draw all maps with exactly the same colour normalization and limits.
        h.attrs['fixed_signed_absolute_limit']=15.;h.attrs['fixed_signed_normalized_limit']=2.
    print('Verified received-power normalization:',target,flush=True)


def probe(path):
    return json.loads(subprocess.check_output(['ffprobe','-v','error','-select_streams','v:0','-show_entries','stream=width,height,nb_frames,duration','-of','json',str(path)],text=True))['streams'][0]


def join(paths,output):
    infos=[probe(p) for p in paths];duration=sum(float(i['duration']) for i in infos);frames=sum(int(i['nb_frames']) for i in infos)
    with tempfile.NamedTemporaryFile(mode='w',suffix='.txt') as listing:
        for path in paths:listing.write("file '"+str(path.resolve()).replace("'","'\\''")+"'\n")
        listing.flush();subprocess.run(['ffmpeg','-y','-v','error','-f','concat','-safe','0','-i',listing.name,'-c','copy','-movflags','+faststart',str(output)],check=True)
    info=probe(output);assert info['width']==1920 and info['height']==1080 and int(info['nb_frames'])==frames and abs(float(info['duration'])-duration)<.05
    subprocess.run(['ffmpeg','-v','error','-i',str(output),'-f','null','-'],check=True)
    return duration,frames


def main():
    p=argparse.ArgumentParser();p.add_argument('five_minute_root',type=Path);p.add_argument('thirty_minute_root',type=Path);p.add_argument('destination',type=Path);p.add_argument('--workers',type=int,default=4);q=p.parse_args()
    q.destination.mkdir(parents=True,exist_ok=True);tool=Path(__file__).resolve().parent
    products={}
    for session in ['190252','205701']:
        for family,source,beams in [('five',q.five_minute_root/f'session_{session}/restart_registered/range-doppler.h5',[0,1,2,3]),('thirty',q.thirty_minute_root/f'session_{session}/range-doppler.h5',[4])]:
            target=q.destination/f'products/{family}/session_{session}.h5';prepare(source,target,beams);products[(family,session)]=target
    jobs=[(color,b,session) for color in MAPS for b in range(5) for session in ['190252','205701']]
    def render(job):
        color,b,session=job;family='thirty' if b==4 else 'five';out=q.destination/f'{color}/beam_{250+b*100}km/session_{session}';out.mkdir(parents=True,exist_ok=True)
        with (out/'driver.log').open('w') as log:
            subprocess.run([sys.executable,str(tool/'make_movies.py'),str(products[(family,session)]),str(out),'--beam',str(b),'--colormap',color],stdout=log,stderr=subprocess.STDOUT,check=True)
        return job
    with ThreadPoolExecutor(max_workers=q.workers) as pool:
        futures=[pool.submit(render,job) for job in jobs]
        for number,future in enumerate(as_completed(futures),1):print('Completed',number,'/',len(jobs),future.result(),flush=True)
    variants=[];segments=[];manifests=[]
    for color in MAPS:
        paths=[]
        for b in range(5):
            for session in ['190252','205701']:
                manifest=q.destination/f'{color}/beam_{250+b*100}km/session_{session}/movies.h5';manifests.append(manifest)
                with h5py.File(manifest) as m:
                    assert sha(m.attrs['source_product'])==m.attrs['source_sha256']
                    info=json.loads(m.attrs['display_scales_json']);assert info['color_map']==color
                    g=m[f'beam_{250+b*100}km'];movie=Path(g.attrs['movie_path']);assert sha(movie)==g.attrs['sha256'];paths.append(movie)
        output=q.destination/f'syisr_all_beams_{color}.mp4';duration,frames=join(paths,output)
        assert abs(duration-186.5)<.05 and frames==4476
        variants.append(output);segments.append(duration)
        print('Verified all-beam variant',color,duration,'seconds',flush=True)
    output=q.destination/'syisr_all_beams_all_colormaps.mp4';duration,frames=join(variants,output)
    with h5py.File(q.destination/'comparison.h5','w') as h:
        h.attrs['movie_path']=str(output.resolve());h.attrs['sha256']=sha(output);h.attrs['duration_s']=duration;h.attrs['encoded_frames']=frames
        h.attrs['normalization']='excess divided by total received power; signed residuals preserved; same limits and asinh transforms for all variants'
        h.attrs['scientific_frames_per_second']=8.;h.attrs['nominal_altitudes_km']=[250,350,450,550,650];h.attrs['integration_window_s_by_beam']=[300.,300.,300.,300.,1800.]
        h.attrs['step_s_by_beam']=[150.,150.,150.,150.,300.]
        h['colormaps']=np.asarray(MAPS,dtype=h5py.string_dtype());h['colormap_segment_start_s']=np.r_[0.,np.cumsum(segments)[:-1]];h['colormap_segment_end_s']=np.cumsum(segments)
        h['variant_movies']=np.asarray([str(p) for p in variants],dtype=h5py.string_dtype());h['variant_sha256']=np.asarray([sha(p) for p in variants],dtype=h5py.string_dtype())
        h['source_manifests']=np.asarray([str(p) for p in manifests],dtype=h5py.string_dtype());h['source_manifest_sha256']=np.asarray([sha(p) for p in manifests],dtype=h5py.string_dtype())
    qa=q.destination/'qa';qa.mkdir(exist_ok=True)
    # Same 650 km time in each colour-map segment; source sessions are retained.
    for color,variant in zip(MAPS,variants):
        subprocess.run(['ffmpeg','-y','-v','error','-ss','176.8','-i',str(variant),'-frames:v','1',str(qa/f'{color}-650km.png')],check=True)
    print('Complete verified comparison:',output,duration,'seconds',flush=True)


if __name__=='__main__':main()
