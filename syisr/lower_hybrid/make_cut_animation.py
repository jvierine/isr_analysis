"""Inferno movie with two signed spectral cuts beside the range-Doppler image."""
import argparse
from concurrent.futures import ThreadPoolExecutor,as_completed
import json
from pathlib import Path
import shutil
import subprocess
import sys
import h5py
import numpy as np
from compare_colormaps import join,sha


def source_commit(tool,path):
    return subprocess.check_output(['git','log','-1','--format=%H','--',':(top)'+path],cwd=tool,text=True).strip()


def main():
    p=argparse.ArgumentParser();p.add_argument('source',type=Path);p.add_argument('destination',type=Path);p.add_argument('--workers',type=int,default=4);q=p.parse_args()
    tool=Path(__file__).resolve().parent;q.destination.mkdir(parents=True,exist_ok=True);products={}
    version=subprocess.check_output([str(tool/'target/release/render'),'--version'],text=True).strip()
    for family in ['five','thirty']:
        for session in ['190252','205701']:
            source=q.source/f'products/{family}/session_{session}.h5';target=q.destination/f'products/{family}/session_{session}.h5';target.parent.mkdir(parents=True,exist_ok=True)
            shutil.copy2(source,target);subprocess.run([str(tool/'target/release/export-cuts'),str(target)],check=True)
            with h5py.File(target,'r+') as h:
                h.attrs['cut_parent_product']=str(source.resolve());h.attrs['cut_parent_sha256']=sha(source)
                h.attrs['cut_export_git_commit']=source_commit(tool,'syisr/lower_hybrid/src/bin/export-cuts.rs')
                h.attrs['cut_export_binary_sha256']=sha(tool/'target/release/export-cuts')
                chain=json.loads(h.attrs['processing_chain_json']);chain=[c for c in chain if c['program']!='syisr-rd-render']
                chain.append(dict(program='syisr-rd-export-cuts',version='0.1.0',git_commit=h.attrs['cut_export_git_commit']))
                chain.append(dict(program='isr-range-doppler',version='0.3.0',git_commit=source_commit(tool,'crates/range-doppler/src/lib.rs'),stage='range interpolation'))
                chain.append(dict(program='syisr-rd-render',version=version,git_commit=source_commit(tool,'syisr/lower_hybrid/src/bin/render.rs')))
                h.attrs['processing_chain_json']=json.dumps(chain);h.attrs['renderer_version']=version;h.attrs['display_colormap']='inferno'
                for b in range(5):
                    if f'beam_{b}' not in h:continue
                    g=h[f'beam_{b}'];raw=g['background_subtracted_psd_adu2_per_hz'][:];r=h['slant_range_km'][:];targets=g['cut_slant_range_km'][:]
                    assert abs((targets[1]-targets[0])/0.149896229-900.)<1e-9
                    for i,name in enumerate(['perpendicular_cut_psd_adu2_per_hz','reference_cut_psd_adu2_per_hz']):
                        ids=g['cut_source_gate_indices'][i];w=g['cut_source_gate_weights'][i]
                        assert abs(w.sum()-1)<1e-14 and (w>=0).all()
                        assert abs(np.dot(w,r[ids])-targets[i])<1e-10
                        expected=raw[:,ids[0],:]*w[0]+raw[:,ids[1],:]*w[1]
                        np.testing.assert_allclose(g[name][:],expected,rtol=1e-13,atol=1e-14)
            products[(family,session)]=target
    linear=[];logarithmic=[];ranges=[]
    for b in range(5):
        paths=[products[('thirty' if b==4 else 'five',s)] for s in ['190252','205701']]
        arrays=[]
        for path in paths:
            with h5py.File(path) as h:
                g=h[f'beam_{b}'];arrays.extend([g['perpendicular_cut_psd_adu2_per_hz'][:].ravel(),g['reference_cut_psd_adu2_per_hz'][:].ravel()]);centres=g['cut_slant_range_km'][:]
        values=np.concatenate(arrays);assert np.isfinite(values).all();span=max(float(values.max()-values.min()),1e-12)
        limits=[min(0.,float(values.min()))-.05*span,max(0.,float(values.max()))+.05*span]
        positive=values[values>0];assert len(positive)>0
        loglimits=[10.**np.floor(np.log10(np.quantile(positive,.01))),10.**np.ceil(np.log10(positive.max()*1.05))]
        if loglimits[0]==loglimits[1]:loglimits[0]/=10
        for path in paths:
            with h5py.File(path,'r+') as h:
                g=h[f'beam_{b}'];perp=g['perpendicular_cut_psd_adu2_per_hz'][:];ref=g['reference_cut_psd_adu2_per_hz'][:]
                lo=np.minimum(0,np.minimum(perp.min(-1),ref.min(-1)));hi=np.maximum(0,np.maximum(perp.max(-1),ref.max(-1)));extent=np.maximum(hi-lo,1e-12)
                g['cut_linear_limits']=np.column_stack([lo-.05*extent,hi+.08*extent]);g['cut_log_limits']=loglimits
                g.attrs['cut_axes']='Log limits fixed across both sessions; linear limits cover the full signed extent of both curves in each frame. Logarithmic panel has gaps at nonpositive estimates, without floor or absolute value.'
        linear.append(limits);logarithmic.append(loglimits);ranges.append(centres)
        print('Verified exact cuts, common axes:',b,centres,limits,loglimits,flush=True)
    jobs=[(b,s) for b in range(5) for s in ['190252','205701']]
    def render(job):
        b,session=job;out=q.destination/f'beam_{250+b*100}km/session_{session}';out.mkdir(parents=True,exist_ok=True)
        with (out/'driver.log').open('w') as log:
            subprocess.run([sys.executable,str(tool/'make_movies.py'),str(products[('thirty' if b==4 else 'five',session)]),str(out),'--beam',str(b),'--colormap','inferno','--cuts'],stdout=log,stderr=subprocess.STDOUT,check=True)
        return job
    with ThreadPoolExecutor(max_workers=q.workers) as pool:
        futures=[pool.submit(render,j) for j in jobs]
        for n,future in enumerate(as_completed(futures),1):print('Rendered',n,'/',len(jobs),future.result(),flush=True)
    manifests=[];movies=[]
    for b,session in jobs:
        path=q.destination/f'beam_{250+b*100}km/session_{session}/movies.h5';manifests.append(path)
        with h5py.File(path) as m:
            assert sha(m.attrs['source_product'])==m.attrs['source_sha256']
            g=m[f'beam_{250+b*100}km'];movie=Path(g.attrs['movie_path']);assert sha(movie)==g.attrs['sha256'];movies.append(movie)
    output=q.destination/'syisr_all_beams_inferno_perpendicular_cuts.mp4';duration,frames=join(movies,output)
    assert abs(duration-186.5)<.05 and frames==4476
    with h5py.File(q.destination/'movie.h5','w') as h:
        h.attrs['movie_path']=str(output.resolve());h.attrs['sha256']=sha(output);h.attrs['duration_s']=duration;h.attrs['encoded_frames']=frames
        h.attrs['colormap']='inferno';h.attrs['reference_two_way_delay_offset_us']=900.;h.attrs['reference_slant_range_offset_km']=900*.149896229
        h.attrs['spectral_cut_units']='ADU^2/Hz, signed background-subtracted PSD; log panel shows positive estimates with gaps at nonpositive values; linear panel keeps signs'
        h.attrs['range_interpolation']='Linear interpolation between overlapping gate PSDs; exact requested target slant ranges; no improvement in physical range resolution'
        h.attrs['integration_window_s_by_beam']=[300,300,300,300,1800];h.attrs['step_s_by_beam']=[150,150,150,150,300]
        h['cut_slant_ranges_km']=ranges;h['linear_global_envelope_by_beam']=linear;h.attrs['linear_axes']='Both cuts share each frame axis; limits include all signed estimates. Log limits remain fixed across sessions.';h['log_y_limits_by_beam']=logarithmic
        h['source_manifests']=np.asarray([str(p) for p in manifests],dtype=h5py.string_dtype());h['source_manifest_sha256']=np.asarray([sha(p) for p in manifests],dtype=h5py.string_dtype())
    qa=q.destination/'qa';qa.mkdir(exist_ok=True)
    for name,t in [('250km-first',0.),('250km-later',9.8),('650km-first',167.),('650km-middle',176.8),('650km-last',186.375)]:
        subprocess.run(['ffmpeg','-y','-v','error','-ss',str(t),'-i',str(output),'-frames:v','1',str(qa/f'{name}.png')],check=True)
    print('New animation verified:',output,duration,'seconds',flush=True)


if __name__=='__main__':main()
