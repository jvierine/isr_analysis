//! Fixed-scale, fixed-canvas scientific animation frames from measured HDF5.
//! Generator and input provenance stay in code/data, never in the pictures.
use anyhow::{ensure,Result};
use chrono::{DateTime,Utc};
use hdf5::File;
use rustplotlib::*;
use std::{fs,path::Path};
const RENDER_VERSION:&str="0.5.0";

fn stamp(t:f64)->String {DateTime::<Utc>::from_timestamp(t.floor() as i64,0).unwrap().format("%H:%M:%S").to_string()}
fn quantile(v:&mut [f64],p:f64)->f64 {let k=((v.len()-1) as f64*p) as usize;*v.select_nth_unstable_by(k,f64::total_cmp).1}
fn ticks(limits:[f64;2],width:f64)->Vec<f64> {
    let transform=|v:f64|(v/width).asinh();
    let span=transform(limits[1])-transform(limits[0]);
    let mut selected=vec![limits[0],limits[1]];
    if limits[0]<0. && limits[1]>0. {selected.push(0.);}
    for value in [0.001,0.01,0.1,1.,10.,100.] {
        if value>limits[0] && value<limits[1] && selected.iter().all(|&other|(transform(value)-transform(other)).abs()>0.07*span) {selected.push(value);}
    }
    selected.sort_by(f64::total_cmp);selected
}

fn main()->Result<()> {
    let a:Vec<_>=std::env::args().collect();
    ensure!((3..=8).contains(&a.len()),"render SPECTRA.h5 OUTPUT-DIR [first-only] [beam=N] [frame=N] [movie=PATH] [colormap=NAME]");
    let colormap=a[3..].iter().find_map(|s|s.strip_prefix("colormap=")).unwrap_or("plasma");
    let color_label=colormap.replace('_'," ");
    let colors=match colormap {"plasma"=>ColorMap::plasma(),"inferno"=>ColorMap::inferno(),"nipy_spectral"=>ColorMap::nipy_spectral(),_=>anyhow::bail!("Unsupported colormap: {colormap}")};
    let first_only=a[3..].iter().any(|s|s=="first-only");
    let beam_only=a[3..].iter().find_map(|s|s.strip_prefix("beam=").map(|s|s.parse::<usize>().unwrap()));
    let frame_only=a[3..].iter().find_map(|s|s.strip_prefix("frame=").map(|s|s.parse::<usize>().unwrap()));
    let movie=a[3..].iter().find_map(|s|s.strip_prefix("movie="));
    ensure!(movie.is_none() || beam_only.is_some(),"Movie output requires one beam");
    let h=File::open(&a[1])?;let out=Path::new(&a[2]);fs::create_dir_all(out)?;
    let starts=h.dataset("window_start_utc_s")?.read_raw::<f64>()?;
    let ends=h.dataset("window_end_utc_s")?.read_raw::<f64>()?;
    let range=h.dataset("slant_range_km")?.read_raw::<f64>()?;
    let freq=h.dataset("frequency_hz")?.read_raw::<f64>()?;
    let geometry=h.dataset("beam_geometry")?.read_raw::<f64>()?;
    let nr=range.len();let nf=freq.len();let plane=nr*nf;
    let fixed=h.attr_names()?.iter().any(|n|n=="fixed_signed_absolute_limit");
    let mut raw_banks=vec![Vec::new();5];let mut norm_banks=vec![Vec::new();5];
    let mut raw_scale=Vec::new();let mut norm_scale=Vec::new();
    for b in 0..5 {
        if fixed && beam_only.is_some_and(|j|j!=b) {continue;}
        let g=h.group(&format!("beam_{b}"))?;
        let raw=g.dataset("background_subtracted_psd_adu2_per_hz")?.read_raw::<f64>()?;
        let received=g.dataset("psd_adu2_per_hz")?.read_raw::<f64>()?;
        ensure!(received.len()==raw.len(),"Received/excess shape mismatch");
        let df_hz=freq[1]-freq[0];
        let mut norm=Vec::with_capacity(raw.len());
        for (row,total) in raw.chunks(nf).zip(received.chunks(nf)) {
            norm.extend(isr_range_doppler::normalize_excess_by_received_power(row,total,df_hz)?.into_iter().map(|x|x*1e4));
        }
        for frame in 0..if fixed{0}else{starts.len()} {
            for (r,&rr) in range.iter().enumerate() {
                if !(150. ..=1250.).contains(&rr) {continue;}
                raw_scale.extend(raw[frame*plane+r*nf..frame*plane+(r+1)*nf].iter().map(|x|x.abs()));
                norm_scale.extend(norm[frame*plane+r*nf..frame*plane+(r+1)*nf].iter().map(|x|x.abs()));
            }
        }
        raw_banks[b]=raw;norm_banks[b]=norm;
    }
    raw_scale.retain(|x|x.is_finite());norm_scale.retain(|x|x.is_finite());
    let absolute=h.attr("fixed_signed_absolute_limit").ok().and_then(|a|a.read_scalar::<f64>().ok()).unwrap_or_else(||(quantile(&mut raw_scale,0.9975)/5.).ceil()*5.);
    let normal=h.attr("fixed_signed_normalized_limit").ok().and_then(|a|a.read_scalar::<f64>().ok()).unwrap_or_else(||(quantile(&mut norm_scale,0.9975)*2.).ceil()/2.);
    let abs_limits=[-0.01,absolute];let norm_limits=[-0.01,normal];
    println!("Shared fixed color limits: raw {abs_limits:?}, normalized {norm_limits:?}");
    fs::write(out.join("display-scales.json"),serde_json::to_vec_pretty(&serde_json::json!({
        "signed_absolute_adu2_per_hz":abs_limits,"signed_normalized_1e4_per_hz":norm_limits,
        "method":"fixed global 99.75 percentile magnitude, rounded up; signed asinh width 0.01 in displayed units; signed data unchanged",
        "normalization":"signed background-subtracted PSD divided by positive full-band total received power before subtraction; excess integral S/(S+B), not unit area; no noise-only unit-strength normalization; signs preserved",
        "asinh_width_absolute":0.01,"asinh_width_normalized":0.001,
        "color_map":colormap,"render_version":RENDER_VERSION,"canvas_pixels":[1920,1080],"frame_rate_fps":8,
    }))?)?;
    let dr=range[1]-range[0];let df=(freq[1]-freq[0])/1000.;
    let re:Vec<_>=(0..=nr).map(|i|range[0]-dr/2.+i as f64*dr).collect();
    let fe:Vec<_>=(0..=nf).map(|i|freq[0]/1000.-df/2.+i as f64*df).collect();
    for b in 0..5 {
        if beam_only.is_some_and(|j|j!=b) {continue;}
        let geo=&geometry[b*7..(b+1)*7];let altitude=geo[0] as usize;
        let directory=out.join(format!("beam_{altitude}km"));fs::create_dir_all(&directory)?;
        let mut encoder=if let Some(path)=movie {
            Some(AnimationWriter::new(path,AnimationOptions {
                input_fps:8.,output_fps:24.,..Default::default()
            })?)
        }else{None};
        for frame in 0..if first_only{1}else{starts.len()} {
            if frame_only.is_some_and(|j|j!=frame) {continue;}
            let mut f=Figure::new(IntendedUse::Custom{width_mm:304.8,font_pt:10.});
            f.height_mm=171.45;f.raster_dpi=160.;f.tight_export=false;
            let start_date=DateTime::<Utc>::from_timestamp(starts[frame] as i64,0).unwrap();
            let end_date=DateTime::<Utc>::from_timestamp(ends[frame] as i64,0).unwrap();
            let date=if start_date.date_naive()==end_date.date_naive(){start_date.format("%d %b %Y").to_string()}else{format!("{}--{}",start_date.format("%d %b %Y"),end_date.format("%d %b %Y"))};
            let duration=ends[frame]-starts[frame];
            let integration=if duration>=60. && (duration/60.-(duration/60.).round()).abs()<1e-6 {format!("{:.0} min",duration/60.)}else{format!("{duration:.1} s")};
            f.labels.push(Label::new(format!("SYISR | {altitude} km target | {date} {}--{} UTC | {integration}",stamp(starts[frame]),stamp(ends[frame])),152.4,6.));
            for normalized in [false,true] {
                let x=if normalized{174.}else{24.};
                let limits=if normalized{norm_limits}else{abs_limits};
                let width=if normalized{0.001}else{0.01};
                let scale=f.scale(Scale::new(limits,Normalization::Asinh{width},colors.clone())?);
                let mut axes=Axes::new([fe[0],fe[nf]],[150.,1250.],"Doppler frequency (kHz)",if normalized{""}else{"Slant range (km)"});
                axes.xticks=vec![-40.,-20.,0.,20.,40.];axes.yticks=vec![200.,400.,600.,800.,1000.,1200.];
                if normalized {axes.yticks.clear();}
                let mut p=Panel::new(Rect::new(x,19.,108.,139.),axes);
                let bank=if normalized{&norm_banks[b]}else{&raw_banks[b]};
                p.push(Layer::Pcolormesh(Field::new(fe.clone(),re.clone(),bank[frame*plane..(frame+1)*plane].to_vec(),scale)?));
                let mut line=Line::new(vec![[fe[0],geo[3]],[fe[nf],geo[3]]],"#69747a");
                line.width_pt=0.7;line.opacity=0.5;line.dash=Some("2.0 1.5".into());p.push(Layer::Line(line));
                f.panels.push(p);
                f.labels.push(Label::new(if normalized{format!("Normalized excess ({color_label})")}else{"Background-subtracted spectrum".into()},x+54.,13.));
                let label=if normalized{r"Normalized PSD ($10^{-4}$ Hz$^{-1}$)"}else{r"Signed PSD (ADU$^2$/Hz)"};
                f.colorbars.push(Colorbar::new(Rect::new(x+111.,19.,3.,139.),scale,label,ticks(limits,width)));
            }
            let stem=directory.join(format!("frame_{frame:04}"));
            if let Some(writer)=encoder.as_mut() {writer.push(&f)?;}
            else{f.save(stem.with_extension("png"))?;}
            if frame==0 {f.save(out.join(format!("syisr_{altitude}km_preview.svg")))?;}
            println!("Rendered {altitude} km frame {frame}");
        }
        if let Some(writer)=encoder {writer.finish()?;}
    }
    Ok(())
}
