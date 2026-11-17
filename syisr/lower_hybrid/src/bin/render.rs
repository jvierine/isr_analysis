//! Fixed-scale, fixed-canvas scientific animation frames from measured HDF5.
//! Generator and input provenance stay in code/data, never in the pictures.
use anyhow::{ensure,Result};
use chrono::{DateTime,Utc};
use hdf5::File;
use rustplotlib::*;
use std::{fs,path::Path};
const RENDER_VERSION:&str="0.2.1";

fn stamp(t:f64)->String {DateTime::<Utc>::from_timestamp(t.floor() as i64,0).unwrap().format("%H:%M:%S").to_string()}
fn quantile(v:&mut [f64],p:f64)->f64 {let k=((v.len()-1) as f64*p) as usize;*v.select_nth_unstable_by(k,f64::total_cmp).1}
fn ticks(limits:[f64;2])->Vec<f64> {
    let mut t=vec![limits[0],-100.,-10.,-1.,-0.1,-0.01,0.,0.01,0.1,1.,10.,100.,limits[1]];
    t.retain(|v|*v>=limits[0] && *v<=limits[1]);t.sort_by(f64::total_cmp);t.dedup();
    let span=(limits[1]/0.01).asinh()-(limits[0]/0.01).asinh();
    t.retain(|&v|v==limits[0] || v==limits[1] ||
        ((v/0.01).asinh()-(limits[0]/0.01).asinh()).abs().min(((v/0.01).asinh()-(limits[1]/0.01).asinh()).abs())>0.05*span);
    t
}

fn main()->Result<()> {
    let a:Vec<_>=std::env::args().collect();
    ensure!((3..=6).contains(&a.len()),"render SPECTRA.h5 OUTPUT-DIR [first-only] [beam=N] [frame=N]");
    let first_only=a[3..].iter().any(|s|s=="first-only");
    let beam_only=a[3..].iter().find_map(|s|s.strip_prefix("beam=").map(|s|s.parse::<usize>().unwrap()));
    let frame_only=a[3..].iter().find_map(|s|s.strip_prefix("frame=").map(|s|s.parse::<usize>().unwrap()));
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
        let norm=g.dataset("normalized_signal_psd_per_hz")?.read_raw::<f64>()?;
        let valid=g.dataset("normalization_display_valid")?.read_raw::<u8>()?;
        let norm:Vec<_>=norm.into_iter().enumerate().map(|(i,x)|if valid[i/nf]!=0{x*1e4}else{f64::NAN}).collect();
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
    let correction=h.attr("range_delay_correction_us").ok().and_then(|a|a.read_scalar::<f64>().ok()).unwrap_or(0.);
    let background=h.attr("background_range_km")?.read_raw::<f64>()?;
    let abs_limits=[-absolute,absolute];let norm_limits=[-normal,normal];
    println!("Shared fixed color limits: raw {abs_limits:?}, normalized {norm_limits:?}");
    fs::write(out.join("display-scales.json"),serde_json::to_vec_pretty(&serde_json::json!({
        "signed_absolute_adu2_per_hz":abs_limits,"signed_normalized_1e4_per_hz":norm_limits,
        "method":"fixed global 99.75 percentile magnitude, rounded up; signed asinh width 0.01 in displayed units; signed data unchanged",
        "color_map":"plasma","render_version":RENDER_VERSION,"canvas_pixels":[1920,1080],"frame_rate_fps":2,
    }))?)?;
    let dr=range[1]-range[0];let df=(freq[1]-freq[0])/1000.;
    let re:Vec<_>=(0..=nr).map(|i|range[0]-dr/2.+i as f64*dr).collect();
    let fe:Vec<_>=(0..=nf).map(|i|freq[0]/1000.-df/2.+i as f64*df).collect();
    for b in 0..5 {
        if beam_only.is_some_and(|j|j!=b) {continue;}
        let geo=&geometry[b*7..(b+1)*7];let altitude=geo[0] as usize;
        let directory=out.join(format!("beam_{altitude}km"));fs::create_dir_all(&directory)?;
        let counts=h.dataset(&format!("beam_{b}/pulse_count"))?.read_raw::<u32>()?;
        for frame in 0..if first_only{1}else{starts.len()} {
            if frame_only.is_some_and(|j|j!=frame) {continue;}
            let mut f=Figure::new(IntendedUse::Custom{width_mm:304.8,font_pt:10.});
            f.height_mm=171.45;f.raster_dpi=160.;f.tight_export=false;
            f.labels.push(Label::new(format!("SYISR: perpendicular-to-B beam near {altitude} km altitude"),152.4,7.));
            f.labels.push(Label::new(format!("Azimuth {:.3} deg; elevation {:.3} deg; perpendicular range {:.1} km",geo[1],geo[2],geo[3]),152.4,15.));
            let start_date=DateTime::<Utc>::from_timestamp(starts[frame] as i64,0).unwrap();
            let end_date=DateTime::<Utc>::from_timestamp(ends[frame] as i64,0).unwrap();
            let date=if start_date.date_naive()==end_date.date_naive(){start_date.format("%d %B %Y").to_string()}else{format!("{}--{}",start_date.format("%d %b %Y"),end_date.format("%d %b %Y"))};
            f.labels.push(Label::new(format!("{date}, {}--{} UTC; {:.0} s window; {} pulses",stamp(starts[frame]),stamp(ends[frame]),ends[frame]-starts[frame],counts[frame]),152.4,23.));
            for normalized in [false,true] {
                let x=if normalized{174.}else{24.};
                let limits=if normalized{norm_limits}else{abs_limits};
                let scale=f.scale(Scale::new(limits,Normalization::Asinh{width:0.01},ColorMap::plasma())?);
                let mut axes=Axes::new([fe[0],fe[nf]],[150.,1250.],"Doppler frequency (kHz)",if normalized{""}else{"Slant range (km)"});
                axes.xticks=vec![-40.,-20.,0.,20.,40.];axes.yticks=vec![200.,400.,600.,800.,1000.,1200.];
                if normalized {axes.yticks.clear();}
                let mut p=Panel::new(Rect::new(x,39.,108.,105.),axes);
                let mut backing=Line::new(vec![[fe[0],150.],[fe[nf],150.],[fe[nf],1250.],[fe[0],1250.],[fe[0],150.]],"#b8bdc1");
                backing.fill=Some("#b8bdc1".into());backing.fill_opacity=1.;backing.width_pt=0.;p.push(Layer::Line(backing));
                let bank=if normalized{&norm_banks[b]}else{&raw_banks[b]};
                p.push(Layer::Pcolormesh(Field::new(fe.clone(),re.clone(),bank[frame*plane..(frame+1)*plane].to_vec(),scale)?));
                let mut line=Line::new(vec![[fe[0],geo[3]],[fe[nf],geo[3]]],"#69747a");
                line.width_pt=0.7;line.opacity=0.5;line.dash=Some("2.0 1.5".into());p.push(Layer::Line(line));
                f.panels.push(p);
                f.labels.push(Label::new(if normalized{"Equal integrated signal power"}else{"Screened, background-subtracted spectrum"},x+54.,32.));
                let label=if normalized{r"Normalized PSD ($10^{-4}$ Hz$^{-1}$)"}else{r"Signed PSD (ADU$^2$/Hz)"};
                f.colorbars.push(Colorbar::new(Rect::new(x+111.,39.,3.,105.),scale,label,ticks(limits)));
            }
            let footer=if correction>0. {format!("PROVISIONAL {correction:.0} us delay; background: {:.0}--{:.0} km",background[0],background[1])}else{format!("Background: quiet gates at {:.0}--{:.0} km; signed asinh colour scale",background[0],background[1])};
            f.labels.push(Label::new(footer,152.4,158.5));
            f.labels.push(Label::new(r"Grey: normalization not used; dashed line: $\mathbf{k}\cdot\mathbf{B}=0$",152.4,166.));
            let stem=directory.join(format!("frame_{frame:04}"));
            f.save(stem.with_extension("png"))?;
            if frame==0 {f.save(out.join(format!("syisr_{altitude}km_preview.svg")))?;}
            println!("Rendered {altitude} km frame {frame}");
        }
    }
    Ok(())
}
