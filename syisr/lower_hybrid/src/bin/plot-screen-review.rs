//! Scientific review of a candidate transient screen; provenance is HDF5-only.
use anyhow::{ensure,Result};
use chrono::{DateTime,Utc};
use hdf5::File;
use rustplotlib::*;
use std::path::Path;
fn main()->Result<()>{
    let a:Vec<_>=std::env::args().collect();ensure!(a.len()==3,"plot-screen-review REVIEW.h5 OUTPUT-STEM");
    let h=File::open(&a[1])?;let r=h.dataset("slant_range_km")?.read_raw::<f64>()?;let freq=h.dataset("frequency_hz")?.read_raw::<f64>()?;
    let nr=r.len();let nf=freq.len();let dr=r[1]-r[0];let df=(freq[1]-freq[0])/1000.;
    let re:Vec<_>=(0..=nr).map(|i|r[0]-dr/2.+i as f64*dr).collect();
    let fe:Vec<_>=(0..=nf).map(|i|freq[0]/1000.-df/2.+i as f64*df).collect();
    let mut before=h.dataset("before")?.read_raw::<f64>()?;let mut after=h.dataset("after")?.read_raw::<f64>()?;let mut difference=h.dataset("difference")?.read_raw::<f64>()?;
    let mut magnitudes:Vec<_>=before.iter().chain(&after).enumerate().filter(|(j,_)|(150. ..=1250.).contains(&r[(j/nf)%nr])).map(|(_,x)|x.abs()).collect();let k=magnitudes.len()*999/1000;
    let original_limit=(*magnitudes.select_nth_unstable_by(k,f64::total_cmp).1).max(1.);
    let exponent=original_limit.log10().floor() as i32;let unit=10_f64.powi(exponent);let limit=(original_limit/unit).ceil();
    for bank in [&mut before,&mut after,&mut difference] {for x in bank {*x/=unit;}}
    let mut f=Figure::new(IntendedUse::Custom{width_mm:290.,font_pt:10.});f.height_mm=245.;f.raster_dpi=150.;f.tight_export=false;
    let beam=h.attr("beam")?.read_scalar::<u64>()? as usize;let geo=h.dataset("beam_geometry")?.read_raw::<f64>()?;
    let start=h.attr("window_start_utc_s")?.read_scalar::<f64>()?;let end=h.attr("window_end_utc_s")?.read_scalar::<f64>()?;
    let stamp=|t:f64|DateTime::<Utc>::from_timestamp(t as i64,0).unwrap().format("%d %b %H:%M:%S").to_string();
    f.labels.push(Label::new(format!("Candidate-transient screening: {} km beam",geo[beam*7] as usize),145.,7.));
    f.labels.push(Label::new(format!("{}--{} UTC",stamp(start),stamp(end)),145.,15.));
    let scale=f.scale(Scale::new([-limit,limit],Normalization::Asinh{width:0.01},ColorMap::diverging())?);
    for (i,(label,values)) in [("Before screening",before),("After screening",after),("Before minus after",difference)].into_iter().enumerate(){
        let x=if i==1{170.}else{25.};let y=if i==2{143.}else{38.};
        let mut axes=Axes::new([fe[0],fe[nf]],[150.,1250.],"Doppler frequency (kHz)","Slant range (km)");axes.xticks=vec![-40.,-20.,0.,20.,40.];axes.yticks=vec![200.,600.,1000.];
        let mut p=Panel::new(Rect::new(x,y,98.,72.),axes);p.push(Layer::Pcolormesh(Field::new(fe.clone(),re.clone(),values,scale)?));f.panels.push(p);
        f.labels.push(Label::new(label,x+49.,y-7.));f.colorbars.push(Colorbar::new(Rect::new(x+103.,y,3.,72.),scale,format!(r"Signed PSD ($10^{{{exponent}}}$ ADU$^2$/Hz)"),vec![-limit,0.,limit]));
    }
    let mask=h.dataset("rejection_mask")?;let shape=mask.shape();let nt=shape[1];let values=mask.read_raw::<u8>()?.iter().map(|&x|x as f64).collect();
    let binary=f.scale(Scale::new([0.,1.],Normalization::Linear,ColorMap::diverging())?);
    let mut p=Panel::new(Rect::new(170.,143.,98.,72.),Axes::new([0.,nt as f64],[150.,1250.],"Seconds from integration start","Slant range (km)"));
    p.push(Layer::Pcolormesh(Field::new((0..=nt).map(|i|i as f64).collect(),re,values,binary)?));f.panels.push(p);
    f.labels.push(Label::new("Excluded time/range exposures (red)",219.,136.));
    let delay=h.attr("range_delay_correction_us")?.read_scalar::<f64>()?;
    f.labels.push(Label::new(format!("Range correction: provisional {delay:.0} us; same background in before/after panels"),145.,233.));
    f.labels.push(Label::new("Morphology screening does not independently identify a satellite",145.,241.));
    let out=Path::new(&a[2]);f.save(out.with_extension("png"))?;f.save(out.with_extension("svg"))?;Ok(())
}
