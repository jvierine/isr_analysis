//! Two satellite crossings per pointing; nominal gain versus raw echo diagnostics.
use anyhow::Result;
use hdf5::File;
use rustplotlib::*;
use std::{env,path::Path};
fn main()->Result<()> {
 let a:Vec<_>=env::args().collect();let h=File::open(&a[1])?;let out=Path::new(&a[2]);
 for target in [150,200,250,300,350]{
  let mut f=Figure::new(IntendedUse::Custom{width_mm:190.,font_pt:9.});f.height_mm=230.;f.tight_export=false;
  let first=h.group(&format!("beam_{target}/crossing_0"))?;let el=first.attr("beam_elevation_deg")?.read_scalar::<f64>()?;
  f.labels.push(Label::new(format!("{target} km pointing | az 357.995 deg, el {el:.3} deg"),95.,7.));
  for col in 0..2{
   let g=h.group(&format!("beam_{target}/crossing_{col}"))?;let t=g.dataset("time_from_closest_s")?.read_raw::<f64>()?;
   let id=g.attr("norad_id")?.read_scalar::<f64>()?;let stamp=g.attr("utc_label")?.read_scalar::<hdf5::types::VarLenUnicode>()?;
   let pred=g.attr("predicted_peak_s")?.read_scalar::<f64>()?;let echo=g.attr("echo_peak_s")?.read_scalar::<f64>()?;
   let x=25.+col as f64*86.;
   f.labels.push(Label::new(format!("NORAD {id:.0} | {stamp}"),x+35.,17.));
   let specs=[("predicted_gain_db",[-30.,1.],"Estimated two-way gain (dB)","#333333"),("snr_db",[-5.,85.],"Matched-filter SNR (dB)","#1f78b4"),("delay_residual_us",[-50.,50.],"Delay residual (us)","#1f78b4"),("doppler_residual_hz",[-500.,500.],"Doppler residual (Hz)","#1f78b4")];
   for (row,(key,lim,ylabel,color)) in specs.iter().enumerate(){
    let v=g.dataset(key)?.read_raw::<f64>()?;let mut axes=Axes::new([-8.,8.],*lim,if row==3{"Time from closest approach (s)"}else{""},if col==0{*ylabel}else{""});axes.xticks=vec![-8.,-4.,0.,4.,8.];
    let mut p=Panel::new(Rect::new(x,26.+row as f64*44.,70.,33.),axes);
    if row>=2{let mut zero=Line::new(vec![[-8.,0.],[8.,0.]],"#aaaaaa");zero.width_pt=0.5;p.push(Layer::Line(zero));}
    let mut line=Line::new(t.iter().zip(v.iter()).map(|(&t,&v)|[t,v]).collect(),*color);line.width_pt=1.;p.push(Layer::Line(line));
    for (time,color) in [(pred,"#777777"),(echo,"#e66101")]{let mut line=Line::new(vec![[time,lim[0]],[time,lim[1]]],color);line.width_pt=0.7;line.dash=Some("2 2".into());p.push(Layer::Line(line));}
    f.panels.push(p);
   }
   let da=g.attr("echo_peak_cross_az_deg")?.read_scalar::<f64>()?;let de=g.attr("echo_peak_el_deg")?.read_scalar::<f64>()?;
   f.labels.push(Label::new(format!("Echo peak minus gain peak: {:+.2} s",echo-pred),x+35.,205.));
   f.labels.push(Label::new(format!("At echo peak: cross-az {da:+.2}, el {de:+.2} deg"),x+35.,212.));
  }
  f.labels.push(Label::new("Dashed: nominal gain peak (grey), smoothed range-corrected echo peak (orange)",95.,218.));
  f.labels.push(Label::new("Gain is a Gaussian estimate; residuals show confirmed samples; delay correction 165.8 us",95.,224.));
  f.save(out.join(format!("beam-{target}-crossings.png")))?;f.save(out.join(format!("beam-{target}-crossings.pdf")))?;
 }
 Ok(())
}
