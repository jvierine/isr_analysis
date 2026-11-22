//! Raw IPP and range-response evidence; no generator annotations.
use anyhow::Result;
use chrono::{DateTime,Utc};
use hdf5::File;
use rustplotlib::*;
use std::{env,path::Path};
fn main()->Result<()> {
 let args:Vec<_>=env::args().collect();let h=File::open(&args[1])?;let out=Path::new(&args[2]);
 let g=h.group("raw_ipp")?;let t:Vec<_>=g.dataset("header_fast_time_us")?.read_raw::<f64>()?.iter().map(|v|v/1000.).collect();let v=g.dataset("voltage_i_q_adu")?.read_raw::<f64>()?;
 let power:Vec<_>=v.chunks_exact(2).map(|q|10.*(q[0]*q[0]+q[1]*q[1]).max(1.).log10()).collect();let lo=power.iter().copied().fold(f64::INFINITY,f64::min);let hi=power.iter().copied().fold(f64::NEG_INFINITY,f64::max);
 let utc=g.attr("utc_s")?.read_scalar::<f64>()?;let norad=g.attr("norad_id")?.read_scalar::<f64>()?;let clock=DateTime::<Utc>::from_timestamp(utc as i64,0).unwrap().format("%d %b %Y %H:%M:%S UTC");
 let mut f=Figure::new(IntendedUse::Custom{width_mm:180.,font_pt:9.});f.height_mm=142.;f.tight_export=false;
 f.labels.push(Label::new(format!("One 16 ms IPP | {clock} | NORAD {norad:.0}"),90.,7.));
 let mut p=Panel::new(Rect::new(24.,18.,145.,43.),Axes::new([0.,16.],[lo-3.,hi+3.],"Header-based fast time (ms)","Power (dB ADU$^2$)"));
 p.push(Layer::Line(Line::new(t.iter().zip(&power).map(|(&x,&y)|[x,y]).collect(),"#1f78b4")));
 for x in [0.667,0.987]{let mut l=Line::new(vec![[x,lo-3.],[x,hi+3.]],"#e66101");l.dash=Some("2 2".into());l.width_pt=0.7;p.push(Layer::Line(l));}f.panels.push(p);
 f.labels.push(Label::new("Unrecorded",155.,30.));
 for (i,xlim) in [[0.667,1.147],[3.50,4.50]].iter().enumerate(){
  let peak=t.iter().zip(v.chunks_exact(2)).filter(|(t,_)|**t>=xlim[0]&&**t<=xlim[1]).map(|(_,v)|v[0].abs().max(v[1].abs())/1e6).fold(1e-9,f64::max);
  let mut p=Panel::new(Rect::new(24.+i as f64*80.,86.,65.,35.),Axes::new(*xlim,[-peak*1.05,peak*1.05],"Header-based fast time (ms)",if i==0{"Voltage ($10^6$ ADU)"}else{""}));
  for (j,color) in ["#1f78b4","#e66101"].iter().enumerate(){let mut l=Line::new(t.iter().enumerate().map(|(k,&x)|[x,v[k*2+j]/1e6]).collect(),*color);l.width_pt=0.7;p.push(Layer::Line(l));}f.panels.push(p);
  f.labels.push(Label::new(if i==0{"Early code-bearing clutter"}else{"Satellite echo: I (blue), Q (orange)"},56.5+i as f64*80.,79.));
 }
 f.labels.push(Label::new("Only 0.667--13.337 ms is stored; orange lines mark the 32-sample exclusion",90.,137.));
 f.save(out.join("raw-ipp.png"))?;f.save(out.join("raw-ipp.pdf"))?;
 let mut f=Figure::new(IntendedUse::Custom{width_mm:180.,font_pt:9.});f.height_mm=105.;f.tight_export=false;
 for (i,b) in [1,2].iter().enumerate(){
  let g=h.group(&format!("sim_{b}"))?;let z=g.dataset("altitude_km")?.read_raw::<f64>()?;let truth=g.attr("truth_altitude_km")?.read_scalar::<f64>()?;
  let mut p=Panel::new(Rect::new(24.+i as f64*80.,25.,65.,56.),Axes::new([truth-15.,truth+15.],[0.,1.1],"Altitude (km)",if i==0{"ACF magnitude / target power"}else{""}));
  for (key,color) in [("plain","#1f78b4"),("excluded","#e66101"),("clutter_excluded","#33a02c"),("four_gate","#984ea3")]{
   let v=g.dataset(key)?.read_raw::<f64>()?;let xx=if key=="four_gate"{g.dataset("smoothed_altitude_km")?.read_raw::<f64>()?}else{z.clone()};let mut l=Line::new(xx.iter().zip(&v).map(|(&x,&y)|[x,y]).collect(),color);l.width_pt=1.;if key=="clutter_excluded"{l.dash=Some("2 2".into());}p.push(Layer::Line(l));
  }
  let mut l=Line::new(vec![[truth,0.],[truth,1.1]],"#333333");l.dash=Some("1 2".into());l.width_pt=0.5;p.push(Layer::Line(l));f.panels.push(p);f.labels.push(Label::new(format!("Injected at {truth:.2} km"),56.5+i as f64*80.,17.));
 }
 for (i,(text,color)) in [("No exclusion","#1f78b4"),("32 excluded","#e66101"),("Clutter + exclusion","#33a02c"),("Four-gate mean","#984ea3")].iter().enumerate(){let mut l=Label::new(*text,25.+i as f64*43.,7.);l.color=(*color).into();f.labels.push(l);}
 f.labels.push(Label::new("Coherent point target; production decoder; 20 us lag",90.,99.));f.save(out.join("point-target.png"))?;f.save(out.join("point-target.pdf"))?;
 let mut f=Figure::new(IntendedUse::Custom{width_mm:190.,font_pt:9.});f.height_mm=220.;f.tight_export=false;
 for b in 0..5{for case in 0..2{let g=h.group(&format!("sat_{case}_{b}"))?;let x=g.dataset("relative_range_km")?.read_raw::<f64>()?;let y=g.dataset("excluded")?.read_raw::<f64>()?;
  let mut p=Panel::new(Rect::new(24.+case as f64*86.,20.+b as f64*37.,70.,26.),Axes::new([-25.,25.],[0.,1.1],if b==4{"Range minus raw code-match range (km)"}else{""},if case==0{"ACF magnitude"}else{""}));
  for key in ["predicted_min_relative_km","predicted_max_relative_km"]{let q=g.attr(key)?.read_scalar::<f64>()?;let mut l=Line::new(vec![[q,0.],[q,1.1]],"#bbbbbb");l.width_pt=0.5;p.push(Layer::Line(l));}
  let mut l=Line::new(vec![[0.,0.],[0.,1.1]],"#333333");l.dash=Some("2 2".into());l.width_pt=0.6;p.push(Layer::Line(l));p.push(Layer::Line(Line::new(x.iter().zip(&y).map(|(&x,&y)|[x,y]).collect(),"#1f78b4")));f.panels.push(p);
  f.labels.push(Label::new(format!("{} km pointing, crossing {}",150+50*b,case+1),59.+case as f64*86.,15.+b as f64*37.));
 }}
 f.labels.push(Label::new("Dashed: raw code match; grey: orbit range span during the code cycle",95.,215.));f.save(out.join("decoded-satellites.png"))?;f.save(out.join("decoded-satellites.pdf"))?;
 Ok(())
}
