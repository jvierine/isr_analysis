//! Raw satellite delay evidence; scientific labels only.
use anyhow::Result;
use hdf5::File;
use rustplotlib::*;
use std::{env,path::Path};
fn edges(v:&[f64])->Vec<f64>{(0..=v.len()).map(|i|if i==0{v[0]-(v[1]-v[0])/2.}else if i==v.len(){v[i-1]+(v[i-1]-v[i-2])/2.}else{(v[i-1]+v[i])/2.}).collect()}
fn main()->Result<()> {
 let a:Vec<_>=env::args().collect();let h=File::open(&a[1])?;let out=Path::new(&a[2]);
 let mut f=Figure::new(IntendedUse::Custom{width_mm:315.,font_pt:10.});f.height_mm=130.;f.tight_export=false;
 let scale=f.scale(Scale::new([0.7,5.],Normalization::Linear,ColorMap::inferno())?);
 for (i,target) in [150,200,250,300,350].iter().enumerate(){
  let g=h.group(&format!("beam_{target}"))?;let t=g.dataset("example_time_from_closest_s")?.read_raw::<f64>()?;let d=g.dataset("example_excess_delay_us")?.read_raw::<f64>()?;
  let raw=g.dataset("example_matched_power_ratio")?.read_raw::<f64>()?;let mut values=Vec::new();for j in 0..d.len(){for k in 0..t.len(){values.push(raw[k*d.len()+j].log10());}}
  let mut axes=Axes::new([-5.,5.],[-100.,800.],"Time from closest approach (s)",if i==0{"Excess two-way delay (us)"}else{""});axes.xticks=vec![-5.,0.,5.];axes.yticks=vec![0.,160.,320.,480.,640.,800.];
  let mut p=Panel::new(Rect::new(23.+i as f64*53.,21.,45.,86.),axes);
  p.push(Layer::Pcolormesh(Field::new(edges(&t),edges(&d),values,scale)?));
  for (v,color) in [(0.,"#33ccff"),(640.,"#33ccff")]{let mut line=Line::new(vec![[-5.,v],[5.,v]],color);line.width_pt=0.8;line.dash=Some("2 2".into());p.push(Layer::Line(line));}
  f.panels.push(p);
  let id=g.attr("example_norad_id")?.read_scalar::<f64>()?;
  f.labels.push(Label::new(format!("{target} km beam"),45.5+i as f64*53.,8.));
  f.labels.push(Label::new(format!("NORAD {id:.0}"),45.5+i as f64*53.,15.));
 }
 f.colorbars.push(Colorbar::new(Rect::new(286.,21.,3.,86.),scale,"Log10 matched power / median",vec![1.,2.,3.,4.,5.]));
 f.labels.push(Label::new("Zero = predicted light travel time; dashed lines: 0 and +640 us",150.,124.));
 f.save(out.join("satellite-delay-tracks.png"))?;f.save(out.join("satellite-delay-tracks.pdf"))?;
 let rows=h.dataset("selected_passes")?.read_raw::<f64>()?;
 let mut f=Figure::new(IntendedUse::Custom{width_mm:180.,font_pt:10.});f.height_mm=105.;f.tight_export=false;
 let mut axes=Axes::new([0.5,10.5],[140.,210.],"Confirmed crossing, ordered by closest angular approach","Excess two-way delay (us)");axes.xticks=(1..=10).map(|v|v as f64).collect();
 let mut p=Panel::new(Rect::new(24.,26.,143.,59.),axes);
 for (i,color) in ["#1f78b4","#33a02c","#e66101","#984ea3","#e7298a"].iter().enumerate(){
  let points=(0..10).map(|j|[(j+1) as f64,rows[(i*10+j)*9+5]]).collect();let mut line=Line::new(points,*color);line.width_pt=1.;p.push(Layer::Line(line));
  let mut label=Label::new(format!("{} km",150+50*i),32.+i as f64*31.,13.);label.color=(*color).into();f.labels.push(label);
 }
 f.panels.push(p);f.save(out.join("satellite-delay-summary.png"))?;f.save(out.join("satellite-delay-summary.pdf"))?;
 Ok(())
}
