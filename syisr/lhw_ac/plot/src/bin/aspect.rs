//! Pencil-beam aspect sweep: intrinsic thermal theory and finite lag support.
use anyhow::Result;
use hdf5::File;
use rustplotlib::*;
use std::{env,path::Path};
fn main()->Result<()> {
 let a:Vec<_>=env::args().collect();let h=File::open(&a[1])?;let out=Path::new(&a[2]);
 let alpha=h.dataset("aspect_from_perpendicular_deg")?.read_raw::<f64>()?;let f=h.dataset("native_frequency_hz")?.read_raw::<f64>()?;let s=h.dataset("native_psd_per_hz")?.read_raw::<f64>()?;let fw=h.dataset("windowed_frequency_hz")?.read_raw::<f64>()?;let sw=h.dataset("triangular_psd_per_hz")?.read_raw::<f64>()?;
 let nf=f.len();let na=alpha.len();let mut fig=Figure::new(IntendedUse::Custom{width_mm:210.,font_pt:9.});fig.height_mm=176.;fig.tight_export=false;
 fig.labels.push(Label::new("430 MHz | B=40 uT | Te=2000 K, Ti=1000 K | O+/H+ = 99/1 | pencil beam",105.,6.));
 let scale=fig.scale(Scale::new([-10.,-3.],Normalization::Linear,ColorMap::inferno())?);
 let ye:Vec<_>=(0..=na).map(|i|-0.025+i as f64*0.05).collect();
 for col in 0..2 {
  let (xe,values)=if col==0 {
   let xe:Vec<_>=(0..=500).map(|i|i as f64*0.1).collect();let mut v=vec![];
   for j in 0..na{for bin in 0..500{let start=bin*100;let avg=s[j*nf+start..j*nf+start+100].iter().sum::<f64>()/100.;v.push(avg.log10());}}(xe,v)
  }else{let df=fw[1]-fw[0];let xe:Vec<_>=(0..=256).map(|i|(i as f64*df-df/2.)/1000.).collect();let sw_ref=&sw;let v=(0..na).flat_map(|j|(256..512).map(move|i|sw_ref[j*512+i].log10())).collect();(xe,v)};
  let mut axes=Axes::new([0.,50.],[0.,2.],"Frequency offset (kHz)",if col==0{"Angle from perpendicular (deg)"}else{""});axes.yticks=vec![0.,0.5,1.,1.5,2.];
  let mut panel=Panel::new(Rect::new(24.+col as f64*88.,21.,65.,53.),axes);panel.push(Layer::Pcolormesh(Field::new(xe,ye.clone(),values,scale)?));fig.panels.push(panel);
  fig.labels.push(Label::new(if col==0{"Intrinsic spectrum (100 Hz bins)"}else{"Triangular 0--620 us lag window"},56.5+col as f64*88.,15.));
 }
 fig.colorbars.push(Colorbar::new(Rect::new(181.,21.,3.,53.),scale,"Log10 PSD (Hz$^{-1}$)",vec![-10.,-8.,-6.,-4.]));
 let ids=[0,2,5,10,20,40];let colors=["#440154","#414487","#2a788e","#22a884","#7ad151","#d38e00"];
 for (j,(&i,color)) in ids.iter().zip(colors.iter()).enumerate(){let mut l=Label::new(format!("{:.2} deg",alpha[i]),29.+j as f64*30.,94.);l.color=(*color).into();fig.labels.push(l);}
 for col in 0..2 {
  let mut axes=Axes::new([0.,10.],[1e-7,2e-2],"Frequency offset (kHz)",if col==0{"PSD (Hz$^{-1}$)"}else{""});axes.set_yscale(AxisScale::Log10)?;
  let mut panel=Panel::new(Rect::new(24.+col as f64*88.,107.,65.,45.),axes);
  for (&i,color) in ids.iter().zip(colors.iter()){
   let points=if col==0{f.iter().enumerate().take_while(|(_,v)|**v<=10000.).map(|(j,&f)|[f/1000.,s[i*nf+j]]).collect()}else{fw.iter().enumerate().filter(|(_,v)|**v>=0.&&**v<=10000.).map(|(j,&f)|[f/1000.,sw[i*512+j]]).collect()};
   let mut line=Line::new(points,*color);line.width_pt=0.9;panel.push(Layer::Line(line));
  }fig.panels.push(panel);fig.labels.push(Label::new(if col==0{"Intrinsic line cuts"}else{"Windowed line cuts"},56.5+col as f64*88.,101.));
 }
 fig.labels.push(Label::new("Thermal Brownian collision model; no beam averaging or suprathermal enhancement",105.,170.));
 fig.save(out.join("ISR-aspect-sweep.png"))?;fig.save(out.join("ISR-aspect-sweep.pdf"))?;Ok(())
}
