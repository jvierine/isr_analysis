//! Inspect signed measured complex ACFs; displayed errors are working SEMs.
use anyhow::{ensure,Result};
use rustplotlib::*;
fn main()->Result<()> {
 let a=std::env::args().collect::<Vec<_>>();
 ensure!(a.len()==3,"INPUT_H5 OUTPUT_PNG");
 let input=hdf5::File::open(&a[1])?;
 let time=input.attr("time_utc_s")?.read_scalar::<i64>()?;
 let mut beams=input.member_names()?.into_iter().map(|n|input.group(&n)).collect::<std::result::Result<Vec<_>,_>>()?;
 beams.sort_by(|a,b|b.attr("elevation_deg").unwrap().read_scalar::<f64>().unwrap().total_cmp(&a.attr("elevation_deg").unwrap().read_scalar::<f64>().unwrap()));
 let beam=&beams[0];
 let mut gates=beam.member_names()?.into_iter().map(|n|beam.group(&n)).collect::<std::result::Result<Vec<_>,_>>()?;
 gates.sort_by(|a,b|a.attr("altitude_km").unwrap().read_scalar::<f64>().unwrap().total_cmp(&b.attr("altitude_km").unwrap().read_scalar::<f64>().unwrap()));
 let mut f=Figure::new(IntendedUse::ArticleWide);f.width_mm=200.;f.height_mm=265.;f.font_pt=9.;
 let date=chrono::DateTime::<chrono::Utc>::from_timestamp(time+28800,0).unwrap();
 f.labels.push(Label::new(format!("SYISR zenith; {}; 30 minutes",date.format("%d %B %Y %H:%M China local time")),100.,5.));
 f.labels.push(Label::new("Measured signed ACF and nominal near-zero-H$^+$ fit",100.,11.));
 for(row,g)in gates.iter().enumerate(){
  let lag=g.dataset("lag_us")?.read_raw::<f64>()?;
  let obs=g.dataset("observed")?.read_raw::<f64>()?;
  let pred=g.dataset("predicted")?.read_raw::<f64>()?;
  let sigma=g.dataset("assigned_sigma")?.read_raw::<f64>()?;
  ensure!(!lag.is_empty()&&obs.len()==2*lag.len(),"malformed ACF");
  ensure!((lag[0]-10.).abs()<1e-9,"10us normalization lag absent");
  let norm=obs[0].abs().max(1e-30);
  let z=g.attr("altitude_km")?.read_scalar::<f64>()?;
  for part in 0..2{
   let data=lag.iter().enumerate().map(|(i,x)|[*x,obs[2*i+part]/norm]).collect::<Vec<_>>();
   let model=lag.iter().enumerate().map(|(i,x)|[*x,pred[2*i+part]/norm]).collect::<Vec<_>>();
   let errors=lag.iter().enumerate().map(|(i,_)|[(obs[2*i+part]-sigma[2*i+part])/norm,(obs[2*i+part]+sigma[2*i+part])/norm]).collect::<Vec<_>>();
   let mut lo=0f64;let mut hi=0f64;for e in &errors{lo=lo.min(e[0]);hi=hi.max(e[1]);}for p in &model{lo=lo.min(p[1]);hi=hi.max(p[1]);}let pad=(hi-lo).max(0.1)*0.07;
   let mut axes=Axes::new([0.,480.],[lo-pad,hi+pad],if row+1==gates.len(){r"Lag ($\mu$s)"}else{""},if part==0{"Real ACF"}else{"Imaginary ACF"});
   axes.xticks=vec![0.,100.,200.,300.,400.];if row+1<gates.len(){axes.show_x_axis=false;}
   let rect=Rect::new(20.+part as f64*96.,23.+row as f64*37.,76.,27.);
   let mut p=Panel::new(rect,axes);p.push(Layer::Line(Line::series(model,1)));
   p.errorbars(&data,None,Some(&errors),"black",0.5,0.7,0.4)?;
   p.markers(&mut f,&data,&MarkerStyle{shape:MarkerShape::Circle,size_pt:1.6,color:[0.,0.,0.]})?;
   f.panels.push(p);f.labels.push(Label::new(format!("{z:.0} km"),rect.x+rect.width/2.,rect.y-3.));
  }
 }
 f.labels.push(Label::new(r"Each gate divided by $|\mathrm{Re}\,R(10\,\mu\mathrm{s})|$; bars: assigned working SEM",100.,255.));
 f.save(&a[2])?;Ok(())
}
