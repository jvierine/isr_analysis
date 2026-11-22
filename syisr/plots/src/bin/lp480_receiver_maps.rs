//! Receiver diagnostics: measured spectra and signed background-correction checks.
use anyhow::{ensure,Result};
use rustplotlib::*;
use serde_json::Value;
use std::{collections::BTreeMap,path::Path};
const NT:usize=31*48;const NF:usize=128;const START:i64=1714521600;
struct Field{sum:Vec<f64>,weight:Vec<f64>}
impl Field{fn new()->Self{Self{sum:vec![0.;4*NF*NT],weight:vec![0.;NT]}}fn add(&mut self,t:usize,p:&Value,n:f64){self.weight[t]+=n;for w in 0..4{for k in 0..NF{self.sum[(w*NF+k)*NT+t]+=n*p[w][k].as_f64().unwrap();}}}}
fn main()->Result<()>{let a=std::env::args().collect::<Vec<_>>();ensure!(a.len()==4,"spectra|acfs INPUT_DIRECTORY OUTPUT_DIRECTORY");let out=Path::new(&a[3]);std::fs::create_dir_all(out)?;if a[1]=="spectra"{spectra(Path::new(&a[2]),out)}else{acfs(Path::new(&a[2]),out)}}
fn spectra(root:&Path,out:&Path)->Result<()>{
 let mut fields=BTreeMap::<(i64,i64,String),Field>::new();let mut geometry=None;
 let mut files=std::fs::read_dir(root)?.map(|e|e.map(|e|e.path())).collect::<std::io::Result<Vec<_>>>()?;files.sort();
 for file in files{if file.extension().is_none_or(|x|x!="h5"){continue;}let v:Value=syisr_ac::campaign::load(&file)?;
  let time=v["start_utc_s"].as_i64().unwrap();let t=(time-START)/1800;if !(0..NT as i64).contains(&t){continue;}let t=t as usize;
  let az=(v["azimuth_deg"].as_f64().unwrap()*100.).round()as i64;let el=(v["elevation_deg"].as_f64().unwrap()*100.).round()as i64;
  if geometry.is_none(){geometry=Some(v["geometry"].clone());}
  if let Some(rows)=v["bursts"].as_array(){for row in rows{let n=row["sampled_pulses"].as_f64().unwrap();for(mode,key)in [("raw","raw_power_adu2"),("covariance","pulse_covariance_power_adu2")]{fields.entry((az,el,mode.into())).or_insert_with(Field::new).add(t,&row[key],n);}}}
  else{fields.entry((az,el,"screened".into())).or_insert_with(Field::new).add(t,&v["screened_power_adu2"],v["retained_pulses"].as_f64().unwrap());}
 }
 ensure!(!fields.is_empty(),"no receiver spectra");let geometry=geometry.unwrap();windows(&geometry,out)?;
 coherence(&fields,out)?;
 for((az,el,mode),field)in &fields{draw_spectra(field,*az,*el,mode,&geometry,out,0,NT)?;
  if *el>=8900 && mode!="screened"{for day in 0..31{if field.weight[day*48..(day+1)*48].iter().any(|n|*n>0.){draw_spectra(field,*az,*el,mode,&geometry,out,day*48,48)?;}}}
 }
 Ok(())
}
fn coherence(fields:&BTreeMap<(i64,i64,String),Field>,out:&Path)->Result<()>{
 let mut rows=vec![];let mut zenith=vec![vec![];4];
 for((az,el,mode),raw)in fields{if mode!="raw"{continue;}let Some(cov)=fields.get(&(*az,*el,"covariance".into()))else{continue;};
  for day in 0..31{for w in 0..4{let mut dc=vec![];let mut absolute=vec![];let mut peaks=vec![];
   for t in day*48+24..day*48+44{if raw.weight[t]<=0.||cov.weight[t]<=0.{continue;}
    let a=(0..NF).map(|k|raw.sum[(w*NF+k)*NT+t]/raw.weight[t]).collect::<Vec<_>>();
    let b=(0..NF).map(|k|cov.sum[(w*NF+k)*NT+t]/cov.weight[t]).collect::<Vec<_>>();
    let mut band=b.clone();band.sort_by(f64::total_cmp);let n0=(band[63]+band[64])/2.;if n0<=0.{continue;}
    let field=(63..=65).map(|k|a[k]-b[k]).sum::<f64>()/NF as f64;dc.push(field/n0);absolute.push(field);
    peaks.push(10.*(a.iter().copied().fold(0f64,f64::max)/n0).log10());
   }
   if dc.is_empty(){continue;}for a in [&mut dc,&mut absolute,&mut peaks]{a.sort_by(f64::total_cmp);}
   let median=|v:&Vec<f64>|(v[(v.len()-1)/2]+v[v.len()/2])/2.;
   let fraction=median(&dc);rows.push(serde_json::json!({"day_utc":day+1,"azimuth_deg":*az as f64/100.,"elevation_deg":*el as f64/100.,"window":w+1,"night_slots":dc.len(),"dc_band_coherent_fraction":fraction,"dc_band_coherent_adu2":median(&absolute),"median_peak_over_continuum_db":median(&peaks)}));
   if *el>=8900{zenith[w].push([day as f64+1.,fraction*100.]);}
  }}
 }
 if rows.is_empty(){return Ok(());}syisr_ac::campaign::save(&out.join("nighttime-coherence.h5"),&serde_json::json!({"rows":rows,"method":"Median over12–22UTC slots; signed raw-minus-unbiased-pulse-covariance power in the DC and adjacent Hann bins, divided by NF and covariance spectral median. Coherent-field diagnostic, not a physical-origin assignment or uncertainty interval."}))?;
 let mut f=Figure::new(IntendedUse::ArticleWide);f.width_mm=195.;f.height_mm=95.;f.font_pt=10.;f.latex.cache=out.join("latex-cache");
 let mut lo=0f64;let mut hi=0f64;for v in zenith.iter().flatten(){lo=lo.min(v[1]);hi=hi.max(v[1]);}let pad=(hi-lo).max(0.01)*0.1;
 let mut ax=Axes::new([1.,31.],[lo-pad,hi+pad],"UTC date (May 2024)","Coherent DC band / continuum (\\%)");ax.xticks=vec![1.,5.,9.,13.,17.,21.,25.,29.,31.];
 let mut p=Panel::new(Rect::new(22.,20.,160.,50.),ax);let mut legends=vec![];
 for(w,data)in zenith.into_iter().enumerate(){if data.is_empty(){continue;}let l=Line::series(data,w);legends.push(l.legend(format!("W{}",w+1)));p.push(Layer::Line(l));}p.legend=Some(Legend::new(legends));f.panels.push(p);
 f.labels.push(Label::new("SYISR zenith: nighttime pulse-coherent DC-band power",97.5,5.));f.labels.push(Label::new("12--22 UTC medians; 32-pulse covariance; four receive windows",97.5,88.));f.save(out.join("nighttime-coherence.png"))?;Ok(())
}
fn draw_spectra(field:&Field,az:i64,el:i64,mode:&str,g:&Value,out:&Path,first:usize,nt:usize)->Result<()>{
 let mut f=Figure::new(IntendedUse::ArticleWide);f.width_mm=205.;f.height_mm=225.;f.font_pt=9.;f.raster_dpi=170.;f.latex.cache=out.join("latex-cache");
 let label=match mode{"raw"=>"Unfiltered raw I/Q samples","covariance"=>"Covariance within sampled pulse bursts",_=>"Existing screened pulse averages"};
 let date=if nt==NT{"1--31 May 2024".into()}else{format!("{} May 2024",first/48+1)};
 f.labels.push(Label::new(format!("SYISR; azimuth {:.2}$^\\circ$, elevation {:.2}$^\\circ$; {date}",az as f64/100.,el as f64/100.),102.,5.));
 f.labels.push(Label::new(format!("{label}; 30-minute spectral averages"),102.,11.));
 let scale=f.scale(Scale::new([-3.,6.],Normalization::Linear,ColorMap::storm())?);
 let x=(0..=nt).map(|i|if nt==NT{i as f64/48.}else{i as f64/2.}).collect::<Vec<_>>();
 let y=(0..=NF).map(|k|(-64.5+k as f64)*100./128.).collect::<Vec<_>>();
 for w in 0..4{
  let mut data=vec![f64::NAN;NF*nt];
  for j in 0..nt{let t=first+j;if field.weight[t]<=0.{continue;}let mut band=(0..NF).map(|k|field.sum[(w*NF+k)*NT+t]/field.weight[t]).collect::<Vec<_>>();
   band.sort_by(f64::total_cmp);let baseline=(band[63]+band[64])/2.;if baseline<=0.{continue;}
   for k in 0..NF{let p=field.sum[(w*NF+k)*NT+t]/field.weight[t];if p>0.{data[k*nt+j]=10.*(p/baseline).log10();}}
  }
  let mut ax=Axes::new([0.,*x.last().unwrap()],[-50.,50.],if w==3{if nt==NT{"UTC date (May 2024)"}else{"UTC hour"}}else{""},"Frequency (kHz)");ax.yticks=vec![-40.,-20.,0.,20.,40.];
  ax.xticks=if nt==NT{(0..=28).step_by(4).map(|d|d as f64).chain(std::iter::once(31.)).collect()}else{vec![0.,4.,8.,12.,16.,20.,24.]};
  if nt==NT{ax.xticklabels=Some(ax.xticks.iter().map(|d|if *d==31.{"1 Jun".into()}else{format!("{}",*d as usize+1)}).collect());}if w<3{ax.show_x_axis=false;}
  let rect=Rect::new(22.,25.+w as f64*45.,151.,31.);let mut p=Panel::new(rect,ax);p.pcolormesh(x.clone(),y.clone(),data,scale)?;f.panels.push(p);
  f.colorbars.push(Colorbar::new(Rect::new(177.,rect.y,3.,rect.height),scale,"Power / band median (dB)",vec![-3.,0.,3.,6.]));
  let r=&g["pulse_center_slant_range_edges_km"][w];f.labels.push(Label::new(format!("W{}: {:.0}--{:.0} km slant range",w+1,r[0].as_f64().unwrap(),r[1].as_f64().unwrap()),97.,rect.y-3.));
 }
 let name=if nt==NT{format!("month-{mode}-az{az}-el{el}")}else{format!("day-{:02}-{mode}-zenith",first/48+1)};
 f.labels.push(Label::new(if mode=="screened"{"All retained pulse groups; all frequency bins shown; missing averages left blank"}else{"Up to32pulses per10sblock; all frequency bins shown; missing averages left blank"},102.,213.));f.save(out.join(format!("{name}.png")))?;f.save(out.join(format!("{name}.pdf")))?;Ok(())
}
fn windows(g:&Value,out:&Path)->Result<()>{
 let mut f=Figure::new(IntendedUse::ArticleWide);f.width_mm=200.;f.height_mm=105.;f.font_pt=10.;f.latex.cache=out.join("latex-cache");
 let mut ax=Axes::new([0.,16.],[-0.5,6.5],"Time after transmission starts (ms)","");ax.xticks=(0..=8).map(|i|i as f64*2.).collect();ax.yticks=vec![0.,1.,2.,3.,4.,5.,6.];ax.yticklabels=Some(vec!["W4","W3","W2","W1","Noise ACF starts","RX record","TX"].into_iter().map(String::from).collect());
 let mut p=Panel::new(Rect::new(31.,17.,160.,66.),ax);
 let mut line=|a:f64,b:f64,y:f64,c:&str|{let mut l=Line::new(vec![[a,y],[b,y]],c);l.width_pt=6.;p.push(Layer::Line(l));};
 line(0.,0.48,6.,"#ad4050");line(0.767,14.007,5.,"#657f8c");line(12.407,13.537,4.,"#bc9a37");
 for w in 0..4{let a=g["raw_sample_time_edges_ms"][w][0].as_f64().unwrap();let b=g["raw_sample_time_edges_ms"][w][1].as_f64().unwrap();line(a,b,(3-w)as f64,"#2e687b");}
 f.panels.push(p);f.labels.push(Label::new("SYISR 480 $\\mu$s; end-of-record spectral windows",100.,5.));
 f.labels.push(Label::new("Noise-ACF lag pairs extend up to 470 $\\mu$s beyond their start samples",100.,98.));f.save(out.join("receiver-windows.png"))?;Ok(())
}
fn acfs(root:&Path,out:&Path)->Result<()>{
 for file in std::fs::read_dir(root)?{let file=file?.path();if file.extension().is_none_or(|x|x!="h5"){continue;}let v:Value=syisr_ac::campaign::load(&file)?;if v["elevation_deg"].as_f64().unwrap()<89.{continue;}
  let time=v["time_utc_s"].as_i64().unwrap();let date=chrono::DateTime::<chrono::Utc>::from_timestamp(time+28800,0).unwrap();let gates=v["gates"].as_array().unwrap();
  let mut f=Figure::new(IntendedUse::ArticleWide);f.width_mm=200.;f.height_mm=265.;f.font_pt=9.;f.latex.cache=out.join("latex-cache");
  f.labels.push(Label::new(format!("SYISR zenith; {} China local time",date.format("%d %B %Y %H:%M")),100.,5.));f.labels.push(Label::new("Measured ACFs before and after receiver-background correction",100.,11.));
  for(row,z)in [700.,900.,1100.,1300.,1500.,1700.].iter().enumerate(){let g=(0..gates.len()).min_by(|a,b|(gates[*a]["altitude_km"].as_f64().unwrap()-z).abs().total_cmp(&(gates[*b]["altitude_km"].as_f64().unwrap()-z).abs())).unwrap();
   let ids=v["measurement_gate"].as_array().unwrap().iter().enumerate().filter(|(_,x)|x.as_u64()==Some(g as u64)).map(|(i,_)|i).collect::<Vec<_>>();if ids.is_empty(){continue;}
   for part in 0..2{let mut a=vec![];let mut b=vec![];let mut errors=vec![];let(mut lo,mut hi)=(0f64,0f64);
    for i in &ids{let x=v["measurement_lag"][*i].as_f64().unwrap()*10.;let old=v["old_observed"][*i]["value"][part].as_f64().unwrap();let new=v["corrected_observed"][*i]["value"][part].as_f64().unwrap();let e=v["corrected_observed"][*i]["sigma"][part].as_f64().unwrap();a.push([x,old]);b.push([x,new]);errors.push([new-e,new+e]);lo=lo.min(old).min(new-e);hi=hi.max(old).max(new+e);}
    let pad=(hi-lo).max(0.1)*0.08;let mut ax=Axes::new([0.,480.],[lo-pad,hi+pad],if row==5{r"Lag ($\mu$s)"}else{""},if part==0{r"Real ACF (ADU$^2$)"}else{r"Imaginary ACF (ADU$^2$)"});ax.xticks=vec![0.,100.,200.,300.,400.];if row<5{ax.show_x_axis=false;}
    let rect=Rect::new(23.+part as f64*96.,25.+row as f64*37.,72.,26.);let mut p=Panel::new(rect,ax);let old=Line::new(a,"#999999");let new=Line::new(b.clone(),"#176c86");if row==0&&part==0{p.legend=Some(Legend::new(vec![old.legend("Original"),new.legend("Corrected")]));}p.push(Layer::Line(old));p.push(Layer::Line(new));p.errorbars(&b,None,Some(&errors),"#176c86",0.4,0.7,0.4)?;f.panels.push(p);f.labels.push(Label::new(format!("{:.0} km",gates[g]["altitude_km"].as_f64().unwrap()),rect.x+rect.width/2.,rect.y-3.));
   }
  }
  f.labels.push(Label::new("Signed measured products; bars are working SEMs; no plasma refit",100.,255.));f.save(out.join(format!("corrected-acfs-{time}.png")))?;f.save(out.join(format!("corrected-acfs-{time}.pdf")))?;
 }
 Ok(())
}
