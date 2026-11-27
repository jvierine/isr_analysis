// Native scalar grid -> rustplotlib. No Python plotting or image inputs.
use anyhow::Result;
use rustplotlib::*;
use std::path::PathBuf;
fn edges(v:&[f64])->Vec<f64>{let d=v[1]-v[0];(0..=v.len()).map(|i|v[0]+(i as f64-0.5)*d).collect()}
fn main()->Result<()>{
 let a:Vec<String>=std::env::args().collect();let input=PathBuf::from(&a[1]);let out=PathBuf::from(&a[2]);std::fs::create_dir_all(&out)?;let carrier:f64=a.get(3).map(|s|s.parse()).transpose()?.unwrap_or(0.);
 for i in 1..=12 {
  let path=input.join(format!("profile-{i:02}-native.h5"));if !path.exists(){continue;}
  let f=hdf5::File::open(path)?;let freq:Vec<f64>=f.dataset("frequency_hz")?.read_raw::<f64>()?.iter().map(|v|v/1e6).collect();let r=f.dataset("slant_range_km")?.read_raw::<f64>()?;let lim=f.dataset("color_limits_db")?.read_raw::<f64>()?;
  let mut values:Vec<f64>=f.dataset("display_index")?.read_raw::<u8>()?.into_iter().map(|q|if q==255{f64::NAN}else{lim[0]+q as f64/254.*(lim[1]-lim[0])}).collect();
  let bin_hz=(freq[1]-freq[0])*1e6;let shift=(-carrier/bin_hz).round()as isize;anyhow::ensure!((shift as f64*bin_hz+carrier).abs()<1e-3,"carrier offset must align to a native Fourier bin for exact display rotation");for row in values.chunks_exact_mut(freq.len()){row.rotate_right(shift.rem_euclid(freq.len()as isize)as usize);}
  let start=f.attr("integration_start_utc_s")?.read_scalar::<f64>()? as i64;let t=chrono::DateTime::from_timestamp(start,0).unwrap();let count=f.attr("accepted_pulses")?.read_scalar::<f64>()?;let duration=f.attr("integration_duration_s").ok().and_then(|a|a.read_scalar::<f64>().ok()).unwrap_or(60.);
  let mut fig=Figure::new(IntendedUse::ArticleWide);fig.width_mm=200.;fig.height_mm=115.;fig.raster_dpi=220.;fig.tight_export=false;
  let scale=fig.scale(Scale::new([lim[0],lim[1]],Normalization::Linear,ColorMap::inferno())?);
  let xlabel=if carrier==0.{"Recorded frequency offset (MHz)"}else{"Frequency offset from 430 MHz (MHz)"};let mut axes=Axes::new([-8.,8.],[50.,800.],xlabel,"Slant range (km)");axes.xticks=vec![-8.,-6.,-4.,-2.,0.,2.,4.,6.,8.];axes.yticks=vec![100.,200.,300.,400.,500.,600.,700.,800.];
  let mut panel=Panel::new(Rect::new(20.,16.,140.,80.),axes);let mut field=Field::new(edges(&freq),edges(&r),values,scale)?;field.interpolate=false;panel.push(Layer::Pcolormesh(field));fig.panels.push(panel);
  fig.labels.push(Label::new(format!("{} UTC; {duration:.0} s window; {:.2} s accepted",t.format("%Y-%m-%d %H:%M:%S"),count*0.01),100.,6.));
  let tick=(lim[1].min(-lim[0])*10.).floor()/10.;fig.colorbars.push(Colorbar::new(Rect::new(164.,16.,3.,80.),scale,"Normalized power (dB)",vec![-tick,0.,tick]));
  fig.raster_dpi=8192.*25.4/140.;let full=out.join(format!("profile-{i:02}-rustplotlib-full.png"));fig.save(&full)?;
  // Screen previews must integrate the native raster footprint, rather than
  // point-sample the scientific field and skip narrow spectral ridges.
  let img=image::open(&full)?;let height=(img.height()as f64*1800./img.width()as f64).round()as u32;img.resize_exact(1800,height,image::imageops::FilterType::Lanczos3).save(out.join(format!("profile-{i:02}-rustplotlib-preview.png")))?;
  println!("RENDERED rustplotlib profile {i}; native {} x {}; 800km maximum",freq.len(),r.len());
 }
 let meta=hdf5::File::create(out.join("frequency-reference.h5"))?;for(name,value)in[("recorded_carrier_hz",carrier),("rf_carrier_hz",430e6),("range_max_km",800.),("range_step_us",2.)]{meta.new_attr::<f64>().create(name)?.write_scalar(&value)?;}
 Ok(())
}
