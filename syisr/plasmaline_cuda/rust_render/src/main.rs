// Native scalar grid -> rustplotlib. No Python plotting or image inputs.
use anyhow::Result;
use rustplotlib::*;
use std::path::PathBuf;
fn edges(v:&[f64])->Vec<f64>{let d=v[1]-v[0];(0..=v.len()).map(|i|v[0]+(i as f64-0.5)*d).collect()}
fn white_label(text:impl Into<String>,x:f64,y:f64)->Label{let mut l=Label::new(text,x,y);l.color="#ffffff".into();l}
fn main()->Result<()>{
 let a:Vec<String>=std::env::args().collect();let input=PathBuf::from(&a[1]);let out=PathBuf::from(&a[2]);std::fs::create_dir_all(&out)?;let rf_mode=a.get(3).is_some_and(|s|s=="rf");let carrier:f64=if rf_mode{0.}else{a.get(3).map(|s|s.parse()).transpose()?.unwrap_or(0.)};
 for i in 1..=12 {
  let path=input.join(format!("profile-{i:02}-native.h5"));if !path.exists(){continue;}
  let f=hdf5::File::open(path)?;let freq:Vec<f64>=f.dataset("frequency_hz")?.read_raw::<f64>()?.iter().map(|v|v/1e6-if rf_mode{430.}else{0.}).collect();let r=f.dataset("slant_range_km")?.read_raw::<f64>()?;let lim=f.dataset("color_limits_db")?.read_raw::<f64>()?;
  let mut values:Vec<f64>=f.dataset("display_index")?.read_raw::<u8>()?.into_iter().map(|q|if q==255{f64::NAN}else{lim[0]+q as f64/254.*(lim[1]-lim[0])}).collect();
  let bin_hz=(freq[1]-freq[0])*1e6;let shift=(-carrier/bin_hz).round()as isize;anyhow::ensure!((shift as f64*bin_hz+carrier).abs()<1e-3,"carrier offset must align to a native Fourier bin for exact display rotation");for row in values.chunks_exact_mut(freq.len()){row.rotate_right(shift.rem_euclid(freq.len()as isize)as usize);}
  let first_valid=values.chunks_exact(freq.len()).position(|row|row.iter().any(|v|v.is_finite())).unwrap_or(0);let ymin=(r[first_valid]-(r[1]-r[0])/2.).max(50.);
  let start=f.attr("integration_start_utc_s")?.read_scalar::<f64>()? as i64;let t=chrono::DateTime::from_timestamp(start,0).unwrap();let count=f.attr("accepted_pulses")?.read_scalar::<f64>()?;let duration=f.attr("integration_duration_s").ok().and_then(|a|a.read_scalar::<f64>().ok()).unwrap_or(60.);
  let mut fig=Figure::new(IntendedUse::ArticleWide);fig.width_mm=200.;fig.height_mm=112.;fig.font_pt=6.5;fig.tight_export=false;fig.check_label_bounds=false;fig.check_label_overlap=false;fig.check_label_geometry=false;fig.auto_colorbar_layout=false;
  let scale=fig.scale(Scale::new([lim[0],lim[1]],Normalization::Linear,ColorMap::inferno())?);
  let xmin=if rf_mode{-2.}else{-8.};let xmax=if rf_mode{22.}else{8.};let xlabel=if carrier==0.&&!rf_mode{"Recorded frequency offset (MHz)"}else{"Frequency offset from 430 MHz (MHz)"};let mut axes=Axes::new([xmin,xmax],[ymin,800.],"","");axes.show_x_axis=false;axes.show_y_axis=false;
  let mut panel=Panel::new(Rect::new(0.,0.,200.,112.),axes);let mut field=Field::new(edges(&freq),edges(&r),values,scale)?;field.interpolate=false;panel.push(Layer::Pcolormesh(field));
  let ticks:Vec<f64>=(0..=((xmax-xmin)/2.)as usize).map(|i|xmin+i as f64*2.).collect();for v in ticks{let mut line=Line::new(vec![[v,ymin],[v,ymin+7.]],"#ffffff");line.width_pt=0.3;panel.push(Layer::Line(line));fig.labels.push(white_label(format!("{v:.0}"),((v-xmin)/(xmax-xmin)*200.).clamp(3.,197.),110.));}
  for v in [100.,200.,300.,400.,500.,600.,700.,800.]{let mut line=Line::new(vec![[xmin,v],[xmin+(xmax-xmin)*0.005,v]],"#ffffff");line.width_pt=0.3;panel.push(Layer::Line(line));fig.labels.push(white_label(format!("{v:.0}"),4.,((800.-v)/(800.-ymin)*112.).clamp(2.,109.)));}
  fig.panels.push(panel);fig.labels.push(white_label(xlabel,100.,105.5));let mut ylabel=white_label("Slant range (km)",7.5,56.);ylabel.rotation=-90.;fig.labels.push(ylabel);
  fig.labels.push(white_label(format!("{} UTC",t.format("%Y-%m-%d %H:%M:%S")),42.,3.5));let exposure=if rf_mode{let c=f.dataset("accepted_pulses_per_channel")?.read_raw::<f64>()?;format!("{duration:.0} s window; 436/444 MHz: {:.2}/{:.2} s accepted",c[0]*0.01,c[1]*0.01)}else{format!("{duration:.0} s window; {:.2} s accepted",count*0.01)};fig.labels.push(white_label(exposure,48.,7.));
  // Inset color scale and inward ticks, deliberately overlaid on the data.
  let mut bar_axes=Axes::new([0.,1.],[lim[0],lim[1]],"","");bar_axes.show_x_axis=false;bar_axes.show_y_axis=false;let mut bar=Panel::new(Rect::new(195.,10.,1.3,28.),bar_axes);
  let bar_y:Vec<f64>=(0..=256).map(|j|lim[0]+j as f64/256.*(lim[1]-lim[0])).collect();let bar_values:Vec<f64>=(0..256).map(|j|lim[0]+(j as f64+0.5)/256.*(lim[1]-lim[0])).collect();bar.push(Layer::Pcolormesh(Field::new(vec![0.,1.],bar_y,bar_values,scale)?));fig.panels.push(bar);
  let color_label=if f.attr("gaussian_sigma_frequency_bins").is_ok(){"High-pass contrast (dB)"}else{"Normalized power (dB)"};fig.labels.push(white_label(color_label,180.,4.5));let tick=(lim[1].min(-lim[0])*10.).floor()/10.;for v in [-tick,0.,tick]{fig.labels.push(white_label(format!("{v:.1}"),191.,10.+(lim[1]-v)/(lim[1]-lim[0])*28.));}
  let native_width=if rf_mode{12288}else{8192};fig.raster_dpi=native_width as f64*25.4/200.;let full=out.join(format!("profile-{i:02}-rustplotlib-full.png"));let video_only=a.get(4).is_some_and(|s|s=="video");
  // Screen previews must integrate the native raster footprint, rather than
  // point-sample the scientific field and skip narrow spectral ridges.
  let img=if video_only{let svg=fig.svg()?;let tree=resvg::usvg::Tree::from_str(&svg,&resvg::usvg::Options::default())?;let w=native_width;let h=(w as f64*fig.height_mm/fig.width_mm).round()as u32;let mut pix=resvg::tiny_skia::Pixmap::new(w,h).unwrap();let tr=resvg::tiny_skia::Transform::from_scale(w as f32/tree.size().width(),h as f32/tree.size().height());resvg::render(&tree,tr,&mut pix.as_mut());image::DynamicImage::ImageRgba8(image::RgbaImage::from_raw(w,h,pix.take()).unwrap())}else{fig.save(&full)?;image::open(&full)?};let height=(img.height()as f64*1800./img.width()as f64).round()as u32;img.resize_exact(1800,height,image::imageops::FilterType::Lanczos3).save(out.join(format!("profile-{i:02}-rustplotlib-preview.png")))?;
  println!("RENDERED rustplotlib profile {i}; native {} x {}; 800km maximum",freq.len(),r.len());
 }
 let meta=hdf5::File::create(out.join("frequency-reference.h5"))?;for(name,value)in[("recorded_carrier_hz",carrier),("rf_carrier_hz",430e6),("range_max_km",800.),("range_step_us",2.)]{meta.new_attr::<f64>().create(name)?.write_scalar(&value)?;}
 if rf_mode{for(name,value)in[("receiver_1_center_hz",436e6),("receiver_2_center_hz",444e6),("splice_rf_hz",440e6),("rf_lower_edge_hz",428e6),("rf_upper_edge_hz",452e6)]{meta.new_attr::<f64>().create(name)?.write_scalar(&value)?;}}
 Ok(())
}
