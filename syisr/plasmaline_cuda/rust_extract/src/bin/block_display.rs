// Native block display preparation; each decoded frequency normalized independently.
use anyhow::Result;
use std::path::PathBuf;
fn quantile(v:&mut[f64],q:f64)->f64{let k=((v.len()-1)as f64*q).round()as usize;*v.select_nth_unstable_by(k,f64::total_cmp).1}
fn main()->Result<()>{
 let a:Vec<String>=std::env::args().collect();let root=PathBuf::from(&a[1]);let dest=PathBuf::from(&a[2]);std::fs::create_dir_all(&dest)?;let fixed:Option<Vec<f64>>=a.get(3).map(|p|->Result<_>{Ok(hdf5::File::open(p)?.dataset("color_limits_db")?.read_raw::<f64>()?)}).transpose()?;
 let mut paths:Vec<_>=std::fs::read_dir(root)?.filter_map(|x|x.ok().map(|x|x.path())).filter(|p|p.file_name().unwrap().to_string_lossy().starts_with("block-")&&p.extension().is_some_and(|x|x=="h5")).collect();paths.sort();
 for(i,path)in paths.iter().enumerate(){let f=hdf5::File::open(path)?;let freq=f.dataset("frequency_offset_hz")?.read_raw::<f64>()?;let r=f.dataset("slant_range_km")?.read_raw::<f64>()?;let p=f.dataset("whitened_matched_power_ratio")?.read_raw::<f64>()?;let nf=freq.len();let nr=r.iter().take_while(|&&r|r<=800.).count();
  let mut background=vec![0.;nf];let mut column=Vec::with_capacity(r.len());for k in 0..nf{column.clear();for row in p.chunks_exact(nf){let x=row[k];if x.is_finite()&&x>0.{column.push(x);}}background[k]=quantile(&mut column,0.5);}
  let db:Vec<f64>=p[..nr*nf].iter().enumerate().map(|(j,&x)|if x.is_finite()&&x>0.{10.*(x/background[j%nf]).log10()}else{f64::NAN}).collect();let mut finite:Vec<f64>=db.iter().copied().filter(|v|v.is_finite()).collect();let q1=quantile(&mut finite,0.25);let q3=quantile(&mut finite,0.75);let limits=[q1-1.5*(q3-q1),q3+1.5*(q3-q1)];
  let limits=if let Some(v)=&fixed{[v[0],v[1]]}else{limits};let display:Vec<u8>=db.iter().map(|&x|if x.is_finite(){((x-limits[0])/(limits[1]-limits[0])*254.).round().clamp(0.,254.)as u8}else{255}).collect();let out=hdf5::File::create(dest.join(format!("profile-{:02}-native.h5",i+1)))?;
  out.new_dataset::<u8>().shape((nr,nf)).chunk((16,nf)).deflate(1).create("display_index")?.write_raw(&display)?;
  for(name,values)in[("frequency_hz",freq.as_slice()),("slant_range_km",&r[..nr]),("color_limits_db",limits.as_slice()),("decoded_frequency_background",background.as_slice())]{out.new_dataset::<f64>().shape(values.len()).create(name)?.write_raw(values)?;}
  for name in ["integration_start_utc_s","accepted_pulses"]{out.new_attr::<f64>().create(name)?.write_scalar(&f.attr(name)?.read_scalar::<f64>()?)?;}let duration=f.attr("integration_duration_s").ok().and_then(|v|v.read_scalar::<f64>().ok()).unwrap_or(60.);out.new_attr::<f64>().create("integration_duration_s")?.write_scalar(&duration)?;
  println!("DISPLAY {} <- {}; IQR {}..{} dB",i+1,path.display(),limits[0],limits[1]);
 }Ok(())
}
