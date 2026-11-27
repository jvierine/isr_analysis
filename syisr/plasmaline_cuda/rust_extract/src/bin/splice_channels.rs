// Independent decoded-column normalization; hard RF splice without wrapping.
use anyhow::{Result,ensure};
use std::path::PathBuf;
fn quantile(v:&mut[f64],q:f64)->f64{let k=((v.len()-1)as f64*q).round()as usize;*v.select_nth_unstable_by(k,f64::total_cmp).1}
fn main()->Result<()>{
 let a:Vec<String>=std::env::args().collect();let root=PathBuf::from(&a[1]);let dest=PathBuf::from(&a[2]);std::fs::create_dir_all(&dest)?;
 let mut ranges=Vec::new();let mut spectra=Vec::new();let mut frequencies=Vec::new();let mut counts=Vec::new();let mut start=0.;let mut backgrounds=Vec::new();
 for center in [436e6,444e6]{let f=hdf5::File::open(root.join(format!("{center:.0}/matched-step2us.h5")))?;
  ensure!(f.attr("receiver_center_hz")?.read_scalar::<f64>()?==center,"mixed channel product");ensure!(f.attr("range_step_us")?.read_scalar::<f64>()?==2.,"range step");
  let t=f.attr("integration_start_utc_s")?.read_scalar::<f64>()?;if spectra.is_empty(){start=t;}else{ensure!(t==start,"channel time mismatch");}
  ensure!(f.attr("integration_stop_utc_s")?.read_scalar::<f64>()?-t==60.,"integration duration");
  let r=f.dataset("slant_range_km")?.read_raw::<f64>()?;if ranges.is_empty(){ranges=r.clone();}else{ensure!(ranges==r,"range grids differ");}
  let freq=f.dataset("rf_frequency_hz")?.read_raw::<f64>()?;let nf=freq.len();let mut p=f.dataset("whitened_matched_power_ratio")?.read_raw::<f64>()?;ensure!(p.len()==r.len()*nf,"shape");
  let mut background=vec![0.;nf];let mut col=Vec::with_capacity(r.len());for k in 0..nf{col.clear();for row in p.chunks_exact(nf){if row[k].is_finite()&&row[k]>0.{col.push(row[k]);}}ensure!(!col.is_empty(),"empty frequency column");background[k]=quantile(&mut col,0.5);}
  for(j,x)in p.iter_mut().enumerate(){*x=if x.is_finite()&&*x>0.{10.*(*x/background[j%nf]).log10()}else{f64::NAN};}
  counts.push(f.attr("accepted_pulses")?.read_scalar::<f64>()?);frequencies.push(freq);spectra.push(p);backgrounds.push(background);
 }
 let nr=ranges.iter().take_while(|&&r|r<=800.).count();let selections:Vec<Vec<usize>>=frequencies.iter().enumerate().map(|(ch,f)|f.iter().enumerate().filter(|&(_,v)|if ch==0{*v<440e6}else{*v>=440e6}).map(|(i,_)|i).collect()).collect();
 let mut rf=Vec::new();for ch in 0..2{for &i in &selections[ch]{rf.push(frequencies[ch][i]);}}let df=frequencies[0][1]-frequencies[0][0];ensure!(rf.windows(2).all(|v|(v[1]-v[0]-df).abs()<1e-6),"RF seam gap or duplicate");ensure!(rf[0]==428e6&&rf.len()==12288,"RF coverage");
 let mut db=Vec::with_capacity(nr*rf.len());for r in 0..nr{for ch in 0..2{for &k in &selections[ch]{db.push(spectra[ch][r*frequencies[ch].len()+k]);}}}
 let sigma:Option<f64>=a.get(4).map(|s|s.parse()).transpose()?;let range_sigma:Option<f64>=a.get(5).map(|s|s.parse()).transpose()?;if let Some(s)=sigma{db=plasma_display_extract::highpass_axes(&db,rf.len(),s,range_sigma.unwrap_or(s));}
 let mut finite:Vec<f64>=db.iter().copied().filter(|v|v.is_finite()).collect();let q1=quantile(&mut finite,0.25);let q3=quantile(&mut finite,0.75);let mut limits=[q1-1.5*(q3-q1),q3+1.5*(q3-q1)];if let Some(path)=a.get(3){limits.copy_from_slice(&hdf5::File::open(path)?.dataset("color_limits_db")?.read_raw::<f64>()?);}
 let display:Vec<u8>=db.iter().map(|&x|if x.is_finite(){((x-limits[0])/(limits[1]-limits[0])*254.).round().clamp(0.,254.)as u8}else{255}).collect();let out=hdf5::File::create(dest.join("profile-01-native.h5"))?;out.new_dataset::<u8>().shape((nr,rf.len())).chunk((16,1024)).deflate(1).create("display_index")?.write_raw(&display)?;
 for(name,v)in[("frequency_hz",rf.as_slice()),("slant_range_km",&ranges[..nr]),("color_limits_db",limits.as_slice()),("accepted_pulses_per_channel",counts.as_slice()),("receiver_center_hz",[436e6,444e6].as_slice())]{out.new_dataset::<f64>().shape(v.len()).create(name)?.write_raw(v)?;}
 for(name,v)in[("integration_start_utc_s",start),("integration_duration_s",60.),("accepted_pulses",counts[0]+counts[1]),("splice_rf_hz",440e6),("transmit_frequency_hz",430e6)]{out.new_attr::<f64>().create(name)?.write_scalar(&v)?;}
 if let Some(s)=sigma{out.new_dataset::<f32>().shape((nr,rf.len())).chunk((16,1024)).shuffle().deflate(1).create("signed_highpass_db")?.write_raw(&db.iter().map(|&x|x as f32).collect::<Vec<_>>())?;for(name,v)in[("gaussian_sigma_frequency_bins",s),("gaussian_sigma_range_bins",range_sigma.unwrap_or(s)),("gaussian_sigma_frequency_hz",s*df),("gaussian_sigma_slant_range_km",range_sigma.unwrap_or(s)*(ranges[1]-ranges[0]))]{out.new_attr::<f64>().create(name)?.write_scalar(&v)?;}}
 // Keep both normalized overlap estimates for inspection; neither is overwritten.
 let audit=hdf5::File::create(dest.join("channel-overlap.h5"))?;let mut overlap_rf=Vec::new();for ch in 0..2{let ix:Vec<_>=frequencies[ch].iter().enumerate().filter(|&(_,v)|*v>=436e6&&*v<444e6).map(|(i,_)|i).collect();let mut values=Vec::with_capacity(nr*ix.len());for r in 0..nr{for &k in &ix{values.push(spectra[ch][r*8192+k]as f32);}}audit.new_dataset::<f32>().shape((nr,ix.len())).chunk((16,1024)).shuffle().deflate(1).create(format!("normalized_db_{}",if ch==0{"436"}else{"444"}).as_str())?.write_raw(&values)?;if ch==0{overlap_rf=ix.iter().map(|&i|frequencies[ch][i]).collect();}else{ensure!(overlap_rf==ix.iter().map(|&i|frequencies[ch][i]).collect::<Vec<_>>(),"overlap RF alignment");}audit.new_dataset::<f64>().shape(8192).create(format!("decoded_column_background_{}",if ch==0{"436"}else{"444"}).as_str())?.write_raw(&backgrounds[ch])?;}
 audit.new_dataset::<f64>().shape(overlap_rf.len()).create("rf_frequency_hz")?.write_raw(&overlap_rf)?;println!("PASS RF splice 428–452 MHz at 440 MHz; {} x {}; accepted {:?}; IQR {:?}",rf.len(),nr,counts,limits);Ok(())
}
