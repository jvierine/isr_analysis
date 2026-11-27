// Five-second wall-clock blocks of the contaminated integration. Retain all
// accepted pulses and quantify interference instead of deleting the minute.
use anyhow::{Result,ensure};
use hdf5::types::VarLenUnicode;
use libloading::{Library,Symbol};
use std::{ffi::c_void,fs::File,io::{Read,Seek,SeekFrom},path::PathBuf};
fn main()->Result<()>{unsafe{
 let a:Vec<String>=std::env::args().collect();let root=PathBuf::from(&a[1]);let raw=PathBuf::from(&a[2]);let output=PathBuf::from(&a[3]);std::fs::create_dir_all(&output)?;
 let selection=hdf5::File::open(root.join("selection.h5"))?;let paths=selection.dataset("matched_path")?.read_raw::<VarLenUnicode>()?;let refs=selection.dataset("diagnostic_path")?.read_raw::<VarLenUnicode>()?;
 let m=hdf5::File::open(paths[2].as_str())?;let d=hdf5::File::open(refs[2].as_str())?;
 let times=m.dataset("pulse_time_utc_s")?.read_raw::<f64>()?;let good=m.dataset("pulse_accepted")?.read_raw::<bool>()?;
 let names=d.dataset("source_filename")?.read_raw::<VarLenUnicode>()?;let offsets=d.dataset("source_byte_offset")?.read_raw::<u64>()?;let phases=d.dataset("pulse_phase")?.read_raw::<u16>()?;
 let gain=m.dataset("whitening_gain_unshifted")?.read_raw::<f32>()?;let signs=m.dataset("ideal_code_phase_signs")?.read_raw::<i8>()?;let codes:Vec<f32>=signs.iter().flat_map(|&x|std::iter::repeat_n(x as f32,160)).collect();
 let start=m.attr("integration_start_utc_s")?.read_scalar::<f64>()?;let range=m.dataset("slant_range_km")?.read_raw::<f64>()?;let freq=m.dataset("frequency_offset_hz")?.read_raw::<f64>()?;
 let valid=m.dataset("range_valid")?.read_raw::<bool>()?;let nr=range.len();let nf=freq.len();let ne=149436;let nt=5120;let library=Library::new(&a[4])?;
 let create:Symbol<unsafe extern "C" fn(i32,i32,i32,i32,i32,i32,*const f32,*const f32,*mut *mut c_void)->i32>=library.get(b"pl_match_create")?;
 let add:Symbol<unsafe extern "C" fn(*mut c_void,*const i32,i32)->i32>=library.get(b"pl_match_accumulate")?;
 let results:Symbol<unsafe extern "C" fn(*mut c_void,*mut f64,*mut f64)->i32>=library.get(b"pl_match_results")?;
 let destroy:Symbol<unsafe extern "C" fn(*mut c_void)>=library.get(b"pl_match_destroy")?;
 let mut quality=Vec::<[f64;5]>::new();
 for block in 0..12 {
  let indices:Vec<usize>=(0..times.len()).filter(|&j|good[j]&&times[j]>=start+block as f64*5.&&times[j]<start+(block+1)as f64*5.).collect();if indices.is_empty(){continue;}
  let mut context=std::ptr::null_mut();ensure!(create(0,ne as i32,nt,8192,32,nf as i32,codes.as_ptr(),gain.as_ptr(),&mut context)==0,"GPU create failed");
  let mut crest=Vec::new();let mut total_power=Vec::new();
  for &j in &indices {
   let mut file=File::open(raw.join(names[j].as_str()))?;file.seek(SeekFrom::Start(offsets[j]+1024))?;let mut qi=vec![0i32;ne*2];file.read_exact(std::slice::from_raw_parts_mut(qi.as_mut_ptr().cast::<u8>(),ne*8))?;
   let mut sum=0.;let mut peak=0f64;for x in qi.chunks_exact(2){let p=(x[0]as f64).powi(2)+(x[1]as f64).powi(2);sum+=p;peak=peak.max(p);}let mean=sum/ne as f64;crest.push(peak/mean);total_power.push(mean);
   ensure!(add(context,qi.as_ptr(),phases[j]as i32)==0,"GPU accumulation failed");
  }
  let mut sum=vec![0.;nr*nf];let mut squares=vec![0.;nr*nf];ensure!(results(context,sum.as_mut_ptr(),squares.as_mut_ptr())==0,"GPU results failed");destroy(context);
  for(r,row)in sum.chunks_exact_mut(nf).enumerate(){for x in row{*x=if valid[r]{*x/indices.len()as f64}else{f64::NAN};}}let out=hdf5::File::create(output.join(format!("block-{block:02}.h5")))?;
  out.new_dataset::<f64>().shape((nr,nf)).create("whitened_matched_power_ratio")?.write_raw(&sum)?;
  for(name,v)in[("slant_range_km",&range),("frequency_offset_hz",&freq)]{out.new_dataset::<f64>().shape(v.len()).create(name)?.write_raw(v)?;}
  for(name,v)in[("integration_start_utc_s",start+block as f64*5.),("accepted_pulses",indices.len()as f64),("integration_duration_s",5.)]{out.new_attr::<f64>().create(name)?.write_scalar(&v)?;}
  let mut sorted=crest.clone();sorted.sort_by(f64::total_cmp);let c=sorted[sorted.len()/2];let max=crest.iter().copied().fold(0f64,f64::max);let p=total_power.iter().sum::<f64>()/total_power.len()as f64;
  quality.push([block as f64,indices.len()as f64,c,max,p]);println!("BLOCK {block}: {} IPPs; median crest {c:.1}, maximum crest {max:.1}, mean ADC power {p:.3e}",indices.len());
 }
 let out=hdf5::File::create(output.join("quality.h5"))?;out.new_dataset::<[f64;5]>().shape(quality.len()).create("block_pulse_count_median_crest_max_crest_mean_power")?.write_raw(&quality)?;
 Ok(())
}}
