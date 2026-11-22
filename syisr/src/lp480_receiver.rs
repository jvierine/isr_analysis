//! Receiver-only diagnostics. These routines never optimize plasma parameters.
use crate::{campaign,lp480::{self,Decoded,Manifest},lp480_fit::{self,Background}};
use anyhow::{ensure,Result};
use num_complex::{Complex32,Complex64};
use rustfft::FftPlanner;
use serde_json::json;
use std::{collections::BTreeMap,io::{BufReader,Seek,SeekFrom},path::Path};

fn windows()->Vec<usize>{
 let nw=(lp480::NS-crate::lp480_screen::START-crate::lp480_screen::NF)/crate::lp480_screen::HOP+1;
 (nw-4..nw).map(|w|crate::lp480_screen::START+w*crate::lp480_screen::HOP).collect()
}
fn geometry(front:f64)->serde_json::Value{
 let nf=crate::lp480_screen::NF;
 json!({"sample_windows":windows().iter().map(|k|[*k,*k+nf]).collect::<Vec<_>>(),
  "raw_sample_time_edges_ms":windows().iter().map(|k|[front/1000.+*k as f64*0.01,front/1000.+(*k+nf)as f64*0.01]).collect::<Vec<_>>(),
  "pulse_center_slant_range_edges_km":windows().iter().map(|k|[(*k as f64*lp480::DT+front*1e-6-lp480::PW/2.)*lp480::C_KM/2.,((*k+nf)as f64*lp480::DT+front*1e-6-lp480::PW/2.)*lp480::C_KM/2.]).collect::<Vec<_>>(),
  "noise_product_first_sample":lp480::NS-160,"noise_product_last_start_sample_exclusive":lp480::NS-lp480::NL,
  "record_samples":lp480::NS,"sample_interval_us":10,"pulse_us":480,"ipp_us":16000,
  "frequency_hz":(0..nf).map(|i|(i as f64-nf as f64/2.)*1e5/nf as f64).collect::<Vec<_>>(),
  "frequency_convention":"FFT of recorded complex I/Q, shifted to [-50,+50)kHz; not a verified LOS-velocity sign"})
}
fn spectra(frames:&[Vec<Complex32>])->(Vec<Vec<f64>>,Vec<Vec<f64>>){
 let n=frames.len();let nf=crate::lp480_screen::NF;
 let fft=FftPlanner::<f64>::new().plan_fft_forward(nf);
 let hann=(0..nf).map(|j|0.5-0.5*(std::f64::consts::TAU*j as f64/(nf-1)as f64).cos()).collect::<Vec<_>>();
 let energy=hann.iter().map(|v|v*v).sum::<f64>();
 let mut raw=vec![];let mut cov=vec![];
 for k in windows(){
  let mut power=vec![0.;nf];let mut field=vec![Complex64::default();nf];
  for x in frames{
   let mut work=(0..nf).map(|j|Complex64::new(x[k+j].re as f64,x[k+j].im as f64)*hann[j]).collect::<Vec<_>>();
   fft.process(&mut work);for j in 0..nf{power[j]+=work[j].norm_sqr()/n as f64/energy;field[j]+=work[j]/n as f64;}
  }
  let centered=(0..nf).map(|j|(power[j]-field[j].norm_sqr()/energy)*n as f64/(n-1)as f64).collect::<Vec<_>>();
  raw.push((0..nf).map(|j|power[(j+nf/2)%nf]).collect());
  cov.push((0..nf).map(|j|centered[(j+nf/2)%nf]).collect());
 }
 (raw,cov)
}
pub fn raw_tail(index:&Path,out:&Path)->Result<()>{
 let m:Manifest=campaign::load(index)?;let(rank,size)=lp480::worker()?;
 std::fs::create_dir_all(out)?;
 for(i,job)in m.jobs.iter().enumerate(){
  if i%size!=rank{continue;}let dest=out.join(format!("{}.h5",job.id));if dest.exists(){continue;}
  let mut groups=BTreeMap::new();for part in &job.parts{groups.entry(part.block).or_insert_with(Vec::new).push(part);}
  let mut rows=vec![];
  for(start,parts)in groups{
   let mut frames=vec![];
   for part in parts{
    if frames.len()>=32{break;}let mut f=BufReader::new(std::fs::File::open(&part.file)?);f.seek(SeekFrom::Start(part.offset))?;
    for _ in 0..part.pulses.min(32-frames.len()){
     let frame=crate::reader::read_frame(&mut f)?.ok_or_else(||anyhow::anyhow!("EOF in validated diagnostic segment"))?;
     ensure!(frame.iq.len()==lp480::NS&&frame.header.pulse_width_us==480&&frame.header.ipp_us==16000,"unsupported raw receiver layout");frames.push(frame.iq);
    }
   }
   if frames.len()<2{continue;}let(raw,covariance)=spectra(&frames);
   rows.push(json!({"start_utc_s":start,"sampled_pulses":frames.len(),"raw_power_adu2":raw,"pulse_covariance_power_adu2":covariance}));
  }
  campaign::save(&dest,&json!({"start_utc_s":job.start,"azimuth_deg":job.az,"elevation_deg":job.el,"geometry":geometry(job.front_us),"bursts":rows,
   "method":"First up to32validated pulses per10sblock, no satellite/RFI rejection; raw and across-pulse-mean-removed Hann periodograms, n/(n-1) correction. Sparse raw diagnostic, not complete transient coverage or a plasma fit. No frequency bins excised."}))?;
  println!("RAW_TAIL_DONE {}",job.id);
 }
 Ok(())
}
pub fn stored_tail(root:&Path,out:&Path)->Result<()>{
 let m:Manifest=campaign::load(&root.join("index.h5"))?;let(rank,size)=lp480::worker()?;let nf=crate::lp480_screen::NF;
 std::fs::create_dir_all(out)?;
 for(i,job)in m.jobs.iter().enumerate(){
  if i%size!=rank{continue;}let dest=out.join(format!("{}.h5",job.id));if dest.exists(){continue;}
  let p=root.join("products").join(format!("{}.h5",job.id));if !p.exists(){continue;}
  let d:Decoded=campaign::load(&p)?;
  let mut sum=vec![];let mut count=0usize;
  if let Some(a)=d.averaged{sum=a.range_doppler_raw;count=a.range_doppler_pulses;}else{
   for b in d.blocks{if sum.is_empty(){sum=vec![0.;b.range_doppler_raw.len()];}ensure!(sum.len()==b.range_doppler_raw.len(),"spectral dimensions changed");count+=b.range_doppler_pulses;
    for(x,v)in sum.iter_mut().zip(b.range_doppler_raw){*x+=v*b.range_doppler_pulses as f64;}}
   if count>0{sum.iter_mut().for_each(|v|*v/=count as f64);}
  }
  if count==0{continue;}let nw=sum.len()/nf;ensure!(nw>=4&&sum.len()%nf==0,"far spectral shape");
  campaign::save(&dest,&json!({"start_utc_s":job.start,"azimuth_deg":job.az,"elevation_deg":job.el,"geometry":geometry(job.front_us),"retained_pulses":count,
   "screened_power_adu2":(nw-4..nw).map(|w|sum[w*nf..(w+1)*nf].to_vec()).collect::<Vec<_>>(),
   "method":"All existing retained eight-pulse groups after the production satellite screen; native Hann periodograms, no noise subtraction or spectral notch. Missing/rejected groups remain excluded."}))?;
 }
 Ok(())
}
pub fn copy_covariance(source:&Path,out:&Path)->Result<()>{
 let m:Manifest=campaign::load(&source.join("index.h5"))?;std::fs::create_dir_all(out.join("products"))?;
 for j in &m.jobs{let dest=out.join("products").join(format!("{}.h5",j.id));if dest.exists(){continue;}
  let mut d:Decoded=campaign::load(&source.join("products").join(format!("{}.h5",j.id)))?;
  ensure!(d.schema.contains("pulse-covariance")&&d.averaged.is_none(),"unchanged raw pulse-covariance blocks required for re-export");
  d.schema="syisr-lp480-pooled-screen-v4-pulse-covariance".into();
  d.rejection.push_str(" Raw pulse-covariance sums unchanged; re-exported for v4 receiver subtraction.");campaign::save(&dest,&d)?;
 }campaign::save(&out.join("index.h5"),&m)?;Ok(())
}
pub fn affine_reference(quiet_covariance:&Path,variation:&Path,out:&Path)->Result<()>{
 let marker:serde_json::Value=campaign::load(&quiet_covariance.join("sensitivity.h5"))?;
 ensure!(marker["mode"]=="pulse-covariance-pilot","covariance quiet reference required");
 let m:Manifest=campaign::load(&quiet_covariance.join("index.h5"))?;
 let d:Decoded=campaign::load(&quiet_covariance.join("products").join(format!("{}.h5",m.jobs[0].id)))?;
 ensure!(d.schema.contains("pulse-covariance"),"incorrect reference class");
 let mut b:Background=campaign::load(&quiet_covariance.join("background.h5"))?;b.pulse_covariance=true;
 let audit:serde_json::Value=campaign::load(variation)?;
 let mut rows:Vec<(i64,usize,Vec<[f64;2]>)>=serde_json::from_value(audit["rows"].clone())?;
 let heldout=1715472000;rows.retain(|r|r.0.div_euclid(86400)*86400!=heldout);
 let(response,se,clusters)=lp480_fit::detrended_noise_response(&rows,b.acf[0][0]);
 ensure!(clusters>=12&&response.len()==lp480::NL+1,"insufficient independent variation calibration");
 ensure!((response[0][0]-1.).abs()<1e-8&&response[0][1].abs()<1e-8,"response zero-lag convention");
 let psd_min=|acf:&[[f64;2]]|{let mut a=vec![Complex64::default();1024];a[0]=Complex64::new(acf[0][0],0.);
  for l in 1..acf.len(){a[l]=Complex64::new(acf[l][0],acf[l][1]);a[1024-l]=a[l].conj();}
  FftPlanner::<f64>::new().plan_fft_forward(1024).process(&mut a);a.iter().map(|v|v.re).fold(f64::INFINITY,f64::min)};
 ensure!(psd_min(&response)>0.,"incremental response is not a nonnegative noise spectrum");
 b.variable_response=response;b.variable_response_sigma=se;
 b.method.push_str(" Affine continuum response calibrated from11/13/14May noise variations after removing separate window offsets and linear drifts;12May excluded and held out. Conditional dwell-cluster response errors; not a full parameter covariance.");
 std::fs::create_dir_all(out)?;campaign::save(&out.join("background.h5"),&b)?;
 let mut validation=vec![];
 for job in m.jobs.iter().filter(|j|j.el>89.&&j.start.div_euclid(86400)*86400==heldout){
  let d:Decoded=campaign::load(&quiet_covariance.join("products").join(format!("{}.h5",job.id)))?;
  for block in d.blocks.iter().filter(|v|v.noise_pairs.iter().all(|n|*n>0)){
   let measured=block.noise.iter().zip(&block.noise_pairs).map(|(v,n)|[v[0]/ *n as f64,v[1]/ *n as f64]).collect::<Vec<_>>();
   let n0=measured[0][0];let model=b.acf.iter().zip(&b.variable_response).map(|(q,r)|[q[0]+(n0-b.acf[0][0])*r[0],q[1]+(n0-b.acf[0][0])*r[1]]).collect::<Vec<_>>();
   ensure!(psd_min(&model)>0.,"held-out background spectrum is not physical");
   validation.push(json!({"time_utc_s":block.start,"window_utc_s":job.start,"noise_gain":n0/b.acf[0][0],"measured":measured,"quiet_scaled":b.acf.iter().map(|q|[q[0]*n0/b.acf[0][0],q[1]*n0/b.acf[0][0]]).collect::<Vec<_>>(),"affine":model}));
  }
 }
 campaign::save(&out.join("validation.h5"),&json!({"heldout_day_utc":"2024-05-12","training_dwell_clusters":clusters,"reference_zero_lag_adu2":b.acf[0][0],"response":b.variable_response,"response_conditional_sigma":b.variable_response_sigma,"validation":validation,"note":"Independent12May covariance observations are held out from the11/13/14May variation fit. Quiet covariance anchor is from the earlier pilot. No plasma fits; no confidence-coverage claim."}))?;
 Ok(())
}
pub fn report(covariance:&Path,original:&Path,quiet_covariance:&Path,out:&Path,calibration:Option<&Path>)->Result<()>{
 let marker:serde_json::Value=campaign::load(&quiet_covariance.join("sensitivity.h5"))?;
 ensure!(marker["mode"]=="pulse-covariance-pilot","reference must be derived from pulse covariance");
 let mut quiet:Background=campaign::load(&quiet_covariance.join("background.h5"))?;
 // The original pilot predates the explicit reference-kind field. Its
 // immutable mode marker and decoded schema jointly establish its origin.
 let reference_index:Manifest=campaign::load(&quiet_covariance.join("index.h5"))?;
 let first=reference_index.jobs.first().ok_or_else(||anyhow::anyhow!("empty covariance reference"))?;
 let reference_decoded:Decoded=campaign::load(&quiet_covariance.join("products").join(format!("{}.h5",first.id)))?;
 ensure!(reference_decoded.schema.contains("pulse-covariance"),"raw reference cannot be relabelled as covariance");
 quiet.pulse_covariance=true;
 if let Some(path)=calibration{let calibrated:Background=campaign::load(path)?;
  ensure!(calibrated.pulse_covariance&&calibrated.acf==quiet.acf&&calibrated.sigma==quiet.sigma&&calibrated.variable_response.len()==lp480::NL+1,"calibration/quiet reference mismatch");quiet=calibrated;}
 let old:Background=campaign::load(&original.join("background.h5"))?;
 let m:Manifest=campaign::load(&covariance.join("index.h5"))?;
 campaign::save(&covariance.join("background.h5"),&quiet)?;
 for name in ["calibration.h5","calibration_metadata.h5"]{if original.join(name).exists(){std::fs::copy(original.join(name),covariance.join(name))?;}}
 if !covariance.join("references").exists(){std::os::unix::fs::symlink(std::fs::canonicalize(original.join("references"))?,covariance.join("references"))?;}
 std::fs::create_dir_all(out)?;
 for job in &m.jobs{
  let d:Decoded=campaign::load(&covariance.join("products").join(format!("{}.h5",job.id)))?;
  ensure!(d.schema.contains("pulse-covariance")&&!d.blocks.is_empty(),"raw covariance blocks required");
  let baseline:Decoded=campaign::load(&original.join("products").join(format!("{}.h5",job.id)))?;
  let(before,og,ol)=lp480_fit::observations(&baseline,&old);let(after,ag,al)=lp480_fit::observations(&d,&quiet);
  ensure!(og==ag&&ol==al,"changed measurement support; inspect separately");
  let far=d.blocks.iter().filter(|b|b.noise_pairs.iter().all(|n|*n>0)).map(|b|{
   let measured=b.noise.iter().zip(&b.noise_pairs).map(|(v,n)|[v[0]/ *n as f64,v[1]/ *n as f64]).collect::<Vec<_>>();
   let scale=measured[0][0]/quiet.acf[0][0];
   let residual=measured.iter().zip(&quiet.acf).enumerate().map(|(l,(v,q))|{
    let prediction=if quiet.variable_response.is_empty(){[scale*q[0],scale*q[1]]}else{[q[0]+(measured[0][0]-quiet.acf[0][0])*quiet.variable_response[l][0],q[1]+(measured[0][0]-quiet.acf[0][0])*quiet.variable_response[l][1]]};
    [(v[0]-prediction[0])/measured[0][0],(v[1]-prediction[1])/measured[0][0]]}).collect::<Vec<_>>();
   json!({"time_utc_s":b.start,"zero_lag_adu2":measured[0][0],"acf":measured,"residual_over_zero_lag":residual})
  }).collect::<Vec<_>>();
  campaign::save(&out.join(format!("{}.h5",job.id)),&json!({"time_utc_s":job.start,"azimuth_deg":job.az,"elevation_deg":job.el,
   "gates":d.gates,"measurement_gate":ag,"measurement_lag":al,"old_observed":before,"corrected_observed":after,"covariance_far_noise":far,
   "old_quiet_acf":old.acf,"covariance_quiet_acf":quiet.acf,
   "method":"Receiver-only correction: identical full-window satellite screen, unbiased pulse covariance removes a static coherent voltage, per-block continuum scaling with matching gate/lag weights; independent quiet covariance reference. No plasma fits. Upper-range plasma and changing interference remain explicit checks."}))?;
 }
 lp480_fit::compact(covariance,&covariance.join("corrected-input"))?;
 if !covariance.join("corrected-input/references").exists(){std::os::unix::fs::symlink(std::fs::canonicalize(original.join("references"))?,covariance.join("corrected-input/references"))?;}
 for name in ["calibration.h5","calibration_metadata.h5"]{if original.join(name).exists(){std::fs::copy(original.join(name),covariance.join("corrected-input").join(name))?;}}
 Ok(())
}
#[cfg(test)]mod tests{
 use super::*;
 #[test]fn coherent_line_is_removed_without_erasing_incoherent_variance(){
  let fields=(0..lp480::NS).map(|k|Complex32::from_polar(4.,std::f32::consts::TAU*7.*k as f32/128.)).collect::<Vec<_>>();
  let frames=[-3.,-1.,1.,3.].iter().map(|n|fields.iter().enumerate().map(|(k,c)|*c+Complex32::new(*n*if k%2==0{1.}else{-1.},0.)).collect()).collect::<Vec<Vec<_>>>();
  let(raw,cov)=spectra(&frames);let(_,reference)=spectra(&frames.iter().map(|x|x.iter().zip(&fields).map(|(a,b)|*a-*b).collect()).collect::<Vec<_>>());
  for (c,r)in cov.iter().flatten().zip(reference.iter().flatten()){assert!((c-r).abs()<1e-5*c.abs().max(1.));}
  assert!(raw[0].iter().zip(&cov[0]).any(|(a,b)|a-b>100.));
 }
 #[test]fn tail_windows_stay_inside_record_and_cover_reference(){for k in windows(){assert!(k+128<=lp480::NS);}assert!(windows()[3]>=lp480::NS-160);}
}
