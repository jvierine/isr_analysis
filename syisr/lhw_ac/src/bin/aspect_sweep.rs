//! Pencil-beam angular sweep using the existing memo-50 Brownian response.
//! Source of included response: HPLA memo050/lower-hybrid/model/coulomb_response.inc.rs.
//! Constant friction/diffusion is not a full velocity-dependent Landau operator.
use anyhow::{ensure,Result};
use num_complex::Complex64 as C;
use rayon::prelude::*;
use std::{env,f64::consts::PI};
const E:f64=1.602176634e-19;const ME:f64=9.1093837139e-31;const KB:f64=1.380649e-23;const EPS:f64=8.8541878188e-12;const U:f64=1.66053906892e-27;
const K:f64=4.*PI*430e6/299792458.;
include!("../coulomb_ou.inc.rs");
fn calculate(alpha:f64,df:f64,refine:usize)->Vec<f64>{
 let f:Vec<_>=(0..=(50000./df) as usize).map(|i|i as f64*df).collect();
 let n=3e11;let b=40e-6;
 let species=[(n*0.99,1000.,16.*U,2.07353708,2.07353708),(n*0.01,1000.,U,11.63749802,11.63749802),(n,2000.,ME,182.79855716,312.05665652)];
 let responses:Vec<_>=species.iter().map(|&(n,t,m,p,q)|coulomb_response(&f,n,t,m,p,q,0.,b,alpha,refine)).collect();
 (0..f.len()).map(|j|{let ce=responses[2][j].0;let ci=responses[0][j].0+responses[1][j].0;let qi=0.99*responses[0][j].1+0.01*responses[1][j].1;
 let eps=C::new(1.,0.)+ce+ci;let s=((C::new(1.,0.)+ci).norm_sqr()*responses[2][j].1+ce.norm_sqr()*qi)/eps.norm_sqr();assert!(s.is_finite()&&s>=0.);s}).collect()
}
fn fold(s:&[f64],df:f64)->Vec<f64>{
 let half=s.len()-1;let n=2*half;let mut bins=vec![C::new(0.,0.);n];
 for j in 0..n{let k=if j<=half{j}else{n-j};bins[j]=C::new(s[k],0.);}
 rustfft::FftPlanner::<f64>::new().plan_fft_inverse(n).process(&mut bins);
 let mut lag=vec![C::new(0.,0.);512];lag[0]=bins[0]*df;
 for j in 1..63{let v=bins[j]*df*(1.-j as f64/63.);lag[j]=v;lag[512-j]=v.conj();}
 rustfft::FftPlanner::<f64>::new().plan_fft_forward(512).process(&mut lag);
 (0..512).map(|j|lag[(j+256)%512].re*1e-5).collect()
}
fn main()->Result<()>{
 let a:Vec<_>=env::args().collect();let df=a.get(2).map(|x|x.parse::<f64>()).transpose()?.unwrap_or(1.);
 let refine=a.get(3).map(|x|x.parse::<usize>()).transpose()?.unwrap_or(1);
 let angle:Vec<f64>=if let Some(list)=a.get(4){list.split(',').map(|v|v.parse().unwrap()).collect()}else{(0..=40).map(|i|i as f64*0.05).collect()};
 let rows:Vec<_>=angle.par_iter().map(|&alpha|{let s=calculate(alpha,df,refine);let windowed=fold(&s,df);println!("ANGLE {alpha:.2} integral {}",s.iter().sum::<f64>()*2.*df-s[0]*df);(s,windowed)}).collect();
 let h=hdf5::File::create(&a[1])?;h.new_dataset_builder().with_data(&angle).create("aspect_from_perpendicular_deg")?;
 let f:Vec<_>=(0..rows[0].0.len()).map(|j|j as f64*df).collect();h.new_dataset_builder().with_data(&f).create("native_frequency_hz")?;
 let ff:Vec<_>=(0..512).map(|j|(j as f64-256.)*1e5/512.).collect();h.new_dataset_builder().with_data(&ff).create("windowed_frequency_hz")?;
 for (key,n,which) in [("native_psd_per_hz",f.len(),0),("triangular_psd_per_hz",512,1)]{let v:Vec<_>=rows.iter().flat_map(|r|if which==0{r.0.clone()}else{r.1.clone()}).collect();h.new_dataset::<f64>().shape([angle.len(),n]).deflate(4).create(key)?.write_raw(&v)?;}
 let metadata=serde_json::json!({"model":"Magnetized Brownian OU/Fokker-Planck displacement; constant parallel/perpendicular collision rates; thermal electrons and magnetized O+/H+ fluctuation terms","beam_average":false,"ne_m3":3e11,"Te_K":2000.,"Ti_K":1000.,"B_T":40e-6,"radar_hz":430e6,"O_fraction":0.99,"H_fraction":0.01,"neutral_collisions":0.,"species_parallel_collision_s":[2.07353708,11.63749802,182.79855716],"species_perpendicular_collision_s":[2.07353708,11.63749802,312.05665652],"frequency_step_hz":df,"time_refinement":refine,"window":"10 us lag spacing; lags 0..62; triangular 1-lag/63; ideal +/-50 kHz receiver cutoff; no range or beam convolution","limitations":"Illustrative homogeneous thermal plasma, not a fit; no suprathermal enhancement; constant Brownian friction is an approximation to Coulomb kinetics; an angular sweep alone does not give a pointing confidence interval"});
 h.new_dataset_builder().with_data(metadata.to_string().as_bytes()).create("metadata_json_utf8")?;
 ensure!(rows.iter().all(|r|r.1.iter().all(|v|v.is_finite())),"nonfinite folded spectrum");Ok(())
}
