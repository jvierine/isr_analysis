use anyhow::Result;
use std::path::PathBuf;
fn main()->Result<()>{let a:Vec<String>=std::env::args().collect();let root=PathBuf::from(&a[1]);let f=hdf5::File::open(root.join("interference-0619-blocks/block-08.h5"))?;let p=f.dataset("whitened_matched_power_ratio")?.read_raw::<f64>()?;let r=f.dataset("slant_range_km")?.read_raw::<f64>()?;let freq=f.dataset("frequency_offset_hz")?.read_raw::<f64>()?;let nf=freq.len();
for control in ["plus1","minus1"]{let f=hdf5::File::open(root.join(format!("interference-0619-phase-{control}/block-08.h5")))?;let q=f.dataset("whitened_matched_power_ratio")?.read_raw::<f64>()?;
 for &(lo,hi)in &[(-6.03e6,-5.97e6),(-3e6,2e6),(3e6,7.5e6)]{let mut s=0.;let mut t=0.;for(j,&rr)in r.iter().enumerate(){if !(100. ..=800.).contains(&rr){continue;}for(k,&ff)in freq.iter().enumerate(){if ff>=lo&&ff<=hi{s+=p[j*nf+k];t+=q[j*nf+k];}}}println!("control {control}: band {}..{} MHz, correct/control={:.4} ({:.3}dB)",lo/1e6,hi/1e6,s/t,10.*(s/t).log10());}
 for &rr in &[150.,230.,300.,470.,600.]{let j=r.iter().enumerate().min_by(|(_,x),(_,y)|(*x-rr).abs().total_cmp(&(*y-rr).abs())).unwrap().0;let mut s=0.;let mut t=0.;for(k,&ff)in freq.iter().enumerate(){if(ff+6e6).abs()<20000.{s+=p[j*nf+k];t+=q[j*nf+k];}}println!("control {control}: {rr}km carrier-band ratio {:.3}dB",10.*(s/t).log10());}
}Ok(())}
