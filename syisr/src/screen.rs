//! Code/Doppler-aware transient screening. No decoded amplitude-outlier rejection.
//! Full sampled Doppler band is searched every fourth pulse; masks span the
//! bracketing pulses, full640us waveform and receiver margins. Strong single-
//! pulse power bursts and ADC clipping are checked on every pulse.
use num_complex::{Complex32 as C32,Complex64 as C};
use rustfft::FftPlanner;
use serde::{Serialize,Deserialize};
#[derive(Clone,Serialize,Deserialize)]pub struct Summary{pub masked:usize,pub total:usize,pub candidates:usize,pub probes:usize,pub max_fraction:f64,pub doppler_step_hz:f64,pub pulse_stride:usize}
pub fn median(x:&mut[f64])->f64{let n=x.len();x.select_nth_unstable_by(n/2,|a,b|a.total_cmp(b));x[n/2]}
pub fn screen(x:&[C32],np:usize,n:usize,codes:&[Vec<f64>],stride:usize)->(Vec<u8>,Summary){
 let mut mask=vec![0u8;np*n];let nb=n/8;let mut power=vec![0f64;np*nb];let mut reference=vec![0.;nb];
 for p in 0..np{for b in 0..nb{power[p*nb+b]=x[p*n+b*8..p*n+(b+1)*8].iter().map(|v|v.norm_sqr()as f64).sum::<f64>()/8.;}}
 for b in 0..nb{let mut v:Vec<_>=(0..np).map(|p|power[p*nb+b]).collect();reference[b]=median(&mut v).max(1e-20);}
 let dilate=|mask:&mut[u8],p:usize,k:usize,len:usize,margin:usize,reason:u8|{for q in p.saturating_sub(margin)..=(p+margin).min(np-1){for j in k.saturating_sub(12)..(k+len+12).min(n){mask[q*n+j]|=reason;}}};
 for p in 0..np{if x[p*n..(p+1)*n].iter().any(|z|z.re.abs()>2.1e9||z.im.abs()>2.1e9){mask[p*n..(p+1)*n].fill(8);continue;}for b in 0..nb{if power[p*nb+b]>64.*reference[b]{dilate(&mut mask,p,b*8,8,3,1);}}}
 let nf=2048;let mut planner=FftPlanner::<f32>::new();let fft=planner.plan_fft_forward(nf);let ifft=planner.plan_fft_inverse(nf);
 let mut templates=vec![];let df=1000.;
 for c in codes{let mut bank=vec![];for f in -50..50{let freq=f as f64*df;let mut t=vec![C32::default();nf];for k in 0..64{let a=C::from_polar(c[k/2],std::f64::consts::TAU*freq*k as f64*1e-5);t[k]=C32::new(a.re as f32,a.im as f32);}fft.process(&mut t);bank.push(t);}templates.push(bank);}
 let mut candidates=0;let mut max_fraction=0f64;let mut probes=0;
 for p in (0..np).step_by(stride){let cid=p%64;let mut y=vec![C32::default();nf];y[..n].copy_from_slice(&x[p*n..(p+1)*n]);fft.process(&mut y);let mut peak=vec![0f64;n-63];
 // Sliding energy uses f64: large early coherent voltages otherwise destroy
 // float32 cumulative-sum precision at quiet ranges.
 let mut energy=vec![0.;n+1];for k in 0..n{energy[k+1]=energy[k]+x[p*n+k].norm_sqr()as f64;}
 for t in &templates[cid]{let mut a:Vec<_>=y.iter().zip(t).map(|(a,b)|*a*b.conj()).collect();ifft.process(&mut a);for k in 0..peak.len(){peak[k]=peak[k].max(a[k].norm_sqr()as f64/(nf*nf*64)as f64);}}
 probes+=1;
 for k in 0..peak.len(){let frac=peak[k]/(energy[k+64]-energy[k]).max(1e-20);max_fraction=max_fraction.max(frac);
 // High-confidence code-matched enhancements are excluded with temporal
 // bracketing, rather than interpolated or subtracted from plasma products.
 if frac>0.30{candidates+=1;dilate(&mut mask,p,k,64,stride+3,2);}}
 }
 let summary=Summary{masked:mask.iter().filter(|v|**v!=0).count(),total:mask.len(),candidates,probes,max_fraction,doppler_step_hz:df,pulse_stride:stride};(mask,summary)
}
#[cfg(test)]mod tests{use super::*;#[test]fn f64_energy_keeps_quiet_tail(){let mut e=0f64;e+=1e15;let a=e;e+=6000.;assert_eq!(e-a,6000.);}}
