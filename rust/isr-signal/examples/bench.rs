use num_complex::Complex64 as C;
use std::{hint::black_box,time::Instant};
fn main(){
 let n=250000;let pulses=16;let d=50;let lags:Vec<_>=(0..32).map(|l|l*250).collect();
 let z:Vec<_>=(0..n*pulses).map(|i|C::new((i as f64*0.17).sin(),(i as f64*0.31).cos())).collect();
 for trial in 0..3 {
 let start=Instant::now();
 for p in z.chunks_exact(n){for &l in &lags{let tmp:Vec<_>=p[..n-l].iter().zip(&p[l..]).map(|(a,b)|a*b.conj()).collect();let y:Vec<C>=tmp.chunks_exact(d).map(|v|v.iter().sum()).collect();black_box(y);}}
 let old=start.elapsed();let start=Instant::now();
 for p in z.chunks_exact(n){for &l in &lags{black_box(isr_signal::lag_products(p,l,d).unwrap());}}
 println!("trial {trial}: original={old:?}, SIMD={:?}",start.elapsed());
 }
 #[cfg(feature="cuda")]{
 let gpu=isr_signal::cuda::CudaLagProducts::new(0,256*1024*1024).unwrap();
 for trial in 0..3{let start=Instant::now();black_box(gpu.products(&z,n,&lags,d).unwrap());println!("trial {trial}: GPU including upload/download={:?}",start.elapsed());}
 }
}
