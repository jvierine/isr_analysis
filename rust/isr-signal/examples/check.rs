use isr_signal::lag_products;
use num_complex::Complex64 as C;
fn main(){
 let n=32771;let pulses=16;
 let z:Vec<_>=(0..n*pulses).map(|i|C::new((i as f64*0.71).sin(),(i as f64*0.13).cos())).collect();
 let lags=[0,1,7,14,131,32770];
 #[cfg(feature="cuda")]{
 let gpu=isr_signal::cuda::CudaLagProducts::new(0,128*1024*1024).unwrap();
 for d in [1,3,14,31]{
 let actual=gpu.products(&z,n,&lags,d).unwrap();
 for (l,got) in lags.iter().zip(actual){
 let want:Vec<_>=z.chunks_exact(n).flat_map(|p|lag_products(p,*l,d).unwrap()).collect();
 assert_eq!(got.len(),want.len());
 let err=got.iter().zip(&want).map(|(a,b)|(*a-*b).norm()).fold(0.,f64::max);
 assert!(err<1e-11,"lag={l}, d={d}, error={err}");
 println!("GPU validated: lag={l} decimation={d} gates={} max_error={err:e}",got.len());
 }
 }
 }
 let start=std::time::Instant::now();
 let mut bins=0;for p in z.chunks_exact(n){for l in lags{bins+=lag_products(p,l,14).unwrap().len();}}
 println!("CPU {} bins in {:?}",bins,start.elapsed());
}
