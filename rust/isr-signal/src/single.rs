//! Native float32 / complex64 kernels for wideband recording processing.
use num_complex::Complex32 as C;
pub fn lag_products(z:&[C],lag:usize,d:usize)->Result<Vec<C>,&'static str>{
 if d==0 || lag>=z.len(){return Err("invalid lag or decimation");}
 cross_products(&z[..z.len()-lag],&z[lag..],d)
}
/// Sum a * conj(b) in nonoverlapping bins, discarding an incomplete final bin.
pub fn cross_products(a:&[C],b:&[C],d:usize)->Result<Vec<C>,&'static str>{
 if d==0 || a.len()!=b.len(){return Err("invalid cross-product shape");}
 let mut out=vec![C::default();a.len()/d];
 #[cfg(target_arch="x86_64")]
 if std::is_x86_feature_detected!("avx") {unsafe{avx(a,b,d,&mut out)};return Ok(out);}
 for(g,y)in out.iter_mut().enumerate(){for i in g*d..(g+1)*d{*y+=a[i]*b[i].conj();}}
 Ok(out)
}
#[cfg(target_arch="x86_64")]
#[target_feature(enable="avx")]
unsafe fn avx(za:&[C],zb:&[C],d:usize,out:&mut[C]){
 use std::arch::x86_64::*;
 let p=za.as_ptr().cast::<f32>();let q=zb.as_ptr().cast::<f32>();
 for(g,y)in out.iter_mut().enumerate(){
 let mut i=g*d;let end=i+d;
 let mut r=_mm256_setzero_ps();let mut c=_mm256_setzero_ps();
 while i+3<end{
 let a=_mm256_loadu_ps(p.add(2*i));let b=_mm256_loadu_ps(q.add(2*i));
 r=_mm256_add_ps(r,_mm256_mul_ps(a,b));
 c=_mm256_add_ps(c,_mm256_mul_ps(a,_mm256_permute_ps(b,0b10110001)));
 i+=4;
 }
 let mut rr=[0.;8];let mut cc=[0.;8];
 _mm256_storeu_ps(rr.as_mut_ptr(),r);_mm256_storeu_ps(cc.as_mut_ptr(),c);
 for k in 0..4{*y+=C::new(rr[2*k]+rr[2*k+1],cc[2*k+1]-cc[2*k]);}
 for j in i..end{*y+=za[j]*zb[j].conj();}
 }
}
#[cfg(test)]
mod tests{
 use super::*;
 #[test]fn reference_precision(){
 let z:Vec<_>=(0..1027).map(|i|C::new((i as f32*0.13).sin()*32767.,(i as f32*0.71).cos()*32767.)).collect();
 let reference:Vec<_>=z.iter().map(|v|num_complex::Complex64::new(v.re as f64,v.im as f64)).collect();
 for lag in [0,1,13,301,1026]{for d in [1,3,14,31,250,2048]{
 let got=lag_products(&z,lag,d).unwrap();let want=crate::lag_products(&reference,lag,d).unwrap();
 for(a,b)in got.iter().zip(want){let error=(num_complex::Complex64::new(a.re as f64,a.im as f64)-b).norm();assert!(error<2e-6*d as f64*32767_f64.powi(2),"{error}");}
 }}
 }
}
