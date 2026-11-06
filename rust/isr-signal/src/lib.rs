//! Fused signed lag products. No clipping, interpolation, or sample reordering.
//! R(l) = x(t) conj(x(t+l)); incomplete decimation bins are discarded.
use num_complex::Complex64 as C;
#[cfg(feature = "cuda")]
pub mod cuda;

pub fn lag_products(z: &[C], lag: usize, decimation: usize) -> Result<Vec<C>, &'static str> {
    if decimation == 0 || lag >= z.len() { return Err("invalid lag or decimation"); }
    let n = (z.len() - lag) / decimation;
    let mut out = vec![C::default(); n];
    #[cfg(target_arch = "x86_64")]
    if std::is_x86_feature_detected!("avx") {
        // SAFETY: runtime feature check; implementation bounds both input slices.
        unsafe { avx(z, lag, decimation, &mut out); }
        return Ok(out);
    }
    scalar(z, lag, decimation, &mut out);
    Ok(out)
}
fn scalar(z: &[C], lag: usize, d: usize, out: &mut [C]) {
    for (g, y) in out.iter_mut().enumerate() {
        for i in g*d..(g+1)*d { *y += z[i] * z[i+lag].conj(); }
    }
}
#[cfg(target_arch = "x86_64")]
#[target_feature(enable = "avx")]
unsafe fn avx(z: &[C], lag: usize, d: usize, out: &mut [C]) {
    use std::arch::x86_64::*;
    // Complex64 is repr(C), with adjacent real and imaginary f64 components.
    let p = z.as_ptr().cast::<f64>();
    for (g, y) in out.iter_mut().enumerate() {
        let start = g*d; let end = start+d;
        let mut i = start;
        let mut direct_sum = _mm256_setzero_pd();
        let mut cross_sum = _mm256_setzero_pd();
        while i+1 < end {
            let a = _mm256_loadu_pd(p.add(2*i));
            let b = _mm256_loadu_pd(p.add(2*(i+lag)));
            let direct = _mm256_mul_pd(a,b);
            let cross = _mm256_mul_pd(a,_mm256_permute_pd(b,0b0101));
            direct_sum = _mm256_add_pd(direct_sum, direct);
            cross_sum = _mm256_add_pd(cross_sum, cross);
            i+=2;
        }
        let mut r=[0.;4]; let mut c=[0.;4];
        _mm256_storeu_pd(r.as_mut_ptr(),direct_sum);
        _mm256_storeu_pd(c.as_mut_ptr(),cross_sum);
        *y=C::new((r[0]+r[1])+(r[2]+r[3]),(c[1]-c[0])+(c[3]-c[2]));
        if i < end { *y += z[i]*z[i+lag].conj(); }
    }
}
#[cfg(test)]
mod tests {
    use super::*;
    #[test]
    fn scalar_equivalence_tails_and_sign() {
        let z:Vec<_>=(0..257).map(|i|C::new((i as f64*0.71).sin(),(i as f64*0.13).cos())).collect();
        for lag in [0,1,7,128,256] { for d in [1,2,3,14,31,512] {
            let actual=lag_products(&z,lag,d).unwrap();
            let mut expected=vec![C::default();actual.len()];
            scalar(&z,lag,d,&mut expected);
            for (a,b) in actual.iter().zip(expected) { assert!((*a-b).norm()<1e-12*(1.+b.norm()),"lag={lag}, d={d}"); }
        }}
        assert_eq!(lag_products(&[C::new(1.,0.),C::new(0.,1.)],1,1).unwrap(),[C::new(0.,-1.)]);
    }
    #[test] fn invalid_and_nonfinite() {
        assert!(lag_products(&[],0,1).is_err());
        assert!(lag_products(&[C::default()],0,0).is_err());
        assert!(lag_products(&[C::new(f64::NAN,0.)],0,1).unwrap()[0].re.is_nan());
    }
}

pub mod single;

#[cfg(feature = "cuda")]
pub mod cuda32;
