//! Batched real-spectrum/complex-response products. Each kernel is read once.
use num_complex::Complex64 as C;
pub fn reference_into(kernel: &[C], channels: &[Vec<f64>], out: &mut [C]) {
    assert_eq!(channels.len(), out.len());
    for (o, s) in out.iter_mut().zip(channels) {
        assert_eq!(kernel.len(), s.len());
        *o = kernel.iter().zip(s).map(|(k, s)| *k * *s).sum();
    }
}
pub fn fused_into(kernel: &[C], channels: &[Vec<f64>], out: &mut [C]) {
    assert!(channels.len() <= 7);
    assert_eq!(channels.len(), out.len());
    assert!(channels.iter().all(|s| s.len() == kernel.len()));
    let mut a = [C::new(0., 0.); 7];
    for (i, k) in kernel.iter().enumerate() {
        for (c, s) in channels.iter().enumerate() {
            a[c].re += k.re * s[i];
            a[c].im += k.im * s[i];
        }
    }
    out.copy_from_slice(&a[..out.len()]);
}
#[cfg(test)]
mod tests {
    use super::*;
    #[test]
    fn preserves_signed_complex_products_and_all_tails() {
        for n in [0, 1, 2, 3, 7, 31, 201, 257] {
            for count in 1..=7 {
                let k: Vec<_> = (0..n)
                    .map(|j| C::new((j as f64 * 0.17).sin(), (j as f64 * 0.23).cos()))
                    .collect();
                let s: Vec<Vec<f64>> = (0..count)
                    .map(|c| {
                        (0..n)
                            .map(|j| {
                                ((j + c) as f64 * 0.11).cos() * if j % 2 == 0 { 1e5 } else { 1e-5 }
                            })
                            .collect()
                    })
                    .collect();
                let mut a = vec![C::default(); count];
                let mut b = a.clone();
                reference_into(&k, &s, &mut a);
                fused_into(&k, &s, &mut b);
                for (x, y) in a.iter().zip(b) {
                    assert!((*x - y).norm() < 1e-10 * x.norm().max(1.));
                }
            }
        }
    }
}
