use crate::C;
use anyhow::{ensure, Result};
use rustfft::FftPlanner;
use std::f64::consts::PI;
/// Linear, zero-padded FIR; the sample origin is unchanged. Unlike circular FFT
/// filtering, an edge cannot wrap into another interpulse gate.
pub fn lowpass(z: &[C], sample_rate: f64, cutoff: f64, half: usize) -> Result<Vec<C>> {
    ensure!(
        sample_rate.is_finite()
            && cutoff.is_finite()
            && sample_rate > 0.
            && cutoff > 0.
            && cutoff < sample_rate / 2.
            && half > 0,
        "invalid FIR configuration"
    );
    let mut h: Vec<f64> = (0..2 * half)
        .map(|j| {
            let m = j as f64 - half as f64 + 1e-6;
            let window = 0.5 - 0.5 * (2. * PI * j as f64 / (2 * half - 1) as f64).cos();
            window * (2. * PI * cutoff / sample_rate * m).sin() / (PI * m)
        })
        .collect();
    let sum = h.iter().sum::<f64>();
    for v in &mut h {
        *v /= sum
    }
    let mut out = vec![C::default(); z.len()];
    for (i, y) in out.iter_mut().enumerate() {
        for (j, &v) in h.iter().enumerate() {
            let k = i as isize + half as isize - j as isize;
            if k >= 0 && (k as usize) < z.len() {
                *y += z[k as usize] * v
            }
        }
    }
    Ok(out)
}
pub fn decimate_sum(z: &[C], factor: usize) -> Result<Vec<C>> {
    ensure!(factor > 0, "zero decimation");
    Ok(z.chunks_exact(factor).map(|x| x.iter().sum()).collect())
}
/// Signed sample lag convention R(l)=x(t)conj(x(t+l)), matching the Millstone chain.
pub fn lag_products(z: &[C], lag: usize, decimation: usize) -> Result<Vec<C>> {
    isr_signal::lag_products(z, lag, decimation).map_err(anyhow::Error::msg)
}
/// Exact linear-interpolated percentile, as NumPy's default percentile method.
pub fn percentile(values: &[f64], q: f64) -> f64 {
    let mut a: Vec<_> = values.iter().copied().filter(|x| x.is_finite()).collect();
    a.sort_by(f64::total_cmp);
    if a.is_empty() {
        return f64::NAN;
    }
    let p = q.clamp(0., 1.) * (a.len() - 1) as f64;
    let i = p.floor() as usize;
    let f = p - i as f64;
    a[i] * (1. - f) + a[(i + 1).min(a.len() - 1)] * f
}
/// Hann-windowed matched-filter range–Doppler power, signed fftshift frequency.
pub fn range_doppler(echo: &[C], tx: &[C], ranges: &[usize], fft_len: usize) -> Result<Vec<f64>> {
    ensure!(
        tx.len() > 1 && fft_len >= tx.len() && ranges.iter().all(|&r| r + tx.len() <= echo.len()),
        "invalid range-Doppler support"
    );
    let mut planner = FftPlanner::<f64>::new();
    let fft = planner.plan_fft_forward(fft_len);
    let mut out = Vec::with_capacity(ranges.len() * fft_len);
    for &r in ranges {
        let mut z = vec![C::default(); fft_len];
        for j in 0..tx.len() {
            let w = 0.5 - 0.5 * (2. * PI * j as f64 / (tx.len() - 1) as f64).cos();
            z[j] = w * tx[j].conj() * echo[r + j];
        }
        fft.process(&mut z);
        for j in 0..fft_len {
            out.push(z[(j + fft_len.div_ceil(2)) % fft_len].norm_sqr())
        }
    }
    Ok(out)
}
#[cfg(test)]
mod tests {
    use super::*;
    #[test]
    fn filter_has_no_wrap() {
        let mut z = vec![C::default(); 100];
        z[99] = C::new(1., 0.);
        let y = lowpass(&z, 1e6, 60e3, 10).unwrap();
        assert!(y[..80].iter().all(|v| v.norm() < 1e-15));
    }
    #[test]
    fn decimation_preserves_sum() {
        let z: Vec<_> = (0..120).map(|i| C::new(i as f64, -1.)).collect();
        assert_eq!(
            decimate_sum(&z, 30).unwrap().iter().sum::<C>(),
            z.iter().sum::<C>()
        );
    }
    #[test]
    fn dft_sign_and_exact_power() {
        let n = 64;
        let tx = vec![C::new(1., 0.); n];
        let z: Vec<_> = (0..n)
            .map(|j| C::from_polar(1., 2. * PI * 7. * j as f64 / n as f64))
            .collect();
        let p = range_doppler(&z, &tx, &[0], n).unwrap();
        assert_eq!(
            p.iter()
                .enumerate()
                .max_by(|a, b| a.1.total_cmp(b.1))
                .unwrap()
                .0,
            n / 2 + 7
        );
        assert!((p[n / 2 + 7] - ((n - 1) as f64 / 2.).powi(2)).abs() < 1e-8);
    }
}
