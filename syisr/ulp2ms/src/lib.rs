//! 2-ms SYISR long-pulse diagnostics. Provenance stays in code/HDF5.
pub mod iqr_screen;
pub mod reader;
pub mod screen;
use num_complex::Complex64 as C;
pub const PULSE: usize = 200;
pub const LAGS: usize = 61;
pub const NU: usize = 21;
/// Unit-area scatter-range weighting, from direct enumeration of transmit
/// overlap and summation cells. Same extent and centroid, not identical shape.
pub fn kernel(lag: usize) -> Vec<f64> {
    let mut k = vec![0.; PULSE + NU - 1];
    for j in 0..lag + NU {
        for s in j..j + PULSE - lag {
            k[s] += 1.;
        }
    }
    let sum: f64 = k.iter().sum();
    k.iter_mut().for_each(|v| *v /= sum);
    k
}
/// sum_j x[j-l] conj(x[j]), j=i..i+l+nu-1; remove background BEFORE
/// pulse-overlap correction. Return equivalent full-pulse signal power.
pub fn gate(profile: &[C], i: usize, lag: usize, noise: C) -> C {
    let sum: C = profile[i - lag..i + NU].iter().sum();
    (sum / (lag + NU) as f64 - noise) * PULSE as f64 / (PULSE - lag) as f64
}
#[cfg(test)]
mod tests {
    use super::*;
    #[test]
    fn common_support_centroid_and_area() {
        for l in 0..LAGS {
            let k = kernel(l);
            assert_eq!(k.len(), 220);
            assert!((k.iter().sum::<f64>() - 1.).abs() < 1e-12);
            let center = k.iter().enumerate().map(|(i, v)| i as f64 * v).sum::<f64>();
            assert!((center - 109.5).abs() < 1e-10);
            assert!(k.iter().all(|v| *v > 0.));
        }
    }
    #[test]
    fn uniform_scatter_and_colored_noise_recovery() {
        for l in 0..LAGS {
            let signal = C::new((-0.015 * l as f64).exp(), 0.1 * l as f64);
            let noise = C::new(30. * (-0.3 * l as f64).exp(), -0.4);
            let profile = vec![signal * (PULSE - l) as f64 / PULSE as f64 + noise; 500];
            assert!((gate(&profile, 200, l, noise) - signal).norm() < 1e-12);
        }
    }
    #[test]
    fn nonuniform_scatter_matches_explicit_kernel() {
        for l in 0..LAGS {
            let i = 250;
            let sky: Vec<f64> = (0..700)
                .map(|j| (-((j as f64 - 240.) / 40.).powi(2)).exp() + 0.2)
                .collect();
            let profile: Vec<C> = (0..500)
                .map(|k| {
                    let sum: f64 = (k + l + 1..=k + PULSE).map(|s| sky[s]).sum();
                    C::new(sum / PULSE as f64, 0.)
                })
                .collect();
            // Product convention here uses reversed scatter coordinate; symmetric kernels
            // have identical weights, so the independently enumerated convolution agrees.
            let want: f64 = kernel(l)
                .iter()
                .enumerate()
                .map(|(s, w)| w * sky[i + 1 + s])
                .sum();
            assert!((gate(&profile, i, l, C::new(0., 0.)).re - want).abs() < 1e-12);
        }
    }
}

pub mod retrieval;
pub mod theory;

pub mod fit_response;

pub mod ambiguity_audit;
pub mod geomagnetic;
pub mod geometry;
pub mod measurement_covariance;
pub mod noise_blocks;
pub mod profile;
pub mod profile_response;
pub mod structured_covariance;

pub mod composition;

pub mod temporal;
