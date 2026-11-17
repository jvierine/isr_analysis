//! Voltage-filter + finite rectangular pulse + actual trapezoidal summation.
//! Local plasma shape is constant inside the broad footprint; signal strength
//! follows a local exponential. This is a first broad-volume fit, not a profile inversion.
use num_complex::Complex64 as C;
use std::f64::consts::PI;
pub fn receiver_fir(rho: &[C]) -> (Vec<(f64, C)>, f64) {
    let n = 256;
    let psd: Vec<f64> = (0..n)
        .map(|k| {
            1. + (1..rho.len())
                .map(|l| {
                    2. * (rho[l] * C::from_polar(1., 2. * PI * k as f64 * l as f64 / n as f64)).re
                })
                .sum::<f64>()
        })
        .collect();
    let negative = psd.iter().filter(|&&v| v < 0.).map(|v| -v).sum::<f64>()
        / psd.iter().map(|v| v.abs()).sum::<f64>();
    let root: Vec<f64> = psd.iter().map(|v| v.max(1e-10).sqrt()).collect();
    let mut taps: Vec<(f64, C)> = (-20..=20)
        .map(|j| {
            (
                j as f64 * 1e-5,
                root.iter()
                    .enumerate()
                    .map(|(k, &v)| {
                        C::from_polar(v / n as f64, 2. * PI * k as f64 * j as f64 / n as f64)
                    })
                    .sum(),
            )
        })
        .collect();
    let norm = taps.iter().map(|(_, v)| v.norm_sqr()).sum::<f64>().sqrt();
    for (_, v) in &mut taps {
        *v /= norm;
    }
    (taps, negative)
}
pub fn kernel(frequency: &[f64], fir: &[(f64, C)], gamma: f64) -> Vec<Vec<C>> {
    let df = frequency[1] - frequency[0];
    let width = 0.002;
    (0..61)
        .map(|l| {
            let tau = l as f64 * 1e-5;
            let n = l + 21;
            let gate = (0..n)
                .map(|j| (gamma * 1.49896229 * (j as f64 - (n - 1) as f64 / 2.)).exp())
                .sum::<f64>()
                / n as f64;
            let mut weights = vec![(0., C::default()); 81];
            for &(a, ha) in fir {
                for &(b, hb) in fir {
                    let t = tau + a - b;
                    let length = (width - t.abs()).max(0.);
                    let d = gamma * 299792.458 * length / 4.;
                    let sinhc = if d.abs() < 1e-8 { 1. } else { d.sinh() / d };
                    let weight = ha
                        * hb.conj()
                        * (length / width)
                        * sinhc
                        * (-gamma * 299792.458 * (a + b) / 4.).exp()
                        * gate
                        / (1. - tau / width);
                    let i = ((a - b) / 1e-5).round() as isize + 40;
                    weights[i as usize].0 = t;
                    weights[i as usize].1 += weight;
                }
            }
            frequency
                .iter()
                .enumerate()
                .map(|(j, &f)| {
                    weights
                        .iter()
                        .map(|(t, w)| *w * C::from_polar(1., -2. * PI * f * t))
                        .sum::<C>()
                        * df
                        * if j == 0 || j + 1 == frequency.len() {
                            0.5
                        } else {
                            1.
                        }
                })
                .collect()
        })
        .collect()
}
#[cfg(test)]
mod tests {
    use super::*;
    #[test]
    fn delta_receiver_matches_enumerated_range_kernels() {
        let f = [-10000., 0., 10000.];
        for gamma in [-0.015, 0., 0.01] {
            let k = kernel(&f, &[(0., C::new(1., 0.))], gamma);
            for l in 0..61 {
                // Continuum scatter integral vs exact discrete unit-area box convolution.
                let discrete: f64 = crate::kernel(l)
                    .iter()
                    .enumerate()
                    .map(|(i, w)| w * (gamma * 1.49896229 * (i as f64 - 109.5)).exp())
                    .sum();
                let predicted = k[l][1].re / 10000.;
                assert!((predicted / discrete - 1.).abs() < 3e-5);
            }
        }
    }
}
#[cfg(test)]
mod recovery_tests {
    use super::*;
    use crate::{
        retrieval::{initial, Observation},
        theory::Theory,
    };
    #[test]
    fn recover_mixture_from_finer_frequency_grid_with_free_background() {
        let fine = Theory::at_field(30000., 25., 250.).unwrap();
        let coarse = Theory::at_field(30000., 25., 500.).unwrap();
        let rho = vec![
            C::new(1., 0.),
            C::new(-0.03, 0.),
            C::new(0.02, 0.),
            C::new(-0.01, 0.),
        ];
        let (fir, negative) = receiver_fir(&rho);
        assert_eq!(negative, 0.);
        let kfine = kernel(&fine.f, &fir, -0.008);
        let kcoarse = kernel(&coarse.f, &fir, -0.008);
        let mut bg = vec![C::default(); 61];
        for l in 0..rho.len() {
            bg[l] = rho[l] * 200. / (200 - l) as f64;
        }
        for (hydrogen, helium) in [(0.02, 0.05), (0.4, 0.15)] {
            let s = fine.spectrum(1800., 2600., 70., hydrogen, helium / (1. - hydrogen), 5e10);
            let y: Vec<C> = kfine
                .iter()
                .enumerate()
                .map(|(l, k)| {
                    k.iter().zip(&s).map(|(k, s)| *k * s).sum::<C>() * 10000. - bg[l] * 210.
                })
                .collect();
            let obs = Observation {
                value: y,
                sigma: vec![[5., 5.]; 61],
                kernel: kcoarse.clone(),
                rho: bg.clone(),
                background_sigma: 1e9,
                lag_min: 0,
            };
            let f = obs.fit(&coarse, &initial(1400., 2200., 0.2), 5e10, None, 150);
            assert!(
                (f.h_fraction - hydrogen).abs() < 0.004,
                "H {} vs {}",
                f.h_fraction,
                hydrogen
            );
            assert!(
                (f.he_fraction - helium).abs() < 0.004,
                "He {} vs {}",
                f.he_fraction,
                helium
            );
            assert!((f.ti_k / 1800. - 1.).abs() < 0.01);
            assert!((f.te_k / 2600. - 1.).abs() < 0.01);
            assert!((f.background_adjustment_adu2 + 210.).abs() < 5.);
            let later = Observation {
                value: obs.value[1..].to_vec(),
                sigma: obs.sigma[1..].to_vec(),
                kernel: obs.kernel[1..].to_vec(),
                rho: obs.rho[1..].to_vec(),
                background_sigma: 1e9,
                lag_min: 1,
            };
            let excluded = later.fit(&coarse, &initial(1400., 2200., 0.2), 5e10, None, 150);
            assert!((excluded.h_fraction - hydrogen).abs() < 0.004);
            assert!((excluded.he_fraction - helium).abs() < 0.004);
            let full_error = obs.local_errors(&coarse, &f, 5e10).unwrap();
            let later_error = later.local_errors(&coarse, &excluded, 5e10).unwrap();
            assert!(later_error.iter().all(|e| e.is_finite() && *e > 0.));
            assert!(later_error[0] >= 0.98 * full_error[0]);
        }
    }
}
