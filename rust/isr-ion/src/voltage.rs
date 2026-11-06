//! Range-dependent proper Gaussian voltage model, before correlation/decoding.
//! Each statistically independent range cell contributes D(tx) R D(tx)*.
//! Receiver filters act on voltages (H R H*), never on decoded error bars.
//! All times, delays and FIR taps use the caller's explicit sample grid.
use crate::{
    noise::{lag_covariance, ProductTerm},
    Complex as C,
};
use anyhow::{ensure, Result};
use nalgebra::DMatrix;

/// Add one independent scattering cell. `acf[d]` is E[s(t+d) conj(s(t))],
/// in voltage-power units, including the cell's strength; delay can be negative.
/// The caller supplies a positive-semidefinite autocorrelation sequence.
pub fn add_range_cell(r: &mut DMatrix<C>, tx: &[C], delay: isize, acf: &[C]) -> Result<()> {
    let n = r.nrows();
    ensure!(
        r.is_square() && n > 0 && acf.len() >= n,
        "invalid voltage/ACF dimensions"
    );
    ensure!(
        tx.iter()
            .chain(acf)
            .all(|z| z.re.is_finite() && z.im.is_finite()),
        "nonfinite waveform/ACF"
    );
    ensure!(
        acf[0].re >= 0. && acf[0].im.abs() <= 1e-12 * acf[0].re.max(1.),
        "invalid zero-lag power"
    );
    let waveform = |i: usize| {
        let k = i as isize - delay;
        if k >= 0 {
            tx.get(k as usize).copied().unwrap_or_default()
        } else {
            C::default()
        }
    };
    for a in 0..n {
        let ta = waveform(a);
        if ta.norm_sqr() == 0. {
            continue;
        }
        for b in 0..=a {
            let v = ta * waveform(b).conj() * acf[a - b];
            r[(a, b)] += v;
            if a != b {
                r[(b, a)] += v.conj();
            }
        }
    }
    Ok(())
}

/// FIR plus arbitrary selected output sample indices. Input must include the
/// filter history; reject absent history rather than silently zero-padding it.
/// White receiver noise is added to the input covariance BEFORE this operation.
pub fn filter_voltage(r: &DMatrix<C>, taps: &[C], output: &[usize]) -> Result<DMatrix<C>> {
    ensure!(
        r.is_square() && !r.is_empty() && !taps.is_empty() && !output.is_empty(),
        "empty filter input"
    );
    ensure!(
        taps.iter().all(|z| z.re.is_finite() && z.im.is_finite()),
        "nonfinite FIR"
    );
    ensure!(
        output.iter().all(|&i| i < r.nrows() && i + 1 >= taps.len()),
        "missing FIR history or output beyond input"
    );
    let mut out = DMatrix::zeros(output.len(), output.len());
    for (a, &i) in output.iter().enumerate() {
        for (b, &j) in output[..=a].iter().enumerate() {
            let mut v = C::default();
            for (k, &h) in taps.iter().enumerate() {
                for (l, &g) in taps.iter().enumerate() {
                    v += h * g.conj() * r[(i - k, j - l)];
                }
            }
            out[(a, b)] = v;
            out[(b, a)] = v.conj();
        }
    }
    Ok(out)
}

/// Mean and full real/imaginary covariance of decoded lag estimates from one
/// independent code/pulse block. Sum block moments with squared normalization
/// weights for covariance; repetitions must not include overlapping dumps.
pub fn decoded_moments(
    r: &DMatrix<C>,
    estimators: &[Vec<ProductTerm>],
) -> Result<(Vec<C>, DMatrix<f64>)> {
    let covariance = lag_covariance(r, estimators, 1.)?;
    let mean = estimators
        .iter()
        .map(|terms| {
            terms
                .iter()
                .map(|t| t.weight * r[(t.first, t.second)])
                .sum()
        })
        .collect();
    Ok((mean, covariance))
}

#[cfg(test)]
mod tests {
    use super::*;
    #[test]
    fn range_cells_and_fir_match_explicit_linear_gaussian_model() {
        let n = 8;
        let tx = vec![
            C::new(1., 0.),
            C::new(-1., 0.),
            C::new(0., 1.),
            C::new(1., 0.),
        ];
        let mut r = DMatrix::from_diagonal_element(n, n, C::new(0.4, 0.));
        let mut explicit = r.clone();
        for (delay, power) in [(-1, 2.), (2, 5.)] {
            // Perfectly coherent scattering: one independent random amplitude per cell.
            add_range_cell(&mut r, &tx, delay, &vec![C::new(power, 0.); n]).unwrap();
            let v = DMatrix::from_fn(n, 1, |i, _| {
                let k = i as isize - delay;
                if k >= 0 {
                    tx.get(k as usize).copied().unwrap_or_default() * power.sqrt()
                } else {
                    C::default()
                }
            });
            explicit += &v * v.adjoint();
        }
        assert!((&r - &explicit).norm() < 1e-12);
        let taps = [C::new(0.5, 0.), C::new(0., 0.25), C::new(0.25, 0.)];
        let output = [2usize, 4, 7];
        let h = DMatrix::from_fn(3, n, |a, b| {
            output[a]
                .checked_sub(b)
                .and_then(|k| taps.get(k).copied())
                .unwrap_or_default()
        });
        let expected = &h * explicit * h.adjoint();
        let actual = filter_voltage(&r, &taps, &output).unwrap();
        assert!((actual - expected).norm() < 1e-12);
    }
    #[test]
    fn rejects_missing_filter_history() {
        let r = DMatrix::identity(4, 4);
        assert!(filter_voltage(&r, &[C::new(1., 0.); 3], &[1]).is_err());
    }
    #[test]
    fn coherent_echo_has_cross_gate_self_noise() {
        let mut r = DMatrix::zeros(5, 5);
        add_range_cell(&mut r, &[C::new(1., 0.); 5], 0, &[C::new(3., 0.); 5]).unwrap();
        let terms = vec![
            vec![ProductTerm {
                first: 1,
                second: 0,
                weight: C::new(1., 0.),
            }],
            vec![ProductTerm {
                first: 4,
                second: 3,
                weight: C::new(1., 0.),
            }],
        ];
        let (mean, cov) = decoded_moments(&r, &terms).unwrap();
        assert_eq!(mean, vec![C::new(3., 0.); 2]);
        assert!((cov[(0, 2)] - 9.).abs() < 1e-12);
        assert_eq!(cov[(1, 3)], 0.);
    }
}

#[cfg(test)]
mod monte_carlo {
    use super::*;
    #[test]
    fn partial_pulse_decoded_covariance_matches_voltage_simulation() {
        // Independent implementation: generate range-cell AR(1) voltages,
        // add receiver noise, then execute the decoder on each realization.
        let n = 70;
        let code: Vec<f64> = (0..32)
            .map(|i| if (i * i + 3 * i + 1) % 7 < 3 { -1. } else { 1. })
            .collect();
        let tx: Vec<C> = code.iter().flat_map(|&c| [C::new(c, 0.); 2]).collect();
        let cells = [(-40isize, 3.0f64, 0.91f64), (-10, 1.5, 0.75)];
        let receiver = 0.7;
        let mut r = DMatrix::from_diagonal_element(n, n, C::new(receiver, 0.));
        for &(delay, power, rho) in &cells {
            let acf: Vec<_> = (0..n)
                .map(|k| C::new(power * rho.powi(k as i32), 0.))
                .collect();
            add_range_cell(&mut r, &tx, delay, &acf).unwrap();
        }
        let gates = [0, 20, 42, 50, 60];
        let decoder = &crate::beata::blocks()[0];
        let estimators: Vec<_> = gates
            .iter()
            .map(|&g| decoder.terms(&code, g).unwrap())
            .collect();
        let (mean, cov) = decoded_moments(&r, &estimators).unwrap();
        let d = 2 * gates.len();
        let mut sum = vec![0.; d];
        let mut cross = DMatrix::<f64>::zeros(d, d);
        let mut seed = 918273u64;
        let mut uniform = || {
            seed = seed.wrapping_mul(6364136223846793005).wrapping_add(1);
            ((seed >> 11) as f64 + 0.5) / 9007199254740992.
        };
        let mut gaussian = || {
            let radius = (-uniform().ln()).sqrt();
            let phase = std::f64::consts::TAU * uniform();
            C::from_polar(radius, phase)
        };
        let draws = 40000;
        for _ in 0..draws {
            let mut z: Vec<C> = (0..n).map(|_| gaussian() * receiver.sqrt()).collect();
            for &(delay, power, rho) in &cells {
                let mut s = gaussian() * power.sqrt();
                for (i, v) in z.iter_mut().enumerate() {
                    if i > 0 {
                        s = s * rho + gaussian() * (power * (1. - rho * rho)).sqrt();
                    }
                    let k = i as isize - delay;
                    if k >= 0 {
                        *v += s * tx.get(k as usize).copied().unwrap_or_default();
                    }
                }
            }
            let mut y = Vec::new();
            for terms in &estimators {
                let q: C = terms
                    .iter()
                    .map(|t| t.weight * z[t.first] * z[t.second].conj())
                    .sum();
                y.extend([q.re, q.im]);
            }
            for i in 0..d {
                sum[i] += y[i];
                for j in 0..d {
                    cross[(i, j)] += y[i] * y[j];
                }
            }
        }
        for i in 0..d {
            let expected = if i % 2 == 0 {
                mean[i / 2].re
            } else {
                mean[i / 2].im
            };
            assert!(
                (sum[i] / draws as f64 - expected).abs() < 6. * (cov[(i, i)] / draws as f64).sqrt()
            );
            for j in 0..d {
                let empirical =
                    cross[(i, j)] / draws as f64 - sum[i] * sum[j] / (draws as f64).powi(2);
                let scale = (cov[(i, i)] * cov[(j, j)]).sqrt();
                assert!(
                    (empirical - cov[(i, j)]).abs() < 0.055 * scale,
                    "covariance ({i},{j}): {empirical} vs {}",
                    cov[(i, j)]
                );
            }
        }
    }
}

/// Add a receiver-filtered AR(1) scattering cell without constructing the huge
/// covariance on the ADC grid. Exact for the supplied grid, waveform and FIR.
/// `rho` is the real correlation per input sample; independent receiver noise
/// can be represented by a constant waveform and rho=0 in a separate call.
pub fn add_filtered_ar1_cell(
    out: &mut DMatrix<C>,
    tx: &[C],
    delay: isize,
    power: f64,
    rho: f64,
    taps: &[C],
    samples: &[usize],
) -> Result<()> {
    ensure!(
        !taps.is_empty() && !samples.is_empty() && out.shape() == (samples.len(), samples.len()),
        "invalid filtered covariance shape"
    );
    ensure!(
        power.is_finite() && power >= 0. && rho.is_finite() && rho >= 0. && rho <= 1.,
        "invalid AR power/correlation"
    );
    ensure!(
        tx.iter()
            .chain(taps)
            .all(|z| z.re.is_finite() && z.im.is_finite()),
        "nonfinite waveform/filter"
    );
    ensure!(
        samples.windows(2).all(|x| x[1] > x[0]) && samples[0] + 1 >= taps.len(),
        "invalid output indices or missing FIR history"
    );
    if power == 0. {
        return Ok(());
    }
    let h = taps.len();
    let weights: Vec<Vec<C>> = samples
        .iter()
        .map(|&o| {
            taps.iter()
                .enumerate()
                .map(|(k, &v)| {
                    let t = (o - k) as isize - delay;
                    v * if t >= 0 {
                        tx.get(t as usize).copied().unwrap_or_default()
                    } else {
                        C::default()
                    }
                })
                .collect()
        })
        .collect();
    let active: Vec<_> = weights
        .iter()
        .map(|w| w.iter().any(|z| z.norm_sqr() > 0.))
        .collect();
    // Powers measured from opposite ends avoid negative powers at rho=0.
    let left: Vec<C> = weights
        .iter()
        .map(|w| {
            w.iter()
                .enumerate()
                .map(|(k, z)| z * rho.powi((h - 1 - k) as i32))
                .sum()
        })
        .collect();
    let right: Vec<C> = weights
        .iter()
        .map(|w| {
            w.iter()
                .enumerate()
                .map(|(k, z)| z * rho.powi(k as i32))
                .sum()
        })
        .collect();
    for a in 0..samples.len() {
        if !active[a] {
            continue;
        }
        for b in 0..=a {
            if !active[b] {
                continue;
            }
            let delta = samples[a] - samples[b];
            let v = if delta >= h - 1 {
                left[a] * right[b].conj() * rho.powi((delta - (h - 1)) as i32)
            } else {
                let start = samples[b] + 1 - h;
                let len = h + delta;
                let mut values = vec![C::default(); len];
                for (k, &w) in weights[b].iter().enumerate() {
                    values[samples[b] - k - start] = w;
                }
                let mut filtered = values.clone();
                for i in 1..len {
                    let previous = filtered[i - 1];
                    filtered[i] += rho * previous;
                }
                let mut backward = C::default();
                for i in (0..len).rev() {
                    filtered[i] += rho * backward;
                    backward = values[i] + rho * backward;
                }
                weights[a]
                    .iter()
                    .enumerate()
                    .map(|(k, w)| w * filtered[samples[a] - k - start].conj())
                    .sum()
            } * power;
            out[(a, b)] += v;
            if a != b {
                out[(b, a)] += v.conj();
            }
        }
    }
    Ok(())
}
#[cfg(test)]
mod filtered_ar_tests {
    use super::*;
    #[test]
    fn fast_adc_grid_covariance_matches_dense_filter_at_boundaries() {
        let tx: Vec<_> = (0..24)
            .map(|i| {
                C::new(
                    if i % 3 == 0 { -1. } else { 1. },
                    if i % 5 == 0 { 0.2 } else { 0. },
                )
            })
            .collect();
        let taps = [
            C::new(0.2, 0.),
            C::new(0.3, 0.1),
            C::new(0.4, 0.),
            C::new(0.1, -0.1),
        ];
        let samples = [3, 4, 6, 9, 15, 23, 26];
        for rho in [0.0f64, 0.7, 0.999, 1.] {
            for delay in [-5, 0, 7] {
                let mut raw = DMatrix::zeros(27, 27);
                let acf: Vec<_> = (0..27).map(|k| C::new(2. * rho.powi(k), 0.)).collect();
                add_range_cell(&mut raw, &tx, delay, &acf).unwrap();
                let expected = filter_voltage(&raw, &taps, &samples).unwrap();
                let mut actual = DMatrix::zeros(samples.len(), samples.len());
                add_filtered_ar1_cell(&mut actual, &tx, delay, 2., rho, &taps, &samples).unwrap();
                assert!(
                    (actual - expected).norm() < 1e-11,
                    "rho={rho},delay={delay}"
                );
            }
        }
    }
}

/// Receiver-filtered covariance of one independent scattering cell, integrated
/// over an explicit spectrum. `angular_frequency` is radians per input sample;
/// `spectral_mass` includes quadrature weights and has voltage-power units.
/// This supports physical ISR spectra (including Doppler shifts), without an
/// assumed AR correlation. FIR history and range delay use the input grid.
/// The result is positive semidefinite for nonnegative spectral masses.
/// Frequency/range quadrature accuracy remains the caller's responsibility.
pub fn add_filtered_spectrum_cell(
    out: &mut DMatrix<C>,
    tx: &[C],
    delay: isize,
    angular_frequency: &[f64],
    spectral_mass: &[f64],
    taps: &[C],
    samples: &[usize],
) -> Result<()> {
    ensure!(
        !samples.is_empty() && !taps.is_empty() && !tx.is_empty(),
        "empty spectrum response geometry"
    );
    ensure!(
        out.shape() == (samples.len(), samples.len()),
        "invalid covariance dimensions"
    );
    ensure!(
        !angular_frequency.is_empty() && angular_frequency.len() == spectral_mass.len(),
        "invalid spectrum dimensions"
    );
    ensure!(
        angular_frequency.iter().all(|x| x.is_finite())
            && spectral_mass.iter().all(|x| x.is_finite() && *x >= 0.),
        "invalid spectrum frequencies or masses"
    );
    ensure!(
        tx.iter()
            .chain(taps)
            .all(|z| z.re.is_finite() && z.im.is_finite()),
        "nonfinite waveform/filter"
    );
    ensure!(
        samples.windows(2).all(|p| p[1] > p[0]) && samples[0] >= taps.len() - 1,
        "invalid output indices or missing FIR history"
    );
    // Save only nonzero H D(tx) entries. No dense ADC covariance is needed.
    let rows: Vec<Vec<(usize, C)>> = samples
        .iter()
        .map(|&o| {
            taps.iter()
                .enumerate()
                .filter_map(|(k, &h)| {
                    let input = o - k;
                    let t = input as i128 - delay as i128;
                    let v = if t >= 0 && t < tx.len() as i128 {
                        h * tx[t as usize]
                    } else {
                        C::default()
                    };
                    (v.norm_sqr() > 0.).then_some((input, v))
                })
                .collect()
        })
        .collect();
    let active: Vec<usize> = rows
        .iter()
        .enumerate()
        .filter_map(|(i, r)| (!r.is_empty()).then_some(i))
        .collect();
    let mut response = vec![C::default(); samples.len()];
    for (&omega, &mass) in angular_frequency.iter().zip(spectral_mass) {
        if mass == 0. {
            continue;
        }
        for &i in &active {
            // A common phase offset cancels in v v*, and reduces phase error.
            response[i] = rows[i]
                .iter()
                .map(|&(t, h)| h * C::from_polar(1., omega * (t as f64 - samples[0] as f64)))
                .sum();
        }
        for (ia, &a) in active.iter().enumerate() {
            for &b in &active[..=ia] {
                let value = response[a] * response[b].conj() * mass;
                out[(a, b)] += value;
                if a != b {
                    out[(b, a)] += value.conj();
                }
            }
        }
    }
    Ok(())
}

#[cfg(test)]
mod filtered_spectrum_tests {
    use super::*;
    #[test]
    fn shifted_multicomponent_spectrum_matches_dense_adc_filter_and_decoder() {
        let n = 21;
        let tx: Vec<_> = (0..14)
            .map(|i| C::from_polar(1., if i % 3 == 0 { 0.3 } else { 3.1 }))
            .collect();
        let taps = vec![C::new(0.2, 0.1), C::new(0.6, -0.1), C::new(0.2, 0.)];
        let omega: [f64; 4] = [-0.7, -0.03, 0.13, 0.6];
        let masses = [0.2, 1.3, 0.8, 0.05];
        let acf: Vec<C> = (0..n)
            .map(|k| {
                omega
                    .iter()
                    .zip(masses)
                    .map(|(&w, m)| C::from_polar(m, w * k as f64))
                    .sum()
            })
            .collect();
        let samples = vec![2, 4, 5, 8, 12, 17, 20];
        for delay in [-4, 0, 5, 24] {
            let mut dense = DMatrix::zeros(n, n);
            add_range_cell(&mut dense, &tx, delay, &acf).unwrap();
            let reference = filter_voltage(&dense, &taps, &samples).unwrap();
            let mut got = DMatrix::zeros(samples.len(), samples.len());
            add_filtered_spectrum_cell(&mut got, &tx, delay, &omega, &masses, &taps, &samples)
                .unwrap();
            assert!((&reference - &got).norm() < 1e-12);
            let estimators = vec![
                vec![
                    ProductTerm {
                        first: 3,
                        second: 1,
                        weight: C::new(-0.5, 0.),
                    },
                    ProductTerm {
                        first: 4,
                        second: 2,
                        weight: C::new(0.25, 0.),
                    },
                ],
                vec![ProductTerm {
                    first: 2,
                    second: 2,
                    weight: C::new(1., 0.),
                }],
            ];
            let (mu, cov) = decoded_moments(&got, &estimators).unwrap();
            let (expected_mu, expected_cov) = decoded_moments(&reference, &estimators).unwrap();
            assert!(mu
                .iter()
                .zip(expected_mu)
                .all(|(a, b)| (*a - b).norm() < 1e-12));
            assert!((cov - expected_cov).norm() < 1e-12);
        }
    }
    #[test]
    fn spectrum_rejects_negative_mass_and_missing_history() {
        let mut out = DMatrix::zeros(1, 1);
        assert!(add_filtered_spectrum_cell(
            &mut out,
            &[C::new(1., 0.)],
            0,
            &[0.],
            &[-1.],
            &[C::new(1., 0.)],
            &[0]
        )
        .is_err());
        assert!(add_filtered_spectrum_cell(
            &mut out,
            &[C::new(1., 0.)],
            0,
            &[0.],
            &[1.],
            &[C::new(1., 0.); 2],
            &[0]
        )
        .is_err());
        assert_eq!(out.norm(), 0.);
    }
}
