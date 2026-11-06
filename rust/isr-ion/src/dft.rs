//! Masked Hermitian matrix DFT of measured ACFs. This is a sampling-window
//! spectrum, not an inversion/deconvolution of missing lags or range ambiguity.
use anyhow::{ensure, Result};
use nalgebra::{DMatrix, DVector};
use std::collections::BTreeMap;

pub struct AcfDft {
    matrix: DMatrix<f64>,
    pub unique_lags: usize,
    pub missing_lags: usize,
    pub has_zero_lag: bool,
}
impl AcfDft {
    /// R(tau)=integral S(f) exp(+i 2pi f tau) df. Each positive lag
    /// contributes 2 dt [Re(R) cos(2pi f tau)+Im(R) sin(2pi f tau)].
    /// Duplicate response groups receive equal weights within their lag;
    /// absent lags have no columns and are never interpolated or renormalized.
    pub fn new(lags_s: &[f64], frequencies_hz: &[f64], dt_s: f64) -> Result<Self> {
        ensure!(
            !lags_s.is_empty() && dt_s.is_finite() && dt_s > 0.,
            "No lags or invalid lag spacing"
        );
        ensure!(
            frequencies_hz.iter().all(|f| f.is_finite()),
            "Invalid frequency"
        );
        let mut counts = BTreeMap::new();
        let mut slots = Vec::new();
        for &lag in lags_s {
            let slot = (lag / dt_s).round();
            ensure!(
                lag.is_finite() && lag >= 0. && slot < 1e6 && (lag / dt_s - slot).abs() < 1e-6,
                "Lag is off the declared sampling grid"
            );
            let slot = slot as usize;
            *counts.entry(slot).or_insert(0usize) += 1;
            slots.push(slot);
        }
        let matrix = DMatrix::from_fn(frequencies_hz.len(), 2 * lags_s.len(), |row, col| {
            let i = col / 2;
            let slot = slots[i];
            let weight = dt_s / counts[&slot] as f64;
            if slot == 0 {
                return if col % 2 == 0 { weight } else { 0. };
            }
            let phase = std::f64::consts::TAU * frequencies_hz[row] * lags_s[i];
            2. * weight
                * if col % 2 == 0 {
                    phase.cos()
                } else {
                    phase.sin()
                }
        });
        Ok(Self {
            unique_lags: counts.len(),
            missing_lags: counts.keys().last().unwrap() + 1 - counts.len(),
            has_zero_lag: counts.contains_key(&0),
            matrix,
        })
    }
    pub fn transform(&self, acf: &[[f64; 2]]) -> Result<Vec<f64>> {
        ensure!(
            acf.len() * 2 == self.matrix.ncols() && acf.iter().flatten().all(|v| v.is_finite()),
            "Invalid ACF"
        );
        let input = DVector::from_iterator(acf.len() * 2, acf.iter().flatten().copied());
        Ok((&self.matrix * input).as_slice().to_vec())
    }
    /// Diagonal-only noise propagation. Correlations between lags, components
    /// and repeated groups are not available in the exported ACF product.
    pub fn diagonal_standard_error(&self, sigma: &[[f64; 2]]) -> Result<Vec<f64>> {
        ensure!(
            sigma.len() * 2 == self.matrix.ncols()
                && sigma.iter().flatten().all(|v| v.is_finite() && *v >= 0.),
            "Invalid ACF errors"
        );
        Ok((0..self.matrix.nrows())
            .map(|row| {
                self.matrix
                    .row(row)
                    .iter()
                    .zip(sigma.iter().flatten())
                    .map(|(a, s)| (a * s).powi(2))
                    .sum::<f64>()
                    .sqrt()
            })
            .collect())
    }
}
#[cfg(test)]
mod tests {
    use super::*;
    #[test]
    fn complete_grid_matches_hermitian_dft_and_frequency_sign() {
        let dt = 0.1;
        let lags: Vec<_> = (0..=4).map(|i| i as f64 * dt).collect();
        let f: Vec<_> = (-4..=4).map(|i| i as f64 / (9. * dt)).collect();
        let acf: Vec<_> = lags
            .iter()
            .map(|t| {
                let p = std::f64::consts::TAU * f[6] * t;
                [p.cos(), p.sin()]
            })
            .collect();
        let op = AcfDft::new(&lags, &f, dt).unwrap();
        let s = op.transform(&acf).unwrap();
        for (i, value) in s.iter().enumerate() {
            assert!((value - if i == 6 { 9. * dt } else { 0. }).abs() < 1e-12);
        }
        assert_eq!(op.missing_lags, 0);
    }
    #[test]
    fn missing_zero_is_not_invented_and_duplicates_do_not_change_weight() {
        let f = [0., 0.25, 0.5];
        let op = AcfDft::new(&[1., 3.], &f, 1.).unwrap();
        assert!(!op.has_zero_lag);
        assert_eq!(op.missing_lags, 2);
        let s = op.transform(&[[1., 0.], [1., 0.]]).unwrap();
        assert!((s[0] - 4.).abs() < 1e-12);
        assert!((s[2] + 4.).abs() < 1e-12);
        let repeated = AcfDft::new(&[1., 1., 3.], &f, 1.).unwrap();
        assert_eq!(
            s,
            repeated.transform(&[[1., 0.], [1., 0.], [1., 0.]]).unwrap()
        );
        let e = op.diagonal_standard_error(&[[2., 3.], [4., 5.]]).unwrap();
        assert!((e[0] - 80f64.sqrt()).abs() < 1e-12);
        assert!((e[1] - 136f64.sqrt()).abs() < 1e-12);
    }
    #[test]
    fn reject_invalid_inputs() {
        assert!(AcfDft::new(&[1.1], &[0.], 1.).is_err());
        assert!(AcfDft::new(&[], &[0.], 1.).is_err());
        let op = AcfDft::new(&[1.], &[0.], 1.).unwrap();
        assert!(op.transform(&[]).is_err());
        assert!(op.diagonal_standard_error(&[[-1., 0.]]).is_err());
    }
}
