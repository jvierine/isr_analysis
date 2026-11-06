//! Explicit voltage-domain temporal filtering and a signal-dependent joint likelihood.
//! No cadence is treated as independent merely because it is stored separately.
use crate::Complex as C;
use anyhow::{ensure, Result};
use nalgebra::{DMatrix, DVector};

/// The clutter.c FFT mask, including its stationary-power normalization.
/// None means no filter; Some(0) removes DC. These are not equivalent.
/// This models floating-point filtering, before online int16 rounding.
pub fn clutter_operator(cycles: usize, notch: Option<usize>) -> Result<DMatrix<f64>> {
    ensure!(cycles > 1, "at least two cycles required");
    let Some(notch) = notch else {
        return Ok(DMatrix::identity(cycles, cycles));
    };
    ensure!(
        2 * notch + 1 < cycles,
        "notch removes all voltage degrees of freedom"
    );
    let keep = cycles - 2 * notch - 1;
    let scale = (cycles as f64 / keep as f64).sqrt();
    Ok(DMatrix::from_fn(cycles, cycles, |i, j| {
        let phase = std::f64::consts::TAU * (i as f64 - j as f64) / cycles as f64;
        let removed = 1. + 2. * (1..=notch).map(|k| (k as f64 * phase).cos()).sum::<f64>();
        scale * ((if i == j { 1. } else { 0. }) - removed / cycles as f64)
    }))
}

/// Independent input cycles with arbitrary (nonstationary) voltage covariance.
/// Output block (a,b) is sum_t F[a,t] F[b,t] R_t, not delta_ab R_mean.
pub fn filtered_voltage_block(
    f: &DMatrix<f64>,
    r: &[DMatrix<C>],
    a: usize,
    b: usize,
) -> Result<DMatrix<C>> {
    ensure!(
        !r.is_empty() && f.ncols() == r.len() && a < f.nrows() && b < f.nrows(),
        "invalid filter dimensions"
    );
    let n = r[0].nrows();
    ensure!(
        n > 0 && r.iter().all(|x| x.shape() == (n, n)),
        "invalid voltage blocks"
    );
    ensure!(f.iter().all(|x| x.is_finite()), "nonfinite filter");
    let mut out = DMatrix::zeros(n, n);
    for (t, rt) in r.iter().enumerate() {
        out += rt.scale(f[(a, t)] * f[(b, t)]);
    }
    Ok(out)
}

/// E[q] for explicit complex product weights and supplied voltage covariance.
pub fn lag_mean(r: &DMatrix<C>, terms: &[crate::noise::ProductTerm]) -> Result<C> {
    ensure!(r.is_square() && !terms.is_empty(), "invalid mean operator");
    let mut value = C::new(0., 0.);
    for t in terms {
        ensure!(
            t.first < r.nrows() && t.second < r.ncols(),
            "sample outside covariance"
        );
        value += t.weight * r[(t.first, t.second)];
    }
    Ok(value)
}

/// Gaussian approximation to the distribution of accumulated lag estimates.
/// Keep log|C(x)| when comparing candidate profiles x: dropping it lets the
/// optimizer explain residuals by inflating its predicted self-noise.
/// Singular redundant observations fail, rather than receiving arbitrary jitter.
pub fn gaussian_nll(residual: &[f64], covariance: &DMatrix<f64>) -> Result<f64> {
    let n = residual.len();
    ensure!(
        n > 0 && covariance.shape() == (n, n),
        "likelihood dimensions differ"
    );
    ensure!(
        residual.iter().all(|x| x.is_finite()) && covariance.iter().all(|x| x.is_finite()),
        "nonfinite likelihood input"
    );
    let scale = covariance.amax().max(f64::MIN_POSITIVE);
    ensure!(
        (covariance - covariance.transpose()).amax() <= 1e-10 * scale,
        "asymmetric covariance"
    );
    let ch = covariance.clone().cholesky().ok_or_else(|| {
        anyhow::anyhow!("joint covariance is not positive definite; check duplicated measurements")
    })?;
    let e = DVector::from_column_slice(residual);
    let quadratic = e.dot(&ch.solve(&e));
    let logdet = 2. * ch.l().diagonal().iter().map(|x| x.ln()).sum::<f64>();
    Ok(0.5 * (quadratic + logdet + n as f64 * std::f64::consts::TAU.ln()))
}

/// Negative log marginal Student-t gradient prior, with the scale-normalization
/// term retained so a learned scale cannot drift freely. Units of delta and
/// scale must match; separate time and altitude scales belong to the caller.
pub fn student_gradient_nll(delta: f64, scale: f64, nu: f64) -> Result<f64> {
    ensure!(
        delta.is_finite() && scale.is_finite() && scale > 0. && nu.is_finite() && nu > 0.,
        "invalid gradient prior"
    );
    Ok(scale.ln() + 0.5 * (nu + 1.) * (delta * delta / (nu * scale * scale)).ln_1p())
}

#[cfg(test)]
mod tests {
    use super::*;
    #[test]
    fn dc_notch_preserves_stationary_power_but_correlates_cycles() {
        let f = clutter_operator(14, Some(0)).unwrap();
        let c = &f * f.transpose();
        for i in 0..14 {
            assert!(f.row(i).sum().abs() < 1e-12);
            assert!((c[(i, i)] - 1.).abs() < 1e-12);
        }
        assert!((c[(0, 1)] + 1. / 13.).abs() < 1e-12);
        // A power covariance is |R_ij|^2, so normalized cycle correlations
        // are 1/169 even for stationary white voltage.
        assert!((c[(0, 1)].powi(2) - 1. / 169.).abs() < 1e-12);
    }
    #[test]
    fn transient_power_is_not_unchanged_by_stationary_normalization() {
        let f = clutter_operator(14, Some(0)).unwrap();
        let r = (0..14)
            .map(|t| DMatrix::from_element(1, 1, C::new(if t == 6 { 100. } else { 1. }, 0.)))
            .collect::<Vec<_>>();
        let peak = filtered_voltage_block(&f, &r, 6, 6).unwrap()[(0, 0)].re;
        let outside = filtered_voltage_block(&f, &r, 0, 0).unwrap()[(0, 0)].re;
        assert!((peak - (1. + 99. * 13. / 14.)).abs() < 1e-10);
        assert!((outside - (1. + 99. / 182.)).abs() < 1e-10);
    }
    #[test]
    fn likelihood_penalizes_inflating_noise() {
        let small = gaussian_nll(&[0., 0.], &DMatrix::identity(2, 2)).unwrap();
        let large = gaussian_nll(&[0., 0.], &(DMatrix::identity(2, 2) * 100.)).unwrap();
        assert!((large - small - 100f64.ln()).abs() < 1e-12);
    }
    #[test]
    fn redundant_sum_is_rejected() {
        let c = DMatrix::from_row_slice(3, 3, &[1., 0., 1., 0., 1., 1., 1., 1., 2.]);
        assert!(gaussian_nll(&[1., 2., 3.], &c).is_err());
    }
    #[test]
    fn no_filter_does_not_mix_cycles() {
        let f = clutter_operator(14, None).unwrap();
        assert_eq!(f, DMatrix::identity(14, 14));
        assert!(clutter_operator(14, Some(7)).is_err());
    }
}
