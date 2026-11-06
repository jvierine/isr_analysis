//! Gaussian voltage fourth moments propagated through explicit lag-product weights.
//! The caller supplies signal + receiver voltage covariance for the actual sample
//! times and range-dependent ambiguity response. No stationary or diagonal shortcut.
use crate::Complex as C;
use anyhow::{ensure, Result};
use nalgebra::DMatrix;
#[derive(Clone, Copy, Debug)]
pub struct ProductTerm {
    pub first: usize,
    pub second: usize,
    pub weight: C,
}
/// Covariance of [Re q0, Im q0, Re q1, Im q1, ...], for
/// q_i = sum w_ab z_a conj(z_b), averaged over independent identical repetitions.
/// R[a,b] = E[z_a conj(z_b)]. Proper, zero-mean complex Gaussian voltages assumed.
/// Independent pulse/code blocks must be represented in R, not counted twice.
pub fn lag_covariance(
    r: &DMatrix<C>,
    estimators: &[Vec<ProductTerm>],
    repetitions: f64,
) -> Result<DMatrix<f64>> {
    ensure!(
        r.is_square() && !r.is_empty() && repetitions.is_finite() && repetitions > 0.,
        "invalid covariance dimensions or repetitions"
    );
    let scale = r
        .iter()
        .map(|z| z.norm())
        .fold(0., f64::max)
        .max(f64::MIN_POSITIVE);
    ensure!(
        r.iter().all(|z| z.re.is_finite() && z.im.is_finite()),
        "nonfinite voltage covariance"
    );
    for a in 0..r.nrows() {
        ensure!(r[(a, a)].re >= 0., "negative voltage power");
        for b in 0..r.ncols() {
            ensure!(
                (r[(a, b)] - r[(b, a)].conj()).norm() <= 1e-10 * scale,
                "non-Hermitian voltage covariance"
            );
        }
    }
    ensure!(
        estimators.iter().all(|e| !e.is_empty()
            && e.iter().all(|t| t.first < r.nrows()
                && t.second < r.nrows()
                && t.weight.re.is_finite()
                && t.weight.im.is_finite())),
        "invalid estimator weights"
    );
    let mut out = DMatrix::zeros(2 * estimators.len(), 2 * estimators.len());
    for (i, left) in estimators.iter().enumerate() {
        for (j, right) in estimators.iter().enumerate() {
            let mut c = C::new(0., 0.);
            let mut p = C::new(0., 0.);
            for a in left {
                for b in right {
                    c += a.weight
                        * b.weight.conj()
                        * r[(a.first, b.first)]
                        * r[(b.second, a.second)];
                    p += a.weight * b.weight * r[(a.first, b.second)] * r[(b.first, a.second)];
                }
            }
            out[(2 * i, 2 * j)] = (c.re + p.re) / (2. * repetitions);
            out[(2 * i + 1, 2 * j + 1)] = (c.re - p.re) / (2. * repetitions);
            out[(2 * i, 2 * j + 1)] = (p.im - c.im) / (2. * repetitions);
            out[(2 * i + 1, 2 * j)] = (p.im + c.im) / (2. * repetitions);
        }
    }
    ensure!(
        out.iter().all(|x| x.is_finite()),
        "nonfinite lag covariance"
    );
    Ok(out)
}
#[cfg(test)]
mod tests {
    use super::*;
    fn term(a: usize, b: usize) -> Vec<ProductTerm> {
        vec![ProductTerm {
            first: a,
            second: b,
            weight: C::new(1., 0.),
        }]
    }
    #[test]
    fn power_variance_contains_signal_squared_and_cross_term() {
        let s = 7.;
        let n = 3.;
        let r = DMatrix::from_element(1, 1, C::new(s + n, 0.));
        let v = lag_covariance(&r, &[term(0, 0)], 10.).unwrap();
        assert!((v[(0, 0)] - (s * s + 2. * s * n + n * n) / 10.).abs() < 1e-12);
        assert_eq!(v[(1, 1)], 0.);
    }
    #[test]
    fn complex_cross_covariance_has_correct_sign() {
        let mut r = DMatrix::from_diagonal_element(2, 2, C::new(2., 0.));
        r[(0, 1)] = C::new(0.3, 0.4);
        r[(1, 0)] = r[(0, 1)].conj();
        let v = lag_covariance(&r, &[term(0, 0), term(0, 1)], 1.).unwrap();
        assert!((v[(0, 2)] - 0.6).abs() < 1e-12);
        assert!((v[(0, 3)] - 0.8).abs() < 1e-12);
        assert!((&v - v.transpose()).amax() < 1e-12);
        assert!((v[(2, 2)] - 1.965).abs() < 1e-12);
        assert!((v[(2, 3)] - 0.12).abs() < 1e-12);
    }
    #[test]
    fn decoding_preserves_correlation_not_independent_error_sum() {
        let r = DMatrix::from_diagonal_element(2, 2, C::new(1., 0.));
        let mut combined = term(0, 1);
        combined.push(ProductTerm {
            first: 0,
            second: 1,
            weight: C::new(-1., 0.),
        });
        let v = lag_covariance(&r, &[combined], 1.).unwrap();
        assert_eq!(v.amax(), 0.);
    }
}

#[cfg(test)]
mod monte_carlo {
    use super::*;
    #[test]
    fn fourth_moments_match_correlated_voltage_simulation() {
        let mut seed = 173u64;
        let mut uniform = || {
            seed = seed.wrapping_mul(6364136223846793005).wrapping_add(1);
            ((seed >> 11) as f64 + 0.5) / 9007199254740992.
        };
        let mut normal = || {
            let u = uniform();
            let v = uniform();
            (-2. * u.ln()).sqrt() * (std::f64::consts::TAU * v).cos()
        };
        let mut r = DMatrix::from_diagonal_element(2, 2, C::new(2., 0.));
        r[(0, 1)] = C::new(0.3, 0.4);
        r[(1, 0)] = r[(0, 1)].conj();
        let estimators = vec![
            vec![ProductTerm {
                first: 0,
                second: 0,
                weight: C::new(1., 0.),
            }],
            vec![ProductTerm {
                first: 0,
                second: 1,
                weight: C::new(1., 0.),
            }],
        ];
        let expected = lag_covariance(&r, &estimators, 1.).unwrap();
        let mut sum = [0.; 4];
        let mut products = DMatrix::<f64>::zeros(4, 4);
        let n = 100000;
        for _ in 0..n {
            let z0 = C::new(normal(), normal());
            let z1 = r[(1, 0)] / 2. * z0 + C::new(normal(), normal()) * (1. - 0.25 / 4_f64).sqrt();
            let q = z0 * z1.conj();
            let x = [z0.norm_sqr(), 0., q.re, q.im];
            for i in 0..4 {
                sum[i] += x[i];
                for j in 0..4 {
                    products[(i, j)] += x[i] * x[j];
                }
            }
        }
        for i in 0..4 {
            for j in 0..4 {
                let actual = (products[(i, j)] - sum[i] * sum[j] / n as f64) / (n - 1) as f64;
                assert!(
                    (actual - expected[(i, j)]).abs() < 0.07,
                    "{i},{j}: {actual} vs {}",
                    expected[(i, j)]
                );
            }
        }
    }
}
