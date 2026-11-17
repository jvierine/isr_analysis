//! Shrunk joint real/imaginary delete-block covariance, with low-rank whitening.
use anyhow::{ensure, Result};
use nalgebra::DMatrix as M;
#[cfg(target_os = "macos")]
#[link(name = "Accelerate", kind = "framework")]
extern "C" {
    fn dpotrf_(uplo: *const i8, n: *const i32, a: *mut f64, lda: *const i32, info: *mut i32)
        -> i32;
    fn cblas_dtrsm(
        order: i32,
        side: i32,
        uplo: i32,
        trans: i32,
        diag: i32,
        m: i32,
        n: i32,
        alpha: f64,
        a: *const f64,
        lda: i32,
        b: *mut f64,
        ldb: i32,
    );
}
pub struct DenseWhitening {
    pub lower: M<f64>,
    pub log_determinant: f64,
}
impl DenseWhitening {
    pub fn new(mut covariance: M<f64>) -> Result<Self> {
        ensure!(
            covariance.is_square() && covariance.iter().all(|v| v.is_finite()),
            "invalid dense covariance"
        );
        let n = covariance.nrows();
        #[cfg(target_os = "macos")]
        {
            let size = i32::try_from(n)?;
            let mut info = 0;
            unsafe {
                dpotrf_(
                    &(b'L' as i8),
                    &size,
                    covariance.as_mut_slice().as_mut_ptr(),
                    &size,
                    &mut info,
                );
            };
            ensure!(info == 0, "dense covariance Cholesky failed at {info}");
            for j in 0..n {
                for i in 0..j {
                    covariance[(i, j)] = 0.;
                }
            }
        }
        #[cfg(not(target_os = "macos"))]
        {
            covariance = covariance
                .cholesky()
                .ok_or_else(|| anyhow::anyhow!("covariance not positive definite"))?
                .l();
        }
        let log_determinant = 2. * (0..n).map(|i| covariance[(i, i)].ln()).sum::<f64>();
        Ok(Self {
            lower: covariance,
            log_determinant,
        })
    }
    pub fn apply(&self, x: &M<f64>) -> M<f64> {
        let mut b = x.clone();
        assert_eq!(b.nrows(), self.lower.nrows());
        #[cfg(target_os = "macos")]
        unsafe {
            let n = b.nrows() as i32;
            cblas_dtrsm(
                102,
                141,
                122,
                111,
                131,
                n,
                b.ncols() as i32,
                1.,
                self.lower.as_slice().as_ptr(),
                n,
                b.as_mut_slice().as_mut_ptr(),
                n,
            );
        }
        #[cfg(not(target_os = "macos"))]
        {
            b = self
                .lower
                .solve_lower_triangular(&b)
                .expect("nonsingular Cholesky factor");
        }
        b
    }
}
pub struct Whitening {
    pub dense: Option<DenseWhitening>,
    pub modes: M<f64>,
    pub correction: Vec<f64>,
    pub alpha: f64,
}
impl Whitening {
    pub fn new(u: &M<f64>, alpha: f64) -> Result<Self> {
        ensure!(
            alpha > 0. && alpha <= 1.,
            "covariance shrinkage must lie in (0,1]"
        );
        let eigen = (u.transpose() * u).symmetric_eigen();
        let keep: Vec<usize> = eigen
            .eigenvalues
            .iter()
            .enumerate()
            .filter(|(_, v)| **v > 1e-10)
            .map(|(i, _)| i)
            .collect();
        let mut modes = M::zeros(u.nrows(), keep.len());
        let mut correction = vec![];
        for (j, &i) in keep.iter().enumerate() {
            let lam = eigen.eigenvalues[i];
            let q = u * eigen.eigenvectors.column(i) / lam.sqrt();
            modes.set_column(j, &q);
            correction.push((1. + (1. - alpha) * lam / alpha).sqrt().recip() - 1.);
        }
        Ok(Self {
            dense: None,
            modes,
            correction,
            alpha,
        })
    }
    /// Applied after division by the saved per-measurement standard errors.
    pub fn apply(&self, x: &M<f64>) -> M<f64> {
        let transformed;
        let x = if let Some(dense) = &self.dense {
            transformed = dense.apply(x);
            &transformed
        } else {
            x
        };
        let mut q = self.modes.transpose() * x;
        for i in 0..q.nrows() {
            q.row_mut(i).scale_mut(self.correction[i]);
        }
        (x + &self.modes * q) / self.alpha.sqrt()
    }
    pub fn with_structured(covariance: M<f64>, empirical: &M<f64>, alpha: f64) -> Result<Self> {
        let dense = DenseWhitening::new(covariance)?;
        Self::with_dense(dense, empirical, alpha)
    }
    pub fn with_dense(dense: DenseWhitening, empirical: &M<f64>, alpha: f64) -> Result<Self> {
        let transformed = dense.apply(empirical);
        let mut result = Self::new(&transformed, alpha)?;
        result.dense = Some(dense);
        Ok(result)
    }
    pub fn from_product(path: &std::path::Path) -> Result<Self> {
        let f = hdf5::File::open(path)?;
        let d = f.dataset("structured_cholesky_lower")?;
        let s = d.shape();
        ensure!(
            s.len() == 2 && s[0] == s[1],
            "invalid structured factor dimensions"
        );
        let lower = M::from_row_slice(s[0], s[1], &d.read_raw::<f64>()?);
        ensure!(
            lower.iter().all(|v| v.is_finite()) && (0..s[0]).all(|i| lower[(i, i)] > 0.),
            "invalid covariance factor"
        );
        let log_determinant = 2. * (0..s[0]).map(|i| lower[(i, i)].ln()).sum::<f64>();
        let d = f.dataset("normalized_empirical_modes")?;
        let s = d.shape();
        let empirical = M::from_row_slice(s[0], s[1], &d.read_raw::<f64>()?);
        let alpha = f.attr("structured_fraction")?.read_scalar::<f64>()?;
        Self::with_dense(
            DenseWhitening {
                lower,
                log_determinant,
            },
            &empirical,
            alpha,
        )
    }
}
/// Equal delete-block jackknife convention matches the existing saved SEMs;
/// pulse-count weights appear inside each leave-one-block mean.
pub fn jackknife_modes(blocks: &M<f64>, counts: &[f64], sigma: &[f64]) -> Result<M<f64>> {
    ensure!(
        blocks.ncols() == counts.len() && blocks.nrows() == sigma.len(),
        "covariance dimensions"
    );
    let good: Vec<usize> = counts
        .iter()
        .enumerate()
        .filter(|(_, n)| **n > 0.)
        .map(|(i, _)| i)
        .collect();
    ensure!(good.len() >= 3, "insufficient error blocks");
    let total: f64 = counts.iter().sum();
    let n = good.len();
    let mut u = M::zeros(blocks.nrows(), n);
    for row in 0..blocks.nrows() {
        let avg: f64 = good
            .iter()
            .map(|&b| blocks[(row, b)] * counts[b] / total)
            .sum();
        let leave: Vec<f64> = good
            .iter()
            .map(|&b| {
                let w = counts[b] / total;
                (avg - w * blocks[(row, b)]) / (1. - w)
            })
            .collect();
        let center = leave.iter().sum::<f64>() / n as f64;
        for b in 0..n {
            u[(row, b)] = (leave[b] - center) * ((n - 1) as f64 / n as f64).sqrt() / sigma[row];
        }
    }
    ensure!(
        u.iter().all(|v| v.is_finite()),
        "nonfinite block covariance"
    );
    Ok(u)
}
#[cfg(test)]
mod tests {
    #[test]
    fn saved_full_covariance_replays_identical_whitening() {
        let c = M::from_row_slice(3, 3, &[2., 0.3, 0.1, 0.3, 1., -0.2, 0.1, -0.2, 0.7]);
        let u = M::from_row_slice(3, 2, &[0.2, 0.8, -0.5, 0.1, 0.3, -0.2]);
        let w = Whitening::with_structured(c, &u, 0.7).unwrap();
        let path = std::env::temp_dir().join(format!(
            "syisr-covariance-roundtrip-{}.h5",
            std::process::id()
        ));
        let h = hdf5::File::create(&path).unwrap();
        for (name, m) in [
            (
                "structured_cholesky_lower",
                &w.dense.as_ref().unwrap().lower,
            ),
            ("normalized_empirical_modes", &u),
        ] {
            let row: Vec<f64> = (0..m.nrows())
                .flat_map(|i| (0..m.ncols()).map(move |j| m[(i, j)]))
                .collect();
            h.new_dataset::<f64>()
                .shape((m.nrows(), m.ncols()))
                .create(name)
                .unwrap()
                .write_raw(&row)
                .unwrap();
        }
        h.new_attr::<f64>()
            .create("structured_fraction")
            .unwrap()
            .write_scalar(&0.7)
            .unwrap();
        drop(h);
        let restored = Whitening::from_product(&path).unwrap();
        assert!((restored.apply(&M::identity(3, 3)) - w.apply(&M::identity(3, 3))).norm() < 1e-12);
        std::fs::remove_file(path).unwrap();
    }
    #[test]
    fn structured_mixture_whitens_the_full_covariance() {
        let c = M::from_row_slice(3, 3, &[2., 0.3, 0.1, 0.3, 1., -0.2, 0.1, -0.2, 0.7]);
        let u = M::from_row_slice(3, 2, &[0.2, 0.8, -0.5, 0.1, 0.3, -0.2]);
        let alpha = 0.6;
        let target = &c * alpha + (&u * u.transpose()) * (1. - alpha);
        let whitening = super::Whitening::with_structured(c, &u, alpha).unwrap();
        let w = whitening.apply(&M::identity(3, 3));
        assert!((&w * target * w.transpose() - M::identity(3, 3)).norm() < 1e-12);
    }
    use super::*;
    #[test]
    fn native_dense_whitening_matches_known_covariance() {
        let a = M::from_fn(15, 8, |i, j| ((i * 3 + j * 7) as f64).sin());
        let c = &a * a.transpose() + M::identity(15, 15) * 0.3;
        let w = DenseWhitening::new(c.clone()).unwrap();
        let q = w.apply(&M::identity(15, 15));
        assert!((&q * &c * q.transpose() - M::identity(15, 15)).amax() < 1e-11);
        assert!((w.log_determinant - c.clone().determinant().ln()).abs() < 1e-10);
        let x = M::from_fn(15, 4, |i, j| (i + j) as f64);
        assert!((&w.lower * w.apply(&x) - x).norm() < 1e-10);
    }
    #[test]
    fn whitening_matches_dense_inverse() {
        let u = M::from_fn(9, 4, |i, j| ((i * 7 + j * 3) as f64).sin() / 2.);
        let alpha = 0.1;
        let w = Whitening::new(&u, alpha).unwrap();
        let c = M::identity(9, 9) * alpha + &u * u.transpose() * (1. - alpha);
        let a = w.apply(&M::identity(9, 9));
        assert!((&a * c * a.transpose() - M::identity(9, 9)).amax() < 1e-10);
    }
    #[test]
    fn jackknife_diagonal_matches_saved_definition() {
        let b = M::from_row_slice(2, 4, &[1., 3., 2., 6., 2., 4., 8., 1.]);
        let counts = [2., 3., 4., 1.];
        let u = jackknife_modes(&b, &counts, &[1., 1.]).unwrap();
        for i in 0..2 {
            let mean: f64 = (0..4).map(|j| counts[j] * b[(i, j)] / 10.).sum();
            let l: Vec<f64> = (0..4)
                .map(|j| (mean - counts[j] / 10. * b[(i, j)]) / (1. - counts[j] / 10.))
                .collect();
            let center = l.iter().sum::<f64>() / 4.;
            let var = l.iter().map(|v| (v - center).powi(2)).sum::<f64>() * 0.75;
            assert!((u.row(i).norm_squared() - var).abs() < 1e-12);
        }
    }
}
