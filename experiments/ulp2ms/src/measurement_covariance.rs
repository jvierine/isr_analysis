//! Shrunk joint real/imaginary delete-block covariance, with low-rank whitening.
use anyhow::{ensure, Result};
use nalgebra::DMatrix as M;
pub struct Whitening {
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
            modes,
            correction,
            alpha,
        })
    }
    /// Applied after division by the saved per-measurement standard errors.
    pub fn apply(&self, x: &M<f64>) -> M<f64> {
        let mut q = self.modes.transpose() * x;
        for i in 0..q.nrows() {
            q.row_mut(i).scale_mut(self.correction[i]);
        }
        (x + &self.modes * q) / self.alpha.sqrt()
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
    use super::*;
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
