//! Random-walk prediction and Rauch--Tung--Striebel smoothing (Virtanen2021,
//! doi:10.1029/2020JA028700, equations3--7). Each likelihood is used once.
use anyhow::{ensure, Result};
use nalgebra::{DMatrix as M, DVector as V};
pub struct Prior {
    /// Optional pseudo-Huber transition in standardized innovation coordinates.
    pub robust_delta: Option<f64>,
    pub columns: Vec<usize>,
    pub mean: V<f64>,
    pub inverse_cholesky: M<f64>,
}
impl Prior {
    pub fn new(columns: Vec<usize>, mean: V<f64>, cov: M<f64>) -> Result<Self> {
        ensure!(
            columns.len() == mean.len() && cov.shape() == (mean.len(), mean.len()),
            "temporal prior dimensions"
        );
        let ch = cov
            .cholesky()
            .ok_or_else(|| anyhow::anyhow!("nonpositive temporal prediction covariance"))?;
        let l = ch.l();
        let inverse_cholesky = l
            .solve_lower_triangular(&M::identity(mean.len(), mean.len()))
            .unwrap();
        Ok(Self {
            robust_delta: None,
            columns,
            mean,
            inverse_cholesky,
        })
    }
    pub fn load(path: &str) -> Result<Self> {
        let f = hdf5::File::open(path)?;
        let c = f
            .dataset("columns")?
            .read_raw::<u64>()?
            .into_iter()
            .map(|v| v as usize)
            .collect::<Vec<_>>();
        let n = c.len();
        let mut prior = Self::new(
            c,
            V::from_vec(f.dataset("mean")?.read_raw()?),
            M::from_row_slice(n, n, &f.dataset("covariance")?.read_raw::<f64>()?),
        )?;
        if let Ok(d) = f.dataset("robust_delta") {
            let delta = d.read_scalar::<f64>()?;
            ensure!(
                delta.is_finite() && delta > 0.,
                "invalid robust temporal transition"
            );
            prior.robust_delta = Some(delta);
        }
        Ok(prior)
    }
}
pub fn columns(knots: usize, composition: bool) -> Vec<usize> {
    (0..6 * knots)
        .filter(|i| i % 6 < 4 || !composition || crate::composition::COLUMNS.contains(i))
        .collect()
}
/// Weak nominal30min changes: logNe0.7, logTi0.4, logTe0.5, v200m/s;
/// H centre150km, width factor2; He non-H amplitude0.1, centre200km,width factor2.
pub fn process_sigma(column: usize) -> f64 {
    match column % 6 {
        0 => 0.7,
        1 => 0.4,
        2 => 0.5,
        3 => 0.2,
        _ => match column {
            4 => 150. / 1450.,
            10 => 2f64.ln() / (300f64 / 20.).ln(),
            17 => 0.1 / 0.5,
            23 => 200. / 1050.,
            29 => 2f64.ln() / 10f64.ln(),
            _ => 0.3,
        },
    }
}
pub fn predict(
    mean: &V<f64>,
    cov: &M<f64>,
    columns: &[usize],
    dt: f64,
    strength: f64,
) -> Result<(V<f64>, M<f64>)> {
    ensure!(
        dt > 0. && dt <= 3600. && strength > 0.,
        "invalid temporal interval/reset required"
    );
    let mut predicted = cov.clone();
    for (j, &col) in columns.iter().enumerate() {
        predicted[(j, j)] += process_sigma(col).powi(2) * dt / 1800. * strength.powi(2);
    }
    Ok((mean.clone(), predicted))
}
pub fn smooth(
    filtered: &V<f64>,
    cov: &M<f64>,
    predicted: &V<f64>,
    prediction_cov: &M<f64>,
    next: &V<f64>,
    next_cov: &M<f64>,
) -> Result<(V<f64>, M<f64>)> {
    let ch = prediction_cov
        .clone()
        .cholesky()
        .ok_or_else(|| anyhow::anyhow!("singular RTS prediction"))?;
    let gain = ch.solve(&cov.transpose()).transpose();
    let mean = filtered + &gain * (next - predicted);
    let p = cov + &gain * (next_cov - prediction_cov) * gain.transpose();
    Ok((mean, (&p + p.transpose()) * 0.5))
}
#[cfg(test)]
mod tests {
    use super::*;
    #[test]
    fn forward_reverse_equals_joint_gaussian_and_is_time_symmetric() {
        let y = [1., 3., 2., -1.];
        let r = 0.4;
        let q = 0.7;
        let p0 = 4.;
        let n = y.len();
        let mut mf = vec![];
        let mut pf = vec![];
        let (mut m, mut p) = (0., p0);
        for (j, v) in y.iter().enumerate() {
            if j > 0 {
                p += q;
            }
            let k = p / (p + r);
            m += k * (v - m);
            p *= 1. - k;
            mf.push(m);
            pf.push(p);
        }
        let (mut ms, mut ps) = (mf.clone(), pf.clone());
        for j in (0..n - 1).rev() {
            let (a, b) = smooth(
                &V::from_element(1, mf[j]),
                &M::from_element(1, 1, pf[j]),
                &V::from_element(1, mf[j]),
                &M::from_element(1, 1, pf[j] + q),
                &V::from_element(1, ms[j + 1]),
                &M::from_element(1, 1, ps[j + 1]),
            )
            .unwrap();
            ms[j] = a[0];
            ps[j] = b[(0, 0)];
        }
        let mut h = M::identity(n, n) / r;
        h[(0, 0)] += 1. / p0;
        for j in 1..n {
            h[(j, j)] += 1. / q;
            h[(j - 1, j - 1)] += 1. / q;
            h[(j, j - 1)] -= 1. / q;
            h[(j - 1, j)] -= 1. / q;
        }
        let ch = h.cholesky().unwrap();
        let joint = ch.solve(&(V::from_column_slice(&y) / r));
        let covariance = ch.inverse();
        for j in 0..n {
            assert!((ms[j] - joint[j]).abs() < 1e-12);
            assert!((ps[j] - covariance[(j, j)]).abs() < 1e-12);
        }
    }
}

/// Signed square-root pseudo-Huber loss and its exact derivative.
/// r(z)^2 = 2 delta^2 (sqrt(1+(z/delta)^2)-1).
pub fn pseudo_huber_residual(z: f64, delta: f64) -> (f64, f64) {
    let s = (z / delta).hypot(1.);
    (z * (2. / (s + 1.)).sqrt(), ((s + 1.) / 2.).sqrt() / s)
}
#[cfg(test)]
mod robust_tests {
    #[test]
    fn pseudo_huber_gradient_and_bounded_influence() {
        for z in [-100., -3., -0.01, 0., 0.01, 3., 100.] {
            let (r, d) = super::pseudo_huber_residual(z, 2.);
            let eps = 1e-5;
            let fd = (super::pseudo_huber_residual(z + eps, 2.).0
                - super::pseudo_huber_residual(z - eps, 2.).0)
                / (2. * eps);
            assert!((fd - d).abs() < 1e-8);
            assert!((2. * r * d).abs() <= 4.);
        }
        let (r, _) = super::pseudo_huber_residual(1e-5, 2.);
        assert!((r / 1e-5 - 1.).abs() < 1e-8);
    }
}
