//! Research pilot: signal-dependent *marginal* lag likelihood and hierarchical
//! Student-t gradients. This is not a full correlator covariance likelihood.
//! The rectangular-pulse variance neglects off-diagonal voltage covariance;
//! a learned noise scale does not restore those missing correlations.
use anyhow::{ensure, Result};
/// Stationary range covariance estimated from pre-event temporal contrasts.
/// A finite-order AR precision retains a positive definite joint likelihood.
#[derive(Clone, Debug, serde::Serialize, serde::Deserialize)]
pub struct ArNoise {
    pub coefficients: Vec<f64>,
    pub innovation_variance: f64,
    pub initial_precision: Vec<f64>,
}
impl ArNoise {
    pub fn fit(correlation: &[f64], order: usize) -> Result<Self> {
        ensure!(
            order > 0 && correlation.len() > order && correlation[0] > 0.,
            "invalid AR correlation"
        );
        let c = nalgebra::DMatrix::from_fn(order, order, |i, j| {
            correlation[i.abs_diff(j)] / correlation[0]
        });
        let b = nalgebra::DVector::from_iterator(
            order,
            (1..=order).map(|i| correlation[i] / correlation[0]),
        );
        let chol = c
            .cholesky()
            .ok_or_else(|| anyhow::anyhow!("range correlation is not positive definite"))?;
        let coeff = chol.solve(&b);
        let variance = 1. - coeff.dot(&b);
        ensure!(variance > 0., "invalid AR innovation variance");
        let initial = chol.inverse();
        Ok(Self {
            coefficients: coeff.iter().copied().collect(),
            innovation_variance: variance,
            initial_precision: (0..order)
                .flat_map(|i| {
                    (0..order).map({
                        let initial = &initial;
                        move |j| initial[(i, j)]
                    })
                })
                .collect(),
        })
    }
    pub fn precision(&self, x: &[f64]) -> Vec<f64> {
        let p = self.coefficients.len();
        assert!(x.len() >= p);
        let mut q = vec![0.; x.len()];
        for i in 0..p {
            for j in 0..p {
                q[i] += self.initial_precision[i * p + j] * x[j];
            }
        }
        for i in p..x.len() {
            let e = (x[i]
                - (1..=p)
                    .map(|j| self.coefficients[j - 1] * x[i - j])
                    .sum::<f64>())
                / self.innovation_variance;
            q[i] += e;
            for j in 1..=p {
                q[i - j] -= self.coefficients[j - 1] * e;
            }
        }
        q
    }
}
/// Fixed covariance for a feasible-GLS pass. The first-pass covariance is held
/// fixed while fitting the mean; only its explicit common scale is inferred.
/// This is not a substitute for a signal-dependent covariance likelihood.
#[derive(Clone)]
pub struct FixedLagCovariance {
    precision: nalgebra::DMatrix<f64>,
    log_determinant: f64,
}
impl FixedLagCovariance {
    pub fn from_rows(rows: &[Vec<f64>]) -> Result<Self> {
        let n = rows.len();
        ensure!(
            n > 0 && rows.iter().all(|r| r.len() == n),
            "Invalid covariance rows"
        );
        Self::new(nalgebra::DMatrix::from_fn(n, n, |i, j| rows[i][j]))
    }

    pub fn new(matrix: nalgebra::DMatrix<f64>) -> Result<Self> {
        ensure!(
            matrix.is_square()
                && matrix.nrows() > 0
                && matrix.nrows() % 2 == 0
                && matrix.iter().all(|v| v.is_finite()),
            "Invalid fixed covariance"
        );
        ensure!(
            (&matrix - matrix.transpose()).amax() <= 1e-10 * matrix.amax().max(f64::MIN_POSITIVE),
            "Asymmetric fixed covariance"
        );
        let chol = matrix
            .cholesky()
            .ok_or_else(|| anyhow::anyhow!("Fixed covariance must be positive definite"))?;
        let log_determinant = 2. * chol.l().diagonal().iter().map(|x| x.ln()).sum::<f64>();
        Ok(Self {
            precision: chol.inverse(),
            log_determinant,
        })
    }
}
#[derive(Clone)]
pub struct Problem {
    pub gates: usize,
    pub cycles: usize,
    pub range_covariance: [Option<ArNoise>; 2],
    pub received_power: Option<Vec<f64>>,
    pub fixed_variance: Option<Vec<f64>>,
    pub fixed_covariance: Option<std::sync::Arc<Vec<FixedLagCovariance>>>,
    pub range_response: Option<Vec<Vec<(usize, f64)>>>,
    pub real: Vec<f64>,
    pub imaginary: Vec<f64>,
    pub long_mean: Vec<f64>,
    pub long_sigma: Vec<f64>,
    pub phase: Vec<f64>,
    pub external: Vec<f64>,
    pub receiver: f64,
    pub pulse_gain: f64,
    pub dc_notch: bool,
}
impl Problem {
    pub fn validate(&self) -> Result<()> {
        let n = self.gates;
        let m = self.cycles;
        if let Some(c) = &self.fixed_covariance {
            ensure!(
                c.len() == m && c.iter().all(|v| v.precision.nrows() == 2 * n),
                "Fixed covariance dimensions"
            );
            ensure!(!self.dc_notch && self.received_power.is_none() && self.fixed_variance.is_none()
                && self.range_covariance.iter().all(Option::is_none)
                && self.long_sigma.iter().all(|v|!v.is_finite()),"Full-covariance short-only pass cannot combine overlapping or approximate likelihoods");
        }
        ensure!(
            self.fixed_variance
                .as_ref()
                .is_none_or(|v| v.len() == n * m && v.iter().all(|x| x.is_finite() && *x > 0.)),
            "invalid fixed variance"
        );
        ensure!(
            n > 2 && m == 14 && self.external.len() >= n + 62,
            "invalid native shape"
        );
        ensure!(
            self.real.len() == n * m && self.imaginary.len() == n * m,
            "measurement dimensions"
        );
        if let Some(rows) = &self.range_response {
            ensure!(
                rows.len() == n
                    && rows.iter().all(|row| !row.is_empty()
                        && row.iter().all(|(j, w)| *j < n && w.is_finite())
                        && (row.iter().map(|(_, w)| w).sum::<f64>() - 1.).abs() < 1e-8),
                "invalid range response"
            );
        }
        ensure!(
            [&self.long_mean, &self.long_sigma, &self.phase]
                .iter()
                .all(|a| a.len() == n),
            "long-lag dimensions"
        );
        ensure!(
            self.received_power
                .as_ref()
                .is_none_or(|p| p.len() == n + 1 && p.iter().all(|v| v.is_finite() && *v > 0.)),
            "invalid receive-power measurements"
        );
        ensure!(
            self.receiver > 0. && self.pulse_gain > 0.,
            "invalid noise or pulse gain"
        );
        ensure!(
            self.real
                .iter()
                .chain(&self.imaginary)
                .chain(&self.external)
                .chain(&self.phase)
                .all(|v| v.is_finite()),
            "nonfinite measurements"
        );
        ensure!(
            self.external.iter().all(|v| *v >= 0.),
            "negative external power"
        );
        Ok(())
    }
    fn fisher(&self, z: &[f64]) -> Vec<f64> {
        let n = self.gates;
        let m = self.cycles;
        let d = n * m;
        let eta = z[d + 2].exp();
        let x: Vec<_> = z[..d].iter().map(|q| 10. * q.exp()).collect();
        let mut out = vec![0.; d];
        let (a, b) = if self.dc_notch {
            (12. / 13., 1. / 182.)
        } else {
            (1., 0.)
        };
        let sums: Vec<_> = (0..n)
            .map(|r| (0..m).map(|t| x[t * n + r]).sum::<f64>())
            .collect();
        for t in 0..m {
            let mut xx = self.external.clone();
            for r in 0..n {
                xx[r] = a * x[t * n + r] + b * sums[r];
            }
            let p: Vec<_> = (0..=n)
                .map(|k| {
                    self.receiver
                        + self.pulse_gain * xx[k.saturating_sub(2)..k + 62].iter().sum::<f64>()
                })
                .collect();
            for r in 0..n {
                let pairs = (r / 2 + 1).min(31);
                let v = eta
                    * (0..pairs)
                        .map(|j| p[r - 2 * j] * p[r - 2 * j + 1])
                        .sum::<f64>()
                    / (128. * (pairs * pairs) as f64);
                let i = t * n + r;
                let v = self.fixed_variance.as_ref().map_or(v, |a| eta * a[i]);
                out[i] = if let Some(c) = &self.fixed_covariance {
                    let p = &c[t].precision;
                    let ph = self.phase[r];
                    x[i] * x[i]
                        * (p[(2 * r, 2 * r)]
                            + 2. * ph * p[(2 * r, 2 * r + 1)]
                            + ph * ph * p[(2 * r + 1, 2 * r + 1)])
                        / eta
                } else {
                    x[i] * x[i] * (1. + self.phase[r] * self.phase[r]) / v
                };
                if self.long_sigma[r].is_finite() && self.long_sigma[r] > 0. {
                    out[i] += x[i] * x[i] / (self.long_sigma[r].powi(2) * m as f64 * eta);
                }
            }
        }
        out
    }
    /// Log state consists of brightness / 10 counts followed by spatial gradient
    /// scale, temporal gradient scale, and a variance multiplier, all logarithmic.
    pub fn evaluate(&self, z: &[f64]) -> (f64, Vec<f64>) {
        let n = self.gates;
        let m = self.cycles;
        let d = n * m;
        let mut grad = vec![0.; d + 3];
        if z.len() != d + 3 || z.iter().any(|v| !v.is_finite() || v.abs() > 40.) {
            return (f64::INFINITY, grad);
        }
        let x: Vec<_> = z[..d].iter().map(|v| 10. * v.exp()).collect();
        let sr = z[d].exp();
        let st = z[d + 1].exp();
        let eta = z[d + 2].exp();
        let (a, b) = if self.dc_notch {
            (12. / 13., 1. / 182.)
        } else {
            (1., 0.)
        };
        let mut mixed = vec![0.; d];
        let mut gx = vec![0.; d];
        let mut gm = vec![0.; d];
        for r in 0..n {
            let sum = (0..m).map(|t| x[t * n + r]).sum::<f64>();
            for t in 0..m {
                mixed[t * n + r] = a * x[t * n + r] + b * sum;
            }
        }
        let predicted: Vec<f64> = if let Some(rows) = &self.range_response {
            (0..d)
                .map(|i| {
                    rows[i % n]
                        .iter()
                        .map(|(j, w)| mixed[(i / n) * n + j] * w)
                        .sum()
                })
                .collect()
        } else {
            mixed.clone()
        };
        let mut cost = 0.;
        if let Some(covariances) = &self.fixed_covariance {
            for t in 0..m {
                let c = &covariances[t];
                let residual = nalgebra::DVector::from_iterator(
                    2 * n,
                    (0..n).flat_map(|r| {
                        let i = t * n + r;
                        [
                            predicted[i] - self.real[i],
                            predicted[i] * self.phase[r] - self.imaginary[i],
                        ]
                    }),
                );
                let weighted = &c.precision * &residual / eta;
                let q = residual.dot(&weighted);
                cost += 0.5 * (c.log_determinant + 2. * n as f64 * eta.ln() + q);
                grad[d + 2] += n as f64 - 0.5 * q;
                for r in 0..n {
                    let g = weighted[2 * r] + self.phase[r] * weighted[2 * r + 1];
                    if let Some(rows) = &self.range_response {
                        for &(j, w) in &rows[r] {
                            gm[t * n + j] += w * g;
                        }
                    } else {
                        gm[t * n + r] += g;
                    }
                }
            }
        } else {
            let mut powers = vec![0.; m * (n + 1)];
            let mut variance = vec![0.; d];
            let mut standardized = [vec![0.; d], vec![0.; d]];
            for t in 0..m {
                let mut xx = self.external.clone();
                xx[..n].copy_from_slice(&mixed[t * n..(t + 1) * n]);
                let mut prefix = vec![0.; xx.len() + 1];
                for (i, v) in xx.iter().enumerate() {
                    prefix[i + 1] = prefix[i] + v;
                }
                let p = &mut powers[t * (n + 1)..(t + 1) * (n + 1)];
                for k in 0..=n {
                    p[k] = self.receiver
                        + self.pulse_gain
                            * (prefix[(k + 62).min(xx.len())] - prefix[k.saturating_sub(2)]);
                }
                let mut parity = vec![0.; n + 2];
                for k in 0..n {
                    parity[k + 2] = parity[k] + p[k] * p[k + 1];
                }
                for r in 0..n {
                    let pairs = (r / 2 + 1).min(31) as f64;
                    let k0 = r.saturating_sub(60);
                    let k0 = if k0 % 2 == r % 2 { k0 } else { k0 + 1 };
                    let v = self.fixed_variance.as_ref().map_or(
                        (parity[r + 2] - parity[k0]) * eta / (128. * pairs * pairs),
                        |a| eta * a[t * n + r],
                    );
                    if !v.is_finite() || v <= 0. {
                        return (f64::INFINITY, grad);
                    }
                    let i = t * n + r;
                    variance[i] = v;
                    cost += v.ln();
                    standardized[0][i] = (predicted[i] - self.real[i]) / v.sqrt();
                    standardized[1][i] =
                        (predicted[i] * self.phase[r] - self.imaginary[i]) / v.sqrt();
                }
            }
            let mut power_gradient = vec![0.; n + 1];
            if let Some(observed) = &self.received_power {
                for k in 0..=n {
                    let mean = (0..m).map(|t| powers[t * (n + 1) + k]).sum::<f64>() / m as f64;
                    let signal = mean - self.receiver;
                    // Independent-pulse variance plus a documented 10% forward-model
                    // discrepancy allowance for the rectangular receive-power response.
                    let v = eta * (mean * mean / 896. + 0.01 * signal * signal);
                    let e = mean - observed[k];
                    cost += 0.5 * (v.ln() + e * e / v);
                    let dv = 0.5 * (1. / v - e * e / (v * v));
                    grad[d + 2] += dv * v;
                    power_gradient[k] =
                        (e / v + dv * eta * (2. * mean / 896. + 0.02 * signal)) / m as f64;
                }
            }
            let mut weighted = standardized.clone();
            for component in 0..2 {
                if let Some(ar) = &self.range_covariance[component] {
                    for t in 0..m {
                        weighted[component][t * n..(t + 1) * n].copy_from_slice(
                            &ar.precision(&standardized[component][t * n..(t + 1) * n]),
                        );
                    }
                }
                // Stationary Gaussian-voltage DC-notch hypothesis: normalized product
                // covariance between cycles is 1/169. This separable approximation does
                // not claim the exact nonstationary correlator covariance.
                if self.dc_notch {
                    let c = 1. / 169.;
                    let aa = 1. / (1. - c);
                    let bb = -c / ((1. - c) * (1. + (m - 1) as f64 * c));
                    for r in 0..n {
                        let sum = (0..m).map(|t| weighted[component][t * n + r]).sum::<f64>();
                        for t in 0..m {
                            weighted[component][t * n + r] =
                                aa * weighted[component][t * n + r] + bb * sum;
                        }
                    }
                }
            }
            for t in 0..m {
                let p = &powers[t * (n + 1)..(t + 1) * (n + 1)];
                let mut adj = vec![0.; n + 2];
                for r in 0..n {
                    let i = t * n + r;
                    let q =
                        standardized[0][i] * weighted[0][i] + standardized[1][i] * weighted[1][i];
                    cost += 0.5 * q;
                    let mean_gradient =
                        (weighted[0][i] + self.phase[r] * weighted[1][i]) / variance[i].sqrt();
                    if let Some(rows) = &self.range_response {
                        for &(j, w) in &rows[r] {
                            gm[t * n + j] += w * mean_gradient;
                        }
                    } else {
                        gm[i] += mean_gradient;
                    }
                    let dv = (1. - 0.5 * q) / variance[i];
                    grad[d + 2] += dv * variance[i];
                    let pairs = (r / 2 + 1).min(31) as f64;
                    let k0 = r.saturating_sub(60);
                    let k0 = if k0 % 2 == r % 2 { k0 } else { k0 + 1 };
                    let c = if self.fixed_variance.is_some() {
                        0.
                    } else {
                        dv * eta / (128. * pairs * pairs)
                    };
                    adj[k0] += c;
                    adj[r + 2] -= c;
                }
                let mut gp = power_gradient.clone();
                for k in 0..n {
                    if k >= 2 {
                        adj[k] += adj[k - 2];
                    }
                    gp[k] += adj[k] * p[k + 1];
                    gp[k + 1] += adj[k] * p[k];
                }
                let mut diff = vec![0.; self.external.len() + 1];
                for k in 0..=n {
                    let c = gp[k] * self.pulse_gain;
                    diff[k.saturating_sub(2)] += c;
                    diff[(k + 62).min(self.external.len())] -= c;
                }
                let mut run = 0.;
                for r in 0..n {
                    run += diff[r];
                    gm[t * n + r] += run;
                }
            }
        } // marginal approximation or fixed full-covariance GLS pass
        for r in 0..n {
            let total = (0..m).map(|t| gm[t * n + r]).sum::<f64>();
            for t in 0..m {
                gx[t * n + r] = a * gm[t * n + r] + b * total;
            }
            if self.long_mean[r].is_finite()
                && self.long_sigma[r].is_finite()
                && self.long_sigma[r] > 0.
            {
                let avg = (0..m).map(|t| x[t * n + r]).sum::<f64>() / m as f64;
                let v = self.long_sigma[r].powi(2) * eta;
                let e = avg - self.long_mean[r];
                cost += 0.5 * (v.ln() + e * e / v);
                grad[d + 2] += 0.5 * (1. - e * e / v);
                for t in 0..m {
                    gx[t * n + r] += e / (v * m as f64);
                }
            }
        }
        for i in 0..d {
            grad[i] = gx[i] * x[i];
        }
        // A directed innovation model has exactly d-1 independent differences,
        // with a unit triangular Jacobian. Multiplying normalized edge densities
        // over every edge of a cyclic grid would use the wrong partition function.
        cost += 0.5 * z[0] * z[0] / 100.;
        grad[0] += z[0] / 100.;
        let mut add_prior = |terms: &[(usize, f64)], h: usize| {
            let scale = if h == 0 { sr } else { st };
            let delta = terms.iter().map(|(i, w)| z[*i] * w).sum::<f64>();
            let denom = 2. * scale * scale + delta * delta;
            cost += scale.ln() + 1.5 * (delta * delta / (2. * scale * scale)).ln_1p();
            let g = 3. * delta / denom;
            for &(i, w) in terms {
                grad[i] += w * g;
            }
            grad[d + h] += 1. - 3. * delta * delta / denom;
        };
        for r in 1..n {
            add_prior(&[(r, 1.), (r - 1, -1.)], 0);
        }
        for t in 1..m {
            add_prior(&[(t * n, 1.), ((t - 1) * n, -1.)], 1);
            for r in 1..n {
                add_prior(
                    &[
                        (t * n + r, 1.),
                        (t * n + r - 1, -0.5),
                        ((t - 1) * n + r, -0.5),
                    ],
                    1,
                );
            }
        }
        // Proper lognormal hyperpriors: relative range changes, cycle changes and
        // unexplained variance. Priors are visible inputs in exported provenance.
        for (i, center, sd) in [
            (d, 0.3f64.ln(), 1.),
            (d + 1, 0.5f64.ln(), 1.),
            (d + 2, 0., 1.),
        ] {
            let e = (z[i] - center) / sd;
            cost += 0.5 * e * e;
            grad[i] += e / sd;
        }
        (cost, grad)
    }
}
struct Rng(u64);
impl Rng {
    fn uniform(&mut self) -> f64 {
        self.0 = self
            .0
            .wrapping_mul(6364136223846793005)
            .wrapping_add(1442695040888963407);
        ((self.0 >> 11) as f64 + 0.5) / 9007199254740992.
    }
    fn normal(&mut self) -> f64 {
        (-2. * self.uniform().ln()).sqrt() * (std::f64::consts::TAU * self.uniform()).cos()
    }
}
/// Independent innovations: initial altitude profile, subsequent time
/// increments at the first gate, and deviations from the mean of the two
/// preceding neighbours. Unlike mixed differences, a persistent false range
/// step is penalized at every time, not only in the first profile.
fn prior_terms(n: usize, m: usize) -> Vec<(Vec<(usize, f64)>, usize)> {
    let mut out = Vec::with_capacity(n * m - 1);
    for r in 1..n {
        out.push((vec![(r, 1.), (r - 1, -1.)], 0));
    }
    for t in 1..m {
        out.push((vec![(t * n, 1.), ((t - 1) * n, -1.)], 1));
        for r in 1..n {
            out.push((
                vec![
                    (t * n + r, 1.),
                    (t * n + r - 1, -0.5),
                    ((t - 1) * n + r, -0.5),
                ],
                1,
            ));
        }
    }
    out
}
// Exact band Cholesky of the local positive curvature approximation.
struct Mass {
    l: Vec<f64>,
    n: usize,
    m: usize,
    size: usize,
    width: usize,
}
impl Mass {
    fn index(&self, k: usize) -> usize {
        if k < self.n * self.m {
            (k % self.m) * self.n + k / self.m
        } else {
            k
        }
    }
    fn build(problem: &Problem, z: &[f64]) -> Self {
        let n = problem.gates;
        let m = problem.cycles;
        let size = n * m + 3;
        let width = m + 2;
        let mut h = vec![0.; size * width];
        let sr = z[n * m].exp();
        let st = z[n * m + 1].exp();
        let fisher = problem.fisher(z);
        for r in 0..n {
            for t in 0..m {
                let k = r * m + t;
                let i = t * n + r;
                h[k * width] = 0.05 + fisher[i];
            }
        }
        for (terms, which) in prior_terms(n, m) {
            let scale = if which == 0 { sr } else { st };
            let delta = terms.iter().map(|(i, w)| z[*i] * w).sum::<f64>();
            let weight = 3. / (2. * scale * scale + delta * delta);
            for &(oi, wi) in &terms {
                let i = (oi % n) * m + oi / n;
                for &(oj, wj) in &terms {
                    let j = (oj % n) * m + oj / n;
                    if i >= j {
                        h[i * width + i - j] += weight * wi * wj;
                    }
                }
            }
        }
        for k in n * m..size {
            h[k * width] = (n * m) as f64;
        }
        let mut l = vec![0.; size * width];
        for i in 0..size {
            for j in i.saturating_sub(m + 1)..=i {
                let mut v = h[i * width + i - j];
                for k in i.saturating_sub(m + 1)..j {
                    v -= l[i * width + i - k] * l[j * width + j - k];
                }
                l[i * width + i - j] = if i == j {
                    v.max(1e-12).sqrt()
                } else {
                    v / l[j * width]
                };
            }
        }
        Self {
            l,
            n,
            m,
            size,
            width,
        }
    }
    fn momentum(&self, rng: &mut Rng) -> Vec<f64> {
        let noise: Vec<_> = (0..self.size).map(|_| rng.normal()).collect();
        let mut out = vec![0.; self.size];
        for i in 0..self.size {
            out[self.index(i)] = (i.saturating_sub(self.m + 1)..=i)
                .map(|j| self.l[i * self.width + i - j] * noise[j])
                .sum();
        }
        out
    }
    fn solve(&self, p: &[f64]) -> Vec<f64> {
        let mut y = vec![0.; self.size];
        for i in 0..self.size {
            let mut v = p[self.index(i)];
            for j in i.saturating_sub(self.m + 1)..i {
                v -= self.l[i * self.width + i - j] * y[j];
            }
            y[i] = v / self.l[i * self.width];
        }
        let mut x = vec![0.; self.size];
        for i in (0..self.size).rev() {
            let mut v = y[i];
            for j in i + 1..(i + self.m + 2).min(self.size) {
                v -= self.l[j * self.width + j - i] * x[j];
            }
            x[i] = v / self.l[i * self.width];
        }
        let mut out = vec![0.; self.size];
        for i in 0..self.size {
            out[self.index(i)] = x[i];
        }
        out
    }
}
fn integrate(u: &[f64], n: usize, m: usize, sr: f64, st: f64) -> Vec<f64> {
    let d = n * m;
    let mut z = vec![0.; d + 3];
    for t in 0..m {
        for r in 0..n {
            let i = t * n + r;
            let scale = if i == 0 {
                1.
            } else if t == 0 {
                sr
            } else {
                st
            };
            z[i] = u[i] * scale
                + if t > 0 {
                    z[i - n] * if r > 0 { 0.5 } else { 1. }
                } else {
                    0.
                }
                + if r > 0 {
                    z[i - 1] * if t > 0 { 0.5 } else { 1. }
                } else {
                    0.
                };
        }
    }
    z[d..].copy_from_slice(&u[d..]);
    z
}
fn difference(z: &[f64], n: usize, m: usize, sr: f64, st: f64) -> Vec<f64> {
    let d = n * m;
    let mut u = vec![0.; d + 3];
    for t in 0..m {
        for r in 0..n {
            let i = t * n + r;
            let v =
                z[i] - if t > 0 {
                    z[i - n] * if r > 0 { 0.5 } else { 1. }
                } else {
                    0.
                } - if r > 0 {
                    z[i - 1] * if t > 0 { 0.5 } else { 1. }
                } else {
                    0.
                };
            u[i] = v / if i == 0 {
                1.
            } else if t == 0 {
                sr
            } else {
                st
            };
        }
    }
    u[d..].copy_from_slice(&z[d..]);
    u
}
fn integrate_transpose(g: &[f64], n: usize, m: usize, sr: f64, st: f64) -> Vec<f64> {
    let d = n * m;
    let mut suffix = g[..d].to_vec();
    for t in (0..m).rev() {
        for r in (0..n).rev() {
            let i = t * n + r;
            let v = suffix[i];
            if t > 0 {
                suffix[i - n] += v * if r > 0 { 0.5 } else { 1. };
            }
            if r > 0 {
                suffix[i - 1] += v * if t > 0 { 0.5 } else { 1. };
            }
        }
    }
    let mut out = g.to_vec();
    for t in 0..m {
        for r in 0..n {
            let i = t * n + r;
            out[i] = suffix[i]
                * if i == 0 {
                    1.
                } else if t == 0 {
                    sr
                } else {
                    st
                };
        }
    }
    out
}
fn difference_transpose(g: &[f64], n: usize, m: usize, sr: f64, st: f64) -> Vec<f64> {
    let d = n * m;
    let mut out = vec![0.; d + 3];
    for t in 0..m {
        for r in 0..n {
            let i = t * n + r;
            let v = g[i]
                / if i == 0 {
                    1.
                } else if t == 0 {
                    sr
                } else {
                    st
                };
            out[i] += v;
            if t > 0 {
                out[i - n] -= v * if r > 0 { 0.5 } else { 1. };
            }
            if r > 0 {
                out[i - 1] -= v * if t > 0 { 0.5 } else { 1. };
            }
        }
    }
    out[d..].copy_from_slice(&g[d..]);
    out
}
fn noncentered(problem: &Problem, u: &[f64]) -> (f64, Vec<f64>) {
    let n = problem.gates;
    let m = problem.cycles;
    let d = n * m;
    let sr = u[d].exp();
    let st = u[d + 1].exp();
    let z = integrate(u, n, m, sr, st);
    let (cost, g) = problem.evaluate(&z);
    if !cost.is_finite() {
        return (cost, vec![0.; u.len()]);
    }
    let mut out = integrate_transpose(&g, n, m, sr, st);
    let mut source = vec![0.; d + 3];
    source[1..n].copy_from_slice(&u[1..n]);
    let spatial = integrate(&source, n, m, sr, st);
    for i in 0..d {
        out[d] += g[i] * spatial[i];
        out[d + 1] += g[i] * (z[i] - u[0] - spatial[i]);
    }
    out[d] -= (n - 1) as f64;
    out[d + 1] -= (n * (m - 1)) as f64;
    (
        cost - (n - 1) as f64 * u[d] - (n * (m - 1)) as f64 * u[d + 1],
        out,
    )
}
struct TransportMass {
    base: Mass,
    sr: f64,
    st: f64,
    coupling: [Vec<f64>; 2],
    inverse_coupling: [Vec<f64>; 2],
    schur: [f64; 3],
}
impl TransportMass {
    fn build(p: &Problem, u: &[f64]) -> Self {
        let d = p.gates * p.cycles;
        let sr = u[d].exp();
        let st = u[d + 1].exp();
        let z = integrate(u, p.gates, p.cycles, sr, st);
        let base = Mass::build(p, &z);
        let fi = p.fisher(&z);
        let mut aa = 1.;
        let mut bb = 1.;
        let mut ab = 0.;
        let mut source = vec![0.; d + 3];
        source[1..p.gates].copy_from_slice(&u[1..p.gates]);
        let spatial = integrate(&source, p.gates, p.cycles, sr, st);
        let mut coupling = [vec![0.; d + 3], vec![0.; d + 3]];
        for i in 0..d {
            let temporal = z[i] - u[0] - spatial[i];
            aa += fi[i] * spatial[i] * spatial[i];
            bb += fi[i] * temporal * temporal;
            ab += fi[i] * spatial[i] * temporal;
            coupling[0][i] = fi[i] * spatial[i];
            coupling[1][i] = fi[i] * temporal;
        }
        let inverse_coupling = [base.solve(&coupling[0]), base.solve(&coupling[1])];
        aa -= coupling[0]
            .iter()
            .zip(&inverse_coupling[0])
            .map(|(a, b)| a * b)
            .sum::<f64>();
        bb -= coupling[1]
            .iter()
            .zip(&inverse_coupling[1])
            .map(|(a, b)| a * b)
            .sum::<f64>();
        ab -= coupling[0]
            .iter()
            .zip(&inverse_coupling[1])
            .map(|(a, b)| a * b)
            .sum::<f64>();
        let l0 = aa.max(1e-6).sqrt();
        let l1 = ab / l0;
        let l2 = (bb - l1 * l1).max(1e-6).sqrt();
        Self {
            base,
            sr,
            st,
            coupling,
            inverse_coupling,
            schur: [l0, l1, l2],
        }
    }
    fn momentum(&self, rng: &mut Rng) -> Vec<f64> {
        let d = self.base.n * self.base.m;
        let mut centered = self.base.momentum(rng);
        centered[d..].fill(0.);
        let velocity = self.base.solve(&centered);
        let mut out = integrate_transpose(&centered, self.base.n, self.base.m, self.sr, self.st);
        let a = rng.normal();
        let b = rng.normal();
        out[d] = self.coupling[0]
            .iter()
            .zip(&velocity)
            .map(|(a, b)| a * b)
            .sum::<f64>()
            + self.schur[0] * a;
        out[d + 1] = self.coupling[1]
            .iter()
            .zip(&velocity)
            .map(|(a, b)| a * b)
            .sum::<f64>()
            + self.schur[1] * a
            + self.schur[2] * b;
        out[d + 2] = (d as f64).sqrt() * rng.normal();
        out
    }
    fn solve(&self, p: &[f64]) -> Vec<f64> {
        let d = self.base.n * self.base.m;
        let mut centered = difference_transpose(p, self.base.n, self.base.m, self.sr, self.st);
        centered[d..].fill(0.);
        let mut velocity = self.base.solve(&centered);
        let a = p[d]
            - self.coupling[0]
                .iter()
                .zip(&velocity)
                .map(|(a, b)| a * b)
                .sum::<f64>();
        let b = p[d + 1]
            - self.coupling[1]
                .iter()
                .zip(&velocity)
                .map(|(a, b)| a * b)
                .sum::<f64>();
        let y0 = a / self.schur[0];
        let y1 = (b - self.schur[1] * y0) / self.schur[2];
        let v1 = y1 / self.schur[2];
        let v0 = (y0 - self.schur[1] * v1) / self.schur[0];
        for (i, v) in velocity.iter_mut().enumerate().take(d) {
            *v -= self.inverse_coupling[0][i] * v0 + self.inverse_coupling[1][i] * v1;
        }
        let mut out = difference(&velocity, self.base.n, self.base.m, self.sr, self.st);
        out[d] = v0;
        out[d + 1] = v1;
        out[d + 2] = p[d + 2] / d as f64;
        out
    }
    fn kinetic(&self, p: &[f64]) -> f64 {
        0.5 * p.iter().zip(self.solve(p)).map(|(p, v)| p * v).sum::<f64>()
    }
}
fn slice_parameter(problem: &Problem, z: &mut [f64], index: usize, rng: &mut Rng) {
    let old = z[index];
    let threshold = problem.evaluate(z).0 - rng.uniform().ln();
    let width = 0.5;
    let mut low = old - width * rng.uniform();
    let mut high = low + width;
    let mut left = (20. * rng.uniform()) as usize;
    let mut right = 19 - left;
    while left > 0 {
        z[index] = low;
        if problem.evaluate(z).0 >= threshold {
            break;
        }
        low -= width;
        left -= 1;
    }
    while right > 0 {
        z[index] = high;
        if problem.evaluate(z).0 >= threshold {
            break;
        }
        high += width;
        right -= 1;
    }
    for _ in 0..1000 {
        let candidate = low + (high - low) * rng.uniform();
        z[index] = candidate;
        if problem.evaluate(z).0 <= threshold {
            return;
        }
        if candidate < old {
            low = candidate;
        } else {
            high = candidate;
        }
    }
    z[index] = old;
}
fn block_moves(problem: &Problem, u: &mut Vec<f64>, rng: &mut Rng) {
    let n = problem.gates;
    let m = problem.cycles;
    let d = n * m;
    let sr = u[d].exp();
    let st = u[d + 1].exp();
    let mut z = integrate(u, n, m, sr, st);
    let mut cost = problem.evaluate(&z).0;
    // Symmetric proposals move coherent layers and nearby edge locations. They
    // leave the target density unchanged; no smoothing or post-hoc relocation.
    for proposal in 0..12 {
        let mut candidate = z.clone();
        let r = (rng.uniform() * n as f64) as usize;
        if proposal < 3 {
            let j = (r + 1).min(n - 1);
            for t in 0..m {
                candidate.swap(t * n + r, t * n + j);
            }
        } else {
            let width = if proposal < 7 {
                1
            } else {
                1usize << ((rng.uniform() * 5.) as usize)
            };
            let (t0, t1) = if proposal < 7 {
                (0, m)
            } else {
                let t = (rng.uniform() * m as f64) as usize;
                (t, (t + 1 + (rng.uniform() * m as f64) as usize).min(m))
            };
            let delta = rng.normal() * [0.05, 0.2, 0.6][(rng.uniform() * 3.) as usize];
            for t in t0..t1 {
                for rr in r..(r + width).min(n) {
                    candidate[t * n + rr] += delta;
                }
            }
        }
        let next = problem.evaluate(&candidate).0;
        if next.is_finite() && rng.uniform().ln() < cost - next {
            z = candidate;
            cost = next;
        }
    }
    *u = difference(&z, n, m, sr, st);
}
pub struct Chain {
    pub samples: Vec<Vec<f64>>,
    pub acceptance: f64,
    pub step: f64,
    pub costs: Vec<f64>,
}
pub fn sample(
    problem: &Problem,
    initial: &[f64],
    seed: u64,
    burn: usize,
    keep: usize,
) -> Result<Chain> {
    problem.validate()?;
    let d = problem.gates * problem.cycles;
    ensure!(initial.len() == d + 3, "initial shape");
    let mut rng = Rng(seed);
    let mut z = initial.to_vec();
    for q in &mut z[..d] {
        *q += 0.05 * rng.normal();
    }
    z = difference(
        &z,
        problem.gates,
        problem.cycles,
        z[d].exp(),
        z[d + 1].exp(),
    );
    let mut mass = TransportMass::build(problem, &z);
    let (mut cost, mut gradient) = noncentered(problem, &z);
    ensure!(cost.is_finite(), "invalid initial posterior");
    let mut step = 0.1;
    let mut samples = Vec::new();
    let mut costs = Vec::new();
    let mut accepted = 0;
    for it in 0..burn + 2 * keep {
        if it < burn && it % 50 == 0 {
            mass = TransportMass::build(problem, &z);
        }
        if it == burn {
            step *= 0.5;
        }
        let mut q = z.clone();
        let mut p = mass.momentum(&mut rng);
        let energy = cost + mass.kinetic(&p);
        for i in 0..p.len() {
            p[i] -= 0.5 * step * gradient[i];
        }
        let leap = if it < burn {
            8 + (rng.uniform() * 8.) as usize
        } else {
            16 + (rng.uniform() * 16.) as usize
        };
        let mut next_cost = f64::INFINITY;
        for l in 0..leap {
            let velocity = mass.solve(&p);
            for i in 0..q.len() {
                q[i] += step * velocity[i];
            }
            let (proposed_cost, next_gradient) = noncentered(problem, &q);
            next_cost = proposed_cost;
            if !next_cost.is_finite() {
                break;
            }
            let weight = if l + 1 == leap { 0.5 } else { 1. };
            for i in 0..p.len() {
                p[i] -= weight * step * next_gradient[i];
            }
        }
        let next_energy = next_cost + mass.kinetic(&p);
        let probability = if next_energy.is_finite() {
            (energy - next_energy).min(0.).exp()
        } else {
            0.
        };
        if rng.uniform() < probability {
            z = q;
            if it >= burn {
                accepted += 1;
            }
        }
        block_moves(problem, &mut z, &mut rng);
        (cost, gradient) = noncentered(problem, &z);
        // Interweave exact centered conditional hyperparameter updates with
        // noncentered trajectories to avoid either form of the hierarchical funnel.
        if it >= burn && it % 10 == 0 {
            let mut centered = integrate(
                &z,
                problem.gates,
                problem.cycles,
                z[d].exp(),
                z[d + 1].exp(),
            );
            for parameter in d..d + 3 {
                slice_parameter(problem, &mut centered, parameter, &mut rng);
            }
            z = difference(
                &centered,
                problem.gates,
                problem.cycles,
                centered[d].exp(),
                centered[d + 1].exp(),
            );
            (cost, gradient) = noncentered(problem, &z);
        }
        if it < burn {
            step = (step * ((probability - 0.7) / (it as f64 + 20.).sqrt()).exp()).clamp(1e-5, 0.7);
        } else if (it - burn) % 2 == 0 {
            samples.push(integrate(
                &z,
                problem.gates,
                problem.cycles,
                z[d].exp(),
                z[d + 1].exp(),
            ));
            costs.push(cost);
        }
    }
    Ok(Chain {
        samples,
        acceptance: accepted as f64 / (2 * keep) as f64,
        step,
        costs,
    })
}
#[cfg(test)]
mod tests {
    use super::*;
    fn problem() -> Problem {
        Problem {
            range_response: Some(
                (0..5)
                    .map(|r| vec![(r, 0.7), ((r + 1).min(4), 0.3)])
                    .collect(),
            ),
            fixed_variance: None,
            fixed_covariance: None,
            received_power: Some(vec![800.; 6]),
            range_covariance: [
                Some(ArNoise::fit(&[1., 0.18, 0.015], 2).unwrap()),
                Some(ArNoise::fit(&[1., 0.07, 0.002], 2).unwrap()),
            ],
            gates: 5,
            cycles: 14,
            real: vec![12.; 70],
            imaginary: vec![1.; 70],
            long_mean: vec![11.; 5],
            long_sigma: vec![3.; 5],
            phase: vec![0.1; 5],
            external: vec![10.; 70],
            receiver: 100.,
            pulse_gain: 1.1,
            dc_notch: true,
        }
    }
    #[test]
    fn full_covariance_gls_gradient_includes_scale_and_range_mixing() {
        let mut p = problem();
        p.dc_notch = false;
        p.received_power = None;
        p.long_mean.fill(f64::NAN);
        p.long_sigma.fill(f64::NAN);
        p.range_covariance = [None, None];
        let matrix = nalgebra::DMatrix::from_fn(10, 10, |i, j| {
            let u = |k: usize| 0.2 * (k + 1) as f64;
            u(i) * u(j) + if i == j { 2. + i as f64 } else { 0. }
        });
        p.fixed_covariance = Some(std::sync::Arc::new(
            (0..14)
                .map(|t| FixedLagCovariance::new(&matrix * (1. + t as f64 / 14.)).unwrap())
                .collect(),
        ));
        p.validate().unwrap();
        let mut z = vec![0.1; 73];
        z[70] = 0.3f64.ln();
        z[71] = 0.5f64.ln();
        z[72] = 0.2;
        let (_, gradient) = p.evaluate(&z);
        for i in 0..z.len() {
            let old = z[i];
            z[i] = old + 1e-6;
            let plus = p.evaluate(&z).0;
            z[i] = old - 1e-6;
            let minus = p.evaluate(&z).0;
            z[i] = old;
            let numeric = (plus - minus) / 2e-6;
            assert!(
                (numeric - gradient[i]).abs() < 1e-5 * (1. + numeric.abs()),
                "index {i}: {numeric} vs {}",
                gradient[i]
            );
        }
        p.dc_notch = true;
        assert!(p.validate().is_err());
    }
    #[test]
    fn transport_momentum_and_inverse_agree() {
        let p = problem();
        let mut u = vec![0.01; 73];
        u[70] = 0.3f64.ln();
        u[71] = 0.5f64.ln();
        let mass = TransportMass::build(&p, &u);
        let mut rng = Rng(109);
        let mut total = 0.;
        for _ in 0..4000 {
            let q = mass.momentum(&mut rng);
            total += 2. * mass.kinetic(&q);
        }
        assert!((total / 4000. - 73.).abs() < 1.5, "{}", total / 4000.);
        let a: Vec<_> = (0..73).map(|i| (i as f64).sin()).collect();
        let b: Vec<_> = (0..73).map(|i| (i as f64).cos()).collect();
        let ab = a
            .iter()
            .zip(mass.solve(&b))
            .map(|(a, b)| a * b)
            .sum::<f64>();
        let ba = b
            .iter()
            .zip(mass.solve(&a))
            .map(|(a, b)| a * b)
            .sum::<f64>();
        assert!((ab - ba).abs() < 1e-8);
    }
    #[test]
    #[ignore = "sampling validation"]
    fn sampler_recovers_weakly_informed_hyperpriors() {
        let mut p = problem();
        p.receiver = 1e50;
        p.received_power = None;
        p.range_covariance = [None, None];
        p.dc_notch = false;
        p.long_mean.fill(f64::NAN);
        p.long_sigma.fill(f64::NAN);
        let mut rng = Rng(719);
        for i in 0..70 {
            let pairs = ((i % 5) / 2 + 1) as f64;
            let sd = 1e50 / (128. * pairs).sqrt();
            p.real[i] = rng.normal() * sd;
            p.imaginary[i] = rng.normal() * sd;
        }
        let mut z = vec![0.; 73];
        z[70] = 0.3f64.ln();
        z[71] = 0.5f64.ln();
        let c = sample(&p, &z, 113, 3000, 2048).unwrap();
        let sr = c.samples.iter().map(|s| s[70]).sum::<f64>() / c.samples.len() as f64;
        let st = c.samples.iter().map(|s| s[71]).sum::<f64>() / c.samples.len() as f64;
        eprintln!("log scale means {sr}, {st}; acceptance {}", c.acceptance);
        assert!((sr - 0.3f64.ln()).abs() < 0.6);
        assert!((st - 0.5f64.ln()).abs() < 0.6);
    }
    #[test]
    fn weak_likelihood_leaves_scale_hyperpriors_unchanged() {
        let mut p = problem();
        p.receiver = 1e50;
        p.received_power = None;
        p.long_mean.fill(f64::NAN);
        p.long_sigma.fill(f64::NAN);
        let mut u = vec![0.1; 73];
        u[70] = 0.3f64.ln();
        u[71] = 0.5f64.ln();
        u[72] = 0.;
        let (_, g) = noncentered(&p, &u);
        assert!(g[70].abs() < 1e-9, "spatial {}", g[70]);
        assert!(g[71].abs() < 1e-9, "temporal {}", g[71]);
        u[70] += 0.5;
        u[71] -= 0.8;
        let (_, g) = noncentered(&p, &u);
        assert!((g[70] - 0.5).abs() < 1e-9);
        assert!((g[71] + 0.8).abs() < 1e-9);
    }
    #[test]
    fn ar_precision_matches_known_stationary_covariance() {
        let ar = ArNoise::fit(&[1., 0.5], 1).unwrap();
        let c = nalgebra::DMatrix::from_fn(8, 8, |i, j| 0.5f64.powi(i.abs_diff(j) as i32));
        let x: Vec<_> = (0..8).map(|i| (i as f64).sin()).collect();
        let q = ar.precision(&x);
        let y = c * nalgebra::DVector::from_column_slice(&q);
        for i in 0..8 {
            assert!((y[i] - x[i]).abs() < 1e-12);
        }
    }
    #[test]
    fn noncentered_gradient_and_transforms_are_correct() {
        let p = problem();
        let mut u = vec![0.01; 73];
        u[70] = 0.3f64.ln();
        u[71] = 0.5f64.ln();
        let z = integrate(&u, 5, 14, 0.3, 0.5);
        let back = difference(&z, 5, 14, 0.3, 0.5);
        for i in 0..73 {
            assert!((u[i] - back[i]).abs() < 1e-12);
        }
        let (_, g) = noncentered(&p, &u);
        for i in 0..73 {
            let mut a = u.clone();
            let mut b = u.clone();
            a[i] += 1e-5;
            b[i] -= 1e-5;
            let fd = (noncentered(&p, &a).0 - noncentered(&p, &b).0) / 2e-5;
            assert!(
                (g[i] - fd).abs() < 1e-4 * (1. + fd.abs()),
                "{i}: {} vs {fd}",
                g[i]
            );
        }
        let v: Vec<_> = (0..73).map(|i| (i as f64).sin()).collect();
        let jtu = integrate_transpose(&v, 5, 14, 0.3, 0.5);
        let lhs = z.iter().zip(&v).map(|(a, b)| a * b).sum::<f64>();
        let rhs = u.iter().zip(&jtu).map(|(a, b)| a * b).sum::<f64>();
        assert!((lhs - rhs).abs() < 1e-10);
    }
    #[test]
    fn innovation_transform_has_unit_jacobian() {
        let (n, m) = (5, 14);
        let mut a = nalgebra::DMatrix::<f64>::zeros(n * m, n * m);
        a[(0, 0)] = 1.;
        for (row, (terms, _)) in prior_terms(n, m).iter().enumerate() {
            for &(col, v) in terms {
                a[(row + 1, col)] = v;
            }
        }
        assert!((a.determinant().abs() - 1.).abs() < 1e-12);
    }
    #[test]
    fn banded_mass_solve_agrees_with_dense_linear_algebra() {
        let p = problem();
        let mut z = vec![0.2; 73];
        z[70] = 0.3f64.ln();
        z[71] = 0.5f64.ln();
        let mass = Mass::build(&p, &z);
        let mut l = nalgebra::DMatrix::<f64>::zeros(73, 73);
        for i in 0usize..73 {
            for j in i.saturating_sub(15)..=i {
                l[(i, j)] = mass.l[i * mass.width + i - j];
            }
        }
        let h = &l * l.transpose();
        let input: Vec<_> = (0..73).map(|i| (i as f64).cos()).collect();
        let answer = mass.solve(&input);
        let permuted = nalgebra::DVector::from_iterator(73, (0..73).map(|i| answer[mass.index(i)]));
        let result = h * permuted;
        for i in 0..73 {
            assert!((result[i] - input[mass.index(i)]).abs() < 1e-9);
        }
    }
    #[test]
    fn analytic_gradient_matches_finite_differences() {
        let p = problem();
        let mut z = vec![0.2; 73];
        z[70] = 0.3f64.ln();
        z[71] = 0.5f64.ln();
        z[72] = 0.1;
        for i in 0..70 {
            z[i] += 0.03 * (i as f64).sin();
        }
        let (_, g) = p.evaluate(&z);
        for i in 0..73 {
            let mut a = z.clone();
            let mut b = z.clone();
            a[i] += 1e-5;
            b[i] -= 1e-5;
            let fd = (p.evaluate(&a).0 - p.evaluate(&b).0) / 2e-5;
            assert!(
                (fd - g[i]).abs() < 1e-5 * (1. + fd.abs()),
                "{i}: {fd} != {}",
                g[i]
            );
        }
    }
    #[test]
    fn fixed_noise_gradient_matches_finite_differences() {
        let mut p = problem();
        p.fixed_variance = Some((0..70).map(|i| 50. + i as f64).collect());
        let mut z = vec![0.2; 73];
        z[70] = 0.3f64.ln();
        z[71] = 0.5f64.ln();
        z[72] = 0.1;
        for i in 0..70 {
            z[i] += 0.03 * (i as f64).sin();
        }
        let (_, g) = p.evaluate(&z);
        for i in 0..73 {
            let mut a = z.clone();
            let mut b = z.clone();
            a[i] += 1e-5;
            b[i] -= 1e-5;
            let fd = (p.evaluate(&a).0 - p.evaluate(&b).0) / 2e-5;
            assert!(
                (fd - g[i]).abs() < 1e-5 * (1. + fd.abs()),
                "{i}: {fd} != {}",
                g[i]
            );
        }
    }
    #[test]
    fn variance_shape_changes_when_density_changes() {
        let p = problem();
        let mut z = vec![0.; 73];
        z[70] = 0.3f64.ln();
        z[71] = 0.5f64.ln();
        let a = p.evaluate(&z).0;
        z[3] += 1.;
        let b = p.evaluate(&z).0;
        assert!(a.is_finite() && b.is_finite() && (a - b).abs() > 1.);
    }
}
