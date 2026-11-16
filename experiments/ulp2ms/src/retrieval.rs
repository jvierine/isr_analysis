// Reused bounded variable-projection fitter from the prior SYISR H+ study.
use crate::theory::Theory;
use nalgebra::{DMatrix as M, DVector as V};
use num_complex::Complex64 as C;
use serde::{Deserialize, Serialize};
#[derive(Clone)]
pub struct Observation {
    pub value: Vec<C>,
    pub sigma: Vec<[f64; 2]>,
    pub kernel: Vec<Vec<C>>,
    pub rho: Vec<C>,
    pub background_sigma: f64,
    pub lag_min: usize,
}
#[derive(Clone, Serialize, Deserialize)]
pub struct Fit {
    pub ti_k: f64,
    pub te_k: f64,
    pub spectral_velocity_m_s: f64,
    pub h_fraction: f64,
    pub he_fraction: f64,
    pub o_fraction: f64,
    pub signal_adu2: f64,
    pub background_adjustment_adu2: f64,
    pub chi_square: f64,
    pub dof: usize,
    pub iterations: usize,
    pub gradient: f64,
    pub status: String,
    pub predicted: Vec<[f64; 2]>,
}
fn bounds(i: usize) -> (f64, f64) {
    match i {
        0 => (0.3_f64.ln(), 5_f64.ln()),
        1 => (0.3_f64.ln(), 12_f64.ln()),
        2 => (-1.5, 1.5),
        _ => (0., 1.),
    }
}
fn encode(f: &Fit) -> V<f64> {
    V::from_vec(vec![
        (f.ti_k / 1000.).ln(),
        (f.te_k / 1000.).ln(),
        f.spectral_velocity_m_s / 1000.,
        f.h_fraction,
        if f.h_fraction < 1. {
            f.he_fraction / (1. - f.h_fraction)
        } else {
            0.
        },
    ])
}
impl Observation {
    /// Local curvature after profiling signal/background amplitudes. Conditional
    /// on the stated diagonal weights and response; not a systematic error budget.
    pub fn local_errors(&self, theory: &Theory, fit: &Fit, ne: f64) -> Option<[f64; 4]> {
        let x = encode(fit);
        let residual = self.evaluate(theory, &x, ne).0;
        let mut j = M::zeros(residual.len(), 5);
        for i in 0..5 {
            let step = if i == 2 { 0.0005 } else { 0.0002 };
            let (lo, hi) = bounds(i);
            let mut p = x.clone();
            let mut m = x.clone();
            p[i] = (p[i] + step).min(hi);
            m[i] = (m[i] - step).max(lo);
            j.set_column(
                i,
                &((self.evaluate(theory, &p, ne).0 - self.evaluate(theory, &m, ne).0)
                    / (p[i] - m[i])),
            );
        }
        let cov = (j.transpose() * j).try_inverse()?;
        let q = x[4];
        let h = x[3];
        let hevar =
            q * q * cov[(3, 3)] + (1. - h).powi(2) * cov[(4, 4)] - 2. * q * (1. - h) * cov[(3, 4)];
        Some([
            cov[(3, 3)].max(0.).sqrt(),
            hevar.max(0.).sqrt(),
            fit.ti_k * cov[(0, 0)].max(0.).sqrt(),
            fit.te_k * cov[(1, 1)].max(0.).sqrt(),
        ])
    }
    fn evaluate(&self, theory: &Theory, x: &V<f64>, ne: f64) -> (V<f64>, f64, f64, Vec<C>) {
        let s = theory.spectrum(
            1000. * x[0].exp(),
            1000. * x[1].exp(),
            1000. * x[2],
            x[3],
            x[4],
            ne,
        );
        let model: Vec<C> = self
            .kernel
            .iter()
            .map(|k| k.iter().zip(&s).map(|(k, s)| k * s).sum())
            .collect();
        let (mut aa, mut ab, mut bb, mut ay, mut by) =
            (0., 0., self.background_sigma.powi(-2), 0., 0.);
        for l in 0..self.value.len() {
            for q in 0..2 {
                if l == 0 && q == 1 && self.lag_min == 0 {
                    continue;
                }
                let m = [model[l].re, model[l].im][q];
                let b = [self.rho[l].re, self.rho[l].im][q];
                let y = [self.value[l].re, self.value[l].im][q];
                let w = self.sigma[l][q].powi(-2);
                aa += w * m * m;
                ab += w * m * b;
                bb += w * b * b;
                ay += w * m * y;
                by += w * b * y;
            }
        }
        let det = aa * bb - ab * ab;
        let a = if det > 0. {
            ((ay * bb - by * ab) / det).max(0.)
        } else {
            0.
        };
        let b = (by - ab * a) / bb;
        let mut r = vec![];
        let predicted: Vec<_> = model
            .iter()
            .zip(&self.rho)
            .map(|(m, r)| a * m + b * r)
            .collect();
        for l in 0..self.value.len() {
            r.push((predicted[l].re - self.value[l].re) / self.sigma[l][0]);
            if l > 0 || self.lag_min > 0 {
                r.push((predicted[l].im - self.value[l].im) / self.sigma[l][1]);
            }
        }
        r.push(b / self.background_sigma);
        (V::from_vec(r), a, b, predicted)
    }
    pub fn fit(
        &self,
        theory: &Theory,
        initial: &Fit,
        ne: f64,
        fixed_h: Option<f64>,
        max_iterations: usize,
    ) -> Fit {
        let mut x = encode(initial);
        if let Some(h) = fixed_h {
            x[3] = h;
        }
        let mut lambda = 0.001;
        let mut iters = 0;
        let mut status = "not_converged";
        let mut gradient = f64::INFINITY;
        let mut residual = self.evaluate(theory, &x, ne).0;
        let mut cost = residual.norm_squared();
        for it in 0..max_iterations {
            iters = it + 1;
            let mut j = M::zeros(residual.len(), 5);
            for f in 0..5 {
                if f == 3 && fixed_h.is_some() {
                    continue;
                }
                let step = if f == 2 { 0.0005 } else { 0.001 };
                let (lo, hi) = bounds(f);
                let mut p = x.clone();
                let mut m = x.clone();
                p[f] = (p[f] + step).min(hi);
                m[f] = (m[f] - step).max(lo);
                let dr = (self.evaluate(theory, &p, ne).0 - self.evaluate(theory, &m, ne).0)
                    / (p[f] - m[f]);
                j.set_column(f, &dr);
            }
            let mut h = j.transpose() * &j;
            let mut g = j.transpose() * &residual;
            let blocked: Vec<_> = (0..5)
                .map(|i| {
                    let (lo, hi) = bounds(i);
                    (i == 3 && fixed_h.is_some())
                        || h[(i, i)] < 1e-18
                        || (x[i] <= lo + 1e-8 && g[i] > 0.)
                        || (x[i] >= hi - 1e-8 && g[i] < 0.)
                })
                .collect();
            gradient = (0..5)
                .filter(|&i| !blocked[i])
                .map(|i| g[i].abs() / h[(i, i)].sqrt())
                .fold(0., f64::max);
            if gradient < 0.001 {
                status = "converged";
                break;
            }
            for i in 0..5 {
                h[(i, i)] += lambda * h[(i, i)].max(1e-12);
                if blocked[i] {
                    for k in 0..5 {
                        h[(i, k)] = 0.;
                        h[(k, i)] = 0.;
                    }
                    h[(i, i)] = 1.;
                    g[i] = 0.;
                }
            }
            let Some(ch) = h.cholesky() else {
                lambda *= 10.;
                continue;
            };
            let mut step = ch.solve(&(-g));
            if step.amax() > 0.5 {
                step *= 0.5 / step.amax();
            }
            let mut next = &x + step;
            for i in 0..5 {
                let (lo, hi) = bounds(i);
                next[i] = next[i].clamp(lo, hi);
            }
            if let Some(h) = fixed_h {
                next[3] = h;
            }
            let nr = self.evaluate(theory, &next, ne).0;
            let nc = nr.norm_squared();
            if nc < cost {
                let dx = (&x - &next).amax();
                let dc = cost - nc;
                x = next;
                residual = nr;
                cost = nc;
                lambda = (lambda / 3.).max(1e-10);
                if dx < 1e-5 && dc < 1e-9 * cost.max(1.) {
                    status = "converged_small_step";
                    break;
                }
            } else {
                lambda *= 8.;
                if lambda > 1e15 {
                    status = "stalled";
                    break;
                }
            }
        }
        let (_, a, b, p) = self.evaluate(theory, &x, ne);
        let he = (1. - x[3]) * x[4];
        Fit {
            ti_k: 1000. * x[0].exp(),
            te_k: 1000. * x[1].exp(),
            spectral_velocity_m_s: 1000. * x[2],
            h_fraction: x[3],
            he_fraction: he,
            o_fraction: 1. - x[3] - he,
            signal_adu2: a,
            background_adjustment_adu2: b,
            chi_square: cost,
            dof: residual
                .len()
                .saturating_sub(if fixed_h.is_some() { 6 } else { 7 }),
            iterations: iters,
            gradient,
            status: status.into(),
            predicted: p.iter().map(|p| [p.re, p.im]).collect(),
        }
    }
    pub fn solve(&self, theory: &Theory, ne: f64) -> (Fit, Vec<Fit>) {
        let mut best = initial(1000., 1500., 0.1);
        best.chi_square = f64::INFINITY;
        for ti in [600., 1200., 2200.] {
            for h in [0.02, 0.3, 0.8] {
                let fit = self.fit(theory, &initial(ti, ti * 1.5, h), ne, None, 65);
                if fit.chi_square < best.chi_square {
                    best = fit;
                }
            }
        }
        let mut profiles = vec![];
        for h in [
            0., 0.02, 0.05, 0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 1.,
        ] {
            let mut b = self.fit(theory, &best, ne, Some(h), 70);
            if h == 0. {
                for ti in [600., 1200., 2200.] {
                    let f = self.fit(theory, &initial(ti, ti * 1.5, 0.), ne, Some(0.), 70);
                    if f.chi_square < b.chi_square {
                        b = f;
                    }
                }
            }
            profiles.push(b);
        }
        (best, profiles)
    }
}
pub fn initial(ti: f64, te: f64, h: f64) -> Fit {
    Fit {
        ti_k: ti,
        te_k: te,
        spectral_velocity_m_s: 0.,
        h_fraction: h,
        he_fraction: 0.05 * (1. - h),
        o_fraction: 0.95 * (1. - h),
        signal_adu2: 1.,
        background_adjustment_adu2: 0.,
        chi_square: 0.,
        dof: 0,
        iterations: 0,
        gradient: 0.,
        status: String::new(),
        predicted: vec![],
    }
}
