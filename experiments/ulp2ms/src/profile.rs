//! Joint regularized plasma-profile inversion in signed measured ACF space.
//! Six fields are interpolated across knots; spatial quadrature cells retain
//! separate spectra, ion-line power per electron, and local geomagnetic inputs.
use crate::{
    profile_response::{Term, TIMES},
    theory::Theory,
};
use anyhow::{ensure, Result};
use nalgebra::{DMatrix as M, DVector as V};
use num_complex::Complex64 as C;
use rayon::prelude::*;
use serde::Serialize;
pub struct Cell {
    pub theory: Theory,
    pub interpolation: Vec<(usize, f64)>,
    pub terms: Vec<Term>,
}
#[derive(Clone)]
pub struct Datum {
    pub value: C,
    pub sigma: [f64; 2],
}
pub struct Problem {
    pub whitening: Option<crate::measurement_covariance::Whitening>,
    pub cells: Vec<Cell>,
    pub data: Vec<Datum>,
    pub knots: Vec<f64>,
    pub lags_per_gate: usize,
    pub background: Vec<C>,
    pub roughness: f64,
    pub density_anchor: Option<usize>,
    pub fixed_calibration: Option<f64>,
    pub fourier: Vec<Vec<C>>,
}
#[derive(Serialize)]
pub struct Fit {
    pub status: String,
    pub iterations: usize,
    pub data_chi_square: f64,
    pub penalty: f64,
    pub calibration_adu2: f64,
    pub background_adu2: Vec<f64>,
    pub objective_history: Vec<f64>,
    pub normalized_gradient: f64,
    pub active_bounds: Vec<usize>,
    pub transformed_profile: Vec<f64>,
    pub conditional_covariance: Option<Vec<f64>>,
    pub resolution_matrix: Option<Vec<f64>>,
    #[serde(skip)]
    pub predicted: Vec<C>,
}
pub struct Evaluation {
    pub residual: V<f64>,
    pub jacobian: M<f64>,
    pub prediction: Vec<C>,
    pub calibration: f64,
    pub background: Vec<f64>,
    pub data_chi_square: f64,
}
pub fn bound(field: usize, anchor: bool) -> (f64, f64) {
    match field {
        0 => (1e-4f64.ln(), if anchor { 10f64.ln() } else { 100f64.ln() }),
        1 => (0.3f64.ln(), 5f64.ln()),
        2 => (0.3f64.ln(), 12f64.ln()),
        3 => (-1.5, 1.5),
        _ => (0., 1.),
    }
}
/// Either a uniform spacing in km or explicit comma-separated slant-range knots.
pub fn knot_grid(spec: &str) -> Result<Vec<f64>> {
    let knots = if spec.contains(',') {
        spec.split(',')
            .map(str::parse::<f64>)
            .collect::<std::result::Result<Vec<_>, _>>()?
    } else {
        let step: f64 = spec.parse()?;
        ensure!(
            step >= 50. && step <= 300. && (1500. / step - (1500. / step).round()).abs() < 1e-8,
            "uniform knots must partition 200..1700 km with step 50..300 km"
        );
        (0..=(1500. / step).round() as usize)
            .map(|i| 200. + step * i as f64)
            .collect()
    };
    ensure!(
        knots.len() >= 3
            && knots.iter().all(|z| z.is_finite())
            && knots[0] == 200.
            && *knots.last().unwrap() == 1700.
            && knots.windows(2).all(|v| v[1] > v[0]),
        "knots must be finite, strictly increasing, and span 200..1700 km"
    );
    Ok(knots)
}
pub fn interpolation(knots: &[f64], z: f64) -> Result<Vec<(usize, f64)>> {
    ensure!(
        knots.len() > 1
            && knots.windows(2).all(|v| v[1] > v[0])
            && z >= knots[0]
            && z <= knots[knots.len() - 1],
        "quadrature outside profile knots"
    );
    let i = knots
        .partition_point(|&v| v <= z)
        .saturating_sub(1)
        .min(knots.len() - 2);
    let u = (z - knots[i]) / (knots[i + 1] - knots[i]);
    Ok(vec![(i, 1. - u), (i + 1, u)])
}
pub fn fourier(f: &[f64]) -> Vec<Vec<C>> {
    let df = f[1] - f[0];
    let n = f.len();
    (0..TIMES)
        .map(|t| {
            f.iter()
                .enumerate()
                .map(|(j, &freq)| {
                    C::from_polar(
                        df * if j == 0 || j + 1 == n { 0.5 } else { 1. },
                        -std::f64::consts::TAU * freq * (t as f64 - 40.) * 1e-5,
                    )
                })
                .collect()
        })
        .collect()
}
fn local_acf(theory: &Theory, p: &[f64; 6], basis: &[Vec<C>]) -> Vec<C> {
    let ne = 1e11 * p[0].exp();
    let (s, mass) = theory.spectrum_with_mass(
        1000. * p[1].exp(),
        1000. * p[2].exp(),
        1000. * p[3],
        p[4],
        p[5],
        ne,
    );
    basis
        .iter()
        .map(|row| {
            row.iter()
                .zip(&s)
                .map(|(a, &s)| *a * (s * mass * p[0].exp()))
                .sum()
        })
        .collect()
}
impl Problem {
    pub fn validate(&self) -> Result<()> {
        ensure!(
            self.knots.len() > 2
                && self.data.len() % self.lags_per_gate == 0
                && self.background.len() == self.data.len(),
            "profile dimensions"
        );
        ensure!(
            self.roughness.is_finite()
                && self.roughness >= 0.
                && self.data.iter().all(|d| d.value.re.is_finite()
                    && d.value.im.is_finite()
                    && d.sigma.iter().all(|s| s.is_finite() && *s > 0.)),
            "invalid profile observations"
        );
        ensure!(
            self.fixed_calibration
                .map(|v| v.is_finite() && v > 0.)
                .unwrap_or(true),
            "invalid calibration"
        );
        ensure!(
            self.fixed_calibration.is_some() || self.density_anchor.is_some(),
            "free calibration needs a density anchor"
        );
        ensure!(
            self.density_anchor
                .map(|i| i < self.knots.len())
                .unwrap_or(true),
            "invalid density anchor"
        );
        ensure!(
            self.cells.iter().all(|c| c
                .interpolation
                .iter()
                .all(|&(k, w)| k < self.knots.len() && w >= 0.)
                && (c.interpolation.iter().map(|v| v.1).sum::<f64>() - 1.).abs() < 1e-10
                && c.terms
                    .iter()
                    .all(|t| t.measurement < self.data.len() && t.time < TIMES)),
            "invalid response cells"
        );
        Ok(())
    }
    pub fn evaluate(&self, x: &V<f64>, derivatives: bool) -> Result<Evaluation> {
        ensure!(
            x.len() == 6 * self.knots.len() && x.iter().all(|v| v.is_finite()),
            "profile parameter dimensions"
        );
        let np = x.len();
        let nm = self.data.len();
        let ng = nm / self.lags_per_gate;
        let pieces = self
            .cells
            .par_iter()
            .map(|cell| {
                let mut p = [0.; 6];
                for &(k, w) in &cell.interpolation {
                    for j in 0..6 {
                        p[j] += w * x[6 * k + j];
                    }
                }
                let value = local_acf(&cell.theory, &p, &self.fourier);
                let mut deriv = vec![];
                if derivatives {
                    for j in 0..6 {
                        let eps = 1e-4;
                        let (lo, hi) = bound(j, self.density_anchor.is_some());
                        let mut a = p;
                        let mut b = p;
                        a[j] = (p[j] + eps).min(hi);
                        b[j] = (p[j] - eps).max(lo);
                        let sa = local_acf(&cell.theory, &a, &self.fourier);
                        let sb = local_acf(&cell.theory, &b, &self.fourier);
                        deriv.push(
                            sa.iter()
                                .zip(sb)
                                .map(|(a, b)| (*a - b) / (a_span(p[j], eps, lo, hi)))
                                .collect::<Vec<_>>(),
                        );
                    }
                }
                (value, deriv)
            })
            .collect::<Vec<_>>();
        let mut model = vec![C::default(); nm];
        let mut dm = M::<f64>::zeros(2 * nm, np);
        for (cell, (value, deriv)) in self.cells.iter().zip(pieces) {
            for t in &cell.terms {
                model[t.measurement] += t.weight * value[t.time];
                if derivatives {
                    for &(k, w) in &cell.interpolation {
                        for j in 0..6 {
                            let d = t.weight * deriv[j][t.time] * w;
                            dm[(2 * t.measurement, 6 * k + j)] += d.re;
                            dm[(2 * t.measurement + 1, 6 * k + j)] += d.im;
                        }
                    }
                }
            }
        }
        // Eliminate per-range background amplitudes, then one common calibration.
        if let Some(w) = &self.whitening {
            let n = 2 * nm;
            let mut a = M::zeros(n, np + 2);
            let mut b = M::zeros(n, ng);
            for i in 0..nm {
                for c in 0..2 {
                    let row = 2 * i + c;
                    let sigma = self.data[i].sigma[c];
                    a[(row, 0)] = [model[i].re, model[i].im][c] / sigma;
                    a[(row, 1)] = [self.data[i].value.re, self.data[i].value.im][c] / sigma;
                    b[(row, i / self.lags_per_gate)] =
                        [self.background[i].re, self.background[i].im][c] / sigma;
                    if derivatives {
                        for j in 0..np {
                            a[(row, j + 2)] = dm[(row, j)] / sigma;
                        }
                    }
                }
            }
            let a = w.apply(&a);
            let b = w.apply(&b);
            let mut normal = b.transpose() * &b;
            for i in 0..ng {
                normal[(i, i)] += 1e-18;
            }
            let ch = normal
                .cholesky()
                .ok_or_else(|| anyhow::anyhow!("singular correlated nuisance model"))?;
            let beta = ch.solve(&(b.transpose() * &a));
            let projected = &a - &b * &beta;
            let m = projected.column(0);
            let y = projected.column(1);
            let den = m.norm_squared();
            ensure!(den > 0., "singular correlated calibration");
            let calibration = self.fixed_calibration.unwrap_or((m.dot(&y) / den).max(0.));
            let bg: Vec<f64> = (0..ng)
                .map(|g| beta[(g, 1)] - calibration * beta[(g, 0)])
                .collect();
            let mut residual = V::zeros(n + (self.knots.len() - 2) * 6);
            let mut jac = M::zeros(residual.len(), np);
            residual.rows_mut(0, n).copy_from(&(m * calibration - y));
            if derivatives {
                for j in 0..np {
                    let d = projected.column(j + 2);
                    let da = if self.fixed_calibration.is_none() && calibration > 0. {
                        (d.dot(&y) - 2. * calibration * m.dot(&d)) / den
                    } else {
                        0.
                    };
                    jac.column_mut(j)
                        .rows_mut(0, n)
                        .copy_from(&(d * calibration + m * da));
                }
            }
            let data_chi_square = residual.rows(0, n).norm_squared();
            self.add_roughness(x, &mut residual, &mut jac, derivatives);
            let prediction = (0..nm)
                .map(|i| calibration * model[i] + bg[i / self.lags_per_gate] * self.background[i])
                .collect();
            return Ok(Evaluation {
                residual,
                jacobian: jac,
                prediction,
                calibration,
                background: bg,
                data_chi_square,
            });
        }
        // Exact derivative of the variable-projection solution is retained.
        let mut bb = vec![1e-18; ng];
        let mut ab = vec![0.; ng];
        let mut by = vec![0.; ng];
        let mut aa = 0.;
        let mut ay = 0.;
        let mut daa = V::<f64>::zeros(np);
        let mut day = V::<f64>::zeros(np);
        let mut dab = M::<f64>::zeros(ng, np);
        for i in 0..nm {
            let g = i / self.lags_per_gate;
            for c in 0..2 {
                let b = [self.background[i].re, self.background[i].im][c];
                let m = [model[i].re, model[i].im][c];
                let y = [self.data[i].value.re, self.data[i].value.im][c];
                let w = self.data[i].sigma[c].powi(-2);
                bb[g] += w * b * b;
                ab[g] += w * m * b;
                by[g] += w * b * y;
                aa += w * m * m;
                ay += w * m * y;
                if derivatives {
                    for k in 0..np {
                        let d = dm[(2 * i + c, k)];
                        daa[k] += 2. * w * m * d;
                        day[k] += w * y * d;
                        dab[(g, k)] += w * b * d;
                    }
                }
            }
        }
        let den = aa - (0..ng).map(|g| ab[g] * ab[g] / bb[g]).sum::<f64>();
        let num = ay - (0..ng).map(|g| ab[g] * by[g] / bb[g]).sum::<f64>();
        ensure!(den.is_finite() && den > 0., "singular profile calibration");
        let calibration = self.fixed_calibration.unwrap_or((num / den).max(0.));
        let mut da = V::<f64>::zeros(np);
        if derivatives && self.fixed_calibration.is_none() && calibration > 0. {
            for k in 0..np {
                let dd = daa[k] - 2. * (0..ng).map(|g| ab[g] * dab[(g, k)] / bb[g]).sum::<f64>();
                let dn = day[k] - (0..ng).map(|g| by[g] * dab[(g, k)] / bb[g]).sum::<f64>();
                da[k] = (dn - calibration * dd) / den;
            }
        }
        let bg = (0..ng)
            .map(|g| (by[g] - calibration * ab[g]) / bb[g])
            .collect::<Vec<_>>();
        let npen = (self.knots.len() - 2) * 6;
        let mut residual = V::zeros(2 * nm + npen);
        let mut jac = M::zeros(residual.len(), np);
        let mut prediction = vec![C::default(); nm];
        for i in 0..nm {
            let g = i / self.lags_per_gate;
            prediction[i] = calibration * model[i] + bg[g] * self.background[i];
            for c in 0..2 {
                let b = [self.background[i].re, self.background[i].im][c];
                let m = [model[i].re, model[i].im][c];
                let s = self.data[i].sigma[c];
                residual[2 * i + c] = ([prediction[i].re, prediction[i].im][c]
                    - [self.data[i].value.re, self.data[i].value.im][c])
                    / s;
                if derivatives {
                    for k in 0..np {
                        let db = (-da[k] * ab[g] - calibration * dab[(g, k)]) / bb[g];
                        jac[(2 * i + c, k)] =
                            (calibration * dm[(2 * i + c, k)] + da[k] * m + db * b) / s;
                    }
                }
            }
        }
        let data_chi_square = residual.rows(0, 2 * nm).norm_squared();
        self.add_roughness(x, &mut residual, &mut jac, derivatives);
        Ok(Evaluation {
            residual,
            jacobian: jac,
            prediction,
            calibration,
            background: bg,
            data_chi_square,
        })
    }
    fn add_roughness(
        &self,
        x: &V<f64>,
        residual: &mut V<f64>,
        jac: &mut M<f64>,
        derivatives: bool,
    ) {
        let nm = self.data.len();
        // Dimensionless second derivatives scaled to 100-km knot spacing. These
        // explicit smoothing priors are not data, and are reported separately.
        let scales = [0.5, 0.25, 0.3, 0.15, 0.15, 0.1];
        for k in 1..self.knots.len() - 1 {
            let dl = self.knots[k] - self.knots[k - 1];
            let dr = self.knots[k + 1] - self.knots[k];
            let co = [
                2e4 / (dl * (dl + dr)),
                -2e4 / (dl * dr),
                2e4 / (dr * (dl + dr)),
            ];
            for f in 0..6 {
                let row = 2 * nm + 6 * (k - 1) + f;
                let fac = self.roughness.sqrt() / scales[f];
                for q in 0..3 {
                    residual[row] += fac * co[q] * x[6 * (k + q - 1) + f];
                    if derivatives {
                        jac[(row, 6 * (k + q - 1) + f)] = fac * co[q];
                    }
                }
            }
        }
    }
    pub fn solve(&self, initial: &[f64], max_iterations: usize) -> Result<Fit> {
        self.validate()?;
        let mut x = V::from_column_slice(initial);
        for i in 0..x.len() {
            let (lo, hi) = bound(i % 6, self.density_anchor.is_some());
            x[i] = x[i].clamp(lo, hi);
        }
        if let Some(k) = self.density_anchor {
            x[6 * k] = 10f64.ln();
        }
        let mut e = self.evaluate(&x, true)?;
        let mut cost = e.residual.norm_squared();
        let mut hist = vec![cost];
        let mut damping = 1e-3;
        let mut status = "iteration_limit";
        let mut iterations = 0;
        for it in 0..max_iterations {
            iterations = it + 1;
            let mut h = e.jacobian.transpose() * &e.jacobian;
            let mut g = e.jacobian.transpose() * &e.residual;
            let blocked = (0..x.len())
                .map(|i| {
                    let (lo, hi) = bound(i % 6, self.density_anchor.is_some());
                    Some(i) == self.density_anchor.map(|k| 6 * k)
                        || h[(i, i)] < 1e-20
                        || (x[i] <= lo + 1e-8 && g[i] > 0.)
                        || (x[i] >= hi - 1e-8 && g[i] < 0.)
                })
                .collect::<Vec<_>>();
            let gradient = (0..x.len())
                .filter(|&i| !blocked[i])
                .map(|i| g[i].abs() / h[(i, i)].sqrt().max(1e-12))
                .fold(0., f64::max);
            eprintln!("profile iteration {it}: objective={cost:.4} data={:.4} gradient={gradient:.5} calibration={:.4}",e.data_chi_square,e.calibration);
            if gradient < 1e-3 {
                status = "converged";
                break;
            }
            for i in 0..x.len() {
                h[(i, i)] += damping * h[(i, i)].max(1e-10);
                if blocked[i] {
                    for j in 0..x.len() {
                        h[(i, j)] = 0.;
                        h[(j, i)] = 0.;
                    }
                    h[(i, i)] = 1.;
                    g[i] = 0.;
                }
            }
            let Some(ch) = h.cholesky() else {
                damping *= 10.;
                continue;
            };
            let mut step = ch.solve(&(-g));
            if step.amax() > 0.5 {
                step *= 0.5 / step.amax();
            }
            let mut candidate = &x + step;
            for i in 0..x.len() {
                let (lo, hi) = bound(i % 6, self.density_anchor.is_some());
                candidate[i] = candidate[i].clamp(lo, hi);
            }
            if let Some(k) = self.density_anchor {
                candidate[6 * k] = 10f64.ln();
            }
            let next = self.evaluate(&candidate, false)?;
            let nc = next.residual.norm_squared();
            if nc < cost {
                x = candidate;
                cost = nc;
                hist.push(cost);
                e = self.evaluate(&x, true)?;
                damping = (damping / 3.).max(1e-10);
            } else {
                damping *= 8.;
                if damping > 1e15 {
                    status = "stalled";
                    break;
                }
            }
        }
        // Recompute final gradient, including an accepted last step.
        let h = e.jacobian.transpose() * &e.jacobian;
        let g = e.jacobian.transpose() * &e.residual;
        let active = (0..x.len())
            .filter(|&i| {
                let (lo, hi) = bound(i % 6, self.density_anchor.is_some());
                x[i] <= lo + 1e-6
                    || x[i] >= hi - 1e-6
                    || Some(i) == self.density_anchor.map(|k| 6 * k)
            })
            .collect::<Vec<_>>();
        let free = (0..x.len())
            .filter(|i| !active.contains(i) && h[(*i, *i)] > 1e-20)
            .collect::<Vec<_>>();
        let gradient = (0..x.len())
            .filter(|&i| {
                let (lo, hi) = bound(i % 6, self.density_anchor.is_some());
                Some(i) != self.density_anchor.map(|k| 6 * k)
                    && h[(i, i)] > 1e-20
                    && !(x[i] <= lo + 1e-8 && g[i] > 0.)
                    && !(x[i] >= hi - 1e-8 && g[i] < 0.)
            })
            .map(|i| g[i].abs() / h[(i, i)].sqrt())
            .fold(0., f64::max);
        if gradient < 1e-3 {
            status = "converged";
        }
        let sub = M::from_fn(free.len(), free.len(), |i, j| h[(free[i], free[j])]);
        let dc = e.jacobian.rows(0, 2 * self.data.len());
        let dh = dc.transpose() * dc;
        let (cov, resolution) = if let Some(ch) = sub.cholesky() {
            let inv = ch.inverse();
            let data = M::from_fn(free.len(), free.len(), |i, j| dh[(free[i], free[j])]);
            let res = &inv * data;
            let mut full = M::from_element(x.len(), x.len(), f64::NAN);
            let mut rr = full.clone();
            for (i, &a) in free.iter().enumerate() {
                for (j, &b) in free.iter().enumerate() {
                    full[(a, b)] = inv[(i, j)];
                    rr[(a, b)] = res[(i, j)];
                }
            }
            (Some(full.as_slice().to_vec()), Some(rr.as_slice().to_vec()))
        } else {
            (None, None)
        };
        Ok(Fit {
            status: status.into(),
            iterations,
            data_chi_square: e.data_chi_square,
            penalty: cost - e.data_chi_square,
            calibration_adu2: e.calibration,
            background_adu2: e.background,
            objective_history: hist,
            normalized_gradient: gradient,
            active_bounds: active,
            transformed_profile: x.as_slice().to_vec(),
            conditional_covariance: cov,
            resolution_matrix: resolution,
            predicted: e.prediction,
        })
    }
}
fn a_span(p: f64, e: f64, lo: f64, hi: f64) -> f64 {
    (p + e).min(hi) - (p - e).max(lo)
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::profile_response::Ambiguity;
    #[test]
    fn nonuniform_grid_reproduces_affine_fields_and_convex_bounds() {
        let z = knot_grid("200,300,400,500,600,700,800,1000,1250,1500,1700").unwrap();
        assert_eq!(z.len(), 11);
        for r in [200., 550., 900., 1125., 1650., 1700.] {
            let w = interpolation(&z, r).unwrap();
            let sum: f64 = w.iter().map(|v| v.1).sum();
            let affine: f64 = w.iter().map(|&(i, w)| w * (2. + 0.003 * z[i])).sum();
            assert!((sum - 1.).abs() < 1e-14);
            assert!((affine - (2. + 0.003 * r)).abs() < 1e-14);
            assert!(w.iter().all(|v| v.1 >= 0. && v.1 <= 1.));
        }
        for bad in [
            "200,800,700,1700",
            "200,800,800,1700",
            "200,NaN,1700",
            "300,800,1700",
            "200,800,1600",
        ] {
            assert!(knot_grid(bad).is_err());
        }
        assert_eq!(knot_grid("100").unwrap().len(), 16);
    }
    fn fixture(free_calibration: bool) -> (Problem, Vec<f64>) {
        let knots = vec![400., 800., 1200.];
        let edges = (0..=16).map(|j| 400. + 50. * j as f64).collect::<Vec<_>>();
        let ranges = (0..=6).map(|j| 650. + 50. * j as f64).collect::<Vec<_>>();
        let lags = (1..61).collect::<Vec<_>>();
        let fir = [
            (-1e-5, C::new(0.15, 0.02)),
            (0., C::new(0.95, 0.)),
            (1e-5, C::new(0.15, -0.02)),
        ];
        let terms = Ambiguity::new(&fir)
            .unwrap()
            .terms(&ranges, &edges, &lags, true)
            .unwrap();
        let cells = terms
            .into_par_iter()
            .enumerate()
            .map(|(i, terms)| Cell {
                theory: Theory::at_field(30000., 25., 500.).unwrap(),
                interpolation: interpolation(&knots, 0.5 * (edges[i] + edges[i + 1])).unwrap(),
                terms,
            })
            .collect::<Vec<_>>();
        let f = fourier(&cells[0].theory.f);
        let background = (0..ranges.len())
            .flat_map(|_| {
                lags.iter().map(|&l| {
                    C::new(
                        0.03 * (-(l as f64) / 8.).exp() * (l as f64).cos(),
                        0.002 * (-(l as f64) / 8.).exp(),
                    )
                })
            })
            .collect::<Vec<_>>();
        let mut p = Problem {
            whitening: None,
            cells,
            data: vec![
                Datum {
                    value: C::new(1., 0.),
                    sigma: [1., 1.]
                };
                ranges.len() * 60
            ],
            knots,
            lags_per_gate: 60,
            background,
            roughness: 0.,
            density_anchor: free_calibration.then_some(1),
            fixed_calibration: Some(1300.),
            fourier: f,
        };
        let x = vec![
            3f64.ln(),
            0.9f64.ln(),
            1.4f64.ln(),
            0.03,
            0.08,
            0.06,
            10f64.ln(),
            1.7f64.ln(),
            2.4f64.ln(),
            0.05,
            0.4,
            0.1,
            0.3f64.ln(),
            2.5f64.ln(),
            4f64.ln(),
            0.08,
            0.75,
            0.12,
        ];
        // Setting y=0 makes profiled background cancel a component of the signal.
        // Recover the unprojected model through an evaluation with zero basis.
        let old = p.background.clone();
        p.background.fill(C::default());
        let prediction = p
            .evaluate(&V::from_column_slice(&x), false)
            .unwrap()
            .prediction;
        p.background = old;
        for (i, z) in prediction.into_iter().enumerate() {
            p.data[i].value = z + p.background[i] * (i / 60) as f64 * 5.;
        }
        if free_calibration {
            p.fixed_calibration = None;
        }
        (p, x)
    }
    #[test]
    fn profiled_calibration_and_range_background_jacobian_matches_differences() {
        let (mut p, x) = fixture(true);
        p.roughness = 0.7;
        let x = V::from_column_slice(&x);
        let e = p.evaluate(&x, true).unwrap();
        assert!((e.calibration - 1300.).abs() < 1e-8);
        for k in 0..x.len() {
            let mut a = x.clone();
            let mut b = x.clone();
            let eps = 2e-5;
            a[k] += eps;
            b[k] -= eps;
            let d = (p.evaluate(&a, false).unwrap().residual
                - p.evaluate(&b, false).unwrap().residual)
                / (2. * eps);
            let err = (d - e.jacobian.column(k)).norm() / e.jacobian.column(k).norm().max(1e-8);
            assert!(err < 1e-3, "coordinate {k}, relative Jacobian error {err}");
        }
    }
    #[test]
    fn identity_covariance_reproduces_diagonal_projection() {
        for free in [false, true] {
            let (mut p, x) = fixture(free);
            let x = V::from_column_slice(&x);
            let old = p.evaluate(&x, true).unwrap();
            p.whitening = Some(
                crate::measurement_covariance::Whitening::new(&M::zeros(2 * p.data.len(), 1), 1.)
                    .unwrap(),
            );
            let new = p.evaluate(&x, true).unwrap();
            assert!((new.calibration - old.calibration).abs() < 1e-7);
            assert!((&new.residual - &old.residual).norm() < 1e-7);
            assert!((&new.jacobian - &old.jacobian).norm() / old.jacobian.norm() < 1e-7);
        }
    }
    #[test]
    fn correlated_projection_and_jacobian_match_finite_differences() {
        let (mut p, x) = fixture(true);
        let n = 2 * p.data.len();
        let u = M::from_fn(n, 8, |i, j| ((i * 3 + j * 7) as f64).sin() / 3.);
        p.whitening = Some(crate::measurement_covariance::Whitening::new(&u, 0.15).unwrap());
        p.roughness = 0.7;
        let x = V::from_column_slice(&x);
        let e = p.evaluate(&x, true).unwrap();
        assert!((e.calibration - 1300.).abs() < 1e-7);
        for k in 0..x.len() {
            let mut a = x.clone();
            let mut b = x.clone();
            let eps = 2e-5;
            a[k] += eps;
            b[k] -= eps;
            let d = (p.evaluate(&a, false).unwrap().residual
                - p.evaluate(&b, false).unwrap().residual)
                / (2. * eps);
            let err = (d - e.jacobian.column(k)).norm() / e.jacobian.column(k).norm().max(1e-8);
            assert!(err < 1e-3, "correlated Jacobian coordinate {k}: {err}");
        }
    }
    #[test]
    fn recover_stratified_density_temperature_and_composition() {
        let (p, x) = fixture(false);
        let mut initial = x.clone();
        for k in 0..3 {
            initial[6 * k] += 0.08;
            initial[6 * k + 1] -= 0.08;
            initial[6 * k + 2] += 0.08;
            initial[6 * k + 4] += 0.03;
            initial[6 * k + 5] -= 0.02;
        }
        let fit = p.solve(&initial, 80).unwrap();
        assert!(
            fit.data_chi_square < 1e-5,
            "{}: {}",
            fit.status,
            fit.data_chi_square
        );
        for (i, (a, b)) in fit.transformed_profile.iter().zip(x).enumerate() {
            assert!((a - b).abs() < 0.01, "coordinate {i}: {a} versus {b}");
        }
    }
}
