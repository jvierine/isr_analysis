//! Range-coupled H+/He+/O+ profile fitting in measured complex-product space.
//! Amplitude is uncalibrated; density controls Debye shape only. No ACF inversion.
use crate::theory::Theory;
use anyhow::{ensure, Result};
use nalgebra::{DMatrix as M, DVector as V};
use num_complex::Complex64 as C;
use rayon::prelude::*;
use serde::{Deserialize, Serialize};
use std::sync::Arc;

#[derive(Clone, Serialize, Deserialize)]
pub struct Product {
    pub value: [f64; 2],
    pub sigma: [f64; 2],
}
#[derive(Clone)]
pub struct Term {
    pub measurement: usize,
    pub kernel: Vec<C>,
}
pub struct Cell {
    pub theory: Arc<Theory>,
    /// Optional prescribed Debye-shape density for this physical range cell.
    pub shape_density_m3: Option<f64>,
    /// When set, control amplitude is Ne/1e11; tuple is empirical gain, range km, H taper.
    pub density_calibration: Option<(f64, f64, f64)>,
    pub knot_weights: Vec<(usize, f64)>,
    /// Optional independent H-only spline basis; all other fields use knot_weights.
    pub h_knot_weights: Option<Vec<(usize, f64)>>,
    pub terms: Vec<Term>,
}
pub struct Problem {
    pub frequency_hz: Vec<f64>,
    pub radar_frequency_hz: f64,
    pub shape_density_m3: f64,
    pub cells: Vec<Cell>,
    pub products: Vec<Product>,
    pub knot_count: usize,
    pub background_basis: Vec<C>,
    pub background_sigma: f64,
}
#[derive(Serialize, Deserialize, Clone, Debug)]
pub struct Node {
    pub power_adu2: f64,
    pub ti_k: f64,
    pub te_k: f64,
    pub velocity_m_s: f64,
    pub h_fraction: f64,
    pub he_fraction: f64,
    pub o_fraction: f64,
}
#[derive(Serialize, Deserialize)]
pub struct Fit {
    pub status: String,
    pub iterations: usize,
    pub chi_square: f64,
    pub degrees_of_freedom: usize,
    pub reduced_chi_square: f64,
    pub nodes: Vec<Node>,
    /// Node-major: power, Ti, Te, velocity, H fraction, He/(1-H).
    /// Last entry is not the marginal error of total He fraction.
    pub conditional_standard_error: Vec<Option<f64>>,
    pub transformed_covariance: Option<Vec<f64>>,
    pub predicted: Vec<[f64; 2]>,
    pub standardized_residual: Vec<[f64; 2]>,
    pub objective_history: Vec<f64>,
    pub active_bounds: Vec<usize>,
    pub normalized_gradient: f64,
    pub uncertainty: String,
}
fn bound(field: usize) -> (f64, f64) {
    match field {
        0 => (-24., 24.),
        1 => (0.15f64.ln(), 6f64.ln()),
        2 => (0.15f64.ln(), 12f64.ln()),
        3 => (-2., 2.),
        4 | 5 => (0., 1.),
        _ => unreachable!(),
    }
}
fn node(p: &[f64]) -> Node {
    let h = p[4];
    let he = (1. - h) * p[5];
    Node {
        power_adu2: p[0].exp(),
        ti_k: 1000. * p[1].exp(),
        te_k: 1000. * p[2].exp(),
        velocity_m_s: 1000. * p[3],
        h_fraction: h,
        he_fraction: he,
        o_fraction: 1. - h - he,
    }
}
pub fn encode(nodes: &[Node]) -> V<f64> {
    V::from_iterator(
        nodes.len() * 6,
        nodes.iter().flat_map(|n| {
            [
                n.power_adu2.ln(),
                (n.ti_k / 1000.).ln(),
                (n.te_k / 1000.).ln(),
                n.velocity_m_s / 1000.,
                n.h_fraction,
                if n.h_fraction < 1. {
                    n.he_fraction / (1. - n.h_fraction)
                } else {
                    0.
                },
            ]
        }),
    )
}
impl Problem {
    fn local(&self, x: &V<f64>, cell: &Cell) -> [f64; 6] {
        let mut p = [0.; 6];
        for &(k, w) in &cell.knot_weights {
            for f in 0..6 {
                if f != 4 || cell.h_knot_weights.is_none() {
                    p[f] += w * x[6 * k + f];
                }
            }
        }
        if let Some(weights) = &cell.h_knot_weights {
            p[4] = weights.iter().map(|(k, w)| w * x[6 * k + 4]).sum();
        }
        p
    }
    fn spectrum(
        &self,
        theory: &Theory,
        p: &[f64; 6],
        density: f64,
        physical: Option<(f64, f64, f64)>,
    ) -> Result<Vec<f64>> {
        if let Some((gain, range, hweight)) = physical {
            let n = node(p);
            let ne = n.power_adu2 * 1e11;
            let (s, mass) = theory.spectrum_with_mass(
                n.ti_k,
                n.te_k,
                n.velocity_m_s,
                n.h_fraction * hweight,
                0.,
                ne,
            );
            return Ok(s
                .into_iter()
                .map(|v| v * ne * mass / (gain * range * range))
                .collect());
        }
        let n = node(p);
        let s = theory.spectrum(
            n.ti_k,
            n.te_k,
            n.velocity_m_s,
            n.h_fraction,
            if n.h_fraction < 1. {
                n.he_fraction / (1. - n.h_fraction)
            } else {
                0.
            },
            density,
        );
        let integral: f64 = s
            .windows(2)
            .zip(self.frequency_hz.windows(2))
            .map(|(s, f)| 0.5 * (s[0] + s[1]) * (f[1] - f[0]))
            .sum();
        ensure!(integral > 0. && integral.is_finite(), "bad spectral mass");
        Ok(s.into_iter().map(|v| v * n.power_adu2 / integral).collect())
    }
    pub fn evaluate(&self, x: &V<f64>, derivatives: bool) -> Result<(V<f64>, M<f64>)> {
        self.evaluate_mode(
            x,
            derivatives,
            std::env::var_os("SYISR_FAST_JOINT").is_some(),
        )
    }
    /// Exact physical model with reference or checked analytic/batched derivatives.
    pub fn evaluate_mode(
        &self,
        x: &V<f64>,
        derivatives: bool,
        fast: bool,
    ) -> Result<(V<f64>, M<f64>)> {
        ensure!(
            x.len() == 6 * self.knot_count && x.iter().all(|v| v.is_finite()),
            "composition parameter dimensions"
        );
        let pieces = self
            .cells
            .par_iter()
            .map(|cell| -> Result<_> {
                let p = self.local(x, cell);
                if fast || std::env::var_os("SYISR_FUSED_JOINT").is_some() {
                    if let Some((gain, range, hw)) = cell.density_calibration {
                        let mut channels = if derivatives && !fast {
                            let mut channels = vec![self.spectrum(
                                &cell.theory,
                                &p,
                                1e11,
                                cell.density_calibration,
                            )?];
                            for f in 0..5 {
                                if f == 4 && hw == 0. {
                                    channels.push(vec![0.; self.frequency_hz.len()]);
                                    continue;
                                }
                                let e = 1e-4;
                                let (lo, hi) = bound(f);
                                let mut a = p;
                                let mut b = p;
                                a[f] = (p[f] + e).min(hi);
                                b[f] = (p[f] - e).max(lo);
                                let sa = self.spectrum(
                                    &cell.theory,
                                    &a,
                                    1e11,
                                    cell.density_calibration,
                                )?;
                                let sb = self.spectrum(
                                    &cell.theory,
                                    &b,
                                    1e11,
                                    cell.density_calibration,
                                )?;
                                channels.push(
                                    sa.iter()
                                        .zip(sb)
                                        .map(|(a, b)| (a - b) / a_span(p[f], e, lo, hi))
                                        .collect(),
                                );
                            }
                            channels
                        } else if derivatives {
                            cell.theory.physical_channels(
                                1000. * p[1].exp(),
                                1000. * p[2].exp(),
                                1000. * p[3],
                                p[4] * hw,
                                p[0].exp() * 1e11,
                                gain,
                                range,
                                hw,
                            )
                        } else {
                            vec![self.spectrum(&cell.theory, &p, 1e11, cell.density_calibration)?]
                        };
                        if derivatives {
                            channels.truncate(if hw == 0. { 5 } else { 6 });
                        }
                        let mut pred = vec![C::default(); cell.terms.len()];
                        let mut deriv = if derivatives {
                            vec![vec![C::default(); cell.terms.len()]; 6]
                        } else {
                            vec![]
                        };
                        for (j, t) in cell.terms.iter().enumerate() {
                            let mut values = [C::default(); 7];
                            crate::response_dot::fused_into(
                                &t.kernel,
                                &channels,
                                &mut values[..channels.len()],
                            );
                            pred[j] = values[0];
                            if derivatives {
                                for f in 0..channels.len() - 1 {
                                    deriv[f][j] = values[f + 1];
                                }
                            }
                        }
                        return Ok((pred, deriv));
                    }
                }
                let spectrum = self.spectrum(
                    &cell.theory,
                    &p,
                    cell.shape_density_m3.unwrap_or(self.shape_density_m3),
                    cell.density_calibration,
                )?;
                let pred: Vec<C> = cell
                    .terms
                    .iter()
                    .map(|t| t.kernel.iter().zip(&spectrum).map(|(a, b)| *a * *b).sum())
                    .collect();
                let mut deriv = vec![];
                if derivatives {
                    for f in 0..6 {
                        if cell
                            .density_calibration
                            .is_some_and(|(_, _, w)| f == 5 || (f == 4 && w == 0.))
                        {
                            deriv.push(vec![C::default(); cell.terms.len()]);
                            continue;
                        }
                        let eps = 1e-4;
                        let (lo, hi) = bound(f);
                        let mut a = p;
                        let mut b = p;
                        a[f] = (p[f] + eps).min(hi);
                        b[f] = (p[f] - eps).max(lo);
                        let sa = self.spectrum(
                            &cell.theory,
                            &a,
                            cell.shape_density_m3.unwrap_or(self.shape_density_m3),
                            cell.density_calibration,
                        )?;
                        let sb = self.spectrum(
                            &cell.theory,
                            &b,
                            cell.shape_density_m3.unwrap_or(self.shape_density_m3),
                            cell.density_calibration,
                        )?;
                        let ds: Vec<_> = sa
                            .iter()
                            .zip(sb)
                            .map(|(a, b)| (a - b) / (a_span(p[f], eps, lo, hi)))
                            .collect();
                        let d: Vec<C> = cell
                            .terms
                            .iter()
                            .map(|t| t.kernel.iter().zip(&ds).map(|(a, b)| *a * *b).sum())
                            .collect();
                        deriv.push(d);
                    }
                }
                Ok((pred, deriv))
            })
            .collect::<Result<Vec<_>>>()?;
        let mut prediction = V::zeros(self.products.len() * 2 + 1);
        let mut jac = M::zeros(self.products.len() * 2 + 1, x.len());
        for (cell, (pred, deriv)) in self.cells.iter().zip(pieces) {
            for (j, t) in cell.terms.iter().enumerate() {
                let m = t.measurement;
                prediction[2 * m] += pred[j].re;
                prediction[2 * m + 1] += pred[j].im;
                if derivatives {
                    for &(k, w) in &cell.knot_weights {
                        for f in 0..6 {
                            if f != 4 || cell.h_knot_weights.is_none() {
                                jac[(2 * m, 6 * k + f)] += w * deriv[f][j].re;
                                jac[(2 * m + 1, 6 * k + f)] += w * deriv[f][j].im;
                            }
                        }
                    }
                    if let Some(weights) = &cell.h_knot_weights {
                        for &(k, w) in weights {
                            jac[(2 * m, 6 * k + 4)] += w * deriv[4][j].re;
                            jac[(2 * m + 1, 6 * k + 4)] += w * deriv[4][j].im;
                        }
                    }
                }
            }
        }
        // Profile one shared complex-background amplitude against its independent prior.
        let mut den = self.background_sigma.powi(-2);
        let mut num = 0.;
        let mut derivative = V::zeros(x.len());
        for (i, p) in self.products.iter().enumerate() {
            for c in 0..2 {
                let b = [self.background_basis[i].re, self.background_basis[i].im][c];
                let w = p.sigma[c].powi(-2);
                den += w * b * b;
                num += w * b * (p.value[c] - prediction[2 * i + c]);
                if derivatives {
                    for k in 0..x.len() {
                        derivative[k] -= w * b * jac[(2 * i + c, k)];
                    }
                }
            }
        }
        let adjustment = num / den;
        derivative /= den;
        for i in 0..self.products.len() {
            for c in 0..2 {
                let b = [self.background_basis[i].re, self.background_basis[i].im][c];
                prediction[2 * i + c] += adjustment * b;
                if derivatives {
                    for k in 0..x.len() {
                        jac[(2 * i + c, k)] += b * derivative[k];
                    }
                }
            }
        }
        let last = self.products.len() * 2;
        prediction[last] = adjustment / self.background_sigma;
        if derivatives {
            for k in 0..x.len() {
                jac[(last, k)] = derivative[k] / self.background_sigma;
            }
        }
        for (i, p) in self.products.iter().enumerate() {
            for c in 0..2 {
                prediction[2 * i + c] = (prediction[2 * i + c] - p.value[c]) / p.sigma[c];
                if derivatives {
                    jac.row_mut(2 * i + c).scale_mut(1. / p.sigma[c]);
                }
            }
        }
        Ok((prediction, jac))
    }
    pub fn solve(
        &self,
        initial: &[Node],
        max_iterations: usize,
        fixed: &[(usize, f64)],
    ) -> Result<Fit> {
        self.solve_with_tolerance(initial, max_iterations, fixed, 1e-3)
    }
    /// Optimizer score tolerance in local conditional-curvature standard errors.
    pub fn solve_with_tolerance(
        &self,
        initial: &[Node],
        max_iterations: usize,
        fixed: &[(usize, f64)],
        score_tolerance: f64,
    ) -> Result<Fit> {
        ensure!(
            score_tolerance.is_finite() && score_tolerance > 0. && score_tolerance <= 0.1,
            "invalid optimizer tolerance"
        );
        ensure!(
            initial.len() == self.knot_count
                && self
                    .products
                    .iter()
                    .all(|p| p.sigma.iter().all(|s| s.is_finite() && *s > 0.)),
            "invalid fit input"
        );
        ensure!(
            self.frequency_hz.len() > 1
                && self
                    .frequency_hz
                    .windows(2)
                    .all(|f| f[0].is_finite() && f[1].is_finite() && f[1] > f[0]),
            "invalid frequency grid"
        );
        ensure!(
            self.cells.iter().all(|c| !c.knot_weights.is_empty()
                && c.knot_weights
                    .iter()
                    .all(|&(k, w)| k < self.knot_count && w.is_finite() && w >= 0.)
                && (c.knot_weights.iter().map(|(_, w)| w).sum::<f64>() - 1.).abs() < 1e-10
                && c.h_knot_weights
                    .as_ref()
                    .is_none_or(|weights| !weights.is_empty()
                        && weights
                            .iter()
                            .all(|(k, w)| *k < self.knot_count && w.is_finite() && *w >= 0.)
                        && (weights.iter().map(|(_, w)| w).sum::<f64>() - 1.).abs() < 1e-10)
                && c.terms.iter().all(|t| t.measurement < self.products.len()
                    && t.kernel.len() == self.frequency_hz.len())),
            "invalid response or knot weights"
        );
        ensure!(
            fixed.iter().all(|&(i, v)| i < 6 * self.knot_count
                && v.is_finite()
                && v >= bound(i % 6).0
                && v <= bound(i % 6).1),
            "invalid fixed parameter"
        );
        let mut x = encode(initial);
        for &(i, v) in fixed {
            x[i] = v;
        }
        let (mut r, mut j) = self.evaluate(&x, true)?;
        let mut cost = r.norm_squared();
        let mut hist = vec![cost];
        let mut lambda = 1e-3;
        let mut status = "not_converged";
        let mut iterations = 0;
        let mut ng = f64::INFINITY;
        for it in 0..max_iterations {
            iterations = it + 1;
            let mut h = j.transpose() * &j;
            let mut g = j.transpose() * &r;
            let mut blocked = vec![false; x.len()];
            for i in 0..x.len() {
                let (lo, hi) = bound(i % 6);
                blocked[i] = fixed.iter().any(|&(k, _)| k == i)
                    || h[(i, i)] < 1e-20
                    || (x[i] <= lo + 1e-8 && g[i] > 0.)
                    || (x[i] >= hi - 1e-8 && g[i] < 0.);
            }
            ng = (0..x.len())
                .filter(|&i| !blocked[i])
                .map(|i| g[i].abs() / h[(i, i)].sqrt().max(1e-12))
                .fold(0., f64::max);
            if std::env::var_os("SYISR_JOINT_PROGRESS").is_some() && it % 10 == 0 {
                eprintln!("JOINT_ITER {it} Q={cost:.6e} gradient={ng:.4e}");
            }
            if ng < score_tolerance {
                status = "converged";
                break;
            }
            for i in 0..x.len() {
                h[(i, i)] += lambda * h[(i, i)].max(1e-10);
                if blocked[i] {
                    for k in 0..x.len() {
                        h[(i, k)] = 0.;
                        h[(k, i)] = 0.;
                    }
                    h[(i, i)] = 1.;
                    g[i] = 0.;
                }
            }
            let Some(chol) = h.cholesky() else {
                lambda *= 10.;
                continue;
            };
            let mut step = chol.solve(&(-g));
            if self.cells.iter().any(|c| c.density_calibration.is_some()) {
                // A weak halo/composition coordinate must not throttle every
                // identifiable field in the joint profile. Project individual
                // transformed coordinates; accept only if the exact cost falls.
                for i in 0..step.len() {
                    let radius = match i % 6 {
                        0 => 0.5,
                        1 | 2 => 0.3,
                        3 => 0.25,
                        _ => 0.15,
                    };
                    step[i] = step[i].clamp(-radius, radius);
                }
            } else if step.amax() > 1. {
                step /= step.amax();
            }
            let mut candidate = &x + step;
            for i in 0..x.len() {
                let (lo, hi) = bound(i % 6);
                candidate[i] = candidate[i].clamp(lo, hi);
            }
            for &(i, v) in fixed {
                candidate[i] = v;
            }
            let (nr, _) = self.evaluate(&candidate, false)?;
            let nc = nr.norm_squared();
            if nc < cost {
                let change = (&candidate - &x).amax();
                let improvement = cost - nc;
                x = candidate;
                cost = nc;
                hist.push(cost);
                (r, j) = self.evaluate(&x, true)?;
                lambda = (lambda / 3.).max(1e-10);
                if change < 1e-6 && improvement < 1e-10 * cost.max(1.) {
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
        let active: Vec<_> = (0..x.len())
            .filter(|&i| {
                let (lo, hi) = bound(i % 6);
                x[i] <= lo + 1e-6 || x[i] >= hi - 1e-6
            })
            .collect();
        let h = j.transpose() * &j;
        // Curvature is conditional on fixed and active-bound coordinates.
        let free: Vec<_> = (0..x.len())
            .filter(|i| {
                !active.contains(i) && !fixed.iter().any(|&(k, _)| k == *i) && h[(*i, *i)] > 1e-20
            })
            .collect();
        let sub = M::from_fn(free.len(), free.len(), |i, j| h[(free[i], free[j])]);
        let cov = sub
            .cholesky()
            .map(|c| c.inverse())
            .filter(|c| c.iter().all(|v| v.is_finite()))
            .map(|c| {
                let mut full = M::zeros(x.len(), x.len());
                for (i, &a) in free.iter().enumerate() {
                    for (j, &b) in free.iter().enumerate() {
                        full[(a, b)] = c[(i, j)];
                    }
                }
                full
            });
        let se = (0..x.len())
            .map(|i| {
                if !free.contains(&i) {
                    None
                } else {
                    cov.as_ref().map(|c| {
                        c[(i, i)].sqrt()
                            * match i % 6 {
                                0 => x[i].exp(),
                                1 | 2 => 1000. * x[i].exp(),
                                3 => 1000.,
                                _ => 1.,
                            }
                    })
                }
            })
            .collect();
        let dof = (self
            .products
            .iter()
            .map(|p| p.sigma.iter().filter(|&&s| s < 1e20).count())
            .sum::<usize>()
            + 1)
        .saturating_sub(x.len() - fixed.len());
        Ok(Fit{status:status.into(),iterations,chi_square:cost,degrees_of_freedom:dof,reduced_chi_square:cost/dof.max(1)as f64,nodes:x.as_slice().chunks(6).map(node).collect(),conditional_standard_error:se,transformed_covariance:cov.map(|c|c.as_slice().to_vec()),predicted:self.products.iter().enumerate().map(|(i,p)|[r[2*i]*p.sigma[0]+p.value[0],r[2*i+1]*p.sigma[1]+p.value[1]]).collect(),standardized_residual:r.as_slice()[..2*self.products.len()].chunks(2).map(|a|[a[0],a[1]]).collect(),objective_history:hist,active_bounds:active,normalized_gradient:ng,uncertainty:"Conditional diagonal-weight curvature; range/lag correlations and response uncertainty are not marginalized. Fraction errors at bounds are withheld.".into()})
    }
}
fn a_span(p: f64, eps: f64, lo: f64, hi: f64) -> f64 {
    (p + eps).min(hi) - (p - eps).max(lo)
}
#[cfg(test)]
mod tests {
    use super::*;
    fn fixture() -> (Problem, Node) {
        let theory = Arc::new(Theory::new(725., 500.).unwrap());
        let f = theory.f.clone();
        let terms = (0..41)
            .map(|l| Term {
                measurement: l,
                kernel: f
                    .iter()
                    .enumerate()
                    .map(|(j, &f)| {
                        C::from_polar(
                            500. * if j == 0 || j == 200 { 0.5 } else { 1. },
                            -std::f64::consts::TAU * f * l as f64 * 1e-5,
                        )
                    })
                    .collect(),
            })
            .collect();
        let mut p = Problem {
            frequency_hz: f,
            radar_frequency_hz: 430.2e6,
            shape_density_m3: 1e11,
            cells: vec![Cell {
                theory,
                shape_density_m3: None,
                density_calibration: None,
                h_knot_weights: None,
                knot_weights: vec![(0, 1.)],
                terms,
            }],
            products: (0..41)
                .map(|l| Product {
                    value: [0., 0.],
                    sigma: [0.1, if l == 0 { 1e30 } else { 0.1 }],
                })
                .collect(),
            knot_count: 1,
            background_basis: (0..41)
                .map(|l| C::new(if l == 0 { 1. } else { 0.01 }, 0.))
                .collect(),
            background_sigma: 1.,
        };
        let n = Node {
            power_adu2: 1000.,
            ti_k: 1000.,
            te_k: 1500.,
            velocity_m_s: 50.,
            h_fraction: 0.3,
            he_fraction: 0.1,
            o_fraction: 0.6,
        };
        let x = encode(&[n.clone()]);
        let local = p.local(&x, &p.cells[0]);
        let s = p.spectrum(&p.cells[0].theory, &local, 1e11, None).unwrap();
        for t in &p.cells[0].terms {
            let z: C = t.kernel.iter().zip(&s).map(|(k, s)| *k * *s).sum();
            p.products[t.measurement].value = [z.re, z.im];
        }
        (p, n)
    }
    #[test]
    fn profiled_background_jacobian_matches_finite_differences() {
        let (p, n) = fixture();
        let x = encode(&[n]);
        let (_, j) = p.evaluate(&x, true).unwrap();
        for k in 0..x.len() {
            let eps = 1e-5;
            let mut a = x.clone();
            let mut b = x.clone();
            a[k] += eps;
            b[k] -= eps;
            let (ra, _) = p.evaluate(&a, false).unwrap();
            let (rb, _) = p.evaluate(&b, false).unwrap();
            let d = (ra - rb) / (2. * eps);
            let err = (d - j.column(k)).norm() / j.column(k).norm().max(1e-8);
            assert!(err < 2e-4, "field {k}: {err}");
        }
    }
    #[test]
    fn noiseless_h_he_mix_recovered_with_background_profiled() {
        let (p, n) = fixture();
        let mut seed = n.clone();
        seed.h_fraction = 0.1;
        seed.he_fraction = 0.2;
        seed.o_fraction = 0.7;
        seed.ti_k = 800.;
        seed.te_k = 1700.;
        let f = p.solve(&[seed], 100, &[]).unwrap();
        assert!((f.nodes[0].h_fraction - n.h_fraction).abs() < 0.002);
        assert!((f.nodes[0].he_fraction - n.he_fraction).abs() < 0.002);
        assert!(f.chi_square < 1e-4, "{}", f.chi_square);
    }
    #[test]
    fn physical_density_jacobian_tracks_debye_shape_and_mass() {
        let (mut p, mut n) = fixture();
        n.power_adu2 = 1.2;
        n.he_fraction = 0.;
        n.h_fraction = 0.2;
        p.cells[0].density_calibration = Some((700., 500., 0.8));
        let x = encode(&[n]);
        let (_, j) = p.evaluate(&x, true).unwrap();
        for k in 0..5 {
            let eps = 1e-5;
            let mut a = x.clone();
            let mut b = x.clone();
            a[k] += eps;
            b[k] -= eps;
            let (ra, _) = p.evaluate(&a, false).unwrap();
            let (rb, _) = p.evaluate(&b, false).unwrap();
            let d = (ra - rb) / (2. * eps);
            let err = (d - j.column(k)).norm() / j.column(k).norm().max(1e-8);
            assert!(err < 3e-4, "physical field{k}: {err}");
        }
    }
    #[test]
    fn analytic_batched_profile_matches_reference_residual_and_jacobian() {
        let (mut p, mut n) = fixture();
        n.power_adu2 = 1.2;
        n.h_fraction = 0.3;
        n.he_fraction = 0.;
        n.velocity_m_s = 37.123;
        n.te_k = 1500.123;
        p.cells[0].density_calibration = Some((700., 500., 0.8));
        let x = encode(&[n]);
        let (a, ja) = p.evaluate_mode(&x, true, false).unwrap();
        let (b, jb) = p.evaluate_mode(&x, true, true).unwrap();
        assert!((&a - &b).norm() / a.norm() < 1e-12);
        for k in 0..x.len() {
            let scale = ja.column(k).norm().max(1e-10);
            assert!(
                (ja.column(k) - jb.column(k)).norm() / scale < 4e-4,
                "column{k}"
            );
        }
    }
    #[test]
    fn independent_h_basis_jacobian_matches_finite_differences() {
        let (mut p, mut a) = fixture();
        a.power_adu2 = 1.1;
        a.he_fraction = 0.;
        a.h_fraction = 0.2;
        a.velocity_m_s = 37.123;
        let mut b = a.clone();
        b.h_fraction = 0.7;
        b.ti_k = 1300.;
        p.knot_count = 2;
        p.cells[0].knot_weights = vec![(0, 0.2), (1, 0.8)];
        p.cells[0].h_knot_weights = Some(vec![(0, 0.75), (1, 0.25)]);
        p.cells[0].density_calibration = Some((700., 500., 0.8));
        let x = encode(&[a, b]);
        let (_, j) = p.evaluate(&x, true).unwrap();
        for k in 0..x.len() {
            if k % 6 == 5 {
                continue;
            }
            let e = 1e-5;
            let mut a = x.clone();
            let mut b = x.clone();
            a[k] += e;
            b[k] -= e;
            let (ra, _) = p.evaluate(&a, false).unwrap();
            let (rb, _) = p.evaluate(&b, false).unwrap();
            let d = (ra - rb) / (2. * e);
            assert!(
                (d - j.column(k)).norm() / j.column(k).norm().max(1e-8) < 4e-4,
                "field{k}"
            );
        }
    }
}
