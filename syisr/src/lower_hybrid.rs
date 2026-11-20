//! Explicit near-perpendicular, finite-beam BGK pilot. Ordinary theory guards remain.
//! Number-conserving BGK is a sensitivity model, not a Coulomb collision operator.
use crate::{
    autodiff::{D, Z},
    campaign,
    lp480::{self, Decoded, Manifest, C_KM, DT, NL},
    lp480_fit::{self, Background, Reference},
};
use anyhow::{ensure, Result};
use errorfunctions::ComplexErrorFunctions;
use nalgebra::{DMatrix as M, DVector as V};
use num_complex::Complex64 as C;
use rayon::prelude::*;
use std::{
    f64::consts::{PI, TAU},
    path::Path,
};
const KB: f64 = 1.380649e-23;
const ME: f64 = 9.1093837139e-31;
const AMU: f64 = 1.66053906892e-27;
const E: f64 = 1.602176634e-19;
const EPS: f64 = 8.8541878188e-12;
fn real(x: D) -> Z {
    Z {
        v: C::new(x.v, 0.),
        d: x.d.map(|v| C::new(v, 0.)),
    }
}
fn sub(a: Z, b: Z) -> Z {
    Z {
        v: a.v - b.v,
        d: std::array::from_fn(|i| a.d[i] - b.d[i]),
    }
}
fn div(a: Z, b: Z) -> Z {
    let v = a.v / b.v;
    Z {
        v,
        d: std::array::from_fn(|i| (a.d[i] - v * b.d[i]) / b.v),
    }
}
fn wd(z: Z) -> Z {
    let v = z.v.w();
    let d = -2. * z.v * v + C::new(0., 2. / PI.sqrt());
    Z {
        v,
        d: z.d.map(|x| x * d),
    }
}
fn exp(x: D) -> D {
    let v = x.v.exp();
    D {
        v,
        d: x.d.map(|a| a * v),
    }
}
fn sin(x: D) -> D {
    D {
        v: x.v.sin(),
        d: x.d.map(|a| a * x.v.cos()),
    }
}
fn cos(x: D) -> D {
    D {
        v: x.v.cos(),
        d: x.d.map(|a| -a * x.v.sin()),
    }
}
fn scaled_i(n: usize, x: f64) -> f64 {
    let mut t = 1.;
    for j in 1..=n {
        t *= x / (2. * j as f64);
    }
    let mut s = t;
    for j in 1..300 {
        t *= x * x / (4. * j as f64 * (j + n) as f64);
        s += t;
        if t.abs() < s.abs() * 1e-16 {
            break;
        }
    }
    s * (-x).exp()
}
fn lambda(n: usize, b: D) -> D {
    let v = scaled_i(n, b.v);
    let derivative =
        0.5 * (scaled_i(n + 1, b.v) + scaled_i(if n == 0 { 1 } else { n - 1 }, b.v)) - v;
    D {
        v,
        d: b.d.map(|x| x * derivative),
    }
}
struct Electron {
    a: D,
    gyro: f64,
    weights: Vec<(i32, D)>,
    nu: f64,
}
impl Electron {
    fn new(te: D, b: f64, angle: D, nu: f64) -> Self {
        let k = 4. * PI * 430.2e6 / 299792458.;
        let gyro = E * b / ME;
        let mut kp = sin(angle) * k;
        if kp.v < 0. {
            kp = -kp;
        }
        let kt = cos(angle) * k;
        let v = (te * (KB / ME)).sqrt();
        let bb = kt * kt * v * v / (gyro * gyro);
        let weights = (-24_i32..=24)
            .map(|n| (n, lambda(n.unsigned_abs() as usize, bb)))
            .filter(|(_, w)| w.v > 1e-22)
            .collect();
        Self {
            a: kp * v * 2f64.sqrt(),
            gyro,
            weights,
            nu,
        }
    }
    fn g(&self, w: D) -> Z {
        let one = Z::constant(C::new(1., 0.));
        let mut g = Z::constant(C::default());
        for (n, weight) in &self.weights {
            let shifted = w - D::constant(*n as f64 * self.gyro);
            let term = if self.a.v < 1e-8 {
                div(one, sub(Z::constant(C::new(self.nu, 0.)), Z::i(shifted)))
            } else {
                let z = div(
                    real(shifted) + Z::constant(C::new(0., self.nu)),
                    real(self.a),
                );
                wd(z) * (D::constant(PI.sqrt()) / self.a)
            };
            g = g + term * *weight;
        }
        div(g, sub(one, g * D::constant(self.nu)))
    }
}
fn ion_g(a: D, w: D) -> Z {
    wd(div(real(w), real(a))) * (D::constant(PI.sqrt()) / a)
}
fn spectrum(ne: D, ti: D, te: D, v: D, frequency: f64, e: &Electron, heavy: [f64; 4]) -> D {
    spectrum_mixture(
        ne,
        ti,
        te,
        v,
        D::constant(0.),
        D::constant(0.),
        frequency,
        e,
        heavy,
        1.,
        1.,
    )
}
fn spectrum_mixture(
    ne: D,
    ti: D,
    te: D,
    v: D,
    h: D,
    he: D,
    frequency: f64,
    e: &Electron,
    heavy: [f64; 4],
    h_ratio: f64,
    electron_noise: f64,
) -> D {
    let k = 4. * PI * 430.2e6 / 299792458.;
    let w = D::constant(TAU * frequency) - v * k;
    let alpha = ne * (E * E / (EPS * KB * k * k));
    let ge = e.g(w);
    let one = Z::constant(C::new(1., 0.));
    let ce = (one + Z::i(w) * ge) * (alpha / te);
    let mut ci = Z::constant(C::default());
    let mut fi = D::constant(0.);
    let heavy_total = (D::constant(1.) - h) * (D::constant(1.) - he);
    let fractions = [
        h,
        (D::constant(1.) - h) * he,
        heavy_total * heavy[0],
        heavy_total * heavy[1],
        heavy_total * heavy[2],
        heavy_total * heavy[3],
    ];
    for (m, f) in [1., 4., 16., 30., 32., 14.].into_iter().zip(fractions) {
        if f.v == 0. && f.d.iter().all(|x| *x == 0.) {
            continue;
        }
        let temp = ti * if m == 1. { h_ratio } else { 1. };
        let a = (temp * (2. * KB / (m * AMU))).sqrt() * k;
        let gi = ion_g(a, w);
        ci = ci + (one + Z::i(w) * gi) * (alpha / temp * f);
        fi = fi + gi.re() * f * 2.;
    }
    ((one + ci).norm_sqr() * ge.re() * (2. * electron_noise) + ce.norm_sqr() * fi)
        / (one + ci + ce).norm_sqr()
        * ne
}
/// Direct, smooth magnetized response: derivatives do not cross a bilinear cache.
#[allow(clippy::too_many_arguments)]
pub(crate) fn physical_channels(
    b: f64,
    aspect: f64,
    nu: f64,
    width: f64,
    frequency: &[f64],
    ti: f64,
    te: f64,
    v: f64,
    h: f64,
    he: f64,
    ne: f64,
    gain: f64,
    range: f64,
    hw: f64,
    heavy: [f64; 4],
    h_ratio: f64,
    ef: bool,
    derivatives: bool,
) -> Vec<Vec<f64>> {
    let seed = |v, k, s| {
        if derivatives {
            D::seed(v, k, s)
        } else {
            D::constant(v)
        }
    };
    let ne = seed(ne, 0, ne);
    let ti = seed(ti, 1, ti);
    let te = seed(te, 2, te);
    let v = seed(v, 3, 1000.);
    let h = seed(h, 4, hw);
    let he = seed(he, 5, hw);
    let angles = if width == 0. {
        vec![(0., 1.)]
    } else {
        quadrature()
    };
    let mut sum = vec![D::constant(0.); frequency.len()];
    for (a, w) in angles {
        let angle = D::constant((aspect + width * a / (8. * 2f64.ln()).sqrt()) * PI / 180.);
        let electron = Electron::new(te, b, angle, nu);
        for (i, f) in frequency.iter().enumerate() {
            sum[i] = sum[i]
                + spectrum_mixture(
                    ne,
                    ti,
                    te,
                    v,
                    h,
                    he,
                    *f,
                    &electron,
                    heavy,
                    h_ratio,
                    if ef { 1. } else { 0. },
                ) * w;
        }
    }
    let mut out = vec![vec![0.; frequency.len()]; if derivatives { 7 } else { 1 }];
    let scale = 1. / (gain * range * range);
    for (i, s) in sum.into_iter().enumerate() {
        out[0][i] = s.v * scale;
        if derivatives {
            for k in 0..6 {
                out[k + 1][i] = s.d[k] * scale;
            }
        }
    }
    out
}

fn quadrature() -> Vec<(f64, f64)> {
    let n = std::env::var("SYISR_LH_ANGLES")
        .ok()
        .and_then(|v| v.parse::<usize>().ok())
        .unwrap_or(9);
    assert!((3..=65).contains(&n));
    let mut j = M::zeros(n, n);
    for i in 1..n {
        j[(i - 1, i)] = (i as f64).sqrt();
        j[(i, i - 1)] = (i as f64).sqrt();
    }
    let e = j.symmetric_eigen();
    (0..n)
        .map(|i| (e.eigenvalues[i], e.eigenvectors[(0, i)].powi(2)))
        .collect()
}

struct Cell {
    z: f64,
    range: f64,
    b: f64,
    angle: f64,
    heavy: [f64; 4],
    weights: Vec<f64>,
}
struct Model {
    cells: Vec<Cell>,
    frequency: Vec<f64>,
    phase: Vec<Vec<C>>,
    center: f64,
    nu: f64,
    angles: Vec<(f64, f64)>,
}
impl Model {
    fn new(d: &Decoded, g: usize, r: &Reference, nu: f64, refine: usize) -> Result<Self> {
        ensure!(
            nu > 0. && refine > 0,
            "positive explicit collision rate and refinement required"
        );
        let mut frequency = vec![];
        for i in -500 * refine as i32..=500 * refine as i32 {
            let f = i as f64 * 100. / refine as f64;
            if f.abs() > 1000. {
                frequency.push(f);
            }
        }
        for i in -100 * refine as i32..=100 * refine as i32 {
            frequency.push(i as f64 * 10. / refine as f64);
        }
        frequency.sort_by(f64::total_cmp);
        let df: Vec<_> = (0..frequency.len())
            .map(|i| {
                let left = if i == 0 {
                    frequency[i]
                } else {
                    frequency[i - 1]
                };
                let right = if i + 1 == frequency.len() {
                    frequency[i]
                } else {
                    frequency[i + 1]
                };
                (right - left) / 2.
            })
            .collect();
        let phase = (1..=NL)
            .map(|l| {
                frequency
                    .iter()
                    .zip(&df)
                    .map(|(f, w)| C::from_polar(*w, -TAU * f * l as f64 * DT))
                    .collect()
            })
            .collect();
        let center = d.gates[g].altitude_km;
        let mut cells = vec![];
        for zi in (100..600).step_by(5) {
            let z = zi as f64 + 2.5;
            let edges = [
                2. * lp480::range(zi as f64, d.job.el) / C_KM,
                2. * lp480::range(zi as f64 + 5., d.job.el) / C_KM,
            ];
            let weights = (1..=NL)
                .map(|l| lp480::weight(&d.gates[g], l, d.job.front_us, edges))
                .collect::<Vec<_>>();
            if weights.iter().all(|w| *w == 0.) {
                continue;
            }
            let range = lp480::range(z, d.job.el);
            let (b, angle) = crate::theory::Theory::beam_field_at_year(
                range,
                d.job.az,
                d.job.el,
                lp480::decimal_year(d.job.start)?,
            )?;
            let heavy = std::array::from_fn(|c| {
                lp480_fit::interpolate(
                    &r.altitude_km,
                    &r.heavy_fractions.iter().map(|x| x[c]).collect::<Vec<_>>(),
                    z,
                )
            });
            cells.push(Cell {
                z,
                range,
                b: b * 1e-9,
                angle: angle.to_radians(),
                heavy,
                weights,
            });
        }
        ensure!(!cells.is_empty(), "no finite-pulse cells");
        Ok(Self {
            cells,
            frequency,
            phase,
            center,
            nu,
            angles: quadrature(),
        })
    }
    fn evaluate(&self, q: &[f64; 6], gain: f64) -> Vec<[C; 7]> {
        let ne = D::seed(q[0].exp(), 0, q[0].exp());
        let ti = D::seed(q[1].exp(), 1, q[1].exp());
        let te = D::seed(q[2].exp(), 2, q[2].exp());
        let v = D::seed(q[3] * 1000., 3, 1000.);
        let width = D::seed(q[4].exp(), 4, q[4].exp());
        let gradient = D::seed(q[5], 5, 1.);
        let pieces: Vec<_> = self
            .cells
            .par_iter()
            .map(|cell| {
                let nz = ne * exp(gradient * ((cell.z - self.center) / 100.));
                let mut averaged = vec![D::constant(0.); self.frequency.len()];
                for &(a, w) in &self.angles {
                    let angle =
                        D::constant(cell.angle) + width * (PI / 180. / (8. * 2f64.ln()).sqrt() * a);
                    let el = Electron::new(te, cell.b, angle, self.nu);
                    for (i, f) in self.frequency.iter().enumerate() {
                        averaged[i] =
                            averaged[i] + spectrum(nz, ti, te, v, *f, &el, cell.heavy) * w;
                    }
                }
                let scale = 1. / (gain * cell.range.powi(2));
                (0..NL)
                    .map(|l| {
                        let mut out = [C::default(); 7];
                        for (s, p) in averaged.iter().zip(&self.phase[l]) {
                            let w = *p * (cell.weights[l] * scale);
                            out[0] += w * s.v;
                            for j in 0..6 {
                                out[j + 1] += w * s.d[j];
                            }
                        }
                        out
                    })
                    .collect::<Vec<_>>()
            })
            .collect();
        let mut out = vec![[C::default(); 7]; NL];
        for p in pieces {
            for (l, a) in p.into_iter().enumerate() {
                for j in 0..7 {
                    out[l][j] += a[j];
                }
            }
        }
        out
    }
}
fn loss(r: f64) -> (f64, f64) {
    if r.abs() <= 3. {
        (r * r, 1.)
    } else {
        (6. * r.abs() - 9., 3. / r.abs())
    }
}
fn bounds(i: usize) -> (f64, f64) {
    match i {
        0 => (1e8f64.ln(), 1e14f64.ln()),
        1 => (150f64.ln(), 6000f64.ln()),
        2 => (150f64.ln(), 12000f64.ln()),
        3 => (-2., 2.),
        4 => (0.4f64.ln(), 2f64.ln()),
        _ => (-3., 3.),
    }
}
fn median(v: &mut [f64]) -> f64 {
    v.sort_by(f64::total_cmp);
    v[v.len() / 2]
}
fn objective(
    model: &Model,
    q: &[f64; 6],
    gain: f64,
    obs: &[(usize, crate::coupled::Product)],
) -> (f64, M<f64>, V<f64>, Vec<[C; 7]>) {
    let pred = model.evaluate(q, gain);
    let mut cost = 0.;
    let mut h = M::zeros(6, 6);
    let mut g = V::zeros(6);
    for (l, p) in obs {
        for c in 0..2 {
            let get = |z: C| if c == 0 { z.re } else { z.im };
            let r = (get(pred[l - 1][0]) - p.value[c]) / p.sigma[c];
            let (v, w) = loss(r);
            cost += v;
            for a in 0..6 {
                let d = get(pred[l - 1][a + 1]) / p.sigma[c];
                g[a] += w * r * d;
                for b in 0..6 {
                    h[(a, b)] += w * d * get(pred[l - 1][b + 1]) / p.sigma[c];
                }
            }
        }
    }
    for (i, mean, sigma) in [(4, 1.22f64.ln(), 1.5f64.ln()), (5, 0., 1.5)] {
        let r = (q[i] - mean) / sigma;
        cost += r * r;
        g[i] += r / sigma;
        h[(i, i)] += 1. / sigma.powi(2);
    }
    (cost, h, g, pred)
}
pub fn fit(root: &Path, time: i64, altitude: f64, out: &Path, nu: f64) -> Result<()> {
    let manifest: Manifest = campaign::load(&root.join("index.h5"))?;
    let j = manifest
        .jobs
        .iter()
        .find(|j| j.start == time && (j.az - 357.99).abs() < 0.1 && (j.el - 60.).abs() < 0.1)
        .ok_or_else(|| anyhow::anyhow!("requested lower-hybrid beam missing"))?;
    let d: Decoded = campaign::load(&root.join("products").join(format!("{}.h5", j.id)))?;
    let r: Reference = campaign::load(&root.join("references").join(format!("{time}.h5")))?;
    let bg: Background = campaign::load(&root.join("background.h5"))?;
    let gate = (0..d.gates.len())
        .min_by(|a, b| {
            (d.gates[*a].altitude_km - altitude)
                .abs()
                .total_cmp(&(d.gates[*b].altitude_km - altitude).abs())
        })
        .unwrap();
    ensure!(
        d.gates[gate].altitude_km < 450.,
        "initial LH pilot supports bottomside gates only; no unvalidated H extension"
    );
    let refine = std::env::var("SYISR_LH_REFINE")
        .unwrap_or("1".into())
        .parse()?;
    let model = Model::new(&d, gate, &r, nu, refine)?;
    let (p, gs, ls) = lp480_fit::observations(&d, &bg);
    let obs: Vec<_> = p
        .into_iter()
        .zip(gs)
        .zip(ls)
        .filter(|((_, g), _)| *g == gate)
        .map(|((p, _), l)| (l, p))
        .collect();
    ensure!(obs.len() > 20, "insufficient LH lag support");
    let initial = |r: &Reference| {
        [
            lp480_fit::interpolate(&r.altitude_km, &r.ne_m3, model.center).ln(),
            lp480_fit::interpolate(&r.altitude_km, &r.ti_k, model.center).ln(),
            lp480_fit::interpolate(&r.altitude_km, &r.te_k, model.center).ln(),
            0.,
            1.22f64.ln(),
            0.,
        ]
    };
    let mut gains = vec![];
    for c in manifest
        .jobs
        .iter()
        .filter(|c| c.start % 7200 == 0 && (c.az - j.az).abs() < 0.01 && (c.el - j.el).abs() < 0.01)
    {
        let cd: Decoded = campaign::load(&root.join("products").join(format!("{}.h5", c.id)))?;
        let cr: Reference =
            campaign::load(&root.join("references").join(format!("{}.h5", c.start)))?;
        let (p, g, l) = lp480_fit::observations(&cd, &bg);
        if let Some(p) = p
            .iter()
            .zip(g)
            .zip(l)
            .find(|((p, g), l)| *g == gate && *l == 1 && p.value[0] > 5. * p.sigma[0])
        {
            let predicted = model.evaluate(&initial(&cr), 1.)[0][0].re;
            let gain = predicted / p.0 .0.value[0];
            if gain.is_finite() && gain > 0. {
                gains.push(gain);
            }
        }
    }
    ensure!(
        gains.len() > 3,
        "insufficient empirical LH density calibration"
    );
    let gain = median(&mut gains);
    let mut q = initial(&r);
    let mut lambda = 1e-3;
    let mut status = "unresolved";
    let mut iterations = 0;
    let mut history = vec![];
    let checkpoint = out.with_extension("checkpoint.h5");
    if checkpoint.exists() {
        let saved: serde_json::Value = campaign::load(&checkpoint)?;
        ensure!(
            saved["nu_s"].as_f64() == Some(nu)
                && saved["refinement"].as_u64() == Some(refine as u64),
            "LH checkpoint model mismatch"
        );
        q = serde_json::from_value(saved["q"].clone())?;
    }
    let (mut cost, mut h, mut g, mut pred) = objective(&model, &q, gain, &obs);
    let mut score = f64::INFINITY;
    for it in 0..10000 {
        iterations = it + 1;
        score = (0..6)
            .filter(|i| {
                let (lo, hi) = bounds(*i);
                !(q[*i] <= lo + 1e-8 && g[*i] > 0. || q[*i] >= hi - 1e-8 && g[*i] < 0.)
            })
            .map(|i| g[i].abs() / h[(i, i)].sqrt().max(1e-12))
            .fold(0., f64::max);
        if it % 5 == 0 {
            eprintln!("LH_ITER {it} objective={cost:.7e} gradient={score:.4e} lambda={lambda:.3e}");
        }
        if score < 0.03 {
            status = "converged";
            break;
        }
        let mut damped = h.clone();
        let mut grad = g.clone();
        for i in 0..6 {
            let (lo, hi) = bounds(i);
            damped[(i, i)] += lambda * h[(i, i)].max(1e-10);
            if q[i] <= lo + 1e-8 && g[i] > 0. || q[i] >= hi - 1e-8 && g[i] < 0. {
                damped.row_mut(i).fill(0.);
                damped.column_mut(i).fill(0.);
                damped[(i, i)] = 1.;
                grad[i] = 0.;
            }
        }
        let Some(chol) = damped.cholesky() else {
            lambda *= 10.;
            continue;
        };
        let step = chol.solve(&(-grad));
        let candidate = std::array::from_fn(|i| {
            let (lo, hi) = bounds(i);
            (q[i] + step[i].clamp(-0.3, 0.3)).clamp(lo, hi)
        });
        let (c, hh, gg, pp) = objective(&model, &candidate, gain, &obs);
        if c < cost {
            q = candidate;
            cost = c;
            h = hh;
            g = gg;
            pred = pp;
            lambda = (lambda / 3.).max(1e-10);
            history.push(cost);
        } else {
            lambda *= 8.;
            if lambda > 1e15 {
                status = "stalled";
                break;
            }
        }
        if it % 25 == 0 {
            campaign::save(
                &checkpoint,
                &serde_json::json!({"q":q,"nu_s":nu,"refinement":refine,"cost":cost,"iterations":iterations}),
            )?;
        }
    }
    let p:Vec<_>=obs.iter().map(|(l,p)|serde_json::json!({"lag_us":*l as f64*10.,"observed":p.value,"assigned_sigma":p.sigma,"predicted":[pred[l-1][0].re,pred[l-1][0].im]})).collect();
    let result = serde_json::json!({"status":status,"iterations":iterations,"normalized_gradient":score,"objective":cost,"objective_history":history,"utc_start":time,"azimuth_deg":j.az,"elevation_deg":j.el,"altitude_km":model.center,"slant_range_km":d.gates[gate].range_km,"ne_m3":q[0].exp(),"ti_k":q[1].exp(),"te_k":q[2].exp(),"velocity_m_s":1000.*q[3],"two_way_aspect_fwhm_deg":q[4].exp(),"log_ne_gradient_per100km":q[5],"effective_BGK_rate_s":nu,"frequency_hz":model.frequency,"angular_quadrature_nodes":model.angles.len(),"frequency_refinement":refine,"empirical_IRI_gain":gain,"calibration_windows":gains.len(),"lags":p,"physical_model":"Full magnetized electron Bessel/Faddeeva response with number-conserving BGK and unmagnetized large-k-rhoi heavy ions; no cold LH frequency fit; exact finite480us source-delay weights; Gaussian two-way aspect quadrature (node count recorded separately); six-direction automatic differentiation","limitations":"Local gated pilot: constant Ti/Te/LOS velocity over pulse; exponential density gradient; fitted effective beamwidth nuisance with weak instrumental prior; BGK is a fixed sensitivity scenario, not Coulomb collision inference. IRI calibration is empirical. No parameter confidence intervals. Requires grid, angle, collision and higher-altitude validation before full-day inclusion."});
    campaign::save(out, &result)?;
    eprintln!("LH_DONE {status} iterations={iterations} gradient={score} objective={cost}");
    Ok(())
}

/// Independent per-electron tables for SciPy/Kinetic cross-checks, not plotted as fits.
pub fn validation_table(out: &Path) -> Result<()> {
    let f: Vec<_> = (-200..=200).map(|i| i as f64 * 100.).collect();
    let mut rows = vec![];
    for angle in [0., 0.25, 2., 15.] {
        for nu in [100., 1000.] {
            let ne = D::constant(1e12);
            let ti = D::constant(1200.);
            let te = D::constant(1800.);
            let electron = Electron::new(te, 38e-6, D::constant(angle * PI / 180.), nu);
            let s: Vec<_> = f
                .iter()
                .map(|f| {
                    spectrum(ne, ti, te, D::constant(0.), *f, &electron, [1., 0., 0., 0.]).v / 1e12
                })
                .collect();
            rows.push(
                serde_json::json!({"aspect_deg":angle,"nu_s":nu,"spectrum_per_electron_Hz":s}),
            );
        }
    }
    campaign::save(
        out,
        &serde_json::json!({"frequency_hz":f,"ne_m3":1e12,"ti_k":1200.,"te_k":1800.,"B_T":38e-6,"radar_Hz":430.2e6,"ion_mass_amu":16.,"ion_response":"Unmagnetized large-k-rhoi limit; compare reference with ion B=0 and electron B=38uT","cases":rows}),
    )?;
    Ok(())
}
#[cfg(test)]
mod tests {
    use super::*;
    #[test]
    fn finite_perpendicular_response_and_parallel_limit() {
        let te = D::constant(1800.);
        for nu in [100., 1000.] {
            for a in [0., 0.25, 2.] {
                let e = Electron::new(te, 38e-6, D::constant(a * PI / 180.), nu);
                for f in [0., 10., 1000., 6400., 17000.] {
                    let p = spectrum(
                        D::constant(1e12),
                        D::constant(1200.),
                        te,
                        D::constant(0.),
                        f,
                        &e,
                        [1., 0., 0., 0.],
                    )
                    .v;
                    let m = spectrum(
                        D::constant(1e12),
                        D::constant(1200.),
                        te,
                        D::constant(0.),
                        -f,
                        &e,
                        [1., 0., 0., 0.],
                    )
                    .v;
                    assert!(p.is_finite() && p >= 0.);
                    assert!((p - m).abs() < 1e-10 * p.abs().max(1.));
                }
            }
            let a = Electron::new(te, 38e-6, D::constant(PI / 2.), nu);
            let b = Electron::new(te, 15e-6, D::constant(PI / 2.), nu);
            assert!((a.g(D::constant(3000.)).v - b.g(D::constant(3000.)).v).norm() < 1e-14);
        }
    }
    fn point(q: [f64; 6]) -> D {
        let ne = D::seed(q[0].exp(), 0, q[0].exp()) * exp(D::seed(q[5], 5, 1.) * 0.4);
        let ti = D::seed(q[1].exp(), 1, q[1].exp());
        let te = D::seed(q[2].exp(), 2, q[2].exp());
        let velocity = D::seed(q[3] * 1000., 3, 1000.);
        let angle =
            D::constant(0.25 * PI / 180.) + D::seed(q[4].exp(), 4, q[4].exp()) * (0.37 * PI / 180.);
        let e = Electron::new(te, 38e-6, angle, 1000.);
        spectrum(ne, ti, te, velocity, 8500., &e, [0.8, 0.1, 0.1, 0.])
    }
    #[test]
    fn near_perpendicular_six_direction_ad_matches_finite_difference() {
        let q = [
            1e12f64.ln(),
            1200f64.ln(),
            1800f64.ln(),
            0.03,
            1.2f64.ln(),
            0.2,
        ];
        let a = point(q);
        for k in 0..6 {
            let mut p = q;
            let mut m = q;
            p[k] += 1e-5;
            m[k] -= 1e-5;
            let fd = (point(p).v - point(m).v) / 2e-5;
            let err = (a.d[k] - fd).abs() / fd.abs().max(a.v * 1e-6);
            assert!(err < 2e-5, "field{k} AD{} FD{fd} relative{err}", a.d[k]);
        }
    }
}
