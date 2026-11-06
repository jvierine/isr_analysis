//! Optional joint physical profile inversion with Student-t gradient priors.
//! Kernels map each range's spectrum into actual measured complex lag products.
use crate::{
    spectrum::{PlasmaState, SpectrumModel},
    Complex as C,
};
use anyhow::{ensure, Result};
#[path="full_profile_smooth.rs"]
pub mod smooth;
use nalgebra::{DMatrix as M, DVector as V};
use rayon::prelude::*;
use serde::{Deserialize, Serialize};
#[derive(Clone, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct Gate {
    pub altitude_km: f64,
    /// If present, hold Debye shielding at this prescribed density while the
    /// first optimizer coordinate controls only a nuisance amplitude. Such
    /// outputs are NOT calibrated electron-density estimates.
    #[serde(default)]
    pub shape_density_m3: Option<f64>,
    pub model: SpectrumModel,
    pub initial: PlasmaState,
    pub tie_temperatures: bool,
    /// Optional fitted fields in Ne, Ti, Te, velocity order. False keeps the
    /// supplied initial value prescribed, not a fitted measurement.
    #[serde(default)]
    pub free_parameters: Option<[bool; 4]>,
}
#[derive(Clone, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct Response {
    pub gate: usize,
    /// Includes frequency quadrature and absolute calibration.
    pub kernel: Vec<[f64; 2]>,
}
#[derive(Clone, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct Measurement {
    pub lag_s: f64,
    pub value: [f64; 2],
    pub background: [f64; 2],
    pub sigma: [f64; 2],
    pub ranges: Vec<Response>,
}
#[derive(Clone, Serialize, Deserialize)]
#[serde(default, deny_unknown_fields)]
pub struct Settings {
    pub smooth: Option<smooth::SmoothSettings>,
    pub max_iterations: usize,
    pub gradient_tolerance: f64,
    pub student_df: f64,
    pub hyper_shape: f64,
    /// Prior mean variances of changes per 10 km: ln Ne, ln Ti, ln Te, velocity/1000 m/s.
    pub prior_gradient_variance: [f64; 4],
    /// Fixed local gradient correlation scales in km, one row per gate and
    /// field (ln Ne, ln Ti, ln Te, velocity). None preserves legacy 10 km.
    pub correlation_length_km: Option<Vec<[f64; 4]>>,
    /// Permit interior cells constrained only by adjacent profile priors.
    /// Unsupported boundary cells still fail; output is conditional on the prior.
    pub allow_prior_only_interior: bool,
    /// Requires trapezoidal frequency weights in the supplied response kernels.
    pub integrate_spectrum_cells: bool,
}
impl Default for Settings {
    fn default() -> Self {
        Self {
            smooth: None,
            max_iterations: 120,
            gradient_tolerance: 1e-4,
            student_df: 3.,
            hyper_shape: 2.,
            prior_gradient_variance: [0.25, 0.04, 0.04, 0.01],
            correlation_length_km: None,
            allow_prior_only_interior: false,
            integrate_spectrum_cells: false,
        }
    }
}
#[derive(Clone, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct Input {
    pub schema: String,
    pub response_provenance: String,
    pub calibration_provenance: String,
    pub measurement_units: String,
    pub frequency_hz: Vec<f64>,
    pub gates: Vec<Gate>,
    pub measurements: Vec<Measurement>,
    /// Full interleaved Re/Im covariance; absent means explicitly diagonal errors.
    pub covariance: Option<Vec<Vec<f64>>>,
    pub settings: Settings,
}
#[derive(Serialize, Deserialize)]
pub struct Output {
    #[serde(default)]
    pub smooth_profile: Option<smooth::Representation>,
    pub schema: String,
    pub status: String,
    pub plasma: Vec<PlasmaState>,
    pub predicted: Vec<[f64; 2]>,
    pub objective_history: Vec<f64>,
    pub chi_square: f64,
    pub inferred_gradient_variance: [f64; 4],
    pub edge_precision: Vec<[f64; 4]>,
    pub gradient_max: f64,
    pub parameter_order: Vec<(usize, usize)>,
    /// Conditional IRLS curvature, NOT marginal posterior covariance.
    pub conditional_standard_error: Vec<Option<f64>>,
    pub uncertainty: String,
    pub scientific_acceptance: String,
}
enum Whitening {
    Diagonal(V<f64>),
    Dense(M<f64>),
}
impl Whitening {
    fn apply(&self, v: V<f64>) -> V<f64> {
        match self {
            Self::Diagonal(w) => v.component_mul(w),
            Self::Dense(w) => w * v,
        }
    }
}
struct Edge {
    field: usize,
    a: usize,
    b: usize,
    scale: f64,
}
fn state(g: &Gate, x: &V<f64>, ids: &[(usize, usize)], gate: usize) -> PlasmaState {
    let mut p = g.initial;
    for (i, &(k, f)) in ids.iter().enumerate() {
        if k != gate {
            continue;
        }
        match f {
            0 => p.electron_density_m3 = 1e11 * x[i].exp(),
            1 => p.ion_temperature_k = 1000. * x[i].exp(),
            2 => p.electron_temperature_k = 1000. * x[i].exp(),
            3 => p.velocity_m_s = 1000. * x[i],
            _ => unreachable!(),
        }
    }
    if g.tie_temperatures {
        p.electron_temperature_k = p.ion_temperature_k;
    }
    p
}
fn physical_spectrum(input:&Input,g:&Gate,mut p:PlasmaState)->Result<Vec<f64>> {
 if let Some(ne)=g.shape_density_m3 {ensure!(ne.is_finite()&&ne>0.,"invalid prescribed shape density");p.electron_density_m3=ne;}
 if input.settings.integrate_spectrum_cells {g.model.evaluate_integrated(p,&input.frequency_hz)} else {g.model.evaluate(p,&input.frequency_hz)}
}
fn prediction(input: &Input, x: &V<f64>, ids: &[(usize, usize)]) -> Result<V<f64>> {
    let spectra = input
        .gates
        .par_iter()
        .enumerate()
        .map(|(i, g)| {
            let p = state(g, x, ids, i);
            Ok(physical_spectrum(input,g,p)?
                .iter()
                .map(|s| s * p.electron_density_m3)
                .collect::<Vec<_>>())
        })
        .collect::<Result<Vec<_>>>()?;
    Ok(V::from_iterator(
        2 * input.measurements.len(),
        input.measurements.iter().flat_map(|m| {
            let mut z = C::new(0., 0.);
            for r in &m.ranges {
                for (k, s) in r.kernel.iter().zip(&spectra[r.gate]) {
                    z += C::new(k[0], k[1]) * s;
                }
            }
            [z.re, z.im]
        }),
    ))
}
fn prior(x: &V<f64>, edges: &[Edge], v: &[f64; 4], s: &Settings) -> f64 {
    let mut cost = 0.;
    for e in edges {
        let d = (x[e.b] - x[e.a]) * e.scale;
        cost += 0.5 * v[e.field].ln()
            + 0.5 * (s.student_df + 1.) * (d * d / (s.student_df * v[e.field])).ln_1p();
    }
    for k in 0..4 {
        cost += (s.hyper_shape + 1.) * v[k].ln()
            + (s.hyper_shape - 1.) * s.prior_gradient_variance[k] / v[k];
    }
    cost
}
fn update_hyper(x: &V<f64>, edges: &[Edge], v: &mut [f64; 4], s: &Settings) {
    // Bounded scalar minimization of each proper inverse-gamma hyperposterior.
    for k in 0..4 {
        let old = v[k];
        let mut lo = -24.;
        let mut hi = 12.;
        for _ in 0..80 {
            let a = lo + (hi - lo) / 3.;
            let b = hi - (hi - lo) / 3.;
            v[k] = f64::exp(a);
            let ca = prior(x, edges, v, s);
            v[k] = f64::exp(b);
            let cb = prior(x, edges, v, s);
            if ca < cb {
                hi = b
            } else {
                lo = a
            }
        }
        v[k] = ((lo + hi) / 2.).exp();
        let new = prior(x, edges, v, s);
        let candidate = v[k];
        v[k] = old;
        if new < prior(x, edges, v, s) {
            v[k] = candidate;
        }
    }
}
fn edge_weight(d: f64, v: f64, nu: f64) -> f64 {
    (nu + 1.) / (nu * v + d * d)
}
fn local_derivative(input: &Input, x: &V<f64>, ids: &[(usize, usize)], k: usize) -> Result<V<f64>> {
    let cell = ids[k].0;
    let gate = &input.gates[cell];
    let mut a = x.clone();
    let mut b = x.clone();
    a[k] += 1e-4;
    b[k] -= 1e-4;
    let pa = state(gate, &a, ids, cell);
    let pb = state(gate, &b, ids, cell);
    let sa = physical_spectrum(input,gate,pa)?;
    let sb = physical_spectrum(input,gate,pb)?;
    let ds: Vec<_> = sa
        .iter()
        .zip(sb)
        .map(|(a, b)| (a * pa.electron_density_m3 - b * pb.electron_density_m3) / 2e-4)
        .collect();
    Ok(V::from_iterator(
        2 * input.measurements.len(),
        input.measurements.iter().flat_map(|m| {
            let z = m
                .ranges
                .iter()
                .filter(|r| r.gate == cell)
                .flat_map(|r| r.kernel.iter().zip(&ds))
                .map(|(k, s)| C::new(k[0], k[1]) * s)
                .sum::<C>();
            [z.re, z.im]
        }),
    ))
}
fn linearize(
    input: &Input,
    x: &V<f64>,
    ids: &[(usize, usize)],
    w: &Whitening,
    edges: &[Edge],
    v: &[f64; 4],
    residual: &V<f64>,
) -> Result<(M<f64>, V<f64>)> {
    // A parameter changes one cell's spectrum, not all other cells. Contract
    // only that derivative through the shared response and retain the complete
    // covariance whitening. Rayon uses the campaign's bounded worker pool.
    let columns = (0..x.len())
        .into_par_iter()
        .map(|k| Ok(w.apply(local_derivative(input, x, ids, k)?)))
        .collect::<Result<Vec<_>>>()?;
    let j = M::from_columns(&columns);
    let mut h = j.transpose() * &j;
    let mut grad = j.transpose() * residual;
    for e in edges {
        let d = (x[e.b] - x[e.a]) * e.scale;
        let weight = edge_weight(d, v[e.field], input.settings.student_df);
        let curvature = weight * e.scale * e.scale;
        h[(e.a, e.a)] += curvature;
        h[(e.b, e.b)] += curvature;
        h[(e.a, e.b)] -= curvature;
        h[(e.b, e.a)] -= curvature;
        grad[e.a] -= weight * d * e.scale;
        grad[e.b] += weight * d * e.scale;
    }
    Ok((h, grad))
}
/// Residual curvature omitted by Gauss-Newton. The forward model is a sum
/// over cells, so second derivatives couple parameters within a cell only,
/// even when the likelihood covariance couples every measurement.
fn residual_curvature(
    input: &Input,
    x: &V<f64>,
    ids: &[(usize, usize)],
    w: &Whitening,
    residual: &V<f64>,
) -> Result<M<f64>> {
    let score = match w {
        Whitening::Diagonal(d) => residual.component_mul(d),
        Whitening::Dense(a) => a.transpose() * residual,
    };
    let blocks = (0..input.gates.len())
        .into_par_iter()
        .map(|cell| -> Result<Vec<(usize, usize, f64)>> {
            let mut sensitivity = vec![0.; input.frequency_hz.len()];
            for (i, m) in input.measurements.iter().enumerate() {
                for r in m.ranges.iter().filter(|r| r.gate == cell) {
                    for (v, k) in sensitivity.iter_mut().zip(&r.kernel) {
                        *v += score[2 * i] * k[0] + score[2 * i + 1] * k[1];
                    }
                }
            }
            let scalar = |x: &V<f64>| -> Result<f64> {
                let p = state(&input.gates[cell], x, ids, cell);
                Ok(physical_spectrum(input,&input.gates[cell],p)?
                    .iter()
                    .zip(&sensitivity)
                    .map(|(a, b)| a * b * p.electron_density_m3)
                    .sum())
            };
            let indices: Vec<_> = ids
                .iter()
                .enumerate()
                .filter_map(|(i, k)| (k.0 == cell).then_some(i))
                .collect();
            let base = scalar(x)?;
            let epsilon = 1e-3;
            let mut result = vec![];
            for (a, &i) in indices.iter().enumerate() {
                for &j in &indices[a..] {
                    let value = if i == j {
                        let mut p = x.clone();
                        let mut m = x.clone();
                        p[i] += epsilon;
                        m[i] -= epsilon;
                        (scalar(&p)? - 2. * base + scalar(&m)?) / (epsilon * epsilon)
                    } else {
                        let mut total = 0.;
                        for (si, sj) in [(1., 1.), (-1., -1.), (1., -1.), (-1., 1.)] {
                            let mut q = x.clone();
                            q[i] += si * epsilon;
                            q[j] += sj * epsilon;
                            total += si * sj * scalar(&q)?;
                        }
                        total / (4. * epsilon * epsilon)
                    };
                    result.push((i, j, value));
                }
            }
            Ok(result)
        })
        .collect::<Result<Vec<_>>>()?;
    let mut h = M::zeros(x.len(), x.len());
    for block in blocks {
        for (i, j, v) in block {
            h[(i, j)] = v;
            h[(j, i)] = v;
        }
    }
    Ok(h)
}
/// Replace the IRLS prior majorizer by the curvature of the profiled
/// Student-t / inverse-gamma objective. Eliminating the four log-variances
/// requires the Schur complement; holding them fixed makes flat directions
/// artificially stiff. The damped solve below handles indefinite curvature.
fn profile_prior_curvature(h: &mut M<f64>, x: &V<f64>, edges: &[Edge], v: &[f64; 4], s: &Settings) {
    for field in 0..4 {
        let mut cross = V::<f64>::zeros(x.len());
        let mut hyper = (s.hyper_shape - 1.) * s.prior_gradient_variance[field] / v[field];
        for e in edges.iter().filter(|e| e.field == field) {
            let d = (x[e.b] - x[e.a]) * e.scale;
            let nv = s.student_df * v[field];
            let denominator = nv + d * d;
            let weight = edge_weight(d, v[field], s.student_df);
            let correction = -2. * weight * d * d / denominator * e.scale * e.scale;
            h[(e.a, e.a)] += correction;
            h[(e.b, e.b)] += correction;
            h[(e.a, e.b)] -= correction;
            h[(e.b, e.a)] -= correction;
            let mixed = -weight * d * e.scale * nv / denominator;
            cross[e.a] -= mixed;
            cross[e.b] += mixed;
            hyper += 0.5 * (s.student_df + 1.) * nv * d * d / denominator.powi(2);
        }
        *h -= (&cross * cross.transpose()) / hyper;
    }
}
pub fn solve(input: &Input) -> Result<Output> {
    if let Some(s)=&input.settings.smooth{return smooth::solve(input,s);}
    let s = &input.settings;
    ensure!(
        input.schema == "ruisdap-full-profile-input-v1",
        "unsupported profile input schema"
    );
    ensure!(
        !input.response_provenance.trim().is_empty()
            && !input.calibration_provenance.trim().is_empty()
            && !input.measurement_units.trim().is_empty(),
        "response, calibration and units required"
    );
    ensure!(
        input.gates.len() >= 2 && !input.measurements.is_empty(),
        "need a profile and measurements"
    );
    ensure!(
        input.frequency_hz.len() > 2
            && input.frequency_hz.iter().all(|x| x.is_finite())
            && input.frequency_hz.windows(2).all(|w| w[1] > w[0]),
        "invalid frequency grid"
    );
    ensure!(
        s.max_iterations > 0
            && s.gradient_tolerance.is_finite()
            && s.gradient_tolerance > 0.
            && s.student_df.is_finite()
            && s.student_df > 0.
            && s.hyper_shape.is_finite()
            && s.hyper_shape > 1.
            && s.prior_gradient_variance
                .iter()
                .all(|v| v.is_finite() && *v > 0.),
        "invalid prior/solver settings"
    );
    ensure!(
        input.gates.iter().all(|g| g.altitude_km.is_finite())
            && input
                .gates
                .windows(2)
                .all(|w| w[1].altitude_km > w[0].altitude_km),
        "altitudes must increase"
    );
    let mut ids = vec![];
    let mut initial = vec![];
    let mut bounds = vec![];
    for (g, gate) in input.gates.iter().enumerate() {
        gate.model.validate()?;
        let p = gate.initial;
        ensure!(
            [
                p.electron_density_m3,
                p.ion_temperature_k,
                p.electron_temperature_k,
                p.velocity_m_s,
                p.ion_collision_s
            ]
            .iter()
            .all(|v| v.is_finite())
                && p.electron_density_m3 > 0.
                && p.ion_temperature_k > 0.
                && p.electron_temperature_k > 0.
                && p.ion_collision_s >= 0.,
            "invalid initial physical state"
        );
        for f in 0..4 {
            if (f == 2 && gate.tie_temperatures) || !gate.free_parameters.unwrap_or([true; 4])[f] {
                continue;
            }
            ids.push((g, f));
            let (value, bound) = match f {
                0 => (
                    (p.electron_density_m3 / 1e11).ln(),
                    ((1e7f64 / 1e11).ln(), (1e14f64 / 1e11).ln()),
                ),
                1 => (
                    (p.ion_temperature_k / 1000.).ln(),
                    (0.08f64.ln(), 10f64.ln()),
                ),
                2 => (
                    (p.electron_temperature_k / 1000.).ln(),
                    (0.08f64.ln(), 20f64.ln()),
                ),
                _ => (p.velocity_m_s / 1000., (-3., 3.)),
            };
            initial.push(value.clamp(bound.0, bound.1));
            bounds.push(bound);
        }
    }
    ensure!(!ids.is_empty(), "profile has no free parameters");
    if let Some(lengths)=&s.correlation_length_km {
        ensure!(lengths.len()==input.gates.len() && lengths.iter().flatten().all(|v|v.is_finite()&&*v>0.),"invalid per-gate correlation lengths");
    }
    let mut edges = vec![];
    for g in 1..input.gates.len() {
        for f in 0..4 {
            let a = ids.iter().position(|v| *v == (g - 1, f));
            let b = ids.iter().position(|v| *v == (g, f));
            if let (Some(a), Some(b)) = (a, b) {
                edges.push(Edge {
                    field: f,
                    a,
                    b,
                    scale: s.correlation_length_km.as_ref().map_or(10.,|lengths| {
                        // Harmonic mean: a short-scale region does not inherit
                        // a stronger smoothing scale from its neighbour.
                        2./(1./lengths[g-1][f]+1./lengths[g][f])
                    }) / (input.gates[g].altitude_km - input.gates[g - 1].altitude_km),
                });
            }
        }
    }
    let dim = 2 * input.measurements.len();
    let mut response_active = vec![false; input.gates.len()];
    for m in &input.measurements {
        ensure!(
            m.lag_s.is_finite()
                && m.lag_s >= 0.
                && m.value.iter().chain(&m.background).all(|v| v.is_finite())
                && m.sigma.iter().all(|v| v.is_finite() && *v > 0.),
            "invalid measurements"
        );
        let mut seen = std::collections::BTreeSet::new();
        for r in &m.ranges {
            ensure!(
                r.gate < input.gates.len()
                    && seen.insert(r.gate)
                    && r.kernel.len() == input.frequency_hz.len()
                    && r.kernel.iter().flatten().all(|v| v.is_finite()),
                "invalid or duplicate range response"
            );
            response_active[r.gate] |= r.kernel.iter().flatten().any(|v| *v != 0.);
        }
    }
    ensure!(response_active.iter().enumerate().all(|(g,v)|*v || (s.allow_prior_only_interior && response_active[..g].iter().any(|v|*v) && response_active[g+1..].iter().any(|v|*v))),"a range has no measured response; only explicitly enabled interior prior interpolation is supported");
    let w = if let Some(c) = &input.covariance {
        ensure!(
            c.len() == dim
                && c.iter()
                    .all(|r| r.len() == dim && r.iter().all(|v| v.is_finite())),
            "invalid full covariance"
        );
        let c = M::from_fn(dim, dim, |i, j| c[i][j]);
        ensure!(
            (&c - c.transpose()).amax() <= 1e-10 * c.amax().max(1e-300),
            "asymmetric covariance"
        );
        Whitening::Dense(
            c.cholesky()
                .ok_or_else(|| anyhow::anyhow!("covariance is not positive definite"))?
                .l()
                .try_inverse()
                .ok_or_else(|| anyhow::anyhow!("singular whitening"))?,
        )
    } else {
        Whitening::Diagonal(V::from_iterator(
            dim,
            input
                .measurements
                .iter()
                .flat_map(|m| [1. / m.sigma[0], 1. / m.sigma[1]]),
        ))
    };
    let observed = V::from_iterator(
        dim,
        input
            .measurements
            .iter()
            .flat_map(|m| [m.value[0] - m.background[0], m.value[1] - m.background[1]]),
    );
    let mut x = V::from_vec(initial);
    let mut variance = s.prior_gradient_variance;
    update_hyper(&x, &edges, &mut variance, s);
    let mut residual = w.apply(prediction(input, &x, &ids)? - &observed);
    let mut objective = 0.5 * residual.norm_squared() + prior(&x, &edges, &variance, s);
    let mut history = vec![objective];
    let mut damping = 1e-3;
    let mut status = "not_converged";
    for _ in 0..s.max_iterations {
        let (mut h, g) = linearize(input, &x, &ids, &w, &edges, &variance, &residual)?;
        profile_prior_curvature(&mut h, &x, &edges, &variance, s);
        h += residual_curvature(input, &x, &ids, &w, &residual)?;
        if g.amax() < s.gradient_tolerance {
            status = "optimizer_converged";
            break;
        }
        let mut damped = h.clone();
        for k in 0..x.len() {
            damped[(k, k)] += damping * h[(k, k)].abs().max(1.);
        }
        let Some(chol) = damped.cholesky() else {
            damping *= 10.;
            continue;
        };
        let step = chol.solve(&(-g));
        let mut candidate = &x + step;
        for i in 0..x.len() {
            candidate[i] = candidate[i].clamp(bounds[i].0, bounds[i].1);
        }
        let r = w.apply(prediction(input, &candidate, &ids)? - &observed);
        let mut v = variance;
        update_hyper(&candidate, &edges, &mut v, s);
        let cost = 0.5 * r.norm_squared() + prior(&candidate, &edges, &v, s);
        if cost.is_finite() && cost < objective {
            x = candidate;
            variance = v;
            residual = r;
            objective = cost;
            history.push(cost);
            damping = (damping / 3.).max(1e-10);
        } else {
            damping *= 10.;
            if damping > 1e16 {
                break;
            }
        }
    }
    let (h, g) = linearize(input, &x, &ids, &w, &edges, &variance, &residual)?;
    if g.amax() < s.gradient_tolerance {
        status = "optimizer_converged";
    }
    if x.iter()
        .enumerate()
        .any(|(i, v)| (*v - bounds[i].0).abs() < 1e-5 || (*v - bounds[i].1).abs() < 1e-5)
    {
        status = "parameter_boundary";
    }
    let covariance = h.cholesky().map(|c| c.inverse());
    let mut errors = vec![None; x.len()];
    if let Some(c) = covariance {
        for (i, &(gate, f)) in ids.iter().enumerate() {
            let p = state(&input.gates[gate], &x, &ids, gate);
            let scale = match f {
                0 => p.electron_density_m3,
                1 => p.ion_temperature_k,
                2 => p.electron_temperature_k,
                _ => 1000.,
            };
            errors[i] = Some(c[(i, i)].max(0.).sqrt() * scale);
        }
    } else {
        status = "singular";
    }
    let mut precision = vec![[0.; 4]; input.gates.len() - 1];
    for e in &edges {
        precision[ids[e.b].0 - 1][e.field] =
            edge_weight((x[e.b] - x[e.a]) * e.scale, variance[e.field], s.student_df);
    }
    let pred = prediction(input, &x, &ids)?;
    Ok(Output{smooth_profile:None,schema:"ruisdap-full-profile-map-v1".into(),status:status.into(),plasma:input.gates.iter().enumerate().map(|(i,g)|state(g,&x,&ids,i)).collect(),predicted:pred.as_slice().chunks_exact(2).map(|v|[v[0],v[1]]).collect(),objective_history:history,chi_square:residual.norm_squared(),inferred_gradient_variance:variance,edge_precision:precision,gradient_max:g.amax(),parameter_order:ids,conditional_standard_error:errors,uncertainty:"Conditional IRLS/Gauss-Newton curvature at the hierarchical MAP. Not marginal posterior intervals; excludes hyperparameter, model and absolute-calibration uncertainty. Student-t objective is nonconvex; compare multiple starts.".into(),scientific_acceptance:"experimental_not_accepted".into()})
}
#[cfg(test)]
mod tests {
    use super::*;
    use crate::spectrum::IonSpecies;
    #[test]
    fn constant_local_scales_preserve_original_fit_and_invalid_scales_fail() {
        let (input,_)=fixture();let original=solve(&input).unwrap();
        let mut local=input.clone();local.settings.correlation_length_km=Some(vec![[10.;4];local.gates.len()]);
        let result=solve(&local).unwrap();
        assert!((result.chi_square-original.chi_square).abs()<1e-12);
        for (a,b) in result.plasma.iter().zip(&original.plasma){assert!((a.electron_density_m3/b.electron_density_m3-1.).abs()<1e-12);}
        local.settings.correlation_length_km=Some(vec![[f64::NAN;4];local.gates.len()]);assert!(solve(&local).is_err());
    }
    #[test]
    fn explicit_interior_gap_is_prior_constrained_but_boundary_gap_fails() {
        let (mut input,truth)=fixture();
        for g in &mut input.gates{g.initial=truth[0];g.free_parameters=Some([true,false,false,false]);}
        let spectrum=input.gates[0].model.evaluate(truth[0],&input.frequency_hz).unwrap();
        for m in &mut input.measurements{
            m.ranges.retain(|r|r.gate!=1);
            let z:C=m.ranges.iter().flat_map(|r|r.kernel.iter().zip(&spectrum)).map(|(k,s)|C::new(k[0],k[1])*s*truth[0].electron_density_m3).sum();m.value=[z.re,z.im];
        }
        assert!(solve(&input).is_err());input.settings.allow_prior_only_interior=true;
        let result=solve(&input).unwrap();assert!((result.plasma[1].electron_density_m3/1e11-1.).abs()<0.01);
        for m in &mut input.measurements{m.ranges.retain(|r|r.gate!=0);}
        assert!(solve(&input).is_err());
    }
    #[test]
    fn spline_recovers_smooth_physical_profile_from_signed_mixed_lags() {
        let (mut input,_)=fixture();
        input.settings.correlation_length_km=Some(vec![[30.;4];4]);
        input.settings.smooth=Some(smooth::SmoothSettings::default());
        input.settings.gradient_tolerance=1e-5;
        for g in &mut input.gates {g.tie_temperatures=false;g.initial.electron_density_m3=2e11;g.initial.ion_temperature_k=1000.;g.initial.electron_temperature_k=1600.;g.initial.velocity_m_s=40.;}
        let ids:Vec<_>=(0..4).flat_map(|g|(0..4).map(move|f|(g,f))).collect();
        let x=V::from_iterator(16,(0..4).flat_map(|_|[2_f64.ln(),0.,1.6_f64.ln(),0.04]));
        let pred=prediction(&input,&x,&ids).unwrap();
        for(i,m)in input.measurements.iter_mut().enumerate(){m.value=[pred[2*i],pred[2*i+1]];}
        for g in &mut input.gates{g.initial.electron_density_m3=1.6e11;g.initial.ion_temperature_k=900.;g.initial.electron_temperature_k=1400.;g.initial.velocity_m_s=0.;}
        let fit=solve(&input).unwrap();assert_eq!(fit.status,"optimizer_converged");
        for p in fit.plasma{assert!((p.electron_density_m3/2e11-1.).abs()<0.02);assert!((p.ion_temperature_k/1000.-1.).abs()<0.02);assert!((p.electron_temperature_k/1600.-1.).abs()<0.02);assert!((p.velocity_m_s-40.).abs()<2.);}
    }
    fn fixture() -> (Input, Vec<PlasmaState>) {
        let f: Vec<_> = (-80..=80).map(|i| i as f64 * 250.).collect();
        let model = SpectrumModel {
            radar_frequency_hz: 930e6,
            ions: vec![IonSpecies {
                mass_amu: 16.,
                fraction: 1.,
            }],
            electron_collision_s: 0.,
        };
        let truth: Vec<_> = (0..4)
            .map(|i| PlasmaState {
                electron_density_m3: if i < 2 { 1e11 } else { 4e11 },
                ion_temperature_k: 900.,
                electron_temperature_k: 900.,
                velocity_m_s: 50.,
                ion_collision_s: 20.,
            })
            .collect();
        let gates = truth
            .iter()
            .enumerate()
            .map(|(i, p)| Gate {
                shape_density_m3: None,
                altitude_km: 150. + 10. * i as f64,
                model: model.clone(),
                initial: PlasmaState {
                    electron_density_m3: 2e11,
                    ion_temperature_k: 1100.,
                    velocity_m_s: 0.,
                    ..*p
                },
                tie_temperatures: true,
                free_parameters: None,
            })
            .collect();
        let spectra: Vec<_> = truth
            .iter()
            .map(|p| model.evaluate(*p, &f).unwrap())
            .collect();
        let mut measurements = vec![];
        // A genuine coupled response, including signed sidelobes and irregular lags.
        for row in 0usize..4 {
            for lag in [10., 30., 50., 80., 120., 160., 220., 310., 410.] {
                let t = lag * 1e-6;
                let mut ranges = vec![];
                let mut value = C::new(0., 0.);
                for g in 0..4 {
                    let weight = if row == g {
                        0.8
                    } else if row.abs_diff(g) == 1 {
                        0.15
                    } else {
                        -0.025
                    };
                    let kernel: Vec<_> = f
                        .iter()
                        .map(|freq| {
                            let z =
                                C::from_polar(250e-11 * weight, -std::f64::consts::TAU * freq * t);
                            [z.re, z.im]
                        })
                        .collect();
                    for (k, s) in kernel.iter().zip(&spectra[g]) {
                        value += C::new(k[0], k[1]) * (*s * truth[g].electron_density_m3);
                    }
                    ranges.push(Response { gate: g, kernel });
                }
                measurements.push(Measurement {
                    lag_s: t,
                    value: [value.re, value.im],
                    background: [0.; 2],
                    sigma: [0.001; 2],
                    ranges,
                });
            }
        }
        (
            Input {
                schema: "ruisdap-full-profile-input-v1".into(),
                response_provenance: "synthetic signed ambiguity".into(),
                calibration_provenance: "synthetic absolute".into(),
                measurement_units: "K".into(),
                frequency_hz: f,
                gates,
                measurements,
                covariance: None,
                settings: Settings {
                    max_iterations: 160,
                    gradient_tolerance: 0.005,
                    ..Settings::default()
                },
            },
            truth,
        )
    }
    #[test]
    fn native_amplitude_does_not_change_prescribed_debye_shape() {
        let (mut input, _) = fixture();
        for g in &mut input.gates {g.shape_density_m3=Some(1e12);}
        let g=&input.gates[0];let p=g.initial;let mut other=p;other.electron_density_m3*=100.;
        assert_eq!(physical_spectrum(&input,g,p).unwrap(),physical_spectrum(&input,g,other).unwrap());
        let ids:Vec<_>=(0..4).map(|i|(i,0)).collect();
        let x=V::zeros(4);let shifted=V::from_element(4,2f64.ln());
        let a=prediction(&input,&x,&ids).unwrap();let b=prediction(&input,&shifted,&ids).unwrap();
        assert!((b-2.*a).norm()<1e-12);
        input.gates[0].shape_density_m3=Some(-1.);
        assert!(physical_spectrum(&input,&input.gates[0],p).is_err());
    }
    #[test]
    fn prescribed_fields_remain_fixed_while_density_is_recovered() {
        let (mut input, truth) = fixture();
        for (g, p) in input.gates.iter_mut().zip(&truth) {
            g.initial = *p;
            g.initial.electron_density_m3 = 2e11;
            g.free_parameters = Some([true, false, false, false]);
        }
        let out = solve(&input).unwrap();
        assert_eq!(out.status, "optimizer_converged");
        assert!(out.parameter_order.iter().all(|(_, f)| *f == 0));
        for (p, t) in out.plasma.iter().zip(truth) {
            assert_eq!(p.ion_temperature_k, t.ion_temperature_k);
            assert_eq!(p.electron_temperature_k, t.electron_temperature_k);
            assert_eq!(p.velocity_m_s, t.velocity_m_s);
            assert!((p.electron_density_m3 / t.electron_density_m3 - 1.).abs() < 0.03);
        }
    }
    #[test]
    fn residual_curvature_matches_likelihood_hessian() {
        let (input, _) = fixture();
        let ids: Vec<_> = (0..input.gates.len())
            .flat_map(|g| [0, 1, 3].map(|f| (g, f)))
            .collect();
        let x = V::zeros(ids.len());
        let w = Whitening::Diagonal(V::from_element(input.measurements.len() * 2, 1000.));
        let observed = V::from_iterator(
            input.measurements.len() * 2,
            input.measurements.iter().flat_map(|m| m.value),
        );
        let r = w.apply(prediction(&input, &x, &ids).unwrap() - &observed);
        let (jtj, _) = linearize(
            &input,
            &x,
            &ids,
            &w,
            &[],
            &input.settings.prior_gradient_variance,
            &r,
        )
        .unwrap();
        let h = jtj + residual_curvature(&input, &x, &ids, &w, &r).unwrap();
        for k in 0..ids.len() {
            let mut a = x.clone();
            let mut b = x.clone();
            a[k] += 2e-4;
            b[k] -= 2e-4;
            let gradient = |q: &V<f64>| {
                let r = w.apply(prediction(&input, q, &ids).unwrap() - &observed);
                linearize(
                    &input,
                    q,
                    &ids,
                    &w,
                    &[],
                    &input.settings.prior_gradient_variance,
                    &r,
                )
                .unwrap()
                .1
            };
            let numeric = (gradient(&a) - gradient(&b)) / 4e-4;
            assert!((h.column(k) - &numeric).norm() < 1e-4 * numeric.norm().max(1.));
        }
    }
    #[test]
    fn profiled_prior_curvature_matches_numerical_gradient() {
        let s = Settings::default();
        let edges = vec![
            Edge {
                field: 0,
                a: 0,
                b: 1,
                scale: 1.,
            },
            Edge {
                field: 0,
                a: 1,
                b: 2,
                scale: 0.5,
            },
        ];
        let x = V::from_vec(vec![0., 0.2, 1.7]);
        let gradient = |x: &V<f64>| {
            let mut v = s.prior_gradient_variance;
            update_hyper(x, &edges, &mut v, &s);
            let mut g = V::zeros(x.len());
            for e in &edges {
                let d = (x[e.b] - x[e.a]) * e.scale;
                let q = edge_weight(d, v[e.field], s.student_df) * d * e.scale;
                g[e.a] -= q;
                g[e.b] += q;
            }
            (g, v)
        };
        let (_, v) = gradient(&x);
        let mut h = M::zeros(x.len(), x.len());
        for e in &edges {
            let d = (x[e.b] - x[e.a]) * e.scale;
            let q = edge_weight(d, v[e.field], s.student_df) * e.scale * e.scale;
            h[(e.a, e.a)] += q;
            h[(e.b, e.b)] += q;
            h[(e.a, e.b)] -= q;
            h[(e.b, e.a)] -= q;
        }
        profile_prior_curvature(&mut h, &x, &edges, &v, &s);
        for k in 0..x.len() {
            let mut a = x.clone();
            let mut b = x.clone();
            a[k] += 1e-3;
            b[k] -= 1e-3;
            let numeric = (gradient(&a).0 - gradient(&b).0) / 2e-3;
            assert!((h.column(k) - numeric).amax() < 2e-3);
        }
    }
    #[test]
    fn sparse_cell_derivatives_match_full_forward_difference() {
        let (input, _) = fixture();
        let ids: Vec<_> = (0..input.gates.len())
            .flat_map(|g| [0, 1, 3].map(|f| (g, f)))
            .collect();
        let x = V::zeros(ids.len());
        for k in 0..ids.len() {
            let mut a = x.clone();
            let mut b = x.clone();
            a[k] += 1e-4;
            b[k] -= 1e-4;
            let reference = (prediction(&input, &a, &ids).unwrap()
                - prediction(&input, &b, &ids).unwrap())
                / 2e-4;
            let actual = local_derivative(&input, &x, &ids, k).unwrap();
            assert!((&actual - &reference).norm() < 1e-7 * reference.norm().max(1e-12));
        }
    }
    #[test]
    fn coupled_physical_profile_recovers_edge() {
        let (input, truth) = fixture();
        let out = solve(&input).unwrap();
        eprintln!(
            "{} gradient {} history {}",
            out.status,
            out.gradient_max,
            out.objective_history.len()
        );
        assert_eq!(out.status, "optimizer_converged");
        for (p, t) in out.plasma.iter().zip(truth) {
            assert!((p.electron_density_m3 / t.electron_density_m3 - 1.).abs() < 0.03);
            assert!((p.ion_temperature_k - t.ion_temperature_k).abs() < 20.);
            assert!((p.velocity_m_s - t.velocity_m_s).abs() < 2.);
            assert_eq!(p.ion_collision_s, t.ion_collision_s);
        }
        assert!(out.objective_history.windows(2).all(|w| w[1] <= w[0]));
        assert!(out.edge_precision[1][0] < out.edge_precision[0][0] * 0.15);
        assert_ne!(
            out.inferred_gradient_variance,
            input.settings.prior_gradient_variance
        );
    }
    #[test]
    fn full_covariance_matches_diagonal() {
        let (mut input, _) = fixture();
        let a = solve(&input).unwrap();
        let n = input.measurements.len() * 2;
        input.covariance = Some(
            (0..n)
                .map(|i| (0..n).map(|j| if i == j { 1e-6 } else { 0. }).collect())
                .collect(),
        );
        let b = solve(&input).unwrap();
        for (a, b) in a.plasma.iter().zip(b.plasma) {
            assert!((a.electron_density_m3 / b.electron_density_m3 - 1.).abs() < 1e-8);
        }
    }
    #[test]
    fn correlated_likelihood_uses_off_diagonal_terms() {
        let (mut input, _) = fixture();
        for (i, m) in input.measurements.iter_mut().enumerate() {
            m.value[0] += 0.001 * (i as f64).sin();
        }
        let n = input.measurements.len() * 2;
        let c = M::from_fn(n, n, |i, j| 1e-6 * 0.35f64.powi(i.abs_diff(j) as i32));
        input.covariance = Some(
            (0..n)
                .map(|i| (0..n).map(|j| c[(i, j)]).collect())
                .collect(),
        );
        let out = solve(&input).unwrap();
        let residual = V::from_iterator(
            n,
            input
                .measurements
                .iter()
                .zip(&out.predicted)
                .flat_map(|(m, p)| [p[0] - m.value[0], p[1] - m.value[1]]),
        );
        let exact = residual.dot(&c.cholesky().unwrap().solve(&residual));
        assert!((exact - out.chi_square).abs() < 1e-8);
        assert!((exact - residual.norm_squared() / 1e-6).abs() > 0.01);
    }
    #[test]
    fn malformed_response_and_covariance_fail() {
        let (mut input, _) = fixture();
        input.measurements[0].ranges[0].kernel.pop();
        assert!(solve(&input).is_err());
        let (mut input, _) = fixture();
        let n = input.measurements.len() * 2;
        input.covariance = Some(vec![vec![0.; n]; n]);
        assert!(solve(&input).is_err());
    }
    #[test]
    fn prior_derivative_and_edge_adaptation() {
        let s = Settings::default();
        let edges = vec![Edge {
            field: 0,
            a: 0,
            b: 1,
            scale: 0.5,
        }];
        let mut x = V::from_vec(vec![0., 2.]);
        let v = s.prior_gradient_variance;
        let eps = 1e-6;
        x[1] += eps;
        let a = prior(&x, &edges, &v, &s);
        x[1] -= 2. * eps;
        let b = prior(&x, &edges, &v, &s);
        let expected = edge_weight(1., v[0], s.student_df) * 0.5;
        assert!(((a - b) / (2. * eps) - expected).abs() < 1e-7);
        assert!(edge_weight(10., v[0], s.student_df) < edge_weight(0., v[0], s.student_df) / 100.);
    }
}
