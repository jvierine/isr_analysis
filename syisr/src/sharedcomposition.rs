//! Shared composition with independent beam plasma profiles and block elimination.
//! Huber loss acts on standardized signed complex lag residuals, not fitted parameters.
use crate::{
    coupled::{encode, Node, Problem},
    h_transition::Shape,
};
use anyhow::{ensure, Result};
use nalgebra::{DMatrix as M, DVector as V};
use serde::{Deserialize, Serialize};
use std::sync::Arc;
pub const SHARED: [usize; 5] = [4, 10, 17, 23, 29];
#[derive(Clone, Serialize, Deserialize)]
pub struct TimeNeighbor {
    pub separation_s: f64,
    pub h_fraction: Vec<f64>,
}
#[derive(Clone, Serialize, Deserialize)]
pub struct Temporal {
    pub altitude_km: Vec<f64>,
    pub sigma_per_30min: f64,
    pub neighbors: Vec<TimeNeighbor>,
}
#[derive(Serialize, Deserialize)]
pub struct Solution {
    pub nodes: Vec<Vec<Node>>,
    pub shared: [f64; 5],
    pub status: String,
    pub iterations: usize,
    pub normalized_gradient: f64,
    pub objective_history: Vec<f64>,
    pub robust_objective: f64,
    pub temporal_objective: f64,
    #[serde(default)]
    pub diagonal_steps: usize,
    #[serde(default)]
    pub data_stability: Option<serde_json::Value>,
}
struct Normal {
    h: M<f64>,
    g: V<f64>,
    columns: Vec<usize>,
    residual: V<f64>,
}
fn bounds(i: usize) -> (f64, f64) {
    match i % 6 {
        0 => (-24., 24.),
        1 => (0.15f64.ln(), 6f64.ln()),
        2 => (0.15f64.ln(), 12f64.ln()),
        3 => (-2., 2.),
        _ => (0., 1.),
    }
}
fn local_columns(p: &Problem) -> Vec<usize> {
    (0..p.knot_count)
        .flat_map(|k| (0..4).map(move |f| 6 * k + f))
        .collect()
}
fn unpack(x: &V<f64>) -> Vec<Node> {
    x.as_slice()
        .chunks_exact(6)
        .map(|p| Node {
            power_adu2: p[0].exp(),
            ti_k: 1000. * p[1].exp(),
            te_k: 1000. * p[2].exp(),
            velocity_m_s: 1000. * p[3],
            h_fraction: p[4],
            he_fraction: (1. - p[4]) * p[5],
            o_fraction: (1. - p[4]) * (1. - p[5]),
        })
        .collect()
}
fn huber(r: f64) -> (f64, f64) {
    let a = r.abs();
    if a <= 3. {
        (r * r, 1.)
    } else {
        (6. * a - 9., 3. / a)
    }
}
fn data(p: &Problem, x: &V<f64>, derivatives: bool) -> Result<(f64, Option<Normal>)> {
    let (r, j) = p.evaluate(x, derivatives)?;
    let mut cost: f64 = r.iter().map(|v| huber(*v).0).sum();
    for v in &p.coordinate_priors {
        let z = (v.coefficients.iter().map(|(c, w)| x[*c] * w).sum::<f64>() - v.mean) / v.sigma;
        cost += 2. * v.delta.powi(2) * ((1. + (z / v.delta).powi(2)).sqrt() - 1.);
    }
    if !derivatives {
        return Ok((cost, None));
    }
    let mut columns = local_columns(p);
    columns.extend(SHARED);
    let mut a = M::from_fn(r.len(), columns.len(), |i, k| j[(i, columns[k])]);
    if std::env::var("SYISR_FIX_HELIUM_ZERO").ok().as_deref() == Some("1") {
        let nc = a.ncols();
        for j in nc - 3..nc {
            a.column_mut(j).fill(0.);
        }
    }
    let mut rw = r.clone();
    for i in 0..r.len() {
        let w = huber(r[i]).1.sqrt();
        rw[i] *= w;
        a.row_mut(i).scale_mut(w);
    }
    let mut h = a.transpose() * &a;
    let mut g = a.transpose() * rw;
    for v in &p.coordinate_priors {
        let terms = v
            .coefficients
            .iter()
            .map(|(c, w)| {
                columns
                    .iter()
                    .position(|k| k == c)
                    .map(|i| (i, *w))
                    .ok_or_else(|| anyhow::anyhow!("prior on unsupported coordinate"))
            })
            .collect::<Result<Vec<_>>>()?;
        let z = (v.coefficients.iter().map(|(c, w)| x[*c] * w).sum::<f64>() - v.mean) / v.sigma;
        let w = 1. / (1. + (z / v.delta).powi(2)).sqrt();
        for &(i, a) in &terms {
            g[i] += w * z * a / v.sigma;
            for &(j, b) in &terms {
                h[(i, j)] += w * a * b / v.sigma.powi(2);
            }
        }
    }
    Ok((
        cost,
        Some(Normal {
            h,
            g,
            columns,
            residual: r,
        }),
    ))
}
fn temporal(t: Option<&Temporal>, shape: &Shape, q: &[f64; 5]) -> (f64, M<f64>, V<f64>) {
    let mut h = M::zeros(5, 5);
    let mut g = V::zeros(5);
    let mut cost = 0.;
    if let Some(t) = t {
        for n in &t.neighbors {
            for (i, &z) in t.altitude_km.iter().enumerate() {
                let (v, d) = shape.evaluate(z, [q[0], q[1]]);
                let sigma = t.sigma_per_30min * (n.separation_s.abs() / 1800.).sqrt();
                let r = (v - n.h_fraction[i]) / sigma;
                let delta = std::env::var("SYISR_TEMPORAL_ROBUST_DELTA")
                    .ok()
                    .and_then(|s| s.parse::<f64>().ok());
                let w = if let Some(delta) = delta {
                    cost += 2. * delta.powi(2) * ((1. + (r / delta).powi(2)).sqrt() - 1.);
                    1. / (1. + (r / delta).powi(2)).sqrt()
                } else {
                    cost += r * r;
                    1.
                };
                for a in 0..2 {
                    g[a] += w * d[a] / sigma * r;
                    for b in 0..2 {
                        h[(a, b)] += w * d[a] * d[b] / sigma.powi(2);
                    }
                }
            }
        }
    }
    (cost, h, g)
}
/// Solve the arrow-shaped normal equations. No dense all-beam Jacobian is formed.
fn arrow(
    normals: &[Normal],
    xs: &[V<f64>],
    q: &[f64; 5],
    th: &M<f64>,
    tg: &V<f64>,
    lambda: f64,
    fixed_h: bool,
) -> Result<(Vec<V<f64>>, V<f64>, f64)> {
    let mut ch = th.clone();
    let mut cg = tg.clone();
    let mut maxscore = 0f64;
    for n in normals {
        let m = n.h.nrows() - 5;
        ch += n.h.view((m, m), (5, 5));
        cg += n.g.rows(m, 5);
    }
    let shared_blocked: Vec<_> = (0..5)
        .map(|i| {
            (fixed_h && i < 2) || (ch[(i, i)] < 1e-20)
                || (q[i] <= 1e-8 && cg[i] > 0.)
                || (q[i] >= 1. - 1e-8 && cg[i] < 0.)
        })
        .collect();
    for i in 0..5 {
        if !shared_blocked[i] {
            maxscore = maxscore.max(cg[i].abs() / ch[(i, i)].sqrt().max(1e-12));
        }
        ch[(i, i)] += lambda * ch[(i, i)].max(1e-10);
    }
    let mut inverse = vec![];
    for (n, x) in normals.iter().zip(xs) {
        let m = n.h.nrows() - 5;
        let mut a = n.h.view((0, 0), (m, m)).into_owned();
        let mut b = n.h.view((0, m), (m, 5)).into_owned();
        let mut g = n.g.rows(0, m).into_owned();
        for i in 0..m {
            let (lo, hi) = bounds(n.columns[i]);
            let v = x[n.columns[i]];
            let blocked =
                a[(i, i)] < 1e-20 || (v <= lo + 1e-8 && g[i] > 0.) || (v >= hi - 1e-8 && g[i] < 0.);
            if !blocked {
                maxscore = maxscore.max(g[i].abs() / a[(i, i)].sqrt().max(1e-12));
            }
            a[(i, i)] += lambda * a[(i, i)].max(1e-10);
            if blocked {
                a.row_mut(i).fill(0.);
                a.column_mut(i).fill(0.);
                a[(i, i)] = 1.;
                b.row_mut(i).fill(0.);
                g[i] = 0.;
            }
        }
        for k in 0..5 {
            if shared_blocked[k] {
                b.column_mut(k).fill(0.);
            }
        }
        let chol = a
            .cholesky()
            .ok_or_else(|| anyhow::anyhow!("nonpositive beam block"))?;
        let ib = chol.solve(&b);
        let ig = chol.solve(&g);
        ch -= b.transpose() * &ib;
        cg -= b.transpose() * &ig;
        inverse.push((ib, ig));
    }
    for k in 0..5 {
        if shared_blocked[k] {
            ch.row_mut(k).fill(0.);
            ch.column_mut(k).fill(0.);
            ch[(k, k)] = 1.;
            cg[k] = 0.;
        }
    }
    let ds = ch
        .cholesky()
        .ok_or_else(|| anyhow::anyhow!("nonpositive shared Schur block"))?
        .solve(&(-cg));
    let dl = inverse.into_iter().map(|(b, g)| -g - b * &ds).collect();
    Ok((dl, ds, maxscore))
}
// JSON/HDF5 round trips can change the last floating-point bit. Reject changes
// in prior configuration, while accepting roundoff in the same saved context.
/// Diagonal Gauss--Newton/Fisher proposal; projection preserves feasible descent.
fn fraction_change(shape: &Shape, a: &[f64; 5], b: &[f64; 5]) -> f64 {
    (500..=crate::lp480::configured_top_km())
        .step_by(10)
        .map(|z| {
            let z = z as f64;
            let hw = crate::spline::h_weight(z);
            let ha = shape.evaluate(z, [a[0], a[1]]).0 * hw;
            let hb = shape.evaluate(z, [b[0], b[1]]).0 * hw;
            let ea = (1. - ha) * crate::h_transition::helium(z, [a[2], a[3], a[4]]).0 * hw;
            let eb = (1. - hb) * crate::h_transition::helium(z, [b[2], b[3], b[4]]).0 * hw;
            (ha - hb).abs().max((ea - eb).abs())
        })
        .fold(0., f64::max)
}
fn diagonal_candidate(
    normals: &[Normal],
    xs: &[V<f64>],
    q: &[f64; 5],
    th: &M<f64>,
    tg: &V<f64>,
) -> (Vec<V<f64>>, [f64; 5]) {
    let mut shared_g = tg.clone();
    let mut shared_d = V::from_iterator(5, (0..5).map(|i| th[(i, i)]));
    for n in normals {
        let m = n.columns.len() - 5;
        for i in 0..5 {
            shared_g[i] += n.g[m + i];
            shared_d[i] += n.h[(m + i, m + i)];
        }
    }
    let dq = std::array::from_fn(|i| {
        if shared_d[i] > 1e-20 {
            (q[i] + (-shared_g[i] / shared_d[i]).clamp(-0.15, 0.15)).clamp(0., 1.)
        } else {
            q[i]
        }
    });
    let mut candidate = xs.to_vec();
    for (n, x) in normals.iter().zip(&mut candidate) {
        let m = n.columns.len() - 5;
        for i in 0..m {
            let k = n.columns[i];
            let radius = match k % 6 {
                0 => 0.5,
                1 | 2 => 0.3,
                _ => 0.25,
            };
            let (lo, hi) = bounds(k);
            if n.h[(i, i)] > 1e-20 {
                x[k] = (x[k] + (-n.g[i] / n.h[(i, i)]).clamp(-radius, radius)).clamp(lo, hi);
            }
        }
        for (i, k) in SHARED.iter().enumerate() {
            x[*k] = dq[i];
        }
    }
    (candidate, dq)
}
fn same_context(a: &serde_json::Value, b: &serde_json::Value) -> bool {
    match (a, b) {
        (serde_json::Value::Number(a), serde_json::Value::Number(b)) => {
            let x = a.as_f64().unwrap();
            let y = b.as_f64().unwrap();
            (x - y).abs() <= 1e-13 * x.abs().max(y.abs()).max(1.)
        }
        (serde_json::Value::Array(a), serde_json::Value::Array(b)) => {
            a.len() == b.len() && a.iter().zip(b).all(|(x, y)| same_context(x, y))
        }
        (serde_json::Value::Object(a), serde_json::Value::Object(b)) => {
            a.len() == b.len()
                && a.iter()
                    .all(|(k, v)| b.get(k).is_some_and(|w| same_context(v, w)))
        }
        _ => a == b,
    }
}
fn save_checkpoint(
    path: &std::path::Path,
    context: &serde_json::Value,
    xs: &[V<f64>],
    lambda: f64,
    history: &[f64],
    iterations: usize,
    gradient: f64,
) -> Result<()> {
    let nodes: Vec<_> = xs.iter().map(unpack).collect();
    crate::campaign::save(
        path,
        &serde_json::json!({"schema":"syisr-shared-optimizer-checkpoint-v1","context":context,"nodes":nodes,"lambda":lambda,"objective_history":history,"iterations":iterations,"normalized_gradient":if gradient.is_finite(){Some(gradient)}else{None},"status":"optimizer_checkpoint_not_accepted_fit"}),
    )
}
fn optimizer_context(
    problems: &[&Problem],
    shape: &Shape,
    prior: Option<&Temporal>,
) -> Result<serde_json::Value> {
    let context = if std::env::var("SYISR_FIX_HELIUM_ZERO").ok().as_deref() == Some("1")
        || problems.iter().any(|p| !p.coordinate_priors.is_empty())
    {
        serde_json::json!({"shape":shape,"prior":prior,"helium_fixed_zero":true,"coordinate_priors":problems.iter().map(|p|&p.coordinate_priors).collect::<Vec<_>>(),"temporal_robust_delta":std::env::var("SYISR_TEMPORAL_ROBUST_DELTA").ok()})
    } else {
        serde_json::to_value((shape, prior))?
    };
    let mut context = context;
    if std::env::var_os("SYISR_LP480_TOP_KM").is_some() {
        context["lp480_top_km"] = serde_json::json!(crate::lp480::configured_top_km());
    }
    if std::env::var_os("SYISR_EXACT_MAGNETIZED_RESPONSE").is_some() {
        context["exact_magnetized_evaluation"] = serde_json::json!(true);
    }
    Ok(context)
}
/// Exact shared objective used to compare initializations, never a substitute for refitting nuisance profiles.
pub fn objective_at(
    problems: &[&Problem],
    nodes: &[Vec<Node>],
    shape: &Shape,
    prior: Option<&Temporal>,
) -> Result<f64> {
    let mut cost = 0.;
    for (p, n) in problems.iter().zip(nodes) {
        cost += data(p, &encode(n), false)?.0;
    }
    let x = encode(&nodes[0]);
    let q = SHARED.map(|i| x[i]);
    Ok(cost + temporal(prior, shape, &q).0)
}
pub fn solve(
    problems: &[&Problem],
    initial: &[Vec<Node>],
    shape: Arc<Shape>,
    prior: Option<&Temporal>,
    max_iterations: usize,
) -> Result<Solution> {
    solve_checkpointed(problems, initial, shape, prior, max_iterations, None)
}
pub fn solve_checkpointed(
    problems: &[&Problem],
    initial: &[Vec<Node>],
    shape: Arc<Shape>,
    prior: Option<&Temporal>,
    max_iterations: usize,
    checkpoint: Option<&std::path::Path>,
) -> Result<Solution> {
    solve_internal(problems, initial, shape, prior, max_iterations, checkpoint, None)
}
/// Conditional composition profile: exactly fix H height/width coordinates,
/// refitting every beam's density, temperatures and velocity. Not an error bar
/// until likelihood/correlation calibration and width profiling are validated.
pub fn solve_fixed_h(
    problems: &[&Problem], initial: &[Vec<Node>], shape: Arc<Shape>,
    q: [f64; 2], max_iterations: usize, checkpoint: Option<&std::path::Path>,
) -> Result<Solution> {
    ensure!(q.iter().all(|v|v.is_finite()&&(0.0..=1.0).contains(v)),"invalid fixed H coordinates");
    ensure!(std::env::var_os("SYISR_DIAGONAL_STEP").is_none(),"fixed H does not support experimental diagonal proposals");
    solve_internal(problems, initial, shape, None, max_iterations, checkpoint, Some(q))
}
fn solve_internal(
    problems: &[&Problem], initial: &[Vec<Node>], shape: Arc<Shape>,
    prior: Option<&Temporal>, max_iterations: usize,
    checkpoint: Option<&std::path::Path>, fixed_h: Option<[f64; 2]>,
) -> Result<Solution> {
    ensure!(
        !problems.is_empty() && problems.len() == initial.len(),
        "beam dimensions"
    );
    shape.validate()?;
    if let Some(t) = prior {
        ensure!(
            t.sigma_per_30min > 0.
                && t.sigma_per_30min.is_finite()
                && t.neighbors.iter().all(|n| n.separation_s.is_finite()
                    && n.separation_s.abs() > 0.
                    && n.h_fraction.len() == t.altitude_km.len()
                    && n.h_fraction
                        .iter()
                        .all(|v| v.is_finite() && (0.0..=1.0).contains(v))),
            "temporal prior dimensions"
        );
    }
    let mut xs: Vec<_> = initial.iter().map(|v| encode(v)).collect();
    let mut q = SHARED.map(|c| xs[0][c]);
    if let Some(h) = fixed_h { q[..2].copy_from_slice(&h); }
    if std::env::var("SYISR_FIX_HELIUM_ZERO").ok().as_deref() == Some("1") {
        q[2] = 0.;
    }
    for x in &mut xs {
        for (i, c) in SHARED.iter().enumerate() {
            x[*c] = q[i];
        }
    }
    let mut lambda = 1e-3;
    let mut history = vec![];
    let mut status = "iteration_limit";
    let mut gradient = f64::INFINITY;
    let mut iters = 0;
    let mut context = optimizer_context(problems, &shape, prior)?;
    if let Some(h) = fixed_h { context["fixed_h_coordinates"] = serde_json::json!(h); }
    if let Some(path) = checkpoint.filter(|p| p.exists()) {
        let saved: serde_json::Value = crate::campaign::load(path)?;
        ensure!(
            same_context(&saved["context"], &context),
            "checkpoint temporal/shape context differs"
        );
        let nodes: Vec<Vec<Node>> = serde_json::from_value(saved["nodes"].clone())?;
        ensure!(
            nodes.len() == xs.len() && nodes.iter().zip(&xs).all(|(n, x)| n.len() * 6 == x.len()),
            "checkpoint dimensions differ"
        );
        xs = nodes.iter().map(|n| encode(n)).collect();
        q = SHARED.map(|c| xs[0][c]);
        lambda = saved["lambda"].as_f64().unwrap_or(1e-3);
        history = serde_json::from_value(saved["objective_history"].clone())?;
        iters = saved["iterations"].as_u64().unwrap_or(0) as usize;
        eprintln!("SHARED_RESUME iterations={iters} lambda={lambda}");
    }
    let resumed_iterations = iters;
    let mut diagonal_steps = 0;
    let mut normals = vec![];
    let mut dc = 0.;
    for (p, x) in problems.iter().zip(&xs) {
        let (c, n) = data(p, x, true)?;
        dc += c;
        normals.push(n.unwrap());
    }
    let mut tc = temporal(prior, &shape, &q).0;
    let mut cost = dc + tc;
    history.push(cost);
    let mut stable_q = q;
    let mut stable_r: Vec<_> = normals.iter().map(|n| n.residual.clone()).collect();
    let mut stable_cost = cost;
    let mut stable_intervals = 0;
    let mut data_stability = None;
    for it in 0..if max_iterations == 0 {
        usize::MAX
    } else {
        max_iterations
    } {
        iters = resumed_iterations + it + 1;
        if it % 25 == 0 {
            if let Some(path) = checkpoint {
                save_checkpoint(path, &context, &xs, lambda, &history, iters, gradient)?;
            }
        }
        let (_, th, tg) = temporal(prior, &shape, &q);
        let step = arrow(&normals, &xs, &q, &th, &tg, lambda, fixed_h.is_some());
        let (dl, ds, score) = match step {
            Ok(v) => v,
            Err(_) => {
                lambda *= 10.;
                if lambda > 1e15 {
                    status = "stalled";
                    break;
                }
                continue;
            }
        };
        gradient = score;
        if it % 5 == 0 {
            eprintln!(
                "SHARED_ITER {it} objective={cost:.6e} gradient={gradient:.5e} lambda={lambda:.3e}"
            );
        }
        if gradient < 0.03 {
            status = "converged";
            break;
        }
        let mut nq = std::array::from_fn(|i| (q[i] + ds[i].clamp(-0.15, 0.15)).clamp(0., 1.));
        let mut candidate = xs.clone();
        for (b, ((x, step), n)) in candidate.iter_mut().zip(dl).zip(&normals).enumerate() {
            for (i, c) in n.columns[..n.columns.len() - 5].iter().enumerate() {
                let radius = match c % 6 {
                    0 => 0.5,
                    1 | 2 => 0.3,
                    _ => 0.25,
                };
                let (lo, hi) = bounds(*c);
                x[*c] = (xs[b][*c] + step[i].clamp(-radius, radius)).clamp(lo, hi);
            }
            for (i, c) in SHARED.iter().enumerate() {
                x[*c] = nq[i];
            }
        }
        let mut nd = 0.;
        for (p, x) in problems.iter().zip(&candidate) {
            nd += data(p, x, false)?.0;
        }
        let mut nt = temporal(prior, &shape, &nq).0;
        let mut diagonal_selected = false;
        if std::env::var_os("SYISR_DIAGONAL_STEP").is_some()
            && it % 5 == 0
            && cost - nd - nt < 1e-6 * cost.max(1.)
        {
            let (proposal, dq) = diagonal_candidate(&normals, &xs, &q, &th, &tg);
            let mut proposal_data = 0.;
            for (p, x) in problems.iter().zip(&proposal) {
                proposal_data += data(p, x, false)?.0;
            }
            let proposal_time = temporal(prior, &shape, &dq).0;
            if proposal_data + proposal_time < (nd + nt).min(cost) {
                candidate = proposal;
                nq = dq;
                nd = proposal_data;
                nt = proposal_time;
                diagonal_selected = true;
                diagonal_steps += 1;
            }
        }
        if nd + nt < cost {
            xs = candidate;
            q = nq;
            cost = nd + nt;
            tc = nt;
            dc = nd;
            history.push(cost);
            normals.clear();
            for (p, x) in problems.iter().zip(&xs) {
                normals.push(data(p, x, true)?.1.unwrap());
            }
            if !diagonal_selected {
                lambda = (lambda / 3.).max(1e-10);
            }
        } else {
            lambda *= 8.;
            if lambda > 1e15 {
                status = "stalled";
                break;
            }
        }
        if std::env::var_os("SYISR_DATA_STABILITY_STOP").is_some() && (it + 1) % 25 == 0 {
            let mut dr2 = 0.;
            for ((p, n), old) in problems.iter().zip(&normals).zip(&stable_r) {
                for (i, product) in p.products.iter().enumerate() {
                    for k in 0..2 {
                        if product.sigma[k] < 1e20 {
                            dr2 += (n.residual[2 * i + k] - old[2 * i + k]).powi(2);
                        }
                    }
                }
            }
            let change = fraction_change(&shape, &q, &stable_q);
            let dq = (stable_cost - cost).abs();
            let prediction_change = dr2.sqrt();
            if dq < 0.1 && prediction_change < 0.1 && change < 0.0025 {
                stable_intervals += 1;
            } else {
                stable_intervals = 0;
            }
            data_stability = Some(
                serde_json::json!({"objective_change":dq,"prediction_change_norm_assigned_sigma":prediction_change,"maximum_H_He_fraction_change":change,"consecutive_intervals":stable_intervals,"iterations_per_interval":25,"objective_tolerance":0.1,"prediction_norm_tolerance":0.1,"fraction_tolerance":0.0025,"qualification":"Data-space stability; does not assert gradient convergence or parameter identifiability"}),
            );
            eprintln!("DATA_STABILITY iterations={iters} objective_change={dq:.4e} prediction_norm={prediction_change:.4e} fraction_change={change:.4e} intervals={stable_intervals}");
            stable_q = q;
            stable_cost = cost;
            stable_r = normals.iter().map(|n| n.residual.clone()).collect();
            if stable_intervals >= 2 {
                status = "data_stable_gradient_pending";
                break;
            }
        }
    }
    if let Some(path) = checkpoint {
        save_checkpoint(path, &context, &xs, lambda, &history, iters, gradient)?;
    }
    // Recompute the score at the returned point, including accepted final iterations.
    let (_, th, tg) = temporal(prior, &shape, &q);
    if let Ok((_, _, g)) = arrow(&normals, &xs, &q, &th, &tg, 1e-3, fixed_h.is_some()) {
        gradient = g;
        if g < 0.03 {
            status = "converged";
        }
    }
    Ok(Solution {
        nodes: xs.iter().map(unpack).collect(),
        shared: q,
        status: status.into(),
        iterations: iters,
        normalized_gradient: gradient,
        objective_history: history,
        robust_objective: dc,
        temporal_objective: tc,
        diagonal_steps,
        data_stability,
    })
}
#[cfg(test)]
mod tests {
    use super::*;
    #[test]
    fn robust_linear_profile_prior_gradient_matches_differences() {
        use crate::coupled::CoordinatePrior;
        let p = Problem {
            coordinate_priors: vec![CoordinatePrior {
                coefficients: vec![(0, 1.), (6, -2.), (12, 1.)],
                mean: 0.,
                sigma: 0.2,
                delta: 1.5,
            }],
            frequency_hz: vec![-1., 0., 1.],
            radar_frequency_hz: 430.2e6,
            shape_density_m3: 1e11,
            cells: vec![],
            products: vec![crate::coupled::Product {
                value: [0., 0.],
                sigma: [1., 1.],
            }],
            knot_count: 5,
            background_basis: vec![num_complex::Complex64::default()],
            background_sigma: 1.,
        };
        let mut x = V::zeros(30);
        x[0] = 0.7;
        x[6] = -0.3;
        x[12] = 0.2;
        let n = data(&p, &x, true).unwrap().1.unwrap();
        for k in [0, 6, 12] {
            let mut a = x.clone();
            let mut b = x.clone();
            a[k] += 1e-6;
            b[k] -= 1e-6;
            let fd = (data(&p, &a, false).unwrap().0 - data(&p, &b, false).unwrap().0) / 2e-6;
            let i = n.columns.iter().position(|c| *c == k).unwrap();
            assert!((fd - 2. * n.g[i]).abs() < 1e-7);
        }
    }
    #[test]
    fn schur_step_matches_dense_joint_solve() {
        let mut ns = vec![];
        let mut xs = vec![];
        for b in 0..3 {
            let j = M::from_fn(24, 9, |i, k| {
                ((i * 7 + k * 11 + b * 3) as f64).sin() + if i == k { 2. } else { 0. }
            });
            let r = V::from_fn(24, |i, _| (i as f64).cos());
            ns.push(Normal {
                h: j.transpose() * &j,
                g: j.transpose() * r,
                columns: vec![0, 1, 2, 3, 4, 10, 17, 23, 29],
                residual: V::zeros(0),
            });
            xs.push(V::from_element(30, 0.5));
        }
        let (trial, shared_trial) =
            diagonal_candidate(&ns, &xs, &[0.5; 5], &M::zeros(5, 5), &V::zeros(5));
        let mut slope = 0.;
        for (b, n) in ns.iter().enumerate() {
            for i in 0..4 {
                slope += n.g[i] * (trial[b][n.columns[i]] - xs[b][n.columns[i]]);
            }
            for i in 0..5 {
                slope += n.g[4 + i] * (shared_trial[i] - 0.5);
            }
        }
        assert!(
            slope < 0.,
            "diagonal proposal must be a feasible descent direction"
        );
        let q = [0.5; 5];
        let (d, s, _) = arrow(&ns, &xs, &q, &M::zeros(5, 5), &V::zeros(5), 0.02, false).unwrap();
        let mut h = M::zeros(17, 17);
        let mut g = V::zeros(17);
        for (b, n) in ns.iter().enumerate() {
            let ix: Vec<_> = (0..4).map(|i| b * 4 + i).chain(12..17).collect();
            for i in 0..9 {
                g[ix[i]] += n.g[i];
                for k in 0..9 {
                    h[(ix[i], ix[k])] += n.h[(i, k)];
                }
            }
        }
        for i in 0..17 {
            h[(i, i)] *= 1.02;
        }
        let exact = h.cholesky().unwrap().solve(&(-g));
        let actual = V::from_iterator(
            17,
            d.into_iter()
                .flat_map(|v| v.as_slice().to_vec())
                .chain(s.iter().copied()),
        );
        assert!((exact - actual).amax() < 1e-10);
        let (d, s, _) = arrow(&ns, &xs, &q, &M::zeros(5,5), &V::zeros(5), 0.02, true).unwrap();
        assert_eq!(s[0],0.); assert_eq!(s[1],0.);
        // With the two H coordinates fixed, nuisance updates must match the
        // corresponding reduced dense system, including all cross-correlations.
        let free:Vec<_>=(0..17).filter(|i|*i!=12&&*i!=13).collect();
        let mut full_h=M::<f64>::zeros(17,17);let mut full_g=V::<f64>::zeros(17);
        for (b,n) in ns.iter().enumerate(){let ix:Vec<_>=(0..4).map(|i|b*4+i).chain(12..17).collect();
            for i in 0..9{full_g[ix[i]]+=n.g[i];for k in 0..9{full_h[(ix[i],ix[k])]+=n.h[(i,k)];}}}
        for i in 0..17{full_h[(i,i)]*=1.02;}
        let hh=M::from_fn(free.len(),free.len(),|i,j|full_h[(free[i],free[j])]);
        let gg=V::from_iterator(free.len(),free.iter().map(|i|full_g[*i]));
        let expected=hh.cholesky().unwrap().solve(&(-gg));
        let actual=V::from_iterator(17,d.into_iter().flat_map(|v|v.as_slice().to_vec()).chain(s.iter().copied()));
        for(i,k)in free.iter().enumerate(){assert!((expected[i]-actual[*k]).abs()<1e-10);}
    }
    #[test]
    fn temporal_fraction_jacobian_and_real_time_spacing() {
        let shape = Shape {
            altitude_km: vec![500., 1000., 2000.],
            scale_height_km: vec![60., 90., 150.],
            source: "test".into(),
        };
        let q = [0.5, 0.3, 0.1, 0.4, 0.5];
        let t = Temporal {
            altitude_km: vec![600., 900., 1200.],
            sigma_per_30min: 0.15,
            neighbors: vec![TimeNeighbor {
                separation_s: 1800.,
                h_fraction: vec![0.1, 0.3, 0.7],
            }],
        };
        let (c, _, g) = temporal(Some(&t), &shape, &q);
        for k in 0..2 {
            let mut a = q;
            let mut b = q;
            a[k] += 1e-6;
            b[k] -= 1e-6;
            let fd = (temporal(Some(&t), &shape, &a).0 - temporal(Some(&t), &shape, &b).0) / 2e-6;
            assert!((fd - 2. * g[k]).abs() < 1e-5);
        }
        let mut twice = t.clone();
        twice.neighbors[0].separation_s = 3600.;
        assert!((temporal(Some(&twice), &shape, &q).0 - c / 2.).abs() < 1e-10);
    }
    #[test]
    fn robust_cost_gradient_is_continuous() {
        for v in [2.99, 3., 3.01, 100., -100.] {
            let e = 1e-6;
            let fd = (huber(v + e).0 - huber(v - e).0) / (2. * e);
            assert!((fd - 2. * huber(v).1 * v).abs() < 2e-6);
        }
    }
}

#[cfg(test)]
mod physical_tests {
    use super::*;
    use crate::{
        coupled::{Cell, Product, Term},
        theory::Theory,
    };
    use num_complex::Complex64 as C;
    #[test]
    fn two_beams_share_composition_without_tying_temperatures_or_velocity() {
        let shape = Arc::new(Shape {
            altitude_km: vec![500., 1000., 2000.],
            scale_height_km: vec![60., 90., 150.],
            source: "test".into(),
        });
        let mut problems = vec![];
        let mut seeds = vec![];
        for b in 0..2 {
            let theory = Arc::new(Theory::new(850., 500.).unwrap());
            let f = theory.f.clone();
            let mut p = Problem {
                coordinate_priors: vec![],
                frequency_hz: f.clone(),
                radar_frequency_hz: 430.2e6,
                shape_density_m3: 1e11,
                cells: vec![Cell {
                    theory,
                    shape_density_m3: None,
                    density_calibration: Some((700., 850., 1.)),
                    knot_weights: vec![(0, 1.)],
                    h_knot_weights: None,
                    composition: Some((shape.clone(), 850.)),
                    terms: (0..25)
                        .map(|l| Term {
                            reuse: None,
                            measurement: l,
                            kernel: f
                                .iter()
                                .enumerate()
                                .map(|(j, &freq)| {
                                    C::from_polar(
                                        500. * if j == 0 || j == 200 { 0.5 } else { 1. },
                                        -std::f64::consts::TAU * freq * l as f64 * 1e-5,
                                    )
                                })
                                .collect(),
                        })
                        .collect(),
                }],
                products: vec![
                    Product {
                        value: [0., 0.],
                        sigma: [1e-7, 1e-7]
                    };
                    25
                ],
                knot_count: 5,
                background_basis: vec![C::default(); 25],
                background_sigma: 1.,
            };
            let mut nodes = vec![
                Node {
                    power_adu2: 1. + b as f64,
                    ti_k: 800. + 300. * b as f64,
                    te_k: 1600. + 400. * b as f64,
                    velocity_m_s: -40. + 90. * b as f64,
                    h_fraction: 0.,
                    he_fraction: 0.,
                    o_fraction: 1.
                };
                5
            ];
            let h = Shape::encode(900., 100.);
            let he = crate::h_transition::helium_encode(0.1, 850., 150.);
            nodes[0].h_fraction = h[0];
            nodes[0].o_fraction = 1. - h[0];
            nodes[1].h_fraction = h[1];
            nodes[1].o_fraction = 1. - h[1];
            for k in 0..3 {
                nodes[k + 2].he_fraction = he[k];
                nodes[k + 2].o_fraction = 1. - he[k];
            }
            let (r, _) = p.evaluate(&encode(&nodes), false).unwrap();
            for i in 0..25 {
                p.products[i].value = [r[2 * i] * 1e-7, r[2 * i + 1] * 1e-7];
            }
            problems.push(p);
            seeds.push(nodes);
        }
        let refs: Vec<_> = problems.iter().collect();
        let solution = solve(&refs, &seeds, shape.clone(), None, 3).unwrap();
        assert_eq!(solution.status, "converged");
        assert!(solution.robust_objective < 1e-12);
        assert_ne!(solution.nodes[0][0].ti_k, solution.nodes[1][0].ti_k);
        assert_ne!(
            solution.nodes[0][0].velocity_m_s,
            solution.nodes[1][0].velocity_m_s
        );
        let a = encode(&solution.nodes[0]);
        let b = encode(&solution.nodes[1]);
        for i in SHARED {
            assert_eq!(a[i], b[i]);
        }
        let mut perturbed = seeds.clone();
        for n in &mut perturbed {
            n[0].ti_k *= 1.03;
            n[0].te_k *= 0.98;
            n[0].velocity_m_s += 2.;
            n[0].h_fraction += 0.01;
        }
        let improved = solve_lbfgs(&refs, &perturbed, shape.clone(), None, 100, None).unwrap();
        assert!(
            improved.objective_history.last().unwrap() < &(improved.objective_history[0] * 0.05)
        );
        assert!(improved.objective_history.windows(2).all(|w| w[1] <= w[0]));
        let lp = std::env::temp_dir().join(format!("syisr-lbfgs-{}.h5", std::process::id()));
        let _ = std::fs::remove_file(&lp);
        let two = solve_lbfgs(&refs, &perturbed, shape.clone(), None, 2, None).unwrap();
        solve_lbfgs(&refs, &perturbed, shape.clone(), None, 1, Some(&lp)).unwrap();
        let split = solve_lbfgs(&refs, &perturbed, shape.clone(), None, 1, Some(&lp)).unwrap();
        assert!(
            (two.robust_objective - split.robust_objective).abs()
                < 1e-10 * two.robust_objective.max(1.)
        );
        for (a, b) in two.shared.iter().zip(&split.shared) {
            assert!((a - b).abs() < 1e-8);
        }
        std::fs::remove_file(lp).unwrap();
        let checkpoint =
            std::env::temp_dir().join(format!("syisr-shared-checkpoint-{}.h5", std::process::id()));
        let _ = std::fs::remove_file(&checkpoint);
        let first =
            solve_checkpointed(&refs, &seeds, shape.clone(), None, 0, Some(&checkpoint)).unwrap();
        let resumed =
            solve_checkpointed(&refs, &seeds, shape.clone(), None, 0, Some(&checkpoint)).unwrap();
        assert_eq!(resumed.status, "converged");
        assert!((resumed.robust_objective - first.robust_objective).abs() < 1e-10);
        for (a, b) in resumed.shared.iter().zip(&first.shared) {
            assert!((a - b).abs() < 1e-14);
        }
        let changed = Temporal {
            altitude_km: vec![800.],
            sigma_per_30min: 0.15,
            neighbors: vec![TimeNeighbor {
                separation_s: 1800.,
                h_fraction: vec![0.2],
            }],
        };
        assert!(
            solve_checkpointed(&refs, &seeds, shape, Some(&changed), 1, Some(&checkpoint)).is_err()
        );
        std::fs::remove_file(&checkpoint).unwrap();
    }
}

/// Experimental preconditioned L-BFGS. Same Huber/temporal objective and score tolerance.
/// Avoids J^T J and dense normal solves; automatic differentiation supplies gradients.
pub fn solve_lbfgs(
    problems: &[&Problem],
    initial: &[Vec<Node>],
    shape: Arc<Shape>,
    prior: Option<&Temporal>,
    max_iterations: usize,
    checkpoint: Option<&std::path::Path>,
) -> Result<Solution> {
    use std::collections::VecDeque;
    let memory_limit = std::env::var("SYISR_LBFGS_MEMORY")
        .unwrap_or("15".into())
        .parse::<usize>()?;
    ensure!(memory_limit <= 30, "invalid L-BFGS history length");
    let angle = std::env::var("SYISR_LBFGS_ANGLE")
        .ok()
        .map(|v| v.parse::<f64>())
        .transpose()?
        .unwrap_or(0.);
    ensure!(
        (0.0..1.0).contains(&angle),
        "invalid direction angle safeguard"
    );
    let adaptive = memory_limit == 0 && std::env::var_os("SYISR_ADAPTIVE_DIAGONAL").is_some();
    ensure!(
        !problems.is_empty() && problems.len() == initial.len(),
        "beam dimensions"
    );
    shape.validate()?;
    let context = optimizer_context(problems, &shape, prior)?;
    let mut xs: Vec<_> = initial.iter().map(|n| encode(n)).collect();
    let mut iterations_base = 0;
    let mut saved_state = None;
    if let Some(path) = checkpoint.filter(|p| p.exists()) {
        let v: serde_json::Value = crate::campaign::load(path)?;
        ensure!(
            same_context(&v["context"], &context),
            "L-BFGS checkpoint context differs"
        );
        let nodes: Vec<Vec<Node>> = serde_json::from_value(v["nodes"].clone())?;
        ensure!(
            nodes.len() == xs.len() && nodes.iter().zip(&xs).all(|(n, x)| n.len() * 6 == x.len()),
            "checkpoint dimensions"
        );
        xs = nodes.iter().map(|n| encode(n)).collect();
        iterations_base = v["iterations"].as_u64().unwrap_or(0) as usize;
        saved_state = Some(v);
    }
    let columns: Vec<_> = problems.iter().map(|p| local_columns(p)).collect();
    let mut offsets = vec![0];
    for c in &columns {
        offsets.push(offsets.last().unwrap() + c.len());
    }
    let shared_offset = *offsets.last().unwrap();
    let size = shared_offset + 5;
    let mut packed = V::zeros(size);
    for (b, cols) in columns.iter().enumerate() {
        for (i, k) in cols.iter().enumerate() {
            packed[offsets[b] + i] = xs[b][*k];
        }
    }
    for (i, k) in SHARED.iter().enumerate() {
        packed[shared_offset + i] = xs[0][*k];
    }
    if std::env::var("SYISR_FIX_HELIUM_ZERO").ok().as_deref() == Some("1") {
        packed[shared_offset + 2] = 0.;
    }
    let apply = |p: &V<f64>, xs: &mut Vec<V<f64>>| {
        for (b, x) in xs.iter_mut().enumerate() {
            for (i, k) in columns[b].iter().enumerate() {
                x[*k] = p[offsets[b] + i];
            }
            for (i, k) in SHARED.iter().enumerate() {
                x[*k] = p[shared_offset + i];
            }
        }
    };
    apply(&packed, &mut xs);
    let evaluate = |xs: &[V<f64>], grad: bool| -> Result<(f64, f64, V<f64>, V<f64>)> {
        let mut cost = 0.;
        let mut g = V::zeros(size);
        let mut diagonal = V::zeros(size);
        for (b, (p, x)) in problems.iter().zip(xs).enumerate() {
            let (r, j) = p.evaluate(x, grad)?;
            let w: V<f64> = V::from_iterator(r.len(), r.iter().map(|&v| huber(v).1));
            cost += r.iter().map(|&v| huber(v).0).sum::<f64>();
            if grad {
                let wr = r.component_mul(&w);
                for (k, &col) in columns[b].iter().chain(SHARED.iter()).enumerate() {
                    let dest = if k < columns[b].len() {
                        offsets[b] + k
                    } else {
                        shared_offset + k - columns[b].len()
                    };
                    let c = j.column(col);
                    g[dest] += 2. * c.dot(&wr);
                    diagonal[dest] +=
                        2. * c.iter().zip(w.iter()).map(|(a, b)| a * a * b).sum::<f64>();
                }
            }
            for prior in &p.coordinate_priors {
                let z = (prior
                    .coefficients
                    .iter()
                    .map(|(c, w)| x[*c] * w)
                    .sum::<f64>()
                    - prior.mean)
                    / prior.sigma;
                let weight = 1. / (1. + (z / prior.delta).powi(2)).sqrt();
                cost += 2. * prior.delta.powi(2) * ((1. + (z / prior.delta).powi(2)).sqrt() - 1.);
                if grad {
                    for (column, coefficient) in &prior.coefficients {
                        let local =
                            columns[b].iter().position(|c| c == column).ok_or_else(|| {
                                anyhow::anyhow!("L-BFGS prior on unsupported local coordinate")
                            })?;
                        let dest = offsets[b] + local;
                        g[dest] += 2. * weight * z * coefficient / prior.sigma;
                        diagonal[dest] += 2. * weight * (coefficient / prior.sigma).powi(2);
                    }
                }
            }
        }
        let q = std::array::from_fn(|i| xs[0][SHARED[i]]);
        let (tc, h, tg) = temporal(prior, &shape, &q);
        if grad {
            for i in 0..5 {
                g[shared_offset + i] += 2. * tg[i];
                diagonal[shared_offset + i] += 2. * h[(i, i)];
            }
        }
        if std::env::var("SYISR_FIX_HELIUM_ZERO").ok().as_deref() == Some("1") {
            for k in 2..5 {
                g[shared_offset + k] = 0.;
                diagonal[shared_offset + k] = 0.;
            }
        }
        Ok((cost, tc, g, diagonal))
    };
    let allcols: Vec<_> = columns.iter().flatten().copied().chain(SHARED).collect();
    let bounds: Vec<_> = allcols.iter().map(|&i| bounds(i)).collect();
    let (mut dc, mut tc, mut g, mut diag) = evaluate(&xs, true)?;
    let mut cost = dc + tc;
    let mut scale = if let Some(v) = saved_state
        .as_ref()
        .filter(|v| v["lbfgs"]["scale"].is_array())
    {
        let a: Vec<f64> = serde_json::from_value(v["lbfgs"]["scale"].clone())?;
        ensure!(
            a.len() == size && a.iter().all(|v| v.is_finite() && *v > 0.),
            "invalid saved preconditioner"
        );
        V::from_vec(a)
    } else {
        diag.map(|d| 1. / d.sqrt().max(1e-5))
    };
    let mut z = packed.component_div(&scale);
    let mut gz = g.component_mul(&scale);
    let mut history = vec![cost];
    let mut memory: VecDeque<(V<f64>, V<f64>, f64)> = VecDeque::new();
    if let Some(v) = saved_state.as_ref() {
        if let Some(pairs) = v["lbfgs"]["pairs"].as_array() {
            for p in pairs {
                let a: Vec<f64> = serde_json::from_value(p["s"].clone())?;
                let b: Vec<f64> = serde_json::from_value(p["y"].clone())?;
                ensure!(
                    a.len() == size && b.len() == size && a.iter().chain(&b).all(|v| v.is_finite()),
                    "invalid saved L-BFGS pair"
                );
                let a = V::from_vec(a);
                let b = V::from_vec(b);
                let sy = a.dot(&b);
                ensure!(sy > 0., "invalid L-BFGS curvature");
                memory.push_back((a, b, 1. / sy));
            }
        }
        if let Some(old) = v["objective_history"].as_array() {
            history = old.iter().filter_map(|v| v.as_f64()).collect();
            history.push(cost);
        }
    }

    while memory.len() > memory_limit {
        memory.pop_front();
    }
    let mut status = "iteration_limit";
    let mut gradient = f64::INFINITY;
    let mut iterations = iterations_base;
    let prediction_snapshot = |states: &[V<f64>]| -> Result<Vec<f64>> {
        let mut values = vec![];
        for (p, x) in problems.iter().zip(states) {
            let (r, _) = p.evaluate(x, false)?;
            for (i, product) in p.products.iter().enumerate() {
                for c in 0..2 {
                    if product.sigma[c] < 1e20 {
                        values.push(r[2 * i + c]);
                    }
                }
            }
        }
        Ok(values)
    };
    let stability_enabled = std::env::var_os("SYISR_DATA_STABILITY_STOP").is_some();
    let mut stable_r = if stability_enabled {
        prediction_snapshot(&xs)?
    } else {
        vec![]
    };
    let mut stable_cost = cost;
    let mut stable_q = std::array::from_fn(|i| packed[shared_offset + i]);
    let mut stable_intervals = 0;
    let mut data_stability = None;
    for it in 0..if max_iterations == 0 {
        usize::MAX
    } else {
        max_iterations
    } {
        iterations = iterations_base + it + 1;
        let blocked: Vec<_> = (0..size)
            .map(|i| {
                diag[i] < 1e-20
                    || (packed[i] <= bounds[i].0 + 1e-8 && g[i] > 0.)
                    || (packed[i] >= bounds[i].1 - 1e-8 && g[i] < 0.)
            })
            .collect();
        gradient = (0..size)
            .filter(|&i| !blocked[i])
            .map(|i| g[i].abs() / ((2. * diag[i]).sqrt().max(1e-12)))
            .fold(0., f64::max);
        if it % 5 == 0 {
            eprintln!(
                "LBFGS_ITER {it} objective={cost:.6e} gradient={gradient:.5e} history={}",
                memory.len()
            );
        }
        if let Some(path) = checkpoint {
            if it % 25 == 0 {
                save_lbfgs_checkpoint(
                    path, &context, &xs, &history, iterations, gradient, &scale, &memory,
                )?;
            }
        }
        if gradient < 0.03 {
            status = "converged";
            break;
        }
        let mut pg = gz.clone();
        for i in 0..size {
            if blocked[i] {
                pg[i] = 0.;
            }
        }
        let mut q = pg.clone();
        let mut alpha = vec![0.; memory.len()];
        for (k, (s, y, rho)) in memory.iter().enumerate().rev() {
            alpha[k] = rho * s.dot(&q);
            q -= y * alpha[k];
        }
        let gamma = memory
            .back()
            .map(|(s, y, _)| s.dot(y) / y.dot(y).max(1e-30))
            .unwrap_or(1.)
            .clamp(1e-8, 1e8);
        let mut direction = q * gamma;
        for (k, (s, y, rho)) in memory.iter().enumerate() {
            let beta = rho * y.dot(&direction);
            direction += s * (alpha[k] - beta);
        }
        direction = -direction;
        for i in 0..size {
            if blocked[i] {
                direction[i] = 0.;
            }
        }
        if direction.dot(&pg) >= -1e-14 * pg.norm_squared()
            || (angle > 0. && direction.dot(&pg) >= -angle * direction.norm() * pg.norm())
        {
            direction = -pg.clone();
            memory.clear();
        }
        let mut accepted = None;
        for restart in 0..2 {
            if restart == 1 {
                memory.clear();
                direction = -pg.clone();
            }
            let mut step = 1.;
            for _ in 0..24 {
                let trial_z = &z + &direction * step;
                let trial = V::from_iterator(
                    size,
                    (0..size).map(|i| (trial_z[i] * scale[i]).clamp(bounds[i].0, bounds[i].1)),
                );
                let trial_z = trial.component_div(&scale);
                let dz = &trial_z - &z;
                let slope = gz.dot(&dz);
                if slope < 0. {
                    let mut new_xs = xs.clone();
                    apply(&trial, &mut new_xs);
                    let (nd, nt, _, _) = evaluate(&new_xs, false)?;
                    if nd + nt <= cost + 1e-4 * slope {
                        accepted = Some((trial, trial_z, new_xs, nd, nt));
                        break;
                    }
                }
                step *= 0.5;
            }
            if accepted.is_some() {
                break;
            }
        }
        let Some((np, nz, nxs, nd, nt)) = accepted else {
            status = "stalled_line_search";
            break;
        };
        let (_, _, ng, ndiag) = evaluate(&nxs, true)?;
        let ngz = ng.component_mul(&scale);
        let s = &nz - &z;
        let y = &ngz - &gz;
        let curvature = s.dot(&y);
        if memory_limit > 0 && curvature > 1e-10 * s.norm() * y.norm() {
            if memory.len() >= memory_limit {
                memory.pop_front();
            }
            memory.push_back((s, y, 1. / curvature));
        }
        packed = np;
        z = nz;
        xs = nxs;
        dc = nd;
        tc = nt;
        cost = nd + nt;
        g = ng;
        diag = ndiag;
        if adaptive {
            scale = diag.map(|d| 1. / d.sqrt().max(1e-5));
            z = packed.component_div(&scale);
            gz = g.component_mul(&scale);
        } else {
            gz = ngz;
        }
        history.push(cost);
        if stability_enabled && (it + 1) % 25 == 0 {
            let current = prediction_snapshot(&xs)?;
            let prediction_change = current
                .iter()
                .zip(&stable_r)
                .map(|(a, b)| (a - b).powi(2))
                .sum::<f64>()
                .sqrt();
            let q = std::array::from_fn(|i| packed[shared_offset + i]);
            let fraction_change = fraction_change(&shape, &q, &stable_q);
            let objective_change = (cost - stable_cost).abs();
            if objective_change < 0.1 && prediction_change < 0.1 && fraction_change < 0.0025 {
                stable_intervals += 1;
            } else {
                stable_intervals = 0;
            }
            data_stability = Some(
                serde_json::json!({"objective_change":objective_change,"prediction_change_norm_assigned_sigma":prediction_change,"maximum_H_He_fraction_change":fraction_change,"consecutive_intervals":stable_intervals,"iterations_per_interval":25,"objective_tolerance":0.1,"prediction_norm_tolerance":0.1,"fraction_tolerance":0.0025,"qualification":"Data-space stability only, not gradient convergence or identifiability"}),
            );
            eprintln!("LBFGS_STABILITY iterations={iterations} objective_change={objective_change:.4e} prediction_norm={prediction_change:.4e} fraction_change={fraction_change:.4e} intervals={stable_intervals}");
            stable_r = current;
            stable_cost = cost;
            stable_q = q;
            if stable_intervals >= 2 {
                status = "data_stable_gradient_pending";
                break;
            }
        }
    }
    // Report the score at the returned point, not at the preceding accepted step.
    gradient = (0..size)
        .filter(|&i| {
            diag[i] > 1e-20
                && !((packed[i] <= bounds[i].0 + 1e-8 && g[i] > 0.)
                    || (packed[i] >= bounds[i].1 - 1e-8 && g[i] < 0.))
        })
        .map(|i| g[i].abs() / (2. * diag[i]).sqrt().max(1e-12))
        .fold(0., f64::max);
    if gradient < 0.03 {
        status = "converged";
    }
    if let Some(path) = checkpoint {
        save_lbfgs_checkpoint(
            path, &context, &xs, &history, iterations, gradient, &scale, &memory,
        )?;
    }
    Ok(Solution {
        nodes: xs.iter().map(unpack).collect(),
        shared: std::array::from_fn(|i| packed[shared_offset + i]),
        status: status.into(),
        iterations,
        normalized_gradient: gradient,
        objective_history: history,
        robust_objective: dc,
        temporal_objective: tc,
        diagonal_steps: 0,
        data_stability,
    })
}

fn save_lbfgs_checkpoint(
    path: &std::path::Path,
    context: &serde_json::Value,
    xs: &[V<f64>],
    history: &[f64],
    iterations: usize,
    gradient: f64,
    scale: &V<f64>,
    memory: &std::collections::VecDeque<(V<f64>, V<f64>, f64)>,
) -> Result<()> {
    let nodes: Vec<_> = xs.iter().map(unpack).collect();
    let pairs: Vec<_> = memory
        .iter()
        .map(|(s, y, _)| serde_json::json!({"s":s.as_slice(),"y":y.as_slice()}))
        .collect();
    crate::campaign::save(
        path,
        &serde_json::json!({"schema":"syisr-shared-optimizer-checkpoint-v2","context":context,"nodes":nodes,"lambda":1e-3,"objective_history":history,"iterations":iterations,"normalized_gradient":if gradient.is_finite(){Some(gradient)}else{None},"status":"optimizer_checkpoint_not_accepted_fit","lbfgs":{"scale":scale.as_slice(),"pairs":pairs}}),
    )
}

#[cfg(test)]
mod lbfgs_prior_tests {
    use super::*;
    #[test]
    fn lbfgs_keeps_identical_robust_profile_prior_objective() {
        let problem = Problem {
            coordinate_priors: vec![crate::coupled::CoordinatePrior {
                coefficients: vec![(0, 1.), (6, -0.5)],
                mean: 0.7,
                sigma: 0.5,
                delta: 1.5,
            }],
            frequency_hz: vec![-1., 0., 1.],
            radar_frequency_hz: 430.2e6,
            shape_density_m3: 1e11,
            cells: vec![],
            products: vec![],
            knot_count: 5,
            background_basis: vec![],
            background_sigma: 1.,
        };
        let nodes = vec![
            Node {
                power_adu2: 1.,
                ti_k: 1000.,
                te_k: 1500.,
                velocity_m_s: 0.,
                h_fraction: 0.2,
                he_fraction: 0.,
                o_fraction: 0.8
            };
            5
        ];
        let shape = Arc::new(Shape {
            altitude_km: vec![500., 1000., 2000.],
            scale_height_km: vec![60., 90., 150.],
            source: "test".into(),
        });
        let initial = data(&problem, &encode(&nodes), false).unwrap().0;
        let fit = solve_lbfgs(&[&problem], &[nodes], shape, None, 3, None).unwrap();
        let replay = data(&problem, &encode(&fit.nodes[0]), false).unwrap().0;
        assert!((replay - fit.robust_objective).abs() < 1e-12);
        assert!(fit.robust_objective < initial);
    }
}
