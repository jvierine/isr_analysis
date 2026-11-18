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
}
struct Normal {
    h: M<f64>,
    g: V<f64>,
    columns: Vec<usize>,
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
    let cost = r.iter().map(|v| huber(*v).0).sum();
    if !derivatives {
        return Ok((cost, None));
    }
    let mut columns = local_columns(p);
    columns.extend(SHARED);
    let mut a = M::from_fn(r.len(), columns.len(), |i, k| j[(i, columns[k])]);
    let mut rw = r.clone();
    for i in 0..r.len() {
        let w = huber(r[i]).1.sqrt();
        rw[i] *= w;
        a.row_mut(i).scale_mut(w);
    }
    Ok((
        cost,
        Some(Normal {
            h: a.transpose() * &a,
            g: a.transpose() * rw,
            columns,
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
                cost += r * r;
                for a in 0..2 {
                    g[a] += d[a] / sigma * r;
                    for b in 0..2 {
                        h[(a, b)] += d[a] * d[b] / sigma.powi(2);
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
            (ch[(i, i)] < 1e-20)
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
pub fn solve(
    problems: &[&Problem],
    initial: &[Vec<Node>],
    shape: Arc<Shape>,
    prior: Option<&Temporal>,
    max_iterations: usize,
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
    for it in 0..max_iterations {
        iters = it + 1;
        let (_, th, tg) = temporal(prior, &shape, &q);
        let step = arrow(&normals, &xs, &q, &th, &tg, lambda);
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
        let nq = std::array::from_fn(|i| (q[i] + ds[i].clamp(-0.15, 0.15)).clamp(0., 1.));
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
        let nt = temporal(prior, &shape, &nq).0;
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
            lambda = (lambda / 3.).max(1e-10);
        } else {
            lambda *= 8.;
            if lambda > 1e15 {
                status = "stalled";
                break;
            }
        }
    }
    // Recompute the score at the returned point, including accepted final iterations.
    let (_, th, tg) = temporal(prior, &shape, &q);
    if let Ok((_, _, g)) = arrow(&normals, &xs, &q, &th, &tg, 1e-3) {
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
    })
}
#[cfg(test)]
mod tests {
    use super::*;
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
            });
            xs.push(V::from_element(30, 0.5));
        }
        let q = [0.5; 5];
        let (d, s, _) = arrow(&ns, &xs, &q, &M::zeros(5, 5), &V::zeros(5), 0.02).unwrap();
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
        let solution = solve(&refs, &seeds, shape, None, 3).unwrap();
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
    }
}
