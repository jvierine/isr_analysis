//! Weighted bounded projection of gated estimates onto the actual spline basis.
//! This changes only the starting point, never the joint data likelihood.
use crate::{
    coupled::{encode, Node},
    fitting::Profile,
    spline::{h_weight, Spline},
};
use nalgebra::{DMatrix as M, DVector as V};
fn bounds(f: usize) -> (f64, f64) {
    match f {
        0 => (1e-4f64.ln(), 1e3f64.ln()),
        1 => (0.15f64.ln(), 6f64.ln()),
        2 => (0.15f64.ln(), 12f64.ln()),
        3 => (-2., 2.),
        _ => (0., 1.),
    }
}
pub fn project(s: &Spline, p: &Profile, seed: &[Node]) -> Vec<Node> {
    let n = s.count();
    assert_eq!(seed.len(), n);
    let prior = encode(seed);
    let controls = s.controls();
    let mut x = prior.clone();
    for field in 0..5 {
        let mut h = M::<f64>::zeros(n, n);
        let mut b = V::<f64>::zeros(n);
        for g in &p.gates {
            let Some(node) = &g.node else {
                continue;
            };
            let value = match field {
                0 => g.ne_m3.filter(|v| *v > 0.).map(|v| (v / 1e11).ln()),
                1 => {
                    if node.ti_k > 155. && node.ti_k < 5900. {
                        Some((node.ti_k / 1000.).ln())
                    } else {
                        None
                    }
                }
                2 => {
                    if node.te_k > 155. && node.te_k < 11800. {
                        Some((node.te_k / 1000.).ln())
                    } else {
                        None
                    }
                }
                3 => Some(node.velocity_m_s / 1000.),
                _ => {
                    if g.altitude_km > 500. {
                        Some(node.h_fraction)
                    } else {
                        None
                    }
                }
            };
            let Some(value) = value.filter(|v| v.is_finite()) else {
                continue;
            };
            let se = g.conditional_standard_error.get(field).copied().flatten();
            let sigma = match field {
                0 => g.lower95[0]
                    .zip(g.upper95[0])
                    .filter(|(a, b)| *a > 0. && b > a)
                    .map_or(0.3, |(a, b)| (b / a).ln() / 3.92)
                    .clamp(0.15, 2.),
                1 => se.map_or(0.3, |v| v / node.ti_k).clamp(0.05, 2.),
                2 => se.map_or(0.3, |v| v / node.te_k).clamp(0.05, 2.),
                3 => se.map_or(0.2, |v| v / 1000.).clamp(0.03, 2.),
                _ => se.unwrap_or(0.2).clamp(0.05, 1.),
            };
            let mut weight = 1. / (sigma * sigma);
            if g.altitude_km < 200. {
                weight *= 0.05;
            }
            if g.flags.iter().any(|v| v == "weak_power") {
                weight *= 0.01;
            }
            let taper = if field == 4 {
                h_weight(g.altitude_km)
            } else {
                1.
            };
            let a: Vec<_> = s
                .weights(g.altitude_km)
                .into_iter()
                .map(|(i, w)| (i, w * taper))
                .collect();
            for (i, wi) in &a {
                b[*i] += weight * wi * value;
                for (j, wj) in &a {
                    h[(*i, *j)] += weight * wi * wj;
                }
            }
        }
        let scale = (0..n).map(|i| h[(i, i)]).sum::<f64>() / n as f64;
        let ridge = (scale * 1e-5).max(1e-8);
        for i in 0..n {
            h[(i, i)] += ridge;
            b[i] += ridge * prior[6 * i + field];
        }
        let (lo, hi) = bounds(field);
        let mut active: Vec<Option<f64>> = vec![None; n];
        if field == 4 {
            for (i, z) in controls.iter().enumerate() {
                if *z <= 500. {
                    active[i] = Some(0.);
                }
            }
        }
        let mut fit: Vec<_> = (0..n).map(|i| prior[6 * i + field].clamp(lo, hi)).collect();
        for _ in 0..=n {
            let free: Vec<_> = (0..n).filter(|i| active[*i].is_none()).collect();
            for (i, v) in active.iter().enumerate() {
                if let Some(v) = v {
                    fit[i] = *v;
                }
            }
            if free.is_empty() {
                break;
            }
            let matrix = M::from_fn(free.len(), free.len(), |i, j| h[(free[i], free[j])]);
            let rhs = V::from_iterator(
                free.len(),
                free.iter().map(|i| {
                    b[*i]
                        - (0..n)
                            .filter_map(|j| active[j].map(|v| h[(*i, j)] * v))
                            .sum::<f64>()
                }),
            );
            let Some(chol) = matrix.cholesky() else {
                break;
            };
            let solution = chol.solve(&rhs);
            for (i, v) in free.iter().zip(solution.iter()) {
                fit[*i] = *v;
            }
            let worst = free
                .iter()
                .filter_map(|i| {
                    let v = fit[*i];
                    if v < lo {
                        Some((*i, lo, lo - v))
                    } else if v > hi {
                        Some((*i, hi, v - hi))
                    } else {
                        None
                    }
                })
                .max_by(|a, b| a.2.total_cmp(&b.2));
            if let Some((i, v, _)) = worst {
                active[i] = Some(v);
            } else {
                break;
            }
        }
        for i in 0..n {
            x[6 * i + field] = fit[i].clamp(lo, hi);
        }
    }
    (0..n)
        .map(|i| {
            let k = 6 * i;
            let h = x[k + 4];
            Node {
                power_adu2: x[k].exp(),
                ti_k: 1000. * x[k + 1].exp(),
                te_k: 1000. * x[k + 2].exp(),
                velocity_m_s: 1000. * x[k + 3],
                h_fraction: h,
                he_fraction: 0.,
                o_fraction: 1. - h,
            }
        })
        .collect()
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::{campaign::Job, fitting::Gate, grid};
    fn values(s: &Spline, n: &[Node], z: f64) -> [f64; 5] {
        let x = encode(n);
        let mut p = [0.; 5];
        for (k, w) in s.weights(z) {
            for f in 0..5 {
                p[f] += x[6 * k + f] * w;
            }
        }
        p[4] *= h_weight(z);
        p
    }
    #[test]
    fn projection_recovers_curved_spline_instead_of_treating_values_as_coefficients() {
        let s = Spline::adaptive();
        let nodes: Vec<_> = s
            .controls()
            .iter()
            .map(|z| {
                let h = if *z <= 500. {
                    0.
                } else {
                    0.3 + 0.1 * (z / 150.).sin()
                };
                Node {
                    power_adu2: (0.5 + 0.5 * (z / 170.).sin()).exp(),
                    ti_k: 1000. * (0.2 * (z / 200.).cos()).exp(),
                    te_k: 1500. * (0.2 * (z / 180.).sin()).exp(),
                    velocity_m_s: 20. + 30. * (z / 250.).sin(),
                    h_fraction: h,
                    he_fraction: 0.,
                    o_fraction: 1. - h,
                }
            })
            .collect();
        let gates: Vec<_> = grid::build(90., 1000.)
            .iter()
            .map(|g| {
                let v = values(&s, &nodes, g.altitude_km);
                let h = v[4];
                Gate {
                    altitude_km: g.altitude_km,
                    range_km: g.slant_range_km,
                    width_km: g.upper_altitude_km - g.lower_altitude_km,
                    node: Some(Node {
                        power_adu2: 1.,
                        ti_k: 1000. * v[1].exp(),
                        te_k: 1000. * v[2].exp(),
                        velocity_m_s: 1000. * v[3],
                        h_fraction: h,
                        he_fraction: 0.,
                        o_fraction: 1. - h,
                    }),
                    ne_m3: Some(1e11 * v[0].exp()),
                    lower68: vec![None; 5],
                    upper68: vec![None; 5],
                    lower95: vec![None; 5],
                    upper95: vec![None; 5],
                    bootstrap_converged: 0,
                    bootstrap_requested: 0,
                    flags: vec![],
                    chi_square: None,
                    observed: vec![],
                    prediction: vec![],
                    conditional_standard_error: vec![None; 5],
                    h_profile: vec![],
                    bootstrap_nodes: vec![],
                }
            })
            .collect();
        let p = Profile {
            schema: "test".into(),
            job: Job {
                id: "test".into(),
                start: 0.,
                end: 1.,
                az: 0.,
                el: 90.,
                dwells: vec![],
            },
            exposure_s: 1.,
            cycles: 64,
            screen_fraction: 0.,
            calibration_gain: Some(1.),
            calibration_fractional_error: Some(0.15),
            gates,
            bootstrap: "test".into(),
            source_dwells: vec![],
            altitude_edges_km: vec![],
        };
        let seed: Vec<_> = s
            .controls()
            .iter()
            .map(|z| {
                let g = p
                    .gates
                    .iter()
                    .min_by(|a, b| {
                        (a.altitude_km - z)
                            .abs()
                            .total_cmp(&(b.altitude_km - z).abs())
                    })
                    .unwrap();
                let mut n = g.node.clone().unwrap();
                n.power_adu2 = g.ne_m3.unwrap() / 1e11;
                n
            })
            .collect();
        let fitted = project(&s, &p, &seed);
        let err = |n: &[Node]| {
            p.gates
                .iter()
                .map(|g| {
                    let a = values(&s, n, g.altitude_km);
                    let b = values(&s, &nodes, g.altitude_km);
                    (0..5).map(|f| (a[f] - b[f]).powi(2)).sum::<f64>()
                })
                .sum::<f64>()
        };
        assert!(
            err(&fitted) < 0.02 * err(&seed),
            "{} vs {}",
            err(&fitted),
            err(&seed)
        );
        assert!(fitted.iter().all(|n| (0.0..=1.0).contains(&n.h_fraction)));
    }
}
