//! SYISR adapter for shared range-coupled atomic composition inference.
use anyhow::{ensure, Result};
use num_complex::Complex64 as C;
use serde::Deserialize;
use serde_json::json;
use std::fs;
use std::sync::Arc;
use syisr_hplus_night::{
    coupled::{Cell, Node, Problem, Product, Term},
    theory::Theory,
};
#[derive(Deserialize)]
struct Datum {
    sample: usize,
    lag: usize,
    value: [f64; 2],
    sigma: [f64; 2],
}
#[derive(Deserialize)]
struct Input {
    radar_frequency_hz: f64,
    shape_density_m3: f64,
    frequency_step_hz: f64,
    frequency_limit_hz: f64,
    sample_rate_hz: f64,
    gate_front_s: f64,
    pulse_width_s: f64,
    knot_range_km: Vec<f64>,
    cell_edges_km: Vec<f64>,
    fir: Vec<(f64, [f64; 2])>,
    products: Vec<Datum>,
    initial: Vec<Node>,
    provenance: serde_json::Value,
    #[serde(default)]
    prescribed_node_fields: Vec<[bool; 6]>,
    background_basis: Vec<[f64; 2]>,
    background_sigma: f64,
    #[serde(default)]
    score_tolerance: Option<f64>,
    #[serde(default)]
    shape_density_at_knots_m3: Option<Vec<f64>>,
}
fn weights(r: f64, k: &[f64]) -> Vec<(usize, f64)> {
    if r <= k[0] {
        return vec![(0, 1.)];
    }
    if r >= k[k.len() - 1] {
        return vec![(k.len() - 1, 1.)];
    }
    let j = k.partition_point(|&x| x < r);
    let a = (r - k[j - 1]) / (k[j] - k[j - 1]);
    vec![(j - 1, 1. - a), (j, a)]
}
fn kernel(
    t: f64,
    lag: f64,
    width: f64,
    edges: [f64; 2],
    fir: &[(f64, C)],
    frequency: &[f64],
    dt: f64,
    phasors: &[Vec<C>],
    offset: isize,
) -> Vec<C> {
    let mut coeff = vec![C::default(); phasors.len()];
    for &(a, ha) in fir {
        for &(b, hb) in fir {
            let first = t - a;
            let second = t + lag - b;
            let overlap = (first.min(second).min(edges[1])
                - (first - width).max(second - width).max(edges[0]))
            .max(0.);
            if overlap == 0. {
                continue;
            }
            let tau = lag + a - b;
            let j = (tau / dt).round() as isize + offset;
            coeff[j as usize] += ha * hb.conj() * (overlap / width);
        }
    }
    let mut out = vec![C::default(); frequency.len()];
    for (j, c) in coeff.iter().enumerate() {
        if c.norm_sqr() == 0. {
            continue;
        }
        for (z, p) in out.iter_mut().zip(&phasors[j]) {
            *z += *c * *p;
        }
    }
    for (j, z) in out.iter_mut().enumerate() {
        let q = if j == 0 || j + 1 == frequency.len() {
            0.5
        } else {
            1.
        };
        *z *= q * (frequency[1] - frequency[0]);
    }
    out
}
fn main() -> Result<()> {
    let args: Vec<_> = std::env::args().collect();
    ensure!(
        args.len() >= 3,
        "input.json output.json [iterations] [scenario]"
    );
    let mut input: Input = serde_json::from_slice(&fs::read(&args[1])?)?;
    let iterations = args.get(3).map(|v| v.parse()).transpose()?.unwrap_or(80);
    let scenario = args.get(4).map(String::as_str).unwrap_or("free");
    if scenario == "density-high" {
        input.shape_density_m3 = 1e12;
    }
    if scenario == "density-low" {
        input.shape_density_m3 = 1e10;
    }
    if scenario == "density-very-low" {
        input.shape_density_m3 = 1e9;
    }
    if scenario == "ideal-receiver" {
        input.fir = vec![(0., [1., 0.])];
    }
    if scenario == "frequency-fine" {
        input.frequency_step_hz /= 2.;
    }
    if let Some(ns) = &input.shape_density_at_knots_m3 {
        ensure!(
            ns.len() == input.knot_range_km.len() && ns.iter().all(|n| n.is_finite() && *n > 0.),
            "invalid prescribed density profile"
        );
    }
    let dt = 1. / input.sample_rate_hz;
    let count = (input.frequency_limit_hz / input.frequency_step_hz).round() as isize;
    let frequency: Vec<f64> = (-count..=count)
        .map(|i| i as f64 * input.frequency_step_hz)
        .collect();
    let maxlag = input.products.iter().map(|p| p.lag).max().unwrap();
    let half = (input.fir.iter().map(|(t, _)| t.abs()).fold(0., f64::max) / dt).round() as usize;
    let offset = (2 * half) as isize;
    let phasors: Vec<Vec<C>> = (-(2 * half as isize)..=maxlag as isize + 2 * half as isize)
        .map(|i| {
            frequency
                .iter()
                .map(|&f| C::from_polar(1., -std::f64::consts::TAU * f * i as f64 * dt))
                .collect()
        })
        .collect();
    let fir: Vec<_> = input
        .fir
        .iter()
        .map(|(t, z)| (*t, C::new(z[0], z[1])))
        .collect();
    let mut cells = vec![];
    for edge in input.cell_edges_km.windows(2) {
        let edges = [edge[0] / 149896.229, edge[1] / 149896.229];
        let mut terms = vec![];
        for (m, p) in input.products.iter().enumerate() {
            let t = input.gate_front_s + p.sample as f64 * dt;
            let tau = p.lag as f64 * dt;
            let padding = half as f64 * dt;
            if edges[1] < t + tau - input.pulse_width_s - padding || edges[0] > t + padding {
                continue;
            }
            let k = kernel(
                t,
                tau,
                input.pulse_width_s,
                edges,
                &fir,
                &frequency,
                dt,
                &phasors,
                offset,
            );
            if k.iter().any(|z| z.norm_sqr() > 0.) {
                terms.push(Term {
                    measurement: m,
                    kernel: k,
                });
            }
        }
        if !terms.is_empty() {
            let mut theory = if let (Some(az), Some(el)) = (
                input.provenance["azimuth_deg"].as_f64(),
                input.provenance["elevation_deg"].as_f64(),
            ) {
                Theory::from_beam((edge[0] + edge[1]) / 2., az, el, input.frequency_step_hz)?
            } else {
                Theory::new((edge[0] + edge[1]) / 2. + 0.05, input.frequency_step_hz)?
            };
            theory.electron_magnetized = scenario != "unmagnetized" && scenario != "other-theory";
            theory.include_electron_fluctuations = scenario != "other-theory";
            cells.push(Cell {
                theory: Arc::new(theory),
                shape_density_m3: input.shape_density_at_knots_m3.as_ref().map(|ns| {
                    weights((edge[0] + edge[1]) / 2., &input.knot_range_km)
                        .iter()
                        .map(|&(i, w)| w * ns[i].ln())
                        .sum::<f64>()
                        .exp()
                }),
                knot_weights: weights((edge[0] + edge[1]) / 2., &input.knot_range_km),
                terms,
            });
        }
    }
    eprintln!(
        "{} cells, {} measured products, {} frequencies",
        cells.len(),
        input.products.len(),
        frequency.len()
    );
    let problem = Problem {
        frequency_hz: frequency,
        radar_frequency_hz: input.radar_frequency_hz,
        shape_density_m3: input.shape_density_m3,
        cells,
        products: input
            .products
            .iter()
            .map(|p| Product {
                value: p.value,
                sigma: p.sigma,
            })
            .collect(),
        knot_count: input.knot_range_km.len(),
        background_basis: input
            .products
            .iter()
            .map(|p| {
                C::new(
                    input.background_basis[p.lag][0],
                    input.background_basis[p.lag][1],
                )
            })
            .collect(),
        background_sigma: input.background_sigma,
    };
    let mut fixed = vec![];
    if !input.prescribed_node_fields.is_empty() {
        ensure!(
            input.prescribed_node_fields.len() == problem.knot_count,
            "prescribed-field dimensions"
        );
        let encoded = syisr_hplus_night::coupled::encode(&input.initial);
        for (k, row) in input.prescribed_node_fields.iter().enumerate() {
            for (f, &held) in row.iter().enumerate() {
                if held {
                    fixed.push((6 * k + f, encoded[6 * k + f]));
                }
            }
        }
    }
    if scenario == "no-hydrogen" {
        for k in 0..problem.knot_count {
            fixed.push((6 * k + 4, 0.));
        }
    }
    if scenario == "no-helium" {
        for k in 0..problem.knot_count {
            fixed.push((6 * k + 5, 0.));
        }
    }
    if scenario == "tied-temperatures" {
        return Err(anyhow::anyhow!("temperature tie not implemented"));
    }
    if scenario.starts_with("profile-") {
        let parts: Vec<_> = scenario.split('-').collect();
        ensure!(parts.len() == 3, "profile-knot-fraction");
        let k: usize = parts[1].parse()?;
        let h: f64 = parts[2].parse()?;
        ensure!(
            k < problem.knot_count && (0. ..=1.).contains(&h),
            "profile bounds"
        );
        fixed.push((6 * k + 4, h));
    }
    let mut runs = vec![];
    let starts = if scenario.starts_with("profile-") {
        vec![(0.2, 1000.), (0.2, 1001.)]
    } else if scenario == "helium-rich" {
        vec![(0.05, 350.)]
    } else if scenario == "free" {
        vec![(0.05, 800.), (0.4, 1300.), (0.8, 2000.), (0.02, 400.)]
    } else {
        vec![(0.2, 1000.)]
    };
    for (h, ti) in starts {
        let initial: Vec<_> = input
            .initial
            .iter()
            .map(|n| {
                let mut n = n.clone();
                if scenario.starts_with("profile-")
                    || scenario == "warm"
                    || scenario == "unmagnetized"
                    || scenario == "other-theory"
                {
                    if scenario.starts_with("profile-") && ti == 1001. {
                        n.he_fraction = 0.5 * (1. - n.h_fraction);
                        n.o_fraction = 1. - n.h_fraction - n.he_fraction;
                        n.ti_k *= 0.7;
                        n.te_k *= 1.2;
                    }
                    return n;
                }
                n.h_fraction = h;
                n.he_fraction = (1. - h)
                    * if scenario == "helium-rich" || ti == 400. {
                        0.8
                    } else {
                        0.05
                    };
                n.o_fraction = 1. - n.h_fraction - n.he_fraction;
                n.ti_k = ti;
                n.te_k = ti * 1.5;
                n
            })
            .collect();
        let fit = problem.solve_with_tolerance(
            &initial,
            iterations,
            &fixed,
            input.score_tolerance.unwrap_or(1e-3),
        )?;
        eprintln!(
            "start H={h} Ti={ti}: {} chi2={} iterations={}",
            fit.status, fit.chi_square, fit.iterations
        );
        runs.push(fit);
    }
    let best = runs
        .iter()
        .enumerate()
        .min_by(|(_, a), (_, b)| a.chi_square.total_cmp(&b.chi_square))
        .unwrap()
        .0;
    let output = json!({"schema":"syisr-hplus-combined-v2","scenario":scenario,"provenance":input.provenance,"knot_range_km":input.knot_range_km,"shape_density_m3":input.shape_density_m3,"shape_density_at_knots_m3":input.shape_density_at_knots_m3,"frequency_step_hz":input.frequency_step_hz,"score_tolerance":input.score_tolerance,"prescribed_node_fields":input.prescribed_node_fields,"products":input.products.iter().map(|p|json!({"sample":p.sample,"lag":p.lag,"value":p.value,"sigma":p.sigma})).collect::<Vec<_>>(),"best_run":best,"runs":runs});
    fs::write(&args[2], serde_json::to_vec(&output)?)?;
    Ok(())
}

#[cfg(test)]
mod tests {
    use super::*;
    #[test]
    fn optimized_kernel_matches_exact_filtered_range_response() {
        let fir = vec![
            (-1e-5, C::new(0.2, 0.1)),
            (0., C::new(0.8, 0.)),
            (1e-5, C::new(0.1, -0.1)),
        ];
        let f = vec![-15000., -5000., 5000., 15000.];
        let dt = 1e-5;
        let phasors: Vec<Vec<C>> = (-2..=12)
            .map(|i| {
                f.iter()
                    .map(|&v| C::from_polar(1., -std::f64::consts::TAU * v * i as f64 * dt))
                    .collect()
            })
            .collect();
        let k = kernel(
            0.001,
            0.0001,
            0.00048,
            [0.00065, 0.0008],
            &fir,
            &f,
            dt,
            &phasors,
            2,
        );
        let exact = isr_ion::pulse_response::rectangular_cell(
            0.001,
            0.0001,
            0.00048,
            [0.00065, 0.0008],
            &fir,
            &f,
        )
        .unwrap();
        for (i, (a, b)) in k.iter().zip(exact).enumerate() {
            let w = 10000. * if i == 0 || i == 3 { 0.5 } else { 1. };
            assert!((*a - b * w).norm() < 1e-8);
        }
    }
}
