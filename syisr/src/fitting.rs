//! Independent gated fits, covariance shrinkage, complete-code-cycle block bootstrap.
use crate::{
    campaign::{load, save, Campaign, Job},
    coupled::{Cell, Node, Problem, Product, Term},
    decode::{ProductDwell, NL},
    screen::median,
    theory::Theory,
};
use anyhow::{ensure, Result};
use nalgebra::{DMatrix as M, DVector as V};
use num_complex::Complex64 as C;
use serde::{Deserialize, Serialize};
use std::{path::Path, sync::Arc};
#[derive(Clone, Serialize, Deserialize)]
pub struct Gate {
    pub altitude_km: f64,
    pub range_km: f64,
    pub width_km: f64,
    pub node: Option<Node>,
    pub ne_m3: Option<f64>,
    pub lower68: Vec<Option<f64>>,
    pub upper68: Vec<Option<f64>>,
    pub lower95: Vec<Option<f64>>,
    pub upper95: Vec<Option<f64>>,
    pub bootstrap_converged: usize,
    pub bootstrap_requested: usize,
    pub flags: Vec<String>,
    pub chi_square: Option<f64>,
    pub observed: Vec<[f64; 2]>,
    pub prediction: Vec<[f64; 2]>,
    pub conditional_standard_error: Vec<Option<f64>>,
    pub h_profile: Vec<[f64; 2]>,
    pub bootstrap_nodes: Vec<Node>,
}
#[derive(Clone, Serialize, Deserialize)]
pub struct Profile {
    pub schema: String,
    pub job: Job,
    pub exposure_s: f64,
    pub cycles: usize,
    pub screen_fraction: f64,
    pub calibration_gain: Option<f64>,
    pub calibration_fractional_error: Option<f64>,
    pub gates: Vec<Gate>,
    pub bootstrap: String,
    pub source_dwells: Vec<usize>,
    pub altitude_edges_km: Vec<f64>,
}
fn gram(codes: &[Vec<f64>]) -> Vec<f64> {
    let mut g = vec![0.; 1024 * 1024];
    for a in 0..1024 {
        for b in 0..1024 {
            g[a * 1024 + b] = (0..64)
                .map(|c| codes[c][a / 32] * codes[c][a % 32] * codes[c][b / 32] * codes[c][b % 32])
                .sum::<f64>()
                / 64.;
        }
    }
    g
}
/// Integrated homogeneous coded-response coefficients. Exact sample-cell quadrature
/// is summed analytically over source delay; all filter cross-lags are retained.
pub fn kernel(codes: &[Vec<f64>], noise: &[C]) -> Vec<Vec<C>> {
    let g = gram(codes);
    let nfreq = 201;
    let mut rho = vec![C::default(); 11];
    for l in 0..11 {
        rho[l] =
            noise[l] / noise[0].re * (0.5 + 0.5 * (std::f64::consts::PI * l as f64 / 11.).cos());
    }
    let psd: Vec<_> = (0..512)
        .map(|j| {
            let f = j as f64 / 512.;
            (1. + (1..11)
                .map(|l| 2. * (rho[l] * C::from_polar(1., std::f64::consts::TAU * f * l as f64)).re)
                .sum::<f64>())
            .max(1e-8)
        })
        .collect();
    let taps: Vec<C> = (-10..=10)
        .map(|t| {
            (0..512)
                .map(|j| {
                    C::from_polar(
                        psd[j].sqrt() / 512.,
                        std::f64::consts::TAU * j as f64 * t as f64 / 512.,
                    )
                })
                .sum()
        })
        .collect();
    (1..=NL)
        .map(|lag| {
            let branches: Vec<_> = [lag / 2, (lag + 1) / 2]
                .into_iter()
                .filter(|l| *l > 0)
                .collect();
            let mut coeff = vec![C::default(); 41];
            for l in &branches {
                for a in -10i32..=10 {
                    for b in -10i32..=10 {
                        let tau = lag as i32 + a - b;
                        let mut overlap = 0.;
                        for baud in 0..32 - l {
                            let p = baud * 32 + baud + l;
                            for sample in 0i32..64 {
                                let j = (sample + tau).div_euclid(2);
                                if (0..32).contains(&j) {
                                    let i = sample as usize / 2;
                                    overlap += g[p * 1024 + i * 32 + j as usize];
                                }
                            }
                        }
                        coeff[(a - b + 20) as usize] += taps[(a + 10) as usize]
                            * taps[(b + 10) as usize].conj()
                            * overlap
                            * (if lag % 2 == 1 { 2. } else { 1. })
                            / (2. * branches.len() as f64 * (32 - l) as f64);
                    }
                }
            }
            (0..nfreq)
                .map(|j| {
                    let f = -50000. + j as f64 * 500.;
                    coeff
                        .iter()
                        .enumerate()
                        .map(|(k, c)| {
                            *c * C::from_polar(
                                1.,
                                -std::f64::consts::TAU
                                    * f
                                    * (lag as i32 + k as i32 - 20) as f64
                                    * 1e-5,
                            )
                        })
                        .sum()
                })
                .collect()
        })
        .collect()
}
fn make(theory: Arc<Theory>, ne: f64, kernel: &[Vec<C>], data: &[f64]) -> Problem {
    let nf = 201;
    Problem {
        frequency_hz: (0..nf).map(|j| -50000. + j as f64 * 500.).collect(),
        radar_frequency_hz: 430.2e6,
        shape_density_m3: ne,
        cells: vec![Cell {
            theory,
            shape_density_m3: Some(ne),
            density_calibration: None,
            h_knot_weights: None,
                composition: None,
            knot_weights: vec![(0, 1.)],
            terms: kernel
                .iter()
                .enumerate()
                .map(|(i, k)| Term {
                    measurement: i,
                    kernel: k
                        .iter()
                        .enumerate()
                        .map(|(j, z)| *z * 500. * if j == 0 || j + 1 == nf { 0.5 } else { 1. })
                        .collect(),
                })
                .collect(),
        }],
        products: data
            .iter()
            .map(|x| Product {
                value: [*x, 0.],
                sigma: [1., 1e30],
            })
            .collect(),
        knot_count: 1,
        background_basis: vec![C::default(); data.len()],
        background_sigma: 1.,
    }
}
fn initial(power: f64, h: f64, ti: f64) -> Node {
    Node {
        power_adu2: power.max(1e-5),
        ti_k: ti,
        te_k: 1800.,
        velocity_m_s: 0.,
        h_fraction: h,
        he_fraction: 0.,
        o_fraction: 1. - h,
    }
}
fn solve(p: &Problem, start: &Node, freeh: bool, robust: bool) -> Result<crate::coupled::Fit> {
    let fixed = if freeh {
        vec![(5, 0.)]
    } else {
        vec![(4, 0.), (5, 0.)]
    };
    let mut fits = vec![p.solve_with_tolerance(&[start.clone()], 200, &fixed, 0.005)?];
    if robust {
        for (h, ti) in if freeh {
            vec![(0., 800.), (0.5, 1500.), (0.95, 1500.)]
        } else {
            vec![(0., 1500.)]
        } {
            fits.push(p.solve_with_tolerance(
                &[initial(start.power_adu2, h, ti)],
                200,
                &fixed,
                0.005,
            )?);
        }
    }
    fits.into_iter()
        .min_by(|a, b| a.chi_square.total_cmp(&b.chi_square))
        .ok_or_else(|| anyhow::anyhow!("no fit"))
}
fn q(v: &mut [f64], p: f64) -> Option<f64> {
    if v.is_empty() {
        return None;
    }
    v.sort_by(f64::total_cmp);
    let i = p * (v.len() - 1) as f64;
    let j = i.floor() as usize;
    Some(v[j] * (1. - i.fract()) + v[(j + 1).min(v.len() - 1)] * i.fract())
}
struct Rng(u64);
impl Rng {
    fn next(&mut self) -> f64 {
        self.0 = self.0.wrapping_mul(6364136223846793005).wrapping_add(1);
        ((self.0 >> 32) as u32 as f64 + 0.5) / 4294967296.
    }
    fn normal(&mut self) -> f64 {
        (-2. * self.next().ln()).sqrt() * (std::f64::consts::TAU * self.next()).cos()
    }
}
fn density(t: &Theory, n: &Node, gain: f64, r: f64) -> f64 {
    let target = gain * n.power_adu2 * r * r;
    let (mut lo, mut hi) = (1e6f64.ln(), 1e14f64.ln());
    for _ in 0..35 {
        let ne = ((lo + hi) * 0.5).exp();
        let mass = t
            .spectrum_with_mass(n.ti_k, n.te_k, n.velocity_m_s, n.h_fraction, 0., ne)
            .1;
        if ne * mass < target {
            lo = ne.ln()
        } else {
            hi = ne.ln()
        }
    }
    ((lo + hi) * 0.5).exp()
}
pub fn fit(
    c: &Campaign,
    job: &Job,
    root: &Path,
    codes: &[Vec<f64>],
    nboot: usize,
) -> Result<Profile> {
    let dwells: Vec<ProductDwell> = job
        .dwells
        .iter()
        .map(|d| load(&root.join("decoded").join(format!("{d:06}.h5"))))
        .collect::<Result<_>>()?;
    let ng = dwells[0].heights.len();
    ensure!(
        dwells
            .iter()
            .all(|d| d.schema == "syisr-ac-native-adaptive-dwell-v2"
                && d.heights == dwells[0].heights),
        "incompatible gate grid/schema; no legacy mixing"
    );
    let nc: usize = dwells.iter().map(|d| d.cycles).sum();
    ensure!(nc >= 8, "insufficient code cycles");
    let exposure = dwells.iter().map(|d| d.pulses as f64 * 0.02133).sum();
    let masked: usize = dwells.iter().map(|d| d.mask_summary.masked).sum();
    let total: usize = dwells.iter().map(|d| d.mask_summary.total).sum();
    let mut noise = vec![C::default(); NL + 1];
    for d in &dwells {
        for l in 0..=NL {
            noise[l] += C::new(d.noise[l][0], d.noise[l][1]) * d.pulses as f64;
        }
    }
    let pulses: usize = dwells.iter().map(|d| d.pulses).sum();
    for v in &mut noise {
        *v /= pulses as f64;
    }
    let response = kernel(codes, &noise);
    let mut gates = vec![];
    let mut calibration = vec![];
    let mut gainscatter = vec![];
    for gate in 0..ng {
        let height = dwells[0].heights[gate];
        let range = dwells[0].ranges[gate];
        let mut flags = vec![];
        if height < 200. {
            flags.push("molecular_species_omitted".into())
        }
        let mut obs = vec![C::default(); NL];
        let mut count = vec![0usize; NL];
        let mut groups: Vec<Vec<f64>> = vec![];
        for d in &dwells {
            for cycle in 0..d.cycles {
                for l in 0..NL {
                    let i = cycle * ng * NL + gate * NL + l;
                    if d.valid[i] {
                        obs[l] += C::new(d.values[i][0], d.values[i][1]);
                        count[l] += 1;
                    }
                }
            }
            // Blocks never cross a beam revisit or an acquisition gap.
            for first in (0..d.cycles).step_by(4) {
                let end = (first + 4).min(d.cycles);
                if end - first < 2 {
                    continue;
                }
                let mut v = vec![0.; NL * 2];
                let mut good = true;
                for l in 0..NL {
                    let mut s = C::default();
                    let mut n = 0;
                    for cycle in first..end {
                        let i = cycle * ng * NL + gate * NL + l;
                        if d.valid[i] {
                            s += C::new(d.values[i][0], d.values[i][1]);
                            n += 1;
                        }
                    }
                    if n < 2 {
                        good = false;
                        break;
                    }
                    s /= n as f64;
                    v[l] = s.re;
                    v[l + NL] = s.im;
                }
                if good {
                    groups.push(v)
                }
            }
        }
        let observed: Vec<_> = obs
            .iter()
            .zip(&count)
            .map(|(s, n)| if *n > 0 { *s / *n as f64 } else { C::default() })
            .collect();
        let raw: Vec<_> = observed
            .iter()
            .map(|v| v.re)
            .chain(observed.iter().map(|v| v.im))
            .collect();
        if groups.len() < 6 || count.iter().any(|n| *n < 8) {
            gates.push(Gate {
                altitude_km: height,
                range_km: range,
                width_km: dwells[0].widths[gate],
                node: None,
                ne_m3: None,
                lower68: vec![None; 5],
                upper68: vec![None; 5],
                lower95: vec![None; 5],
                upper95: vec![None; 5],
                bootstrap_converged: 0,
                bootstrap_requested: nboot,
                flags: vec!["insufficient_independent_blocks".into()],
                chi_square: None,
                observed: observed.iter().map(|v| [v.re, v.im]).collect(),
                prediction: vec![],
                conditional_standard_error: vec![],
                h_profile: vec![],
                bootstrap_nodes: vec![],
            });
            continue;
        }
        let m = NL * 2;
        let nb = groups.len();
        let mean: Vec<f64> = (0..m)
            .map(|j| groups.iter().map(|g| g[j]).sum::<f64>() / nb as f64)
            .collect();
        let mut cov = M::zeros(m, m);
        for group in &groups {
            for i in 0..m {
                for j in 0..m {
                    cov[(i, j)] +=
                        (group[i] - mean[i]) * (group[j] - mean[j]) / ((nb - 1) * nb) as f64;
                }
            }
        }
        let mut diagonal: Vec<_> = (0..m).map(|i| cov[(i, i)]).collect();
        let floor = median(&mut diagonal).max(1e-8) * 1e-4;
        for i in 0..m {
            for j in 0..m {
                if i != j {
                    cov[(i, j)] *= 0.5;
                }
            }
            cov[(i, i)] += floor;
        }
        let whiten = cov
            .cholesky()
            .ok_or_else(|| anyhow::anyhow!("covariance not positive"))?
            .l()
            .try_inverse()
            .ok_or_else(|| anyhow::anyhow!("whitening inverse"))?;
        let data = &whiten * V::from_vec(raw);
        let kernel: Vec<Vec<C>> = (0..m)
            .map(|i| {
                (0..201)
                    .map(|j| {
                        (0..NL)
                            .map(|l| C::new(whiten[(i, l)], -whiten[(i, l + NL)]) * response[l][j])
                            .sum()
                    })
                    .collect()
            })
            .collect();
        let theory = match Theory::from_beam(range, job.az, job.el, 500.) {
            Ok(t) => Arc::new(t),
            Err(_) => {
                gates.push(Gate {
                    altitude_km: height,
                    range_km: range,
                    width_km: dwells[0].widths[gate],
                    node: None,
                    ne_m3: None,
                    lower68: vec![None; 5],
                    upper68: vec![None; 5],
                    lower95: vec![None; 5],
                    upper95: vec![None; 5],
                    bootstrap_converged: 0,
                    bootstrap_requested: nboot,
                    flags: vec!["unsupported_near_perpendicular_electron_response".into()],
                    chi_square: None,
                    observed: observed.iter().map(|v| [v.re, v.im]).collect(),
                    prediction: vec![],
                    conditional_standard_error: vec![],
                    h_profile: vec![],
                    bootstrap_nodes: vec![],
                });
                continue;
            }
        };
        let mut refne: Vec<f64> = dwells
            .iter()
            .filter_map(|d| d.reference_ne[gate])
            .filter(|v| v.is_finite() && *v > 1e7)
            .collect();
        let reference = if refne.is_empty() {
            None
        } else {
            Some(median(&mut refne))
        };
        let mut ne = reference.unwrap_or(1e11);
        let free = height > 500.;
        let mut problem = make(theory.clone(), ne, &kernel, data.as_slice());
        let mut fitted = solve(
            &problem,
            &initial(observed[0].re, if free { 0.2 } else { 0. }, 1000.),
            free,
            true,
        )?;
        // One empirical system gain is derived from available L2 density anchors.
        if (300.0..=450.0).contains(&height) {
            if let Some(reference) = reference {
                let n = &fitted.nodes[0];
                let mass = theory
                    .spectrum_with_mass(n.ti_k, n.te_k, n.velocity_m_s, n.h_fraction, 0., reference)
                    .1;
                let gain = reference * mass / (n.power_adu2 * range * range);
                if gain.is_finite() && gain > 0. {
                    calibration.push(gain);
                }
            }
        }
        let gain = if calibration.len() >= 3 {
            let mut v = calibration.clone();
            Some(median(&mut v))
        } else {
            None
        };
        if let Some(gain) = gain {
            for _ in 0..2 {
                ne = density(&theory, &fitted.nodes[0], gain, range);
                problem = make(theory.clone(), ne, &kernel, data.as_slice());
                fitted = solve(&problem, &fitted.nodes[0], free, false)?;
            }
        }
        let mut h_profile = vec![];
        if free {
            for h in [0., 0.25, 0.5, 0.75, 1.] {
                let mut start = fitted.nodes[0].clone();
                start.h_fraction = h;
                start.o_fraction = 1. - h;
                let f = problem.solve_with_tolerance(&[start], 200, &[(4, h), (5, 0.)], 0.005)?;
                h_profile.push([h, f.chi_square]);
                if f.chi_square < fitted.chi_square {
                    fitted = solve(&problem, &f.nodes[0], true, true)?;
                }
            }
        }
        let n = fitted.nodes[0].clone();
        if fitted.status != "converged" {
            flags.push("not_converged".into())
        }
        if n.ti_k < 155.
            || n.ti_k > 5900.
            || n.te_k < 155.
            || n.te_k > 11800.
            || n.velocity_m_s.abs() > 1990.
        {
            flags.push("parameter_bound".into())
        }
        if fitted.reduced_chi_square > 3. {
            flags.push("large_working_residual".into())
        }
        let mut rng = Rng((job.start as u64) ^ (gate as u64 + 71) * 91939);
        let mut draws: Vec<Vec<f64>> = vec![];
        let mut bootstrap_nodes = vec![];
        for draw in 0..nboot {
            let mut sample = vec![0.; m];
            for _ in 0..nb {
                let group = &groups[(rng.next() * nb as f64) as usize % nb];
                for j in 0..m {
                    sample[j] += group[j] / nb as f64;
                }
            }
            let y = &whiten * V::from_vec(sample);
            let bp = make(theory.clone(), ne, &kernel, y.as_slice());
            let mut start = n.clone();
            if free && draw % 2 == 1 {
                start.h_fraction = 1. - start.h_fraction;
                start.o_fraction = 1. - start.h_fraction;
                start.ti_k = 1000.;
            }
            let f = solve(&bp, &start, free, free)?;
            if f.status == "converged" {
                let node = &f.nodes[0];
                bootstrap_nodes.push(node.clone());
                let gainfactor = (0.15 * rng.normal()).exp();
                let density = if let Some(gain) = gain {
                    density(&theory, node, gain * gainfactor, range)
                } else {
                    f64::NAN
                };
                draws.push(vec![
                    density,
                    node.ti_k,
                    node.te_k,
                    node.velocity_m_s,
                    node.h_fraction,
                ]);
            }
        }
        if draws.len() * 5 < nboot * 4 {
            flags.push("bootstrap_failures".into())
        }
        let mut lo68 = vec![];
        let mut hi68 = vec![];
        let mut lo95 = vec![];
        let mut hi95 = vec![];
        for j in 0..5 {
            let mut v: Vec<_> = draws
                .iter()
                .map(|a| a[j])
                .filter(|v| v.is_finite())
                .collect();
            lo68.push(q(&mut v, 0.16));
            hi68.push(q(&mut v, 0.84));
            lo95.push(q(&mut v, 0.025));
            hi95.push(q(&mut v, 0.975));
        }
        if let Some(se) = fitted.conditional_standard_error[0] {
            if n.power_adu2 < 3. * se {
                flags.push("weak_power".into())
            }
        }
        if hi95[1]
            .zip(lo95[1])
            .map(|(a, b)| a - b > n.ti_k * 2.)
            .unwrap_or(true)
            || hi95[2]
                .zip(lo95[2])
                .map(|(a, b)| a - b > n.te_k * 2.)
                .unwrap_or(true)
        {
            flags.push("wide_joint_uncertainty".into())
        }
        let (spec, _) =
            theory.spectrum_with_mass(n.ti_k, n.te_k, n.velocity_m_s, n.h_fraction, 0., ne);
        let prediction = response
            .iter()
            .map(|k| {
                let v: C = k
                    .iter()
                    .zip(&spec)
                    .enumerate()
                    .map(|(j, (a, b))| {
                        *a * *b * n.power_adu2 * 500. * if j == 0 || j == 200 { 0.5 } else { 1. }
                    })
                    .sum();
                [v.re, v.im]
            })
            .collect();
        gates.push(Gate {
            altitude_km: height,
            range_km: range,
            width_km: dwells[0].widths[gate],
            node: Some(n),
            ne_m3: gain.map(|g| density(&theory, &fitted.nodes[0], g, range)),
            lower68: lo68,
            upper68: hi68,
            lower95: lo95,
            upper95: hi95,
            bootstrap_converged: draws.len(),
            bootstrap_requested: nboot,
            flags,
            chi_square: Some(fitted.chi_square),
            observed: observed.iter().map(|v| [v.re, v.im]).collect(),
            prediction,
            conditional_standard_error: fitted.conditional_standard_error,
            h_profile,
            bootstrap_nodes,
        });
    }
    // One final robust gain and its uncertainty apply to every gate, not a
    // different running median as gates are traversed.
    let gain = if calibration.len() >= 3 {
        Some(median(&mut calibration))
    } else {
        None
    };
    let fraction = if let Some(gain) = gain {
        gainscatter = calibration.iter().map(|v| (v / gain - 1.).abs()).collect();
        Some((1.4826 * median(&mut gainscatter)).max(0.15))
    } else {
        None
    };
    if let Some(gain) = gain {
        for (i, g) in gates.iter_mut().enumerate() {
            if let Some(n) = &g.node {
                if let Ok(t) = Theory::from_beam(g.range_km, job.az, job.el, 500.) {
                    g.ne_m3 = Some(density(&t, n, gain, g.range_km));
                    let mut random = Rng((job.start as u64) ^ (i as u64 + 733) * 314159);
                    let mut values: Vec<f64> = g
                        .bootstrap_nodes
                        .iter()
                        .map(|node| {
                            density(
                                &t,
                                node,
                                gain * (fraction.unwrap() * random.normal()).exp(),
                                g.range_km,
                            )
                        })
                        .collect();
                    g.lower68[0] = q(&mut values, 0.16);
                    g.upper68[0] = q(&mut values, 0.84);
                    g.lower95[0] = q(&mut values, 0.025);
                    g.upper95[0] = q(&mut values, 0.975);
                }
            }
        }
    } else {
        for g in &mut gates {
            g.flags.push("no_empirical_density_calibration".into());
        }
    }
    let result=Profile{schema:"syisr-ac-native-adaptive-profile-v2".into(),job:job.clone(),exposure_s:exposure,cycles:nc,screen_fraction:masked as f64/total as f64,calibration_gain:gain,calibration_fractional_error:fraction,gates,bootstrap:"Complete64-code cycles in4-cycle blocks,never across dwells;all nuisance parameters refit;50percent covariance shrinkage;empirical gain scatter with15percent systematic floor; common final gain;conditional on H/O model and receiver response;not absolute confidence calibration".into(),source_dwells:job.dwells.clone(),altitude_edges_km:dwells[0].altitude_edges_km.clone()};
    save(&root.join("fits").join(format!("{}.h5", job.id)), &result)?;
    let _ = c;
    Ok(result)
}
