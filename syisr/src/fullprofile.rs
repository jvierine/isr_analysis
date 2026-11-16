//! Joint spline-profile inversion of the original signed decoded complex lag products.
//! All native source cells enter the gate integrals; knots never replace gate geometry.
use crate::{
    ac_range,
    campaign::{self, Campaign, Job},
    coupled::{encode, Cell, Fit, Node, Problem, Product, Term},
    decode::{ProductDwell, NL},
    fitting::{Gate, Profile},
    grid,
    spline::{h_weight, Spline},
    theory::Theory,
};
use anyhow::{ensure, Result};
use num_complex::Complex64 as C;
use serde::Serialize;
use std::{collections::BTreeMap, path::Path, sync::Arc};
#[derive(Clone)]
struct Block {
    dwell: usize,
    sum: Vec<C>,
    count: Vec<usize>,
}
#[derive(Serialize)]
pub struct Joint {
    #[serde(flatten)]
    pub profile: Profile,
    pub spline: Spline,
    pub control_altitude_km: Vec<f64>,
    #[serde(serialize_with = "serialize_controls")]
    pub control_nodes: Vec<Node>,
    #[serde(serialize_with = "serialize_fit")]
    pub nominal_fit: Fit,
    #[serde(serialize_with = "serialize_bootstrap_controls")]
    pub bootstrap_control_nodes: Vec<Vec<Node>>,
    pub product_sigma: Vec<[f64; 2]>,
    pub bootstrap_status: Vec<String>,
    pub bootstrap_ne_m3: Vec<Vec<f64>>,
    pub likelihood: String,
    pub forward_model: String,
}
fn control(n: &Node) -> serde_json::Value {
    serde_json::json!({"ne_m3":n.power_adu2*1e11,"ti_k":n.ti_k,"te_k":n.te_k,"velocity_m_s":n.velocity_m_s,"h_coefficient":n.h_fraction})
}
fn serialize_controls<S: serde::Serializer>(
    v: &[Node],
    s: S,
) -> std::result::Result<S::Ok, S::Error> {
    v.iter().map(control).collect::<Vec<_>>().serialize(s)
}
fn serialize_bootstrap_controls<S: serde::Serializer>(
    v: &[Vec<Node>],
    s: S,
) -> std::result::Result<S::Ok, S::Error> {
    v.iter()
        .map(|n| n.iter().map(control).collect::<Vec<_>>())
        .collect::<Vec<_>>()
        .serialize(s)
}
fn serialize_fit<S: serde::Serializer>(v: &Fit, s: S) -> std::result::Result<S::Ok, S::Error> {
    serde_json::json!({"status":v.status,"iterations":v.iterations,"chi_square":v.chi_square,"degrees_of_freedom":v.degrees_of_freedom,"reduced_chi_square":v.reduced_chi_square,"normalized_gradient":v.normalized_gradient,"active_bounds":v.active_bounds,"objective_history":v.objective_history,"transformed_covariance":v.transformed_covariance,"control_coordinates":"control-major log(Ne/1e11),log(Ti/1000),log(Te/1000),vi/1000,H coefficient,He coefficient fixed zero; column-major covariance; conditional working curvature"}).serialize(s)
}
fn sample(s: &Spline, n: &[Node], z: f64) -> Node {
    let x = encode(n);
    let mut p = [0.; 6];
    for (k, w) in s.weights(z) {
        for f in 0..6 {
            p[f] += x[6 * k + f] * w;
        }
    }
    let h = p[4] * h_weight(z);
    Node {
        power_adu2: p[0].exp(),
        ti_k: 1000. * p[1].exp(),
        te_k: 1000. * p[2].exp(),
        velocity_m_s: 1000. * p[3],
        h_fraction: h,
        he_fraction: 0.,
        o_fraction: 1. - h,
    }
}
fn amplitude(problem: &Problem, range: f64, n: &Node, ne: f64, gain: f64) -> f64 {
    let cell = problem
        .cells
        .iter()
        .min_by(|a, b| {
            let ra = a.density_calibration.unwrap().1;
            let rb = b.density_calibration.unwrap().1;
            (ra - range).abs().total_cmp(&(rb - range).abs())
        })
        .unwrap();
    let mass = cell
        .theory
        .spectrum_with_mass(n.ti_k, n.te_k, n.velocity_m_s, n.h_fraction, 0., ne)
        .1;
    ne * mass / (gain * range * range)
}
fn percentile(v: &mut Vec<f64>, q: f64) -> Option<f64> {
    v.retain(|v| v.is_finite());
    if v.is_empty() {
        return None;
    }
    v.sort_by(f64::total_cmp);
    Some(v[((v.len() - 1) as f64 * q).round() as usize])
}
fn mean(blocks: &[Block], indices: &[usize], m: usize) -> (Vec<C>, Vec<usize>) {
    let mut sums = vec![C::default(); m];
    let mut n = vec![0; m];
    for i in indices {
        for j in 0..m {
            sums[j] += blocks[*i].sum[j];
            n[j] += blocks[*i].count[j];
        }
    }
    for j in 0..m {
        if n[j] > 0 {
            sums[j] /= n[j] as f64;
        }
    }
    (sums, n)
}
struct Rng(u64);
impl Rng {
    fn next(&mut self) -> f64 {
        self.0 ^= self.0 << 13;
        self.0 ^= self.0 >> 7;
        self.0 ^= self.0 << 17;
        (self.0 >> 11) as f64 / ((1u64 << 53) as f64)
    }
    fn normal(&mut self) -> f64 {
        (-2. * self.next().max(1e-12).ln()).sqrt() * (std::f64::consts::TAU * self.next()).cos()
    }
}
pub fn fit(
    c: &Campaign,
    job: &Job,
    root: &Path,
    out: &Path,
    codes: &[Vec<f64>],
    nboot: usize,
) -> Result<Joint> {
    let baseline: Profile = campaign::load(&root.join("fits").join(format!("{}.h5", job.id)))?;
    let gain = baseline
        .calibration_gain
        .ok_or_else(|| anyhow::anyhow!("No empirical density gain for this profile"))?;
    let gain_error = baseline
        .calibration_fractional_error
        .unwrap_or(0.15)
        .max(0.15);
    let dwells: Vec<ProductDwell> = job
        .dwells
        .iter()
        .map(|d| campaign::load(&root.join("decoded").join(format!("{d:06}.h5"))))
        .collect::<Result<_>>()?;
    let ng = dwells[0].heights.len();
    let m = ng * NL;
    ensure!(
        dwells
            .iter()
            .all(|d| d.heights == dwells[0].heights
                && d.schema == "syisr-ac-native-adaptive-dwell-v2"),
        "native/adaptive data only"
    );
    let mut blocks = vec![];
    let mut noise = vec![[0.; 2]; NL + 1];
    let exposure = dwells.iter().map(|d| d.pulses as f64 * 0.02133).sum();
    let pulses: usize = dwells.iter().map(|d| d.pulses).sum();
    for (di, d) in dwells.iter().enumerate() {
        for l in 0..=NL {
            for k in 0..2 {
                noise[l][k] += d.noise[l][k] * d.pulses as f64 / pulses as f64;
            }
        }
        for first in (0..d.cycles).step_by(4) {
            let end = (first + 4).min(d.cycles);
            if end - first < 2 {
                continue;
            }
            let mut b = Block {
                dwell: di,
                sum: vec![C::default(); m],
                count: vec![0; m],
            };
            for cy in first..end {
                for j in 0..m {
                    let k = cy * m + j;
                    if d.valid[k] {
                        b.sum[j] += C::new(d.values[k][0], d.values[k][1]);
                        b.count[j] += 1;
                    }
                }
            }
            blocks.push(b);
        }
    }
    ensure!(
        blocks.len() >= 8,
        "insufficient code-cycle blocks for profile"
    );
    let all: Vec<_> = (0..blocks.len()).collect();
    let (observed, counts) = mean(&blocks, &all, m);
    let mut products = vec![];
    for j in 0..m {
        let vals: Vec<_> = blocks
            .iter()
            .filter(|b| b.count[j] >= 2)
            .map(|b| b.sum[j] / b.count[j] as f64)
            .collect();
        let mut sd = [1e30; 2];
        if counts[j] >= 8 && vals.len() >= 6 {
            for k in 0..2 {
                let mu = if k == 0 {
                    observed[j].re
                } else {
                    observed[j].im
                };
                let variance = vals
                    .iter()
                    .map(|v| {
                        let z = if k == 0 { v.re } else { v.im };
                        (z - mu).powi(2)
                    })
                    .sum::<f64>()
                    / ((vals.len() - 1) * vals.len()) as f64;
                sd[k] = variance.sqrt().max(noise[0][0].abs() * 1e-6);
            }
        }
        products.push(Product {
            value: [observed[j].re, observed[j].im],
            sigma: sd,
        });
    }
    eprintln!(
        "JOINT {} building range-resolved AC response, {} gates, {} blocks",
        job.id,
        ng,
        blocks.len()
    );
    let freq: Vec<_> = (0..201).map(|j| -50000. + j as f64 * 500.).collect();
    let response = ac_range::stencil(codes, &ac_range::fir(&noise), NL, &freq);
    let gategrid = grid::build(job.el, 1000.);
    ensure!(gategrid.len() == ng, "grid differs from measured products");
    let mut terms: BTreeMap<i32, Vec<Term>> = BTreeMap::new();
    for (g, bin) in gategrid.iter().enumerate() {
        let mut kernels: BTreeMap<i32, Vec<C>> = BTreeMap::new();
        for lag in 0..NL {
            kernels.clear();
            for (j, w) in bin.native_cells.iter().zip(&bin.weights) {
                for d in ac_range::FIRST..=ac_range::LAST {
                    let source = *j as i32 + d;
                    let r = 667e-6 * 149896.229 + source as f64 * grid::NATIVE_RANGE_KM;
                    if r < 10. {
                        continue;
                    }
                    let kernel = &response[lag][(d - ac_range::FIRST) as usize];
                    if kernel.iter().all(|v| v.norm() < 1e-13) {
                        continue;
                    }
                    let v = kernels
                        .entry(source)
                        .or_insert_with(|| vec![C::default(); freq.len()]);
                    for (k, z) in v.iter_mut().enumerate() {
                        *z += kernel[k] * w * 500. * if k == 0 || k == 200 { 0.5 } else { 1. };
                    }
                }
            }
            for (source, kernel) in &kernels {
                terms.entry(*source).or_default().push(Term {
                    measurement: g * NL + lag,
                    kernel: kernel.clone(),
                });
            }
        }
    }
    let spline = Spline::adaptive();
    let controls = spline.controls();
    let mut cells = vec![];
    eprintln!(
        "JOINT {} {} native source cells, {} spline controls; building magnetized response",
        job.id,
        terms.len(),
        spline.count()
    );
    for (source, terms) in terms {
        let range = 667e-6 * 149896.229 + source as f64 * grid::NATIVE_RANGE_KM;
        let z = campaign::altitude(range, job.el);
        let theory = Arc::new(Theory::from_beam(range, job.az, job.el, 500.)?);
        cells.push(Cell {
            theory,
            shape_density_m3: None,
            density_calibration: Some((gain, range, h_weight(z))),
            knot_weights: spline.weights(z),
            terms,
        });
    }
    let initial: Vec<_> = controls
        .iter()
        .map(|z| {
            let g = baseline
                .gates
                .iter()
                .filter(|g| g.node.is_some() && g.ne_m3.is_some())
                .min_by(|a, b| {
                    (a.altitude_km - z)
                        .abs()
                        .total_cmp(&(b.altitude_km - z).abs())
                })
                .unwrap();
            let n = g.node.as_ref().unwrap();
            let h = if *z <= 500. {
                0.
            } else {
                n.h_fraction.clamp(0.01, 0.8)
            };
            Node {
                power_adu2: (g.ne_m3.unwrap() / 1e11).clamp(1e-5, 1e4),
                ti_k: n.ti_k.clamp(300., 4000.),
                te_k: n.te_k.clamp(300., 8000.),
                velocity_m_s: n.velocity_m_s.clamp(-1000., 1000.),
                h_fraction: h,
                he_fraction: 0.,
                o_fraction: 1. - h,
            }
        })
        .collect();
    let mut fixed = vec![];
    for (k, z) in controls.iter().enumerate() {
        fixed.push((6 * k + 5, 0.));
        if *z <= 500. {
            fixed.push((6 * k + 4, 0.));
        }
    }
    let mut problem = Problem {
        frequency_hz: freq,
        radar_frequency_hz: 430.2e6,
        shape_density_m3: 1e11,
        cells,
        products,
        knot_count: spline.count(),
        background_basis: vec![C::default(); m],
        background_sigma: 1.,
    };
    let mut fitted = problem.solve_with_tolerance(&initial, 200, &fixed, 0.03)?;
    eprintln!(
        "JOINT {} first solve {} Q={} gradient={}",
        job.id, fitted.status, fitted.chi_square, fitted.normalized_gradient
    );
    for hv in [0.05, 0.5] {
        let mut start = fitted.nodes.clone();
        for (k, n) in start.iter_mut().enumerate() {
            if controls[k] > 500. {
                n.h_fraction = hv;
                n.o_fraction = 1. - hv;
            }
        }
        let alternative = problem.solve_with_tolerance(&start, 60, &fixed, 0.03)?;
        if alternative.chi_square < fitted.chi_square {
            fitted = alternative;
        }
    }
    for continuation in 0..3 {
        if fitted.status.starts_with("converged") && fitted.normalized_gradient < 0.05 {
            break;
        }
        let retry = problem.solve_with_tolerance(&fitted.nodes, 200, &fixed, 0.03)?;
        eprintln!(
            "JOINT {} continuation {} {} Q={} gradient={}",
            job.id, continuation, retry.status, retry.chi_square, retry.normalized_gradient
        );
        if retry.chi_square <= fitted.chi_square {
            fitted = retry;
        } else {
            break;
        }
    }
    let mut gates = vec![];
    for (g, bin) in gategrid.iter().enumerate() {
        let n = sample(&spline, &fitted.nodes, bin.altitude_km);
        let mut flags = vec![];
        if bin.altitude_km < 200. {
            flags.push("molecular_species_omitted".into());
        }
        if bin.altitude_km < 130. || bin.altitude_km > 970. {
            flags.push("constant_extrapolated_profile_halo".into());
        }
        if !fitted.status.starts_with("converged") {
            flags.push("joint_optimizer_not_converged".into());
        }
        if problem.products[g * NL..(g + 1) * NL]
            .iter()
            .any(|p| p.sigma[0] >= 1e20)
        {
            flags.push("insufficient_cycles_in_lag".into());
        }
        let mut reported = n.clone();
        reported.power_adu2 =
            amplitude(&problem, bin.slant_range_km, &n, n.power_adu2 * 1e11, gain);
        gates.push(Gate {
            altitude_km: bin.altitude_km,
            range_km: bin.slant_range_km,
            width_km: bin.upper_altitude_km - bin.lower_altitude_km,
            node: Some(reported),
            ne_m3: Some(n.power_adu2 * 1e11),
            lower68: vec![None; 5],
            upper68: vec![None; 5],
            lower95: vec![None; 5],
            upper95: vec![None; 5],
            bootstrap_converged: 0,
            bootstrap_requested: nboot,
            flags,
            chi_square: None,
            observed: observed[g * NL..(g + 1) * NL]
                .iter()
                .map(|z| [z.re, z.im])
                .collect(),
            prediction: fitted.predicted[g * NL..(g + 1) * NL].to_vec(),
            conditional_standard_error: vec![None; 5],
            h_profile: vec![],
            bootstrap_nodes: vec![],
        });
    }
    let profile=Profile{schema:"syisr-ac-joint-spline-profile-v3".into(),job:job.clone(),exposure_s:exposure,cycles:dwells.iter().map(|d|d.cycles).sum(),screen_fraction:baseline.screen_fraction,calibration_gain:Some(gain),calibration_fractional_error:Some(gain_error),gates,bootstrap:"Stratified four-complete-code-cycle blocks, synchronized over all altitudes and lags; full joint nuisance refit in each draw; one common calibration perturbation per draw; conditional on spline spacing, H/O composition and receiver model".into(),source_dwells:job.dwells.clone(),altitude_edges_km:baseline.altitude_edges_km};
    let mut result=Joint{profile,spline,control_altitude_km:controls,control_nodes:fitted.nodes.clone(),product_sigma:problem.products.iter().map(|p|p.sigma).collect(),nominal_fit:fitted,bootstrap_control_nodes:vec![],bootstrap_status:vec![],bootstrap_ne_m3:vec![vec![];ng],likelihood:"Joint signed real+imag lag likelihood with empirical block variances; off-diagonal covariance not used in the working objective; common block bootstrap retains lag/range correlation in conditional intervals".into(),forward_model:"Cubic B-splines for log Ne, log Ti, log Te, velocity, bounded H; control spacing 15km below400km increasing linearly to150km at2000km; native20us source cells integrated exactly in two10us subcells through 64-code whole/half-baud decoder, empirical voltage FIR, 1/range^2 and density-dependent spectral mass; H identically zero below500km with C2 onset over500-530km; He zero; lower molecular ions omitted and flagged; native-cell response extends beyond100–1000km with endpoint-constant extrapolation, boundary estimates flagged".into()};
    campaign::save(
        &out.join("joint-nominal").join(format!("{}.h5", job.id)),
        &result,
    )?;
    let mut strata = BTreeMap::<usize, Vec<usize>>::new();
    for (i, b) in blocks.iter().enumerate() {
        strata.entry(b.dwell).or_default().push(i);
    }
    let mut random = Rng(job.start as u64 ^ 0x3521321);
    for draw in 0..nboot {
        let mut indices = vec![];
        for group in strata.values() {
            for _ in group {
                indices.push(group[(random.next() * group.len() as f64) as usize % group.len()]);
            }
        }
        let (y, _) = mean(&blocks, &indices, m);
        for (j, z) in y.iter().enumerate() {
            problem.products[j].value = [z.re, z.im];
        }
        let perturbed = gain * (gain_error * random.normal()).exp();
        for cell in &mut problem.cells {
            if let Some((_, r, w)) = cell.density_calibration {
                cell.density_calibration = Some((perturbed, r, w));
            }
        }
        let fit = problem.solve_with_tolerance(&result.control_nodes, 80, &fixed, 0.05)?;
        let valid = fit.status.starts_with("converged") && fit.normalized_gradient < 0.1;
        result.bootstrap_status.push(fit.status.clone());
        if valid {
            for (i, g) in result.profile.gates.iter_mut().enumerate() {
                let mut n = sample(&result.spline, &fit.nodes, g.altitude_km);
                let ne = n.power_adu2 * 1e11;
                result.bootstrap_ne_m3[i].push(ne);
                n.power_adu2 = amplitude(&problem, g.range_km, &n, ne, perturbed);
                g.bootstrap_nodes.push(n);
            }
            result.bootstrap_control_nodes.push(fit.nodes);
        }
        eprintln!(
            "JOINT {} bootstrap {}/{} {}",
            job.id,
            draw + 1,
            nboot,
            fit.status
        );
    }
    for (i, g) in result.profile.gates.iter_mut().enumerate() {
        g.bootstrap_converged = g.bootstrap_nodes.len();
        if nboot == 0 {
            g.flags.push("joint_bootstrap_pending".into());
        } else if g.bootstrap_converged * 5 < nboot * 4 {
            g.flags.push("joint_bootstrap_failures".into());
        }
        if g.bootstrap_converged >= 32 {
            for f in 0..5 {
                let mut v: Vec<_> = if f == 0 {
                    result.bootstrap_ne_m3[i].clone()
                } else {
                    g.bootstrap_nodes
                        .iter()
                        .map(|n| match f {
                            1 => n.ti_k,
                            2 => n.te_k,
                            3 => n.velocity_m_s,
                            _ => n.h_fraction,
                        })
                        .collect()
                };
                g.lower68[f] = percentile(&mut v, 0.16);
                g.upper68[f] = percentile(&mut v, 0.84);
                g.lower95[f] = percentile(&mut v, 0.025);
                g.upper95[f] = percentile(&mut v, 0.975);
            }
        }
    }
    campaign::save(
        &out.join("joint-fits").join(format!("{}.h5", job.id)),
        &result,
    )?;
    let _ = c;
    Ok(result)
}

#[cfg(test)]
mod tests {
    use super::*;
    #[test]
    fn noiseless_joint_profile_is_recovered_through_range_mixing() {
        let spline = Spline {
            knots: vec![100., 100., 100., 100., 1000., 1000., 1000., 1000.],
        };
        let theory = Arc::new(Theory::new(500., 500.).unwrap());
        let freq = theory.f.clone();
        let mut cells = vec![];
        let groups = 8;
        let lags = 20;
        for i in 0..31 {
            let z = 100. + i as f64 * 30.;
            let mut terms = vec![];
            for g in 0..groups {
                let center = 100. + g as f64 * 900. / (groups - 1) as f64;
                let norm = (0..31)
                    .map(|j| (-0.5 * ((100. + j as f64 * 30. - center) / 55.).powi(2)).exp())
                    .sum::<f64>();
                let w = (-0.5 * ((z - center) / 55.).powi(2)).exp() / norm;
                for l in 1..=lags {
                    terms.push(Term {
                        measurement: g * lags + l - 1,
                        kernel: freq
                            .iter()
                            .enumerate()
                            .map(|(k, f)| {
                                C::from_polar(
                                    w * 500. * if k == 0 || k == 200 { 0.5 } else { 1. },
                                    -std::f64::consts::TAU * f * l as f64 * 1e-5,
                                )
                            })
                            .collect(),
                    });
                }
            }
            cells.push(Cell {
                theory: theory.clone(),
                shape_density_m3: None,
                density_calibration: Some((700., z + 100., h_weight(z))),
                knot_weights: spline.weights(z),
                terms,
            });
        }
        let mut p = Problem {
            frequency_hz: freq,
            radar_frequency_hz: 430.2e6,
            shape_density_m3: 1e11,
            cells,
            products: vec![
                Product {
                    value: [0., 0.],
                    sigma: [1., 1.]
                };
                groups * lags
            ],
            knot_count: 4,
            background_basis: vec![C::default(); groups * lags],
            background_sigma: 1.,
        };
        let truth: Vec<_> = (0..4)
            .map(|i| Node {
                power_adu2: [0.8, 1.3, 0.9, 0.3][i],
                ti_k: 800. + i as f64 * 150.,
                te_k: 1200. + i as f64 * 400.,
                velocity_m_s: 10. + i as f64 * 10.,
                h_fraction: if i < 2 { 0. } else { (i - 1) as f64 * 0.2 },
                he_fraction: 0.,
                o_fraction: 1. - if i < 2 { 0. } else { (i - 1) as f64 * 0.2 },
            })
            .collect();
        let (y, _) = p.evaluate(&encode(&truth), false).unwrap();
        for (i, obs) in p.products.iter_mut().enumerate() {
            obs.value = [y[2 * i], y[2 * i + 1]];
        }
        let mut seed = truth.clone();
        for n in &mut seed {
            n.power_adu2 *= 1.08;
            n.ti_k *= 0.96;
            n.te_k *= 1.04;
            n.velocity_m_s += 5.;
        }
        seed[2].h_fraction += 0.02;
        seed[3].h_fraction += 0.02;
        let fixed = [(5, 0.), (11, 0.), (17, 0.), (23, 0.), (4, 0.), (10, 0.)];
        let fit = p.solve_with_tolerance(&seed, 100, &fixed, 0.001).unwrap();
        assert!(
            fit.chi_square < 1e-5,
            "{} Q{} gradient{}",
            fit.status,
            fit.chi_square,
            fit.normalized_gradient
        );
        for i in 0..4 {
            assert!((fit.nodes[i].power_adu2 / truth[i].power_adu2 - 1.).abs() < 0.01);
            assert!((fit.nodes[i].h_fraction - truth[i].h_fraction).abs() < 0.01);
        }
    }
}
