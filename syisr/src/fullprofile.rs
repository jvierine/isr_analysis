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
    pub h_spline: Spline,
    pub h_control_altitude_km: Vec<f64>,
    pub h_smoothing_spacing_multiplier: f64,
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
fn sample(s: &Spline, hs: &Spline, n: &[Node], z: f64) -> Node {
    let x = encode(n);
    let mut p = [0.; 6];
    for (k, w) in s.weights(z) {
        for f in 0..6 {
            p[f] += x[6 * k + f] * w;
        }
    }
    let h = hs
        .weights(z)
        .iter()
        .map(|(k, w)| w * n[*k].h_fraction)
        .sum::<f64>()
        * h_weight(z);
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
    let gategrid = grid::build(job.el, grid::maximum_altitude_km());
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
    let spline = Spline::adaptive_to(grid::maximum_altitude_km());
    let h_spline = spline.doubled_spacing();
    let h_controls = h_spline.controls();
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
            h_knot_weights: Some(h_spline.weights(z)),
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
    let gated_seed = initial.clone();
    let mut initial = crate::initialization::project(&spline, &baseline, &initial);
    let h_seed: Vec<_> = h_controls
        .iter()
        .map(|z| {
            initial[controls
                .iter()
                .enumerate()
                .min_by(|(_, a), (_, b)| (*a - z).abs().total_cmp(&(*b - z).abs()))
                .unwrap()
                .0]
                .clone()
        })
        .collect();
    let h_initial = crate::initialization::project(&h_spline, &baseline, &h_seed);
    for (i, n) in initial.iter_mut().enumerate() {
        n.h_fraction = h_initial.get(i).map_or(0., |n| n.h_fraction);
        n.o_fraction = 1. - n.h_fraction;
    }

    let mut fixed = vec![];
    for (k, z) in controls.iter().enumerate() {
        fixed.push((6 * k + 5, 0.));
        if k >= h_controls.len() || h_controls[k] <= 500. {
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

    let mut reused_joint_seed = false;
    if let Some(seed_root) = std::env::var_os("SYISR_JOINT_SEED_ROOT") {
        let path = Path::new(&seed_root).join(format!("{}.h5", job.id));
        if let Ok(v) = campaign::load::<serde_json::Value>(&path) {
            if v["nominal_fit"]["status"]
                .as_str()
                .is_some_and(|s| s.starts_with("converged"))
                && v["nominal_fit"]["normalized_gradient"]
                    .as_f64()
                    .unwrap_or(1e9)
                    < 0.05
            {
                let prior_profile: Profile = campaign::load(&path)?;
                let mut nodes = crate::initialization::project(&spline, &prior_profile, &initial);
                let hfit = crate::initialization::project(&h_spline, &prior_profile, &h_seed);
                for (i,n) in nodes.iter_mut().enumerate() {
                    n.h_fraction=hfit.get(i).map_or(0.,|n|n.h_fraction);
                    n.o_fraction=1.-n.h_fraction;
                }
                let candidate=problem.evaluate(&encode(&nodes),false)?.0.norm_squared();
                let current=problem.evaluate(&encode(&initial),false)?.0.norm_squared();
                if candidate < current {
                    initial=nodes;
                    reused_joint_seed=v["h_spline"]["knots"]==serde_json::to_value(&h_spline.knots)? && v["spline"]["knots"]==serde_json::to_value(&spline.knots)?;
                    eprintln!("JOINT {} projected converged same-window seed Q={candidate}",job.id);
                }
            }
        }
    }
    if let Some(path) = std::env::var_os("SYISR_JOINT_BENCH_FILE") {
        use std::time::Instant;
        let x = encode(&initial);
        let mut trials = vec![];
        let mut previous = None;
        let mut residual_error = 0f64;
        let mut jacobian_error = 0f64;
        for trial in 0..5 {
            for fast in if trial % 2 == 0 {
                [false, true]
            } else {
                [true, false]
            } {
                let start = Instant::now();
                let (r, j) = problem.evaluate_mode(&x, true, fast)?;
                let wall = start.elapsed().as_secs_f64();
                trials.push(serde_json::json!({"trial":trial,"fast":fast,"wall_s":wall}));
                if !fast {
                    previous = Some((r, j));
                } else if let Some((a, b)) = &previous {
                    residual_error = residual_error.max((&r - a).norm() / a.norm().max(1e-30));
                    jacobian_error = jacobian_error.max((&j - b).norm() / b.norm().max(1e-30));
                }
            }
        }
        let q_gated = problem
            .evaluate_mode(&encode(&gated_seed), false, true)?
            .0
            .norm_squared();
        let q_projected = problem.evaluate_mode(&x, false, true)?.0.norm_squared();
        let mut quality = serde_json::Value::Null;
        if let Some(reference) = std::env::var_os("SYISR_JOINT_REFERENCE") {
            let v: serde_json::Value = campaign::load(Path::new(&reference))?;
            let old_nodes: Vec<Node> = v["control_nodes"]
                .as_array()
                .unwrap()
                .iter()
                .map(|n| {
                    let h = n["h_coefficient"].as_f64().unwrap();
                    Node {
                        power_adu2: n["ne_m3"].as_f64().unwrap() / 1e11,
                        ti_k: n["ti_k"].as_f64().unwrap(),
                        te_k: n["te_k"].as_f64().unwrap(),
                        velocity_m_s: n["velocity_m_s"].as_f64().unwrap(),
                        h_fraction: h,
                        he_fraction: 0.,
                        o_fraction: 1. - h,
                    }
                })
                .collect();
            let (r, _) = problem.evaluate_mode(&encode(&old_nodes), false, true)?;
            let prediction: Vec<_> = (0..m)
                .map(|j| {
                    [
                        r[2 * j] * problem.products[j].sigma[0] + problem.products[j].value[0],
                        r[2 * j + 1] * problem.products[j].sigma[1] + problem.products[j].value[1],
                    ]
                })
                .collect();
            let mut joint_q = vec![];
            let mut gated_q = vec![];
            for g in 0..ng {
                joint_q.push(
                    (0..NL)
                        .map(|l| r[2 * (g * NL + l)].powi(2) + r[2 * (g * NL + l) + 1].powi(2))
                        .sum::<f64>(),
                );
                let old = &baseline.gates[g].prediction;
                gated_q.push(if old.len() == NL {
                    Some(
                        (0..NL)
                            .map(|l| {
                                let j = g * NL + l;
                                ((old[l][0] - observed[j].re) / problem.products[j].sigma[0])
                                    .powi(2)
                                    + ((old[l][1] - observed[j].im) / problem.products[j].sigma[1])
                                        .powi(2)
                            })
                            .sum::<f64>(),
                    )
                } else {
                    None
                });
            }
            quality = serde_json::json!({"height_km":dwells[0].heights,"observed":observed.iter().map(|z|[z.re,z.im]).collect::<Vec<_>>(),"sigma":problem.products.iter().map(|p|p.sigma).collect::<Vec<_>>(),"joint_prediction":prediction,"gated_prediction":baseline.gates.iter().map(|g|g.prediction.clone()).collect::<Vec<_>>(),"joint_q_by_gate":joint_q,"gated_q_by_gate":gated_q,"same_diagonal_working_variances":true});
        }
        let kernel_bytes = problem
            .cells
            .iter()
            .flat_map(|c| &c.terms)
            .map(|t| t.kernel.len() * 16)
            .sum::<usize>();
        campaign::save(
            Path::new(&path),
            &serde_json::json!({"job":job,"source_cells":problem.cells.len(),"products":m,"controls":spline.count(),"kernel_bytes":kernel_bytes,"initial_objective_nearest_gated_controls":q_gated,"initial_objective_projected_controls":q_projected,"residual_relative_error":residual_error,"jacobian_relative_error":jacobian_error,"trials":trials,"quality":quality,"cpu_info":std::fs::read_to_string("/proc/cpuinfo").ok(),"rayon_threads":std::env::var("RAYON_NUM_THREADS").ok()}),
        )?;
        anyhow::bail!("benchmark-only: wrote comparison; no fitted product created");
    }
    let mut fitted = problem.solve_with_tolerance(&initial, 200, &fixed, 0.03)?;
    eprintln!(
        "JOINT {} first solve {} Q={} gradient={}",
        job.id, fitted.status, fitted.chi_square, fitted.normalized_gradient
    );
    for hv in if reused_joint_seed && fitted.status.starts_with("converged") {
        vec![]
    } else {
        vec![0.05, 0.5]
    } {
        let mut start = fitted.nodes.clone();
        for (k, n) in start.iter_mut().enumerate() {
            if k < h_controls.len() && h_controls[k] > 500. {
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
        let n = sample(&spline, &h_spline, &fitted.nodes, bin.altitude_km);
        let mut flags = vec![];
        if bin.altitude_km < 200. {
            flags.push("molecular_species_omitted".into());
        }
        if bin.altitude_km < 130. || bin.altitude_km > grid::maximum_altitude_km() - 30. {
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
    let profile=Profile{schema:"syisr-ac-joint-spline-profile-v3".into(),job:job.clone(),exposure_s:exposure,cycles:dwells.iter().map(|d|d.cycles).sum(),screen_fraction:baseline.screen_fraction,calibration_gain:Some(gain),calibration_fractional_error:Some(gain_error),gates,bootstrap:"Stratified four-complete-code-cycle blocks, synchronized over all altitudes and lags; full joint nuisance refit in each draw; one common calibration perturbation per draw; conditional on spline spacing, H/O composition and receiver model".into(),source_dwells:job.dwells.clone(),altitude_edges_km:gategrid.iter().map(|g|g.lower_altitude_km).chain(std::iter::once(gategrid.last().unwrap().upper_altitude_km)).collect()};
    let mut result=Joint{profile,spline,h_spline,h_control_altitude_km:h_controls,h_smoothing_spacing_multiplier:2.,control_altitude_km:controls,control_nodes:fitted.nodes.clone(),product_sigma:problem.products.iter().map(|p|p.sigma).collect(),nominal_fit:fitted,bootstrap_control_nodes:vec![],bootstrap_status:vec![],bootstrap_ne_m3:vec![vec![];ng],likelihood:"Joint signed real+imag lag likelihood with empirical block variances; off-diagonal covariance not used in the working objective; common block bootstrap retains lag/range correlation in conditional intervals".into(),forward_model:"Cubic B-splines for log Ne, log Ti, log Te, velocity, bounded H; Ne/Ti/Te/vi control spacing 15km below400km increasing linearly to150km at2000km; H-only control spacing doubled by retaining every second original breakpoint; H coefficients use h_control_altitude_km and first h_spline.count() H slots; native20us source cells integrated exactly in two10us subcells through 64-code whole/half-baud decoder, empirical voltage FIR, 1/range^2 and density-dependent spectral mass; H identically zero below500km with C2 onset over500-530km; He zero; lower molecular ions omitted and flagged; native-cell response extends beyond the fitted altitude domain with endpoint-constant extrapolation, boundary estimates flagged".into()};
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
                let mut n = sample(&result.spline, &result.h_spline, &fit.nodes, g.altitude_km);
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
                h_knot_weights: None,
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
