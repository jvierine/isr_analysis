//! Shared H/O composition, beam-specific spline profiles and robust adjacent-time priors.
use crate::{
    campaign,
    coupled::{self, Cell, CoordinatePrior, Node, Problem, Product, Term},
    h_transition::Shape,
    lp480::{self, Decoded, Manifest, C_KM, DT, NL},
    sharedcomposition::{self, Temporal, TimeNeighbor},
    spline::{self, Spline},
    theory::Theory,
};
use anyhow::{ensure, Result};
use num_complex::Complex64 as C;
use serde::{Deserialize, Serialize};
use std::{collections::BTreeMap, path::Path, sync::Arc};
#[derive(Clone, Serialize, Deserialize)]
pub struct Reference {
    pub altitude_km: Vec<f64>,
    pub ne_m3: Vec<f64>,
    pub ti_k: Vec<f64>,
    pub te_k: Vec<f64>,
    pub heavy_fractions: Vec<[f64; 4]>,
    pub shape: Shape,
    pub provenance: serde_json::Value,
}
#[derive(Clone, Serialize, Deserialize)]
pub struct Background {
    pub acf: Vec<[f64; 2]>,
    pub sigma: Vec<[f64; 2]>,
    pub blocks: usize,
    pub method: String,
}
#[derive(Serialize, Deserialize)]
pub struct BeamFit {
    pub id: String,
    pub azimuth_deg: f64,
    pub elevation_deg: f64,
    pub exposure_s: f64,
    pub input_pulses: usize,
    pub kept_pulses: usize,
    pub spline: Spline,
    pub controls: Vec<Node>,
    pub altitude_km: Vec<f64>,
    pub ne_m3: Vec<f64>,
    pub ti_k: Vec<f64>,
    pub te_k: Vec<f64>,
    pub velocity_m_s: Vec<f64>,
    pub h_fraction: Vec<f64>,
    pub raw_reduced_chi_square: f64,
    pub residual_rms_sigma: f64,
    pub measurement_gate: Vec<usize>,
    pub measurement_lag: Vec<usize>,
    pub residual: Vec<[f64; 2]>,
    pub observed: Vec<[f64; 2]>,
    pub predicted: Vec<[f64; 2]>,
    pub assigned_sigma: Vec<[f64; 2]>,
}
#[derive(Serialize, Deserialize)]
pub struct WindowFit {
    pub start_utc_s: i64,
    pub end_utc_s: i64,
    pub shape: Shape,
    pub shared_parameters: [f64; 5],
    pub status: String,
    pub iterations: usize,
    pub normalized_gradient: f64,
    pub objective_history: Vec<f64>,
    pub minimized_objective: f64,
    pub h_temporal_objective: f64,
    pub beams: Vec<BeamFit>,
    pub metadata: serde_json::Value,
}
fn median(v: &mut [f64]) -> f64 {
    v.sort_by(f64::total_cmp);
    v[v.len() / 2]
}
pub fn interpolate(z: &[f64], v: &[f64], x: f64) -> f64 {
    let k = z
        .partition_point(|a| *a <= x)
        .saturating_sub(1)
        .min(z.len() - 2);
    let u = ((x - z[k]) / (z[k + 1] - z[k])).clamp(0., 1.);
    v[k] * (1. - u) + v[k + 1] * u
}
pub fn background(root: &Path, out: &Path) -> Result<Background> {
    let m: Manifest = campaign::load(&root.join("index.h5"))?;
    let mut blocks = vec![];
    for j in &m.jobs {
        if j.el < 89. {
            continue;
        }
        let d: Decoded = campaign::load(&root.join("products").join(format!("{}.h5", j.id)))?;
        for b in d.blocks {
            if b.kept >= 100 && b.noise_pairs[0] > 0 {
                blocks.push((b.noise[0][0] / b.noise_pairs[0] as f64, b));
            }
        }
    }
    ensure!(blocks.len() > 20, "insufficient zenith background blocks");
    blocks.sort_by(|a, b| a.0.total_cmp(&b.0));
    let take = (blocks.len() / 5).max(20);
    let mut acf = vec![[0.; 2]; NL + 1];
    let mut sigma = acf.clone();
    for l in 0..=NL {
        let vals: Vec<_> = blocks[..take]
            .iter()
            .filter(|(_, b)| b.noise_pairs[l] > 0)
            .map(|(_, b)| {
                [
                    b.noise[l][0] / b.noise_pairs[l] as f64,
                    b.noise[l][1] / b.noise_pairs[l] as f64,
                ]
            })
            .collect();
        for c in 0..2 {
            let mean = vals.iter().map(|x| x[c]).sum::<f64>() / vals.len() as f64;
            acf[l][c] = mean;
            sigma[l][c] = (vals.iter().map(|x| (x[c] - mean).powi(2)).sum::<f64>()
                / (vals.len() * (vals.len() - 1)) as f64)
                .sqrt();
        }
    }
    let b=Background{acf,sigma,blocks:take,method:"Lower 20% power of screened zenith far-range 10 s blocks across campaign; common complex quiet-background reference, subtracted before finite-pulse response. Possible residual plasma in reference is not ruled out; it is not interpreted as a signal receiver FIR.".into()};
    campaign::save(out, &b)?;
    Ok(b)
}
fn one() -> f64 {
    1.
}
#[derive(Serialize, Deserialize)]
pub struct Averaged {
    #[serde(default = "one")]
    pub background_scale: f64,
    #[serde(default)]
    pub background_scale_sem: f64,
    pub products: Vec<Product>,
    pub gates: Vec<usize>,
    pub lags: Vec<usize>,
    pub input_pulses: usize,
    pub kept_pulses: usize,
    pub range_doppler_raw: Vec<f64>,
    pub range_doppler_pulses: usize,
    pub background_acf: Vec<[f64; 2]>,
}
pub fn pulse_counts(d: &Decoded) -> (usize, usize) {
    if let Some(a) = &d.averaged {
        (a.input_pulses, a.kept_pulses)
    } else {
        (
            d.blocks.iter().map(|b| b.pulses).sum(),
            d.blocks.iter().map(|b| b.kept).sum(),
        )
    }
}
pub fn compact(root: &Path, out: &Path) -> Result<()> {
    std::fs::create_dir_all(out.join("products"))?;
    let m: Manifest = campaign::load(&root.join("index.h5"))?;
    let b: Background = campaign::load(&root.join("background.h5"))?;
    campaign::save(&out.join("index.h5"), &m)?;
    campaign::save(&out.join("background.h5"), &b)?;
    for j in &m.jobs {
        let mut d: Decoded = campaign::load(&root.join("products").join(format!("{}.h5", j.id)))?;
        let (p, g, l) = observations(&d, &b);
        let (input, kept) = pulse_counts(&d);
        let len = d.blocks.first().map_or(0, |a| a.range_doppler_raw.len());
        let mut rd = vec![0.; len];
        let mut n = 0;
        for block in &d.blocks {
            ensure!(block.range_doppler_raw.len() == len, "RD dimensions");
            let count = block.range_doppler_pulses;
            n += count;
            for (i, v) in rd.iter_mut().enumerate() {
                *v += block.range_doppler_raw[i] * count as f64;
            }
        }
        if n > 0 {
            rd.iter_mut().for_each(|v| *v /= n as f64);
        }
        let (background_scale, background_scale_sem) = noise_scale(&d, &b);
        d.averaged = Some(Averaged {
            background_scale,
            background_scale_sem,
            products: p,
            gates: g,
            lags: l,
            input_pulses: input,
            kept_pulses: kept,
            range_doppler_raw: rd,
            range_doppler_pulses: n,
            background_acf: b.acf.clone(),
        });
        d.blocks.clear();
        campaign::save(&out.join("products").join(format!("{}.h5", j.id)), &d)?;
    }
    Ok(())
}
pub struct Prepared {
    pub decoded: Decoded,
    pub spline: Spline,
    pub initial: Vec<Node>,
    pub problem: Problem,
    pub gate: Vec<usize>,
    pub lag: Vec<usize>,
}
pub fn noise_scale(d: &Decoded, b: &Background) -> (f64, f64) {
    if let Some(a) = &d.averaged {
        return (a.background_scale, a.background_scale_sem);
    }
    let mut values = d
        .blocks
        .iter()
        .filter(|x| x.noise_pairs[0] > 0)
        .map(|x| x.noise[0][0] / x.noise_pairs[0] as f64)
        .filter(|x| x.is_finite() && *x > 0.)
        .collect::<Vec<_>>();
    if values.is_empty() {
        return (1., 0.);
    }
    let center = median(&mut values);
    let mut deviations = values
        .iter()
        .map(|v| (v - center).abs())
        .collect::<Vec<_>>();
    (
        center / b.acf[0][0],
        1.2533 * 1.4826 * median(&mut deviations) / (values.len() as f64).sqrt() / b.acf[0][0],
    )
}
pub(crate) fn observations(d: &Decoded, b: &Background) -> (Vec<Product>, Vec<usize>, Vec<usize>) {
    if let Some(a) = &d.averaged {
        return (a.products.clone(), a.gates.clone(), a.lags.clone());
    }
    let (scale, scale_sem) = noise_scale(d, b);
    let mut p = vec![];
    let mut gs = vec![];
    let mut ls = vec![];
    for (g, gate) in d.gates.iter().enumerate() {
        for l in 1..=NL {
            if gate.index < l {
                continue;
            }
            let j = g * NL + l - 1;
            let bs: Vec<_> = d.blocks.iter().filter(|a| a.counts[j] > 0).collect();
            if bs.len() < 3 {
                continue;
            }
            let n = bs.iter().map(|a| a.counts[j]).sum::<usize>() as f64;
            let mut mean = [0.; 2];
            let mut sigma = [0.; 2];
            for c in 0..2 {
                mean[c] = bs.iter().map(|a| a.sums[j][c]).sum::<f64>() / n;
                let var = bs
                    .iter()
                    .map(|a| {
                        let k = a.counts[j] as f64;
                        k * (a.sums[j][c] / k - mean[c]).powi(2)
                    })
                    .sum::<f64>()
                    / ((bs.len() - 1) as f64 * n);
                sigma[c] =
                    (var + (scale * b.sigma[l][c]).powi(2) + (scale_sem * b.acf[l][c]).powi(2))
                        .sqrt()
                        .max(b.acf[0][0].abs() * 1e-5);
                mean[c] -= scale * b.acf[l][c];
            }
            p.push(Product { value: mean, sigma });
            gs.push(g);
            ls.push(l);
        }
    }
    (p, gs, ls)
}
pub fn prepare(
    d: Decoded,
    r: &Reference,
    b: &Background,
    gain: f64,
    prior: &[BeamFit],
) -> Result<Prepared> {
    let s = Spline::adaptive_to(1400.);
    let zc = s.controls();
    let products = observations(&d, b);
    ensure!(products.0.len() > 100, "insufficient products");
    let mut initial = zc
        .iter()
        .map(|z| Node {
            power_adu2: interpolate(&r.altitude_km, &r.ne_m3, *z).max(1e8) / 1e11,
            ti_k: interpolate(&r.altitude_km, &r.ti_k, *z).clamp(150., 6000.),
            te_k: interpolate(&r.altitude_km, &r.te_k, *z).clamp(150., 12000.),
            velocity_m_s: 0.,
            h_fraction: 0.,
            he_fraction: 0.,
            o_fraction: 1.,
        })
        .collect::<Vec<_>>();
    let q = Shape::encode(1000., 100.);
    initial[0].h_fraction = q[0];
    initial[1].h_fraction = q[1];
    initial[2].he_fraction = 0.;
    initial[3].he_fraction = 0.3;
    initial[4].he_fraction = 0.4;
    if let Some(previous) = prior.first() {
        ensure!(
            previous.spline.knots.len() == s.knots.len()
                && previous
                    .spline
                    .knots
                    .iter()
                    .zip(&s.knots)
                    .all(|(a, b)| (a - b).abs() < 1e-9)
                && previous.controls.len() == s.count(),
            "temporal grid changed"
        );
        initial = previous.controls.clone();
    }
    let mut priors = vec![];
    for previous in prior {
        let x = coupled::encode(&previous.controls);
        for k in 0..zc.len() {
            for c in 0..4 {
                priors.push(CoordinatePrior {
                    coefficients: vec![(6 * k + c, 1.)],
                    mean: x[6 * k + c],
                    sigma: 2. * [0.7, 0.4, 0.5, 0.2][c],
                    delta: 1.5,
                });
            }
        }
    }
    // Smooth in altitude at a scale comparable with pulse resolution, increasing aloft.
    for k in 1..zc.len() - 1 {
        let a = zc[k] - zc[k - 1];
        let c = zc[k + 1] - zc[k];
        let length = 30. + 0.05 * (zc[k] - 400.).max(0.);
        let w = [2. / (a * (a + c)), -2. / (a * c), 2. / (c * (a + c))];
        for f in 0..4 {
            priors.push(CoordinatePrior {
                coefficients: (0..3)
                    .map(|i| (6 * (k + i - 1) + f, w[i] * length * length))
                    .collect(),
                mean: 0.,
                sigma: [1., 0.5, 0.5, 0.25][f],
                delta: 3.,
            });
        }
    }
    let shape = Arc::new(r.shape.clone());
    let mut cells = vec![];
    let mut cache = BTreeMap::<i64, Arc<Theory>>::new();
    let nf = 201;
    let frequency: Vec<_> = (0..nf).map(|i| -50000. + i as f64 * 500.).collect();
    for zz in (50..1500).step_by(5) {
        let z = zz as f64 + 2.5;
        let edge = [
            2. * lp480::range(zz as f64, d.job.el) / C_KM,
            2. * lp480::range(zz as f64 + 5., d.job.el) / C_KM,
        ];
        let rr = lp480::range(z, d.job.el);
        let bucket = if z < 300. {
            (z / 15.).round() as i64 * 15
        } else {
            (z / 50.).round() as i64 * 50
        };
        let theory = if let Some(t) = cache.get(&bucket) {
            t.clone()
        } else {
            let mut t = Theory::from_beam_at_year(
                lp480::range(bucket as f64, d.job.el),
                d.job.az,
                d.job.el,
                500.,
                2023. + 319. / 365.25,
            )?;
            let heavy: Vec<_> = (0..4)
                .map(|c| {
                    interpolate(
                        &r.altitude_km,
                        &r.heavy_fractions.iter().map(|v| v[c]).collect::<Vec<_>>(),
                        bucket as f64,
                    )
                })
                .collect();
            t.heavy_ion_fractions = heavy.try_into().unwrap();
            let t = Arc::new(t);
            cache.insert(bucket, t.clone());
            t
        };
        let mut terms = vec![];
        for (m, (&g, &l)) in products.1.iter().zip(&products.2).enumerate() {
            let w = lp480::weight(&d.gates[g], l, d.job.front_us, edge);
            if w <= 0. {
                continue;
            }
            let kernel = frequency
                .iter()
                .enumerate()
                .map(|(i, f)| {
                    C::from_polar(
                        w * 500. * if i == 0 || i + 1 == nf { 0.5 } else { 1. },
                        -std::f64::consts::TAU * f * l as f64 * DT,
                    )
                })
                .collect();
            terms.push(Term {
                reuse: Some((l, w)),
                measurement: m,
                kernel,
            });
        }
        if !terms.is_empty() {
            cells.push(Cell {
                theory,
                shape_density_m3: None,
                density_calibration: Some((gain, rr, spline::h_weight(z))),
                knot_weights: s.weights(z),
                h_knot_weights: None,
                composition: Some((shape.clone(), z)),
                terms,
            });
        }
    }
    let (noise_gain, _) = noise_scale(&d, b);
    let bg = products
        .2
        .iter()
        .map(|l| C::new(noise_gain * b.acf[*l][0], noise_gain * b.acf[*l][1]))
        .collect();
    let problem = Problem {
        coordinate_priors: priors,
        frequency_hz: frequency,
        radar_frequency_hz: 430.2e6,
        shape_density_m3: 5e11,
        cells,
        products: products.0,
        knot_count: s.count(),
        background_basis: bg,
        background_sigma: 0.01,
    };
    Ok(Prepared {
        decoded: d,
        spline: s,
        initial,
        problem,
        gate: products.1,
        lag: products.2,
    })
}
pub fn sample(s: &Spline, n: &[Node], shape: &Shape, q: [f64; 5], z: f64) -> [f64; 5] {
    let x = coupled::encode(n);
    let mut a = [0.; 4];
    for (k, w) in s.weights(z) {
        for f in 0..4 {
            a[f] += x[6 * k + f] * w;
        }
    }
    [
        a[0].exp() * 1e11,
        a[1].exp() * 1000.,
        a[2].exp() * 1000.,
        a[3] * 1000.,
        shape.evaluate(z, [q[0], q[1]]).0 * spline::h_weight(z),
    ]
}
/// Preserve the validated collisionless-electron aspect guard over the full pulse support.
pub fn minimum_aspect(j: &lp480::Job) -> Result<f64> {
    let mut minimum = 90f64;
    for zz in (50..1500).step_by(5) {
        let z = zz as f64 + 2.5;
        let bucket = if z < 300. {
            (z / 15.).round() * 15.
        } else {
            (z / 50.).round() * 50.
        };
        let (_, angle) = Theory::beam_field_at_year(
            lp480::range(bucket, j.el),
            j.az,
            j.el,
            2023. + 319. / 365.25,
        )?;
        minimum = minimum.min(angle);
    }
    Ok(minimum)
}
pub fn fit(
    root: &Path,
    time: i64,
    out: &Path,
    previous: Option<&Path>,
    iterations: usize,
) -> Result<()> {
    let manifest: Manifest = campaign::load(&root.join("index.h5"))?;
    let reference: Reference = campaign::load(&root.join("references").join(format!("{time}.h5")))?;
    let bg: Background = campaign::load(&root.join("background.h5"))?;
    let gains: BTreeMap<String, f64> = campaign::load(&root.join("calibration.h5"))?;
    let jobs: Vec<_> = manifest.jobs.iter().filter(|j| j.start == time).collect();
    let mut neighbors = vec![];
    if let Some(pr) = previous {
        for t in [time - 1800, time + 1800] {
            let p = pr.join(format!("{t}.h5"));
            if p.exists() {
                neighbors.push(campaign::load::<WindowFit>(&p)?);
            }
        }
    }
    let mut prep = vec![];
    let mut unsupported_beams = vec![];
    for j in jobs {
        if !gains.contains_key(&format!("{:.2}-{:.2}", j.az, j.el)) {
            unsupported_beams.push(serde_json::json!({"azimuth_deg":j.az,"elevation_deg":j.el,"minimum_aspect_deg":minimum_aspect(j)?,"reason":"Finite-pulse support approaches within 3 degrees of perpendicular B; collisionless electron model is not validated. No fitted values."}));
            continue;
        }
        let d: Decoded = campaign::load(&root.join("products").join(format!("{}.h5", j.id)))?;
        if let Some(a) = &d.averaged {
            ensure!(
                a.background_acf == bg.acf,
                "compact likelihood background mismatch"
            );
        }
        if d.averaged
            .as_ref()
            .map_or(d.blocks.len() < 3, |a| a.products.len() < 100)
        {
            continue;
        }
        let key = format!("{:.2}-{:.2}", j.az, j.el);
        let prior = neighbors
            .iter()
            .filter_map(|n| {
                n.beams
                    .iter()
                    .find(|b| format!("{:.2}-{:.2}", b.azimuth_deg, b.elevation_deg) == key)
            })
            .map(|x| clone_beam(x))
            .collect::<Vec<_>>();
        let mut prepared = prepare(d, &reference, &bg, gains[&key], &prior)?;
        if let Some(dir) = previous {
            let own = dir.join(format!("{time}.h5"));
            if own.exists() {
                let w: WindowFit = campaign::load(&own)?;
                if let Some(b) = w
                    .beams
                    .iter()
                    .find(|b| format!("{:.2}-{:.2}", b.azimuth_deg, b.elevation_deg) == key)
                {
                    prepared.initial = b.controls.clone();
                }
            }
        }
        prep.push(prepared);
    }
    ensure!(!prep.is_empty(), "no supported beam fits");
    let z: Vec<_> = (550..=1400).step_by(100).map(|x| x as f64).collect();
    let temporal = Temporal {
        altitude_km: z.clone(),
        sigma_per_30min: 0.15,
        neighbors: neighbors
            .iter()
            .map(|n| TimeNeighbor {
                separation_s: (time - n.start_utc_s).abs() as f64,
                h_fraction: z
                    .iter()
                    .map(|z| {
                        n.shape
                            .evaluate(*z, [n.shared_parameters[0], n.shared_parameters[1]])
                            .0
                    })
                    .collect(),
            })
            .collect(),
    };
    let refs: Vec<_> = prep.iter().map(|p| &p.problem).collect();
    let seeds: Vec<_> = prep.iter().map(|p| p.initial.clone()).collect();
    std::fs::create_dir_all(out)?;
    let checkpoint = out.join(format!("{time}.checkpoint.h5"));
    let sol = sharedcomposition::solve_checkpointed(
        &refs,
        &seeds,
        Arc::new(reference.shape.clone()),
        Some(&temporal),
        iterations,
        Some(&checkpoint),
    )?;
    let mut beams = vec![];
    let mut data_component = 0.;
    let mut spatial_component = 0.;
    let mut profile_temporal_component = 0.;
    for (p, nodes) in prep.iter().zip(sol.nodes) {
        let (r, _) = p.problem.evaluate(&coupled::encode(&nodes), false)?;
        data_component += r
            .iter()
            .map(|v| {
                if v.abs() <= 3. {
                    v * v
                } else {
                    6. * v.abs() - 9.
                }
            })
            .sum::<f64>();
        let x = coupled::encode(&nodes);
        for pr in &p.problem.coordinate_priors {
            let z =
                (pr.coefficients.iter().map(|(c, w)| x[*c] * w).sum::<f64>() - pr.mean) / pr.sigma;
            let cost = 2. * pr.delta.powi(2) * ((1. + (z / pr.delta).powi(2)).sqrt() - 1.);
            if pr.coefficients.len() == 1 {
                profile_temporal_component += cost;
            } else {
                spatial_component += cost;
            }
        }
        let mut obs = vec![];
        let mut pred = vec![];
        let mut resid = vec![];
        let mut sig = vec![];
        for (i, a) in p.problem.products.iter().enumerate() {
            let rr = [r[2 * i], r[2 * i + 1]];
            obs.push(a.value);
            sig.push(a.sigma);
            resid.push(rr);
            pred.push([
                rr[0] * a.sigma[0] + a.value[0],
                rr[1] * a.sigma[1] + a.value[1],
            ]);
        }
        let zs: Vec<_> = (100..=1400).step_by(5).map(|z| z as f64).collect();
        let samples: Vec<_> = zs
            .iter()
            .map(|z| sample(&p.spline, &nodes, &reference.shape, sol.shared, *z))
            .collect();
        let chi = r.rows(0, 2 * p.problem.products.len()).norm_squared();
        let (pulses, kept) = pulse_counts(&p.decoded);
        beams.push(BeamFit {
            id: p.decoded.job.id.clone(),
            azimuth_deg: p.decoded.job.az,
            elevation_deg: p.decoded.job.el,
            exposure_s: kept as f64 * 0.016,
            input_pulses: pulses,
            kept_pulses: kept,
            spline: p.spline.clone(),
            controls: nodes,
            altitude_km: zs,
            ne_m3: samples.iter().map(|a| a[0]).collect(),
            ti_k: samples.iter().map(|a| a[1]).collect(),
            te_k: samples.iter().map(|a| a[2]).collect(),
            velocity_m_s: samples.iter().map(|a| a[3]).collect(),
            h_fraction: samples.iter().map(|a| a[4]).collect(),
            raw_reduced_chi_square: chi
                / (2 * p.problem.products.len() - 4 * p.spline.count()).max(1) as f64,
            residual_rms_sigma: (chi / (2 * p.problem.products.len()) as f64).sqrt(),
            measurement_gate: p.gate.clone(),
            measurement_lag: p.lag.clone(),
            residual: resid,
            observed: obs,
            predicted: pred,
            assigned_sigma: sig,
        });
    }
    let value = WindowFit {
        start_utc_s: time,
        end_utc_s: time + 1800,
        shape: reference.shape,
        shared_parameters: sol.shared,
        status: sol.status,
        iterations: sol.iterations,
        normalized_gradient: sol.normalized_gradient,
        objective_history: sol.objective_history,
        minimized_objective: sol.robust_objective + sol.temporal_objective,
        h_temporal_objective: sol.temporal_objective,
        beams,
        metadata: serde_json::json!({"unsupported_beams":unsupported_beams,"objective_components":{"data":data_component,"spatial":spatial_component,"profile_temporal":profile_temporal_component},"helium_fixed_zero":true,"density":"Fixed beam-specific empirical gain anchored to IRI, not absolute system calibration","likelihood":"Signed complex products; block SEM diagonal working weights including reference SEM; correlations not fully whitened; no uncertainty intervals","gating":"Lag-expanded summation with common 480us+nu support; exact rectangular pulse/source-cell overlap; signal FIR delta, no noise-as-signal-FIR inference","temporal":"Adjacent 30-minute states; pseudo-Huber delta1.5; weak log-profile process scales2x[.7,.4,.5,.2], shared H fraction sigma.15; each edge counted once in global audit","background":bg.method,"background_amplitude":"Quiet complex ACF shape scaled by each beam-window median far-range zero-lag noise; scale SEM propagated into working weights. Potential far-range plasma contamination remains a limitation.","guard_cells":"50-1500km; parameter spline100-1400km clamped outside. No synthetic fits beyond requested altitude domain."}),
    };
    campaign::save(&out.join(format!("{time}.h5")), &value)?;
    eprintln!("FIT_DONE {time} {} {}", value.status, value.iterations);
    Ok(())
}
fn clone_beam(x: &BeamFit) -> BeamFit {
    serde_json::from_value(serde_json::to_value(x).unwrap()).unwrap()
}
pub fn calibrate(root: &Path) -> Result<()> {
    let m: Manifest = campaign::load(&root.join("index.h5"))?;
    let b: Background = campaign::load(&root.join("background.h5"))?;
    let mut all = BTreeMap::<String, Vec<f64>>::new();
    let mut theory_cache = BTreeMap::new();
    let mut aspect_cache = BTreeMap::new();
    for j in m.jobs.iter().filter(|j| j.start % 7200 == 0) {
        let beam_key = (j.az.to_bits(), j.el.to_bits());
        if !aspect_cache.contains_key(&beam_key) {
            aspect_cache.insert(beam_key, minimum_aspect(j)?);
        }
        if aspect_cache[&beam_key] <= 3. {
            continue;
        }
        let d: Decoded = campaign::load(&root.join("products").join(format!("{}.h5", j.id)))?;
        let r: Reference =
            campaign::load(&root.join("references").join(format!("{}.h5", j.start)))?;
        let (p, gs, ls) = observations(&d, &b);
        let mut v = vec![];
        for (i, (&g, &l)) in gs.iter().zip(&ls).enumerate() {
            let gate = &d.gates[g];
            if l != 1
                || !(250.0..=400.).contains(&gate.altitude_km)
                || p[i].value[0] <= 5. * p[i].sigma[0]
            {
                continue;
            }
            let ne = interpolate(&r.altitude_km, &r.ne_m3, gate.altitude_km);
            let ti = interpolate(&r.altitude_km, &r.ti_k, gate.altitude_km);
            let te = interpolate(&r.altitude_km, &r.te_k, gate.altitude_km);
            let key = (beam_key, gate.range_km.to_bits());
            if !theory_cache.contains_key(&key) {
                theory_cache.insert(
                    key,
                    Theory::from_beam_at_year(
                        gate.range_km,
                        j.az,
                        j.el,
                        500.,
                        2023. + 319. / 365.25,
                    )?,
                );
            }
            let theory = theory_cache.get_mut(&key).unwrap();
            for c in 0..4 {
                theory.heavy_ion_fractions[c] = interpolate(
                    &r.altitude_km,
                    &r.heavy_fractions.iter().map(|v| v[c]).collect::<Vec<_>>(),
                    gate.altitude_km,
                );
            }
            let mut density_weight = 0.;
            for zz in (50..1500).step_by(5) {
                let z = zz as f64 + 2.5;
                let rr = lp480::range(z, j.el);
                let edge = [
                    2. * lp480::range(zz as f64, j.el) / C_KM,
                    2. * lp480::range(zz as f64 + 5., j.el) / C_KM,
                ];
                density_weight += lp480::weight(gate, 1, j.front_us, edge)
                    * interpolate(&r.altitude_km, &r.ne_m3, z)
                    / rr.powi(2);
            }
            let (s, mass) = theory.spectrum_with_mass(ti, te, 0., 0., 0., ne);
            let acf = s
                .iter()
                .zip(&theory.f)
                .map(|(s, f)| s * (std::f64::consts::TAU * f * DT).cos() * 500.)
                .sum::<f64>()
                * mass
                * density_weight;
            let gain = acf / p[i].value[0];
            if gain.is_finite() && gain > 0. {
                v.push(gain);
            }
        }
        if !v.is_empty() {
            all.entry(format!("{:.2}-{:.2}", j.az, j.el))
                .or_default()
                .push(median(&mut v));
        }
    }
    let gains: BTreeMap<_, _> = all
        .into_iter()
        .map(|(k, mut v)| (k, median(&mut v)))
        .collect();
    ensure!(!gains.is_empty(), "no density calibration support");
    campaign::save(&root.join("calibration.h5"), &gains)?;
    Ok(())
}

#[cfg(test)]
mod tests {
    use super::*;
    #[test]
    fn changing_receiver_noise_preserves_injected_signed_acf() {
        let mut acf = vec![[-20., 10.]; NL + 1];
        acf[0] = [1000., 0.];
        let b = Background {
            acf: acf.clone(),
            sigma: vec![[0.1; 2]; NL + 1],
            blocks: 100,
            method: "test".into(),
        };
        let blocks = (0..3)
            .map(|i| lp480::Block {
                start: i,
                pulses: 1000,
                kept: 1000,
                range_doppler_raw: vec![],
                range_doppler_pulses: 0,
                sums: (1..=NL)
                    .map(|l| [(2. * acf[l][0] - 7.) * 1000., (2. * acf[l][1] + 4.) * 1000.])
                    .collect(),
                counts: vec![1000; NL],
                noise: acf.iter().map(|v| [v[0] * 2000., v[1] * 2000.]).collect(),
                noise_pairs: vec![1000; NL + 1],
            })
            .collect();
        let d = Decoded {
            schema: "test".into(),
            averaged: None,
            job: lp480::Job {
                id: "test".into(),
                start: 0,
                az: 0.,
                el: 90.,
                front_us: 767.,
                parts: vec![],
            },
            gates: vec![lp480::Gate {
                altitude_km: 300.,
                range_km: 300.,
                index: 100,
                nu: 3,
            }],
            blocks,
            lags_us: vec![],
            rejection: "test".into(),
        };
        let p = observations(&d, &b).0;
        assert_eq!(p.len(), NL);
        for p in p {
            assert!((p.value[0] + 7.).abs() < 1e-12);
            assert!((p.value[1] - 4.).abs() < 1e-12);
        }
    }
    #[test]
    fn molecular_mixture_is_normalized_and_shared_fraction_is_identical() {
        let shape = Shape {
            altitude_km: vec![500., 1000., 2000.],
            scale_height_km: vec![60., 90., 150.],
            source: "test".into(),
        };
        let s = Spline::adaptive_to(1400.);
        let nodes = s
            .controls()
            .iter()
            .map(|_| Node {
                power_adu2: 1.,
                ti_k: 1000.,
                te_k: 1500.,
                velocity_m_s: 0.,
                h_fraction: 0.,
                he_fraction: 0.,
                o_fraction: 1.,
            })
            .collect::<Vec<_>>();
        let q = [0.3, 0.4, 0., 0.3, 0.4];
        for z in [100., 500., 700., 1400.] {
            let a = sample(&s, &nodes, &shape, q, z);
            assert!(a[4] >= 0. && a[4] <= 1.);
            assert_eq!(a[4], sample(&s, &nodes, &shape, q, z)[4]);
            if z <= 500. {
                assert_eq!(a[4], 0.);
            }
        }
    }
}

fn penalty(z: f64, delta: f64) -> f64 {
    2. * delta.powi(2) * ((1. + (z / delta).powi(2)).sqrt() - 1.)
}
pub fn audit(directory: &Path, previous: Option<&Path>) -> Result<bool> {
    let mut fits = BTreeMap::<i64, WindowFit>::new();
    for p in std::fs::read_dir(directory)? {
        let p = p?.path();
        if p.extension().is_some_and(|e| e == "h5") && !p.to_string_lossy().contains("checkpoint") {
            if let Ok(w) = campaign::load::<WindowFit>(&p) {
                fits.insert(w.start_utc_s, w);
            }
        }
    }
    ensure!(!fits.is_empty(), "no fit windows");
    let mut counts = BTreeMap::<String, usize>::new();
    let mut objective = 0.;
    let mut max_h = 0f64;
    let mut max_pred = 0f64;
    let mut comparisons = 0usize;
    for (t, w) in &fits {
        *counts.entry(w.status.clone()).or_default() += 1;
        objective += w.metadata["objective_components"]["data"]
            .as_f64()
            .unwrap_or(f64::NAN)
            + w.metadata["objective_components"]["spatial"]
                .as_f64()
                .unwrap_or(f64::NAN);
        if let Some(next) = fits.get(&(t + 1800)) {
            for z in (550..=1400).step_by(100) {
                let a = w
                    .shape
                    .evaluate(z as f64, [w.shared_parameters[0], w.shared_parameters[1]])
                    .0;
                let b = next
                    .shape
                    .evaluate(
                        z as f64,
                        [next.shared_parameters[0], next.shared_parameters[1]],
                    )
                    .0;
                objective += penalty((a - b) / 0.15, 1.5);
            }
            for b in &w.beams {
                if let Some(c) = next.beams.iter().find(|c| {
                    (c.azimuth_deg - b.azimuth_deg).abs() < 0.01
                        && (c.elevation_deg - b.elevation_deg).abs() < 0.01
                }) {
                    let x = coupled::encode(&b.controls);
                    let y = coupled::encode(&c.controls);
                    for k in 0..b.controls.len() {
                        for q in 0..4 {
                            objective += penalty(
                                (x[6 * k + q] - y[6 * k + q]) / (2. * [0.7, 0.4, 0.5, 0.2][q]),
                                1.5,
                            );
                        }
                    }
                }
            }
        }
        if let Some(dir) = previous {
            let p = dir.join(format!("{t}.h5"));
            if p.exists() {
                let old: WindowFit = campaign::load(&p)?;
                comparisons += 1;
                for z in (500..=1400).step_by(5) {
                    let a = w
                        .shape
                        .evaluate(z as f64, [w.shared_parameters[0], w.shared_parameters[1]])
                        .0;
                    let b = old
                        .shape
                        .evaluate(
                            z as f64,
                            [old.shared_parameters[0], old.shared_parameters[1]],
                        )
                        .0;
                    max_h = max_h.max((a - b).abs());
                }
                for b in &w.beams {
                    if let Some(c) = old.beams.iter().find(|c| c.id == b.id) {
                        let value = b
                            .predicted
                            .iter()
                            .zip(&c.predicted)
                            .zip(&c.assigned_sigma)
                            .map(|((a, b), s)| {
                                (0..2).map(|i| ((a[i] - b[i]) / s[i]).powi(2)).sum::<f64>()
                            })
                            .sum::<f64>()
                            .sqrt();
                        max_pred = max_pred.max(value);
                    }
                }
            }
        }
    }
    let unresolved = counts
        .iter()
        .filter(|(s, _)| s.as_str() != "converged" && s.as_str() != "data_stable_gradient_pending")
        .map(|(_, n)| *n)
        .sum::<usize>();
    let ready = unresolved == 0 && comparisons == fits.len() && max_h < 0.005 && max_pred < 0.1;
    let v = serde_json::json!({"windows":fits.len(),"statuses":counts,"unresolved_fit_windows":unresolved,"joint_objective_edges_counted_once":objective,"compared_windows":comparisons,"maximum_h_fraction_change":max_h,"maximum_beam_prediction_change_sigma_norm":max_pred,"temporal_stable":ready,"note":"Working diagonal-likelihood objective; no covariance-calibrated confidence claim. All finite signed predictions retained."});
    campaign::save(&directory.join("audit.h5"), &v)?;
    println!("{}", v);
    Ok(ready)
}
