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
    #[serde(default)]
    pub short_lag_detection_sigma: Vec<f64>,
    #[serde(default)]
    pub temperature_data_supported: Vec<bool>,
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
/// Curvature of the evaluated profiles, integrated on a physical altitude grid.
/// This avoids treating cubic-spline coefficients as altitude samples, and keeps
/// regularization independent of the number of fine native source cells.
fn profile_curvature_priors(s: &Spline, elevation_deg: f64) -> Vec<CoordinatePrior> {
    let top = *s.knots.last().unwrap();
    let mut priors = Vec::new();
    let h = 2.5;
    let dz = 5.;
    let mut z = 100. + h;
    while z <= top - h {
        let length = (72. * elevation_deg.to_radians().sin())
            .max(45.) + 0.06 * (z - 400f64).max(0.);
        let mut curvature = BTreeMap::<usize, f64>::new();
        for (at, factor) in [(z - h, 1.), (z, -2.), (z + h, 1.)] {
            for (k, weight) in s.weights(at) {
                *curvature.entry(k).or_default() += factor * weight / (h * h);
            }
        }
        let scale = length * length * (dz / length).sqrt();
        for (field, sigma) in [0.35, 0.15, 0.20, 0.025].into_iter().enumerate() {
            priors.push(CoordinatePrior {
                coefficients: curvature.iter().filter(|(_, w)| w.abs() > 1e-14)
                    .map(|(k, w)| (6 * k + field, w * scale)).collect(),
                mean: 0., sigma, delta: 30.,
            });
        }
        z += dz;
    }
    priors
}

pub fn prepare(
    d: Decoded,
    r: &Reference,
    b: &Background,
    gain: f64,
    prior: &[BeamFit],
) -> Result<Prepared> {
    let top = lp480::beam_top_km(&d.job);
    let s = Spline::adaptive_to(top as f64);
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
    if std::env::var_os("SYISR_LP480_PROFILE_CURVATURE").is_some() {
        priors.extend(profile_curvature_priors(&s, d.job.el));
    } else {
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
    }
    if std::env::var_os("SYISR_TOPSIDE_TEMPERATURE_SLOPE").is_some() {
        // A soft gradient anchors weak topside temperatures to the measured
        // lower-F profile, without imposing IRI temperatures or hard clipping.
        for k in 0..zc.len() - 1 {
            if zc[k + 1] <= 400. {
                continue;
            }
            let dz = (zc[k + 1] - zc[k]) / 100.;
            for (field, sigma) in [(1, 0.20), (2, 0.30)] {
                priors.push(CoordinatePrior {
                    coefficients: vec![(6 * k + field, -1.), (6 * (k + 1) + field, 1.)],
                    mean: 0.,
                    sigma: sigma * dz,
                    delta: 3.,
                });
            }
        }
    }
    let shape = Arc::new(r.shape.clone());
    let mut cells = vec![];
    let mut cache = BTreeMap::<i64, Arc<Theory>>::new();
    let nf = 201;
    let frequency: Vec<_> = (0..nf).map(|i| -50000. + i as f64 * 500.).collect();
    for zz in (50..top + 100).step_by(5) {
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
                lp480::decimal_year(d.job.start)?,
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
/// Reproject an earlier profile as an initialization, never as new measurements.
pub fn project_initial(old: &BeamFit, s: &Spline) -> Result<Vec<Node>> {
    project_controls(&old.spline, &old.controls, s)
}
fn project_controls(old_s: &Spline, old_nodes: &[Node], s: &Spline) -> Result<Vec<Node>> {
    let z = s.controls();
    let n = z.len();
    let x = coupled::encode(old_nodes);
    let mut a = nalgebra::DMatrix::<f64>::zeros(n, n);
    let mut y = nalgebra::DMatrix::<f64>::zeros(n, 4);
    for (i, z) in z.iter().enumerate() {
        for (k, w) in s.weights(*z) {
            a[(i, k)] = w;
        }
        for (k, w) in old_s.weights(*z) {
            for f in 0..4 {
                y[(i, f)] += w * x[6 * k + f];
            }
        }
    }
    let c = a
        .lu()
        .solve(&y)
        .ok_or_else(|| anyhow::anyhow!("singular warm-start projection"))?;
    let mut nodes = (0..n)
        .map(|i| Node {
            power_adu2: c[(i, 0)].exp(),
            ti_k: (1000. * c[(i, 1)].exp()).clamp(150., 6000.),
            te_k: (1000. * c[(i, 2)].exp()).clamp(150., 12000.),
            velocity_m_s: (1000. * c[(i, 3)]).clamp(-2000., 2000.),
            h_fraction: 0.,
            he_fraction: 0.,
            o_fraction: 1.,
        })
        .collect::<Vec<_>>();
    nodes[0].h_fraction = old_nodes[0].h_fraction;
    nodes[1].h_fraction = old_nodes[1].h_fraction;
    for k in 2..5 {
        nodes[k].he_fraction = old_nodes[k].he_fraction;
    }
    Ok(nodes)
}
/// Preserve the validated collisionless-electron aspect guard over the full pulse support.
pub fn minimum_aspect(j: &lp480::Job) -> Result<f64> {
    let mut minimum = 90f64;
    for zz in (50..lp480::beam_top_km(j) + 100).step_by(5) {
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
            lp480::decimal_year(j.start)?,
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
    let mut own_previous = None;
    if let Some(dir) = previous {
        let p = dir.join(format!("{time}.h5"));
        if p.exists() {
            own_previous = Some(campaign::load::<WindowFit>(&p)?);
        }
    }
    if own_previous.is_none() {
        if let Ok(dir) = std::env::var("SYISR_INITIAL_PROFILE_DIR") {
            let p = Path::new(&dir).join(format!("{time}.h5"));
            if p.exists() {
                own_previous = Some(campaign::load::<WindowFit>(&p)?);
            }
        }
    }
    let stabilized_previous = own_previous.as_ref().is_some_and(|w| {
        w.metadata["topside_temperature_gradient_prior"]
            .as_str()
            .is_some_and(|s| s != "off")
            && w.metadata["fit_top_altitude_km"].as_u64().unwrap_or(1400)
                == lp480::configured_top_km() as u64
    });
    let mut prep = vec![];
    let mut unsupported_beams = vec![];
    for j in jobs {
        if !gains.contains_key(&format!("{:.2}-{:.2}", j.az, j.el)) {
            let aspect = minimum_aspect(j)?;
            unsupported_beams.push(serde_json::json!({"azimuth_deg":j.az,"elevation_deg":j.el,"minimum_aspect_deg":aspect,"reason":if aspect<=3.{"Finite-pulse support approaches within3degrees of perpendicular B; collisionless model not validated"}else{"No supported empirical density calibration"}}));
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
        if let Some(w) = &own_previous {
            if let Some(b) = w
                .beams
                .iter()
                .find(|b| format!("{:.2}-{:.2}", b.azimuth_deg, b.elevation_deg) == key)
            {
                prepared.initial = if b.spline.knots == prepared.spline.knots {
                    b.controls.clone()
                } else {
                    project_initial(b, &prepared.spline)?
                };
            }
        }
        if let Some(dir) = std::env::var("SYISR_WARMSTART_DIR").ok().filter(|_| own_previous.is_none()) {
            let cp = Path::new(&dir).join(format!("{time}.checkpoint.h5"));
            if cp.exists() {
                let v: serde_json::Value = campaign::load(&cp)?;
                let nodes: Vec<Vec<Node>> = serde_json::from_value(v["nodes"].clone())?;
                let position = prep.len();
                ensure!(
                    position < nodes.len() && nodes[position].len() == prepared.initial.len(),
                    "warm-start dimensions"
                );
                prepared.initial = nodes[position].clone();
            }
        }
        prep.push(prepared);
    }
    ensure!(!prep.is_empty(), "no supported beam fits");
    let z: Vec<_> = (550..=lp480::configured_top_km())
        .step_by(100)
        .map(|x| x as f64)
        .collect();
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
    let solver = if std::env::var_os("SYISR_LBFGS").is_some() {
        sharedcomposition::solve_lbfgs
    } else {
        sharedcomposition::solve_checkpointed
    };
    let mut multistart_report = vec![];
    if std::env::var_os("SYISR_MULTISTART").is_some()
        && !stabilized_previous
        && !checkpoint.exists()
    {
        let mut cooled = seeds.clone();
        for (nodes, p) in cooled.iter_mut().zip(&prep) {
            let weights = p.spline.weights(450.);
            let anchor = |field: usize| {
                let x = coupled::encode(nodes);
                weights
                    .iter()
                    .map(|(k, w)| w * x[6 * k + field])
                    .sum::<f64>()
            };
            let ti = (anchor(1)).exp() * 1000.;
            let te = (anchor(2)).exp() * 1000.;
            let velocity = anchor(3) * 1000.;
            for (node, z) in nodes.iter_mut().zip(p.spline.controls()) {
                if z > 450. {
                    node.ti_k = ti * (z / 450.).powf(0.25);
                    node.te_k = te * (z / 450.).powf(0.4);
                    node.velocity_m_s = velocity;
                }
            }
        }
        let mut best_cooled = None;
        let mut best_score = f64::INFINITY;
        for center in [650., 800., 1000., 1250., 1550., 1800.] {
            for width in [60., 150., 300.] {
                let q = Shape::encode(center, width);
                let mut candidate = cooled.clone();
                for b in &mut candidate {
                    b[0].h_fraction = q[0];
                    b[1].h_fraction = q[1];
                }
                let cost = sharedcomposition::objective_at(
                    &refs,
                    &candidate,
                    &reference.shape,
                    Some(&temporal),
                )?;
                if cost < best_score {
                    best_score = cost;
                    best_cooled = Some(candidate);
                }
            }
        }
        let candidates = [seeds.clone(), best_cooled.unwrap()];
        let mut best = f64::INFINITY;
        let mut chosen = None;
        for (i, candidate) in candidates.iter().enumerate() {
            let file = out.join(format!("{time}.seed-{i}.checkpoint.h5"));
            let solution = solver(
                &refs,
                candidate,
                Arc::new(reference.shape.clone()),
                Some(&temporal),
                100,
                Some(&file),
            )?;
            let cost = solution.robust_objective + solution.temporal_objective;
            multistart_report.push(serde_json::json!({"seed":if i==0{"previous model"}else{"lower-F anchored temperature plus H initialization grid"},"penalized_objective":cost,"status":solution.status,"shared_parameters":solution.shared}));
            if cost < best {
                best = cost;
                chosen = Some(file);
            }
        }
        std::fs::copy(chosen.unwrap(), &checkpoint)?;
    }
    let sol = solver(
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
        let zs: Vec<_> = (100..=lp480::beam_top_km(&p.decoded.job))
            .step_by(5)
            .map(|z| z as f64)
            .collect();
        let mut detection_by_gate = std::collections::BTreeMap::new();
        let mut lag_support = std::collections::BTreeMap::<usize, usize>::new();
        for ((a, g), l) in p.problem.products.iter().zip(&p.gate).zip(&p.lag) {
            *lag_support.entry(*g).or_default() += 1;
            if *l == 1 {
                detection_by_gate.insert(*g, a.value[0] / a.sigma[0]);
            }
        }
        let detection: Vec<f64> = zs
            .iter()
            .map(|z| {
                detection_by_gate
                    .iter()
                    .min_by(|(a, _), (b, _)| {
                        (p.decoded.gates[**a].altitude_km - z)
                            .abs()
                            .total_cmp(&(p.decoded.gates[**b].altitude_km - z).abs())
                    })
                    .map_or(0., |(_, v)| *v)
            })
            .collect();
        let sufficient_lags = zs
            .iter()
            .map(|z| {
                detection_by_gate
                    .keys()
                    .min_by(|a, b| {
                        (p.decoded.gates[**a].altitude_km - z)
                            .abs()
                            .total_cmp(&(p.decoded.gates[**b].altitude_km - z).abs())
                    })
                    .is_some_and(|g| lag_support.get(g).copied().unwrap_or(0) >= 8)
            })
            .collect::<Vec<_>>();
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
            temperature_data_supported: samples
                .iter()
                .zip(&detection)
                .zip(&sufficient_lags)
                .map(|((s, snr), enough)| {
                    *enough
                        && *snr >= 5.
                        && s[1] > 151.
                        && s[1] < 5900.
                        && s[2] > 151.
                        && s[2] < 11800.
                })
                .collect(),
            short_lag_detection_sigma: detection,
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
    let mut value = WindowFit {
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
        metadata: serde_json::json!({"multistart":multistart_report,"topside_temperature_gradient_prior":if std::env::var_os("SYISR_TOPSIDE_TEMPERATURE_SLOPE").is_some(){"Soft log-Ti/log-Te increments above400km, sigma0.20/0.30 per100km, pseudo-Huber delta3; no IRI temperature amplitude prior"}else{"off"},"quality_flag":"Short-lag detection>=5assigned sigma and temperatures not at bounds is only a necessary data-support flag, not calibrated parameter uncertainty","optimizer":if std::env::var_os("SYISR_LBFGS").is_some(){"L-BFGS"}else{"Levenberg-Marquardt"},"optimizer_data_stability":sol.data_stability,"unsupported_beams":unsupported_beams,"objective_components":{"data":data_component,"spatial":spatial_component,"profile_temporal":profile_temporal_component},"helium_fixed_zero":true,"density":"Fixed beam-specific empirical gain anchored to IRI, not absolute system calibration","likelihood":"Signed complex products; block SEM diagonal working weights including reference SEM; correlations not fully whitened; no uncertainty intervals","gating":"Lag-expanded summation with common 480us+nu support; exact rectangular pulse/source-cell overlap; signal FIR delta, no noise-as-signal-FIR inference","temporal":"Adjacent 30-minute states; pseudo-Huber delta1.5; weak log-profile process scales2x[.7,.4,.5,.2], shared H fraction sigma.15; each edge counted once in global audit","background":bg.method,"background_amplitude":"Quiet complex ACF shape scaled by each beam-window median far-range zero-lag noise; scale SEM propagated into working weights. Potential far-range plasma contamination remains a limitation.","guard_cells":"50-1500km; parameter spline100-1400km clamped outside. No synthetic fits beyond requested altitude domain."}),
    };
    let calibration: serde_json::Value = if root.join("calibration_metadata.h5").exists() {
        campaign::load(&root.join("calibration_metadata.h5"))?
    } else {
        serde_json::json!({"density_source":"IRI","method":"Fixed empirical gain per beam"})
    };
    value.metadata["density"] = serde_json::json!(format!(
        "Fixed empirical gain per beam anchored to {}; not an absolute system calibration",
        calibration["density_source"]
            .as_str()
            .unwrap_or("unspecified reference")
    ));
    value.metadata["density_calibration"] = calibration;
    value.metadata["fit_top_altitude_km"] = serde_json::json!(lp480::configured_top_km());
    value.metadata["altitude_regularization"] = if std::env::var_os("SYISR_LP480_PROFILE_CURVATURE").is_some() {
        serde_json::json!({"version":"evaluated-curvature-v2","quadrature_km":5.,"derivative_half_step_km":2.5,"length_km":"max(45,72*sin(el))+0.06*max(z-400,0)","sigmas_logNe_logTi_logTe_vi_over1000":[0.35,0.15,0.20,0.025],"robust_delta":30.,"note":"Altitude smoothness prior, not observational resolution or uncertainty"})
    } else { serde_json::json!({"version":"control-differences-v1"}) };
    value.metadata["quality_flag"]=serde_json::json!("Temperature support requires >=5 assigned-sigma short-lag detection, >=8 valid lags, and no temperature bound; preliminary support, not validated parameter uncertainty.");
    value.metadata["guard_cells"]=serde_json::json!("5km source cells from50km to100km beyond each beam's fitted top; clamped profiles in guard cells. Beam-specific top limited by valid recorded lag support.");
    value.metadata["warm_start"]=serde_json::json!("Earlier controls, when present on a different grid, are projected in logNe/logTi/logTe/velocity coordinates; clamped continuation is initialization only and all new gates are refitted.");
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
    let l2: Option<crate::lp480_l2::Profiles> = if root.join("density_reference.h5").exists() {
        Some(campaign::load(&root.join("density_reference.h5"))?)
    } else {
        None
    };
    let mut all = BTreeMap::<String, Vec<f64>>::new();
    let mut theory_cache = BTreeMap::new();
    let mut aspect_cache = BTreeMap::new();
    for j in m.jobs.iter().filter(|j| j.start % 7200 == 0) {
        let density_profile = l2.as_ref().and_then(|v| v.get(&j.id));
        if l2.is_some() && density_profile.is_none() {
            continue;
        }
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
            let (density_z, density_ne) =
                density_profile.map_or((&r.altitude_km, &r.ne_m3), |v| (&v.altitude_km, &v.ne_m3));
            // Require coverage over the complete calibration gate pulse support.
            if gate.altitude_km - 50. < density_z[0]
                || gate.altitude_km + 50. > *density_z.last().unwrap()
            {
                continue;
            }
            let ne = interpolate(density_z, density_ne, gate.altitude_km);
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
                        lp480::decimal_year(j.start)?,
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
                    * interpolate(density_z, density_ne, z)
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
    campaign::save(
        &root.join("calibration_metadata.h5"),
        &serde_json::json!({
            "density_source":if l2.is_some(){"Supplied L2"}else{"IRI"},
            "method":"Fixed empirical gain per beam; screened F-region short-lag samples at2-hour intervals; finite480us pulse density weighting",
            "qualification":"Not an absolute system calibration; IRI temperatures and heavy-ion proportions enter reference spectral shape",
            "calibrated_beams":gains.len()
        }),
    )?;
    Ok(())
}

#[cfg(test)]
mod tests {
    use super::*;
    #[test]
    fn physical_curvature_preserves_linear_profiles_and_penalizes_ringing() {
        let spline = Spline::adaptive_to(1800.);
        let controls = spline.controls();
        let priors = profile_curvature_priors(&spline, 90.);
        for p in &priors {
            let constant = p.coefficients.iter().map(|(_, w)| w).sum::<f64>();
            let linear = p.coefficients.iter().map(|(i,w)| w * controls[i / 6]).sum::<f64>();
            assert!(constant.abs() < 1e-9);
            assert!(linear.abs() < 1e-6);
        }
        let cost = |alternating: bool| priors.iter().filter(|p|p.coefficients[0].0 % 6 == 3).map(|p| {
            let r = p.coefficients.iter().map(|(i,w)| {
                let k = i / 6;
                w * if alternating { if k % 2 == 0 {0.05} else {-0.05} }
                    else {0.05 * (controls[k] / 400.).sin()}
            }).sum::<f64>() / p.sigma;
            r*r
        }).sum::<f64>();
        assert!(cost(true) > 100. * cost(false));
    }
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
        let extended = Spline::adaptive_to(1800.);
        let projected = project_controls(&s, &nodes, &extended).unwrap();
        for z in [100., 500., 1400., 1800.] {
            let p = sample(&extended, &projected, &shape, q, z);
            assert!((p[0] / 1e11 - 1.).abs() < 1e-12);
            assert!((p[1] - 1000.).abs() < 1e-9);
            assert!((p[2] - 1500.).abs() < 1e-9);
        }
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
            for z in (550..=lp480::configured_top_km()).step_by(100) {
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
                for z in (500..=lp480::configured_top_km()).step_by(5) {
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
