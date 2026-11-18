//! Joint altitude-profile LOS inversion with cubic splines and a curvature penalty.
use anyhow::{ensure, Result};
use nalgebra::{DMatrix as M, DVector as V};
use serde_json::{json, Value};
use std::{
    collections::{BTreeMap, BTreeSet},
    path::Path,
};
use syisr_ac::{campaign, spline::Spline};
#[derive(Clone)]
struct Obs {
    z: f64,
    id: String,
    a: [f64; 3],
    y: f64,
}
fn spline() -> Spline {
    let mut k = vec![250.; 4];
    for z in (280..650).step_by(30) {
        k.push(z as f64);
    }
    k.extend([650.; 4]);
    Spline { knots: k }
}
fn solve(obs: &[Obs], s: &Spline, length: f64, skip: Option<&str>) -> V<f64> {
    let n = s.count() * 3;
    let mut penalty = M::zeros(n, n);
    for z in (260..650).step_by(10) {
        let mut d = vec![0.; s.count()];
        for (zz, c) in [(z - 10, 1.), (z, -2.), (z + 10, 1.)] {
            for (k, w) in s.weights(zz as f64) {
                d[k] += c * w * length * length / 100.;
            }
        }
        for q in 0..3 {
            for i in 0..d.len() {
                for j in 0..d.len() {
                    penalty[(3 * i + q, 3 * j + q)] += d[i] * d[j];
                }
            }
        }
    }
    let mut x = V::zeros(n);
    for _ in 0..8 {
        let mut h = penalty.clone();
        let mut g = V::zeros(n);
        for o in obs.iter().filter(|o| skip != Some(o.id.as_str())) {
            let b = s.weights(o.z);
            let mut aa = vec![];
            for (k, w) in b {
                for q in 0..3 {
                    aa.push((3 * k + q, w * o.a[q]));
                }
            }
            let pred = aa.iter().map(|(i, a)| x[*i] * a).sum::<f64>();
            let w = (30. / (o.y - pred).abs().max(1e-10)).min(1.);
            for (i, a) in &aa {
                g[*i] += w * a * o.y;
                for (j, b) in &aa {
                    h[(*i, *j)] += w * a * b;
                }
            }
        }
        let next = h
            .cholesky()
            .expect("profile geometry must have full rank")
            .solve(&g);
        let change = (&next - &x).norm();
        x = next;
        if change < 1e-5 {
            break;
        }
    }
    x
}
fn value(x: &V<f64>, s: &Spline, z: f64) -> [f64; 3] {
    let mut v = [0.; 3];
    for (k, w) in s.weights(z) {
        for q in 0..3 {
            v[q] += w * x[3 * k + q];
        }
    }
    v
}
fn dot(a: [f64; 3], b: [f64; 3]) -> f64 {
    a.iter().zip(b).map(|(a, b)| a * b).sum()
}
fn loss(r: f64) -> f64 {
    let a = r.abs();
    if a <= 30. {
        a * a
    } else {
        60. * a - 900.
    }
}
fn main() -> Result<()> {
    let a: Vec<_> = std::env::args().collect();
    ensure!(a.len() == 3, "DRIFT_INPUT OUTPUT");
    let mut v: Value = campaign::load(Path::new(&a[1]))?;
    let rows = v["rows"].as_array().unwrap();
    let mut groups: BTreeMap<i64, Vec<usize>> = BTreeMap::new();
    for (i, r) in rows.iter().enumerate() {
        if r["altitude_km"].as_i64().unwrap() <= 650 {
            groups
                .entry(r["start_utc_s"].as_i64().unwrap())
                .or_default()
                .push(i);
        }
    }
    let s = spline();
    let mut reports = vec![];
    for (time, idx) in groups {
        for kind in ["magnetic", "geographic"] {
            let mut obs = vec![];
            for i in &idx {
                let r = &v["rows"][*i];
                for b in r["beams"].as_array().unwrap() {
                    let id = format!(
                        "{:.3}-{:.3}",
                        b["azimuth_deg"].as_f64().unwrap(),
                        b["elevation_deg"].as_f64().unwrap()
                    );
                    let ar = &b["geometry"][if kind == "magnetic" {
                        "magnetic_row"
                    } else {
                        "enu_row"
                    }];
                    obs.push(Obs {
                        z: r["altitude_km"].as_f64().unwrap(),
                        id,
                        a: [
                            ar[0].as_f64().unwrap(),
                            ar[1].as_f64().unwrap(),
                            ar[2].as_f64().unwrap(),
                        ],
                        y: b["los_m_s"].as_f64().unwrap(),
                    });
                }
            }
            let ids: Vec<_> = obs
                .iter()
                .map(|o| o.id.clone())
                .collect::<BTreeSet<_>>()
                .into_iter()
                .collect();
            let mut candidates = vec![];
            let mut best = (f64::INFINITY, 0., vec![]);
            for length in [0., 15., 30., 60.] {
                let folds: Vec<_> = ids
                    .iter()
                    .map(|id| solve(&obs, &s, length, Some(id)))
                    .collect();
                let mut sum = 0.;
                let mut count = 0;
                for (id, x) in ids.iter().zip(&folds) {
                    for o in obs
                        .iter()
                        .filter(|o| &o.id == id && o.z >= 300. && o.z <= 600.)
                    {
                        sum += loss(o.y - dot(o.a, value(x, &s, o.z)));
                        count += 1;
                    }
                }
                let score = sum / count as f64;
                candidates
                    .push(json!({"curvature_length_km":length,"held_beam_mean_huber_m2_s2":score}));
                if length > 0. && score < best.0 {
                    best = (score, length, folds);
                }
            }
            let (_, length, folds) = best;
            let x = solve(&obs, &s, length, None);
            for i in &idx {
                let z = v["rows"][*i]["altitude_km"].as_f64().unwrap();
                let vel = value(&x, &s, z);
                let here: Vec<_> = obs.iter().filter(|o| o.z == z).collect();
                let residual: Vec<_> = here.iter().map(|o| o.y - dot(o.a, vel)).collect();
                let rms =
                    (residual.iter().map(|v| v * v).sum::<f64>() / residual.len() as f64).sqrt();
                let loov: Vec<_> = folds.iter().map(|x| value(x, &s, z)).collect();
                let mut spread = [0.; 3];
                for q in 0..3 {
                    let mean = loov.iter().map(|v| v[q]).sum::<f64>() / loov.len() as f64;
                    spread[q] = ((loov.len() - 1) as f64 / loov.len() as f64
                        * loov.iter().map(|v| (v[q] - mean).powi(2)).sum::<f64>())
                    .sqrt();
                }
                let maxerr = here
                    .iter()
                    .map(|o| {
                        let j = ids.iter().position(|id| id == &o.id).unwrap();
                        (o.y - dot(o.a, loov[j])).abs()
                    })
                    .fold(0., f64::max);
                let old = v["rows"][*i][kind].clone();
                let cond = old["condition_number"].as_f64().unwrap();
                let ok = cond < 15. && rms < 30. && maxerr < 75. && spread.iter().all(|v| *v < 50.);
                v["rows"][*i][format!("unregularized_{kind}")] = old;
                v["rows"][*i][kind] = json!({"velocity_m_s":vel,"condition_number":cond,"residual_m_s":residual,"residual_rms_m_s":rms,"jackknife_spread_m_s":spread,"maximum_leave_one_out_error_m_s":maxerr,"consistency_ok":ok,"curvature_length_km":length});
            }
            reports.push(json!({"start_utc_s":time,"basis":kind,"selected_length_km":length,"candidates":candidates}));
        }
    }
    v["regularization"] = json!({"altitude_domain_km":[250,650],"model":"Joint cubic spline velocity profile fit to beam LOS values, sum robust residuals + squared length^2 times second altitude derivative at10km samples; natural free endpoint behavior; controls30km, evaluated10km","selection":"15,30,60km curvature lengths compared by withholding each entire beam; lowest300-600km mean Huber prediction loss selected separately per time and component basis","qualification":"Applied to velocity reconstruction, not to the original AC plasma parameter likelihood. Altitude samples inherit correlation from the AC spline fits. No statistical error-bar claim. No temporal smoothing or extrapolation beyond650km.","reports":reports});
    v["schema"] = json!("syisr-regional-drift-altitude-regularized-v1");
    campaign::save(Path::new(&a[2]), &v)?;
    println!(
        "regularized profiles {}",
        v["regularization"]["reports"].as_array().unwrap().len()
    );
    Ok(())
}

#[cfg(test)]
mod tests {
    use super::*;
    #[test]
    fn curvature_penalty_preserves_affine_flow_and_predicts_omitted_beam() {
        let s = spline();
        let mut obs = vec![];
        for z in (250..=650).step_by(10) {
            for (i, (az, el)) in [
                (88., 60.),
                (268., 60.),
                (178., 45.),
                (178., 75.),
                (358., 75.),
                (358., 90.),
            ]
            .iter()
            .enumerate()
            {
                let a = syisr_ac::drift::geometry(z as f64, *az, *el)
                    .unwrap()
                    .magnetic_row;
                let truth = [20. + 0.1 * (z as f64 - 400.), -15., 5.];
                obs.push(Obs {
                    z: z as f64,
                    id: i.to_string(),
                    a,
                    y: dot(a, truth),
                });
            }
        }
        for length in [0., 15., 60.] {
            let x = solve(&obs, &s, length, Some("0"));
            for z in [250., 400., 650.] {
                let fit = value(&x, &s, z);
                let truth = [20. + 0.1 * (z - 400.), -15., 5.];
                for q in 0..3 {
                    assert!(
                        (fit[q] - truth[q]).abs() < 1e-7,
                        "length={length} z={z} q={q}"
                    );
                }
            }
        }
    }
}
