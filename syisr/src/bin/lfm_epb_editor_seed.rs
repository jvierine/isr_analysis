//! Conditional field-aligned density-contrast proxy with frozen cross-field advection.
//! Uses measured signed echo ratios and empirical weights; no absolute Ne calibration.
use anyhow::{ensure, Result};
use hdf5::File;
use std::{fs, path::Path};
const OBS_COLS: usize = 34;
const NX: usize = 701;
const NY: usize = 351;
const DX: f64 = 2.;
const X0: f64 = -700.;
const Y0: f64 = -300.;
const TREF: f64 = 1772369100.; // 2026-03-01 12:45 UTC
fn vec(h: &File, k: &str, v: &[f64]) -> Result<()> {
    h.new_dataset_builder().with_data(v).create(k)?;
    Ok(())
}
fn mat(h: &File, k: &str, v: &[f64], nr: usize, nc: usize) -> Result<()> {
    h.new_dataset::<f64>()
        .shape((nr, nc))
        .create(k)?
        .write_raw(v)?;
    Ok(())
}
fn med(v: &mut [f64]) -> f64 {
    v.sort_by(f64::total_cmp);
    (v[(v.len() - 1) / 2] + v[v.len() / 2]) / 2.
}
fn prepare(root: &Path, out: &Path) -> Result<()> {
    let geom = File::open(root.join("geometry.h5"))?;
    let mut paths: Vec<_> = fs::read_dir(root.join("beams"))?
        .filter_map(|e| e.ok().map(|e| e.path()))
        .filter(|p| p.is_dir())
        .collect();
    paths.sort();
    let mut rows = Vec::new();
    let mut masks = Vec::new();
    let mut beamstats = Vec::new();
    for (beam, path) in paths.iter().enumerate() {
        let h = File::open(path.join("inversion.h5"))?;
        let name = path.file_name().unwrap().to_str().unwrap();
        let t = h.dataset("time_unix_s")?.read_raw::<f64>()?;
        let z = h.dataset("altitude_km")?.read_raw::<f64>()?;
        let ranges = h.dataset("range_km")?.read_raw::<f64>()?;
        let sig = h.dataset("median_signal")?.read_raw::<f64>()?;
        let noise = h.dataset("noise_power")?.read_raw::<f64>()?;
        let mapped = geom
            .group(name)?
            .dataset("field_line_reference_plane_km")?
            .read_raw::<f64>()?;
        let linear = geom
            .group(name)?
            .dataset("straight_B_projection_km")?
            .read_raw::<f64>()?;
        let quad = geom
            .group(name)?
            .dataset("beam_quadrature_field_line_km")?
            .read_raw::<f64>()?;
        let aspect = geom
            .group(name)?
            .dataset("aspect_deviation_from_perpendicular_deg")?
            .read_raw::<f64>()?;
        let width = geom
            .group(name)?
            .dataset("oneway_fwhm_elevation_azimuth_deg")?
            .read_raw::<f64>()?;
        let nr = z.len();
        let mut nn = noise.clone();
        let nmed = med(&mut nn);
        let mut nused = 0.;
        let mut nexcluded = 0.;
        for k in (0..nr).step_by(2) {
            if z[k] < 200. || z[k] > 900. {
                continue;
            }
            let mut quiet: Vec<f64> = t
                .iter()
                .enumerate()
                .filter(|(_, t)| **t >= 1772362800. && **t < 1772364600.)
                .map(|(j, _)| sig[j * nr + k])
                .filter(|v| v.is_finite())
                .collect();
            if quiet.len() < 20 {
                continue;
            }
            let reference = med(&mut quiet);
            if reference < 0.05 * nmed {
                continue;
            }
            for (j, &time) in t.iter().enumerate() {
                if time < 1772365500. || time >= 1772372700. {
                    continue;
                } //11:45-13:45 UTC
                let signal = sig[j * nr + k];
                let ratio = signal / reference;
                let sigma = ((0.08 * noise[j] / reference).powi(2) + 0.1f64.powi(2)).sqrt();
                let usable = ratio.is_finite() && ratio < 6. && sigma.is_finite();
                rows.extend([
                    time,
                    beam as f64,
                    ranges[k],
                    z[k],
                    mapped[3 * k],
                    mapped[3 * k + 1],
                    ratio,
                    sigma,
                    reference,
                    signal,
                    noise[j],
                    linear[3 * k],
                    linear[3 * k + 1],
                ]);
                for q in 0..9 {
                    rows.extend([quad[(k * 9 + q) * 3], quad[(k * 9 + q) * 3 + 1]]);
                }
                rows.extend([aspect[k], width[0], width[1]]);
                masks.push(if usable {
                    nused += 1.;
                    1.
                } else {
                    nexcluded += 1.;
                    0.
                });
            }
        }
        beamstats.extend([
            beam as f64,
            h.attr("beam_azimuth_deg")?.read_scalar::<f64>()?,
            h.attr("beam_elevation_deg")?.read_scalar::<f64>()?,
            nused,
            nexcluded,
        ]);
    }
    let out = File::create(out)?;
    mat(&out, "observations", &rows, masks.len(), OBS_COLS)?;
    vec(&out, "eligible", &masks)?;
    mat(&out, "beam_statistics", &beamstats, paths.len(), 5)?;
    println!(
        "PREPARED {} observations across {} beams",
        masks.len(),
        paths.len()
    );
    Ok(())
}
#[derive(Clone)]
struct Row {
    index: Vec<usize>,
    a: Vec<f64>,
    y: f64,
    w: f64,
    beam: usize,
    obs: usize,
}
fn interpolation(x: f64, y: f64) -> Option<([usize; 4], [f64; 4])> {
    let u = (x - X0) / DX;
    let v = (y - Y0) / DX;
    if u < 0. || v < 0. || u >= (NX - 1) as f64 || v >= (NY - 1) as f64 {
        return None;
    }
    let i = u.floor() as usize;
    let j = v.floor() as usize;
    let a = u - i as f64;
    let b = v - j as f64;
    let k = j * NX + i;
    Some((
        [k, k + 1, k + NX, k + NX + 1],
        [(1. - a) * (1. - b), a * (1. - b), (1. - a) * b, a * b],
    ))
}
fn predict(row: &Row, x: &[f64]) -> f64 {
    row.index.iter().zip(&row.a).map(|(i, a)| a * x[*i]).sum()
}
fn heldout(beam: usize, fold: i32) -> bool {
    if fold < 0 {
        false
    } else if fold >= 100 {
        beam == (fold - 100) as usize
    } else {
        beam % 3 == fold as usize
    }
}
fn apply(x: &[f64], rows: &[Row], lambda: f64, ridge: f64) -> Vec<f64> {
    let mut y: Vec<f64> = x.iter().map(|v| v * ridge).collect();
    for r in rows {
        let s = predict(r, x) * r.w;
        for k in 0..r.index.len() {
            y[r.index[k]] += r.a[k] * s;
        }
    }
    for j in 0..NY {
        for i in 0..NX {
            let k = j * NX + i;
            for n in [
                if i + 1 < NX { Some(k + 1) } else { None },
                if j + 1 < NY { Some(k + NX) } else { None },
            ]
            .into_iter()
            .flatten()
            {
                let d = lambda * (x[k] - x[n]);
                y[k] += d;
                y[n] -= d;
            }
        }
    }
    y
}
fn objective(x: &[f64], rows: &[Row], lambda: f64, ridge: f64) -> f64 {
    let mut v = rows
        .iter()
        .map(|r| r.w * (predict(r, x) - r.y).powi(2))
        .sum::<f64>();
    v += ridge * x.iter().map(|x| x * x).sum::<f64>();
    for j in 0..NY {
        for i in 0..NX {
            let k = j * NX + i;
            if i + 1 < NX {
                v += lambda * (x[k] - x[k + 1]).powi(2);
            }
            if j + 1 < NY {
                v += lambda * (x[k] - x[k + NX]).powi(2);
            }
        }
    }
    v
}
fn solve(rows: &[Row], lambda: f64) -> (Vec<f64>, f64, usize, f64, Vec<f64>) {
    solve_loss(rows, lambda, None)
}
fn solve_loss(
    rows: &[Row],
    lambda: f64,
    depletion: Option<f64>,
) -> (Vec<f64>, f64, usize, f64, Vec<f64>) {
    let n = NX * NY;
    let ridge = 0.1;
    let mut rhs = vec![0.; n];
    let mut support = vec![0.; n];
    for r in rows {
        for k in 0..r.index.len() {
            rhs[r.index[k]] += r.w * r.a[k] * r.y;
            support[r.index[k]] += r.w * r.a[k];
        }
    }
    let bound = support.iter().copied().fold(0., f64::max) + 8. * lambda + ridge;
    let mut x = vec![0.; n];
    let mut momentum = x.clone();
    let mut accel: f64 = 1.;
    let mut iterations = 0;
    let mut change = f64::INFINITY;
    // Projected accelerated gradient; parameters enforce nonnegative density ratio.
    let max_iterations = std::env::var("SYISR_CAVITY_ITERATIONS")
        .ok()
        .and_then(|s| s.parse::<usize>().ok())
        .unwrap_or(2000);
    for iter in 0..max_iterations {
        let mut ax = apply(&momentum, rows, lambda, ridge);
        if let Some(delta) = depletion {
            for r in rows {
                let residual = predict(r, &momentum) - r.y;
                let correction = r.w * (residual.max(-delta) - residual);
                for k in 0..r.index.len() {
                    ax[r.index[k]] += r.a[k] * correction;
                }
            }
        }
        let next: Vec<f64> = (0..n)
            .map(|i| {
                (momentum[i] - (ax[i] - rhs[i]) / bound)
                    .clamp(-1., if depletion.is_some() { 0. } else { 3. })
            })
            .collect();
        change = next
            .iter()
            .zip(&x)
            .map(|(a, b)| (a - b).abs())
            .fold(0., f64::max);
        let new_accel = (1. + (1. + 4. * accel * accel).sqrt()) / 2.;
        momentum = next
            .iter()
            .zip(&x)
            .map(|(a, b)| a + (accel - 1.) / new_accel * (a - b))
            .collect();
        x = next;
        accel = new_accel;
        iterations = iter + 1;
        if iter > 40 && change < 1e-5 {
            break;
        }
    }
    let mut obj = objective(&x, rows, lambda, ridge);
    if let Some(delta) = depletion {
        for r in rows {
            let e = predict(r, &x) - r.y;
            if e < -delta {
                obj += r.w * (-2. * delta * e - delta * delta - e * e);
            }
        }
    }
    (x, obj, iterations, change, support)
}
fn run(
    input: &Path,
    output: &Path,
    vx: f64,
    vy: f64,
    lambda: f64,
    fold: i32,
    straight: bool,
    point: bool,
    aspect_cut: f64,
) -> Result<()> {
    let h = File::open(input)?;
    let ds = h.dataset("observations")?;
    let shape = ds.shape();
    let obs = ds.read_raw::<f64>()?;
    let masks = h.dataset("eligible")?.read_raw::<f64>()?;
    let mut train = Vec::new();
    let mut test = Vec::new();
    let mut all = Vec::new();
    for (i, p) in obs.chunks_exact(OBS_COLS).enumerate() {
        if masks[i] == 0. || p[31].abs() < aspect_cut {
            continue;
        }
        let dt = p[0] - TREF;
        let mut index: Vec<usize> = Vec::new();
        let mut a: Vec<f64> = Vec::new();
        let weights = [1. / 6., 2. / 3., 1. / 6.];
        for q in 0..if point { 1 } else { 9 } {
            let offset = if point { 4 } else { 13 + 2 * q };
            let mut xx = p[offset];
            let mut yy = p[offset + 1];
            if straight {
                xx += p[11] - p[4];
                yy += p[12] - p[5];
            }
            if let Some((ii, aa)) = interpolation(xx - vx * dt, yy - vy * dt) {
                let weight = if point {
                    1.
                } else {
                    weights[q / 3] * weights[q % 3]
                };
                for k in 0..4 {
                    if aa[k] * weight < 1e-12 {
                        continue;
                    }
                    if let Some(j) = index.iter().position(|v| *v == ii[k]) {
                        a[j] += aa[k] * weight;
                    } else {
                        index.push(ii[k]);
                        a.push(aa[k] * weight);
                    }
                }
            }
        }
        if (a.iter().sum::<f64>() - 1.).abs() < 1e-8 {
            let row = Row {
                index,
                a,
                y: p[6] - 1.,
                w: 1. / p[7].powi(2),
                beam: p[1] as usize,
                obs: i,
            };
            all.push(row.clone());
            if heldout(row.beam, fold) {
                test.push(row);
            } else {
                train.push(row);
            }
        }
    }
    ensure!(!train.is_empty(), "empty training set");
    let depletion = std::env::var("SYISR_CAVITY_DEPLETION")
        .ok()
        .map(|s| s.parse::<f64>())
        .transpose()?;
    ensure!(
        depletion.map_or(true, |v| v.is_finite() && v > 0.),
        "invalid depletion threshold"
    );
    let (x, obj, it, change, support) = solve_loss(&train, lambda, depletion);
    let eval = if fold < 0 { &train } else { &test };
    let sse = eval
        .iter()
        .map(|r| (predict(r, &x) - r.y).powi(2))
        .sum::<f64>();
    let baseline = eval.iter().map(|r| r.y * r.y).sum::<f64>();
    let f = File::create(output)?;
    if let Some(delta) = depletion {
        vec(&f, "one_sided_huber_threshold", &[delta])?;
    }
    vec(
        &f,
        "parameters",
        &[
            vx,
            vy,
            lambda,
            fold as f64,
            TREF,
            straight as u8 as f64,
            point as u8 as f64,
            aspect_cut,
        ],
    )?;
    vec(
        &f,
        "diagnostics",
        &[
            obj,
            it as f64,
            change,
            sse,
            eval.len() as f64,
            baseline,
            train.len() as f64,
        ],
    )?;
    if fold < 0 || fold >= 100 {
        f.new_dataset::<f64>()
            .shape((NY, NX))
            .create("contrast")?
            .write_raw(&x)?;
        f.new_dataset::<f64>()
            .shape((NY, NX))
            .create("support_weight")?
            .write_raw(&support)?;
        vec(
            &f,
            "x_km",
            &(0..NX).map(|i| X0 + i as f64 * DX).collect::<Vec<_>>(),
        )?;
        vec(
            &f,
            "y_km",
            &(0..NY).map(|i| Y0 + i as f64 * DX).collect::<Vec<_>>(),
        )?;
        let mut predicted = vec![f64::NAN; shape[0]];
        for r in &all {
            predicted[r.obs] = 1. + predict(r, &x);
        }
        vec(&f, "predicted_ratio", &predicted)?;
    }
    println!("FIT vx={vx} vy={vy} lambda={lambda} fold={fold} rows={} RMSE={} baseline={} iterations={it} delta={change}",train.len(),(sse/eval.len()as f64).sqrt(),(baseline/eval.len()as f64).sqrt());
    Ok(())
}
fn inject(input: &Path, output: &Path, vx: f64, vy: f64, lambda: f64) -> Result<()> {
    let h = File::open(input)?;
    let mut obs = h.dataset("observations")?.read_raw::<f64>()?;
    let mask = h.dataset("eligible")?.read_raw::<f64>()?;
    let stats = h.dataset("beam_statistics")?.read_raw::<f64>()?;
    let weights = [1. / 6., 2. / 3., 1. / 6.];
    for p in obs.chunks_exact_mut(OBS_COLS) {
        let dt = p[0] - TREF;
        let mut delta = 0.;
        for q in 0..9 {
            let u = p[13 + 2 * q] - vx * dt;
            let v = p[14 + 2 * q] - vy * dt;
            delta += weights[q / 3]
                * weights[q % 3]
                * (-0.8)
                * (-0.5 * (((u - 40.) / 15.).powi(2) + ((v + 20.) / 35.).powi(2))).exp();
        }
        p[6] = 1. + delta;
        p[9] = p[6] * p[8];
    }
    let path = output.with_extension("input.h5");
    let f = File::create(&path)?;
    mat(&f, "observations", &obs, mask.len(), OBS_COLS)?;
    vec(&f, "eligible", &mask)?;
    mat(&f, "beam_statistics", &stats, stats.len() / 5, 5)?;
    drop(f);
    run(&path, output, vx, vy, lambda, -1, false, false, 2.)?;
    let f = File::open_rw(output)?;
    let truth: Vec<f64> = (0..NY)
        .flat_map(|j| {
            (0..NX).map(move |i| {
                let u = X0 + i as f64 * DX;
                let v = Y0 + j as f64 * DX;
                -0.8 * (-0.5 * (((u - 40.) / 15.).powi(2) + ((v + 20.) / 35.).powi(2))).exp()
            })
        })
        .collect();
    f.new_dataset::<f64>()
        .shape((NY, NX))
        .create("injected_contrast")?
        .write_raw(&truth)?;
    vec(
        &f,
        "injection_parameters_x_y_sigma_x_sigma_y_depth",
        &[40., -20., 15., 35., 0.8],
    )?;
    Ok(())
}
fn main() -> Result<()> {
    let a: Vec<_> = std::env::args().collect();
    ensure!(
        a.len() >= 2,
        "prepare ROOT OBS | fit OBS OUT VX VY LAMBDA FOLD [straight]"
    );
    match a[1].as_str() {
        "prepare" => {
            ensure!(a.len() == 4, "prepare ROOT OBS");
            prepare(Path::new(&a[2]), Path::new(&a[3]))
        }
        "inject" => {
            ensure!(a.len() == 7, "inject OBS OUT VX VY LAMBDA");
            inject(
                Path::new(&a[2]),
                Path::new(&a[3]),
                a[4].parse()?,
                a[5].parse()?,
                a[6].parse()?,
            )
        }
        "fit" => {
            ensure!(a.len() >= 8, "fit OBS OUT VX VY LAMBDA FOLD");
            run(
                Path::new(&a[2]),
                Path::new(&a[3]),
                a[4].parse()?,
                a[5].parse()?,
                a[6].parse()?,
                a[7].parse()?,
                a[8..].iter().any(|s| s == "straight"),
                a[8..].iter().any(|s| s == "point"),
                if a[8..].iter().any(|s| s == "include_perp") {
                    0.
                } else if a[8..].iter().any(|s| s == "strict_perp") {
                    5.
                } else {
                    2.
                },
            )
        }
        _ => anyhow::bail!("unknown command"),
    }
}
#[cfg(test)]
mod tests {
    use super::*;
    #[test]
    fn interpolation_conserves_constant() {
        for (x, y) in [(0., 0.), (12.3, 24.5), (-123.4, -55.5)] {
            let (i, a) = interpolation(x, y).unwrap();
            assert!((a.iter().sum::<f64>() - 1.).abs() < 1e-12);
            let row = Row {
                index: i.to_vec(),
                a: a.to_vec(),
                y: 0.,
                w: 1.,
                beam: 0,
                obs: 0,
            };
            assert!((predict(&row, &vec![2.; NX * NY]) - 2.).abs() < 1e-12);
        }
    }
    #[test]
    fn bounded_solution_predicts_synthetic_cavity() {
        let mut rows = Vec::new();
        for j in 0..11 {
            for i in 0..11 {
                let x = (i as f64 - 5.) * 10.;
                let y = (j as f64 - 5.) * 10.;
                let (index, a) = interpolation(x, y).unwrap();
                let v = -0.7 * (-(x * x + y * y) / 800.).exp();
                rows.push(Row {
                    index: index.to_vec(),
                    a: a.to_vec(),
                    y: v,
                    w: 100.,
                    beam: i % 3,
                    obs: rows.len(),
                });
            }
        }
        let (x, _, _, _, _) = solve(&rows, 0.01);
        let rms = (rows
            .iter()
            .map(|r| (predict(r, &x) - r.y).powi(2))
            .sum::<f64>()
            / rows.len() as f64)
            .sqrt();
        assert!(rms < 0.01, "synthetic RMS {rms}");
        assert!(x.iter().all(|v| *v >= -1. && *v <= 3.));
    }
    #[test]
    fn depletion_loss_resists_positive_contamination() {
        let (ii, aa) = interpolation(0., 0.).unwrap();
        let rows: Vec<_> = (0..20)
            .map(|j| Row {
                index: ii.to_vec(),
                a: aa.to_vec(),
                y: if j < 16 { -0.7 } else { 3. },
                w: 100.,
                beam: 0,
                obs: j,
            })
            .collect();
        let (plain, _, _, _, _) = solve(&rows, 0.01);
        let (robust, _, _, _, _) = solve_loss(&rows, 0.01, Some(0.2));
        assert!((predict(&rows[0], &robust) + 0.7).abs() < 0.08);
        assert!(predict(&rows[0], &plain) > -0.1);
        assert!(robust.iter().all(|v| *v >= -1. && *v <= 0.));
    }
}
