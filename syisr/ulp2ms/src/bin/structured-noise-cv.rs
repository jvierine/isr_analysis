//! Held-out prediction with a Gaussian-overlap + common-reference covariance target.
use anyhow::{ensure, Result};
use nalgebra::DMatrix as M;
use rayon::prelude::*;
use serde_json::json;
use std::{fs, path::Path, time::Instant};
use syisr_long_pulse::{
    measurement_covariance::{DenseWhitening, Whitening},
    noise_blocks::{difference_factors, NoiseBlocks},
    structured_covariance::StructuredModel,
};
const ALPHA: [f64; 7] = [0.05, 0.1, 0.2, 0.35, 0.5, 0.7, 1.];
struct Check {
    summary: serde_json::Value,
    kernels: Vec<f64>,
}
fn covariance_kernel(native: &[usize], r: &M<f64>) -> Vec<f64> {
    let mut sum = vec![0.; 2 * 13 * 25];
    let mut count = vec![0usize; 13 * 25];
    for g in 0..native.len() {
        for h in g..native.len() {
            let dr = (native[h] - native[g]) as f64 * 1.49896229;
            let b = (dr / 25.).round() as usize;
            if b >= 13 || (dr - b as f64 * 25.).abs() > 6. {
                continue;
            }
            for j in 0..25 {
                let dl = j as isize - 12;
                for l in 0..60 {
                    let m = l as isize + dl;
                    if !(0..60).contains(&m) {
                        continue;
                    }
                    count[b * 25 + j] += 1;
                    for c in 0..2 {
                        let i = 2 * (g * 60 + l) + c;
                        let k = 2 * (h * 60 + m as usize) + c;
                        sum[c * 13 * 25 + b * 25 + j] += r[(i, k)] / (r[(i, i)] * r[(k, k)]).sqrt();
                    }
                }
            }
        }
    }
    for c in 0..2 {
        for j in 0..13 * 25 {
            sum[c * 13 * 25 + j] /= count[j] as f64;
        }
    }
    sum
}
fn check(root: &Path, t: usize, out: &Path) -> Result<Check> {
    let start = Instant::now();
    let b = NoiseBlocks::load(root, t)?;
    let n = b.pairs.len();
    let p = b.sigma.len();
    let mut groups = [
        (0..n / 2).collect::<Vec<_>>(),
        (n / 2..n).collect::<Vec<_>>(),
    ];
    let interleaved = std::env::var("SYISR_COVARIANCE_INTERLEAVED").is_ok();
    if interleaved {
        groups = [(0..n).step_by(2).collect(), (1..n).step_by(2).collect()];
    }
    let mut scores = vec![0.; 7];
    let mut energies = scores.clone();
    let mut diag = 0.;
    let mut pure = 0.;
    let mut kernels = vec![0.; 3 * 2 * 13 * 25];
    let mut details = vec![];
    for fold in 0..2 {
        let tick = Instant::now();
        let train = difference_factors(&b.differences, &groups[fold], b.total);
        let test = difference_factors(&b.differences, &groups[1 - fold], b.total);
        let variance: Vec<f64> = (0..p).map(|i| train.row(i).norm_squared()).collect();
        let sigma: Vec<f64> = variance.iter().map(|v| v.sqrt()).collect();
        let (gamma, power, neg) = b.voltage_noise(&groups[fold])?;
        let local_acf = b.local_voltage_acfs(&groups[fold]);
        let mut model = StructuredModel::with_self_noise(
            gamma.clone(),
            &b.native,
            &local_acf,
            power,
            &variance,
            &sigma,
            power.powi(2) / b.total,
        )?;
        let nugget: f64 = std::env::var("SYISR_COVARIANCE_NUGGET")
            .unwrap_or_else(|_| "0".into())
            .parse()?;
        anyhow::ensure!((0.0..=1.0).contains(&nugget), "invalid covariance nugget");
        model.normalized *= 1.0 - nugget;
        for i in 0..p {
            model.normalized[(i, i)] += nugget;
        }
        let build_seconds = tick.elapsed().as_secs_f64();
        let mut ut = train;
        let mut vt = test;
        for i in 0..p {
            ut.row_mut(i).scale_mut(1. / sigma[i]);
            vt.row_mut(i).scale_mut(1. / sigma[i]);
        }
        let dscore = vt.norm_squared() / p as f64;
        diag += dscore * 0.5;
        let tick = Instant::now();
        let dense = DenseWhitening::new(model.normalized.clone())?;
        let factor_seconds = tick.elapsed().as_secs_f64();
        let train_white = dense.apply(&ut);
        let test_white = dense.apply(&vt);
        pure += 0.5 * test_white.norm_squared() / p as f64;
        let mut local_scores = vec![];
        let mut local_energies = vec![];
        for (k, &alpha) in ALPHA.iter().enumerate() {
            let w = Whitening::new(&train_white, alpha)?;
            let energy = w.apply(&test_white).norm_squared() / p as f64;
            let logdet = dense.log_determinant / p as f64 + alpha.ln()
                - 2. * w.correction.iter().map(|c| (1. + c).ln()).sum::<f64>() / p as f64;
            scores[k] += 0.5 * (logdet + energy);
            energies[k] += 0.5 * energy;
            local_scores.push(logdet + energy);
            local_energies.push(energy);
        }
        let best = (0..7)
            .min_by(|&a, &c| local_scores[a].total_cmp(&local_scores[c]))
            .unwrap();
        let empirical = &ut * ut.transpose();
        let observed = &vt * vt.transpose();
        let predicted = &model.normalized * ALPHA[best] + &empirical * (1. - ALPHA[best]);
        for (m, mat) in [&observed, &model.normalized, &predicted]
            .iter()
            .enumerate()
        {
            let k = covariance_kernel(&b.native, mat);
            for j in 0..k.len() {
                kernels[m * k.len() + j] += 0.5 * k[j];
            }
        }
        details.push(json!({"fold":fold,"diagonal_score":dscore,"score":local_scores,"heldout_energy":local_energies,"clamped_local_variances":model.clamped,"receiver_PSD_negative_mass":neg,"target_logdet_per_coordinate":dense.log_determinant/p as f64,"target_build_seconds":build_seconds,"target_factor_seconds":factor_seconds,"gamma_first_lags":gamma[..6].iter().map(|g|[g.re,g.im]).collect::<Vec<_>>()}));
        eprintln!("window {t} fold {fold}: target {build_seconds:.2}s, factor {factor_seconds:.2}s, diagonal {dscore:.3}, structured {:.3}",local_energies[6]);
    }
    let best = (0..7)
        .min_by(|&a, &c| scores[a].total_cmp(&scores[c]))
        .unwrap();
    let summary = json!({"window":t,"alpha":ALPHA,"score":scores,"heldout_energy":energies,"diagonal_score":diag,"selected_alpha":ALPHA[best],"selected_score_gain_vs_diagonal":diag-scores[best],"selected_heldout_energy":energies[best],"pure_structured_heldout_energy":pure,"folds":details,"seconds":start.elapsed().as_secs_f64()});
    let mut summary = summary;
    let mut ksample = vec![];
    for m in 0..3 {
        for c in 0..2 {
            let a = |b: usize, j: usize| kernels[(m * 2 + c) * 13 * 25 + b * 25 + j];
            ksample.push(json!({"mode":m,"component":c,"lag10us":a(0,13),"range50km":a(2,12),"range300km":a(12,12)}));
        }
    }
    summary["kernel_samples"] = json!(ksample);
    summary["diagonal_nugget"] = json!(std::env::var("SYISR_COVARIANCE_NUGGET")
        .unwrap_or_else(|_| "0".into())
        .parse::<f64>()?);
    summary["fold_split"] = json!(if interleaved {
        "alternating disjoint pairs"
    } else {
        "temporal halves"
    });
    fs::write(
        out.join(format!("window-{t:03}.json")),
        serde_json::to_vec_pretty(&summary)?,
    )?;
    Ok(Check { summary, kernels })
}
fn main() -> Result<()> {
    let a: Vec<String> = std::env::args().collect();
    ensure!(
        a.len() == 3 || a.len() == 4,
        "structured-noise-cv fit-root output-root [window-list]"
    );
    let root = Path::new(&a[1]);
    let out = Path::new(&a[2]);
    fs::create_dir_all(out)?;
    let windows: Vec<usize> = a
        .get(3)
        .map(String::as_str)
        .unwrap_or("0,14,24,40")
        .split(',')
        .map(str::parse)
        .collect::<std::result::Result<_, _>>()?;
    let checks: Vec<Result<Check>> = windows.par_iter().map(|&t| check(root, t, out)).collect();
    let checks = checks.into_iter().collect::<Result<Vec<_>>>()?;
    let h = hdf5::File::create(out.join("validation.h5"))?;
    h.new_dataset::<f64>()
        .shape((checks.len(), 3, 2, 13, 25))
        .create("correlation_kernels")?
        .write_raw(
            &checks
                .iter()
                .flat_map(|c| c.kernels.clone())
                .collect::<Vec<_>>(),
        )?;
    h.new_dataset_builder()
        .with_data(&windows.iter().map(|&t| t as u64).collect::<Vec<_>>())
        .create("window_index")?;
    h.new_attr_builder().with_data("Held-out, Gaussian structured target, structured-target/empirical mixture; real/imag; delta range 0..300 by25km, delta lag -120..120 by10us. Each fold uses reference filter, local voltage ACFs and marginal variance estimated only from its training temporal half of disjoint block pairs. Positive spectral projection and smooth voltage amplitudes model plasma self-noise. Common reference covariance and exact partial-product sums are included. Hyperparameter selection uses the validation scores; these are development, not nested test-set results.").create("method")?;
    let summary = json!({"windows":checks.iter().map(|c|c.summary.clone()).collect::<Vec<_>>(),"scope":"Two-fold covariance prediction; target from Gaussian voltage moments with exact partial weights and common reference; empirical correction selected without fit residuals. No change to plasma campaign."});
    fs::write(
        out.join("summary.json"),
        serde_json::to_vec_pretty(&summary)?,
    )?;
    println!("{}", serde_json::to_string_pretty(&summary)?);
    Ok(())
}
