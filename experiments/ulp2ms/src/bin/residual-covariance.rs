//! Empirical range-lag noise audit with disjoint block differences and held-out covariance tests.
use anyhow::{ensure, Result};
use nalgebra::DMatrix as M;
use rayon::prelude::*;
use serde_json::json;
use std::{fs, path::Path};
use syisr_long_pulse::measurement_covariance::{jackknife_modes, Whitening};
const ALPHAS: [f64; 7] = [0.05, 0.1, 0.2, 0.35, 0.5, 0.7, 1.];
const NBIN: usize = 13; // 0..300 km in 25-km bins, actual offsets within 6 km.
const NL: usize = 25; // -120..120 us.
struct Audit {
    real_imag: Vec<f64>,
    range: Vec<f64>,
    residual: Vec<f64>,
    kernels: Vec<f64>,
    kernel_count: Vec<u64>,
    summary: serde_json::Value,
    cv: Vec<f64>,
    cv_energy: Vec<f64>,
    variance_ratio: Vec<f64>,
    time_corr: Vec<f64>,
}
fn get(f: &hdf5::File, n: &str) -> Result<Vec<f64>> {
    Ok(f.dataset(n)?.read_raw::<f64>()?)
}
fn rows(m: &M<f64>) -> Vec<Vec<f64>> {
    (0..m.nrows())
        .map(|i| m.row(i).iter().copied().collect())
        .collect()
}
fn dot(a: &[f64], b: &[f64]) -> f64 {
    a.iter().zip(b).map(|(a, b)| a * b).sum()
}
fn median(v: &[f64]) -> f64 {
    let mut v: Vec<_> = v.iter().copied().filter(|x| x.is_finite()).collect();
    if v.is_empty() {
        return f64::NAN;
    }
    v.sort_by(f64::total_cmp);
    if v.len() % 2 == 0 {
        0.5 * (v[v.len() / 2 - 1] + v[v.len() / 2])
    } else {
        v[v.len() / 2]
    }
}
fn factors(z: &M<f64>, columns: &[usize], total: f64) -> M<f64> {
    let n = columns.len();
    M::from_fn(z.nrows(), n, |i, j| {
        let mean = columns.iter().map(|&b| z[(i, b)]).sum::<f64>() / n as f64;
        (z[(i, columns[j])] - mean) / ((n - 1) as f64 * total).sqrt()
    })
}
fn cv(z: &M<f64>, total: f64) -> Result<(Vec<f64>, Vec<f64>)> {
    let n = z.ncols();
    let p = z.nrows();
    let groups = [(0..n / 2).collect::<Vec<_>>(), (n / 2..n).collect()];
    ensure!(
        groups.iter().all(|v| v.len() >= 10),
        "too few independent block pairs"
    );
    let mut scores = vec![0.; ALPHAS.len()];
    let mut energies = scores.clone();
    for fold in 0..2 {
        let mut train = factors(z, &groups[fold], total);
        let mut test = factors(z, &groups[1 - fold], total);
        for i in 0..p {
            let d = train.row(i).norm().max(1e-20);
            train.row_mut(i).scale_mut(1. / d);
            test.row_mut(i).scale_mut(1. / d);
        }
        for (k, &alpha) in ALPHAS.iter().enumerate() {
            let w = Whitening::new(&train, alpha)?;
            let energy = w.apply(&test).norm_squared() / p as f64;
            let logdet = alpha.ln()
                - 2. * w.correction.iter().map(|c| (1. + c).ln()).sum::<f64>() / p as f64;
            scores[k] += 0.5 * (energy + logdet);
            energies[k] += 0.5 * energy;
        }
    }
    Ok((scores, energies))
}
fn audit(oldroot: &Path, root: &Path, t: usize) -> Result<Audit> {
    let f = hdf5::File::open(root.join(format!("windows/{t:03}/profile.h5")))?;
    let old = hdf5::File::open(oldroot.join(format!("windows/{t:03}/lpm.h5")))?;
    let early = hdf5::File::open(root.join(format!("windows/{t:03}/partial.h5")))?;
    let range = get(&f, "observed_slant_range_km")?;
    let native = f.dataset("observed_native_gate_index")?.read_raw::<u64>()?;
    let ng = range.len();
    let nr = old.dataset("slant_range_km")?.size();
    let ne = early.dataset("slant_range_km")?.size();
    let counts = get(&old, "block_retained_pulses")?;
    let nb = counts.len();
    let total: f64 = counts.iter().sum();
    let n = ng * 60 * 2;
    let br = get(&old, "block_acf_real_adu2")?;
    let bi = get(&old, "block_acf_imag_adu2")?;
    let er = get(&early, "block_acf_real_adu2")?;
    let ei = get(&early, "block_acf_imag_adu2")?;
    let bg_r = get(&old, "background_acf_real_adu2")?;
    let bg_i = get(&old, "background_acf_imag_adu2")?;
    let mut sigma = vec![0.; n];
    let mut residual = sigma.clone();
    for (c, part) in ["real", "imag"].iter().enumerate() {
        let obs = get(&f, &format!("observed_{part}_adu2"))?;
        let model = get(&f, &format!("model_{part}_adu2"))?;
        let sem = get(&f, &format!("sem_{part}_adu2"))?;
        for i in 0..obs.len() {
            sigma[2 * i + c] = sem[i];
            residual[2 * i + c] = (obs[i] - model[i]) / sem[i];
        }
    }
    let blocks = M::from_fn(n, nb, |row, b| {
        let g = row / 120;
        let l = (row / 2) % 60 + 1;
        let i = native[g] as usize;
        if i < 90 {
            let j = (b * ne + (i - 20) / 5) * 61 + l;
            if row % 2 == 0 {
                er[j]
            } else {
                ei[j]
            }
        } else {
            let j = (b * nr + (i - 60) / 5) * 61 + l;
            if row % 2 == 0 {
                br[j]
            } else {
                bi[j]
            }
        }
    });
    let before = M::from_fn(n, nb, |row, b| {
        let l = (row / 2) % 60 + 1;
        blocks[(row, b)]
            + if row % 2 == 0 {
                bg_r[b * 61 + l]
            } else {
                bg_i[b * 61 + l]
            } * 200.
                / (200 - l) as f64
    });
    let up = jackknife_modes(&blocks, &counts, &sigma)?;
    let ub = jackknife_modes(&before, &counts, &sigma)?;
    let pairs: Vec<usize> = (0..nb - 1)
        .step_by(2)
        .filter(|&b| counts[b] > 0. && counts[b + 1] > 0.)
        .collect();
    let z = M::from_fn(n, pairs.len(), |i, j| {
        let b = pairs[j];
        (blocks[(i, b + 1)] - blocks[(i, b)]) / (1. / counts[b] + 1. / counts[b + 1]).sqrt()
    });
    let all: Vec<usize> = (0..pairs.len()).collect();
    let raw_diff = factors(&z, &all, total);
    let mut ud = raw_diff.clone();
    for i in 0..n {
        ud.row_mut(i).scale_mut(1. / sigma[i]);
    }
    let variance_ratio = (0..n).map(|i| ud.row(i).norm_squared()).collect::<Vec<_>>();
    let (cv, cv_energy) = cv(&z, total)?;
    let best = (0..ALPHAS.len())
        .min_by(|&a, &b| cv[a].total_cmp(&cv[b]))
        .unwrap();
    let modes = [rows(&up), rows(&ub), rows(&ud)];
    let norms: Vec<Vec<f64>> = modes
        .iter()
        .map(|u| u.iter().map(|r| dot(r, r).sqrt()).collect())
        .collect();
    // Six panels: real/imag for full-block post-subtraction, pre-subtraction, differenced post.
    let mut kernels = vec![0.; 6 * NBIN * NL];
    let mut num = vec![0u64; NBIN * NL];
    for g in 0..ng {
        for h in g..ng {
            let dr = range[h] - range[g];
            let b = (dr / 25.).round() as usize;
            if b >= NBIN || (dr - b as f64 * 25.).abs() > 6. {
                continue;
            }
            for dl in 0..NL {
                for l in 0..60 {
                    let shifted = l as isize + dl as isize - 12;
                    if !(0..60).contains(&shifted) {
                        continue;
                    }
                    let j = b * NL + dl;
                    num[j] += 1;
                    for c in 0..2 {
                        let i = 2 * (g * 60 + l) + c;
                        let k = 2 * (h * 60 + shifted as usize) + c;
                        for m in 0..3 {
                            kernels[(2 * m + c) * NBIN * NL + j] += dot(&modes[m][i], &modes[m][k])
                                / (norms[m][i] * norms[m][k]).max(1e-100);
                        }
                    }
                }
            }
        }
    }
    for p in 0..6 {
        for j in 0..NBIN * NL {
            kernels[p * NBIN * NL + j] = if num[j] > 0 {
                kernels[p * NBIN * NL + j] / num[j] as f64
            } else {
                f64::NAN
            };
        }
    }
    // Check adjacent correlation using independent halves of the 20-second differences.
    let groups = [
        (0..pairs.len() / 2).collect::<Vec<_>>(),
        (pairs.len() / 2..pairs.len()).collect(),
    ];
    let split: Vec<_> = groups
        .iter()
        .map(|g| rows(&factors(&z, g, total)))
        .collect();
    let mut heldout_adj = vec![];
    for u in &split {
        let mut corr = vec![];
        for g in 0..ng {
            for l in 0..59 {
                for c in 0..2 {
                    let i = 2 * (g * 60 + l) + c;
                    corr.push(
                        dot(&u[i], &u[i + 2])
                            / (dot(&u[i], &u[i]) * dot(&u[i + 2], &u[i + 2]))
                                .sqrt()
                                .max(1e-100),
                    );
                }
            }
        }
        heldout_adj.push(median(&corr));
    }
    let mut time_corr = vec![];
    for offset in [1usize, 2, 6] {
        let mut values = vec![];
        for i in 0..n {
            let good: Vec<usize> = (0..nb).filter(|&b| counts[b] > 0.).collect();
            let sw: f64 = good.iter().map(|&b| counts[b]).sum();
            let mt = good.iter().map(|&b| counts[b] * b as f64).sum::<f64>() / sw;
            let my = good
                .iter()
                .map(|&b| counts[b] * blocks[(i, b)])
                .sum::<f64>()
                / sw;
            let slope = good
                .iter()
                .map(|&b| counts[b] * (b as f64 - mt) * (blocks[(i, b)] - my))
                .sum::<f64>()
                / good
                    .iter()
                    .map(|&b| counts[b] * (b as f64 - mt).powi(2))
                    .sum::<f64>();
            let d: Vec<f64> = (0..nb)
                .map(|b| (blocks[(i, b)] - my - slope * (b as f64 - mt)) * counts[b].sqrt())
                .collect();
            let mut xy = 0.;
            let mut xx = 0.;
            let mut yy = 0.;
            for b in 0..nb - offset {
                if counts[b] > 0. && counts[b + offset] > 0. {
                    xy += d[b] * d[b + offset];
                    xx += d[b] * d[b];
                    yy += d[b + offset] * d[b + offset];
                }
            }
            values.push(xy / (xx * yy).sqrt().max(1e-100));
        }
        time_corr.push(median(&values));
    }
    let real: Vec<f64> = residual.iter().step_by(2).copied().collect();
    let imag: Vec<f64> = residual.iter().skip(1).step_by(2).copied().collect();
    let chi = residual.iter().map(|v| v * v).sum::<f64>();
    let summary = json!({"window":t,"real_residual_rms_SEM":(dot(&real,&real)/real.len() as f64).sqrt(),"imag_residual_rms_SEM":(dot(&imag,&imag)/imag.len() as f64).sqrt(),"data_chi_square":chi,"real_measurements":n,"fraction_abs_residual_gt3":residual.iter().filter(|v|v.abs()>3.).count() as f64/n as f64,"difference_to_saved_variance_median":median(&variance_ratio),"heldout_difference_adjacent_lag_correlation":heldout_adj,"cv_selected_shrinkage":ALPHAS[best],"cv_heldout_energy_per_dimension":cv_energy[best],"cv_diagonal_energy_per_dimension":cv_energy[6],"cv_score_improvement_over_diagonal":cv[6]-cv[best],"usable_blocks":counts.iter().filter(|&&n|n>0.).count(),"disjoint_difference_pairs":pairs.len(),"detrended_block_temporal_correlation_10_20_60_seconds":time_corr});
    let mut summary = summary;
    for (m, label) in [(0, "full_block"), (2, "differenced")] {
        for (c, component) in ["real", "imag"].iter().enumerate() {
            let at = |b: usize, dl: usize| kernels[(2 * m + c) * NBIN * NL + b * NL + dl + 12];
            summary[format!("{label}_lag10us_{component}")] = json!(at(0, 1));
            for (b, d) in [(1, 25), (2, 50), (4, 100), (12, 300)] {
                summary[format!("{label}_range{d}km_{component}")] = json!(at(b, 0));
            }
        }
    }
    let mut real_imag = vec![];
    for m in [0usize, 2] {
        let v: Vec<f64> = (0..n)
            .step_by(2)
            .map(|i| {
                dot(&modes[m][i], &modes[m][i + 1]) / (norms[m][i] * norms[m][i + 1]).max(1e-100)
            })
            .collect();
        summary[if m == 0 {
            "full_block_real_imag_abs_median"
        } else {
            "differenced_real_imag_abs_median"
        }] = json!(median(&v.iter().map(|v| v.abs()).collect::<Vec<_>>()));
        real_imag.extend(v);
    }
    let mut kurt = vec![];
    let mut diff_time = vec![];
    let nd = pairs.len() as f64;
    for row in &modes[2] {
        let s2 = dot(row, row);
        let g2 = nd * row.iter().map(|v| v.powi(4)).sum::<f64>() / s2.powi(2) - 3.;
        kurt.push((nd - 1.) / ((nd - 2.) * (nd - 3.)) * ((nd + 1.) * g2 + 6.));
        let mut xy = 0.;
        let mut xx = 0.;
        let mut yy = 0.;
        for j in 0..row.len() - 1 {
            if pairs[j + 1] == pairs[j] + 2 {
                xy += row[j] * row[j + 1];
                xx += row[j].powi(2);
                yy += row[j + 1].powi(2);
            }
        }
        diff_time.push(xy / (xx * yy).sqrt().max(1e-100));
    }
    summary["difference_excess_kurtosis_median"] = json!(median(&kurt));
    summary["difference_temporal_correlation_20s_median"] = json!(median(&diff_time));
    for (label, lo, hi) in [
        ("lower", 0., 450.),
        ("middle", 450., 900.),
        ("upper", 900., 2000.),
    ] {
        for (c, part) in ["real", "imag"].iter().enumerate() {
            let v: Vec<f64> = (0..ng)
                .filter(|&g| range[g] >= lo && range[g] < hi)
                .flat_map(|g| (0..60).map(move |l| 2 * (g * 60 + l) + c))
                .map(|j| residual[j])
                .collect();
            summary[format!("{label}_residual_rms_{part}_SEM")] =
                json!((dot(&v, &v) / v.len() as f64).sqrt());
        }
    }
    Ok(Audit {
        real_imag,
        range,
        residual,
        kernels,
        kernel_count: num,
        summary,
        cv,
        cv_energy,
        variance_ratio,
        time_corr,
    })
}
fn main() -> Result<()> {
    let a: Vec<String> = std::env::args().collect();
    ensure!(
        a.len() == 4,
        "residual-covariance old-root fit-root output-root"
    );
    let out = Path::new(&a[3]);
    fs::create_dir_all(out)?;
    let results: Vec<Result<Audit>> = (0..48usize)
        .into_par_iter()
        .map(|t| {
            let r = audit(Path::new(&a[1]), Path::new(&a[2]), t);
            eprintln!(
                "covariance audit window {t}: {}",
                if r.is_ok() { "complete" } else { "failed" }
            );
            r
        })
        .collect();
    let all = results.into_iter().collect::<Result<Vec<_>>>()?;
    let ng = all[0].range.len();
    let f = hdf5::File::create(out.join("audit.h5"))?;
    let write = |name: &str, shape: &[usize], v: &[f64]| -> Result<()> {
        f.new_dataset::<f64>()
            .shape(shape)
            .create(name)?
            .write_raw(v)?;
        Ok(())
    };
    let residual: Vec<f64> = all.iter().flat_map(|v| v.residual.clone()).collect();
    write("standardized_residual", &[48, ng, 60, 2], &residual)?;
    let mut mean = vec![0.; ng * 120];
    let mut rms = mean.clone();
    for v in &all {
        for j in 0..mean.len() {
            mean[j] += v.residual[j] / 48.;
            rms[j] += v.residual[j].powi(2) / 48.;
        }
    }
    rms.iter_mut().for_each(|v| *v = v.sqrt());
    write("mean_standardized_residual", &[ng, 60, 2], &mean)?;
    write("rms_standardized_residual", &[ng, 60, 2], &rms)?;
    let ks: Vec<f64> = all.iter().flat_map(|v| v.kernels.clone()).collect();
    write(
        "same_product_real_imag_correlation",
        &[48, 2, ng, 60],
        &all.iter()
            .flat_map(|v| v.real_imag.clone())
            .collect::<Vec<_>>(),
    )?;
    write("correlation_kernel", &[48, 3, 2, NBIN, NL], &ks)?;
    write(
        "cv_score",
        &[48, ALPHAS.len()],
        &all.iter().flat_map(|v| v.cv.clone()).collect::<Vec<_>>(),
    )?;
    write(
        "cv_energy_per_dimension",
        &[48, ALPHAS.len()],
        &all.iter()
            .flat_map(|v| v.cv_energy.clone())
            .collect::<Vec<_>>(),
    )?;
    write(
        "difference_to_saved_variance_ratio",
        &[48, ng, 60, 2],
        &all.iter()
            .flat_map(|v| v.variance_ratio.clone())
            .collect::<Vec<_>>(),
    )?;
    write(
        "block_temporal_correlation",
        &[48, 3],
        &all.iter()
            .flat_map(|v| v.time_corr.clone())
            .collect::<Vec<_>>(),
    )?;
    f.new_dataset_builder()
        .with_data(&all[0].range)
        .create("nominal_gate_slant_range_km")?;
    f.new_dataset_builder()
        .with_data(&ALPHAS)
        .create("shrinkage_grid")?;
    f.new_dataset_builder()
        .with_data(&all[0].kernel_count)
        .create("correlation_pair_count")?;
    f.new_attr_builder().with_data("Correlation types: full delete-block post background, before background, and disjoint adjacent 10-second-block differences post background; components real,imag. Delta-range bins 25 km within 6 km; delta-lag -120..120 us in 10 us steps. Residual=(observed-model)/saved SEM; fitting modifies its covariance. CV uses non-overlapping temporal halves of disjoint block-pair differences, centered separately; diagonal target estimated without fit residuals. Temporal independence and stationarity are tested, not guaranteed.").create("method")?;
    let mut summary = json!({"windows":all.iter().map(|v|v.summary.clone()).collect::<Vec<_>>(),"limitations":"20-second differences suppress slow physical variation but do not guarantee pure thermal noise or temporal independence. Fitted residual covariance is altered by fitting and regularization. Covariance rank is limited by block count; shrinkage selection is independent of fit residuals. No raw-recording reprocessing and no change to plasma fits."});
    summary["limitations"]=json!("Differences of adjacent 10-second blocks, in disjoint 20-second pairs, suppress slow variation without guaranteeing pure thermal noise or independence. Fitting, bounds and regularization alter residual covariance. Covariance rank is limited by block count; shrinkage is selected without fit residuals. Audit does not change campaign fits; separate diagnostic refits are identified explicitly.");
    let per = summary["windows"].as_array().unwrap();
    let metric = |key: &str| {
        median(
            &per.iter()
                .map(|v| v[key].as_f64().unwrap())
                .collect::<Vec<_>>(),
        )
    };
    summary["campaign_medians"] = json!({"real_residual_rms_SEM":metric("real_residual_rms_SEM"),"imag_residual_rms_SEM":metric("imag_residual_rms_SEM"),"fraction_abs_residual_gt3":metric("fraction_abs_residual_gt3"),"difference_to_saved_variance_ratio":metric("difference_to_saved_variance_median"),"selected_shrinkage":metric("cv_selected_shrinkage"),"cv_score_improvement_over_diagonal":metric("cv_score_improvement_over_diagonal"),"cv_heldout_energy_per_dimension":metric("cv_heldout_energy_per_dimension"),"cv_diagonal_energy_per_dimension":metric("cv_diagonal_energy_per_dimension")});
    fs::write(
        out.join("summary.json"),
        serde_json::to_vec_pretty(&summary)?,
    )?;
    println!(
        "{}",
        serde_json::to_string_pretty(&summary["campaign_medians"])?
    );
    Ok(())
}
#[cfg(test)]
mod tests {
    use super::*;
    struct Rng(u64);
    impl Rng {
        fn uniform(&mut self) -> f64 {
            self.0 ^= self.0 << 13;
            self.0 ^= self.0 >> 7;
            self.0 ^= self.0 << 17;
            ((self.0 >> 11) as f64 + 0.5) / (1u64 << 53) as f64
        }
        fn normal(&mut self) -> f64 {
            (-2. * self.uniform().ln()).sqrt() * (std::f64::consts::TAU * self.uniform()).cos()
        }
    }
    fn synthetic(rho: f64) -> (M<f64>, f64) {
        let mut rng = Rng(9172315);
        let counts: Vec<f64> = (0..240).map(|b| 100. + ((b * 37) % 150) as f64).collect();
        let mut y = M::zeros(64, 240);
        for k in 0..32 {
            for b in 0..240 {
                let u = rng.normal();
                let v = rng.normal();
                y[(2 * k, b)] = u / counts[b].sqrt();
                y[(2 * k + 1, b)] = (rho * u + (1. - rho * rho).sqrt() * v) / counts[b].sqrt();
            }
        }
        let z = M::from_fn(64, 120, |i, j| {
            (y[(i, 2 * j + 1)] - y[(i, 2 * j)])
                / (1. / counts[2 * j] + 1. / counts[2 * j + 1]).sqrt()
        });
        (z, counts.iter().sum())
    }
    #[test]
    fn independent_noise_is_not_artificially_correlated() {
        let (z, total) = synthetic(0.);
        let u = factors(&z, &(0..120).collect::<Vec<_>>(), total);
        let variance: Vec<f64> = (0..64).map(|i| u.row(i).norm_squared() * total).collect();
        assert!((median(&variance) - 1.).abs() < 0.1);
        let u = rows(&u);
        let c: Vec<f64> = (0..32)
            .map(|i| {
                dot(&u[2 * i], &u[2 * i + 1])
                    / (dot(&u[2 * i], &u[2 * i]) * dot(&u[2 * i + 1], &u[2 * i + 1])).sqrt()
            })
            .collect();
        assert!(c.iter().sum::<f64>().abs() / 32. < 0.04);
        let (s, e) = cv(&z, total).unwrap();
        assert!((e[6] - 1.).abs() < 0.2);
        assert!(s[6] - s.iter().copied().fold(f64::INFINITY, f64::min) < 0.02);
    }
    #[test]
    fn known_correlations_survive_differencing_and_predict_heldout_noise() {
        let (z, total) = synthetic(0.7);
        let u = rows(&factors(&z, &(0..120).collect::<Vec<_>>(), total));
        let c: Vec<f64> = (0..32)
            .map(|i| {
                dot(&u[2 * i], &u[2 * i + 1])
                    / (dot(&u[2 * i], &u[2 * i]) * dot(&u[2 * i + 1], &u[2 * i + 1])).sqrt()
            })
            .collect();
        assert!((median(&c) - 0.7).abs() < 0.04);
        let (s, _) = cv(&z, total).unwrap();
        assert!(s[6] - s.iter().copied().fold(f64::INFINITY, f64::min) > 0.005);
    }
}
