//! Full-profile fit of an existing screened product; never rescreens or modifies it.
use anyhow::{ensure, Result};
use num_complex::Complex64 as C;
use rayon::prelude::*;
use syisr_long_pulse::{
    fit_response,
    profile::{self, Cell, Datum, Problem},
    profile_response::Ambiguity,
    theory::Theory,
};
fn main() -> Result<()> {
    let a = std::env::args().collect::<Vec<_>>();
    ensure!(a.len()>=4&&a.len()<=9,"profile-fit lpm.h5 seed-fits.h5 output.h5 [knot-spacing-or-comma-list=100] [roughness=1] [calibration=auto] [iterations=100] [cell_step_km=25]");
    let mut knots = profile::knot_grid(a.get(4).map(String::as_str).unwrap_or("100"))?;
    let roughness: f64 = a.get(5).map(|s| s.parse()).transpose()?.unwrap_or(1.);
    let fixed_calibration = a
        .get(6)
        .filter(|s| s.as_str() != "auto")
        .map(|s| s.parse::<f64>())
        .transpose()?;
    let bottomside = std::env::var("SYISR_BOTTOMSIDE_CHAPMAN")
        .ok()
        .map(|s| -> Result<(usize, f64)> {
            let v = s
                .split(':')
                .map(str::parse::<f64>)
                .collect::<std::result::Result<Vec<_>, _>>()?;
            ensure!(
                v.len() == 2 && v[1] >= 30. && v[1] <= 150.,
                "bottomside requires peak_range_km:scale_height_km with H=30..150 km"
            );
            let k = knots
                .iter()
                .position(|r| (*r - v[0]).abs() < 1e-8)
                .ok_or_else(|| anyhow::anyhow!("bottomside peak must be an existing knot"))?;
            ensure!(
                k > 0 && k + 2 < knots.len(),
                "bottomside peak must be interior"
            );
            Ok((k, v[1]))
        })
        .transpose()?;
    let iterations: usize = a.get(7).map(|s| s.parse()).transpose()?.unwrap_or(100);
    let cell_step: f64 = a.get(8).map(|s| s.parse()).transpose()?.unwrap_or(25.);
    ensure!(
        cell_step >= 5.
            && cell_step <= 50.
            && (1500. / cell_step - (1500. / cell_step).round()).abs() < 1e-8,
        "quadrature must partition 200..1700 km, step 5..50 km"
    );
    ensure!(
        !std::path::Path::new(&a[3]).exists(),
        "refusing to overwrite a profile product"
    );
    let input = hdf5::File::open(&a[1])?;
    let get = |n: &str| input.dataset(n).unwrap().read_raw::<f64>().unwrap();
    let meta: serde_json::Value =
        serde_json::from_slice(&input.attr("metadata_json")?.read_raw::<u8>()?)?;
    ensure!(
        meta["header"]["radar_frequency_hz"].as_f64() == Some(430.2e6),
        "unsupported frequency"
    );
    let partial_path = std::env::var("SYISR_PARTIAL_PRODUCT").ok();
    let partial = partial_path.as_ref().map(hdf5::File::open).transpose()?;
    if let Some(p) = &partial {
        ensure!(
            p.attr("first_stable_sample")?.read_scalar::<u64>()? == 30
                && p.dataset("native_gate_index")?.read_raw::<u64>()?
                    == (20u64..90).step_by(5).collect::<Vec<_>>(),
            "unsupported partial-product layout"
        );
    }
    let prefix_count = partial
        .as_ref()
        .map(|f| f.dataset("slant_range_km").unwrap().size())
        .unwrap_or(0);
    let merged = |name: &str| -> Vec<f64> {
        let mut v = partial
            .as_ref()
            .map(|f| f.dataset(name).unwrap().read_raw::<f64>().unwrap())
            .unwrap_or_default();
        let old = get(name);
        let skip = if partial.is_some() {
            if name == "slant_range_km" {
                6
            } else {
                6 * 61
            }
        } else {
            0
        };
        v.extend_from_slice(&old[skip..]);
        v
    };
    let ranges = merged("slant_range_km");
    let re = merged("acf_real_adu2");
    let im = merged("acf_imag_adu2");
    let sr = merged("acf_real_sem_adu2");
    let si = merged("acf_imag_sem_adu2");
    let native = get("native_lpm_real_adu2");
    let native_i = get("native_lpm_imag_adu2");
    let noise = (0..61)
        .map(|l| {
            (2500..3400)
                .map(|j| C::new(native[l * 3782 + j], native_i[l * 3782 + j]))
                .sum::<C>()
                / 900.
        })
        .collect::<Vec<_>>();
    ensure!(noise[0].re > 0., "nonpositive receiver noise");
    let rho = (0..61)
        .map(|l| {
            noise[l] / noise[0].re
                * if l == 0 {
                    1.
                } else if l < 21 {
                    0.5 + 0.5 * (std::f64::consts::PI * l as f64 / 21.).cos()
                } else {
                    0.
                }
        })
        .collect::<Vec<_>>();
    let (fir, negative) = fit_response::receiver_fir(&rho);
    ensure!(negative < 1e-3, "inferred filter PSD not positive");
    let requested: Vec<usize> = if partial.is_some() {
        (200..=425)
            .step_by(25)
            .chain((450..=1450).step_by(50))
            .collect()
    } else {
        (450..=1450).step_by(50).collect()
    };
    let targets = requested
        .into_iter()
        .map(|r| {
            ranges
                .iter()
                .enumerate()
                .min_by(|a, b| (a.1 - r as f64).abs().total_cmp(&(b.1 - r as f64).abs()))
                .unwrap()
                .0
        })
        .collect::<Vec<_>>();
    let selected = targets.iter().map(|&i| ranges[i]).collect::<Vec<_>>();
    let lags = (1..61).collect::<Vec<_>>();
    let peak_range = (0..ranges.len())
        .filter(|&i| ranges[i] >= 300. && ranges[i] < 650.)
        .max_by(|&i, &j| {
            (re[i * 61] * ranges[i].powi(2)).total_cmp(&(re[j * 61] * ranges[j].powi(2)))
        })
        .unwrap();
    let peak = re[peak_range * 61] * ranges[peak_range].powi(2);
    ensure!(peak > 0., "nonpositive density reference");
    // Keep the same assumed peak height when comparing knot spacings, rather
    // than silently moving the density calibration anchor with the grid.
    if fixed_calibration.is_none() {
        let anchor_range = 200. + ((ranges[peak_range] - 200.) / 100.).round() * 100.;
        if !knots.iter().any(|z| (z - anchor_range).abs() < 1e-8) {
            knots.push(anchor_range);
            knots.sort_by(f64::total_cmp);
        }
    }
    let mut edges: Vec<f64> = if partial.is_some() {
        ensure!(knots[0] == 80., "partial-pulse domain must start at 80 km");
        vec![80., 100., 125., 150., 175.]
    } else {
        vec![]
    };
    edges.extend((0..=(1500. / cell_step).round() as usize).map(|i| 200. + cell_step * i as f64));
    let ambiguity = Ambiguity::new(&fir)?;
    let terms = if partial.is_some() {
        let mut combined = vec![vec![]; edges.len() - 1];
        for (g, (&r, &i)) in selected.iter().zip(&targets).enumerate() {
            let native = if i < prefix_count {
                20 + i * 5
            } else {
                90 + 5 * (i - prefix_count)
            };
            let p;
            if native < 90 {
                p = Ambiguity::for_gate_from(&fir, native, 30)?;
            } else {
                p = Ambiguity { weights: vec![] };
            }
            let a = if native < 90 { &p } else { &ambiguity };
            let c = a.terms_with_floor(&[r], &edges, &lags, true, true)?;
            for (k, v) in c.into_iter().enumerate() {
                combined[k].extend(v.into_iter().map(|mut t| {
                    t.measurement += g * 60;
                    t
                }));
            }
        }
        combined
    } else {
        ambiguity.terms(&selected, &edges, &lags, true)?
    };
    let time = get("time_utc_s");
    let date = chrono::DateTime::from_timestamp(time[0] as i64, 0).unwrap();
    use chrono::Datelike;
    let epoch = date.year() as f64 + (date.ordinal0() as f64 + 0.5) / 365.25;
    let cells = terms
        .into_par_iter()
        .enumerate()
        .map(|(i, terms)| -> Result<Cell> {
            let r = 0.5 * (edges[i] + edges[i + 1]);
            let b = syisr_long_pulse::geomagnetic::igrf14(epoch, 18.3492, 109.6222, r + 0.05)?;
            let mag = b.iter().map(|v| v * v).sum::<f64>().sqrt();
            let angle = (b[2].abs() / mag).asin().to_degrees();
            let mut theory = Theory::at_field(mag, angle, 500.)?;
            if partial.is_some() {
                theory.molecular_fraction = ((200. - r) / 50.).clamp(0., 1.);
            }
            Ok(Cell {
                lower_guard: partial.is_some() && r < 200.,
                bottomside_density: bottomside
                    .and_then(|(k, h)| profile::bottomside_mapping(&knots, k, h, r)),
                theory,
                interpolation: profile::interpolation(&knots, r)?,
                terms,
            })
        })
        .collect::<Result<Vec<_>>>()?;
    let validation = cells[cells.len() / 2].theory.validate()?;
    let mut data = vec![];
    let mut bg = vec![];
    for &i in &targets {
        for &l in &lags {
            data.push(Datum {
                value: C::new(re[i * 61 + l], im[i * 61 + l]),
                sigma: [sr[i * 61 + l].max(1.), si[i * 61 + l].max(1.)],
            });
            bg.push(rho[l] * 200. / (200 - l) as f64);
        }
    }
    let seeds = hdf5::File::open(&a[2])?;
    let mut seed = vec![];
    for n in seeds.member_names()? {
        if !n.starts_with("range_") {
            continue;
        }
        let m: serde_json::Value =
            serde_json::from_slice(&seeds.group(&n)?.attr("metadata_json")?.read_raw::<u8>()?)?;
        seed.push(m);
    }
    ensure!(!seed.is_empty(), "no seed fits");
    seed.sort_by(|a, b| {
        a["slant_range_km"]
            .as_f64()
            .unwrap()
            .total_cmp(&b["slant_range_km"].as_f64().unwrap())
    });
    let anchor = knots
        .iter()
        .enumerate()
        .min_by(|a, b| {
            (a.1 - (200. + ((ranges[peak_range] - 200.) / 100.).round() * 100.))
                .abs()
                .total_cmp(
                    &(b.1 - (200. + ((ranges[peak_range] - 200.) / 100.).round() * 100.)).abs(),
                )
        })
        .unwrap()
        .0;
    let mut initial = vec![];
    for &z in &knots {
        let m = seed
            .iter()
            .min_by(|a, b| {
                (a["slant_range_km"].as_f64().unwrap() - z)
                    .abs()
                    .total_cmp(&(b["slant_range_km"].as_f64().unwrap() - z).abs())
            })
            .unwrap();
        let f = &m["fit"];
        let i = ranges
            .iter()
            .enumerate()
            .min_by(|a, b| (a.1 - z).abs().total_cmp(&(b.1 - z).abs()))
            .unwrap()
            .0;
        let mut ne = (10. * re[i * 61] * ranges[i].powi(2) / peak).clamp(1e-3, 10.);
        if partial.is_some() && z < 200. {
            ne = if z < 90. { 1e-4 } else { 0.1 };
        }
        let h = if partial.is_some() && z < 200. {
            0.
        } else {
            f["h_fraction"].as_f64().unwrap()
        };
        let he = if partial.is_some() && z < 200. {
            0.
        } else {
            f["he_fraction"].as_f64().unwrap()
        };
        initial.extend([
            ne.ln(),
            (f["ti_k"].as_f64().unwrap() / 1000.).ln(),
            (f["te_k"].as_f64().unwrap() / 1000.).ln(),
            f["spectral_velocity_m_s"].as_f64().unwrap() / 1000.,
            h,
            he / (1. - h).max(1e-10),
        ]);
    }
    let fourier = profile::fourier(&cells[0].theory.f);
    let shrinkage = std::env::var("SYISR_COVARIANCE_SHRINKAGE")
        .ok()
        .map(|v| v.parse::<f64>())
        .transpose()?;
    let covariance_modes = if shrinkage.is_some() {
        let counts = get("block_retained_pulses");
        let nr = ranges.len();
        let nb = counts.len();
        let block_data = |name: &str| {
            let old = get(name);
            if let Some(f) = &partial {
                let extra = f.dataset(name).unwrap().read_raw::<f64>().unwrap();
                let mut v = vec![];
                for b in 0..nb {
                    v.extend_from_slice(&extra[b * prefix_count * 61..(b + 1) * prefix_count * 61]);
                    let original = nr - prefix_count + 6;
                    v.extend_from_slice(&old[(b * original + 6) * 61..(b + 1) * original * 61]);
                }
                v
            } else {
                old
            }
        };
        let br = block_data("block_acf_real_adu2");
        let bi = block_data("block_acf_imag_adu2");
        let blocks = nalgebra::DMatrix::from_fn(2 * data.len(), nb, |row, b| {
            let j = row / 2;
            let i = targets[j / 60];
            let l = lags[j % 60];
            let offset = (b * nr + i) * 61 + l;
            if row % 2 == 0 {
                br[offset]
            } else {
                bi[offset]
            }
        });
        let sigma: Vec<f64> = data.iter().flat_map(|d| d.sigma).collect();
        Some(syisr_long_pulse::measurement_covariance::jackknife_modes(
            &blocks, &counts, &sigma,
        )?)
    } else {
        None
    };
    let whitening = if let Some(u) = &covariance_modes {
        Some(syisr_long_pulse::measurement_covariance::Whitening::new(
            u,
            shrinkage.unwrap(),
        )?)
    } else {
        None
    };
    let problem = Problem {
        bottomside_join: bottomside.map(|(k, _)| k),
        whitening,
        cells,
        data,
        knots: knots.clone(),
        lags_per_gate: 60,
        background: bg,
        roughness,
        density_anchor: fixed_calibration.is_none().then_some(anchor),
        fixed_calibration,
        fourier,
    };
    eprintln!(
        "Joint fit: {} knots, {} quadrature cells, {} complex products; density anchor {:?}",
        knots.len(),
        problem.cells.len(),
        problem.data.len(),
        problem.density_anchor
    );
    let mut fit = problem.solve(&initial, iterations)?;
    if let Some((k, h)) = bottomside {
        for j in 0..k {
            fit.transformed_profile[6 * j] =
                fit.transformed_profile[6 * k] + profile::chapman_log_ratio(knots[j], knots[k], h);
        }
    }
    let out = hdf5::File::create(&a[3])?;
    if partial.is_some() {
        let native: Vec<u64> = selected
            .iter()
            .map(|r| (r / 1.49896229 - 110.6).round() as u64)
            .collect();
        let counts: Vec<u64> = native
            .iter()
            .flat_map(|&i| (1usize..61).map(move |l| i + 21 - (i.saturating_sub(l as u64)).max(30)))
            .collect();
        out.new_dataset_builder()
            .with_data(&native)
            .create("observed_native_gate_index")?;
        out.new_dataset::<u64>()
            .shape((selected.len(), 60))
            .create("observed_pair_count")?
            .write_raw(&counts)?;
    }
    if let Some(u) = &covariance_modes {
        let row: Vec<f64> = (0..u.nrows())
            .flat_map(|i| (0..u.ncols()).map(move |j| u[(i, j)]))
            .collect();
        out.new_dataset::<f64>()
            .shape((u.nrows(), u.ncols()))
            .create("measurement_covariance_normalized_modes")?
            .write_raw(&row)?;
        out.new_attr::<f64>()
            .create("measurement_covariance_shrinkage")?
            .write_scalar(&shrinkage.unwrap())?;
    }
    let write = |name: &str, values: &[f64]| -> Result<()> {
        out.new_dataset_builder().with_data(values).create(name)?;
        Ok(())
    };
    write("slant_range_km", &knots)?;
    write(
        "altitude_km",
        &knots.iter().map(|r| r + 0.05).collect::<Vec<_>>(),
    )?;
    write("quadrature_edges_slant_range_km", &edges)?;
    write("observed_slant_range_km", &selected)?;
    write(
        "lag_us",
        &lags.iter().map(|&l| l as f64 * 10.).collect::<Vec<_>>(),
    )?;
    write("time_utc_s", &[time[0], time[time.len() - 1] + 0.04])?;
    write(
        "receiver_fir_offset_s",
        &fir.iter().map(|v| v.0).collect::<Vec<_>>(),
    )?;
    write(
        "receiver_fir_real",
        &fir.iter().map(|v| v.1.re).collect::<Vec<_>>(),
    )?;
    write(
        "receiver_fir_imag",
        &fir.iter().map(|v| v.1.im).collect::<Vec<_>>(),
    )?;
    write("residual_background_amplitude_adu2", &fit.background_adu2)?;
    for (name, values) in [
        (
            "ne_m3",
            fit.transformed_profile
                .chunks(6)
                .map(|p| 1e11 * p[0].exp())
                .collect::<Vec<_>>(),
        ),
        (
            "ti_k",
            fit.transformed_profile
                .chunks(6)
                .map(|p| 1000. * p[1].exp())
                .collect(),
        ),
        (
            "te_k",
            fit.transformed_profile
                .chunks(6)
                .map(|p| 1000. * p[2].exp())
                .collect(),
        ),
        (
            "spectral_velocity_m_s",
            fit.transformed_profile
                .chunks(6)
                .map(|p| 1000. * p[3])
                .collect(),
        ),
        (
            "h_fraction",
            fit.transformed_profile.chunks(6).map(|p| p[4]).collect(),
        ),
        (
            "he_fraction",
            fit.transformed_profile
                .chunks(6)
                .map(|p| (1. - p[4]) * p[5])
                .collect(),
        ),
        (
            "o_fraction",
            fit.transformed_profile
                .chunks(6)
                .map(|p| (1. - p[4]) * (1. - p[5]))
                .collect(),
        ),
    ] {
        write(name, &values)?;
    }
    for (name, values) in [
        (
            "observed_real_adu2",
            problem.data.iter().map(|p| p.value.re).collect::<Vec<_>>(),
        ),
        (
            "observed_imag_adu2",
            problem.data.iter().map(|p| p.value.im).collect(),
        ),
        (
            "model_real_adu2",
            fit.predicted.iter().map(|p| p.re).collect(),
        ),
        (
            "model_imag_adu2",
            fit.predicted.iter().map(|p| p.im).collect(),
        ),
        (
            "sem_real_adu2",
            problem.data.iter().map(|p| p.sigma[0]).collect(),
        ),
        (
            "sem_imag_adu2",
            problem.data.iter().map(|p| p.sigma[1]).collect(),
        ),
    ] {
        out.new_dataset::<f64>()
            .shape((selected.len(), 60))
            .create(name)?
            .write_raw(&values)?;
    }
    let n = initial.len();
    if let Some(cov) = &fit.conditional_covariance {
        // nalgebra uses column-major storage: transpose on export.
        let row = (0..n)
            .flat_map(|i| (0..n).map(move |j| cov[j * n + i]))
            .collect::<Vec<_>>();
        out.new_dataset::<f64>()
            .shape((n, n))
            .create("conditional_transformed_covariance")?
            .write_raw(&row)?;
    }
    if let Some(res) = &fit.resolution_matrix {
        let row = (0..n)
            .flat_map(|i| (0..n).map(move |j| res[j * n + i]))
            .collect::<Vec<_>>();
        out.new_dataset::<f64>()
            .shape((n, n))
            .create("conditional_resolution_matrix")?
            .write_raw(&row)?;
    }
    let metadata = serde_json::json!({"fit":fit,"theory_validation":validation,"minimum_fitted_lag_us":10,"range_model":"joint profile; exact discrete trapezoidal pairs, rectangular TX, inferred 41-tap voltage FIR, analytic integral of (1000 km/R)^2 in every native subcell","profile_fields":"log(ne/1e11), log(Ti/1000), log(Te/1000), velocity/1000, H, He/(1-H); piecewise linear between knots","density_anchor_slant_range_km":problem.density_anchor.map(|k|knots[k]),"density":"auto calibration fixes one assumed peak knot to 1e12 m^-3 and caps other knots there; fixed calibration permits day-to-day density changes; no ionosonde calibration","regularization":{"weight":roughness,"second_derivative_scales_100km":[0.5,0.25,0.3,0.15,0.15,0.1]},"uncertainty":"conditional penalized curvature and local resolution; cross-range/lag covariance and FIR/background systematics omitted; fixed and active-bound entries are NaN","zero_lag":"excluded from objective; used for initialization and selecting the assumed peak-density anchor","background":"original top-gate subtraction unchanged; one unpenalized residual-noise amplitude profiled at each observation range","provenance":{"lpm":a[1],"seed_fits":a[2],"satellite_mask":"reused unchanged from input","receiver_negative_psd_mass":negative}});
    let mut metadata = metadata;
    if let Some(path) = partial_path {
        metadata["partial_products"] = serde_json::json!({"source":path,"selection":"175..425 km every 50 km plus previous 450..1450 km; only recorded native pairs; actual pair-count normalization","physical_domain":"80..1700 km; zero ionospheric scattering below 80 km","lower_guard_composition":"fixed 30-amu molecular ion below 150 km, mixed with O+ through 150..200 km; H/He suppressed below 200 km; nuisance assumption, not species retrieval","limitations":"lower guard mixture prescribed; no neutral collision correction; fitted profiles shown from 200 km; same saved pulse mask and highest-gate block background"});
    }
    if metadata["partial_products"].is_object() {
        metadata["partial_products"]["selection"] = serde_json::json!(
            "200..425 by25 +450..1450 by50 km; stable samples >=30; actual-pair normalization"
        );
    }
    if bottomside.is_some() {
        metadata["density_interpolation_transition"]=serde_json::json!("C1 cubic-Hermite collar over first upper knot interval; zero log-density slope at peak, matching next linear interval slope");
    }
    metadata["bottomside_density"] = if let Some((k, h)) = bottomside {
        serde_json::json!({"model":"alpha-Chapman below fixed peak; original hat basis above","join_slant_range_km":knots[k],"scale_height_km":h,"formula":"log(ne/Nm)=0.5*(1-u-exp(-u)), u=(R-Rm)/H","status":"assumed bottomside shape; fitted peak amplitude at fixed gain; width and join prescribed for sensitivity test","density_curvature":"density penalty omitted for stencils touching or below join; other fields unchanged","lower_density_coefficients":"derived values, not independently fitted"})
    } else {
        serde_json::Value::Null
    };
    metadata["measurement_covariance"] = if let Some(alpha) = shrinkage {
        serde_json::json!({"mode":"joint real/imaginary range-lag delete-block covariance","diagonal_shrinkage":alpha,"formula":"C = D [alpha I + (1-alpha) U U^T] D; U from weighted delete-one-block means centered across usable blocks; D saved SEM floored at 1 ADU^2","limitations":"finite-rank empirical covariance; shrinkage is a chosen regularization, blocks may contain physical variability; covariance held fixed during fitting; nuisance backgrounds profiled jointly in whitened space"})
    } else {
        serde_json::json!({"mode":"diagonal SEM weights"})
    };
    if shrinkage.is_some() {
        metadata["uncertainty"]=serde_json::json!("Conditional penalized curvature with fixed empirical shrunk range-lag covariance; covariance-estimation, FIR, background and calibration systematics omitted; active-bound entries withheld");
    }
    out.new_attr_builder()
        .with_data(serde_json::to_string(&metadata)?.as_bytes())
        .create("metadata_json")?;
    println!(
        "{}",
        serde_json::json!({"status":metadata["fit"]["status"],"calibration":metadata["fit"]["calibration_adu2"],"data_chi_square":metadata["fit"]["data_chi_square"],"penalty":metadata["fit"]["penalty"],"gradient":metadata["fit"]["normalized_gradient"]})
    );
    Ok(())
}
