//! Independently replay saved scientific-product invariants (no optimizer calls).
use anyhow::{ensure, Result};
fn main() -> Result<()> {
    let a = std::env::args().collect::<Vec<_>>();
    ensure!(
        a.len() == 2 || a.len() == 3,
        "profile-validate campaign-root [comma-separated-window-indices]"
    );
    let root = std::path::Path::new(&a[1]);
    let mut count = 0;
    let mut gain: Option<f64> = None;
    let mut max_relative_cost = 0f64;
    let mut nodes = 0;
    let windows = if a.len() == 3 {
        a[2].split(',')
            .map(str::parse::<usize>)
            .collect::<std::result::Result<Vec<_>, _>>()?
    } else {
        (0..48).collect()
    };
    for t in windows {
        let f = hdf5::File::open(root.join(format!("windows/{t:03}/profile.h5")))?;
        let get = |n: &str| f.dataset(n).unwrap().read_raw::<f64>().unwrap();
        let m: serde_json::Value =
            serde_json::from_slice(&f.attr("metadata_json")?.read_raw::<u8>()?)?;
        ensure!(
            m["fit"]["status"] == "converged",
            "window {t} did not converge"
        );
        let g = m["fit"]["calibration_adu2"].as_f64().unwrap();
        if let Some(prev) = gain {
            ensure!((g / prev - 1f64).abs() < 1e-12, "inconsistent calibration");
        }
        gain = Some(g);
        ensure!(
            get("lag_us") == (1..61).map(|i| i as f64 * 10.).collect::<Vec<_>>(),
            "window {t} lag selection"
        );
        let z = get("slant_range_km");
        let altitude = get("altitude_km");
        ensure!(
            z.iter()
                .zip(altitude)
                .all(|(r, h)| (h - r - 0.05).abs() < 1e-10),
            "altitude geometry"
        );
        let ne = get("ne_m3");
        let h = get("h_fraction");
        let he = get("he_fraction");
        let o = get("o_fraction");
        let ti = get("ti_k");
        let te = get("te_k");
        ensure!(
            (0..z.len()).all(|i| ne[i] > 0.
                && ne[i].is_finite()
                && h[i] >= 0.
                && he[i] >= 0.
                && o[i] >= 0.
                && (h[i] + he[i] + o[i] - 1.).abs() < 1e-12
                && ti[i] >= 300. - 1e-6
                && ti[i] <= 5000. + 1e-6
                && te[i] >= 300. - 1e-6
                && te[i] <= 12000. + 1e-6),
            "window {t} physical states"
        );
        let nm = get("observed_real_adu2").len();
        let mut normalized_residual = vec![0.; 2 * nm];
        for component in ["real", "imag"] {
            let obs = get(&format!("observed_{component}_adu2"));
            let model = get(&format!("model_{component}_adu2"));
            let sem = get(&format!("sem_{component}_adu2"));
            ensure!(
                obs.len() == nm
                    && model.len() == nm
                    && sem.len() == nm
                    && nm == get("observed_slant_range_km").len() * 60,
                "ACF dimensions"
            );
            for i in 0..obs.len() {
                ensure!(
                    obs[i].is_finite() && model[i].is_finite() && sem[i] > 0.,
                    "invalid ACF"
                );
                normalized_residual[2 * i + usize::from(component == "imag")] =
                    (model[i] - obs[i]) / sem[i];
            }
        }
        let mut whitened = nalgebra::DMatrix::from_column_slice(2 * nm, 1, &normalized_residual);
        if let Some(path) = m["measurement_covariance"]["product"].as_str() {
            let w = syisr_long_pulse::measurement_covariance::Whitening::from_product(
                std::path::Path::new(path),
            )?;
            whitened = w.apply(&whitened);
        } else if let Ok(d) = f.dataset("measurement_covariance_normalized_modes") {
            let shape = d.shape();
            let u = nalgebra::DMatrix::from_row_slice(shape[0], shape[1], &d.read_raw::<f64>()?);
            let alpha = f
                .attr("measurement_covariance_shrinkage")?
                .read_scalar::<f64>()?;
            whitened = syisr_long_pulse::measurement_covariance::Whitening::new(&u, alpha)?
                .apply(&whitened);
        }
        let chi = whitened.norm_squared();
        if let Some(delta) = m["robust_data"]["pseudo_huber_delta"].as_f64() {
            let robust_cost: f64 = whitened
                .iter()
                .map(|z| 2. * z * z / ((1. + (z / delta).powi(2)).sqrt() + 1.))
                .sum();
            let claimed = m["fit"]["data_objective"].as_f64().unwrap();
            ensure!(
                (robust_cost - claimed).abs() / robust_cost.max(1.) < 1e-11,
                "robust data cost mismatch"
            );
        }
        if m["helium_fixed_zero"].as_bool() == Some(true) {
            ensure!(he.iter().all(|v| *v == 0.), "fixed helium is nonzero");
            let columns: Vec<usize> = serde_json::from_value(m["fit"]["filter_columns"].clone())?;
            ensure!(
                [17, 23, 29].iter().all(|c| !columns.contains(c)),
                "fixed helium in fitted covariance"
            );
        }
        let claimed = m["fit"]["data_chi_square"].as_f64().unwrap();
        let rel = (chi - claimed).abs() / chi.max(1.);
        max_relative_cost = max_relative_cost.max(rel);
        ensure!(rel < 1e-11, "window {t} cost mismatch");
        let x = m["fit"]["transformed_profile"]
            .as_array()
            .unwrap()
            .iter()
            .map(|v| v.as_f64().unwrap())
            .collect::<Vec<_>>();
        if let Some(shape) = syisr_long_pulse::composition::from_metadata(&m)? {
            for (j, r) in z.iter().enumerate() {
                let (fractions, _) = shape.fractions(*r + 0.05, &x, None);
                ensure!(
                    (fractions[0] - h[j]).abs() < 1e-12 && (fractions[1] - he[j]).abs() < 1e-12,
                    "window {t} analytic composition export mismatch"
                );
            }
        }
        let scales = [0.5, 0.25, 0.3, 0.15, 0.15, 0.1];
        let mut penalty = 0.;
        for k in 1..z.len() - 1 {
            let dl = z[k] - z[k - 1];
            let dr = z[k + 1] - z[k];
            for p in 0..6 {
                if p >= 4 && m["composition_shape"].is_object() {
                    continue;
                }
                if p == 0
                    && m["bottomside_density"]["join_slant_range_km"]
                        .as_f64()
                        .map(|join| z[k] <= join)
                        .unwrap_or(false)
                {
                    continue;
                }
                let dd = 2e4 / (dl + dr)
                    * ((x[(k + 1) * 6 + p] - x[k * 6 + p]) / dr
                        - (x[k * 6 + p] - x[(k - 1) * 6 + p]) / dl);
                penalty += (dd / scales[p]).powi(2);
            }
        }
        let weight = m["regularization"]["weight"].as_f64().unwrap();
        ensure!(
            (weight * penalty - m["fit"]["penalty"].as_f64().unwrap()).abs() < 1e-7,
            "window {t} penalty mismatch"
        );
        if let Some(path) = m["temporal_prior"]["product"].as_str() {
            let prior = syisr_long_pulse::temporal::Prior::load(path)?;
            let difference = nalgebra::DVector::from_iterator(
                prior.columns.len(),
                prior
                    .columns
                    .iter()
                    .enumerate()
                    .map(|(j, &i)| x[i] - prior.mean[j]),
            );
            let innovation = &prior.inverse_cholesky * difference;
            let penalty = innovation
                .iter()
                .map(|&z| {
                    prior
                        .robust_delta
                        .map(|d| {
                            syisr_long_pulse::temporal::pseudo_huber_residual(z, d)
                                .0
                                .powi(2)
                        })
                        .unwrap_or(z * z)
                })
                .sum::<f64>();
            ensure!(
                (penalty - m["fit"]["temporal_penalty"].as_f64().unwrap()).abs() < 1e-7,
                "temporal prior cost mismatch"
            );
        }
        let hist = m["fit"]["objective_history"].as_array().unwrap();
        ensure!(
            hist.windows(2)
                .all(|v| v[1].as_f64().unwrap() < v[0].as_f64().unwrap()),
            "nonmonotone accepted objective"
        );
        let source = hdf5::File::open(m["provenance"]["lpm"].as_str().unwrap())?;
        let partial = m["partial_products"]["source"]
            .as_str()
            .map(hdf5::File::open)
            .transpose()?;
        let merged = |key: &str| -> Result<Vec<f64>> {
            let mut v = if let Some(p) = &partial {
                p.dataset(key)?.read_raw::<f64>()?
            } else {
                vec![]
            };
            let original = source.dataset(key)?.read_raw::<f64>()?;
            let skip = if partial.is_some() {
                if key == "slant_range_km" {
                    6
                } else {
                    6 * 61
                }
            } else {
                0
            };
            v.extend_from_slice(&original[skip..]);
            Ok(v)
        };
        let source_ranges = merged("slant_range_km")?;
        let observed_ranges = get("observed_slant_range_km");
        let maximum = m["requested_observation_max_slant_range_km"]
            .as_f64()
            .unwrap_or(1450.);
        ensure!(
            (observed_ranges.last().unwrap() - maximum).abs() < 4.,
            "window {t} missing upper observation"
        );
        for (saved, original, sem) in [
            ("observed_real_adu2", "acf_real_adu2", false),
            ("observed_imag_adu2", "acf_imag_adu2", false),
            ("sem_real_adu2", "acf_real_sem_adu2", true),
            ("sem_imag_adu2", "acf_imag_sem_adu2", true),
        ] {
            let data = merged(original)?;
            let values = get(saved);
            for (g, r) in observed_ranges.iter().enumerate() {
                let i = source_ranges
                    .iter()
                    .position(|z| (z - r).abs() < 1e-8)
                    .ok_or_else(|| anyhow::anyhow!("window {t} fabricated observation range"))?;
                for l in 1..61 {
                    let expected = if sem {
                        data[i * 61 + l].max(1.)
                    } else {
                        data[i * 61 + l]
                    };
                    ensure!(
                        values[g * 60 + l - 1] == expected,
                        "window {t} source mismatch: {saved}, gate {g}, lag {l}"
                    );
                }
            }
        }
        let source_metadata: serde_json::Value =
            serde_json::from_slice(&source.attr("metadata_json")?.read_raw::<u8>()?)?;
        let ipp = source_metadata["header"]["ipp_us"].as_f64().unwrap() * 1e-6;
        let time = get("time_utc_s");
        let source_time = source.dataset("time_utc_s")?.read_raw::<f64>()?;
        ensure!(
            time.len() == 2
                && (time[0] - source_time[0]).abs() < 1e-9
                && (time[1] - source_time[source_time.len() - 1] - ipp).abs() < 1e-6,
            "window {t} UTC interval mismatch"
        );
        let mask = source.dataset("pulse_keep")?.read_raw::<u8>()?;
        let sm: serde_json::Value =
            serde_json::from_slice(&source.attr("metadata_json")?.read_raw::<u8>()?)?;
        ensure!(
            mask.iter().map(|&v| v as usize).sum::<usize>()
                == sm["retained_pulse_count"].as_f64().unwrap() as usize,
            "source pulse-mask count"
        );
        nodes += z.len();
        count += 1;
    }
    let result = serde_json::json!({"validated_windows":count,"physical_profile_nodes":nodes,"fixed_calibration":gain,"max_relative_data_cost_difference":max_relative_cost,"checks":"10-600 us only; physical simplex/bounds; UTC-source data; altitude conversion; signed finite observations; independently replayed data and regularization costs; common gain; monotone accepted objective; source pulse-mask counts"});
    std::fs::write(
        root.join("validation.json"),
        serde_json::to_vec_pretty(&result)?,
    )?;
    println!("{}", serde_json::to_string_pretty(&result)?);
    Ok(())
}
