//! Independently replay saved scientific-product invariants (no optimizer calls).
use anyhow::{ensure, Result};
fn main() -> Result<()> {
    let a = std::env::args().collect::<Vec<_>>();
    ensure!(
        a.len() == 2 || a.len() == 3,
        "profile-validate campaign-root [window-index]"
    );
    let root = std::path::Path::new(&a[1]);
    let mut count = 0;
    let mut gain: Option<f64> = None;
    let mut max_relative_cost = 0f64;
    let mut nodes = 0;
    let windows = if a.len() == 3 {
        vec![a[2].parse::<usize>()?]
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
        let mut chi = 0.;
        let mut normalized_residual = vec![0.; 2520];
        for component in ["real", "imag"] {
            let obs = get(&format!("observed_{component}_adu2"));
            let model = get(&format!("model_{component}_adu2"));
            let sem = get(&format!("sem_{component}_adu2"));
            ensure!(
                obs.len() == 1260 && model.len() == 1260 && sem.len() == 1260,
                "ACF dimensions"
            );
            for i in 0..obs.len() {
                ensure!(
                    obs[i].is_finite() && model[i].is_finite() && sem[i] > 0.,
                    "invalid ACF"
                );
                chi += ((obs[i] - model[i]) / sem[i]).powi(2);
                normalized_residual[2 * i + usize::from(component == "imag")] =
                    (model[i] - obs[i]) / sem[i];
            }
        }
        if let Ok(d) = f.dataset("measurement_covariance_normalized_modes") {
            let shape = d.shape();
            let u = nalgebra::DMatrix::from_row_slice(shape[0], shape[1], &d.read_raw::<f64>()?);
            let alpha = f
                .attr("measurement_covariance_shrinkage")?
                .read_scalar::<f64>()?;
            let w = syisr_long_pulse::measurement_covariance::Whitening::new(&u, alpha)?;
            chi = w
                .apply(&nalgebra::DMatrix::from_column_slice(
                    2520,
                    1,
                    &normalized_residual,
                ))
                .norm_squared();
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
        let scales = [0.5, 0.25, 0.3, 0.15, 0.15, 0.1];
        let mut penalty = 0.;
        for k in 1..z.len() - 1 {
            let dl = z[k] - z[k - 1];
            let dr = z[k + 1] - z[k];
            for p in 0..6 {
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
        let hist = m["fit"]["objective_history"].as_array().unwrap();
        ensure!(
            hist.windows(2)
                .all(|v| v[1].as_f64().unwrap() < v[0].as_f64().unwrap()),
            "nonmonotone accepted objective"
        );
        let source = hdf5::File::open(m["provenance"]["lpm"].as_str().unwrap())?;
        let time = get("time_utc_s");
        let source_time = source.dataset("time_utc_s")?.read_raw::<f64>()?;
        ensure!(
            time.len() == 2
                && (time[0] - source_time[0]).abs() < 1e-9
                && (time[1] - source_time[source_time.len() - 1] - 0.04).abs() < 1e-6,
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
