//! Export the actual filtered/RTS state on a display grid; keep uncertainty conditional.
use anyhow::{ensure, Result};
use std::path::Path;
use syisr_long_pulse::{composition, profile};
fn main() -> Result<()> {
    let a: Vec<_> = std::env::args().collect();
    ensure!(
        a.len() == 3 || a.len() == 4,
        "export-temporal session-root filtered|smoothed [fit-directory]"
    );
    let root = Path::new(&a[1]);
    let fit_directory = a.get(3).map(String::as_str).unwrap_or("temporal");
    let smooth = a[2] == "smoothed";
    ensure!(smooth || a[2] == "filtered", "unknown state");
    let plan: serde_json::Value = serde_json::from_slice(&std::fs::read(root.join("plan.json"))?)?;
    let windows = plan["windows"].as_array().unwrap();
    let nt = windows.len();
    let maximum = if plan["layout"]["samples"].as_u64() == Some(1194) {
        1650
    } else {
        5000
    };
    let ranges: Vec<f64> = (200..=maximum).step_by(10).map(|v| v as f64).collect();
    let n = ranges.len() * nt;
    let keys = [
        "ne_m3",
        "ti_k",
        "te_k",
        "spectral_velocity_m_s",
        "h_fraction",
        "he_fraction",
        "o_fraction",
        "sigma_h",
        "sigma_he",
        "sigma_ti_k",
        "sigma_te_k",
        "sigma_log_ne",
        "reduced_chi2",
    ];
    let mut arrays = vec![vec![f64::NAN; n]; keys.len()];
    let mut flags = vec![255u8; n];
    let mut valid = vec![0u8; n];
    let mut done = vec![0u8; nt];
    let mut helium_fixed_zero = false;
    for t in 0..nt {
        let dir = root.join(format!("{fit_directory}/windows/{t:03}"));
        let path = dir.join("profile.h5");
        if !path.exists() {
            continue;
        }
        let f = hdf5::File::open(path)?;
        let m: serde_json::Value =
            serde_json::from_slice(&f.attr("metadata_json")?.read_raw::<u8>()?)?;
        helium_fixed_zero |= m["helium_fixed_zero"].as_bool().unwrap_or(false);
        if m["fit"]["status"] != "converged" {
            continue;
        }
        let z = f.dataset("slant_range_km")?.read_raw::<f64>()?;
        let mut x: Vec<f64> = serde_json::from_value(m["fit"]["transformed_profile"].clone())?;
        let np = x.len();
        let cols: Vec<usize> = serde_json::from_value(m["fit"]["filter_columns"].clone())?;
        let nc = cols.len();
        let mut covariance = vec![0.; np * np];
        if smooth {
            let state = dir.join("smoothed.h5");
            if !state.exists() {
                continue;
            }
            let h = hdf5::File::open(state)?;
            let mean = h.dataset("mean")?.read_raw::<f64>()?;
            let p = h.dataset("covariance")?.read_raw::<f64>()?;
            for (i, &col) in cols.iter().enumerate() {
                x[col] = mean[i];
                for (j, &c) in cols.iter().enumerate() {
                    covariance[col * np + c] = p[i * nc + j];
                }
            }
        } else {
            let p: Vec<f64> = serde_json::from_value(m["fit"]["filter_covariance"].clone())?;
            for (i, &col) in cols.iter().enumerate() {
                for (j, &c) in cols.iter().enumerate() {
                    covariance[col * np + c] = p[j * nc + i];
                }
            }
        }
        let shape = composition::from_metadata(&m)?
            .ok_or_else(|| anyhow::anyhow!("global composition required"))?;
        let ng = f.dataset("observed_slant_range_km")?.shape()[0];
        let red =
            m["fit"]["data_chi_square"].as_f64().unwrap() / ((ng * 119) as f64 - cols.len() as f64);
        done[t] = 1;
        let global_invalid = cols.iter().any(|&i| {
            let (lo, hi) = profile::bound(i % 6, false);
            x[i] < lo - 1e-6 || x[i] > hi + 1e-6
        });
        for (j, &r) in ranges.iter().enumerate() {
            let i = j * nt + t;
            let w = profile::interpolation(&z, r)?;
            let mut q = [0.; 4];
            for (k, v) in &w {
                for b in 0..4 {
                    q[b] += v * x[6 * k + b];
                }
            }
            let variance = |field: usize| {
                let mut v = 0.;
                for (k, a) in &w {
                    for (l, b) in &w {
                        v += a * b * covariance[(6 * k + field) * np + 6 * l + field];
                    }
                }
                v.max(0.).sqrt()
            };
            let (frac, errors) = shape.fractions(r + 0.05, &x, Some(&covariance));
            arrays[0][i] = 1e11 * q[0].exp();
            arrays[1][i] = 1000. * q[1].exp();
            arrays[2][i] = 1000. * q[2].exp();
            arrays[3][i] = 1000. * q[3];
            for b in 0..3 {
                arrays[4 + b][i] = frac[b];
            }
            arrays[7][i] = errors[0];
            arrays[8][i] = errors[1];
            arrays[9][i] = arrays[1][i] * variance(1);
            arrays[10][i] = arrays[2][i] * variance(2);
            arrays[11][i] = variance(0);
            arrays[12][i] = red;
            flags[i] = if global_invalid {
                8
            } else if red > 3. {
                2
            } else {
                0
            };
            if w.iter().filter(|(_, v)| *v > 1e-12).any(|(k, _)| {
                let ti = 1000. * x[6 * k + 1].exp();
                let te = 1000. * x[6 * k + 2].exp();
                ti < 301. || ti > 4999. || te < 301. || te > 11999.
            }) {
                flags[i] |= 4;
            }
            valid[i] = u8::from(flags[i] == 0 && variance(0) < 0.5);
        }
    }
    let out = hdf5::File::create(root.join(format!("{fit_directory}/{}-campaign.h5", a[2])))?;
    out.new_attr::<f64>()
        .create("display_max_slant_range_km")?
        .write_scalar(&(maximum as f64))?;
    out.new_attr_builder()
        .with_data("joint_full_profile_temporal")
        .create("analysis_type")?;
    out.new_attr::<u8>()
        .create("helium_fixed_zero")?
        .write_scalar(&u8::from(helium_fixed_zero))?;
    let config_path = root.join(format!("{fit_directory}/configuration.json"));
    let robust_config: Option<serde_json::Value> = if config_path.exists() {
        Some(serde_json::from_slice(&std::fs::read(config_path)?)?)
    } else {
        None
    };
    if let Some(config) = robust_config {
        out.new_attr_builder().with_data("Nonlinear robust temporal coordinate MAP; each data objective and adjacent process edge counted once; pseudo-Huber temporal/data norms; no Gaussian RTS; uncertainties conditional on fixed neighbours").create("temporal_method")?;
        out.new_attr_builder()
            .with_data(serde_json::to_string(&config)?.as_bytes())
            .create("robust_configuration_json")?;
    } else {
        out.new_attr_builder().with_data(if smooth{"Forward Bayesian filter and reverse RTS; weak random-walk continuity; each lag likelihood used once; local Gaussian and fixed-shape conditional uncertainty; bounds/model discrepancy retained as flags"}else{"Forward nonlinear Bayesian filter; previous fitted state initializes next likelihood with process-noise-widened prior"}).create("temporal_method")?;
    }
    out.new_attr_builder()
        .with_data(if helium_fixed_zero {
            "Exponential-scale-height-skewed sigmoid H; He fixed zero"
        } else {
            "Exponential-scale-height-skewed sigmoid H; Gaussian non-H He envelope"
        })
        .create("composition_model")?;
    for (k, name) in keys.iter().enumerate() {
        out.new_dataset::<f64>()
            .shape((ranges.len(), nt))
            .create(*name)?
            .write_raw(&arrays[k])?;
    }
    for (name, data) in [
        ("slant_range_km", ranges.clone()),
        ("altitude_km", ranges.iter().map(|r| r + 0.05).collect()),
    ] {
        out.new_dataset_builder().with_data(&data).create(name)?;
    }
    let mut edges = vec![windows[0]["utc_start"].as_f64().unwrap()];
    edges.extend(windows.iter().map(|w| w["utc_end"].as_f64().unwrap()));
    out.new_dataset_builder()
        .with_data(&edges)
        .create("time_edges_utc_s")?;
    out.new_dataset_builder()
        .with_data(&done)
        .create("window_complete")?;
    out.new_dataset::<u8>()
        .shape((ranges.len(), nt))
        .create("quality_flags")?
        .write_raw(&flags)?;
    out.new_dataset::<u8>()
        .shape((ranges.len(), nt))
        .create("density_valid")?
        .write_raw(&valid)?;
    Ok(())
}
