//! Reconstruct actual fitted basis on a fine display grid; propagate cross-knot covariance.
use anyhow::{ensure, Result};
use syisr_long_pulse::profile::interpolation;
fn main() -> Result<()> {
    let a: Vec<String> = std::env::args().collect();
    ensure!(a.len() == 3, "interpolate-campaign profile-root output.h5");
    let root = std::path::Path::new(&a[1]);
    let h = hdf5::File::open(root.join("campaign.h5"))?;
    let cz = h.dataset("slant_range_km")?.read_raw::<f64>()?;
    let done = h.dataset("window_complete")?.read_raw::<u8>()?;
    let nt = done.len();
    let cf = h.dataset("quality_flags")?.read_raw::<u8>()?;
    let cv = h.dataset("density_valid")?.read_raw::<u8>()?;
    let maximum = h
        .attr("display_max_slant_range_km")
        .ok()
        .and_then(|a| a.read_scalar::<f64>().ok())
        .unwrap_or(1400.);
    let ranges: Vec<f64> = (200..=maximum as usize)
        .step_by(10)
        .map(|v| v as f64)
        .collect();
    let n = ranges.len() * nt;
    let keys = [
        "ne_m3",
        "ti_k",
        "te_k",
        "h_fraction",
        "he_fraction",
        "o_fraction",
        "sigma_h",
        "sigma_he",
        "sigma_ti_k",
        "sigma_te_k",
        "reduced_chi2",
    ];
    let mut arrays = vec![vec![f64::NAN; n]; keys.len()];
    let mut flags = vec![255u8; n];
    let mut valid = vec![0u8; n];
    for t in 0..nt {
        if done[t] == 0 {
            continue;
        }
        let f = hdf5::File::open(root.join(format!("windows/{t:03}/profile.h5")))?;
        let m: serde_json::Value =
            serde_json::from_slice(&f.attr("metadata_json")?.read_raw::<u8>()?)?;
        let composition = syisr_long_pulse::composition::from_metadata(&m)?;
        let z = f.dataset("slant_range_km")?.read_raw::<f64>()?;
        let np = z.len() * 6;
        let x: Vec<f64> = m["fit"]["transformed_profile"]
            .as_array()
            .unwrap()
            .iter()
            .map(|v| v.as_f64().unwrap())
            .collect();
        let cov = f
            .dataset("conditional_transformed_covariance")
            .ok()
            .map(|d| d.read_raw::<f64>().unwrap())
            .unwrap_or_else(|| vec![f64::NAN; np * np]);
        for (j, &r) in ranges.iter().enumerate() {
            let i = j * nt + t;
            let w = interpolation(&z, r)?;
            let mut q = [0.; 6];
            for &(k, v) in &w {
                for b in 0..6 {
                    q[b] += v * x[k * 6 + b];
                }
            }
            let density_mapping = m["bottomside_density"]["join_slant_range_km"]
                .as_f64()
                .and_then(|join| {
                    let k = z.iter().position(|v| (*v - join).abs() < 1e-8).unwrap();
                    syisr_long_pulse::profile::bottomside_mapping(
                        &z,
                        k,
                        m["bottomside_density"]["scale_height_km"].as_f64().unwrap(),
                        r,
                    )
                });
            if let Some(d) = &density_mapping {
                q[0] = d.offset + d.weights.iter().map(|&(k, w)| w * x[6 * k]).sum::<f64>();
            }
            let variance = |a: usize, b: usize| {
                let mut sum = 0.;
                let wa = if a == 0 {
                    density_mapping.as_ref().map(|d| &d.weights).unwrap_or(&w)
                } else {
                    &w
                };
                let wb = if b == 0 {
                    density_mapping.as_ref().map(|d| &d.weights).unwrap_or(&w)
                } else {
                    &w
                };
                for &(k, v) in wa {
                    for &(l, u) in wb {
                        if v * u != 0. {
                            sum += v * u * cov[(6 * k + a) * np + 6 * l + b];
                        }
                    }
                }
                sum
            };
            arrays[0][i] = 1e11 * q[0].exp();
            arrays[1][i] = 1000. * q[1].exp();
            arrays[2][i] = 1000. * q[2].exp();
            arrays[3][i] = q[4];
            arrays[4][i] = (1. - q[4]) * q[5];
            arrays[5][i] = (1. - q[4]) * (1. - q[5]);
            arrays[6][i] = variance(4, 4).sqrt();
            arrays[7][i] = (q[5].powi(2) * variance(4, 4) + (1. - q[4]).powi(2) * variance(5, 5)
                - 2. * q[5] * (1. - q[4]) * variance(4, 5))
            .sqrt();
            if let Some(shape) = &composition {
                let (fractions, errors) = shape.fractions(r + 0.05, &x, Some(&cov));
                for a in 0..3 {
                    arrays[3 + a][i] = fractions[a];
                }
                arrays[6][i] = errors[0];
                arrays[7][i] = errors[1];
            }
            arrays[8][i] = arrays[1][i] * variance(1, 1).sqrt();
            arrays[9][i] = arrays[2][i] * variance(2, 2).sqrt();
            let ng = f.dataset("observed_slant_range_km")?.size();
            arrays[10][i] = m["fit"]["data_chi_square"].as_f64().unwrap()
                / ((ng * 120 - ng) as f64
                    - m["fitted_parameter_count"].as_u64().unwrap_or(np as u64) as f64);
            // Every contributing knot matters: nearest-knot screening can
            // otherwise paint through a bound-limited neighbour. Zero-weight
            // neighbours must not suppress an exactly sampled valid knot.
            flags[i] = w
                .iter()
                .filter(|(_, v)| *v > 1e-12)
                .fold(0, |a, (k, _)| a | cf[k * nt + t]);
            valid[i] = u8::from(
                w.iter()
                    .filter(|(_, v)| *v > 1e-12)
                    .all(|(k, _)| cv[k * nt + t] > 0)
                    && variance(0, 0).sqrt() < 0.5,
            );
        }
    }
    let out = hdf5::File::create(&a[2])?;
    if h.attr("composition_model").is_ok() {
        out.new_attr_builder().with_data("Analytic IRI-height sigmoid H and Gaussian He envelope evaluated at display altitude; full conditional global-parameter covariance propagated").create("composition_model")?;
    }
    out.new_attr::<f64>()
        .create("display_max_slant_range_km")?
        .write_scalar(&maximum)?;
    out.new_attr_builder()
        .with_data("joint_full_profile_interpolated")
        .create("analysis_type")?;
    if cz[0] < 200. {
        out.new_attr_builder().with_data("Partial-pulse range-lag inversion; physical fit to 200 km and lower nuisance layers; all original flags retained").create("partial_pulse_extension")?;
    }
    out.new_attr_builder().with_data("Actual transformed thermal/density basis sampled every 10 km; composition uses analytic global model when composition_model is present, otherwise nonlinear knot reconstruction; conditional errors include cross-knot covariance. Screening requires every nonzero-weight knot to pass, plus interpolated error cuts. Display sampling is not physical resolution.").create("display_interpolation")?;
    for (k, key) in keys.iter().enumerate() {
        out.new_dataset::<f64>()
            .shape((ranges.len(), nt))
            .create(*key)?
            .write_raw(&arrays[k])?;
    }
    for (name, v) in [
        ("slant_range_km", ranges.clone()),
        ("altitude_km", ranges.iter().map(|r| r + 0.05).collect()),
    ] {
        out.new_dataset_builder().with_data(&v).create(name)?;
    }
    for name in ["time_edges_utc_s", "time_centres_utc_s"] {
        if let Ok(d) = h.dataset(name) {
            out.new_dataset_builder()
                .with_data(&d.read_raw::<f64>()?)
                .create(name)?;
        }
    }
    out.new_dataset_builder()
        .with_data(&done)
        .create("window_complete")?;
    out.new_dataset::<u8>()
        .shape((ranges.len(), nt))
        .create("density_valid")?
        .write_raw(&valid)?;
    out.new_dataset::<u8>()
        .shape((ranges.len(), nt))
        .create("quality_flags")?
        .write_raw(&flags)?;
    Ok(())
}
