//! Compare the filtered-voltage mean with archived range response, using the
//! same piecewise-constant source cells. No measured samples are interpolated.
use anyhow::{ensure, Context, Result};
use serde_json::{json, Value};
fn main() -> Result<()> {
    let a: Vec<_> = std::env::args().collect();
    ensure!(a.len() == 5, "INPUT.json FILTERED.json RANGE.json NEW.json");
    ensure!(!std::path::Path::new(&a[4]).exists(), "output exists");
    let read = |p: &str| -> Result<Value> { Ok(serde_json::from_slice(&std::fs::read(p)?)?) };
    let input = read(&a[1])?;
    let filtered = read(&a[2])?;
    let ranges = read(&a[3])?;
    let profile: Vec<f64> = serde_json::from_value(input["profile_counts"].clone())?;
    let means: Vec<[f64; 2]> = serde_json::from_value(filtered["mean_counts"].clone())?;
    let groups = ranges["groups"].as_array().context("groups")?;
    let mut reference = vec![];
    for gate in 0..means.len() {
        let address = 480. + gate as f64;
        let group = groups
            .iter()
            .find(|g| {
                let start = g["lpg_ra"].as_f64().unwrap();
                let n = g["lpg_nt"].as_f64().unwrap();
                address >= start && address < start + n
            })
            .context("missing gate")?;
        let j = address - group["lpg_ra"].as_f64().unwrap();
        let dt = group["lpg_dt"].as_f64().context("step")?;
        let samples: Vec<[f64; 2]> = serde_json::from_value(group["samples"].clone())?;
        let norm: f64 = samples.iter().map(|p| p[1]).sum();
        ensure!(norm > 0., "invalid response");
        let mut mean = 0.;
        for [sample, w] in samples {
            let cell = ((sample + j * dt - 810.5) / 25. + 0.5).floor() as isize;
            if cell >= 0 && (cell as usize) < profile.len() {
                mean += w * profile[cell as usize] / norm;
            }
        }
        reference.push(mean);
    }
    let residual: Vec<f64> = reference
        .iter()
        .zip(&means)
        .map(|(r, m)| m[0] - r)
        .collect();
    let norm = reference.iter().map(|r| r * r).sum::<f64>().sqrt();
    let relative = residual.iter().map(|r| r * r).sum::<f64>().sqrt() / norm;
    let max_relative = reference
        .iter()
        .zip(&means)
        .filter(|(r, _)| **r > 0.)
        .map(|(r, m)| (m[0] / r - 1.).abs())
        .fold(0., f64::max);
    println!("Relative mean L2 discrepancy {relative:.6e}; max positive-cell discrepancy {max_relative:.6e}");
    std::fs::write(
        &a[4],
        serde_json::to_vec_pretty(
            &json!({"status":"mean_operator_comparison_not_acceptance","relative_l2_discrepancy":relative,"max_relative_discrepancy":max_relative,"archived_mean":reference,"filtered_mean":means,"limitations":"Archive is zero-frequency marginal; finite-width exponential ACF differs physically. Compare coherent limit separately. Discrete archive versus ADC midpoint range quadrature differ."}),
        )?,
    )?;
    Ok(())
}
