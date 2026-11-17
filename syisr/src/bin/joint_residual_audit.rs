//! Same-data short-lag branch audit; no inference from an isolated gate alone.
use anyhow::Result;
use serde_json::{json, Value};
use std::{fs, path::Path};
use syisr_ac::campaign;
fn med(v: &mut Vec<f64>) -> Option<f64> {
    if v.is_empty() {
        None
    } else {
        v.sort_by(f64::total_cmp);
        Some(v[v.len() / 2])
    }
}
fn main() -> Result<()> {
    let a: Vec<_> = std::env::args().collect();
    let mut rows = vec![];
    for e in fs::read_dir(&a[1])? {
        let p = e?.path();
        if p.extension().is_none_or(|v| v != "h5") {
            continue;
        }
        let d: Value = campaign::load(&p)?;
        let Some(sigma) = d["product_sigma"].as_array() else {
            continue;
        };
        let mut num = [0.; 2];
        let mut den = [0.; 2];
        let mut residuals = vec![vec![]; 10];
        for (g, gate) in d["gates"].as_array().unwrap().iter().enumerate() {
            let z = gate["altitude_km"].as_f64().unwrap();
            if !(200.0..=600.0).contains(&z) {
                continue;
            }
            for l in 0..10 {
                let obs = gate["observed"][l][0].as_f64().unwrap();
                let model = gate["prediction"][l][0].as_f64().unwrap();
                let sd = sigma[g * 62 + l][0].as_f64().unwrap();
                if sd >= 1e20 {
                    continue;
                }
                let k = l % 2;
                num[k] += obs * model / (sd * sd);
                den[k] += model * model / (sd * sd);
                residuals[l].push((model - obs) / sd);
            }
        }
        let odd = num[0] / den[0];
        let even = num[1] / den[1];
        rows.push(json!({"job":d["job"],"optimizer_status":d["nominal_fit"]["status"],"gradient":d["nominal_fit"]["normalized_gradient"],"odd_multiplier":odd,"even_multiplier":even,"odd_to_even":odd/even,"median_standardized_real_residual_by_lag":residuals.iter_mut().map(med).collect::<Vec<_>>()}));
    }
    campaign::save(
        Path::new(&a[2]),
        &json!({"schema":"joint-short-lag-branch-audit-v1","altitude_interval_km":[200,600],"lag_us":[10,20,30,40,50,60,70,80,90,100],"limitation":"working diagonal weights; overlapping gate errors correlated; branch factors are diagnostics, not independent calibration measurements","rows":rows}),
    )?;
    Ok(())
}
