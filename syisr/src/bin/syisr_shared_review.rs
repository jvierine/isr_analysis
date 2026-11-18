//! Audit temporal sweeps against the same whole-day penalized objective.
use anyhow::{ensure, Result};
use std::{collections::BTreeMap, fs, path::Path};
use syisr_ac::{campaign, h_transition::Transition};
fn main() -> Result<()> {
    let a: Vec<_> = std::env::args().collect();
    ensure!(a.len() == 3, "STAGE PREVIOUS-or-none");
    let stage = Path::new(&a[1]);
    let mut rows = BTreeMap::new();
    for p in campaign::files(&stage.join("shared-windows"), "h5")? {
        let v: serde_json::Value = campaign::load(&p)?;
        rows.insert(v["start"].as_i64().unwrap(), v);
    }
    ensure!(!rows.is_empty(), "empty stage");
    let mut objective = 0.;
    let mut failed = 0;
    let mut change = 0f64;
    let mut prior: Option<(i64, Transition)> = None;
    for (&time, v) in &rows {
        objective += v["robust_objective"].as_f64().unwrap();
        if v["status"] != "converged" && v["model_inferred_only"] != true {
            failed += 1;
        }
        let tr: Transition = serde_json::from_value(v["h_transition"].clone())?;
        if let Some((old, ref prev)) = prior {
            let dt = (time - old) as f64;
            for z in (600..=1300).step_by(100) {
                let d = tr.shape.evaluate(z as f64, tr.normalized_parameters).0
                    - prev.shape.evaluate(z as f64, prev.normalized_parameters).0;
                objective += d * d / (0.15f64.powi(2) * dt / 1800.);
            }
        }
        if a[2] != "none" {
            let old: serde_json::Value = campaign::load(
                &Path::new(&a[2])
                    .join("shared-windows")
                    .join(format!("{time}.h5")),
            )?;
            let p: Transition = serde_json::from_value(old["h_transition"].clone())?;
            for z in (500..=1300).step_by(10) {
                change = change.max(
                    (tr.shape.evaluate(z as f64, tr.normalized_parameters).0
                        - p.shape.evaluate(z as f64, p.normalized_parameters).0)
                        .abs(),
                );
            }
        }
        prior = Some((time, tr));
    }
    let report = serde_json::json!({"windows":rows.len(),"measured_windows":rows.values().filter(|v|v["model_inferred_only"]!=true).count(),"model_inferred_windows":rows.values().filter(|v|v["model_inferred_only"]==true).count(),"nonconverged_windows":failed,"global_huber_plus_temporal_objective":objective,"maximum_fraction_change_from_previous_sweep":if a[2]=="none"{None}else{Some(change)},"temporal_sigma_per_30min":0.15,"temporal_altitude_samples_km":[600,700,800,900,1000,1100,1200,1300],"missing_measurements_interpolated":false,"uncertainty":"disabled","sweep_fixed_point_tolerance_fraction":0.005});
    campaign::save(&stage.join("review.h5"), &report)?;
    fs::write(
        stage.join("review.json"),
        serde_json::to_vec_pretty(&report)?,
    )?;
    println!("{}", serde_json::to_string(&report)?);
    Ok(())
}
