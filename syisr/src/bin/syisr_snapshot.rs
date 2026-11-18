//! Compact immutable snapshot of completed window outputs; no raw/optimizer data are altered.
use anyhow::{ensure, Result};
use std::{collections::BTreeSet, fs, path::Path};
use syisr_ac::campaign;
fn main() -> Result<()> {
    let a: Vec<_> = std::env::args().collect();
    ensure!(a.len() == 4, "INPUT_ROOT STAGE SNAPSHOT");
    let input = Path::new(&a[1]);
    let stage = Path::new(&a[2]);
    let out = Path::new(&a[3]);
    fs::create_dir_all(out)?;
    fs::copy(input.join("campaign.h5"), out.join("campaign.h5"))?;
    let mut done = BTreeSet::new();
    let mut rows = vec![];
    for p in campaign::files(&stage.join("shared-windows"), "h5")? {
        let v: serde_json::Value = campaign::load(&p)?;
        done.insert(v["start"].as_i64().unwrap());
        rows.push(v.clone());
        campaign::save(&out.join("shared-windows").join(p.file_name().unwrap()), &v)?;
    }
    let mut count = 0;
    for p in campaign::files(&stage.join("joint-fits"), "h5")? {
        let mut v: serde_json::Value = campaign::load(&p)?;
        let t = v["job"]["start"].as_f64().unwrap() as i64;
        if !done.contains(&t) {
            continue;
        }
        for k in [
            "product_sigma",
            "bootstrap_control_nodes",
            "bootstrap_ne_m3",
            "control_nodes",
        ] {
            v.as_object_mut().unwrap().remove(k);
        }
        for g in v["gates"].as_array_mut().unwrap() {
            for key in ["observed", "prediction", "h_profile", "bootstrap_nodes"] {
                g[key] = serde_json::json!([]);
            }
        }
        campaign::save(&out.join("fits").join(p.file_name().unwrap()), &v)?;
        count += 1;
    }
    let summary = serde_json::json!({"completed_windows":rows.len(),"beam_profiles":count,"converged_windows":rows.iter().filter(|v|v["status"]=="converged").count(),"window_status":rows.iter().map(|v|serde_json::json!({"start":v["start"],"status":v["status"],"normalized_gradient":v["normalized_gradient"],"iterations":v["iterations"]})).collect::<Vec<_>>(),"snapshot":"Only completed solver outputs; convergence flags preserved. Active optimizer checkpoints are not included."});
    campaign::save(&out.join("snapshot-summary.h5"), &summary)?;
    fs::write(
        out.join("snapshot-summary.json"),
        serde_json::to_vec_pretty(&summary)?,
    )?;
    println!("{summary}");
    Ok(())
}
