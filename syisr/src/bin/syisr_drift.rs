//! Reconstruct provisional regional drift from the exact fitted beam splines.
use anyhow::{ensure, Result};
use serde_json::{json, Value};
use std::{collections::BTreeMap, path::Path};
use syisr_ac::{campaign, drift, spline::Spline};
fn main() -> Result<()> {
    let a: Vec<_> = std::env::args().collect();
    ensure!(a.len() == 3, "SNAPSHOT OUTPUT.h5");
    let root = Path::new(&a[1]);
    let mut windows: BTreeMap<i64, Vec<Value>> = BTreeMap::new();
    for p in campaign::files(&root.join("fits"), "h5")? {
        let v: Value = campaign::load(&p)?;
        ensure!(
            v["control_nodes"].is_array(),
            "exact spline controls absent: {}",
            p.display()
        );
        windows
            .entry(v["job"]["start"].as_f64().unwrap() as i64)
            .or_default()
            .push(v);
    }
    let mut output = vec![];
    for (start, profiles) in windows {
        for z in (250..=1000).step_by(10) {
            let mut beams = vec![];
            let mut rows = vec![];
            let mut magnetic = vec![];
            let mut y = vec![];
            for v in &profiles {
                let status = v["nominal_fit"]["status"].as_str().unwrap_or("");
                if status != "converged" && status != "data_stable_gradient_pending" {
                    continue;
                }
                let s: Spline = serde_json::from_value(v["spline"].clone())?;
                let nodes = v["control_nodes"].as_array().unwrap();
                ensure!(nodes.len() == s.count(), "spline control count");
                let w = s.weights(z as f64);
                let vel = w
                    .iter()
                    .map(|(k, w)| w * nodes[*k]["velocity_m_s"].as_f64().unwrap())
                    .sum::<f64>();
                let ne = w
                    .iter()
                    .map(|(k, w)| w * nodes[*k]["ne_m3"].as_f64().unwrap().ln())
                    .sum::<f64>()
                    .exp();
                if !vel.is_finite() || vel.abs() > 1800. || ne < 1e10 {
                    continue;
                }
                let az = v["job"]["az"].as_f64().unwrap();
                let el = v["job"]["el"].as_f64().unwrap();
                let g = drift::geometry(z as f64, az, el)?;
                rows.push(g.enu_row);
                magnetic.push(g.magnetic_row);
                y.push(vel);
                beams.push(json!({"id":v["job"]["id"],"azimuth_deg":az,"elevation_deg":el,"los_m_s":vel,"ne_m3":ne,"geometry":g,"optimizer_status":status}));
            }
            if y.len() < 5 {
                continue;
            }
            let enu = drift::invert(&rows, &y)?;
            let mag = drift::invert(&magnetic, &y)?;
            output.push(json!({"start_utc_s":start,"end_utc_s":start+1800,"altitude_km":z,"beams":beams,"geographic":enu,"magnetic":mag}));
        }
    }
    let good = output
        .iter()
        .filter(|r| r["magnetic"]["consistency_ok"] == true)
        .count();
    let n = output.len();
    let t: std::collections::BTreeSet<_> = output
        .iter()
        .map(|r| r["start_utc_s"].as_i64().unwrap())
        .collect();
    let v = json!({"schema":"syisr-regional-drift-v1","rows":output,"metadata":{"input_snapshot":root.to_string_lossy(),"velocity_sign":"Input AC software convention retained. Physical toward/away polarity is not independently calibrated. Reversing LOS polarity reverses all reconstructed vectors.","geographic_components":["east","north","up"],"magnetic_components":["perpendicular_east","perpendicular_meridional_north_up","parallel_along_B"],"magnetic_field":"IGRF14 2025.76; field geometry evaluated at each beam gate","model":"Common local components across sampled beam footprints in each 30-minute elapsed window. Geographic and magnetic inversions are separate regional models.","spline_evaluation":"Exact cubic basis and stored velocity controls; no interpolation of plotted gates","weights":"Equal starting beam weights; Huber IRLS with 1.5 threshold and MAD residual scale floor5m/s. Formal LOS uncertainties unavailable.","consistency_cut":"weighted condition<15; raw LOS residual RMS<30m/s; max leave-one-out prediction error<75m/s; every component jackknife spread<50m/s","uncertainty":"Leave-one-beam-out spread is a spatial/model consistency diagnostic, not a calibrated confidence interval. No covariance from plasma fit.","sami2_driver_ready":false},"summary":{"windows":t.len(),"rows":n,"magnetic_consistent_rows":good}});
    campaign::save(Path::new(&a[2]), &v)?;
    println!("{}", v["summary"]);
    Ok(())
}
