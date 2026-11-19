//! Matched LOS-to-vector comparison: average successful L2 parameters within AC windows.
use anyhow::{ensure, Result};
use chrono::NaiveDateTime;
use hdf5::types::FixedAscii;
use serde_json::{json, Value};
use std::{collections::BTreeMap, path::Path};
use syisr_ac::{campaign, drift};
fn mean(v: &[f64]) -> f64 {
    v.iter().sum::<f64>() / v.len() as f64
}
fn interp(p: &[(f64, f64, f64)], z: f64) -> Option<(f64, f64)> {
    let k = p.partition_point(|p| p.0 < z);
    if k < p.len() && (p[k].0 - z).abs() < 1e-6 {
        return Some((p[k].1, p[k].2));
    }
    if k == 0 || k == p.len() || p[k].0 - p[k - 1].0 > 60. {
        return None;
    }
    let a = (z - p[k - 1].0) / (p[k].0 - p[k - 1].0);
    Some((
        (1. - a) * p[k - 1].1 + a * p[k].1,
        (1. - a) * p[k - 1].2 + a * p[k].2,
    ))
}
fn main() -> Result<()> {
    let a: Vec<_> = std::env::args().collect();
    ensure!(a.len() == 4, "OUR_RAW_DRIFT L2 OUTPUT_ROOT");
    let ours: Value = campaign::load(Path::new(&a[1]))?;
    let f = hdf5::File::open(&a[2])?;
    let shape = f.dataset("Fittedparameter")?.shape();
    let (nb, nt, nr) = (shape[0], shape[1], shape[2]);
    let p = f.dataset("Fittedparameter")?.read_raw::<f64>()?;
    let error = f.dataset("Error")?.read_raw::<f64>()?;
    let state = f.dataset("Fitstate")?.read_raw::<f64>()?;
    let geo = f.dataset("GEO")?.read_raw::<f64>()?;
    let dirs = f.dataset("Beamdir")?.read_raw::<f64>()?;
    let ts = f.dataset("Time")?.read_raw::<FixedAscii<20>>()?;
    let mut samples: BTreeMap<(usize, i64), Vec<Vec<(f64, f64, f64)>>> = BTreeMap::new();
    for b in 0..nb {
        for t in 0..nt {
            let Ok(dt) =
                NaiveDateTime::parse_from_str(ts[b * nt + t].as_str().trim(), "%d-%b-%Y %H:%M:%S")
            else {
                continue;
            };
            let time = dt.and_utc().timestamp();
            let start = time.div_euclid(1800) * 1800;
            let mut points = vec![];
            for r in 0..nr {
                let ix = (b * nt + t) * nr + r;
                let z = geo[ix * 3 + 2];
                let y = p[ix * 4 + 3];
                let ne = p[ix * 4];
                let e = error[ix * 4 + 3];
                if state[ix] == 0.
                    && z.is_finite()
                    && y.is_finite()
                    && y.abs() < 1800.
                    && ne.is_finite()
                    && ne >= 1e10
                {
                    points.push((z, y, if e.is_finite() { e } else { 0. }));
                }
            }
            points.sort_by(|a, b| a.0.total_cmp(&b.0));
            samples.entry((b, start)).or_default().push(points);
        }
    }
    let mut output = [vec![], vec![]];
    let mut paired_los = vec![];
    for row in ours["rows"].as_array().unwrap() {
        let start = row["start_utc_s"].as_i64().unwrap();
        let z = row["altitude_km"].as_f64().unwrap();
        let mut beams = [vec![], vec![]];
        let mut rows = vec![];
        let mut magnetic = vec![];
        let mut values = [vec![], vec![]];
        for b in row["beams"].as_array().unwrap() {
            let az = b["azimuth_deg"].as_f64().unwrap();
            let el = b["elevation_deg"].as_f64().unwrap();
            let beam = (0..nb)
                .min_by(|i, j| {
                    ((dirs[i * 2] - az).abs() + (dirs[i * 2 + 1] - el).abs())
                        .total_cmp(&((dirs[j * 2] - az).abs() + (dirs[j * 2 + 1] - el).abs()))
                })
                .unwrap();
            ensure!(
                (dirs[beam * 2] - az).abs() + (dirs[beam * 2 + 1] - el).abs() < 0.1,
                "unmatched beam"
            );
            let Some(profiles) = samples.get(&(beam, start)) else {
                continue;
            };
            let matched: Vec<_> = profiles.iter().filter_map(|p| interp(p, z)).collect();
            if matched.is_empty() {
                continue;
            }
            let y = mean(&matched.iter().map(|v| v.0).collect::<Vec<_>>());
            let supplied_error = mean(&matched.iter().map(|v| v.1).collect::<Vec<_>>());
            let our_y = b["los_m_s"].as_f64().unwrap();
            let g: drift::Geometry = serde_json::from_value(b["geometry"].clone())?;
            rows.push(g.enu_row);
            magnetic.push(g.magnetic_row);
            values[0].push(our_y);
            values[1].push(y);
            beams[0].push(b.clone());
            let mut lb = b.clone();
            lb["los_m_s"] = json!(y);
            lb["l2_samples"] = json!(matched.len());
            lb["l2_supplied_error_mean_m_s"] = json!(supplied_error);
            beams[1].push(lb);
            paired_los.push(json!({"start_utc_s":start,"altitude_km":z,"beam":beam,"ac_m_s":our_y,"l2_m_s":y,"l2_samples":matched.len(),"l2_supplied_error_mean_m_s":supplied_error}));
        }
        if rows.len() < 5 {
            continue;
        }
        for source in 0..2 {
            let enu = drift::invert(&rows, &values[source])?;
            let mag = drift::invert(&magnetic, &values[source])?;
            output[source].push(json!({"start_utc_s":start,"end_utc_s":start+1800,"altitude_km":z,"beams":beams[source],"geographic":enu,"magnetic":mag}));
        }
    }
    let root = Path::new(&a[3]);
    std::fs::create_dir_all(root)?;
    for (source, name) in ["matched-ac", "matched-l2"].iter().enumerate() {
        let v = json!({"schema":"syisr-l2-matched-vector-v1","rows":output[source],"metadata":{"source":name,"time":"Same30-minuteUTCwindows asAC. L2successful one-minuteLOSvelocities linearlyinterpolatedinheight(maxbracket60km,noextrapolation)thenarithmeticallyaveragedwithineachwindow.","geometry":"Identicalbeamintersectionandgeographic/magneticLOSrowsforeachpairedtime/height. SameIRLScodeandconsistencycriteria.","qualification":"L2containsLOSparameters,nosuppliedvectorproduct. TheseL2vectorsarederivedhere. MeanofL2parametersdiffersfromfittingintegratedrawACdata; L2rawscreeningcannotberedonefromtheL2file. LOSispositivetowardradarperZhang2024Section2.2; invertagainstnegativeoutwardray. Independenthardwarepolaritynotcalibrated.","uncertainty":"SuppliedL2Errorrecordedasdiagnostic,notusedascalibratedACuncertainty or forinversevarianceweighting."}});
        campaign::save(&root.join(format!("{name}.h5")), &v)?;
    }
    campaign::save(&root.join("matched-los.h5"), &json!({"pairs":paired_los}))?;
    println!(
        "paired vectors={} LOSpairs={}",
        output[0].len(),
        paired_los.len()
    );
    Ok(())
}
#[cfg(test)]
mod tests {
    use super::*;
    #[test]
    fn interpolates_only_supported_height_intervals() {
        let p = vec![(300., 10., 2.), (330., 40., 4.), (450., 80., 10.)];
        assert_eq!(interp(&p, 315.), Some((25., 3.)));
        assert!(interp(&p, 280.).is_none());
        assert!(interp(&p, 400.).is_none());
        assert!(interp(&p, 500.).is_none());
    }
}
