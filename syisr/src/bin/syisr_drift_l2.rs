//! Match AC spline velocities to supplied L2; agreement checks relative polarity only.
use anyhow::{ensure, Result};
use chrono::NaiveDateTime;
use hdf5::types::FixedAscii;
use serde_json::{json, Value};
use std::{collections::BTreeMap, path::Path};
use syisr_ac::{campaign, spline::Spline};
fn main() -> Result<()> {
    let a: Vec<_> = std::env::args().collect();
    ensure!(a.len() == 4, "SNAPSHOT L2 OUTPUT");
    let f = hdf5::File::open(&a[2])?;
    let sh = f.dataset("Fittedparameter")?.shape();
    let (nb, nt, nr) = (sh[0], sh[1], sh[2]);
    let p = f.dataset("Fittedparameter")?.read_raw::<f64>()?;
    let st = f.dataset("Fitstate")?.read_raw::<f64>()?;
    let geo = f.dataset("GEO")?.read_raw::<f64>()?;
    let dirs = f.dataset("Beamdir")?.read_raw::<f64>()?;
    let ts = f.dataset("Time")?.read_raw::<FixedAscii<20>>()?;
    let mut lookup = BTreeMap::new();
    for path in campaign::files(&Path::new(&a[1]).join("fits"), "h5")? {
        let v: Value = campaign::load(&path)?;
        let az = v["job"]["az"].as_f64().unwrap();
        let el = v["job"]["el"].as_f64().unwrap();
        let b = (0..nb)
            .min_by(|i, j| {
                ((dirs[i * 2] - az).abs() + (dirs[i * 2 + 1] - el).abs())
                    .total_cmp(&((dirs[j * 2] - az).abs() + (dirs[j * 2 + 1] - el).abs()))
            })
            .unwrap();
        lookup.insert((b, v["job"]["start"].as_f64().unwrap() as i64), v);
    }
    let mut pairs = vec![];
    let mut per_beam = vec![];
    for b in 0..nb {
        let mut matched = vec![];
        for t in 0..nt {
            let Ok(time) =
                NaiveDateTime::parse_from_str(ts[b * nt + t].as_str().trim(), "%d-%b-%Y %H:%M:%S")
            else {
                continue;
            };
            let sec = time.and_utc().timestamp();
            let start = sec.div_euclid(1800) * 1800;
            let Some(v) = lookup.get(&(b, start)) else {
                continue;
            };
            let s: Spline = serde_json::from_value(v["spline"].clone())?;
            let n = v["control_nodes"].as_array().unwrap();
            for r in 0..nr {
                let ix = (b * nt + t) * nr + r;
                let z = geo[ix * 3 + 2];
                let vl = p[ix * 4 + 3];
                if !(300.0..=600.).contains(&z)
                    || st[ix] != 0.
                    || !vl.is_finite()
                    || vl.abs() > 500.
                {
                    continue;
                }
                let ac = s
                    .weights(z)
                    .iter()
                    .map(|(k, w)| w * n[*k]["velocity_m_s"].as_f64().unwrap())
                    .sum::<f64>();
                if ac.abs() > 500. {
                    continue;
                }
                matched.push([ac, vl]);
                pairs.push(json!({"beam":b,"utc_s":sec,"altitude_km":z,"ac_m_s":ac,"l2_m_s":vl}));
            }
        }
        if matched.len() > 5 {
            let n = matched.len() as f64;
            let ma = matched.iter().map(|p| p[0]).sum::<f64>() / n;
            let ml = matched.iter().map(|p| p[1]).sum::<f64>() / n;
            let corr = matched
                .iter()
                .map(|p| (p[0] - ma) * (p[1] - ml))
                .sum::<f64>()
                / (matched.iter().map(|p| (p[0] - ma).powi(2)).sum::<f64>()
                    * matched.iter().map(|p| (p[1] - ml).powi(2)).sum::<f64>())
                .sqrt();
            let same = (matched.iter().map(|p| (p[0] - p[1]).powi(2)).sum::<f64>() / n).sqrt();
            let reverse = (matched.iter().map(|p| (p[0] + p[1]).powi(2)).sum::<f64>() / n).sqrt();
            per_beam.push(json!({"beam":b,"azimuth_deg":dirs[b*2],"elevation_deg":dirs[b*2+1],"pairs":n,"correlation":corr,"same_sign_rms_m_s":same,"opposite_sign_rms_m_s":reverse}));
        }
    }
    let v = json!({"schema":"syisr-ac-l2-velocity-polarity-v1","pairs":pairs,"per_beam":per_beam,"qualification":"Matched successful supplied L2 samples,300-600km,inside the same30-minute AC windows. This checks relative velocity convention; it does not independently calibrate instrument polarity or correct for asynchronous dwell sampling."});
    campaign::save(Path::new(&a[3]), &v)?;
    println!("{}", serde_json::to_string_pretty(&v["per_beam"])?);
    Ok(())
}
