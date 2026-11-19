//! Export fixed IRI reference shapes. IRI is an external empirical model; inversion is Rust.
use anyhow::{ensure, Result};
use chrono::{DateTime, Datelike, Timelike, Utc};
use std::{path::Path, process::Command};
use syisr_ac::{
    campaign::{self, Campaign},
    h_transition::Shape,
};
fn main() -> Result<()> {
    let a: Vec<_> = std::env::args().collect();
    ensure!(
        a.len() == 5,
        "IRI_DRIVER MODEL_DATA_DIRECTORY CAMPAIGN_H5 OUTPUT_DIRECTORY"
    );
    let c: Campaign = campaign::load(Path::new(&a[3]))?;
    let out = Path::new(&a[4]);
    std::fs::create_dir_all(out)?;
    let driver = std::fs::canonicalize(&a[1])?;
    for j in c.jobs {
        let t = DateTime::<Utc>::from_timestamp(((j.start + j.end) / 2.) as i64, 0).unwrap();
        let lat = 18.3492f64.to_radians();
        let lon = 109.6222f64.to_radians();
        let az = j.az.to_radians();
        let el = j.el.to_radians();
        let up = [lat.cos() * lon.cos(), lat.cos() * lon.sin(), lat.sin()];
        let north = [-lat.sin() * lon.cos(), -lat.sin() * lon.sin(), lat.cos()];
        let east = [-lon.sin(), lon.cos(), 0.];
        let radius = 6371.05f64;
        let distance = -radius * el.sin()
            + ((radius * el.sin()).powi(2) + (6371f64 + 900.).powi(2) - radius.powi(2)).sqrt();
        let pos: Vec<_> = (0..3)
            .map(|k| {
                radius * up[k]
                    + distance
                        * (el.sin() * up[k] + el.cos() * (az.cos() * north[k] + az.sin() * east[k]))
            })
            .collect();
        let glat = (pos[2] / pos.iter().map(|x| x * x).sum::<f64>().sqrt())
            .asin()
            .to_degrees();
        let glon = pos[1].atan2(pos[0]).to_degrees();
        let args = [
            t.year().to_string(),
            t.month().to_string(),
            t.day().to_string(),
            t.hour().to_string(),
            t.minute().to_string(),
            t.second().to_string(),
            glat.to_string(),
            glon.to_string(),
            "500".into(),
            "2000".into(),
            "25".into(),
            ".".into(),
        ];
        let output = Command::new(&driver)
            .args(args)
            .current_dir(&a[2])
            .output()?;
        ensure!(output.status.success(), "IRI failed for {}", j.id);
        let text = String::from_utf8(output.stdout)?;
        let mut z = vec![];
        let mut ti = vec![];
        let mut raw = vec![];
        let mut h = vec![];
        let mut last = 0f64;
        for line in text.lines().take(61) {
            let row: Vec<f64> = line
                .split_whitespace()
                .map(str::parse)
                .collect::<std::result::Result<_, _>>()?;
            ensure!(
                row.len() == 12 && row[3].is_finite() && row[3] > 100.,
                "invalid IRI ion temperature for {}",
                j.id
            );
            let g = 9.80665 * (6371. / (6371. + row[0])).powi(2);
            let height = 1.380649e-23 * row[3] / (15. * 1.66053906660e-27 * g) / 1000.;
            z.push(row[0]);
            ti.push(row[3]);
            raw.push(height);
            last = last.max(height);
            h.push(last);
        }
        let shape=Shape{altitude_km:z,scale_height_km:h,source:format!("Official IRI2020 fixed Ti/g differential H/O scale height, monotonic cumulative maximum; UTC {}, nominal900km beam footprint lat {}, lon {}; model data directory hash must be retained with the run",t,glat,glon)};
        ensure!(shape.altitude_km.len() == 61, "incomplete IRI output");
        shape.validate()?;
        let mut value = serde_json::to_value(shape)?;
        value["iri_ti_k"] = serde_json::to_value(ti)?;
        value["unmodified_scale_height_km"] = serde_json::to_value(raw)?;
        campaign::save(&out.join(format!("{}.h5", j.id)), &value)?;
    }
    Ok(())
}
