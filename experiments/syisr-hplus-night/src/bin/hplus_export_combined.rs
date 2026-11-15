use anyhow::Result;
use serde_json::Value;
use std::{fs, str::FromStr};
fn main() -> Result<()> {
    let a: Vec<_> = std::env::args().collect();
    let s = fs::read(&a[1])?;
    let p: Value = serde_json::from_slice(&s)?;
    let out = hdf5::File::create(&a[2])?;
    let night = p["night"].as_array().unwrap();
    let nz = p["primary"]["knots"].as_array().unwrap().len();
    for key in [
        "h_fraction",
        "he_fraction",
        "o_fraction",
        "ti_k",
        "te_k",
        "power_adu2",
    ] {
        let v: Vec<_> = night
            .iter()
            .flat_map(|h| {
                h["best"]["nodes"]
                    .as_array()
                    .unwrap()
                    .iter()
                    .map(move |n| n[key].as_f64().unwrap() as f32)
            })
            .collect();
        let d = out
            .new_dataset::<f32>()
            .shape([night.len(), nz])
            .create(key)?;
        d.write_raw(&v)?;
        let u = if key.ends_with("_k") {
            "K"
        } else if key.ends_with("adu2") {
            "ADU^2"
        } else {
            "1"
        };
        d.new_attr::<hdf5::types::VarLenUnicode>()
            .create("units")?
            .write_scalar(&hdf5::types::VarLenUnicode::from_str(u)?)?;
    }
    let flags: Vec<_> = night
        .iter()
        .flat_map(|h| {
            h["quality_rejected"]
                .as_array()
                .unwrap()
                .iter()
                .map(|v| u8::from(v.as_bool().unwrap()))
        })
        .collect();
    out.new_dataset::<u8>()
        .shape([night.len(), nz])
        .create("quality_rejected")?
        .write_raw(&flags)?;
    let starts: Vec<_> = night
        .iter()
        .map(|h| 1700006400. + h["hour_utc"].as_f64().unwrap() * 3600.)
        .collect();
    let ends: Vec<_> = starts.iter().map(|t| t + 3600.).collect();
    let heights: Vec<_> = p["primary"]["knots"]
        .as_array()
        .unwrap()
        .iter()
        .map(|v| v.as_f64().unwrap() + 0.05)
        .collect();
    for (name, u, v) in [
        ("time_start_unix_s", "s since 1970-01-01T00:00:00Z", starts),
        ("time_end_unix_s", "s since 1970-01-01T00:00:00Z", ends),
        ("altitude_km", "km", heights),
    ] {
        let d = out.new_dataset_builder().with_data(&v).create(name)?;
        d.new_attr::<hdf5::types::VarLenUnicode>()
            .create("units")?
            .write_scalar(&hdf5::types::VarLenUnicode::from_str(u)?)?;
    }
    let profiles = p["profiles"].as_array().unwrap();
    let rows: Vec<f32> = profiles
        .iter()
        .flat_map(|r| {
            [
                r["knot"].as_f64().unwrap() as f32,
                r["h_fraction"].as_f64().unwrap() as f32,
                (r["chi_square"].as_f64().unwrap()
                    - p["primary"]["best"]["chi_square"].as_f64().unwrap()) as f32,
            ]
        })
        .collect();
    out.new_dataset::<f32>()
        .shape([profiles.len(), 3])
        .create("profile_knot_h_fraction_delta_Q")?
        .write_raw(&rows)?;
    out.new_dataset_builder()
        .with_data(&s)
        .deflate(4)
        .create("metadata_json_utf8")?;
    println!(
        "Exported {} x {} cells, {} profile points",
        night.len(),
        nz,
        profiles.len()
    );
    Ok(())
}
