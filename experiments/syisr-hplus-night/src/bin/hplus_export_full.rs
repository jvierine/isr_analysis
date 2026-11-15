//! Compact float32 campaign cubes and small browser tiles; preserve missing calibration.
use anyhow::{ensure, Result};
use serde_json::{json, Value};
use std::{collections::BTreeSet, fs, path::Path, str::FromStr};
fn main() -> Result<()> {
    let a: Vec<_> = std::env::args().collect();
    ensure!(a.len() == 3, "combined-summary.json output-directory");
    let rows: Vec<Value> = serde_json::from_slice(&fs::read(&a[1])?)?;
    let out = Path::new(&a[2]);
    fs::create_dir_all(out.join("diagnostics"))?;
    let times: Vec<i64> = rows
        .iter()
        .filter_map(|r| r["provenance"]["bin_start_utc_s"].as_i64())
        .collect::<BTreeSet<_>>()
        .into_iter()
        .collect();
    let beams: Vec<(i64, i64)> = rows
        .iter()
        .filter_map(|r| {
            Some((
                (r["provenance"]["azimuth_deg"].as_f64()? * 1e6).round() as i64,
                (r["provenance"]["elevation_deg"].as_f64()? * 1e6).round() as i64,
            ))
        })
        .collect::<BTreeSet<_>>()
        .into_iter()
        .collect();
    let ranges: Vec<_> = (0..10).map(|i| 275. + 150. * i as f64).collect();
    let nb = beams.len();
    let nt = times.len();
    let nr = ranges.len();
    let size = nb * nt * nr;
    let names = [
        "electron_density_m3",
        "ion_temperature_k",
        "electron_temperature_k",
        "spectral_velocity_m_s",
        "h_fraction",
        "he_fraction",
        "power_adu2",
    ];
    let keys = [
        "UNAVAILABLE",
        "ti_k",
        "te_k",
        "velocity_m_s",
        "h_fraction",
        "he_fraction",
        "power_adu2",
    ];
    let units = ["m^-3", "K", "K", "m/s", "1", "1", "ADU^2"];
    let mut cubes = vec![vec![f32::NAN; size]; 7];
    let mut errors = vec![vec![f32::NAN; size]; 6];
    let mut flags = vec![2u8; size];
    let mut manifest_beams = vec![];
    let mut index = vec![json!(null); nb * nt];
    for r in &rows {
        let Some(az) = r["provenance"]["azimuth_deg"].as_f64() else {
            continue;
        };
        let el = r["provenance"]["elevation_deg"].as_f64().unwrap();
        let bi = beams
            .iter()
            .position(|b| *b == ((az * 1e6).round() as i64, (el * 1e6).round() as i64))
            .unwrap();
        let ti = times
            .iter()
            .position(|t| Some(*t) == r["provenance"]["bin_start_utc_s"].as_i64())
            .unwrap();
        let Some(nodes) = r["nodes"].as_array() else {
            continue;
        };
        let knots = r["knot_range_km"].as_array().unwrap();
        for (j, node) in nodes.iter().enumerate() {
            let range = knots[j].as_f64().unwrap();
            let Some(ri) = ranges.iter().position(|v| (*v - range).abs() < 1e-4) else {
                continue;
            };
            let idx = (bi * nt + ti) * nr + ri;
            let q = &r["node_quality"][j];
            flags[idx] = u8::from(!q["flags"].as_array().map(|x| x.is_empty()).unwrap_or(false));
            for p in 1..7 {
                cubes[p][idx] = node[keys[p]].as_f64().unwrap_or(f64::NAN) as f32;
            }
            for (p, key) in [
                "conditional_ti_standard_error",
                "conditional_te_standard_error",
                "conditional_velocity_standard_error",
                "conditional_h_fraction_error",
                "conditional_he_standard_error",
                "conditional_power_standard_error",
            ]
            .iter()
            .enumerate()
            {
                errors[p][idx] = q[*key].as_f64().unwrap_or(f64::NAN) as f32;
            }
        }
        let fit_path = Path::new(&a[1])
            .parent()
            .unwrap()
            .join("fits")
            .join(format!("{}.json", r["id"].as_str().unwrap()));
        let full: Value = serde_json::from_slice(&fs::read(fit_path)?)?;
        let runs = full["runs"].as_array().unwrap();
        let converged: Vec<_> = runs
            .iter()
            .filter(|x| x["status"].as_str().unwrap_or("").starts_with("converged"))
            .collect();
        let best = (if converged.is_empty() {
            runs.iter().collect::<Vec<_>>()
        } else {
            converged
        })
        .into_iter()
        .min_by(|x, y| {
            x["chi_square"]
                .as_f64()
                .unwrap()
                .total_cmp(&y["chi_square"].as_f64().unwrap())
        })
        .unwrap();
        let diagnostic = json!({"status":best["status"],"reduced_chi_square":best["reduced_chi_square"],"products":full["products"],"predicted":best["predicted"],"knot_range_km":full["knot_range_km"],"node_quality":r["node_quality"],"provenance":full["provenance"]});
        fs::write(
            out.join("diagnostics")
                .join(format!("{}.json", r["id"].as_str().unwrap())),
            serde_json::to_vec(&diagnostic)?,
        )?;
        index[bi * nt + ti] = json!({"id":r["id"],"file":format!("diagnostics/{}.json",r["id"].as_str().unwrap()),"status":r["status"],"chi_square_reduced":r["reduced_chi_square"]});
    }
    let h = hdf5::File::create(out.join("syisr-full-campaign.h5"))?;
    for p in 0..7 {
        let d = h
            .new_dataset::<f32>()
            .shape((nb, nt, nr))
            .create(names[p])?;
        d.write_raw(&cubes[p])?;
        d.new_attr::<hdf5::types::VarLenUnicode>()
            .create("units")?
            .write_scalar(&hdf5::types::VarLenUnicode::from_str(units[p])?)?;
    }
    h.new_dataset::<u8>()
        .shape((nb, nt, nr))
        .create("quality_flag")?
        .write_raw(&flags)?;
    for (i, name) in [
        "conditional_ti_standard_error_k",
        "conditional_te_standard_error_k",
        "conditional_velocity_standard_error_m_s",
        "conditional_h_fraction_standard_error",
        "conditional_he_fraction_standard_error",
        "conditional_power_standard_error_adu2",
    ]
    .iter()
    .enumerate()
    {
        h.new_dataset::<f32>()
            .shape((nb, nt, nr))
            .create(*name)?
            .write_raw(&errors[i])?;
    }
    let starts: Vec<f64> = times.iter().map(|t| *t as f64).collect();
    h.new_dataset_builder()
        .with_data(&starts)
        .create("time_start_unix_s")?;
    h.new_dataset_builder()
        .with_data(&times.iter().map(|t| (*t + 1800) as f64).collect::<Vec<_>>())
        .create("time_end_unix_s")?;
    h.new_dataset_builder()
        .with_data(&ranges)
        .create("slant_range_km")?;
    let mut altitudes = vec![];
    for (bi, &(az, el)) in beams.iter().enumerate() {
        let az = az as f64 / 1e6;
        let el = el as f64 / 1e6;
        let altitude: Vec<f64> = ranges
            .iter()
            .map(|r| {
                (6371.05_f64.powi(2) + r * r + 2. * 6371.05 * r * el.to_radians().sin()).sqrt()
                    - 6371.
            })
            .collect();
        altitudes.extend_from_slice(&altitude);
        let mut tile = vec![];
        for p in 0..8 {
            for ri in 0..nr {
                for ti in 0..nt {
                    let idx = (bi * nt + ti) * nr + ri;
                    let value = if p == 7 {
                        flags[idx] as f32
                    } else {
                        cubes[p][idx]
                    };
                    tile.extend_from_slice(&value.to_le_bytes());
                }
            }
        }
        let file = format!("beam-{bi:02}.f32");
        fs::write(out.join(&file), tile)?;
        manifest_beams.push(json!({"azimuth_deg":az,"elevation_deg":el,"altitude_km":altitude,"slant_range_km":ranges,"time_unix_s":times.iter().map(|t|*t as f64+900.).collect::<Vec<_>>(),"file":file,"shape":[8,nr,nt],"diagnostics":&index[bi*nt..(bi+1)*nt]}));
    }
    h.new_dataset::<f64>()
        .shape((nb, nr))
        .create("nominal_altitude_km")?
        .write_raw(&altitudes)?;
    for (name, values) in [
        (
            "beam_azimuth_deg",
            beams.iter().map(|x| x.0 as f64 / 1e6).collect::<Vec<_>>(),
        ),
        (
            "beam_elevation_deg",
            beams.iter().map(|x| x.1 as f64 / 1e6).collect::<Vec<_>>(),
        ),
    ] {
        h.new_dataset_builder().with_data(&values).create(name)?;
    }
    let provenance = json!({"schema":"syisr-full-campaign-v2","dimensions":["beam","time","range"],"density_calibration":"unavailable; electron_density_m3 is NaN, never the prescribed Debye-shape density","shape_density_m3":1e11,"velocity":"signed spectral velocity; physical Doppler sign not independently calibrated","range_resolution":"150 km continuous parameter knots; finite-pulse individual lag response; not independent 150 km cells","quality_flags":{"0":"conditional diagnostic accepted","1":"rejected or prescribed nuisance shape","2":"unsupported or unavailable"},"rows":rows});
    h.new_dataset_builder()
        .with_data(&serde_json::to_vec(&provenance)?)
        .deflate(4)
        .create("metadata_json_utf8")?;
    let manifest = json!({"schema":"syisr-full-campaign-v2","integration_s":1800,"density_calibrated":false,"beams":manifest_beams,"fields":names,"note":"Independent Rust H/He/O mixture fits to all 41 measured lags with pair-level nonthermal masks; conditional on receiver response and prescribed Debye density. Absolute Ne and physical Doppler sign are not calibrated."});
    fs::write(out.join("manifest.json"), serde_json::to_vec(&manifest)?)?;
    println!("{nb} beams x {nt} times x {nr} ranges");
    Ok(())
}
