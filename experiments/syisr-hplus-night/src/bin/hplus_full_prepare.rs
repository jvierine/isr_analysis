//! Stream full-campaign products, pool and construct all-lag fit inputs in Rust.
use anyhow::{ensure, Result};
use serde_json::{json, Value};
use std::{
    collections::BTreeMap,
    fs::{self, File},
    io::{BufReader, Read},
    path::Path,
    process::Command,
};
struct Record {
    meta: Value,
    counts: Vec<u32>,
    arrays: Vec<Vec<f32>>,
}
fn u32read(r: &mut impl Read) -> Result<u32> {
    let mut b = [0; 4];
    r.read_exact(&mut b)?;
    Ok(u32::from_le_bytes(b))
}
fn main() -> Result<()> {
    let a: Vec<_> = std::env::args().collect();
    ensure!(
        a.len() == 4,
        "cleaned-products.bin output-directory prepare-binary"
    );
    let out = Path::new(&a[2]);
    fs::create_dir_all(out)?;
    let mut f = BufReader::new(File::open(&a[1])?);
    let mut magic = [0; 8];
    f.read_exact(&mut magic)?;
    ensure!(&magic == b"SYLPC001", "wrong format");
    let n = u32read(&mut f)? as usize;
    let mut bytes = vec![0; n];
    f.read_exact(&mut bytes)?;
    let config: Value = serde_json::from_slice(&bytes)?;
    ensure!(config["lags"] == 41, "requires all 41 lags");
    let samples: Vec<usize> = serde_json::from_value(config["samples"].clone())?;
    let size = 41 * samples.len();
    let mut groups: BTreeMap<(i64, u64, u64), Vec<Record>> = BTreeMap::new();
    loop {
        let mut n = [0; 4];
        let got = f.read(&mut n)?;
        if got == 0 {
            break;
        };
        ensure!(got == 4, "truncated record");
        let mut bytes = vec![0; u32::from_le_bytes(n) as usize];
        f.read_exact(&mut bytes)?;
        let meta: Value = serde_json::from_slice(&bytes)?;
        let mut bytes = vec![0; size * 20];
        f.read_exact(&mut bytes)?;
        let counts = bytes[..size * 4]
            .chunks_exact(4)
            .map(|b| u32::from_le_bytes(b.try_into().unwrap()))
            .collect();
        let arrays = (0..4)
            .map(|q| {
                bytes[(q + 1) * size * 4..(q + 2) * size * 4]
                    .chunks_exact(4)
                    .map(|b| f32::from_le_bytes(b.try_into().unwrap()))
                    .collect()
            })
            .collect();
        let az = meta["azimuth_raw"].as_u64().unwrap();
        let el = meta["elevation_raw"].as_u64().unwrap();
        let txaz = meta["tx_azimuth_raw"].as_i64().unwrap();
        let txel = meta["tx_elevation_raw"].as_i64().unwrap();
        if meta["pulses"].as_u64().unwrap() < 64
            || ((az as i64 - txaz + 32768).rem_euclid(65536) - 32768).abs() > 2
            || (el as i64 - txel).abs() > 2
        {
            continue;
        }
        let bin = (meta["start_utc_s"].as_f64().unwrap() / 1800.).floor() as i64;
        groups.entry((bin, az, el)).or_default().push(Record {
            meta,
            counts,
            arrays,
        });
    }
    let temp = out.join("pooling-temporary.h5");
    let mut jobs = vec![];
    let mut coverage = vec![];
    for ((bin, az, el), records) in groups {
        let pulses: u64 = records
            .iter()
            .map(|r| r.meta["pulses"].as_u64().unwrap())
            .sum();
        let start = bin * 1800;
        let id = format!("{start}_az{az}_el{el}");
        if pulses < 1000 {
            coverage.push(json!({"id":id,"status":"insufficient_pulses","pulses":pulses}));
            continue;
        }
        let nw = records.len();
        let n = 1324;
        let len = nw * 41 * n;
        let mut counts = vec![0u32; len];
        let mut arrays = vec![vec![0f32; len]; 4];
        let mut windows = vec![];
        for (w, r) in records.iter().enumerate() {
            for l in 0..41 {
                for (g, &k) in samples.iter().enumerate() {
                    let j = w * 41 * n + l * n + k;
                    counts[j] = r.counts[l * samples.len() + g];
                    for q in 0..4 {
                        arrays[q][j] = r.arrays[q][l * samples.len() + g];
                    }
                }
            }
            windows
                .push(json!({"utc_start_s":r.meta["start_utc_s"],"pulse_count":r.meta["pulses"]}));
        }
        {
            let h = hdf5::File::create(&temp)?;
            let names = [
                "acf_real_adu2",
                "acf_imag_adu2",
                "standard_error_real_adu2",
                "standard_error_imag_adu2",
            ];
            for (name, data) in names.iter().zip(&arrays) {
                h.new_dataset::<f32>()
                    .shape((nw, 41, n))
                    .create(*name)?
                    .write_raw(data)?;
            }
            h.new_dataset::<u32>()
                .shape((nw, 41, n))
                .create("pair_count")?
                .write_raw(&counts)?;
            let meta = serde_json::to_vec(
                &json!({"windows":windows,"source":a[1],"configuration":config}),
            )?;
            h.new_dataset_builder()
                .with_data(&meta)
                .create("metadata_json_utf8")?;
        }
        drop(arrays);
        drop(counts);
        let file = format!("{id}.json");
        let output = out.join(&file);
        let status = Command::new(&a[3])
            .arg(&temp)
            .arg(&output)
            .args(["taper", "30", &(bin.rem_euclid(2)).to_string()])
            .status()?;
        if !status.success() {
            coverage.push(json!({"id":id,"status":"preparation_failed"}));
            continue;
        }
        let mut input: Value = serde_json::from_slice(&fs::read(&output)?)?;
        let azdeg = az as f64 * 360. / 65536.;
        let eldeg = el as f64 * 360. / 65536.;
        let edges: Vec<f64> = serde_json::from_value(input["cell_edges_km"].clone())?;
        let valid: Vec<bool> = edges
            .windows(2)
            .map(|e| {
                syisr_hplus_night::theory::Theory::beam_field((e[0] + e[1]) / 2., azdeg, eldeg)
                    .map(|(_, alpha)| alpha > 3.)
                    .unwrap_or(false)
            })
            .collect();
        let mut best = (0usize, 0usize);
        let mut begin = 0;
        for end in 0..=valid.len() {
            if end == valid.len() || !valid[end] {
                if end - begin > best.1 - best.0 {
                    best = (begin, end);
                }
                begin = end + 1;
            }
        }
        let lo = edges[best.0];
        let hi = edges[best.1];
        if best != (0, valid.len()) {
            let knots: Vec<f64> = serde_json::from_value(input["knot_range_km"].clone())?;
            let ids: Vec<_> = knots
                .iter()
                .enumerate()
                .filter(|(_, r)| **r >= lo && **r <= hi)
                .map(|(i, _)| i)
                .collect();
            if ids.len() < 3 {
                coverage.push(json!({"id":id,"status":"unsupported_near_perpendicular_geometry","aspect_cutoff_deg":3}));
                fs::remove_file(output)?;
                continue;
            }
            input["knot_range_km"] = json!(ids.iter().map(|&i| knots[i]).collect::<Vec<_>>());
            for key in ["initial", "prescribed_node_fields"] {
                let old = input[key].as_array().unwrap().clone();
                input[key] = json!(ids.iter().map(|&i| old[i].clone()).collect::<Vec<_>>());
            }
            input["cell_edges_km"] = json!(&edges[best.0..=best.1]);
            let pp = input["products"]
                .as_array()
                .unwrap()
                .iter()
                .filter(|p| {
                    let t = 767e-6 + p["sample"].as_f64().unwrap() * 1e-5;
                    let tau = p["lag"].as_f64().unwrap() * 1e-5;
                    (t + tau - 480e-6 - 100e-6) * 149896.229 >= lo
                        && (t + 100e-6) * 149896.229 <= hi
                })
                .cloned()
                .collect::<Vec<_>>();
            input["products"] = json!(pp);
        }
        input["provenance"]["supported_range_km"] = json!([lo, hi]);
        input["provenance"]["geometry_validity"]=json!("Collisionless magnetized Maxwellian electrons: retain contiguous domain >3 deg from perpendicular; excluded domain is not fitted");
        let p = &mut input["provenance"];
        p["id"] = json!(id);
        p["azimuth_deg"] = json!(az as f64 * 360. / 65536.);
        p["elevation_deg"] = json!(el as f64 * 360. / 65536.);
        p["bin_start_utc_s"] = json!(start);
        p["bin_end_utc_s"] = json!(start + 1800);
        p["density_calibration"] =
            json!("Unavailable: power is ADU squared; prescribed Debye density is not measured Ne");
        p["source"] = json!(a[1]);
        p["background"]=json!("per-dwell complex receiver reference, samples 1200:1250; short-lag tapered FIR; common residual-background amplitude profiled");
        fs::write(&output, serde_json::to_vec(&input)?)?;
        jobs.push(json!({"id":id,"input":file,"bin_start_utc_s":start,"azimuth_deg":az as f64*360./65536.,"elevation_deg":el as f64*360./65536.}));
        coverage.push(json!({"id":id,"status":"prepared","pulses":pulses,"dwells":nw}));
    }
    fs::remove_file(temp)?;
    fs::write(out.join("jobs.json"), serde_json::to_vec_pretty(&jobs)?)?;
    fs::write(
        out.join("coverage.json"),
        serde_json::to_vec_pretty(&coverage)?,
    )?;
    println!("{} prepared", jobs.len());
    Ok(())
}
