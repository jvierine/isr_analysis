use anyhow::{ensure, Result};
use serde_json::{json, Value};
use std::fs;
fn main() -> Result<()> {
    let a: Vec<_> = std::env::args().collect();
    ensure!(a.len() == 3, "products_dir output.h5");
    let start = chrono::DateTime::parse_from_rfc3339("2023-11-15T15:00:00Z")?.timestamp() as f64;
    let end = start + 6. * 3600.;
    let mut metadata = vec![];
    let mut arrays: [Vec<f32>; 4] = std::array::from_fn(|_| vec![]);
    let names = [
        "acf_real_adu2",
        "acf_imag_adu2",
        "standard_error_real_adu2",
        "standard_error_imag_adu2",
    ];
    let mut paths: Vec<_> = fs::read_dir(&a[1])?
        .filter_map(|e| e.ok().map(|e| e.path()))
        .filter(|p| p.extension().is_some_and(|x| x == "json"))
        .collect();
    paths.sort();
    for p in paths {
        let d: Value = serde_json::from_slice(&fs::read(&p)?)?;
        let mut selected = vec![];
        for w in d["windows"].as_array().unwrap() {
            let t = w["header_clock_start_s"].as_f64().unwrap() - 28800.;
            if t >= start
                && t < end
                && w["elevation_deg"].as_f64().unwrap() > 89.
                && w["pulse_count"].as_u64().unwrap() > 1
            {
                selected.push((t, w));
            }
        }
        if selected.is_empty() {
            continue;
        }
        let h = hdf5::File::open(p.with_extension("h5"))?;
        for (t, w) in selected {
            let g = h.group(w["group"].as_str().unwrap())?;
            for (i, n) in names.iter().enumerate() {
                let data = g.dataset(n)?.read_raw::<f32>()?;
                ensure!(data.len() == 41 * 1324, "unexpected lag product dimensions");
                arrays[i].extend(data);
            }
            metadata.push(json!({"utc_start_s":t,"pulse_count":w["pulse_count"],"window":w,"header":d["header"],"source_raw":d["source"],"source_product":p,"group":w["group"]}));
        }
    }
    ensure!(!metadata.is_empty(), "no nighttime zenith windows");
    let out = hdf5::File::create(&a[2])?;
    for (i, n) in names.iter().enumerate() {
        out.new_dataset::<f32>()
            .shape([metadata.len(), 41, 1324])
            .chunk([1, 41, 1324])
            .shuffle()
            .deflate(4)
            .create(*n)?
            .write_raw(&arrays[i])?;
    }
    out.new_dataset_builder().with_data(&serde_json::to_vec(&json!({"schema":"syisr-hplus-night-raw-products-v1","windows":metadata,"clock":"UTC = raw header calendar - 8 hours; previously validated SYISR convention","selection":"monostatic SY zenith, 2023-11-15 15:00--21:00 UTC; approximately 22:18--04:18 local solar time","lag_s":(0..41).map(|i|i as f64*1e-5).collect::<Vec<_>>(),"range":"receive sample index; 767 us gate front, 480 us pulse; not altitude","processing":"copy of existing independent Rust per-pulse lag-product means and SEMs, no supplied L2 input; no additional lag removal"}))?).create("metadata_json_utf8")?;
    println!("extracted {} zenith windows to {}", metadata.len(), a[2]);
    Ok(())
}
