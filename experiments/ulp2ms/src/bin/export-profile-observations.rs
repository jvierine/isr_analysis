//! Export only measured products for covariance training before any plasma fit.
use anyhow::{ensure, Result};
use std::{fs, path::Path};
fn main() -> Result<()> {
    let a: Vec<String> = std::env::args().collect();
    ensure!(
        a.len() == 2 || a.len() == 3,
        "export-profile-observations raw-product-root [window]"
    );
    let root = Path::new(&a[1]);
    let windows = if a.len() == 3 {
        vec![a[2].parse::<usize>()?]
    } else {
        (0..48).collect()
    };
    for t in windows {
        let dir = root.join(format!("windows/{t:03}"));
        let out = dir.join("observations.h5");
        if out.exists() {
            continue;
        }
        let old = hdf5::File::open(dir.join("lpm.h5"))?;
        let partial = hdf5::File::open(dir.join("partial.h5"))?;
        let merge = |key: &str| -> Result<Vec<f64>> {
            let mut v = partial.dataset(key)?.read_raw::<f64>()?;
            let b = old.dataset(key)?.read_raw::<f64>()?;
            v.extend_from_slice(&b[if key == "slant_range_km" { 6 } else { 6 * 61 }..]);
            Ok(v)
        };
        let ranges = merge("slant_range_km")?;
        let requested = syisr_long_pulse::profile::observation_ranges(true, 5000)?;
        let indices: Vec<usize> = requested
            .iter()
            .map(|&r| {
                ranges
                    .iter()
                    .enumerate()
                    .min_by(|a, b| (a.1 - r as f64).abs().total_cmp(&(b.1 - r as f64).abs()))
                    .unwrap()
                    .0
            })
            .collect();
        let f = hdf5::File::create(&out)?;
        let selected: Vec<f64> = indices.iter().map(|&i| ranges[i]).collect();
        f.new_dataset_builder()
            .with_data(&selected)
            .create("observed_slant_range_km")?;
        f.new_dataset_builder()
            .with_data(
                &selected
                    .iter()
                    .map(|r| (r / 1.49896229 - 110.6).round() as u64)
                    .collect::<Vec<_>>(),
            )
            .create("observed_native_gate_index")?;
        for (name, source, sem) in [
            ("observed_real_adu2", "acf_real_adu2", false),
            ("observed_imag_adu2", "acf_imag_adu2", false),
            ("sem_real_adu2", "acf_real_sem_adu2", true),
            ("sem_imag_adu2", "acf_imag_sem_adu2", true),
        ] {
            let v = merge(source)?;
            let values: Vec<f64> = indices
                .iter()
                .flat_map(|&i| (1..61).map(move |l| i * 61 + l))
                .map(|j| if sem { v[j].max(1.) } else { v[j] })
                .collect();
            f.new_dataset::<f64>()
                .shape((indices.len(), 60))
                .create(name)?
                .write_raw(&values)?;
        }
        let m = serde_json::json!({"product":"measured observations only; no plasma fit","provenance":{"lpm":dir.join("lpm.h5")},"partial_products":{"source":dir.join("partial.h5")}});
        f.new_attr_builder()
            .with_data(serde_json::to_string(&m)?.as_bytes())
            .create("metadata_json")?;
    }
    let plan: serde_json::Value = serde_json::from_slice(&fs::read(root.join("plan.json"))?)?;
    let w = plan["windows"].as_array().unwrap();
    let mut edges = vec![w[0]["utc_start"].as_f64().unwrap()];
    edges.extend(w.iter().map(|v| v["utc_end"].as_f64().unwrap()));
    let centers: Vec<f64> = w
        .iter()
        .map(|v| (v["utc_start"].as_f64().unwrap() + v["utc_end"].as_f64().unwrap()) / 2.)
        .collect();
    let h = hdf5::File::create(root.join("campaign.h5"))?;
    h.new_dataset_builder()
        .with_data(&edges)
        .create("time_edges_utc_s")?;
    h.new_dataset_builder()
        .with_data(&centers)
        .create("time_centres_utc_s")?;
    h.new_attr_builder()
        .with_data("Observation timeline only; individual window files hold measured products")
        .create("product")?;
    Ok(())
}
