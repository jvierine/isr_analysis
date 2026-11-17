//! Record exact partial-product support and correct descriptive metadata in existing products.
use anyhow::{ensure, Result};
fn padded_attr(f: &hdf5::File, name: &str, text: &[u8]) -> Result<()> {
    let a = f.attr(name)?;
    ensure!(
        text.len() <= a.size(),
        "metadata replacement exceeds original storage"
    );
    let mut p = text.to_vec();
    p.resize(a.size(), b' ');
    a.write_raw(&p)?;
    Ok(())
}
fn main() -> Result<()> {
    let a: Vec<String> = std::env::args().collect();
    ensure!(a.len() == 2, "audit-partial-support campaign-root");
    for t in 0..48 {
        let path = format!("{}/windows/{t:03}", a[1]);
        let p = hdf5::File::open_rw(format!("{path}/partial.h5"))?;
        let text="Recorded stable pairs k=max(30,i-l)..i+20; actual pair count; original pulse mask and block highest-gate subtraction; 200/(200-l) scaling.";
        padded_attr(&p, "estimator", text.as_bytes())?;
        let f = hdf5::File::open_rw(format!("{path}/profile.h5"))?;
        let attr = f.attr("metadata_json")?;
        let mut m: serde_json::Value = serde_json::from_slice(&attr.read_raw::<u8>()?)?;
        ensure!(
            m["fit"]["status"] == "converged",
            "window {t} did not converge"
        );
        m["partial_products"]["selection"] = serde_json::json!(
            "200..425 by25 +450..1450 by50 km; stable samples >=30; actual-pair normalization"
        );
        padded_attr(&f, "metadata_json", &serde_json::to_vec(&m)?)?;
        let r = f.dataset("observed_slant_range_km")?.read_raw::<f64>()?;
        let native: Vec<u64> = r
            .iter()
            .map(|r| (r / 1.49896229 - 110.6).round() as u64)
            .collect();
        let counts: Vec<u64> = native
            .iter()
            .flat_map(|&i| {
                (1usize..61).map(move |l| {
                    let lo = (i as usize).saturating_sub(l).max(30);
                    (i as usize + 21 - lo) as u64
                })
            })
            .collect();
        ensure!(counts.iter().all(|&n| n > 0), "empty lag product");
        if f.dataset("observed_native_gate_index").is_err() {
            f.new_dataset_builder()
                .with_data(&native)
                .create("observed_native_gate_index")?;
            f.new_dataset::<u64>()
                .shape((r.len(), 60))
                .create("observed_pair_count")?
                .write_raw(&counts)?;
        }
        for (&i, &rr) in native.iter().zip(&r) {
            ensure!(
                (syisr_long_pulse::ambiguity_audit::nominal_range(i as usize) - rr).abs() < 1e-7,
                "gate coordinate"
            );
        }
    }
    println!("48 windows: exact native indices and stable pair counts recorded; metadata audited");
    Ok(())
}
