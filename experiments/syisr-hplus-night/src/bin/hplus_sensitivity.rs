use anyhow::Result;
use serde_json::json;
use std::fs;
use syisr_hplus_night::{data::Archive, theory::Theory};
fn main() -> Result<()> {
    let a: Vec<_> = std::env::args().collect();
    let d = Archive::load(&a[1])?;
    let h = d.hour(16);
    let mut out = vec![];
    for ratio in [0.5, 1., 2.] {
        let mut t = Theory::new(700., 500.)?;
        t.h_temperature_ratio = ratio;
        let (o, m) = h.observation(700., &t, 1, 0)?;
        let (b, p) = o.solve(&t, 1e11);
        out.push(
            json!({"proton_to_oxygen_temperature_ratio":ratio,"metadata":m,"fit":b,"profile":p}),
        );
    }
    fs::write(&a[2], serde_json::to_vec_pretty(&out)?)?;
    Ok(())
}
