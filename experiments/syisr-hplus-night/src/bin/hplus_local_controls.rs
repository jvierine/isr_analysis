use anyhow::Result;
use serde_json::json;
use std::fs;
use syisr_hplus_night::{data::Archive, theory::Theory};
fn main() -> Result<()> {
    let a: Vec<_> = std::env::args().collect();
    let archive = Archive::load(&a[1])?;
    let hour = archive.hour(16);
    let mut rows = vec![];
    for h in [725., 875.] {
        let theory = Theory::new(h, 500.)?;
        for receiver in ["full", "taper", "pedestal"] {
            let (obs, meta) = hour.observation_receiver(h, &theory, 1, 0, receiver)?;
            let (best, profile) = obs.solve(&theory, 1e11);
            rows.push(json!({"height_km":h,"receiver":receiver,"best":best,"profile_likelihood":profile,"metadata":meta}));
        }
    }
    fs::write(&a[2], serde_json::to_vec(&rows)?)?;
    Ok(())
}
