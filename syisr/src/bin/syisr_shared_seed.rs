//! Construct temporal initialization from existing independent-beam fits, never new measurements.
use anyhow::{ensure, Result};
use std::{collections::BTreeMap, path::Path};
use syisr_ac::campaign::{self, Campaign};
fn main() -> Result<()> {
    let a: Vec<_> = std::env::args().collect();
    ensure!(a.len() == 4, "INPUT_ROOT INDEPENDENT_FITS OUTPUT");
    let input = Path::new(&a[1]);
    let old = Path::new(&a[2]);
    let out = Path::new(&a[3]);
    let c: Campaign = campaign::load(&input.join("campaign.h5"))?;
    let mut seeds = BTreeMap::new();
    for j in &c.jobs {
        if let Ok(v) = campaign::load::<serde_json::Value>(&old.join(format!("{}.h5", j.id))) {
            if v["h_transition"].is_object() {
                let e = seeds.entry(j.start as i64).or_insert((j.el, v.clone()));
                if j.el > e.0 {
                    *e = (j.el, v);
                }
            }
        }
    }
    ensure!(!seeds.is_empty(), "no composition seeds");
    let mut count = 0;
    for time in c
        .jobs
        .iter()
        .map(|j| j.start as i64)
        .collect::<std::collections::BTreeSet<_>>()
    {
        let (&t, (_, v)) = seeds.iter().min_by_key(|(t, _)| (*t - time).abs()).unwrap();
        campaign::save(
            &out.join("shared-windows").join(format!("{time}.h5")),
            &serde_json::json!({"start":time,"h_transition":v["h_transition"],"helium_layer":v["helium_layer"],"initialization_only":true,"source_time":t,"not_a_shared_fit":true}),
        )?;
        count += 1;
    }
    println!("INITIALIZATION_WINDOWS {count}");
    Ok(())
}
