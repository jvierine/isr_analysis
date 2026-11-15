//! Document the neutral-collision scale behind the topside collisionless limit.
use anyhow::Result;
use serde_json::json;
use std::{f64::consts::PI, fs};
fn main() -> Result<()> {
    let a: Vec<_> = std::env::args().collect();
    let config: serde_json::Value = serde_json::from_slice(&fs::read(&a[1])?)?;
    let d = &config["2023-11-15"];
    let am = PI * (319. - 3.) / 182.6;
    let e = 0.01675;
    let radius = 1. - e * (am.cos() + e * ((2. * am).cos() - 1.) / 2.);
    let mut rows = vec![];
    for h in [400., 500., 600., 700., 800., 900., 1100., 1600.] {
        let input = rmsis::Input::consistent(
            319,
            16.5 * 3600.,
            h,
            18.3492,
            109.6222,
            d["f107_81day"].as_f64().unwrap() / radius.powi(2),
            d["f107_previous_day"].as_f64().unwrap() / radius.powi(2),
            rmsis::MagneticActivity::Daily {
                ap: d["ap_daily"].as_f64().unwrap(),
            },
        );
        let n = rmsis::Model::new().gtd7::<f64>(&input);
        let nu = 1.63e-16 * n.atomic_oxygen
            + 2.99e-16 * n.molecular_nitrogen
            + 4.28e-16 * n.molecular_oxygen;
        rows.push(json!({"height_km":h,"neutral_temperature_k":n.temperature,"nO_m3":n.atomic_oxygen,"nN2_m3":n.molecular_nitrogen,"nO2_m3":n.molecular_oxygen,"nH_m3":n.hydrogen,"reference_effective_ion_neutral_rate_s":nu,"rate_times_longest_lag":nu*400e-6}));
    }
    fs::write(
        &a[2],
        serde_json::to_vec_pretty(
            &json!({"model":"Rust NRLMSISE-00 fc707387617885de749dcc516c306a74227cf1b5","drivers":d,"flux_conversion":"archived IRI adjusted-to-1AU flux converted to observed flux at Earth using same eccentric-orbit approximation as shared atmosphere module","time_utc":"2023-11-15T16:30:00Z","rate_prescription":"Shared legacy effective molecular-ion rate: 1.63e-16*nO + 2.99e-16*nN2 + 4.28e-16*nO2, SI. Used only to check the neutral collision scale, not imposed as a species-resolved H/He/O collision operator.","scope":"Neutral-collision scale check supports a collisionless topside approximation; does not implement detailed Coulomb collisions or independently determine H-neutral collision cross sections.","profile":rows}),
        )?,
    )?;
    Ok(())
}
