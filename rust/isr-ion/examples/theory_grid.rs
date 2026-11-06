//! Independent-oracle input/output harness; no Python participates in rendering
//! or in the native forward model.
use isr_ion::spectrum::{PlasmaState, SpectrumModel};
use serde::{Deserialize, Serialize};
#[derive(Deserialize)]
struct Case {
    model: SpectrumModel,
    plasma: PlasmaState,
    frequency_hz: Vec<f64>,
}
#[derive(Serialize)]
struct Output {
    values: Vec<f64>,
}
fn main() -> anyhow::Result<()> {
    let input = std::env::args().nth(1).expect("JSON input path");
    let cases: Vec<Case> = serde_json::from_slice(&std::fs::read(input)?)?;
    let result: Vec<_> = cases
        .into_iter()
        .map(|c| {
            c.model
                .evaluate(c.plasma, &c.frequency_hz)
                .map(|values| Output { values })
        })
        .collect::<anyhow::Result<_>>()?;
    println!("{}", serde_json::to_string(&result)?);
    Ok(())
}
