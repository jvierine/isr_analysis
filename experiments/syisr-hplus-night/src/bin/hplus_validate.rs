use anyhow::{ensure, Result};
use num_complex::Complex64 as C;
use rayon::prelude::*;
use serde_json::json;
use std::fs;
use syisr_hplus_night::{
    retrieval::Observation,
    theory::Theory,
};
fn main() -> Result<()> {
    let args: Vec<_> = std::env::args().collect();
    let theory = Theory::new(700., 500.)?;
    let fine = Theory::new(700., 250.)?;
    let validation = theory.validate()?;
    let width = 480e-6;
    let taps = [
        (-1e-5, C::new(0.15, 0.)),
        (0., C::new(0.977241, 0.)),
        (1e-5, C::new(0.15, 0.)),
    ];
    let kernel: Vec<Vec<C>> = (0..41)
        .map(|l| {
            let mut k = isr_ion::pulse_response::rectangular_cell(
                0.005,
                l as f64 * 1e-5,
                width,
                [0., 0.01],
                &taps,
                &theory.f,
            )
            .unwrap();
            let n = k.len();
            for (i, k) in k.iter_mut().enumerate() {
                *k *= 500. * if i == 0 || i + 1 == n { 0.5 } else { 1. };
            }
            k
        })
        .collect();
    let cases = [0., 0.1, 0.4, 0.8]
        .par_iter()
        .map(|&h| {
            let s = theory.spectrum(900., 1500., 75., h, 0.1, 1e11);
            let value: Vec<C> = kernel
                .iter()
                .map(|k| k.iter().zip(&s).map(|(k, s)| k * s * 200.).sum())
                .collect();
            let observation = Observation {
                value,
                sigma: vec![[0.2; 2]; 41],
                kernel: kernel.clone(),
                rho: (0..41)
                    .map(|l| C::new(if l == 0 { 1. } else { 0. }, 0.))
                    .collect(),
                background_sigma: 0.2,
                lag_min: 0,
            };
            let (best, _) = observation.solve(&theory, 1e11);
            ensure!(
                (best.h_fraction - h).abs() < 0.005,
                "H recovery failed truth {h}, fit {}",
                best.h_fraction
            );
            ensure!(
                (best.ti_k / 900. - 1.).abs() < 0.01 && (best.te_k / 1500. - 1.).abs() < 0.01,
                "temperature recovery failed"
            );
            Ok(json!({"truth_h":h,"truth_he":(1.-h)*0.1,"fit":best}))
        })
        .collect::<Result<Vec<_>>>()?;
    let a = theory.spectrum(900., 1500., 75., 0.4, 0.1, 1e11);
    let b = fine.spectrum(900., 1500., 75., 0.4, 0.1, 1e11);
    let max = a
        .iter()
        .enumerate()
        .map(|(j, v)| (v - b[j * 2]).abs())
        .fold(0., f64::max)
        / a.iter().copied().fold(0., f64::max);
    ensure!(max < 0.001, "frequency grid did not converge");
    fs::write(
        &args[1],
        serde_json::to_vec_pretty(
            &json!({"theory":validation,"synthetic_recovery":cases,"frequency_grid_500_vs_250_hz_peak_relative_error":max,"scope":"Noiseless independent mixture-parameter recovery through a known filtered finite LP response. Does not validate measured receiver/background or observational confidence."}),
        )?,
    )?;
    println!("mixture recovery and response validation passed");
    Ok(())
}
