//! Audit the recovered BEATA DDF definition, keeping its exact ADC-grid impulse.
use anyhow::{ensure, Result};
use serde_json::json;
fn convolve(a: &[f64], b: &[f64]) -> Vec<f64> {
    let mut out = vec![0.; a.len() + b.len() - 1];
    for (i, x) in a.iter().enumerate() {
        for (j, y) in b.iter().enumerate() {
            out[i + j] += x * y;
        }
    }
    out
}
fn main() -> Result<()> {
    let a: Vec<_> = std::env::args().collect();
    ensure!(a.len() == 3, "FIR_FILE OUTPUT.json");
    let text = std::fs::read_to_string(&a[1])?;
    let mut fields = std::collections::BTreeMap::new();
    let mut taps = std::collections::BTreeMap::new();
    for line in text.lines() {
        let tokens: Vec<_> = line.split('%').next().unwrap().split_whitespace().collect();
        if tokens.len() == 2 {
            fields.insert(tokens[0], tokens[1]);
        } else if tokens.len() == 3 && tokens[0] == "TAP" {
            taps.insert(
                tokens[1].parse::<usize>()?,
                u64::from_str_radix(tokens[2].trim_start_matches("0x"), 16)? as f64,
            );
        }
    }
    for (key, value) in [
        ("H_STAGES", "5"),
        ("H_DRATE", "14"),
        ("H_BYP", "0"),
        ("F_TAPS", "28"),
        ("F_ESYM", "1"),
        ("F_DRATE", "9"),
        ("F_BYP", "0"),
        ("F_0AD", "0"),
    ] {
        ensure!(fields.get(key) == Some(&value), "unsupported {key}");
    }
    ensure!(
        taps.len() == 15 && (0..15).all(|i| taps.contains_key(&i)),
        "incomplete symmetric FIR"
    );
    let mut high = vec![1.];
    for _ in 0..5 {
        high = convolve(&high, &vec![1. / 15.; 15]);
    }
    let mut low: Vec<_> = (0..29).map(|i| taps[&i.min(28 - i)]).collect();
    let sum: f64 = low.iter().sum();
    for v in &mut low {
        *v /= sum;
    }
    let mut up = vec![0.; 421];
    for (i, &v) in low.iter().enumerate() {
        up[15 * i] = v;
    }
    let impulse = convolve(&high, &up);
    let fs = 15e6;
    let neb = fs * impulse.iter().map(|x| x * x).sum::<f64>();
    let centroid = impulse
        .iter()
        .enumerate()
        .map(|(i, h)| i as f64 * h / fs)
        .sum::<f64>();
    let mut noise = vec![];
    for lag in 0..6 {
        let k = lag * 150;
        noise.push(if k < impulse.len() {
            impulse[..impulse.len() - k]
                .iter()
                .zip(&impulse[k..])
                .map(|(x, y)| x * y)
                .sum::<f64>()
                / (neb / fs)
        } else {
            0.
        });
    }
    let gain = |f: f64| {
        let (re, im) = impulse
            .iter()
            .enumerate()
            .fold((0., 0.), |(re, im), (i, h)| {
                let phase = std::f64::consts::TAU * f * i as f64 / fs;
                (re + h * phase.cos(), im + h * phase.sin())
            });
        re * re + im * im
    };
    let (mut lo, mut hi) = (0., 100000.);
    for _ in 0..60 {
        let f = (lo + hi) / 2.;
        if gain(f) > 0.5 {
            lo = f
        } else {
            hi = f
        }
    }
    let half_power = (lo + hi) / 2.;
    ensure!(
        (neb / 73500. - 1.).abs() < 0.01,
        "noise bandwidth inconsistent with definition: {neb}"
    );
    ensure!(
        (half_power / 34700. - 1.).abs() < 0.01,
        "3dB bandwidth inconsistent: {half_power}"
    );
    let out = json!({"source":a[1],"status":"definition_audit_not_reconstruction_acceptance","input_sampling_hz":fs,"output_decimation":150,"output_sampling_hz":100000.,"normalized_adc_impulse":impulse,"noise_equivalent_bandwidth_hz":neb,"half_power_frequency_hz":half_power,"group_delay_s":centroid,"receiver_voltage_correlation_at_10us_lags":noise,"caution":"Filter is from the recovered 2024 bundle, whose available experiment definitions match 2022. ADC/decimation phase and absolute range alignment must be checked against the 2022 ambiguity functions; group delay must not be applied twice."});
    std::fs::write(&a[2], serde_json::to_vec(&out)?)?;
    println!(
        "NEB={neb:.3} Hz, half-power={half_power:.3} Hz, group delay={:.6} us",
        centroid * 1e6
    );
    println!("noise correlation: {noise:?}");
    Ok(())
}
