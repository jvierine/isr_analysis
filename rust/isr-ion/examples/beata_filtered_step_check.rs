//! Deterministic subcycle mean audit, not a fit to the precipitation event.
use anyhow::{ensure, Result};
use serde_json::{json, Value};
fn main() -> Result<()> {
    let a: Vec<_> = std::env::args().collect();
    ensure!(a.len() == 3, "FILTERED_LAYER.json NEW_OUTPUT.json");
    ensure!(!std::path::Path::new(&a[2]).exists(), "output exists");
    let source: Value = serde_json::from_slice(&std::fs::read(&a[1])?)?;
    let means: Vec<Vec<[f64; 2]>> =
        serde_json::from_value(source["per_code_signal_mean_counts"].clone())?;
    ensure!(
        means.len() == 64 && means.iter().all(|x| x.len() >= 66),
        "invalid code means"
    );
    let ng = means[0].len();
    let mut tests = vec![];
    for onset in 0..64 {
        let sum: Vec<f64> = (0..ng)
            .map(|r| means[onset..].iter().map(|m| m[r][0]).sum())
            .collect();
        let mean_notch = sum[44..58].iter().sum::<f64>() / 14.;
        let min_notch = sum[44..58].iter().copied().fold(f64::INFINITY, f64::min);
        tests.push(json!({"onset_pulse":onset,"onset_fraction":onset as f64/64.,"mean_leakage_gates44_57":mean_notch,"minimum_leakage_gates44_57":min_notch,"mean_counts":sum}));
    }
    let mut ramps = vec![];
    for width in [4usize, 8, 16] {
        let mut best = (f64::INFINITY, 0usize, vec![]);
        for onset in 0..64 {
            let sum: Vec<f64> = (0..ng)
                .map(|r| {
                    means
                        .iter()
                        .enumerate()
                        .map(|(c, m)| {
                            let weight =
                                ((c as f64 + 0.5 - onset as f64) / width as f64).clamp(0., 1.);
                            weight * m[r][0]
                        })
                        .sum()
                })
                .collect();
            let value = sum[44..58].iter().sum::<f64>() / 14.;
            if value < best.0 {
                best = (value, onset, sum);
            }
        }
        println!(
            "Ramp {width} pulses: onset {}, mean notch leakage {}",
            best.1, best.0
        );
        ramps.push(json!({"rise_pulses":width,"rise_seconds":width as f64*0.35712/64.,"onset_pulse":best.1,"mean_leakage_gates44_57":best.0,"mean_counts":best.2}));
    }
    let worst = tests
        .iter()
        .min_by(|a, b| {
            a["mean_leakage_gates44_57"]
                .as_f64()
                .unwrap()
                .total_cmp(&b["mean_leakage_gates44_57"].as_f64().unwrap())
        })
        .unwrap();
    println!(
        "Worst mean notch-region leakage: onset {}, mean {}, minimum {}",
        worst["onset_pulse"], worst["mean_leakage_gates44_57"], worst["minimum_leakage_gates44_57"]
    );
    std::fs::write(
        &a[2],
        serde_json::to_vec_pretty(
            &json!({"status":"synthetic_filtered_subcycle_mean_only","source":a[1],"definition":"Unit brightness in source gates 16..35, zero elsewhere. Switches from zero to one before selected code pulse and remains on. Receiver noise mean is not scaled with signal. Decoder normalization remains that of the full 64-pulse dump.","limitations":["Only deterministic mean tested; no posterior inference or noise realization.","Same experimental ADC alignment as filtered covariance audit; archived marginal discretization still under review.","A synthetic artifact demonstrates a mechanism, not that this is the event explanation."],"tests":tests,"finite_rise_time_checks":ramps}),
        )?,
    )?;
    Ok(())
}
