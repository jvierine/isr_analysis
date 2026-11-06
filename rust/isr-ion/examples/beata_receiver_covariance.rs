//! Receiver-only decoded covariance using the recovered DDF impulse response.
//! Diagnostic: excludes signal/self noise, and does not establish acceptance.
use anyhow::{ensure, Result};
use isr_ion::{beata, voltage, Complex as C};
use nalgebra::DMatrix;
use serde_json::{json, Value};
fn main() -> Result<()> {
    let a: Vec<_> = std::env::args().collect();
    ensure!(
        a.len() == 4,
        "COVARIANCE_INPUT.json RECEIVER_AUDIT.json NEW_OUTPUT.json"
    );
    ensure!(
        !std::path::Path::new(&a[3]).exists(),
        "output already exists"
    );
    let input: Value = serde_json::from_slice(&std::fs::read(&a[1])?)?;
    let receiver: Value = serde_json::from_slice(&std::fs::read(&a[2])?)?;
    let codes: Vec<Vec<f64>> = serde_json::from_value(input["codes"].clone())?;
    let observed: Vec<[f64; 2]> = serde_json::from_value(input["observed"].clone())?;
    let taps: Vec<f64> = serde_json::from_value(receiver["normalized_adc_impulse"].clone())?;
    let decimation = receiver["output_decimation"].as_u64().unwrap_or(0) as usize;
    let power = input["receiver_counts"].as_f64().unwrap_or(f64::NAN);
    ensure!(
        codes.len() == 64
            && !observed.is_empty()
            && observed.len() <= 431
            && decimation > 0
            && power.is_finite()
            && power > 0.,
        "invalid input"
    );
    ensure!(
        !taps.is_empty() && taps.iter().all(|v| v.is_finite()),
        "invalid FIR"
    );
    let energy: f64 = taps.iter().map(|v| v * v).sum();
    ensure!(energy > 0., "zero FIR");
    let ng = observed.len();
    let nv = ng + 1;
    let r = DMatrix::from_fn(nv, nv, |i, j| {
        let lag = i.abs_diff(j) * decimation;
        C::new(
            if lag < taps.len() {
                power
                    * taps[..taps.len() - lag]
                        .iter()
                        .zip(&taps[lag..])
                        .map(|(x, y)| x * y)
                        .sum::<f64>()
                    / energy
            } else {
                0.
            },
            0.,
        )
    });
    let white = DMatrix::from_diagonal_element(nv, nv, C::new(power, 0.));
    let mut covariance = DMatrix::<f64>::zeros(2 * ng, 2 * ng);
    let mut baseline = covariance.clone();
    let mut mean = vec![C::default(); ng];
    for code in &codes {
        let terms: Vec<_> = (0..ng)
            .map(|g| -> Result<_> {
                let mut t = beata::blocks()[0].terms(code, g)?;
                let norm = 64. * t.len() as f64;
                for v in &mut t {
                    v.weight /= norm;
                }
                Ok(t)
            })
            .collect::<Result<_>>()?;
        let (mu, cov) = voltage::decoded_moments(&r, &terms)?;
        covariance += cov;
        baseline += voltage::decoded_moments(&white, &terms)?.1;
        for (m, v) in mean.iter_mut().zip(mu) {
            *m += v;
        }
    }
    let mean_max = mean.iter().map(|v| v.norm()).fold(0., f64::max);
    ensure!(
        mean_max < 1e-10 * power,
        "non-cancelling receiver bias: {mean_max}"
    );
    ensure!(
        covariance.clone().cholesky().is_some(),
        "receiver covariance not positive definite"
    );
    let ratio: Vec<_> = (0..ng)
        .map(|i| {
            [
                covariance[(2 * i, 2 * i)] / baseline[(2 * i, 2 * i)],
                covariance[(2 * i + 1, 2 * i + 1)] / baseline[(2 * i + 1, 2 * i + 1)],
            ]
        })
        .collect();
    let adjacent: Vec<_> = (0..ng - 1)
        .map(|i| {
            [
                covariance[(2 * i, 2 * i + 2)]
                    / (covariance[(2 * i, 2 * i)] * covariance[(2 * i + 2, 2 * i + 2)]).sqrt(),
                covariance[(2 * i + 1, 2 * i + 3)]
                    / (covariance[(2 * i + 1, 2 * i + 1)] * covariance[(2 * i + 3, 2 * i + 3)])
                        .sqrt(),
            ]
        })
        .collect();
    let output = json!({"status":"receiver_only_diagnostic_not_reconstruction","input":a[1],"receiver_source":a[2],"receiver_power_counts":power,"maximum_cancelled_mean_counts":mean_max,"variance_ratio_to_white":ratio,"adjacent_gate_correlations":adjacent,"covariance_counts2":(0..2*ng).map(|i|(0..2*ng).map(|j|covariance[(i,j)]).collect::<Vec<_>>()).collect::<Vec<_>>(),"limitations":["Does not include signal self-noise or signal-noise cross terms.","Independent code pulses; no active clutter projection.","Stationary receiver noise requires no absolute range alignment."]});
    std::fs::write(&a[3], serde_json::to_vec(&output)?)?;
    println!("Receiver mean cancellation {mean_max:.3e}; variance ratios first {:?}, last {:?}; first adjacent correlations {:?}",ratio[0],ratio[ng-1],adjacent.first());
    Ok(())
}
