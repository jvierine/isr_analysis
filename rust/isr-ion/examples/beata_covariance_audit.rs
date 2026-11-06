//! Sensitivity diagnostic on the 10-us sample grid; NOT a receiver-filtered
//! production likelihood. Never use this to claim final reconstruction acceptance.
use anyhow::{ensure, Result};
use isr_ion::{beata, voltage, Complex as C};
use nalgebra::{DMatrix, DVector};
use serde_json::{json, Value};
fn main() -> Result<()> {
    let args: Vec<_> = std::env::args().collect();
    ensure!(args.len() == 4, "INPUT.json OUTPUT.json rho");
    let input: Value = serde_json::from_slice(&std::fs::read(&args[1])?)?;
    let rho: f64 = args[3].parse()?;
    ensure!(rho > 0. && rho < 1., "rho must be between zero and one");
    let codes: Vec<Vec<f64>> = serde_json::from_value(input["codes"].clone())?;
    let mut profile: Vec<f64> = serde_json::from_value(input["profile_counts"].clone())?;
    let bridge = std::env::var_os("RUISDAP_NOTCH_TEST").is_some();
    if bridge {
        ensure!(
            profile.len() > 63 && profile[39] > 0. && profile[63] > 0.,
            "invalid bridge endpoints"
        );
        for i in 40..63 {
            let u = (i - 39) as f64 / 24.;
            profile[i] = ((1. - u) * profile[39].ln() + u * profile[63].ln()).exp();
        }
    }
    let observed: Vec<[f64; 2]> = serde_json::from_value(input["observed"].clone())?;
    let receiver = input["receiver_counts"].as_f64().unwrap();
    let ng = observed.len();
    let nv = ng + 1;
    ensure!(
        codes.len() == 64 && profile.len() >= ng + 62,
        "incomplete pulse/profile"
    );
    let mut mean = vec![C::default(); ng];
    let mut covariance = DMatrix::<f64>::zeros(2 * ng, 2 * ng);
    for code in &codes {
        let tx: Vec<_> = code.iter().flat_map(|&c| [C::new(c, 0.); 2]).collect();
        let mut r = DMatrix::from_diagonal_element(nv, nv, C::new(receiver, 0.));
        for (s, &x) in profile.iter().enumerate() {
            let acf: Vec<_> = (0..nv)
                .map(|k| C::new(x / rho * rho.powi(k as i32), 0.))
                .collect();
            voltage::add_range_cell(&mut r, &tx, s as isize - 61, &acf)?;
        }
        let decoder = &beata::blocks()[0];
        let terms: Vec<_> = (0..ng)
            .map(|g| {
                let mut terms = decoder.terms(code, g).unwrap();
                let norm = 64. * terms.len() as f64;
                for t in &mut terms {
                    t.weight /= norm;
                }
                terms
            })
            .collect();
        let (mu, cov) = voltage::decoded_moments(&r, &terms)?;
        for i in 0..ng {
            mean[i] += mu[i];
        }
        covariance += cov;
    }
    let error = (0..ng)
        .map(|i| (mean[i].re - profile[i]).abs())
        .fold(0., f64::max);
    ensure!(
        error < 1e-9 * profile.iter().copied().fold(1., f64::max),
        "mean response mismatch {error}"
    );
    let residual = DVector::from_iterator(
        2 * ng,
        (0..ng).flat_map(|i| [observed[i][0] - mean[i].re, observed[i][1] - mean[i].im]),
    );
    let chol = covariance
        .clone()
        .cholesky()
        .ok_or_else(|| anyhow::anyhow!("non-PD covariance"))?;
    let q = residual.dot(&chol.solve(&residual));
    let logdet = 2. * chol.l().diagonal().iter().map(|v| v.ln()).sum::<f64>();
    let diagonal_q = (0..2 * ng)
        .map(|i| residual[i].powi(2) / covariance[(i, i)])
        .sum::<f64>();
    let correlations: Vec<_> = (35..65.min(ng))
        .map(|i| {
            (35..65.min(ng))
                .map(|j| {
                    covariance[(2 * i, 2 * j)]
                        / (covariance[(2 * i, 2 * i)] * covariance[(2 * j, 2 * j)]).sqrt()
                })
                .collect::<Vec<_>>()
        })
        .collect();
    let output = json!({"status":"diagnostic_only_not_accepted","assumptions":"Independent code pulses; stationary profile within 357.12 ms; rectangular 2-sample baud; exponential scattering ACF; white receiver voltage; no receiver FIR or clutter projection. Do not deploy as final reconstruction.","notch_bridge_hypothesis_only":bridge,"rho":rho,"input":args[1],"mean_identity_max_error":error,"full_covariance_chi_square":q,"diagonal_chi_square":diagonal_q,"gaussian_nll_without_constant":0.5*(q+logdet),"log_determinant":logdet,"dof":2*ng,"mean_counts":mean.iter().map(|v|[v.re,v.im]).collect::<Vec<_>>(),"standard_error_counts":(0..ng).map(|i|[covariance[(2*i,2*i)].sqrt(),covariance[(2*i+1,2*i+1)].sqrt()]).collect::<Vec<_>>(),"notch_range_real_correlation":correlations,"covariance_counts2":(0..2*ng).map(|i|(0..2*ng).map(|j|covariance[(i,j)]).collect::<Vec<_>>()).collect::<Vec<_>>()});
    std::fs::write(&args[2], serde_json::to_vec(&output)?)?;
    println!(
        "rho={rho} full chi2={q:.2} diagonal chi2={diagonal_q:.2} dimension={}",
        2 * ng
    );
    Ok(())
}
