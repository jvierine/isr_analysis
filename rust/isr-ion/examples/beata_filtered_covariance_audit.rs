//! Sensitivity diagnostic on the 10-us sample grid; NOT a receiver-filtered
//! production likelihood. Never use this to claim final reconstruction acceptance.
use anyhow::{ensure, Result};
use isr_ion::{beata, voltage, Complex as C};
use nalgebra::{DMatrix, DVector};
use rayon::prelude::*;
use serde_json::{json, Value};
fn main() -> Result<()> {
    let args: Vec<_> = std::env::args().collect();
    ensure!(
        args.len() == 6,
        "INPUT.json RECEIVER.json NEW_OUTPUT.json rho RANGE_QUADRATURE"
    );
    let input: Value = serde_json::from_slice(&std::fs::read(&args[1])?)?;
    ensure!(!std::path::Path::new(&args[3]).exists(), "output exists");
    let receiver_definition: Value = serde_json::from_slice(&std::fs::read(&args[2])?)?;
    let taps: Vec<f64> =
        serde_json::from_value(receiver_definition["normalized_adc_impulse"].clone())?;
    ensure!(
        taps.len() == 491 && receiver_definition["output_decimation"] == 150,
        "unexpected receiver grid"
    );
    let taps: Vec<C> = taps.into_iter().map(|x| C::new(x, 0.)).collect();
    let quadrature: usize = args[5].parse()?;
    ensure!(
        quadrature > 0 && quadrature <= 25 && 150 % quadrature == 0,
        "quadrature must divide 150"
    );
    let rho: f64 = args[4].parse()?;
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
    // Archived 0.4-us geometry: t1-origin-1 = 2388.5 samples,
    // first source-centre 810.5 samples. Difference 1578 * 6 ADC samples.
    // TX envelope starts at archive sample 1 = six ADC samples.
    // Explicit diagnostic alignment, still requires direct marginal validation.
    let phase: isize = std::env::var("RUISDAP_ADC_PHASE_OFFSET")
        .unwrap_or_else(|_| "0".into())
        .parse()?;
    ensure!(
        phase.abs() <= 6,
        "diagnostic phase outside one archive sample"
    );
    let samples: Vec<usize> = (0..nv)
        .map(|i| (9468isize + phase) as usize + 150 * i)
        .collect();
    let rho_adc = rho.powf(1. / 150.);
    let energy: f64 = taps.iter().map(|z| z.norm_sqr()).sum();
    let pool = rayon::ThreadPoolBuilder::new().num_threads(128).build()?;
    let moments = pool.install(|| {
        codes
            .par_iter()
            .map(|code| -> Result<_> {
                let mut tx = vec![C::default(); 6];
                tx.extend(code.iter().flat_map(|&c| [C::new(c, 0.); 300]));
                let mut r = DMatrix::from_fn(nv, nv, |a, b| {
                    let d = a.abs_diff(b) * 150;
                    C::new(
                        if d < taps.len() {
                            receiver
                                * taps[..taps.len() - d]
                                    .iter()
                                    .zip(&taps[d..])
                                    .map(|(x, y)| (x * y.conj()).re)
                                    .sum::<f64>()
                                / energy
                        } else {
                            0.
                        },
                        0.,
                    )
                });
                let receiver_covariance = r.clone();
                for (s, &x) in profile.iter().enumerate().take(ng + 65) {
                    for q in 0..quadrature {
                        let offset = -75isize + ((q * 150 + 75) / quadrature) as isize;
                        voltage::add_filtered_ar1_cell(
                            &mut r,
                            &tx,
                            s as isize * 150 + offset as isize,
                            x / rho / quadrature as f64,
                            rho_adc,
                            &taps,
                            &samples,
                        )?;
                    }
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
                let signal_mu: Vec<C> = terms
                    .iter()
                    .zip(&mu)
                    .map(|(ts, m)| {
                        *m - ts
                            .iter()
                            .map(|t| t.weight * receiver_covariance[(t.first, t.second)])
                            .sum::<C>()
                    })
                    .collect();
                Ok((mu, cov, signal_mu))
            })
            .collect::<Result<Vec<_>>>()
    })?;
    let per_code_signal_mean: Vec<Vec<[f64; 2]>> = moments
        .iter()
        .map(|(_, _, mu)| mu.iter().map(|v| [v.re, v.im]).collect())
        .collect();
    for (mu, cov, _) in moments {
        for i in 0..ng {
            mean[i] += mu[i];
        }
        covariance += cov;
    }
    let error = (0..ng)
        .map(|i| (mean[i].re - profile[i]).abs())
        .fold(0., f64::max);
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
    let output = json!({"status":"diagnostic_only_not_accepted","assumptions":"Independent code pulses; stationary profile within 357.12 ms; full 15MHz FIR with 150x output decimation; exponential scattering ACF, still a sensitivity assumption; no active clutter projection. Not a validated reconstruction.","notch_bridge_hypothesis_only":bridge,"rho":rho,"input":args[1],"mean_change_from_rectangular_max_counts":error,"range_quadrature":quadrature,"diagnostic_adc_phase_offset":phase,"receiver_definition":args[2],"alignment":"archive t1-origin-1 minus first range centre = 1578*0.4 us; TX starts at 0.4 us; midpoint range quadrature; direct marginal validation pending","full_covariance_chi_square":q,"diagonal_chi_square":diagonal_q,"gaussian_nll_without_constant":0.5*(q+logdet),"log_determinant":logdet,"dof":2*ng,"per_code_signal_mean_counts":per_code_signal_mean,"mean_counts":mean.iter().map(|v|[v.re,v.im]).collect::<Vec<_>>(),"standard_error_counts":(0..ng).map(|i|[covariance[(2*i,2*i)].sqrt(),covariance[(2*i+1,2*i+1)].sqrt()]).collect::<Vec<_>>(),"notch_range_real_correlation":correlations,"covariance_counts2":(0..2*ng).map(|i|(0..2*ng).map(|j|covariance[(i,j)]).collect::<Vec<_>>()).collect::<Vec<_>>()});
    std::fs::write(&args[3], serde_json::to_vec(&output)?)?;
    println!(
        "rho={rho} full chi2={q:.2} diagonal chi2={diagonal_q:.2} dimension={}",
        2 * ng
    );
    Ok(())
}
