use anyhow::{ensure, Result};
use num_complex::Complex64 as C;
use rayon::prelude::*;
use syisr_long_pulse::{
    fit_response,
    retrieval::{initial, Observation},
    theory::Theory,
};
fn write(h: &hdf5::Group, n: &str, v: &[f64]) -> Result<()> {
    h.new_dataset_builder().with_data(v).create(n)?;
    Ok(())
}
fn main() -> Result<()> {
    let a: Vec<_> = std::env::args().collect::<Vec<_>>();
    ensure!(
        (4..=5).contains(&a.len()),
        "fit data.h5 geometry.h5 output.h5 [minimum_lag_index]"
    );
    let lag_min: usize = a.get(4).map(|s| s.parse()).transpose()?.unwrap_or(0);
    ensure!(lag_min <= 10, "minimum lag index must be 0..10");
    let f = hdf5::File::open(&a[1])?;
    let g = hdf5::File::open(&a[2])?;
    let time = f.dataset("time_utc_s")?.read_raw::<f64>()?;
    let get = |name: &str| f.dataset(name).unwrap().read_raw::<f64>().unwrap();
    let ranges = get("slant_range_km");
    let ar = get("acf_real_adu2");
    let ai = get("acf_imag_adu2");
    let sr = get("acf_real_sem_adu2");
    let si = get("acf_imag_sem_adu2");
    let native = get("native_lpm_real_adu2");
    let native_i = get("native_lpm_imag_adu2");
    let field = g.dataset("field_nT")?.read_raw::<f64>()?;
    let angle = g
        .dataset("angle_from_perpendicular_deg")?
        .read_raw::<f64>()?;
    let indices = g.dataset("range_index")?.read_raw::<i64>()?;
    let noise: Vec<C> = (0..61)
        .map(|l| {
            (2500..3400)
                .map(|j| C::new(native[l * 3782 + j], native_i[l * 3782 + j]))
                .sum::<C>()
                / 900.
        })
        .collect::<Vec<_>>();
    let rho: Vec<C> = (0..61)
        .map(|l| {
            noise[l] / noise[0].re
                * if l == 0 {
                    1.
                } else if l < 21 {
                    0.5 + 0.5 * (std::f64::consts::PI * l as f64 / 21.).cos()
                } else {
                    0.
                }
        })
        .collect::<Vec<_>>();
    let (fir, negative) = fit_response::receiver_fir(&rho);
    ensure!(negative < 1e-3, "receiver noise spectrum not positive");
    let measured_peak = (0..ranges.len())
        .filter(|&i| ranges[i] >= 300. && ranges[i] < 650.)
        .map(|i| ar[i * 61] * ranges[i].powi(2))
        .fold(0f64, f64::max);
    let peak = std::env::var("SYISR_PEAK_POWER_RANGE2")
        .ok()
        .map(|s| s.parse::<f64>())
        .transpose()?
        .unwrap_or(measured_peak);
    let results:Vec<_>=indices.par_iter().enumerate().map(|(j,&index)| -> Result<_>{
  let i=index as usize;let r=ranges[i];let ne=1e12*ar[i*61]*r*r/peak;ensure!(ne>0.,"nonpositive density proxy");
  let xy:Vec<_>=(0..ranges.len()).filter(|&k|(ranges[k]-r).abs()<150.&&ar[k*61]>5.*sr[k*61]).map(|k|(ranges[k]-r,ar[k*61].ln())).collect::<Vec<_>>();
  ensure!(xy.len()>=3,"insufficient positive power points for range gradient");
  let mx=xy.iter().map(|v|v.0).sum::<f64>()/xy.len() as f64;let my=xy.iter().map(|v|v.1).sum::<f64>()/xy.len() as f64;
  let gamma=(xy.iter().map(|v|(v.0-mx)*(v.1-my)).sum::<f64>()/xy.iter().map(|v|(v.0-mx).powi(2)).sum::<f64>()).clamp(-0.03,0.03);
  let theory=Theory::at_field(field[j],angle[j],500.)?;let validation=theory.validate()?;
  let response=fit_response::kernel(&theory.f,&fir,gamma);
  let full_obs=Observation{value:(0..61).map(|l|C::new(ar[i*61+l],ai[i*61+l])).collect::<Vec<_>>(),sigma:(0..61).map(|l|[sr[i*61+l].max(1.),si[i*61+l].max(1.)]).collect::<Vec<_>>(),kernel:response,rho:(0..61).map(|l|rho[l]*200./(200-l) as f64).collect::<Vec<_>>(),background_sigma:1e9,lag_min:0};
  let obs=Observation{value:full_obs.value[lag_min..].to_vec(),sigma:full_obs.sigma[lag_min..].to_vec(),kernel:full_obs.kernel[lag_min..].to_vec(),rho:full_obs.rho[lag_min..].to_vec(),background_sigma:full_obs.background_sigma,lag_min};
  let mut best=initial(1000.,1500.,0.1);best.chi_square=f64::INFINITY;
  for ti in [600.,1200.,2200.] {for hydrogen in [0.02,0.3,0.8] {for helium in [0.02,0.4,0.85] {
   let mut start=initial(ti,ti*1.5,hydrogen);start.he_fraction=(1.-hydrogen)*helium;start.o_fraction=1.-start.h_fraction-start.he_fraction;
   let fit=obs.fit(&theory,&start,ne,None,100);if fit.chi_square<best.chi_square{best=fit;}
  }}}
  let mut profiles=vec![];for hydrogen in [0.,0.05,0.1,0.2,0.4,0.6,0.8,1.] {profiles.push(obs.fit(&theory,&best,ne,Some(hydrogen),100));}
  let errors=obs.local_errors(&theory,&best,ne);
  let spec=theory.spectrum(best.ti_k,best.te_k,best.spectral_velocity_m_s,best.h_fraction,best.he_fraction/(1.-best.h_fraction).max(1e-12),ne);
  let predicted_zero=full_obs.kernel[0].iter().zip(&spec).map(|(k,s)|*k*s).sum::<C>()*best.signal_adu2+full_obs.rho[0]*best.background_adjustment_adu2;
  eprintln!("{r:.0} km H={:.3} He={:.3} O={:.3} Ti={:.0} Te={:.0} background={:.1} chi2/dof={:.2} {}",best.h_fraction,best.he_fraction,best.o_fraction,best.ti_k,best.te_k,best.background_adjustment_adu2,best.chi_square/best.dof as f64,best.status);
  Ok((i,ne,gamma,best,profiles,obs,validation,predicted_zero,errors))
 }).collect::<Vec<_>>();
    let output = hdf5::File::create(&a[3])?;
    write(
        &output,
        "time_utc_s",
        &[time[0], time[time.len() - 1] + 0.04],
    )?;
    let mut summary = vec![];
    let mut failures = vec![];
    for (target, result) in results.into_iter().enumerate() {
        let (i, ne, gamma, best, profiles, obs, validation, predicted_zero, errors) = match result {
            Ok(v) => v,
            Err(e) => {
                failures
                    .push(serde_json::json!({"range_index":indices[target],"error":e.to_string()}));
                continue;
            }
        };
        let group = output.create_group(&format!("range_{:.0}", ranges[i]))?;
        write(
            &group,
            "lag_us",
            &(lag_min..61).map(|l| l as f64 * 10.).collect::<Vec<_>>(),
        )?;
        for (name, v) in [
            (
                "observed_real_adu2",
                obs.value.iter().map(|v| v.re).collect::<Vec<_>>(),
            ),
            (
                "observed_imag_adu2",
                obs.value.iter().map(|v| v.im).collect::<Vec<_>>(),
            ),
            (
                "sigma_real_adu2",
                obs.sigma.iter().map(|v| v[0]).collect::<Vec<_>>(),
            ),
            (
                "sigma_imag_adu2",
                obs.sigma.iter().map(|v| v[1]).collect::<Vec<_>>(),
            ),
            (
                "model_real_adu2",
                best.predicted.iter().map(|v| v[0]).collect::<Vec<_>>(),
            ),
            (
                "model_imag_adu2",
                best.predicted.iter().map(|v| v[1]).collect::<Vec<_>>(),
            ),
            (
                "hydrogen_profile_fraction",
                profiles.iter().map(|p| p.h_fraction).collect::<Vec<_>>(),
            ),
            (
                "hydrogen_profile_delta_chi2",
                profiles
                    .iter()
                    .map(|p| p.chi_square - best.chi_square)
                    .collect::<Vec<_>>(),
            ),
        ] {
            write(&group, name, &v)?;
        }
        let meta = serde_json::json!({"slant_range_km":ranges[i],"conditional_local_sigma_H_He_Ti_Te":errors,"minimum_fitted_lag_us":lag_min*10,"held_out_zero_lag_observed_adu2":ar[i*61],"held_out_zero_lag_prediction_adu2":predicted_zero.re,"held_out_zero_lag_sigma_adu2":sr[i*61],"density_shape_assumption_m3":ne,"local_log_power_gradient_per_km":gamma,"fit":best,"theory_validation":validation,"background":"independent free additive amplitude multiplying receiver-noise ACF; virtually unpenalized","density":"P(r)*r^2 proxy scaled to provisional F-region peak 1e12 m^-3; not calibrated; Te/Ti density correction not applied","range_model":"2 ms rectangular pulse, measured trapezoidal summation, inferred voltage FIR, local exponential amplitude profile, constant plasma shape across broad footprint","uncertainty":"diagonal 10-second-block jackknife weights; profile chi-square is conditional and not a formal significance","receiver_negative_psd_mass":negative});
        group
            .new_attr_builder()
            .with_data(meta.to_string().as_str())
            .create("metadata_json")?;
        summary.push(meta);
    }
    output
        .new_attr_builder()
        .with_data(serde_json::to_string(&failures)?.as_str())
        .create("failed_gates_json")?;
    output
        .new_attr::<f64>()
        .create("reference_peak_power_range2")?
        .write_scalar(&peak)?;
    output
        .new_attr_builder()
        .with_data(serde_json::to_string(&summary)?.as_str())
        .create("summary_json")?;
    Ok(())
}
