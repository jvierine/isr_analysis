use anyhow::{ensure, Result};
use num_complex::Complex64 as C;
use rayon::prelude::*;
use serde_json::json;
use std::fs;
use syisr_hplus_night::{
    data::Archive,
    retrieval::initial,
    theory::Theory,
};
fn random(x: &mut u64) -> u64 {
    *x ^= *x << 13;
    *x ^= *x >> 7;
    *x ^= *x << 17;
    *x
}
fn main() -> Result<()> {
    let a: Vec<_> = std::env::args().collect();
    ensure!(a.len() == 3, "clean.h5 output.json");
    let data = Archive::load(&a[1])?;
    let theory = Theory::new(700., 500.)?;
    let indices = data.indices(16);
    ensure!(indices.len() >= 8, "too few independent integrations");
    let hour = data.hour(16);
    let (obs, meta) = hour.observation(700., &theory, 1, 0)?;
    let (best, profile) = obs.solve(&theory, 1e11);
    let null = profile.iter().find(|f| f.h_fraction == 0.).unwrap();
    let observed_delta = null.chi_square - best.chi_square;
    let nboot = 200;
    let boots: Vec<_> = (0..nboot)
        .into_par_iter()
        .map(|i| -> Result<_> {
            let mut rng = (i as u64 + 1) * 987654321;
            let draws: Vec<_> = (0..indices.len())
                .map(|_| indices[random(&mut rng) as usize % indices.len()])
                .collect();
            let hour = data.hour_indices(16, &draws);
            let (b, _) = hour.observation(700., &theory, 1, 0)?;
            let mut fit = b.fit(&theory, &best, 1e11, None, 75);
            for h in [0.05, 0.5, 0.9] {
                let mut start = best.clone();
                start.h_fraction = h;
                let f = b.fit(&theory, &start, 1e11, None, 75);
                if f.chi_square < fit.chi_square {
                    fit = f;
                }
            }
            let null_delta = if i < 100 {
                let mut o = obs.clone();
                o.value = o
                    .value
                    .iter()
                    .enumerate()
                    .map(|(j, y)| {
                        C::new(null.predicted[j][0], null.predicted[j][1]) + b.value[j] - y
                    })
                    .collect();
                let mut f = o.fit(&theory, &best, 1e11, None, 75);
                for h in [0.05, 0.7] {
                    let mut s = best.clone();
                    s.h_fraction = h;
                    let q = o.fit(&theory, &s, 1e11, None, 75);
                    if q.chi_square < f.chi_square {
                        f = q;
                    }
                }
                let mut n = o.fit(&theory, null, 1e11, Some(0.), 75);
                for ti in [600., 1400.] {
                    let q = o.fit(&theory, &initial(ti, ti * 1.5, 0.), 1e11, Some(0.), 75);
                    if q.chi_square < n.chi_square {
                        n = q;
                    }
                }
                Some((n.chi_square - f.chi_square).max(0.))
            } else {
                None
            };
            Ok(json!({"resample":i,"fit":fit,"empirical_null_delta":null_delta}))
        })
        .collect::<Result<_>>()?;
    let mut h: Vec<f64> = boots
        .iter()
        .map(|v| v["fit"]["h_fraction"].as_f64().unwrap())
        .collect();
    h.sort_by(f64::total_cmp);
    let nulls: Vec<_> = boots
        .iter()
        .filter_map(|v| v["empirical_null_delta"].as_f64())
        .collect();
    let p = (1 + nulls.iter().filter(|&&x| x >= observed_delta).count()) as f64
        / (nulls.len() + 1) as f64;
    let per_window:Vec<_>=indices.par_iter().map(|&j|->Result<_>{let hour=data.hour_indices(16,&[j]);let(o,m)=hour.observation(700.,&theory,1,0)?;let(f,profile)=o.solve(&theory,1e11);Ok(json!({"metadata":m,"utc_start_s":data.meta["windows"][j]["utc_start_s"],"best":f,"profile":profile}))}).collect::<Result<_>>()?;
    let mut joint = vec![];
    for k in 0..profile.len() {
        joint.push(json!({"h_fraction":profile[k].h_fraction,"chi_square":per_window.iter().map(|v|v["profile"][k]["chi_square"].as_f64().unwrap()).sum::<f64>()}));
    }
    fs::write(
        &a[2],
        serde_json::to_vec(
            &json!({"schema":"syisr-hplus-block-stability-v1","source":a[1],"selection":"preselected UTC16--17, zenith, 700 km","metadata":meta,"best":best,"profile":profile,"block_count":indices.len(),"bootstrap_count":nboot,"bootstrap_h_fraction_percentiles":[h[5],h[100],h[194]],"empirical_null_draws":nulls.len(),"empirical_null_p_conditional":p,"observed_delta_chi_square":observed_delta,"resamples":boots,"individual_integrations":per_window,"joint_profile_independent_temperatures":joint,"method":"resample whole independent beam-dwell means, retaining all measured lags/ranges together; recompute receiver, background and local slope for each resample. Null test adds centered resampled measurement fluctuations to fitted H=0 mean.","limitations":["Only about 11 beam dwells; bootstrap tail precision limited","Finite-dimensional thermal model and inferred response remain conditional assumptions","Per-window joint profile allows temperatures, He fraction, flow and power to vary while H fraction is shared","Preselected from exploratory data: empirical p is not a look-elsewhere-corrected discovery significance"]}),
        )?,
    )?;
    println!(
        "H={:.3}; block 95% percentiles {:.3}--{:.3}; empirical conditional p={:.4}",
        best.h_fraction, h[5], h[194], p
    );
    Ok(())
}
