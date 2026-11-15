use anyhow::{ensure, Result};
use rayon::prelude::*;
use serde_json::json;
use std::{fs, sync::Arc};
use syisr_hplus_night::{data::Archive, theory::Theory};
fn main() -> Result<()> {
    let a: Vec<_> = std::env::args().collect();
    ensure!(a.len() >= 3, "input.h5 output_dir [pilot]");
    fs::create_dir_all(&a[2])?;
    let data = Archive::load(&a[1])?;
    let pilot = a.get(3).is_some_and(|s| s == "pilot");
    let heights = if pilot {
        vec![500., 700., 900., 1100.]
    } else {
        vec![
            400., 500., 600., 700., 800., 900., 1000., 1100., 1200., 1400., 1600.,
        ]
    };
    let theory: Vec<_> = heights
        .par_iter()
        .map(|&h| Theory::new(h, 500.).map(Arc::new))
        .collect::<Result<_>>()?;
    let validation: Vec<_> = theory.iter().map(|t| t.validate()).collect::<Result<_>>()?;
    fs::write(
        format!("{}/theory-validation.json", a[2]),
        serde_json::to_vec_pretty(&validation)?,
    )?;
    let single_hour = a.get(3).is_some_and(|s| s == "pilot" || s == "clean");
    let requested_hour = a.get(3).and_then(|s| s.parse::<usize>().ok());
    let hours = if let Some(h) = requested_hour {
        vec![h]
    } else if single_hour {
        vec![16]
    } else {
        (15..=20).collect()
    };
    let hours: Vec<_> = hours.iter().map(|&h| data.hour(h)).collect();
    let jobs: Vec<_> = hours
        .iter()
        .flat_map(|hour| {
            heights
                .iter()
                .enumerate()
                .map(move |(i, &height)| (hour, height, i))
        })
        .collect();
    let outputs:Vec<_>=jobs.par_iter().map(|(hour,height,i)|->Result<_>{let (obs,meta)=hour.observation(*height,&theory[*i],1,0)?;let(best,profile)=obs.solve(&theory[*i],1e11);let null_fit=profile.iter().find(|p|p.h_fraction==0.).unwrap();let mut sensitivity=vec![];for ne in[1e10,1e12]{let mut fit=obs.fit(&theory[*i],&best,ne,None,80);for h in[0.02,0.5,0.9]{let mut start=best.clone();start.h_fraction=h;let f=obs.fit(&theory[*i],&start,ne,None,70);if f.chi_square<fit.chi_square{fit=f;}}sensitivity.push(json!({"shape_density_m3":ne,"fit":fit}));}let(bgh,metah)=hour.observation(*height,&theory[*i],2,0)?;let fh=bgh.fit(&theory[*i],&best,1e11,None,80);let(short,metas)=hour.observation(*height,&theory[*i],1,2)?;let fs=short.fit(&theory[*i],&best,1e11,None,80);eprintln!("UTC{} {:.0}km H={:.3} He={:.3} Ti={:.0} Te={:.0} chi/dof={:.2} delta_null={:.1}",hour.hour,height,best.h_fraction,best.he_fraction,best.ti_k,best.te_k,best.chi_square/best.dof as f64,null_fit.chi_square-best.chi_square);Ok(json!({"metadata":meta,"best":best,"profile_likelihood":profile,"null_delta_chi_square":null_fit.chi_square-best.chi_square,"density_shape_sensitivity":sensitivity,"nearer_background_sensitivity":{"metadata":metah,"fit":fh},"short_lag_sensitivity":{"metadata":metas,"fit":fs},"measured_real":obs.value.iter().map(|z|z.re).collect::<Vec<_>>(),"measured_imag":obs.value.iter().map(|z|z.im).collect::<Vec<_>>(),"sem_real":obs.sigma.iter().map(|v|v[0]).collect::<Vec<_>>(),"sem_imag":obs.sigma.iter().map(|v|v[1]).collect::<Vec<_>>() }))}).collect::<Result<_>>()?;
    let product = json!({"schema":"syisr-hplus-night-zenith-poc-v1","source":a[1],"producer":"Independent Rust RUISDAP/isr_analysis mixture fit, supplied L2 and IRI composition are not fitting inputs","ion_species":["H+","He+","O+"],"absolute_density_calibrated":false,"data_selection":data.meta["selection"],"results":outputs,"limitations":["Conditional diagonal-lag weighting with conservative fully-correlated range-average SEM; measured lag covariance not yet marginalized","Common Ti among species; constant thermodynamic/composition within ~100 km LP footprint, local exponential echo-power gradient","Receiver voltage response is inferred from background ACF with zero-phase assumption; background/reference sensitivity retained","Collisionless topside approximation, magnetized Maxwellian electrons, large-k*rho unmagnetized ions; only zenith first","Ne sets Debye shape only: fits at 1e10,1e11,1e12 m^-3 retained; amplitude not calibrated density","Profile likelihood is conditional on response and temperature assumptions, not a detection significance until covariance and contaminants are checked"]});
    fs::write(format!("{}/fits.json", a[2]), serde_json::to_vec(&product)?)?;
    let out = hdf5::File::create(format!("{}/hplus-fits.h5", a[2]))?;
    out.new_dataset_builder()
        .with_data(&serde_json::to_vec(&product)?)
        .deflate(4)
        .create("metadata_json_utf8")?;
    println!("{} fits complete", outputs.len());
    Ok(())
}
