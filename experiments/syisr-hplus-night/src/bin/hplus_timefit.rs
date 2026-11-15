use anyhow::{ensure, Result};
use rayon::prelude::*;
use serde_json::json;
use std::{fs, sync::Arc};
use syisr_hplus_night::{data::Archive, theory::Theory};
fn main() -> Result<()> {
    let a: Vec<_> = std::env::args().collect();
    ensure!(
        a.len() == 5,
        "input.h5 output_dir UTC_hour integration_minutes"
    );
    let hour: usize = a[3].parse()?;
    let minutes: usize = a[4].parse()?;
    ensure!(
        minutes > 0 && 60 % minutes == 0,
        "integration must divide hour"
    );
    fs::create_dir_all(&a[2])?;
    let data = Archive::load(&a[1])?;
    let heights = vec![
        400., 500., 600., 700., 800., 900., 1000., 1100., 1200., 1400., 1600.,
    ];
    let theory: Vec<_> = heights
        .par_iter()
        .map(|&h| Theory::new(h, 500.).map(Arc::new))
        .collect::<Result<_>>()?;
    let base = chrono::DateTime::parse_from_rfc3339("2023-11-15T00:00:00Z")?.timestamp() as f64
        + hour as f64 * 3600.;
    let pools: Vec<_> = (0..60 / minutes)
        .map(|b| {
            let lo = base + (b * minutes * 60) as f64;
            let hi = lo + (minutes * 60) as f64;
            let ids: Vec<_> = data
                .indices(hour)
                .into_iter()
                .filter(|&j| {
                    let t = data.meta["windows"][j]["utc_start_s"].as_f64().unwrap();
                    t >= lo && t < hi
                })
                .collect();
            ensure!(!ids.is_empty(), "empty integration {lo}");
            Ok((lo, hi, data.hour_indices(hour, &ids), ids.len()))
        })
        .collect::<Result<_>>()?;
    let jobs: Vec<_> = pools
        .iter()
        .flat_map(|p| heights.iter().enumerate().map(move |(i, &h)| (p, i, h)))
        .collect();
    let rows: Vec<_> = jobs
        .par_iter()
        .map(|(p, i, h)| -> Result<_> {
            let (obs, mut meta) = p.2.observation(*h, &theory[*i], 1, 0)?;
            let (best, profile) = obs.solve(&theory[*i], 1e11);
            meta["time_start_unix_s"] = json!(p.0);
            meta["time_end_unix_s"] = json!(p.1);
            meta["beam_dwells"] = json!(p.3);
            Ok(json!({"metadata":meta,"best":best,"profile_likelihood":profile}))
        })
        .collect::<Result<_>>()?;
    fs::write(
        format!("{}/fits.json", a[2]),
        serde_json::to_vec(
            &json!({"schema":"syisr-hplus-timefits-v1","integration_s":minutes*60,"source":a[1],"results":rows,"assumptions":"same screened long-pulse local H/He/O model and all measured lags as hourly fits; no bootstrap or systematic-error marginalization for these shorter integrations"}),
        )?,
    )?;
    println!("{} time-altitude fits complete", rows.len());
    Ok(())
}
