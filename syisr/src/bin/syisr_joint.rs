//! MPI campaign entry point, reusing immutable decoded products and gated fits only for initialization/calibration.
use anyhow::{ensure, Result};
use std::{fs, path::Path};
use syisr_ac::{
    campaign::{self, Campaign},
    decode, fullprofile,
};
fn main() -> Result<()> {
    let a: Vec<_> = std::env::args().collect();
    ensure!(
        a.len() == 5,
        "syisr_joint INPUT_ROOT JOB_ID-or-all BOOTSTRAPS OUT_ROOT"
    );
    let input = Path::new(&a[1]);
    let out = Path::new(&a[4]);
    let c: Campaign = campaign::load(&input.join("campaign.h5"))?;
    if a[2] == "validate" {
        let mut checked = 0;
        for e in fs::read_dir(out.join("joint-fits"))? {
            let p = e?.path();
            if p.extension().is_none_or(|v| v != "h5") {
                continue;
            }
            let v: serde_json::Value = campaign::load(&p)?;
            let fit = &v["nominal_fit"];
            ensure!(
                fit["status"]
                    .as_str()
                    .is_some_and(|s| s.starts_with("converged")),
                "joint pilot has not converged"
            );
            ensure!(
                fit["normalized_gradient"].as_f64().unwrap_or(f64::INFINITY) < 0.05,
                "pilot gradient"
            );
            for g in v["gates"].as_array().unwrap() {
                ensure!(
                    g["ne_m3"].as_f64().is_some_and(|n| n > 0.),
                    "invalid density"
                );
                let n = &g["node"];
                ensure!(
                    n["h_fraction"]
                        .as_f64()
                        .is_some_and(|h| (0.0..=1.0).contains(&h)),
                    "invalid H fraction"
                );
            }
            checked += 1;
        }
        ensure!(checked > 0, "no completed joint pilot");
        println!("JOINT_PILOT_VALIDATED {checked}");
        return Ok(());
    }
    let codes = decode::phase(&input.join("m32-reference.bc"))?;
    let boot: usize = a[3].parse()?;
    ensure!(
        boot == 0 || boot >= 32,
        "pilot0 or >=32 joint bootstrap draws"
    );
    let rank = std::env::var("OMPI_COMM_WORLD_RANK")
        .unwrap_or("0".into())
        .parse::<usize>()?;
    let size = std::env::var("OMPI_COMM_WORLD_SIZE")
        .unwrap_or("1".into())
        .parse::<usize>()?;
    let threads = std::env::var("RAYON_NUM_THREADS")
        .unwrap_or("2".into())
        .parse::<usize>()?;
    rayon::ThreadPoolBuilder::new()
        .num_threads(threads)
        .build_global()?;
    fs::create_dir_all(out.join("joint-failures"))?;
    for (i, j) in c.jobs.iter().enumerate() {
        if a[2] != "all" && a[2] != j.id {
            continue;
        }
        if i % size != rank {
            continue;
        }
        if out.join("joint-fits").join(format!("{}.h5", j.id)).exists() {
            continue;
        }
        match fullprofile::fit(&c, j, input, out, &codes, boot) {
            Ok(r) => eprintln!("JOINT_COMPLETE {} {}", j.id, r.nominal_fit.status),
            Err(e) => {
                fs::write(
                    out.join("joint-failures").join(format!("{}.txt", j.id)),
                    format!("{e:#}"),
                )?;
                eprintln!("JOINT_FAILED {} {e:#}", j.id);
            }
        }
    }
    Ok(())
}
