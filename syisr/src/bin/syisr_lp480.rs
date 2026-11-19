use anyhow::{ensure, Result};
use rayon::prelude::*;
use std::path::Path;
use syisr_ac::{campaign, lp480};
fn main() -> Result<()> {
    let a: Vec<_> = std::env::args().collect();
    ensure!(
        a.len() >= 4,
        "index RAW OUT_H5 | prepare INDEX_H5 OUTPUT_DIR"
    );
    match a[1].as_str() {
        "index" => {
            lp480::index(Path::new(&a[2]), Path::new(&a[3]))?;
        }
        "repair-index" => {
            lp480::repair_index(Path::new(&a[2]))?;
        }
        "prepare" => {
            let m: lp480::Manifest = campaign::load(Path::new(&a[2]))?;
            std::fs::create_dir_all(&a[3])?;
            let rank = std::env::var("OMPI_COMM_WORLD_RANK")
                .unwrap_or("0".into())
                .parse::<usize>()?;
            let size = std::env::var("OMPI_COMM_WORLD_SIZE")
                .unwrap_or("1".into())
                .parse::<usize>()?;
            m.jobs
                .par_iter()
                .enumerate()
                .filter(|(i,j)| i % size == rank && std::env::var("SYISR_PREPARE_ONLY").map_or(true,|id|id==j.id))
                .map(|(_, j)| lp480::decode(j, &Path::new(&a[3]).join(format!("{}.h5", j.id))))
                .collect::<Result<Vec<_>>>()?;
        }
        "compact" => {
            syisr_ac::lp480_fit::compact(Path::new(&a[2]), Path::new(&a[3]))?;
        }
        "background" => {
            syisr_ac::lp480_fit::background(Path::new(&a[2]), Path::new(&a[3]))?;
        }
        "calibrate" => {
            syisr_ac::lp480_fit::calibrate(Path::new(&a[2]))?;
        }
        "fit" => {
            ensure!(a.len() == 7, "fit ROOT TIME|all OUT PREVIOUS|none ITER");
            let m: lp480::Manifest = campaign::load(&Path::new(&a[2]).join("index.h5"))?;
            let times: std::collections::BTreeSet<_> = m.jobs.iter().map(|j| j.start).collect();
            let rank = std::env::var("OMPI_COMM_WORLD_RANK")
                .unwrap_or("0".into())
                .parse::<usize>()?;
            let size = std::env::var("OMPI_COMM_WORLD_SIZE")
                .unwrap_or("1".into())
                .parse::<usize>()?;
            std::env::set_var("SYISR_FIX_HELIUM_ZERO", "1");
            std::env::set_var("SYISR_AUTODIFF_JOINT", "1");
            std::env::set_var("SYISR_TEMPORAL_ROBUST_DELTA", "1.5");
            let selected: Vec<_> = times
                .into_iter()
                .enumerate()
                .filter(|(i, t)| match a[3].as_str() {
                    "all" => true,
                    "even" => i % 2 == 0,
                    "odd" => i % 2 == 1,
                    _ => a[3] == t.to_string(),
                })
                .map(|(_, t)| t)
                .collect();
            for (i, t) in selected.into_iter().enumerate() {
                if i % size != rank {
                    continue;
                }
                if Path::new(&a[4]).join(format!("{t}.h5")).exists() {
                    continue;
                }
                syisr_ac::lp480_fit::fit(
                    Path::new(&a[2]),
                    t,
                    Path::new(&a[4]),
                    if a[5] == "none" {
                        None
                    } else {
                        Some(Path::new(&a[5]))
                    },
                    a[6].parse()?,
                )?;
            }
        }
        _ => anyhow::bail!("unsupported command"),
    };
    Ok(())
}
