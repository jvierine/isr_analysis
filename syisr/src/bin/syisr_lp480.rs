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
                .filter(|(i, _)| i % size == rank)
                .map(|(_, j)| lp480::decode(j, &Path::new(&a[3]).join(format!("{}.h5", j.id))))
                .collect::<Result<Vec<_>>>()?;
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
            for (i, t) in times.into_iter().enumerate() {
                if i % size != rank || (a[3] != "all" && a[3] != t.to_string()) {
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
