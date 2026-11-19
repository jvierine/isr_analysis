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
        _ => anyhow::bail!("unsupported command"),
    };
    Ok(())
}
