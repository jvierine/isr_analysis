//! Local MPI replay plus lower-boundary fits; failed ranks do not interrupt others.
use anyhow::{ensure, Result};
use std::{
    fs,
    path::Path,
    process::{Command, Stdio},
};
fn main() -> Result<()> {
    let a: Vec<String> = std::env::args().collect();
    ensure!(a.len() == 3, "partial-campaign old-root new-root");
    let old = Path::new(&a[1]);
    let root = Path::new(&a[2]);
    fs::create_dir_all(root.join("windows"))?;
    let rank = std::env::var("OMPI_COMM_WORLD_RANK")
        .unwrap_or("0".into())
        .parse::<usize>()?;
    let size = std::env::var("OMPI_COMM_WORLD_SIZE")
        .unwrap_or("1".into())
        .parse::<usize>()?;
    let dir = std::env::current_exe()?.parent().unwrap().to_path_buf();
    let mut failures = vec![];
    for t in (rank..48).step_by(size) {
        let dest = root.join(format!("windows/{t:03}"));
        fs::create_dir_all(&dest)?;
        let out = dest.join("profile.h5");
        if out.exists() {
            let f = hdf5::File::open(&out)?;
            let m: serde_json::Value =
                serde_json::from_slice(&f.attr("metadata_json")?.read_raw::<u8>()?)?;
            if m["fit"]["status"] == "converged" {
                continue;
            }
            fs::rename(&out, dest.join("previous-nonconverged.h5"))?;
        }
        let log = fs::File::create(dest.join("run.log"))?;
        let replay = Command::new(dir.join("partial-products"))
            .arg(old)
            .arg(root)
            .arg(t.to_string())
            .stdout(Stdio::from(log.try_clone()?))
            .stderr(Stdio::from(log.try_clone()?))
            .status()?;
        if !replay.success() {
            failures.push(t);
            continue;
        }
        let status = Command::new(dir.join("profile-fit"))
            .arg(old.join(format!("windows/{t:03}/lpm.h5")))
            .arg(old.join(format!("windows/{t:03}/fits.h5")))
            .arg(&out)
            .args([
                "80,100,150,200,250,300,350,400,500,600,700,800,1000,1250,1500,1700",
                "1",
                "25899.08075225199",
                "800",
                "25",
            ])
            .env("RAYON_NUM_THREADS", "1")
            .env("SYISR_PARTIAL_PRODUCT", dest.join("partial.h5"))
            .env_remove("SYISR_BOTTOMSIDE_CHAPMAN")
            .env_remove("SYISR_COVARIANCE_SHRINKAGE")
            .stdout(Stdio::from(log.try_clone()?))
            .stderr(Stdio::from(log))
            .status()?;
        let good = if status.success() {
            let f = hdf5::File::open(&out)?;
            let m: serde_json::Value =
                serde_json::from_slice(&f.attr("metadata_json")?.read_raw::<u8>()?)?;
            m["fit"]["status"] == "converged"
        } else {
            false
        };
        if !good {
            failures.push(t);
        }
        eprintln!(
            "rank {rank}: window {t} {}",
            if good { "converged" } else { "needs review" }
        );
    }
    fs::write(
        root.join(format!("rank-{rank:02}.json")),
        serde_json::to_vec_pretty(&serde_json::json!({"rank":rank,"failures":failures}))?,
    )?;
    Ok(())
}
