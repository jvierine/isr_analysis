//! Reuse the immutable pulse inventory, screen the full recorded range, and
//! export both complete and partial ACFs from the same native block products.
use anyhow::{ensure, Result};
use std::{
    fs,
    io::{Read, Seek, SeekFrom, Write},
    path::Path,
    process::{Command, Stdio},
};
fn main() -> Result<()> {
    let a: Vec<String> = std::env::args().collect();
    ensure!(
        a.len() == 3,
        "reprocess-full-range original-root output-root"
    );
    let old = Path::new(&a[1]);
    let root = Path::new(&a[2]);
    fs::create_dir_all(root)?;
    let plan: serde_json::Value = serde_json::from_slice(&fs::read(old.join("plan.json"))?)?;
    let rank: usize = std::env::var("OMPI_COMM_WORLD_RANK")
        .unwrap_or("0".into())
        .parse()?;
    let size: usize = std::env::var("OMPI_COMM_WORLD_SIZE")
        .unwrap_or("1".into())
        .parse()?;
    if rank == 0 {
        fs::copy(old.join("plan.json"), root.join("plan.json"))?;
    }
    let exe = std::env::current_exe()?.with_file_name("process");
    for t in (rank..48).step_by(size) {
        let dest = root.join(format!("windows/{t:03}"));
        fs::create_dir_all(&dest)?;
        if dest.join("done.json").exists() {
            continue;
        }
        let w = &plan["windows"][t];
        let raw = dest.join("input.dat");
        let mut output = fs::File::create(&raw)?;
        let mut buffer = vec![0u8; 8 * 1024 * 1024];
        for seg in w["segments"].as_array().unwrap() {
            let mut input = fs::File::open(seg["path"].as_str().unwrap())?;
            let first = seg["first_frame"].as_u64().unwrap();
            let last = seg["stop_frame"].as_u64().unwrap();
            input.seek(SeekFrom::Start(first * 31288))?;
            let mut left = (last - first) * 31288;
            while left > 0 {
                let n = left.min(buffer.len() as u64) as usize;
                input.read_exact(&mut buffer[..n])?;
                output.write_all(&buffer[..n])?;
                left -= n as u64;
            }
        }
        output.flush()?;
        drop(output);
        ensure!(
            fs::metadata(&raw)?.len() == w["pulses"].as_u64().unwrap() * 31288,
            "raw selection length"
        );
        let log = fs::File::create(dest.join("run.log"))?;
        let status = Command::new(&exe)
            .arg(&raw)
            .arg(dest.join("lpm.tmp.h5"))
            .arg("--screen-bright")
            .env("SYISR_SCREEN_END_SAMPLE", "3782")
            .env("SYISR_PARTIAL_OUTPUT", dest.join("partial.tmp.h5"))
            .env("RAYON_NUM_THREADS", "1")
            .stdout(Stdio::from(log.try_clone()?))
            .stderr(Stdio::from(log))
            .status()?;
        if !status.success() {
            eprintln!("window {t}: processing failed {status}");
            continue;
        }
        fs::rename(dest.join("lpm.tmp.h5"), dest.join("lpm.h5"))?;
        fs::rename(dest.join("partial.tmp.h5"), dest.join("partial.h5"))?;
        if !dest.join("fits.h5").exists() {
            std::os::unix::fs::symlink(
                old.join(format!("windows/{t:03}/fits.h5")),
                dest.join("fits.h5"),
            )?;
        }
        fs::write(
            dest.join("done.json"),
            serde_json::to_vec_pretty(
                &serde_json::json!({"window":t,"screen_end_sample_exclusive":3782,"seed_fits":"previous broad fits used only for initialization"}),
            )?,
        )?;
        fs::remove_file(raw)?;
        eprintln!("window {t}: full-range screened products complete");
    }
    Ok(())
}
