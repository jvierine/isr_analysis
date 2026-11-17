//! Recover early partial products from raw prefixes, reusing the exact saved pulse mask.
use anyhow::{ensure, Result};
use num_complex::Complex64 as C;
use std::{
    fs,
    io::{BufReader, Read, Seek, SeekFrom},
    path::{Path, PathBuf},
    thread,
    time::Duration,
};
struct DiskLock(PathBuf);
impl Drop for DiskLock {
    fn drop(&mut self) {
        let _ = fs::remove_file(&self.0);
    }
}
use syisr_long_pulse::{ambiguity_audit::nominal_range, reader::Header};
fn main() -> Result<()> {
    let a: Vec<String> = std::env::args().collect();
    ensure!(
        a.len() == 4,
        "partial-products old-root output-root window-index"
    );
    let t: usize = a[3].parse()?;
    let root = Path::new(&a[1]);
    let dest = Path::new(&a[2]);
    fs::create_dir_all(dest.join(format!("windows/{t:03}")))?;
    let path = dest.join(format!("windows/{t:03}/partial.h5"));
    if path.exists() {
        return Ok(());
    }
    let lockpath = dest.join("raw-replay.lock");
    loop {
        match fs::OpenOptions::new()
            .write(true)
            .create_new(true)
            .open(&lockpath)
        {
            Ok(_) => break,
            Err(e) if e.kind() == std::io::ErrorKind::AlreadyExists => {
                thread::sleep(Duration::from_millis(200))
            }
            Err(e) => return Err(e.into()),
        }
    }
    let _lock = DiskLock(lockpath);
    let p: serde_json::Value = serde_json::from_slice(&fs::read(root.join("plan.json"))?)?;
    let w = &p["windows"][t];
    let old = hdf5::File::open(root.join(format!("windows/{t:03}/lpm.h5")))?;
    let keep = old.dataset("pulse_keep")?.read_raw::<u8>()?;
    let counts = old.dataset("block_retained_pulses")?.read_raw::<f64>()?;
    let nb = counts.len();
    let bg_r = old.dataset("background_acf_real_adu2")?.read_raw::<f64>()?;
    let bg_i = old.dataset("background_acf_imag_adu2")?.read_raw::<f64>()?;
    let indices: Vec<usize> = (20..90).step_by(5).collect();
    let nr = indices.len();
    let mut block = vec![C::default(); nb * nr * 61];
    let mut prefix = vec![0u8; 31288];
    let mut n = 0usize;
    let mut x = vec![C::default(); 166];
    for seg in w["segments"].as_array().unwrap() {
        let mut f = BufReader::with_capacity(
            4 * 1024 * 1024,
            fs::File::open(seg["path"].as_str().unwrap())?,
        );
        f.seek(SeekFrom::Start(
            seg["first_frame"].as_u64().unwrap() * 31288,
        ))?;
        for _frame in seg["first_frame"].as_u64().unwrap()..seg["stop_frame"].as_u64().unwrap() {
            f.read_exact(&mut prefix)?;
            if keep[n] > 0 {
                if n == 0 || n % 10000 == 0 {
                    let h = Header::parse(&prefix[..1024])?;
                    ensure!(
                        h.gate_front_us == 2001 && h.samples == 3782 && h.pulse_width_us == 2000,
                        "partial mode mismatch"
                    );
                }
                for j in 0..166 {
                    let o = 1024 + 8 * j;
                    let q = i32::from_le_bytes(prefix[o..o + 4].try_into().unwrap()) as f64;
                    let i = i32::from_le_bytes(prefix[o + 4..o + 8].try_into().unwrap()) as f64;
                    x[j] = C::new(i, q);
                }
                let b = n / 250;
                for l in 0usize..61 {
                    let mut cumulative = [C::new(0., 0.); 107];
                    for k in 30..106 {
                        cumulative[k + 1] = cumulative[k] + x[k] * x[k + l].conj();
                    }
                    for (g, &i) in indices.iter().enumerate() {
                        let lo = i.saturating_sub(l).max(30);
                        let hi = i + 21;
                        block[(b * nr + g) * 61 + l] +=
                            (cumulative[hi] - cumulative[lo]) / (hi - lo) as f64;
                    }
                }
            }
            n += 1;
        }
    }
    ensure!(n == keep.len(), "raw plan/mask length mismatch");
    for b in 0..nb {
        for g in 0..nr {
            for l in 0..61 {
                let j = (b * nr + g) * 61 + l;
                block[j] = if counts[b] > 0. {
                    (block[j] / counts[b] - C::new(bg_r[b * 61 + l], bg_i[b * 61 + l])) * 200.
                        / (200 - l) as f64
                } else {
                    C::new(f64::NAN, f64::NAN)
                };
            }
        }
    }
    let total: f64 = counts.iter().sum();
    let good: Vec<usize> = (0..nb).filter(|&b| counts[b] > 0.).collect();
    let mut mean = vec![C::default(); nr * 61];
    let mut sr = vec![0.; nr * 61];
    let mut si = sr.clone();
    for j in 0..nr * 61 {
        mean[j] = good
            .iter()
            .map(|&b| block[b * nr * 61 + j] * counts[b] / total)
            .sum();
        let leave: Vec<C> = good
            .iter()
            .map(|&b| {
                let w = counts[b] / total;
                (mean[j] - block[b * nr * 61 + j] * w) / (1. - w)
            })
            .collect();
        let center = leave.iter().sum::<C>() / good.len() as f64;
        let fac = (good.len() - 1) as f64 / good.len() as f64;
        sr[j] = (fac
            * leave
                .iter()
                .map(|v| (v.re - center.re).powi(2))
                .sum::<f64>())
        .sqrt();
        si[j] = (fac
            * leave
                .iter()
                .map(|v| (v.im - center.im).powi(2))
                .sum::<f64>())
        .sqrt();
    }
    let temp = path.with_extension("tmp.h5");
    let out = hdf5::File::create(&temp)?;
    for (name, data, shape) in [
        (
            "acf_real_adu2",
            mean.iter().map(|v| v.re).collect::<Vec<_>>(),
            vec![nr, 61],
        ),
        (
            "acf_imag_adu2",
            mean.iter().map(|v| v.im).collect(),
            vec![nr, 61],
        ),
        ("acf_real_sem_adu2", sr, vec![nr, 61]),
        ("acf_imag_sem_adu2", si, vec![nr, 61]),
        (
            "block_acf_real_adu2",
            block.iter().map(|v| v.re).collect(),
            vec![nb, nr, 61],
        ),
        (
            "block_acf_imag_adu2",
            block.iter().map(|v| v.im).collect(),
            vec![nb, nr, 61],
        ),
    ] {
        out.new_dataset::<f64>()
            .shape(shape)
            .create(name)?
            .write_raw(&data)?;
    }
    out.new_dataset_builder()
        .with_data(
            &indices
                .iter()
                .map(|&i| nominal_range(i))
                .collect::<Vec<_>>(),
        )
        .create("slant_range_km")?;
    out.new_dataset_builder()
        .with_data(&indices.iter().map(|&i| i as u64).collect::<Vec<_>>())
        .create("native_gate_index")?;
    out.new_attr::<u64>()
        .create("first_stable_sample")?
        .write_scalar(&30)?;
    out.new_attr_builder().with_data("Recorded stable pairs k=max(30,i-l)..i+20; actual pair count; original pulse mask and block highest-gate subtraction; 200/(200-l) scaling.").create("estimator")?;
    drop(out);
    fs::rename(temp, path)?;
    println!(
        "window {t}: {n} pulses, {} retained; early products complete",
        total
    );
    Ok(())
}
