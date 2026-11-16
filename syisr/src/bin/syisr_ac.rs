use anyhow::{ensure, Result};
use rayon::prelude::*;
use std::{
    fs,
    path::{Path, PathBuf},
    time::Duration,
};
use syisr_ac::{
    campaign::{self, Campaign},
    decode, fitting,
};
fn rank() -> (usize, usize) {
    let get = |a: &str, b: &str| {
        std::env::var(a)
            .or_else(|_| std::env::var(b))
            .ok()
            .and_then(|v| v.parse().ok())
    };
    (
        get("OMPI_COMM_WORLD_RANK", "PMI_RANK").unwrap_or(0),
        get("OMPI_COMM_WORLD_SIZE", "PMI_SIZE").unwrap_or(1),
    )
}
fn reference(
    path: &Path,
    d: &campaign::Dwell,
    heights: &[f64],
) -> Result<(Vec<Option<f64>>, Vec<Option<f64>>)> {
    use chrono::NaiveDateTime;
    use hdf5::types::FixedAscii;
    let f = hdf5::File::open(path)?;
    let ds = f.dataset("Fittedparameter")?;
    let shape = ds.shape();
    ensure!(shape.len() == 4 && shape[3] == 4, "unsupported L2 layout");
    let (nb, nt, nr) = (shape[0], shape[1], shape[2]);
    let dirs = f.dataset("Beamdir")?.read_raw::<f64>()?;
    let times = f.dataset("Time")?.read_raw::<FixedAscii<20>>()?;
    let beam = (0..nb)
        .min_by(|a, b| {
            ((dirs[*a * 2] - d.az).abs() + (dirs[*a * 2 + 1] - d.el).abs())
                .total_cmp(&((dirs[*b * 2] - d.az).abs() + (dirs[*b * 2 + 1] - d.el).abs()))
        })
        .unwrap();
    ensure!(
        (dirs[beam * 2] - d.az).abs() < 0.2 && (dirs[beam * 2 + 1] - d.el).abs() < 0.2,
        "no matching L2 beam"
    );
    let time = (0..nt)
        .filter_map(|j| {
            NaiveDateTime::parse_from_str(times[beam * nt + j].as_str().trim(), "%d-%b-%Y %H:%M:%S")
                .ok()
                .map(|t| (j, (t.and_utc().timestamp() as f64 - d.start).abs()))
        })
        .min_by(|a, b| a.1.total_cmp(&b.1))
        .ok_or_else(|| anyhow::anyhow!("L2 clock"))?;
    ensure!(time.1 < 90., "L2 timestamp not matched");
    let p = ds.read_raw::<f32>()?;
    let error = f.dataset("Error")?.read_raw::<f32>()?;
    let state = f.dataset("Fitstate")?.read_raw::<f32>()?;
    let geo = f.dataset("GEO")?.read_raw::<f64>()?;
    let base = (beam * nt + time.0) * nr;
    let mut ne = vec![];
    let mut se = vec![];
    for h in heights {
        let mut values: Vec<_> = (0..nr)
            .filter(|j| {
                state[base + j] == 0. && p[(base + j) * 4].is_finite() && p[(base + j) * 4] > 1e7
            })
            .map(|j| {
                (
                    (geo[(base + j) * 3 + 2] - h).abs(),
                    p[(base + j) * 4] as f64,
                    error[(base + j) * 4] as f64,
                )
            })
            .collect();
        values.sort_by(|a, b| a.0.total_cmp(&b.0));
        if let Some(v) = values.first().filter(|v| v.0 < 40.) {
            ne.push(Some(v.1));
            se.push(if v.2.is_finite() && v.2 >= 0. {
                Some(v.2)
            } else {
                None
            });
        } else {
            ne.push(None);
            se.push(None);
        }
    }
    Ok((ne, se))
}
fn main() -> Result<()> {
    let a: Vec<String> = std::env::args().collect();
    ensure!(a.len()>=3,"syisr_ac index RAW PHASE L2 OUT | decode ROOT [THREADS] | fit ROOT [BOOTSTRAPS] | status ROOT");
    match a[1].as_str() {
        "index" => {
            ensure!(a.len() == 6, "index rawdir phase l2-or-none campaignroot");
            let root = Path::new(&a[5]);
            fs::create_dir_all(root)?;
            let c = campaign::index(
                Path::new(&a[2]),
                Path::new(&a[3]),
                if a[4] == "none" {
                    None
                } else {
                    Some(a[4].clone())
                },
                &root.join("campaign.h5"),
            )?;
            println!(
                "{} files,{} dwells,{} time/beam jobs",
                c.files.len(),
                c.dwells.len(),
                c.jobs.len()
            );
        }
        "decode" => {
            let root = Path::new(&a[2]);
            let c: Campaign = campaign::load(&root.join("campaign.h5"))?;
            let codes = decode::phase(
                if root.join("m32-reference.bc").exists() {
                    root.join("m32-reference.bc")
                } else {
                    PathBuf::from(&c.phase_table)
                }
                .as_path(),
            )?;
            let threads = a.get(3).and_then(|v| v.parse().ok()).unwrap_or(4);
            rayon::ThreadPoolBuilder::new()
                .num_threads(threads)
                .build_global()?;
            fs::create_dir_all(root.join("decoded"))?;
            fs::create_dir_all(root.join("decode-failures"))?;
            c.dwells.par_iter().for_each(|d| {
                let path = root.join("decoded").join(format!("{:06}.h5", d.id));
                if path.exists() {
                    return;
                }
                let result = (|| -> Result<()> {
                    let mut p = decode::decode(d, &codes, &path, true)?;
                    if let Some(l2) = &c.l2 {
                        match reference(Path::new(l2), d, &p.heights) {
                            Ok((ne, se)) => {
                                p.reference_ne = ne;
                                p.reference_ne_error = se;
                                let temporary = root
                                    .join("decoded")
                                    .join(format!("{:06}.reference.h5", d.id));
                                campaign::save(&temporary, &p)?;
                                let old = hdf5::File::open(&path)?;
                                let mask = old.dataset("voltage_mask")?.read_raw::<u8>()?;
                                drop(old);
                                let f = hdf5::File::open_rw(&temporary)?;
                                f.new_dataset::<u8>()
                                    .shape((p.pulses, 1334))
                                    .deflate(4)
                                    .create("voltage_mask")?
                                    .write_raw(&mask)?;
                                drop(f);
                                fs::rename(temporary, &path)?;
                            }
                            Err(e) => eprintln!("dwell{} L2 unavailable:{e}", d.id),
                        }
                    }
                    println!(
                        "DECODE {} pulses{} mask{:0.6}",
                        d.id,
                        p.pulses,
                        p.mask_summary.masked as f64 / p.mask_summary.total as f64
                    );
                    Ok(())
                })();
                if let Err(e) = result {
                    let _ = fs::remove_file(&path);
                    let _ = fs::write(
                        root.join("decode-failures")
                            .join(format!("{:06}.txt", d.id)),
                        format!("{e:#}"),
                    );
                    eprintln!("FAILED dwell{}:{e:#}", d.id);
                }
            });
            fs::write(
                root.join("DECODE_COMPLETE"),
                "all manifest dwells attempted\n",
            )?;
        }
        "fit" => {
            let root = Path::new(&a[2]);
            let c: Campaign = campaign::load(&root.join("campaign.h5"))?;
            let codes = decode::phase(
                if root.join("m32-reference.bc").exists() {
                    root.join("m32-reference.bc")
                } else {
                    PathBuf::from(&c.phase_table)
                }
                .as_path(),
            )?;
            let nboot = a.get(3).and_then(|v| v.parse().ok()).unwrap_or(48);
            ensure!(nboot >= 32, "at least32 bootstrap draws required");
            let (rank, size) = rank();
            fs::create_dir_all(root.join("fits"))?;
            fs::create_dir_all(root.join("fit-failures"))?;
            let jobs: Vec<_> = c
                .jobs
                .iter()
                .enumerate()
                .filter(|(i, _)| i % size == rank)
                .map(|(_, j)| j)
                .collect();
            loop {
                let mut pending = 0;
                for j in &jobs {
                    let out = root.join("fits").join(format!("{}.h5", j.id));
                    let fail = root.join("fit-failures").join(format!("{}.txt", j.id));
                    if out.exists() || fail.exists() {
                        continue;
                    }
                    if !j
                        .dwells
                        .iter()
                        .all(|d| root.join("decoded").join(format!("{d:06}.h5")).exists())
                    {
                        if root.join("DECODE_COMPLETE").exists() {
                            fs::write(
                                fail,
                                "One or more raw dwell products failed or were unavailable\n",
                            )?;
                        } else {
                            pending += 1;
                        }
                        continue;
                    }
                    match fitting::fit(&c, j, root, &codes, nboot) {
                        Ok(p) => println!(
                            "FIT rank{rank}/{size} {} gates{} exposure{:0.1}",
                            j.id,
                            p.gates.len(),
                            p.exposure_s
                        ),
                        Err(e) => {
                            fs::write(fail, format!("{e:#}"))?;
                            eprintln!("FAILED {}:{e:#}", j.id);
                        }
                    }
                }
                if pending == 0 {
                    break;
                }
                std::thread::sleep(Duration::from_secs(15));
            }
            fs::write(
                root.join(format!("MPI_RANK_{rank:03}_COMPLETE")),
                format!("{size} ranks\n"),
            )?;
        }
        "status" => {
            let root = Path::new(&a[2]);
            let c: Campaign = campaign::load(&root.join("campaign.h5"))?;
            let count = |sub: &str| -> usize {
                fs::read_dir(root.join(sub))
                    .map(|d| {
                        d.filter_map(|v| v.ok())
                            .filter(|v| {
                                v.path()
                                    .extension()
                                    .is_some_and(|e| e == "h5" || e == "txt")
                            })
                            .count()
                    })
                    .unwrap_or(0)
            };
            println!(
                "dwells {}/{}; profiles {}/{}; decode failures{}; fit failures{}",
                count("decoded"),
                c.dwells.len(),
                count("fits"),
                c.jobs.len(),
                count("decode-failures"),
                count("fit-failures")
            );
        }
        _ => anyhow::bail!("unknown mode"),
    };
    Ok(())
}
