//! MPI-launched local campaign driver. Open MPI supplies rank/size via its
//! environment; ranks partition independent time windows without network I/O.
use anyhow::{bail, ensure, Context, Result};
use serde::{Deserialize, Serialize};
use serde_json::{json, Value};
use std::{
    fs,
    io::{Read, Seek, SeekFrom, Write},
    path::{Path, PathBuf},
    process::{Command, Stdio},
    time::{Duration, SystemTime, UNIX_EPOCH},
};
use syisr_long_pulse::reader::Header;
const FRAME: u64 = 31288;
#[derive(Serialize, Deserialize, Clone)]
struct Source {
    path: PathBuf,
    frames: usize,
    start: f64,
    last: f64,
}
#[derive(Serialize, Deserialize, Clone)]
struct Segment {
    path: PathBuf,
    first_frame: usize,
    stop_frame: usize,
}
#[derive(Serialize, Deserialize, Clone)]
struct Window {
    index: usize,
    utc_start: f64,
    utc_end: f64,
    pulses: usize,
    segments: Vec<Segment>,
}
#[derive(Serialize, Deserialize)]
struct Plan {
    windows: Vec<Window>,
    reference_peak_power_range2: f64,
    #[serde(default = "fourteen")]
    mpi_ranks: usize,
}
fn fourteen() -> usize {
    14
}
fn now() -> f64 {
    SystemTime::now()
        .duration_since(UNIX_EPOCH)
        .unwrap()
        .as_secs_f64()
}
fn atomic_json(path: &Path, value: &impl Serialize) -> Result<()> {
    let p = path.with_extension(format!("{}.tmp", std::process::id()));
    fs::write(&p, serde_json::to_vec_pretty(value)?)?;
    fs::rename(p, path)?;
    Ok(())
}
fn read_json<T: for<'a> Deserialize<'a>>(p: &Path) -> Result<T> {
    Ok(serde_json::from_slice(&fs::read(p)?)?)
}
fn raw_header(f: &mut fs::File, offset: u64) -> Result<Header> {
    f.seek(SeekFrom::Start(offset))?;
    let mut b = [0; 1024];
    f.read_exact(&mut b)?;
    let h = Header::parse(&b)?;
    ensure!(
        h.samples == 3782
            && h.sample_rate_hz == 1e5
            && h.waveform == 4
            && h.pulse_width_us == 2000
            && h.gate_front_us == 2001
            && h.ipp_us == 40000
            && h.elevation_raw == 16384
            && h.tx_elevation_raw == 16384,
        "unsupported campaign layout"
    );
    Ok(h)
}
fn plan(data: &Path, root: &Path, reference: &Path) -> Result<()> {
    fs::create_dir_all(root.join("windows"))?;
    let mut files = vec![];
    for e in fs::read_dir(data)? {
        let e = e?;
        let name = e.file_name().to_string_lossy().into_owned();
        if name.starts_with("SY_") && name.contains(".dat1-1-") {
            files.push(e.path());
        }
    }
    ensure!(!files.is_empty(), "no supported SYISR raw files");
    let mut sources = vec![];
    for p in files {
        let mut f = fs::File::open(&p)?;
        let len = f.metadata()?.len();
        ensure!(
            len % FRAME == 0 && len >= FRAME,
            "partial frame in {}",
            p.display()
        );
        let h = raw_header(&mut f, 0)?;
        let last = raw_header(&mut f, len - FRAME)?;
        sources.push(Source {
            path: fs::canonicalize(p)?,
            frames: (len / FRAME) as usize,
            start: h.header_clock_unix_s - 28800.,
            last: last.header_clock_unix_s - 28800.,
        });
    }
    sources.sort_by(|a, b| a.start.total_cmp(&b.start));
    for pair in sources.windows(2) {
        ensure!(
            (pair[1].start - pair[0].last - 0.04).abs() < 26e-6,
            "file-to-file clock gap"
        );
    }
    let start = sources[0].start;
    let end = sources.last().unwrap().last + 0.04;
    let mut windows = vec![];
    let mut lo = (start / 1800.).floor() * 1800.;
    while lo < end {
        let hi = lo + 1800.;
        let mut segments = vec![];
        let mut pulses = 0;
        for s in &sources {
            let first = ((lo - s.start) / 0.04 - 1e-6).ceil().max(0.) as usize;
            let stop = (((hi - s.start) / 0.04 - 1e-6).ceil().max(0.) as usize).min(s.frames);
            if stop > first {
                pulses += stop - first;
                segments.push(Segment {
                    path: s.path.clone(),
                    first_frame: first,
                    stop_frame: stop,
                });
            }
        }
        if pulses > 0 {
            windows.push(Window {
                index: windows.len(),
                utc_start: lo.max(start),
                utc_end: hi.min(end),
                pulses,
                segments,
            });
        }
        lo = hi;
    }
    ensure!(
        windows.iter().map(|w| w.pulses).sum::<usize>()
            == sources.iter().map(|s| s.frames).sum::<usize>(),
        "pulse assignment does not conserve input frames"
    );
    let h = hdf5::File::open(reference)?;
    let r = h.dataset("slant_range_km")?.read_raw::<f64>()?;
    let power = h.dataset("acf_real_adu2")?.read_raw::<f64>()?;
    let n = h.dataset("lag_us")?.size();
    let peak = (0..r.len())
        .filter(|&i| r[i] >= 300. && r[i] < 650.)
        .map(|i| power[i * n] * r[i] * r[i])
        .fold(0f64, f64::max);
    ensure!(peak > 0., "no positive F-region peak in reference");
    syisr_long_pulse::geometry::write(
        &root.join("fit-geometry.h5"),
        &r,
        (start + end) / 2.,
        350.,
        1450.,
        50.,
    )?;
    atomic_json(&root.join("inventory.json"), &sources)?;
    atomic_json(
        &root.join("plan.json"),
        &Plan {
            windows,
            reference_peak_power_range2: peak,
            mpi_ranks: 14,
        },
    )?;
    Ok(())
}
fn run(root: &Path) -> Result<()> {
    let plan: Plan = read_json(&root.join("plan.json"))?;
    let rank: usize = std::env::var("OMPI_COMM_WORLD_RANK")
        .unwrap_or_else(|_| "0".into())
        .parse()?;
    let size: usize = std::env::var("OMPI_COMM_WORLD_SIZE")
        .unwrap_or_else(|_| "1".into())
        .parse()?;
    let bins = std::env::current_exe()?.parent().unwrap().to_owned();
    std::env::set_var("RAYON_NUM_THREADS", "1");
    if rank == 0 {
        atomic_json(
            &root.join("run.json"),
            &json!({"ranks":size,"started":now(),"rust_only":true}),
        )?;
    }
    let finished = root.join(format!("rank-{rank:03}.finished.json"));
    let _ = fs::remove_file(&finished);
    for w in plan.windows.iter().skip(rank).step_by(size) {
        let out = root.join("windows").join(format!("{:03}", w.index));
        fs::create_dir_all(&out)?;
        if out.join("done.json").exists() {
            continue;
        }
        let _ = fs::remove_file(out.join("failed.json"));
        atomic_json(
            &out.join("running.json"),
            &json!({"rank":rank,"started":now(),"window":w}),
        )?;
        let begin = now();
        let result = (|| -> Result<()> {
            let raw = out.join("input.dat");
            if fs::metadata(&raw).map(|m| m.len()).unwrap_or(0) != w.pulses as u64 * FRAME {
                let mut target = fs::File::create(&raw)?;
                let mut buffer = vec![0u8; 8 * 1024 * 1024];
                for s in &w.segments {
                    let mut input = fs::File::open(&s.path)?;
                    input.seek(SeekFrom::Start(s.first_frame as u64 * FRAME))?;
                    let mut left = (s.stop_frame - s.first_frame) as u64 * FRAME;
                    while left > 0 {
                        let n = left.min(buffer.len() as u64) as usize;
                        input.read_exact(&mut buffer[..n])?;
                        target.write_all(&buffer[..n])?;
                        left -= n as u64;
                    }
                }
            }
            let log = fs::OpenOptions::new()
                .create(true)
                .append(true)
                .open(out.join("run.log"))?;
            let lpm = out.join("lpm.partial.h5");
            let fits = out.join("fits.partial.h5");
            for (binary, args) in [
                (
                    "process",
                    vec![raw.clone(), lpm.clone(), PathBuf::from("--screen-bright")],
                ),
                (
                    "fit",
                    vec![
                        lpm.clone(),
                        root.join("fit-geometry.h5"),
                        fits.clone(),
                        PathBuf::from("1"),
                    ],
                ),
            ] {
                let status = Command::new(bins.join(binary))
                    .args(args)
                    .env("RAYON_NUM_THREADS", "1")
                    .env(
                        "SYISR_PEAK_POWER_RANGE2",
                        plan.reference_peak_power_range2.to_string(),
                    )
                    .stdout(Stdio::from(log.try_clone()?))
                    .stderr(Stdio::from(log.try_clone()?))
                    .status()?;
                ensure!(status.success(), "{binary} failed with {status}");
            }
            fs::rename(lpm, out.join("lpm.h5"))?;
            fs::rename(fits, out.join("fits.h5"))?;
            atomic_json(
                &out.join("done.json"),
                &json!({"rank":rank,"window":w,"elapsed_s":now()-begin,"completed_at":now()}),
            )?;
            fs::remove_file(raw)?;
            Ok(())
        })();
        let _ = fs::remove_file(out.join("running.json"));
        if let Err(e) = result {
            atomic_json(
                &out.join("failed.json"),
                &json!({"rank":rank,"window":w,"error":format!("{e:#}")}),
            )?;
        }
    }
    atomic_json(&finished, &json!({"rank":rank,"finished":now()}))?;
    Ok(())
}
fn read_attr(g: &hdf5::Group, name: &str) -> Result<Value> {
    Ok(serde_json::from_slice(&g.attr(name)?.read_raw::<u8>()?)?)
}
fn collect(root: &Path) -> Result<Value> {
    let plan: Plan = read_json(&root.join("plan.json"))?;
    let geo = hdf5::File::open(root.join("fit-geometry.h5"))?;
    let r = geo.dataset("slant_range_km")?.read_raw::<f64>()?;
    let z = geo.dataset("altitude_km")?.read_raw::<f64>()?;
    let nt = plan.windows.len();
    let nr = r.len();
    let keys = [
        "h_fraction",
        "he_fraction",
        "o_fraction",
        "ne_m3",
        "te_k",
        "ti_k",
        "reduced_chi2",
        "sigma_h",
        "sigma_he",
        "sigma_ti_k",
        "sigma_te_k",
        "background_adjustment_adu2",
    ];
    let mut data: std::collections::BTreeMap<&str, Vec<f64>> =
        keys.iter().map(|&k| (k, vec![f64::NAN; nr * nt])).collect();
    let mut flags = vec![255u8; nr * nt];
    let mut density_valid = vec![0u8; nr * nt];
    let mut complete = vec![0u8; nt];
    let mut retained = vec![0.; nt];
    let mut failures = vec![];
    for w in &plan.windows {
        let t = w.index;
        let out = root.join("windows").join(format!("{t:03}"));
        if !out.join("done.json").exists() {
            if out.join("failed.json").exists() {
                failures.push(read_json::<Value>(&out.join("failed.json"))?);
            }
            continue;
        }
        let h = hdf5::File::open(out.join("lpm.h5"))?;
        let rr = h.dataset("slant_range_km")?.read_raw::<f64>()?;
        let ar = h.dataset("acf_real_adu2")?.read_raw::<f64>()?;
        let se = h.dataset("acf_real_sem_adu2")?.read_raw::<f64>()?;
        retained[t] = h
            .dataset("block_retained_pulses")?
            .read_raw::<f64>()?
            .iter()
            .sum();
        for j in 0..nr {
            let ri = (0..rr.len())
                .min_by(|&a, &b| (rr[a] - r[j]).abs().total_cmp(&(rr[b] - r[j]).abs()))
                .unwrap();
            data.get_mut("ne_m3").unwrap()[j * nt + t] =
                ar[ri * 61] * r[j] * r[j] / plan.reference_peak_power_range2 * 1e12;
            density_valid[j * nt + t] = u8::from(ar[ri * 61] > 3. * se[ri * 61]);
        }
        let f = hdf5::File::open(out.join("fits.h5"))?;
        for name in f.member_names()? {
            if !name.starts_with("range_") {
                continue;
            }
            let g = f.group(&name)?;
            let m = read_attr(&g, "metadata_json")?;
            let fit = &m["fit"];
            let range = m["slant_range_km"].as_f64().context("missing range")?;
            let j = (0..nr)
                .min_by(|&a, &b| (r[a] - range).abs().total_cmp(&(r[b] - range).abs()))
                .unwrap();
            let i = j * nt + t;
            for k in [
                "h_fraction",
                "he_fraction",
                "o_fraction",
                "te_k",
                "ti_k",
                "background_adjustment_adu2",
            ] {
                data.get_mut(k).unwrap()[i] = fit[k].as_f64().unwrap_or(f64::NAN);
            }
            let reduced =
                fit["chi_square"].as_f64().unwrap() / fit["dof"].as_u64().unwrap().max(1) as f64;
            data.get_mut("reduced_chi2").unwrap()[i] = reduced;
            let mut flag = 0u8;
            if !fit["status"]
                .as_str()
                .unwrap_or("")
                .starts_with("converged")
            {
                flag |= 1;
            }
            if reduced > 3. {
                flag |= 2;
            }
            let ti = fit["ti_k"].as_f64().unwrap();
            let te = fit["te_k"].as_f64().unwrap();
            if ti <= 301. || ti >= 4999. || te <= 301. || te >= 11999. {
                flag |= 4;
            }
            if density_valid[i] == 0 || fit["signal_adu2"].as_f64().unwrap() <= 0. {
                flag |= 8;
            }
            if let Some(errors) = m["conditional_local_sigma_H_He_Ti_Te"].as_array() {
                for (k, v) in ["sigma_h", "sigma_he", "sigma_ti_k", "sigma_te_k"]
                    .iter()
                    .zip(errors)
                {
                    data.get_mut(k).unwrap()[i] = v.as_f64().unwrap_or(f64::NAN);
                }
            } else {
                flag |= 16;
            }
            flags[i] = flag;
        }
        complete[t] = 1;
    }
    let stage = root.join(format!("campaign.{}.partial.h5", std::process::id()));
    let f = hdf5::File::create(&stage)?;
    for (k, v) in data {
        f.new_dataset::<f64>()
            .shape([nr, nt])
            .create(k)?
            .write_raw(&v)?;
    }
    f.new_dataset::<u8>()
        .shape([nr, nt])
        .create("quality_flags")?
        .write_raw(&flags)?;
    f.new_dataset::<u8>()
        .shape([nr, nt])
        .create("density_valid")?
        .write_raw(&density_valid)?;
    f.new_dataset_builder()
        .with_data(&complete)
        .create("window_complete")?;
    let mut edges = vec![plan.windows[0].utc_start];
    edges.extend(plan.windows.iter().map(|w| w.utc_end));
    for (k, v) in [
        ("slant_range_km", r),
        ("altitude_km", z),
        ("time_edges_utc_s", edges),
        (
            "time_utc_s",
            plan.windows
                .iter()
                .map(|w| (w.utc_start + w.utc_end) / 2.)
                .collect(),
        ),
        (
            "input_pulse_count",
            plan.windows.iter().map(|w| w.pulses as f64).collect(),
        ),
        ("retained_pulse_count", retained.clone()),
    ] {
        f.new_dataset_builder().with_data(&v).create(k)?;
    }
    f.new_attr_builder().with_data("Provisional P(r) r^2 proxy on fixed noon-derived scale, F-peak 1e12 m^-3; no absolute calibration or Te/Ti power correction").create("density")?;
    f.new_attr_builder().with_data("1 nonconverged; 2 reduced chi2>3; 4 temperature boundary; 8 weak/nonpositive signal; 16 singular covariance; 255 missing").create("quality_bits")?;
    f.flush()?;
    drop(f);
    fs::rename(stage, root.join("campaign.h5"))?;
    let completed: Vec<usize> = complete
        .iter()
        .enumerate()
        .filter_map(|(i, &v)| (v > 0).then_some(i))
        .collect();
    let contiguous = complete.iter().take_while(|&&v| v > 0).count();
    let latest = completed.last().copied();
    let status = json!({"total_windows":nt,"completed_windows":completed.len(),"completed_indices":completed,"failed_windows":failures.len(),"failures":failures,"latest_complete_index":latest,"contiguous_through_utc_s":if contiguous>0{Some(plan.windows[contiguous-1].utc_end)}else{None},"latest_completed_utc_s":latest.map(|j|plan.windows[j].utc_end),"input_pulses":plan.windows.iter().filter(|w|complete[w.index]>0).map(|w|w.pulses).sum::<usize>(),"retained_pulses":retained.iter().sum::<f64>(),"updated_unix_s":now()});
    atomic_json(&root.join("status.json"), &status)?;
    Ok(status)
}
fn watch(root: &Path) -> Result<()> {
    let bins = std::env::current_exe()?.parent().unwrap().to_owned();
    let mut prior = usize::MAX;
    loop {
        let status = collect(root)?;
        let completed = status["completed_windows"].as_u64().unwrap() as usize;
        let failed = status["failed_windows"].as_u64().unwrap() as usize;
        if completed > 0 && completed != prior {
            let latest = status["latest_complete_index"].as_u64().unwrap();
            let lpm = root
                .join("windows")
                .join(format!("{latest:03}"))
                .join("lpm.h5");
            for (binary, args) in [
                (
                    "plot-campaign",
                    vec![root.join("campaign.h5"), root.join("time-altitude")],
                ),
                ("plot", vec![lpm.clone(), root.join("latest-lag-profile")]),
                (
                    "plot",
                    vec![
                        lpm,
                        root.join("latest-lag-profile-full"),
                        PathBuf::from("2500"),
                    ],
                ),
            ] {
                let s = Command::new(bins.join(binary)).args(args).status()?;
                ensure!(s.success(), "{binary} renderer failed");
            }
            println!("{}", serde_json::to_string(&status)?);
        }
        prior = completed;
        if completed + failed == status["total_windows"].as_u64().unwrap() as usize {
            break;
        }
        std::thread::sleep(Duration::from_secs(30));
    }
    Ok(())
}
fn main() -> Result<()> {
    let a: Vec<_> = std::env::args_os().collect();
    ensure!(a.len()>=3,"campaign plan RAW_DIR OUTPUT_DIR REFERENCE_LPM.h5 | run OUTPUT_DIR | collect OUTPUT_DIR | watch OUTPUT_DIR");
    match a[1].to_str().unwrap_or("") {
        "plan" => {
            ensure!(
                a.len() == 5,
                "plan requires raw directory, output directory and reference HDF5"
            );
            plan(Path::new(&a[2]), Path::new(&a[3]), Path::new(&a[4]))
        }
        "run" => run(Path::new(&a[2])),
        "collect" => {
            println!("{}", collect(Path::new(&a[2]))?);
            Ok(())
        }
        "watch" => watch(Path::new(&a[2])),
        _ => bail!("unknown command"),
    }
}
