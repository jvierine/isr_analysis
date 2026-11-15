//! Raw SYISR uncoded-LP lag products with auditable voltage sample masks.
use anyhow::{ensure, Result};
use num_complex::Complex32 as C;
use rayon::prelude::*;
use serde_json::{json, Value};
use std::{
    fs::{self, File},
    io::{BufReader, BufWriter, Read, Write},
    path::{Path, PathBuf},
};
use syisr_hplus_night::pulse_screen::{screen, Settings};
const N: usize = 1324;
const STRIDE: usize = 1024 + 8 * N;
const LAGS: usize = 41;
fn u16(b: &[u8], k: usize) -> u16 {
    u16::from_le_bytes(b[k..k + 2].try_into().unwrap())
}
fn days(y: i64, m: i64, d: i64) -> i64 {
    let y = y - if m <= 2 { 1 } else { 0 };
    let era = if y >= 0 { y } else { y - 399 } / 400;
    let yo = y - era * 400;
    let mp = m + if m > 2 { -3 } else { 9 };
    let doy = (153 * mp + 2) / 5 + d - 1;
    let doe = yo * 365 + yo / 4 - yo / 100 + doy;
    era * 146097 + doe - 719468
}
fn timestamp(h: &[u8]) -> Result<f64> {
    ensure!(
        (1..=12).contains(&h[12])
            && (1..=31).contains(&h[15])
            && h[14] < 24
            && h[17] < 60
            && h[16] < 60,
        "invalid calendar header"
    );
    Ok(
        days(2000 + h[13] as i64, h[12] as i64, h[15] as i64) as f64 * 86400.
            + h[14] as f64 * 3600.
            + h[17] as f64 * 60.
            + h[16] as f64
            + u16(h, 18) as f64 * 25e-6
            - 28800.,
    )
}
struct Dwell {
    start: f64,
    end: f64,
    az: u16,
    el: u16,
    txaz: u16,
    txel: u16,
    x: Vec<C>,
    saturated: Vec<bool>,
    origins: Vec<(usize, u64)>,
}
impl Dwell {
    fn new(t: f64, h: &[u8]) -> Self {
        Self {
            start: t,
            end: t,
            az: u16(h, 856),
            el: u16(h, 858),
            txaz: u16(h, 868),
            txel: u16(h, 870),
            x: vec![],
            saturated: vec![],
            origins: vec![],
        }
    }
    fn len(&self) -> usize {
        self.saturated.len()
    }
}
struct Record {
    meta: Value,
    counts: Vec<u32>,
    re: Vec<f32>,
    im: Vec<f32>,
    sr: Vec<f32>,
    si: Vec<f32>,
    qa: Option<Value>,
}
fn process(d: Dwell, samples: &[usize], settings: &Settings) -> Record {
    let np = d.len();
    let mut sc = screen(&d.x, np, N, settings);
    for p in 0..np {
        if d.saturated[p] {
            for k in 0..N {
                sc.mask[p * N + k] |= 8;
            }
        }
    }
    let mut mask_runs = vec![];
    let mut rejected = 0;
    for p in 0..np {
        let mut k = 0;
        let mut any = false;
        while k < N {
            if sc.mask[p * N + k] == 0 {
                k += 1;
                continue;
            }
            let a = k;
            let mut reason = 0;
            while k < N && sc.mask[p * N + k] != 0 {
                reason |= sc.mask[p * N + k];
                k += 1;
            }
            mask_runs.push(json!([p, a, k, reason]));
            any = true;
        }
        if any {
            rejected += 1;
        }
    }
    let nr = samples.len();
    let mut counts = vec![0u32; LAGS * nr];
    let mut re = vec![0f64; LAGS * nr];
    let mut im = vec![0f64; LAGS * nr];
    let mut re2 = vec![0f64; LAGS * nr];
    let mut im2 = vec![0f64; LAGS * nr];
    for p in 0..np {
        let x = &d.x[p * N..(p + 1) * N];
        let mask = &sc.mask[p * N..(p + 1) * N];
        for lag in 0..LAGS {
            for (g, &k) in samples.iter().enumerate() {
                if k + lag >= N || mask[k] != 0 || mask[k + lag] != 0 {
                    continue;
                }
                let z = x[k] * x[k + lag].conj();
                let i = lag * nr + g;
                counts[i] += 1;
                let a = z.re as f64;
                let b = z.im as f64;
                re[i] += a;
                im[i] += b;
                re2[i] += a * a;
                im2[i] += b * b;
            }
        }
    }
    let mut rr = vec![0f32; re.len()];
    let mut ii = rr.clone();
    let mut sr = rr.clone();
    let mut si = rr.clone();
    for i in 0..re.len() {
        let n = counts[i] as f64;
        if n > 0. {
            rr[i] = (re[i] / n) as f32;
            ii[i] = (im[i] / n) as f32;
        }
        if n > 1. {
            sr[i] = ((re2[i] - re[i] * re[i] / n).max(0.) / (n * (n - 1.))).sqrt() as f32;
            si[i] = ((im2[i] - im[i] * im[i] / n).max(0.) / (n * (n - 1.))).sqrt() as f32;
        }
    }
    let masked = sc.mask.iter().filter(|&&v| v != 0).count();
    let mut qa = None;
    if masked > 0 {
        let mut after = vec![0f32; sc.time_bins * sc.range_bins];
        for t in 0..sc.time_bins {
            let a = t * settings.slow_pulses;
            let b = (a + settings.slow_pulses).min(np);
            for g in 0..sc.range_bins {
                let mut sum = 0.;
                let mut n = 0;
                for p in a..b {
                    for k in g * settings.range_bin..(g + 1) * settings.range_bin {
                        if sc.mask[p * N + k] == 0 {
                            sum += d.x[p * N + k].norm_sqr();
                            n += 1;
                        }
                    }
                }
                after[t * sc.range_bins + g] = if n > 0 {
                    sum / n as f32 / sc.reference[g]
                } else {
                    -1.
                };
            }
        }
        qa = Some(
            json!({"start":d.start,"az_raw":d.az,"el_raw":d.el,"pulses":np,"range_bins":sc.range_bins,"time_bins":sc.time_bins,"before_ratio":sc.power_ratio,"after_ratio":after,"mask_fraction":masked as f64/(np*N)as f64}),
        );
    }
    Record {
        meta: json!({"start_utc_s":d.start,"end_utc_s":d.end,"azimuth_raw":d.az,"elevation_raw":d.el,"tx_azimuth_raw":d.txaz,"tx_elevation_raw":d.txel,"pulses":np,"pulses_with_any_mask":rejected,"saturated_pulses":d.saturated.iter().filter(|&&x|x).count(),"masked_voltage_samples":masked,"mask_runs":mask_runs,"detector_flagged_cells":sc.flagged_cells,"source_first":d.origins.first(),"source_last":d.origins.last()}),
        counts,
        re: rr,
        im: ii,
        sr,
        si,
        qa,
    }
}
fn save(writer: &mut BufWriter<File>, record: &Record) -> Result<()> {
    let h = serde_json::to_vec(&record.meta)?;
    writer.write_all(&(h.len() as u32).to_le_bytes())?;
    writer.write_all(&h)?;
    for &c in &record.counts {
        writer.write_all(&c.to_le_bytes())?;
    }
    for array in [&record.re, &record.im, &record.sr, &record.si] {
        for v in array {
            writer.write_all(&v.to_le_bytes())?;
        }
    }
    Ok(())
}
fn flush(
    pending: &mut Vec<Dwell>,
    samples: &[usize],
    settings: &Settings,
    writer: &mut BufWriter<File>,
    qas: &mut Vec<Value>,
    totals: &mut Value,
) -> Result<()> {
    if pending.is_empty() {
        return Ok(());
    }
    let records: Vec<_> = std::mem::take(pending)
        .into_par_iter()
        .map(|d| process(d, samples, settings))
        .collect();
    for r in records {
        save(writer, &r)?;
        for key in [
            "pulses",
            "pulses_with_any_mask",
            "saturated_pulses",
            "masked_voltage_samples",
        ] {
            let value = r.meta[key].as_u64().unwrap();
            totals[key] = json!(totals[key].as_u64().unwrap_or(0) + value);
        }
        totals["dwells"] = json!(totals["dwells"].as_u64().unwrap_or(0) + 1);
        if let Some(q) = r.qa {
            qas.push(q);
            qas.sort_by(|a, b| {
                b["mask_fraction"]
                    .as_f64()
                    .unwrap()
                    .total_cmp(&a["mask_fraction"].as_f64().unwrap())
            });
            qas.truncate(16);
        }
    }
    writer.flush()?;
    Ok(())
}
fn main() -> Result<()> {
    let args: Vec<_> = std::env::args().collect();
    ensure!(
        args.len() >= 3,
        "source-directory output-directory [file-limit]"
    );
    let source = Path::new(&args[1]);
    let out = Path::new(&args[2]);
    fs::create_dir_all(out)?;
    let mount = std::process::Command::new("findmnt")
        .args(["-n", "-o", "SOURCE,FSTYPE", "--target"])
        .arg(source)
        .output()?;
    ensure!(
        mount.status.success()
            && String::from_utf8_lossy(&mount.stdout)
                .split_whitespace()
                .collect::<Vec<_>>()
                == vec!["/dev/sda1", "ext4"],
        "wrong/missing SYISR source mount"
    );
    let mut files: Vec<(f64, PathBuf)> = vec![];
    for p in fs::read_dir(source)? {
        let p = p?.path();
        if p.is_file() {
            let mut f = File::open(&p)?;
            let mut h = [0u8; 1024];
            f.read_exact(&mut h)?;
            files.push((timestamp(&h)?, p));
        }
    }
    files.sort_by(|a, b| a.0.total_cmp(&b.0));
    if let Some(v) = args.get(3) {
        files.truncate(v.parse()?);
    }
    let settings = Settings::default();
    let samples: Vec<usize> = (130..1051).step_by(10).chain(1100..1250).collect();
    let global = json!({"schema":"syisr-clean-lp-v1","samples":samples,"lags":LAGS,"detector":settings,"array_format":"counts uint32; mean real, mean imaginary, SEM real, SEM imaginary float32; all little endian, lag-major","clock":"raw header minus 8 hours","pair_rule":"Both voltage samples must be unmasked; count and SEM retained for each measurement","classification":"Conservative nonthermal echo candidates, including satellite/RFI; no orbital object classification"});
    let mut writer = BufWriter::with_capacity(
        8 * 1024 * 1024,
        File::create(out.join("cleaned-products.bin"))?,
    );
    writer.write_all(b"SYLPC001")?;
    let bytes = serde_json::to_vec(&global)?;
    writer.write_all(&(bytes.len() as u32).to_le_bytes())?;
    writer.write_all(&bytes)?;
    let mut dwell: Option<Dwell> = None;
    let mut pending = vec![];
    let mut qas = vec![];
    let mut manifest = vec![];
    let mut totals = json!({});
    let mut previous = None;
    let mut large_steps = 0;
    let mut nonmonotonic = 0;
    for (fi, (_, path)) in files.iter().enumerate() {
        let size = fs::metadata(path)?.len();
        ensure!(size % STRIDE as u64 == 0, "unaligned LP file");
        let mut f = BufReader::with_capacity(8 * 1024 * 1024, File::open(path)?);
        let frames = size / STRIDE as u64;
        let mut first = None;
        let mut last = None;
        for index in 0..frames {
            let mut h = [0u8; 1024];
            f.read_exact(&mut h)?;
            ensure!(
                &h[..4] == b"\x55\x55\xaa\xaa"
                    && &h[1020..] == b"\xaa\x55\x55\xaa"
                    && u32::from_le_bytes(h[768..772].try_into().unwrap()) as usize == N
                    && h[36] == 0x41
                    && h[764] == 1
                    && u16(&h, 980) == 16000
                    && u16(&h, 988) == 480
                    && u16(&h, 766) == 767,
                "invalid/incompatible LP frame {fi}:{index}"
            );
            let t = timestamp(&h)?;
            if let Some(old) = previous {
                let step: f64 = t - old;
                if step <= 0. {
                    nonmonotonic += 1;
                }
                if (step - 0.016).abs() > 0.0001 {
                    large_steps += 1;
                }
            }
            previous = Some(t);
            if first.is_none() {
                first = Some(t);
            }
            last = Some(t);
            let change = dwell.as_ref().map_or(true, |d| {
                d.az != u16(&h, 856)
                    || d.el != u16(&h, 858)
                    || d.txaz != u16(&h, 868)
                    || d.txel != u16(&h, 870)
                    || (t - d.end - 0.016).abs() > 0.0002
                    || (d.start / 1800.).floor() != (t / 1800.).floor()
                    || d.len() >= 2500
            });
            if change {
                if let Some(d) = dwell.take() {
                    pending.push(d);
                }
                if pending.len() >= 8 {
                    flush(
                        &mut pending,
                        &samples,
                        &settings,
                        &mut writer,
                        &mut qas,
                        &mut totals,
                    )?;
                }
                dwell = Some(Dwell::new(t, &h));
            }
            let mut raw = vec![0u8; N * 8];
            f.read_exact(&mut raw)?;
            let d = dwell.as_mut().unwrap();
            let mut saturated = false;
            for chunk in raw.chunks_exact(8) {
                let i = i32::from_le_bytes(chunk[..4].try_into().unwrap());
                let q = i32::from_le_bytes(chunk[4..].try_into().unwrap());
                if i == i32::MIN || i == i32::MAX || q == i32::MIN || q == i32::MAX {
                    saturated = true;
                }
                d.x.push(C::new(q as f32, i as f32));
            }
            d.end = t;
            d.saturated.push(saturated);
            d.origins.push((fi, index));
        }
        manifest.push(
            json!({"path":path,"bytes":size,"frames":frames,"first_utc_s":first,"last_utc_s":last}),
        );
        eprintln!("{}", json!({"file":fi+1,"of":files.len(),"totals":totals}));
    }
    if let Some(d) = dwell.take() {
        pending.push(d);
    }
    flush(
        &mut pending,
        &samples,
        &settings,
        &mut writer,
        &mut qas,
        &mut totals,
    )?;
    totals["nonmonotonic_clock_steps"] = json!(nonmonotonic);
    totals["clock_steps_departing_ipp_by_over_100us"] = json!(large_steps);
    fs::write(
        out.join("screening-summary.json"),
        serde_json::to_vec_pretty(
            &json!({"totals":totals,"files":manifest,"configuration":global}),
        )?,
    )?;
    fs::write(
        out.join("screening-examples.json"),
        serde_json::to_vec(&qas)?,
    )?;
    println!("{}", totals);
    Ok(())
}
#[cfg(test)]
mod tests {
    use super::*;
    #[test]
    fn utc_offset_and_date() {
        assert_eq!(days(2023, 11, 15), 19676);
        let mut h = [0u8; 1024];
        h[12] = 11;
        h[13] = 23;
        h[15] = 15;
        h[14] = 8;
        assert_eq!(timestamp(&h).unwrap(), 1700006400.);
    }
}
