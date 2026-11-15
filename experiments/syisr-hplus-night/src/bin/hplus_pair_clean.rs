//! Conservative per-IPP range-power CFAR guard before averaging; raw files read-only.
use anyhow::{ensure, Result};
use num_complex::Complex32 as C;
use rayon::prelude::*;
use serde_json::{json, Value};
use std::{
    collections::BTreeSet,
    fs,
    io::{BufReader, Read},
};
use syisr_hplus_night::pulse_screen::{screen, Settings};
fn clock(b: &[u8]) -> Result<f64> {
    let d = chrono::NaiveDate::from_ymd_opt(2000 + b[13] as i32, b[12] as u32, b[15] as u32)
        .ok_or_else(|| anyhow::anyhow!("date"))?;
    let t = d
        .and_hms_opt(b[14] as u32, b[17] as u32, b[16] as u32)
        .ok_or_else(|| anyhow::anyhow!("clock"))?;
    Ok(t.and_utc().timestamp() as f64 + u16::from_le_bytes([b[18], b[19]]) as f64 * 25e-6 - 28800.)
}
struct Pulse {
    t: f64,
    z: Vec<C>,
    saturated: bool,
}
struct Sum {
    n: usize,
    counts: Vec<u32>,
    x: [Vec<f64>; 4],
}
impl Sum {
    fn new() -> Self {
        Self {
            n: 0,
            counts: vec![0; 41 * 1324],
            x: std::array::from_fn(|_| vec![0.; 41 * 1324]),
        }
    }
    fn add(mut self, p: &Pulse, mask: &[u8]) -> Self {
        self.n += 1;
        for l in 0..41 {
            for k in 0..1324 - l {
                if mask[k] != 0 || mask[k + l] != 0 {
                    continue;
                }
                self.counts[l * 1324 + k] += 1;
                let a = p.z[k];
                let b = p.z[k + l];
                let re = (a.re * b.re + a.im * b.im) as f64;
                let im = (a.im * b.re - a.re * b.im) as f64;
                let j = l * 1324 + k;
                self.x[0][j] += re;
                self.x[1][j] += im;
                self.x[2][j] += re * re;
                self.x[3][j] += im * im;
            }
        }
        self
    }
    fn merge(mut self, b: Self) -> Self {
        self.n += b.n;
        for (a, b) in self.counts.iter_mut().zip(&b.counts) {
            *a += b;
        }
        for i in 0..4 {
            for (a, b) in self.x[i].iter_mut().zip(b.x[i].iter()) {
                *a += b;
            }
        }
        self
    }
}
fn main() -> Result<()> {
    let a: Vec<_> = std::env::args().collect();
    ensure!(a.len() == 4, "night_products.h5 output.h5 UTC_hour");
    let hour = a[3].parse::<usize>()?;
    let input = hdf5::File::open(&a[1])?;
    let meta: Value =
        serde_json::from_slice(&input.dataset("metadata_json_utf8")?.read_raw::<u8>()?)?;
    let day = chrono::DateTime::parse_from_rfc3339("2023-11-15T00:00:00Z")?.timestamp() as f64;
    let lo = day + hour as f64 * 3600.;
    let hi = lo + 3600.;
    let selected: Vec<_> = meta["windows"]
        .as_array()
        .unwrap()
        .iter()
        .filter(|w| {
            let t = w["utc_start_s"].as_f64().unwrap();
            t >= lo && t < hi
        })
        .collect();
    let paths: BTreeSet<_> = selected
        .iter()
        .map(|w| w["source_raw"].as_str().unwrap().to_string())
        .collect();
    let mut pulses = vec![];
    let mut fileinfo = vec![];
    let mut total = 0;
    for path in paths {
        let mut f = BufReader::with_capacity(4 * 1024 * 1024, fs::File::open(&path)?);
        let mut b = [0_u8; 1024];
        loop {
            let got = f.read(&mut b[..1])?;
            if got == 0 {
                break;
            }
            f.read_exact(&mut b[1..])?;
            ensure!(
                u32::from_le_bytes(b[0..4].try_into().unwrap()) == 0xaaaa5555
                    && u32::from_le_bytes(b[1020..1024].try_into().unwrap()) == 0xaa5555aa,
                "bad header"
            );
            let n = u32::from_le_bytes(b[768..772].try_into().unwrap()) as usize;
            ensure!(
                n == 1324 && b[764] == 1 && b[36] == 0x41,
                "unexpected LP layout"
            );
            total += 1;
            let t = clock(&b)?;
            let zenith = u16::from_le_bytes(b[858..860].try_into().unwrap()) == 16384
                && u16::from_le_bytes(b[870..872].try_into().unwrap()) == 16384;
            if t < lo || t >= hi || !zenith {
                f.seek_relative((n * 8) as i64)?;
                continue;
            }
            let mut raw = vec![0_u8; n * 8];
            f.read_exact(&mut raw)?;
            let mut saturated = false;
            let z: Vec<C> = raw
                .chunks_exact(8)
                .map(|v| {
                    let im = i32::from_le_bytes(v[..4].try_into().unwrap());
                    let re = i32::from_le_bytes(v[4..].try_into().unwrap());
                    if re.unsigned_abs() > 2_100_000_000 || im.unsigned_abs() > 2_100_000_000 {
                        saturated = true;
                    }
                    C::new(re as f32, im as f32)
                })
                .collect();
            pulses.push(Pulse { t, z, saturated });
        }
        fileinfo.push(json!({"path":path,"bytes":fs::metadata(&path)?.len()}));
        eprintln!("{} selected zenith pulses after {}", pulses.len(), path);
    }
    ensure!(pulses.len() > 1000, "too few pulses");
    let expected: usize = selected
        .iter()
        .map(|w| w["pulse_count"].as_u64().unwrap() as usize)
        .sum();
    ensure!(
        pulses.len() == expected,
        "raw selected {} differs from existing product count {}",
        pulses.len(),
        expected
    );
    let settings = Settings::default();
    let mut masks = vec![];
    let mut pulse_times = vec![];
    let mut all_counts = vec![];
    let mut excluded_pairs = 0u64;
    let mut pairs = 0u64;
    let mut touched = 0usize;
    let mut outputmeta = vec![];
    let mut output: [Vec<f32>; 4] = std::array::from_fn(|_| vec![]);
    for w in selected {
        let t0 = w["utc_start_s"].as_f64().unwrap();
        let selected: Vec<_> = pulses
            .iter()
            .filter(|p| p.t >= t0 && p.t < t0 + 60.)
            .collect();
        let x: Vec<_> = selected.iter().flat_map(|p| p.z.iter().copied()).collect();
        let mut sc = screen(&x, selected.len(), 1324, &settings);
        for (j, p) in selected.iter().enumerate() {
            if p.saturated {
                sc.mask[j * 1324..(j + 1) * 1324].fill(8);
            }
            pulse_times.push(p.t);
        }
        touched += sc
            .mask
            .chunks(1324)
            .filter(|m| m.iter().any(|&x| x != 0))
            .count();
        let masked_samples = sc.mask.iter().filter(|&&m| m != 0).count();
        let sum = selected
            .par_iter()
            .enumerate()
            .fold(Sum::new, |sum, (j, p)| {
                sum.add(p, &sc.mask[j * 1324..(j + 1) * 1324])
            })
            .reduce(Sum::new, |a, b| a.merge(b));
        masks.extend_from_slice(&sc.mask);
        ensure!(sum.n > 1, "too few pulses");
        for j in 0..41 * 1324 {
            let n = sum.counts[j] as f64;
            for q in 0..2 {
                let mean = if n > 0. { sum.x[q][j] / n } else { 0. };
                let sem = if n > 1. {
                    ((sum.x[q + 2][j] - n * mean * mean).max(0.) / (n * (n - 1.))).sqrt()
                } else {
                    0.
                };
                output[q].push(mean as f32);
                output[q + 2].push(sem as f32);
            }
            if j % 1324 + j / 1324 < 1324 {
                pairs += sum.n as u64;
                excluded_pairs += sum.n as u64 - sum.counts[j] as u64;
            }
        }
        all_counts.extend_from_slice(&sum.counts);
        let mut wm = w.clone();
        wm["original_pulse_count"] = wm["pulse_count"].clone();
        wm["pulse_count"] = json!(sum.n);
        wm["masked_voltage_samples"] = json!(masked_samples);
        outputmeta.push(wm);
    }
    let h = hdf5::File::create(&a[2])?;
    let names = [
        "acf_real_adu2",
        "acf_imag_adu2",
        "standard_error_real_adu2",
        "standard_error_imag_adu2",
    ];
    for (i, n) in names.iter().enumerate() {
        h.new_dataset::<f32>()
            .shape([outputmeta.len(), 41, 1324])
            .chunk([1, 41, 1324])
            .shuffle()
            .deflate(4)
            .create(*n)?
            .write_raw(&output[i])?;
    }
    h.new_dataset::<u32>()
        .shape([outputmeta.len(), 41, 1324])
        .shuffle()
        .deflate(4)
        .create("pair_count")?
        .write_raw(&all_counts)?;
    h.new_dataset::<u8>()
        .shape([pulse_times.len(), 1324])
        .deflate(4)
        .create("voltage_mask")?
        .write_raw(&masks)?;
    h.new_dataset_builder()
        .with_data(&pulse_times)
        .create("pulse_utc_s")?;
    let meta = json!({"schema":"syisr-hplus-pair-masked-v2","windows":outputmeta,"selection":format!("zenith hour {hour} UTC"),"frames_checked":total,"raw_selected_pulses":pulses.len(),"pulses_with_any_mask":touched,"masked_voltage_samples":masks.iter().filter(|&&v|v!=0).count(),"excluded_pairs":excluded_pairs,"total_pairs":pairs,"detector":settings,"classification":"satellite/RFI candidates, not object identification; sample pairs excluded if either voltage sample masked","source_files":fileinfo,"clock":"UTC=header-8h","lags":"all measured 0..400 us, no spline or omitted lags"});
    h.new_dataset_builder()
        .with_data(&serde_json::to_vec(&meta)?)
        .create("metadata_json_utf8")?;
    println!(
        "raw={} touched={} masked_pairs={}/{}",
        pulses.len(),
        touched,
        excluded_pairs,
        pairs
    );
    Ok(())
}
