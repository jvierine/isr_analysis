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
    power: Vec<f64>,
    saturated: bool,
}
struct Sum {
    n: usize,
    x: [Vec<f64>; 4],
}
impl Sum {
    fn new() -> Self {
        Self {
            n: 0,
            x: std::array::from_fn(|_| vec![0.; 41 * 1324]),
        }
    }
    fn add(mut self, p: &Pulse) -> Self {
        self.n += 1;
        for l in 0..41 {
            for k in 0..1324 - l {
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
    let gates: Vec<usize> = (400..=1000)
        .step_by(50)
        .map(|h| ((2. * (h as f64 - 0.05) / 299792.458 + 240e-6 - 767e-6) / 1e-5).round() as usize)
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
            let power = gates
                .iter()
                .map(|&k| (k - 8..=k + 8).map(|i| z[i].norm_sqr() as f64).sum::<f64>() / 17.)
                .collect();
            pulses.push(Pulse {
                t,
                z,
                power,
                saturated,
            });
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
    let medians: Vec<_> = (0..gates.len())
        .map(|k| {
            let mut v: Vec<_> = pulses.iter().map(|p| p.power[k]).collect();
            v.sort_by(f64::total_cmp);
            v[v.len() / 2]
        })
        .collect();
    let multiplier = (gates.len() as f64 / 1e-6).ln() / 2_f64.ln();
    let mask: Vec<bool> = pulses
        .iter()
        .map(|p| {
            p.saturated
                || p.power
                    .iter()
                    .zip(&medians)
                    .any(|(&x, &m)| x > multiplier * m)
        })
        .collect();
    let rejected = mask.iter().filter(|&&v| v).count();
    let mut outputmeta = vec![];
    let mut output: [Vec<f32>; 4] = std::array::from_fn(|_| vec![]);
    for w in selected {
        let t0 = w["utc_start_s"].as_f64().unwrap();
        let selected: Vec<_> = pulses
            .iter()
            .zip(&mask)
            .filter(|(p, m)| !**m && p.t >= t0 && p.t < t0 + 60.)
            .map(|(p, _)| p)
            .collect();
        let sum = selected
            .par_iter()
            .fold(Sum::new, |sum, p| sum.add(p))
            .reduce(Sum::new, |a, b| a.merge(b));
        ensure!(sum.n > 1, "window entirely rejected");
        for j in 0..41 * 1324 {
            for q in 0..2 {
                let mean = sum.x[q][j] / sum.n as f64;
                let sem = ((sum.x[q + 2][j] - sum.n as f64 * mean * mean).max(0.)
                    / (sum.n as f64 * (sum.n - 1) as f64))
                    .sqrt();
                output[q].push(mean as f32);
                output[q + 2].push(sem as f32);
            }
        }
        let mut wm = w.clone();
        wm["original_pulse_count"] = wm["pulse_count"].clone();
        wm["pulse_count"] = json!(sum.n);
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
    let times: Vec<f64> = pulses.iter().map(|p| p.t).collect();
    h.new_dataset_builder()
        .with_data(&times)
        .create("pulse_utc_s")?;
    h.new_dataset_builder()
        .with_data(&mask.iter().map(|m| u8::from(*m)).collect::<Vec<_>>())
        .create("excluded_pulse")?;
    h.new_dataset_builder().with_data(&serde_json::to_vec(&json!({"schema":"syisr-hplus-night-clean-v1","windows":outputmeta,"selection":format!("SY zenith 2023-11-15 {hour}:00--{}:00 UTC after conservative pulse-power CFAR guard",hour+1),"frames_checked":total,"raw_selected_pulses":pulses.len(),"excluded_pulses":rejected,"screen":{"method":"whole-IPP exclusion if a 17-bin mean power at ANY guard altitude exceeds a conservative median-based threshold, or int32 near-saturation","guard_altitudes_km":(400..=1000).step_by(50).collect::<Vec<_>>(),"training":"same-hour per-range median across zenith pulses, independent of ion-composition fits","threshold_over_median":multiplier,"nominal_per_pulse_pfa":1e-6,"pfa_limitation":"nominal exponential fully-coherent Gaussian-power approximation with union-bound guard count; not a calibrated satellite classifier or range-Doppler CFAR detection","lag_policy":"all measured lags 0--400 us retained; excluded pulses removed before all range/lag means and SEMs"},"source_files":fileinfo,"clock":"UTC=header-8h; previously validated clock convention","raw_untouched":true}))?).create("metadata_json_utf8")?;
    println!(
        "raw={} rejected={} retained={}",
        pulses.len(),
        rejected,
        pulses.len() - rejected
    );
    Ok(())
}
