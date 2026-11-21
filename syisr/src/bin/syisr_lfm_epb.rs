//! Zenith LFM RTI from SYISR L1 complex voltages. Chirp phase follows
//! lfm_meteor/matched_filter.py; use explicit linear correlation and leading-edge
//! delays, not its legacy mode=same gate convention or 2024 satellite correction.
use anyhow::{ensure, Result};
use chrono::NaiveDate;
use hdf5::types::VarLenUnicode;
use num_complex::Complex32;
use rustfft::{num_traits::Zero, FftPlanner};
use std::{path::Path, str::FromStr};

#[derive(Clone, Copy, hdf5::H5Type)]
#[repr(C)]
struct IQ {
    real: f64,
    imag: f64,
}
const C: f64 = 299792458.;
const VERSION: &str = "lfm-epb-multibeam-v1";
fn chirp(fs: f64, bw: f64, duration: f64, sign: f64) -> Vec<Complex32> {
    (0..(fs * duration).round() as usize)
        .map(|i| {
            let t = i as f64 / fs;
            Complex32::from_polar(
                1.,
                (sign * 2. * std::f64::consts::PI * (bw * t / 2. - bw * t * t / (2. * duration)))
                    as f32,
            )
        })
        .collect()
}
fn text(h: &hdf5::File, name: &str, value: &str) -> Result<()> {
    h.new_attr::<VarLenUnicode>()
        .shape(())
        .create(name)?
        .write_scalar(&VarLenUnicode::from_str(value)?)?;
    Ok(())
}
fn vector(h: &hdf5::File, name: &str, v: &[f64]) -> Result<()> {
    h.new_dataset_builder().with_data(v).create(name)?;
    Ok(())
}
fn matrix(h: &hdf5::File, name: &str, v: &[f64], nr: usize, nc: usize) -> Result<()> {
    h.new_dataset::<f64>()
        .shape((nr, nc))
        .create(name)?
        .write_raw(v)?;
    Ok(())
}
fn median(v: &mut [f64]) -> f64 {
    v.sort_by(f64::total_cmp);
    let n = v.len();
    (v[(n - 1) / 2] + v[n / 2]) / 2.
}
fn utc(tm: &[f64], np: usize, i: usize) -> Result<f64> {
    let a = |j| tm[j * np + i];
    let t = NaiveDate::from_ymd_opt(2000 + a(0) as i32, a(1) as u32, a(2) as u32)
        .and_then(|d| d.and_hms_opt(a(3) as u32, a(4) as u32, 0))
        .ok_or_else(|| anyhow::anyhow!("invalid timestamp"))?;
    ensure!(a(5) >= 0. && a(5) < 60., "invalid seconds");
    Ok(t.and_utc().timestamp() as f64 + a(5) - 28800.)
}
fn run(input: &Path, output: &Path, start: f64, stop: f64) -> Result<()> {
    let h = hdf5::File::open(input)?;
    let p = h.dataset("para")?.read_raw::<f64>()?;
    ensure!(p.len() == 23, "unsupported para length");
    ensure!(
        (p[7] - p[19]).abs() < 0.02
            && (p[6] - p[18]).abs() < 0.02
            && (45. ..=90.01).contains(&p[7]),
        "unsupported non-monostatic pointing"
    );
    ensure!(
        (p[14] - 4.).abs() < 1e-6
            && (p[15] - 4.).abs() < 1e-6
            && p[9] == 480.
            && p[10] == 479.
            && p[11] == 12000.,
        "unsupported waveform layout"
    );
    let ds = h.dataset("data_raw")?;
    let shape = ds.shape();
    ensure!(
        shape.len() == 2 && shape[0] == 44564,
        "unsupported voltage shape"
    );
    let (ns, np) = (shape[0], shape[1]);
    let time_ds = h.dataset("syspara")?;
    ensure!(time_ds.shape() == [7, np], "unsupported timestamp layout");
    let tm = time_ds.read_raw::<f64>()?;
    let times: Vec<f64> = (0..np).map(|i| utc(&tm, np, i)).collect::<Result<_>>()?;
    ensure!(
        times.windows(2).all(|w| w[1] > w[0]),
        "non-increasing pulse times"
    );
    ensure!(
        (0..np).all(|i| tm[6 * np + i] == 0.),
        "unsupported LFM code"
    );
    let selected: Vec<usize> = (0..np)
        .filter(|&i| times[i] >= start && times[i] < stop)
        .collect();
    if selected.is_empty() {
        println!("no overlapping pulses");
        return Ok(());
    }
    let raw = ds.read_raw::<IQ>()?;
    ensure!(
        raw.iter().all(|v| v.real.is_finite() && v.imag.is_finite()),
        "nonfinite voltages"
    );
    let fs = p[14] * 1e6;
    let bw = p[15] * 1e6;
    let duration = p[10] * 1e-6;
    let code = chirp(fs, bw, duration, 1.);
    let nc = code.len();
    let nfft = (ns + nc - 1).next_power_of_two();
    let mut planner = FftPlanner::<f32>::new();
    let fft = planner.plan_fft_forward(nfft);
    let ifft = planner.plan_fft_inverse(nfft);
    let mut filters = Vec::new();
    for sign in [1., -1.] {
        let c = chirp(fs, bw, duration, sign);
        let mut f = vec![Complex32::zero(); nfft];
        for (o, x) in f.iter_mut().zip(c.iter().rev()) {
            *o = x.conj() / (nc as f32).sqrt();
        }
        fft.process(&mut f);
        filters.push(f);
    }
    // para gate coordinates use c=300,000 km/s; retain nominal hardware delay.
    let first_delay = p[12] / 0.15 * 1e-6;
    let dr = C / (2. * fs) / 1000.;
    let r0 = C * first_delay / 2. / 1000.;
    let valid = ns - nc + 1;
    let group = (0.375 / dr).round() as usize;
    let ng = valid / group;
    let n = selected.len();
    let edges: Vec<f64> = (0..=ng)
        .map(|g| r0 + (g * group) as f64 * dr - dr / 2.)
        .collect();
    let ranges: Vec<f64> = edges.windows(2).map(|w| (w[0] + w[1]) / 2.).collect();
    let heights: Vec<f64> = ranges
        .iter()
        .map(|r| {
            syisr_ac::lfm_geometry::geodetic(syisr_ac::lfm_geometry::position(*r, p[6], p[7]))[2]
        })
        .collect();
    let hedge: Vec<f64> = edges
        .iter()
        .map(|r| {
            syisr_ac::lfm_geometry::geodetic(syisr_ac::lfm_geometry::position(*r, p[6], p[7]))[2]
        })
        .collect();
    let mut powers = vec![vec![0.; n * ng]; 2];
    let mut noises = Vec::new();
    let mut raw_top = Vec::new();
    for (j, &i) in selected.iter().enumerate() {
        let mut v = vec![Complex32::zero(); nfft];
        for k in 0..ns {
            let q = raw[k * np + i];
            v[k] = Complex32::new(q.real as f32, q.imag as f32);
        }
        let mut top: Vec<f64> = v[..ns]
            .iter()
            .enumerate()
            .filter(|(k, _)| r0 + *k as f64 * dr >= 1500.)
            .map(|(_, z)| z.norm_sqr() as f64)
            .collect();
        raw_top.push(top.iter().sum::<f64>() / top.len() as f64);
        noises.push(median(&mut top) / std::f64::consts::LN_2);
        fft.process(&mut v);
        for (s, f) in filters.iter().enumerate() {
            let mut c: Vec<Complex32> = v.iter().zip(f).map(|(a, b)| a * b).collect();
            ifft.process(&mut c);
            for g in 0..ng {
                powers[s][j * ng + g] = c[nc - 1 + g * group..nc - 1 + (g + 1) * group]
                    .iter()
                    .map(|z| z.norm_sqr() as f64 / (nfft as f64).powi(2))
                    .sum::<f64>()
                    / group as f64;
            }
        }
    }
    let partial = output.with_extension("h5.partial");
    let out = hdf5::File::create(&partial)?;
    for (key,val) in [("version",VERSION),("source_file",input.to_str().unwrap()),("time_scale","UTC"),
        ("voltage_convention","data_raw.real + i data_raw.imag"),
        ("range_convention","leading edge linear correlation delay; nominal header gate-front; no 2024 meteor offset"),
        ("altitude_convention","WGS84 ray at recorded azimuth/elevation; site 18.3492 N,109.6222 E, assumed 0.05 km ellipsoid height"),
        ("noise_method","median raw |voltage|^2 / ln(2) at nominal range >=1500 km; may contain topside scatter or interference"),
        ("power_units","input voltage units squared; filter has unit energy; range-cell incoherent average"),
        ("limitations","uncalibrated absolute density and instrumental delay; no Doppler correction; 0.375 km cells; retained negative signal; pulse sampling is sparse"),
        ("frequency_convention","DFT exp(-i2pi f t); no velocity inferred; downchirp has decreasing instantaneous frequency"),
        ("script_provenance","syisr/src/bin/syisr_lfm_epb.rs; chirp equation adapted from lfm_meteor/matched_filter.py")] {text(&out,key,val)?;}
    vector(
        &out,
        "times_unix_s",
        &selected.iter().map(|&i| times[i]).collect::<Vec<_>>(),
    )?;
    vector(&out, "range_km", &ranges)?;
    vector(&out, "range_edges_km", &edges)?;
    vector(&out, "altitude_km", &heights)?;
    vector(&out, "altitude_edges_km", &hedge)?;
    vector(&out, "noise_power", &noises)?;
    vector(&out, "raw_topside_mean_power", &raw_top)?;
    vector(&out, "para", &p)?;
    let geo: Vec<f64> = ranges
        .iter()
        .flat_map(|r| {
            syisr_ac::lfm_geometry::geodetic(syisr_ac::lfm_geometry::position(*r, p[6], p[7]))
        })
        .collect();
    matrix(&out, "geodetic_lat_lon_alt", &geo, ng, 3)?;
    matrix(&out, "downchirp_power", &powers[0], n, ng)?;
    matrix(&out, "upchirp_power", &powers[1], n, ng)?;
    let signal: Vec<f64> = powers[0]
        .iter()
        .enumerate()
        .map(|(i, v)| v - noises[i / ng])
        .collect();
    matrix(&out, "downchirp_signal", &signal, n, ng)?;
    out.flush()?;
    drop(out);
    std::fs::rename(partial, output)?;
    println!(
        "pulses={n} range={:.1}..{:.1} km cells={ng}",
        edges[0], edges[ng]
    );
    Ok(())
}
fn main() -> Result<()> {
    let a: Vec<_> = std::env::args().collect();
    ensure!(
        a.len() == 5,
        "INPUT_MAT OUTPUT_H5 START_UTC_UNIX STOP_UTC_UNIX"
    );
    run(
        Path::new(&a[1]),
        Path::new(&a[2]),
        a[3].parse()?,
        a[4].parse()?,
    )
}
#[cfg(test)]
mod tests {
    use super::*;
    #[test]
    fn timestamp_beijing_to_utc() {
        let t = [26., 3., 1., 20., 0., 0.125, 0.];
        assert_eq!(utc(&t, 1, 0).unwrap(), 1772366400.125);
    }
    #[test]
    fn linear_correlation_delay_and_unit_energy() {
        let c = chirp(4e6, 4e6, 479e-6, 1.);
        let n = (5000 + c.len() - 1).next_power_of_two();
        let mut x = vec![Complex32::zero(); n];
        let delay = 817;
        x[delay..delay + c.len()].copy_from_slice(&c);
        let mut f = vec![Complex32::zero(); n];
        for (i, v) in c.iter().rev().enumerate() {
            f[i] = v.conj() / (c.len() as f32).sqrt();
        }
        let mut p = FftPlanner::new();
        let fft = p.plan_fft_forward(n);
        let ifft = p.plan_fft_inverse(n);
        fft.process(&mut x);
        fft.process(&mut f);
        for (i, v) in x.iter_mut().enumerate() {
            *v *= f[i];
        }
        ifft.process(&mut x);
        let peak = (0..n)
            .max_by(|&a, &b| x[a].norm_sqr().total_cmp(&x[b].norm_sqr()))
            .unwrap();
        assert_eq!(peak - c.len() + 1, delay);
        assert!((x[peak].norm_sqr() as f64 / (n * n) as f64 / c.len() as f64 - 1.).abs() < 2e-5);
        // A unit impulse has unit total compressed energy (white-noise gain = 1).
        assert!(
            (c.iter()
                .map(|z| z.norm_sqr() as f64 / c.len() as f64)
                .sum::<f64>()
                - 1.)
                .abs()
                < 1e-6
        );
    }
}
