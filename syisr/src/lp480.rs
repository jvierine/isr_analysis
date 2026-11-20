//! 480-us monostatic uncoded long-pulse products with explicit finite-pulse gating.
//! Explicit recorded-clock offset is applied before UTC binning; filenames are not clocks.
use crate::{campaign, reader::Header};
use anyhow::{ensure, Result};
use num_complex::Complex64 as C;
use serde::{Deserialize, Serialize};
use std::{
    collections::BTreeMap,
    fs,
    io::{BufReader, Read, Seek, SeekFrom},
    path::Path,
};
pub const NS: usize = 1324;
pub const FRAME: usize = 1024 + 8 * NS;
pub const NL: usize = 47;
pub const DT: f64 = 1e-5;
pub const PW: f64 = 480e-6;
pub const C_KM: f64 = 299792.458;
pub fn configured_top_km() -> usize {
    let top = std::env::var("SYISR_LP480_TOP_KM")
        .unwrap_or("1400".into())
        .parse::<usize>()
        .expect("LP480 top must be an integer km");
    assert!(
        (1000..=2000).contains(&top) && top % 5 == 0,
        "LP480 top must be1000..2000km on5km grid"
    );
    top
}
fn gate_nu(z: f64) -> usize {
    if z <= 400. {
        3
    } else {
        (3. + (z - 400.) * 7. / 1000.).round() as usize
    }
}
pub fn range_support(job: &Job) -> serde_json::Value {
    let first = beam_top_for(job, 2200);
    let mut all = first;
    loop {
        let nu = gate_nu(all as f64);
        let i = ((2. * range(all as f64, job.el) / C_KM + PW / 2.
            - (nu as f64 - 1.) * DT / 2.
            - job.front_us * 1e-6)
            / DT)
            .round();
        if i + nu as f64 + NL as f64 <= NS as f64 {
            break;
        }
        all -= 5;
    }
    let raw_outer_range = (job.front_us * 1e-6 + (NS - 1) as f64 * DT) * C_KM / 2.;
    let noise_min = altitude(
        (job.front_us * 1e-6 + (NS - 160) as f64 * DT - PW / 2.) * C_KM / 2.,
        job.el,
    );
    let noise_max = altitude(
        (job.front_us * 1e-6 + (NS - NL - 1) as f64 * DT - PW / 2.) * C_KM / 2.,
        job.el,
    );
    serde_json::json!({"azimuth_deg":job.az,"elevation_deg":job.el,"fit_top_km":beam_top_km(job),"max_10us_lag_gate_centre_altitude_km":first,"max_full_470us_lag_gate_centre_altitude_km":all,"outer_raw_sample_slant_range_km":raw_outer_range,"noise_estimator_gate_centre_altitude_km":[noise_min,noise_max],"qualification":"Geometric/sample support only, not a detection or parameter uncertainty. Pulse-centre convention; complete finite pulse response is fitted. Noise reference may contain upper-range plasma; background sensitivity remains to validate."})
}
/// Conservative domain endpoint with at least the10us lag inside the recording.
pub fn beam_top_km(job: &Job) -> usize {
    beam_top_for(job, configured_top_km())
}
fn beam_top_for(job: &Job, requested: usize) -> usize {
    let mut top = requested;
    loop {
        let nu = gate_nu(top as f64);
        let i = ((2. * range(top as f64, job.el) / C_KM + PW / 2.
            - (nu as f64 - 1.) * DT / 2.
            - job.front_us * 1e-6)
            / DT)
            .round();
        if i >= 1. && i + nu as f64 + 1. <= NS as f64 {
            return top;
        }
        top -= 5;
        assert!(top >= 1000, "insufficient recorded range for profile");
    }
}
/// Magnetic field epoch follows the observation, including leap years.
pub fn decimal_year(utc_s: i64) -> Result<f64> {
    use chrono::{Datelike, TimeZone, Utc};
    let t = Utc
        .timestamp_opt(utc_s, 0)
        .single()
        .ok_or_else(|| anyhow::anyhow!("invalid UTC epoch"))?;
    let start = Utc
        .with_ymd_and_hms(t.year(), 1, 1, 0, 0, 0)
        .unwrap()
        .timestamp();
    let end = Utc
        .with_ymd_and_hms(t.year() + 1, 1, 1, 0, 0, 0)
        .unwrap()
        .timestamp();
    Ok(t.year() as f64 + (utc_s - start) as f64 / (end - start) as f64)
}
/// File-parallel workers use MPI on the original host or native Slurm tasks.
/// A batch-script parent has no SLURM_PROCID and must remain a single worker.
pub fn worker() -> Result<(usize, usize)> {
    let pair = if let Ok(rank) = std::env::var("OMPI_COMM_WORLD_RANK") {
        (rank, std::env::var("OMPI_COMM_WORLD_SIZE")?)
    } else if let Ok(rank) = std::env::var("SLURM_PROCID") {
        (rank, std::env::var("SLURM_NTASKS")?)
    } else {
        return Ok((0, 1));
    };
    let (rank, size): (usize, usize) = (pair.0.parse()?, pair.1.parse()?);
    ensure!(
        size > 0 && rank < size,
        "invalid distributed worker coordinates"
    );
    Ok((rank, size))
}
#[derive(Clone, Serialize, Deserialize, PartialEq)]
pub struct Part {
    pub file: String,
    pub offset: u64,
    pub pulses: usize,
    pub block: i64,
}
#[derive(Clone, Serialize, Deserialize, PartialEq)]
pub struct Job {
    pub id: String,
    pub start: i64,
    pub az: f64,
    pub el: f64,
    pub front_us: f64,
    pub parts: Vec<Part>,
}
#[derive(Serialize, Deserialize)]
pub struct Manifest {
    #[serde(default)]
    pub quarantined_segments: Vec<serde_json::Value>,
    #[serde(default)]
    pub header_utc_offset_s: i64,
    #[serde(default)]
    pub clock_repairs: Vec<serde_json::Value>,
    pub schema: String,
    pub jobs: Vec<Job>,
    pub pulse_us: f64,
    pub ipp_s: f64,
    pub raw_root: String,
}
#[derive(Clone, Serialize, Deserialize)]
pub struct Gate {
    pub altitude_km: f64,
    pub range_km: f64,
    pub index: usize,
    pub nu: usize,
}
#[derive(Serialize, Deserialize)]
pub struct Block {
    #[serde(default)]
    pub range_doppler_raw: Vec<f64>,
    #[serde(default)]
    pub range_doppler_pulses: usize,
    pub start: i64,
    pub pulses: usize,
    pub kept: usize,
    pub sums: Vec<[f64; 2]>,
    pub counts: Vec<usize>,
    pub noise: Vec<[f64; 2]>,
    pub noise_pairs: Vec<usize>,
}
#[derive(Serialize, Deserialize)]
pub struct Decoded {
    #[serde(default)]
    pub averaged: Option<crate::lp480_fit::Averaged>,
    pub schema: String,
    pub job: Job,
    pub gates: Vec<Gate>,
    pub blocks: Vec<Block>,
    pub lags_us: Vec<f64>,
    pub rejection: String,
}
pub fn altitude(r: f64, el: f64) -> f64 {
    let re = 6371.05;
    (re * re + r * r + 2. * re * r * el.to_radians().sin()).sqrt() - re
}
pub fn range(z: f64, el: f64) -> f64 {
    let re = 6371.05;
    let s = el.to_radians().sin();
    -re * s + ((re * s).powi(2) + 2. * re * z + z * z).sqrt()
}
pub fn index(raw: &Path, out: &Path) -> Result<Manifest> {
    let header_utc_offset_s: i64 = std::env::var("SYISR_HEADER_UTC_OFFSET_S")
        .unwrap_or("0".into())
        .parse()?;
    index_with_offset(raw, out, header_utc_offset_s)
}
pub(crate) fn index_with_offset(
    raw: &Path,
    out: &Path,
    header_utc_offset_s: i64,
) -> Result<Manifest> {
    index_with_recovery(
        raw,
        out,
        header_utc_offset_s,
        std::env::var_os("SYISR_ALLOW_PARTIAL_FINAL_FRAME").is_some(),
        std::env::var_os("SYISR_QUARANTINE_CORRUPT_FRAMES").is_some(),
    )
}
fn index_with_options(
    raw: &Path,
    out: &Path,
    header_utc_offset_s: i64,
    allow_partial: bool,
) -> Result<Manifest> {
    index_with_recovery(raw, out, header_utc_offset_s, allow_partial, false)
}
fn resync_verified(f: &mut BufReader<fs::File>, from: u64, bytes: u64) -> Result<Option<u64>> {
    let mut search = from;
    while let Some(candidate) = campaign::resync(f, search, bytes)? {
        f.seek(SeekFrom::Start(candidate))?;
        let mut h = [0u8; 1024];
        f.read_exact(&mut h)?;
        let first = Header::parse(&h)?;
        if first.samples != NS
            || first.waveform != 4
            || first.pulse_width_us != 480
            || first.ipp_us != 16000
        {
            return Ok(Some(candidate)); // Let the normal layout guard reject a real mode change.
        }
        if candidate + FRAME as u64 + 1024 <= bytes {
            f.seek(SeekFrom::Start(candidate + FRAME as u64))?;
            f.read_exact(&mut h)?;
            if let Ok(second) = Header::parse(&h) {
                if (second.header_clock_unix_s - first.header_clock_unix_s - 0.016).abs() < 0.0001 {
                    return Ok(Some(candidate));
                }
            }
        }
        search = candidate + 1;
    }
    Ok(None)
}
fn index_with_recovery(
    raw: &Path,
    out: &Path,
    header_utc_offset_s: i64,
    allow_partial: bool,
    allow_corrupt: bool,
) -> Result<Manifest> {
    ensure!(
        header_utc_offset_s.abs() <= 14 * 3600,
        "invalid recorded clock offset"
    );
    let mut files = campaign::files(raw, ".dat")?;
    files.sort();
    let mut jobs = BTreeMap::<(i64, u16, u16), Job>::new();
    let mut total = 0;
    let mut quarantined_segments = vec![];
    for p in files {
        let mut f = BufReader::new(fs::File::open(&p)?);
        let bytes = fs::metadata(&p)?.len();
        let complete_bytes = bytes / (FRAME as u64) * (FRAME as u64);
        ensure!(
            bytes % FRAME as u64 == 0 || allow_partial,
            "truncated or unsupported frame {}",
            p.display()
        );
        ensure!(complete_bytes > 0, "no complete frame in {}", p.display());
        let mut offset = 0;
        let mut last_good: Option<(u64, (i64, u16, u16))> = None;
        while offset < bytes {
            if bytes - offset < 1024 {
                ensure!(allow_partial, "partial final header in {}", p.display());
                quarantined_segments.push(serde_json::json!({"file":p,"offset":offset,"bytes":bytes-offset,"reason":"Incomplete final header at EOF; raw unchanged"}));
                break;
            }
            let mut h = [0u8; 1024];
            f.read_exact(&mut h)?;
            let framed =
                h[..4] == 0xaaaa5555u32.to_le_bytes() && h[1020..] == 0xaa5555aau32.to_le_bytes();
            if !framed && allow_corrupt {
                // A byte loss can corrupt the preceding pulse's payload while
                // leaving its own header valid. Conservatively roll it back.
                let begin = if let Some((previous, key)) = last_good.take() {
                    let j = jobs.get_mut(&key).unwrap();
                    let part = j.parts.last_mut().unwrap();
                    part.pulses -= 1;
                    if part.pulses == 0 {
                        j.parts.pop();
                    }
                    if j.parts.is_empty() {
                        jobs.remove(&key);
                    }
                    total -= 1;
                    previous
                } else {
                    offset
                };
                let search = if begin < offset {
                    begin + 1024
                } else {
                    offset + 1
                };
                let next = resync_verified(&mut f, search, bytes)?;
                let end = next.unwrap_or(bytes);
                quarantined_segments.push(serde_json::json!({"file":p,"offset":begin,"bytes":end-begin,"reason":"Corrupt framing region, including one preceding pulse conservatively excluded; resume requires two valid headers one complete frame and16ms apart; raw unchanged"}));
                if let Some(next) = next {
                    offset = next;
                    f.seek(SeekFrom::Start(offset))?;
                    continue;
                } else {
                    break;
                }
            }
            let h = Header::parse(&h)?;
            ensure!(
                h.samples == NS
                    && h.waveform == 4
                    && h.pulse_width_us == 480
                    && h.ipp_us == 16000
                    && h.sample_rate_hz == 1e5
                    && (h.radar_frequency_hz - 430.2e6).abs() < 1.,
                "unsupported 480-us layout at {}:{offset}",
                p.display()
            );
            if offset + FRAME as u64 > bytes {
                ensure!(
                    allow_partial,
                    "incomplete final IQ frame in {}",
                    p.display()
                );
                quarantined_segments.push(serde_json::json!({"file":p,"offset":offset,"bytes":bytes-offset,"reason":"Incomplete final IQ frame at EOF explicitly excluded; every complete preceding frame validated; raw unchanged"}));
                break;
            }
            let time = h.header_clock_unix_s - header_utc_offset_s as f64;
            let start = (time as i64).div_euclid(1800) * 1800;
            let block = (time as i64).div_euclid(10) * 10;
            last_good = Some((offset, (start, h.azimuth_raw, h.elevation_raw)));
            let j = jobs
                .entry((start, h.azimuth_raw, h.elevation_raw))
                .or_insert_with(|| Job {
                    id: format!("{start}-{:05}-{:05}", h.azimuth_raw, h.elevation_raw),
                    start,
                    az: h.azimuth_deg(),
                    el: h.elevation_deg(),
                    front_us: h.gate_front_us as f64,
                    parts: vec![],
                });
            ensure!(
                (j.front_us - h.gate_front_us as f64).abs() < 1e-8,
                "range timing changed"
            );
            let file = p.display().to_string();
            if let Some(last) = j.parts.last_mut().filter(|q| {
                q.file == file
                    && q.offset + q.pulses as u64 * FRAME as u64 == offset
                    && q.block == block
            }) {
                last.pulses += 1;
            } else {
                j.parts.push(Part {
                    file,
                    offset,
                    pulses: 1,
                    block,
                });
            }
            offset += FRAME as u64;
            f.seek(SeekFrom::Start(offset))?;
            total += 1;
        }
        eprintln!("INDEX {} pulses={total}", p.display());
    }
    let m = Manifest {
        quarantined_segments,
        header_utc_offset_s,
        clock_repairs: vec![],
        schema: "syisr-lp480-index-v1".into(),
        jobs: jobs.into_values().collect(),
        pulse_us: 480.,
        ipp_s: 0.016,
        raw_root: raw.display().to_string(),
    };
    campaign::save(out, &m)?;
    Ok(m)
}
pub fn gates(job: &Job) -> Vec<Gate> {
    gates_to(job, configured_top_km())
}
fn gates_to(job: &Job, requested: usize) -> Vec<Gate> {
    let top = beam_top_for(job, requested) as f64;
    let mut z = 100f64;
    let mut out = vec![];
    while z <= top + 0.0001 {
        let r = range(z, job.el);
        let nu = gate_nu(z);
        let i = ((2. * r / C_KM + PW / 2. - (nu as f64 - 1.) * DT / 2. - job.front_us * 1e-6) / DT)
            .round();
        if i >= 1.
            && i + nu as f64 + 1. <= NS as f64
            && out
                .last()
                .is_none_or(|g: &Gate| g.index != i as usize || g.nu != nu)
        {
            let t = job.front_us * 1e-6 + i * DT + (nu as f64 - 1.) * DT / 2. - PW / 2.;
            let rr = t * C_KM / 2.;
            out.push(Gate {
                altitude_km: altitude(rr, job.el),
                range_km: rr,
                index: i as usize,
                nu,
            });
        }
        if z == top {
            break;
        }
        let step = if z <= 400. {
            15.
        } else {
            15. + (z - 400.) * 0.085
        };
        z = if z < 1400. && z + step > 1400. {
            1400f64.min(top)
        } else {
            (z + step).min(top)
        };
    }
    out
}
/// Source-delay integral of the exact rectangular pulse for a lag-specific sum.
/// k=i-l..i+nu-1 keeps the union of scatter support independent of lag.
pub fn weight(g: &Gate, l: usize, front: f64, edge: [f64; 2]) -> f64 {
    if g.index < l {
        return 0.;
    }
    let mut w = 0.;
    for k in g.index - l..g.index + g.nu {
        let t = front * 1e-6 + k as f64 * DT;
        w += (t.min(edge[1]) - (t + l as f64 * DT - PW).max(edge[0])).max(0.);
    }
    w / (PW * (l + g.nu) as f64)
}
pub fn decode(job: &Job, out: &Path) -> Result<()> {
    let gates = gates(job);
    if out.exists() {
        let old: Decoded = campaign::load(out)?;
        if old.schema == "syisr-lp480-pooled-screen-v2"
            && serde_json::to_value(&old.job)? == serde_json::to_value(job)?
            && serde_json::to_value(&old.gates)? == serde_json::to_value(&gates)?
            && old
                .blocks
                .first()
                .is_some_and(|b| !b.range_doppler_raw.is_empty())
        {
            return Ok(());
        }
    }

    let m = gates.len() * NL;
    let mut groups = BTreeMap::<i64, Vec<&Part>>::new();
    for p in &job.parts {
        groups.entry(p.block).or_default().push(p);
    }
    let mut frames = vec![];
    let mut spans = vec![];
    for (start, parts) in groups {
        let begin = frames.len();
        for p in parts {
            let mut f = BufReader::new(fs::File::open(&p.file)?);
            f.seek(SeekFrom::Start(p.offset))?;
            for _ in 0..p.pulses {
                let fr =
                    crate::reader::read_frame(&mut f)?.ok_or_else(|| anyhow::anyhow!("EOF"))?;
                frames.push(fr.iq);
            }
        }
        spans.push((start, begin, frames.len()));
    }
    ensure!(frames.len() >= 64, "insufficient pulse support");
    // Pool the complete same-beam integration for robust baselines: a multi-second
    // echo must not contaminate the upper quartile of its own 10-second block.
    let audit = crate::lp480_screen::screen(&frames, NS);
    let mut blocks = vec![];
    for (start, begin, end_block) in spans {
        if end_block - begin < 64 {
            continue;
        }
        let kept = audit.keep[begin..end_block].iter().filter(|x| **x).count();
        let mut sum = vec![C::default(); m];
        let mut count = vec![0usize; m];
        let mut noise = vec![C::default(); NL + 1];
        let mut pairs = vec![0usize; NL + 1];
        for (p, x) in frames[begin..end_block].iter().enumerate() {
            if !audit.keep[begin + p] {
                continue;
            }
            for l in 0..=NL {
                for k in NS - 160..NS - NL {
                    let a = x[k] * x[k + l].conj();
                    noise[l] += C::new(a.re as f64, a.im as f64);
                    pairs[l] += 1;
                }
            }
            for l in 1..=NL {
                let mut prefix = vec![C::default(); NS - l + 1];
                for k in 0..NS - l {
                    let a = x[k] * x[k + l].conj();
                    prefix[k + 1] = prefix[k] + C::new(a.re as f64, a.im as f64);
                }
                for (g, gate) in gates.iter().enumerate() {
                    if gate.index < l || gate.index + gate.nu > NS - l {
                        continue;
                    }
                    let j = g * NL + l - 1;
                    sum[j] += (prefix[gate.index + gate.nu] - prefix[gate.index - l])
                        / (l + gate.nu) as f64;
                    count[j] += 1;
                }
            }
        }
        let nw = audit.nw;
        let nf = crate::lp480_screen::NF;
        let np = crate::lp480_screen::NP;
        let mut rd = vec![0.; nw * nf];
        let mut rdcount = 0.;
        for (t, spectrum) in audit
            .spectra
            .iter()
            .enumerate()
            .take(end_block / np)
            .skip(begin.div_ceil(np))
        {
            let a = t * np;
            let end = ((t + 1) * np).min(frames.len());
            if audit.keep[a..end].iter().all(|b| *b) {
                let count = (end - a) as f64;
                rdcount += count;
                for w in 0..nw {
                    for f in 0..nf {
                        rd[w * nf + f] += spectrum[w * nf + (f + nf / 2) % nf] as f64 * count;
                    }
                }
            }
        }
        if rdcount > 0. {
            rd.iter_mut().for_each(|x| *x /= rdcount);
        }
        blocks.push(Block {
            range_doppler_raw: rd,
            range_doppler_pulses: rdcount as usize,
            start,
            pulses: end_block - begin,
            kept,
            sums: sum.iter().map(|x| [x.re, x.im]).collect(),
            counts: count,
            noise: noise.iter().map(|x| [x.re, x.im]).collect(),
            noise_pairs: pairs,
        });
    }
    let d=Decoded{averaged:None,schema:"syisr-lp480-pooled-screen-v2".into(),job:job.clone(),gates,blocks,lags_us:(1..=NL).map(|l|l as f64*10.).collect(),rejection:"All-range 128-sample Hann Doppler spectra, 8-pulse averages; per-frequency temporal Q3+6IQR pooled over the full same-beam 30-minute integration, with narrow-feature/track support and broadband guard; reject bracketing pulses. Whole-pulse masks applied before any lag product. No decoded-amplitude outlier deletion.".into()};
    campaign::save(out, &d)?;
    eprintln!("PREPARED {} blocks={}", job.id, d.blocks.len());
    Ok(())
}
#[cfg(test)]
mod tests {
    use super::*;
    #[test]
    fn local_header_is_converted_before_window_and_block_assignment() -> Result<()> {
        let root = std::env::temp_dir().join(format!("lp480-clock-{}", std::process::id()));
        fs::create_dir_all(&root)?;
        let mut frame = vec![0u8; FRAME];
        frame[..4].copy_from_slice(&0xaaaa5555u32.to_le_bytes());
        frame[1020..1024].copy_from_slice(&0xaa5555aau32.to_le_bytes());
        frame[12..18].copy_from_slice(&[5, 24, 8, 11, 0, 1]);
        frame[32] = 1;
        frame[36] = 0x41;
        frame[764] = 1;
        frame[766..768].copy_from_slice(&767u16.to_le_bytes());
        frame[768..772].copy_from_slice(&(NS as u32).to_le_bytes());
        frame[858..860].copy_from_slice(&16384u16.to_le_bytes());
        frame[980..982].copy_from_slice(&16000u16.to_le_bytes());
        frame[988..990].copy_from_slice(&480u16.to_le_bytes());
        fs::write(root.join("raw.dat"), &frame)?;
        let m = index_with_offset(&root, &root.join("index.h5"), 28800)?;
        assert_eq!(m.jobs[0].start, 1715385600);
        assert_eq!(m.jobs[0].parts[0].block, 1715385660);
        assert_eq!(m.header_utc_offset_s, 28800);
        use std::io::Write;
        let mut f = fs::OpenOptions::new()
            .append(true)
            .open(root.join("raw.dat"))?;
        f.write_all(&frame[..1500])?;
        drop(f);
        assert!(index_with_options(&root, &root.join("strict.h5"), 28800, false).is_err());
        let partial = index_with_options(&root, &root.join("partial.h5"), 28800, true)?;
        assert_eq!(partial.jobs[0].parts[0].pulses, 1);
        assert_eq!(partial.quarantined_segments[0]["bytes"], 1500);
        let mut bad = vec![0u8; FRAME * 2];
        bad[..FRAME].copy_from_slice(&frame);
        fs::write(root.join("raw.dat"), bad)?;
        assert!(index_with_options(&root, &root.join("bad.h5"), 28800, true).is_err());
        fs::remove_dir_all(root)?;
        Ok(())
    }
    #[test]
    fn upper_gates_respect_beam_geometry_and_missing_long_lags() {
        for el in [45., 60., 75., 90.] {
            let j = Job {
                id: "test".into(),
                start: 0,
                az: 0.,
                el,
                front_us: 767.,
                parts: vec![],
            };
            let top = beam_top_for(&j, 1800);
            assert!(top > 1400 && top <= 1800);
            if el == 45. {
                assert!(top < 1800);
            } else if el == 90. {
                assert_eq!(top, 1800);
            }
            let g = gates_to(&j, 2000);
            assert!(g.windows(2).all(|w| w[1].altitude_km > w[0].altitude_km));
            assert!(g.iter().all(|g| g.index + g.nu + 1 <= NS));
            if el == 90. {
                let last = g.last().unwrap();
                assert!(last.index + last.nu + NL > NS);
            }
        }
    }
    #[test]
    fn byte_deletion_recovers_later_frames_without_using_corrupt_pulse() -> Result<()> {
        let root = std::env::temp_dir().join(format!("lp480-resync-{}", std::process::id()));
        fs::create_dir_all(&root)?;
        let mut frames = vec![];
        for k in 0..4 {
            let mut b = vec![0u8; FRAME];
            b[..4].copy_from_slice(&0xaaaa5555u32.to_le_bytes());
            b[1020..1024].copy_from_slice(&0xaa5555aau32.to_le_bytes());
            b[12..18].copy_from_slice(&[5, 24, 8, 11, 0, 1]);
            b[18..20].copy_from_slice(&(k as u16 * 640).to_le_bytes());
            b[32] = 1;
            b[36] = 0x41;
            b[764] = 1;
            b[766..768].copy_from_slice(&767u16.to_le_bytes());
            b[768..772].copy_from_slice(&(NS as u32).to_le_bytes());
            b[858..860].copy_from_slice(&16384u16.to_le_bytes());
            b[980..982].copy_from_slice(&16000u16.to_le_bytes());
            b[988..990].copy_from_slice(&480u16.to_le_bytes());
            frames.extend(b);
        }
        frames.drain(FRAME + 2048..FRAME + 2048 + 3072);
        fs::write(root.join("raw.dat"), frames)?;
        assert!(index_with_options(&root, &root.join("strict.h5"), 28800, true).is_err());
        let m = index_with_recovery(&root, &root.join("recovered.h5"), 28800, true, true)?;
        assert_eq!(m.jobs[0].parts.iter().map(|p| p.pulses).sum::<usize>(), 3);
        assert_eq!(m.quarantined_segments.len(), 1);
        assert_eq!(m.quarantined_segments[0]["offset"], FRAME as u64);
        assert_eq!(m.quarantined_segments[0]["bytes"], (FRAME - 3072) as u64);
        assert_eq!(m.jobs[0].parts[1].offset, (2 * FRAME - 3072) as u64);
        fs::remove_dir_all(root)?;
        Ok(())
    }
    #[test]
    fn field_epoch_tracks_observation_and_leap_year() {
        use chrono::TimeZone;
        let t = chrono::Utc
            .with_ymd_and_hms(2024, 7, 2, 0, 0, 0)
            .unwrap()
            .timestamp();
        assert_eq!(decimal_year(t).unwrap(), 2024.5);
        let storm = chrono::Utc
            .with_ymd_and_hms(2024, 5, 11, 0, 0, 0)
            .unwrap()
            .timestamp();
        assert!((decimal_year(storm).unwrap() - (2024. + 131. / 366.)).abs() < 1e-10);
    }
    #[test]
    fn finite_pulse_weight_conserves_uniform_acf() {
        let g = Gate {
            altitude_km: 400.,
            range_km: 400.,
            index: 400,
            nu: 3,
        };
        for l in 1..=NL {
            let w: f64 = (0..800)
                .map(|i| weight(&g, l, 767., [i as f64 * DT, (i + 1) as f64 * DT]))
                .sum();
            assert!((w - (1. - l as f64 / 48.)).abs() < 1e-11);
        }
    }
    #[test]
    fn ray_height_roundtrip() {
        for el in [45., 60., 75., 90.] {
            for z in [100., 400., 1400.] {
                assert!((altitude(range(z, el), el) - z).abs() < 1e-8);
            }
        }
    }
}

/// Repair an isolated <1-s date rollover only when adjacent on-disk frames
/// independently bracket it with the exact recorded IPP. No filename clock is used.
pub fn repair_index(path: &Path) -> Result<()> {
    let mut m: Manifest = campaign::load(path)?;
    let times: std::collections::BTreeSet<_> = m.jobs.iter().map(|j| j.start).collect();
    let mut keep = vec![];
    let mut corrected = vec![];
    for j in m.jobs {
        let n = j.parts.iter().map(|p| p.pulses).sum::<usize>();
        let isolated = !times.contains(&(j.start - 1800)) && !times.contains(&(j.start + 1800));
        if n >= 64 || !isolated {
            keep.push(j);
            continue;
        }
        for mut p in j.parts {
            ensure!(p.offset >= FRAME as u64, "unbracketed bad clock");
            let mut f = fs::File::open(&p.file)?;
            let mut h = [0u8; 1024];
            f.seek(SeekFrom::Start(p.offset - FRAME as u64))?;
            f.read_exact(&mut h)?;
            let before = Header::parse(&h)?.header_clock_unix_s - m.header_utc_offset_s as f64;
            f.seek(SeekFrom::Start(p.offset))?;
            f.read_exact(&mut h)?;
            let observed = Header::parse(&h)?.header_clock_unix_s - m.header_utc_offset_s as f64;
            f.seek(SeekFrom::Start(p.offset + p.pulses as u64 * FRAME as u64))?;
            f.read_exact(&mut h)?;
            let after = Header::parse(&h)?.header_clock_unix_s - m.header_utc_offset_s as f64;
            let first = before + 0.016;
            ensure!(
                (after - first - p.pulses as f64 * 0.016).abs() < 0.0001
                    && (observed - first).abs() > 86000.,
                "isolated clock segment lacks exact adjacent-frame evidence"
            );
            let start = (first as i64).div_euclid(1800) * 1800;
            p.block = (first as i64).div_euclid(10) * 10;
            m.clock_repairs.push(serde_json::json!({"file":p.file,"offset":p.offset,"pulses":p.pulses,"original_first_utc_s":observed,"corrected_first_utc_s":first,"method":"Exact preceding/following frame clocks and 16ms cadence bracket an isolated date-rollover error; raw headers unchanged"}));
            corrected.push(Job {
                id: format!(
                    "{start}-{:05}-{:05}",
                    (j.az / 360. * 65536.).round() as u16,
                    (j.el / 360. * 65536.).round() as u16
                ),
                start,
                az: j.az,
                el: j.el,
                front_us: j.front_us,
                parts: vec![p],
            });
        }
    }
    for j in corrected {
        if let Some(old) = keep.iter_mut().find(|x| x.id == j.id) {
            old.parts.extend(j.parts);
        } else {
            keep.push(j);
        }
    }
    keep.sort_by(|a, b| a.id.cmp(&b.id));
    m.jobs = keep;
    campaign::save(path, &m)?;
    Ok(())
}
