//! 480-us monostatic uncoded long-pulse products with explicit finite-pulse gating.
//! Recorded clock fields are retained as UTC; filenames are not observation times.
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
#[derive(Clone, Serialize, Deserialize)]
pub struct Part {
    pub file: String,
    pub offset: u64,
    pub pulses: usize,
    pub block: i64,
}
#[derive(Clone, Serialize, Deserialize)]
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
    let mut files = campaign::files(raw, ".dat")?;
    files.sort();
    let mut jobs = BTreeMap::<(i64, u16, u16), Job>::new();
    let mut total = 0;
    for p in files {
        let mut f = BufReader::new(fs::File::open(&p)?);
        let bytes = fs::metadata(&p)?.len();
        ensure!(
            bytes % FRAME as u64 == 0,
            "truncated or unsupported frame {}",
            p.display()
        );
        let mut offset = 0;
        while offset < bytes {
            let mut h = [0u8; 1024];
            f.read_exact(&mut h)?;
            let h = Header::parse(&h)?;
            ensure!(
                h.samples == NS
                    && h.waveform == 4
                    && h.pulse_width_us == 480
                    && h.ipp_us == 16000
                    && h.sample_rate_hz == 1e5,
                "unsupported 480-us layout at {}:{offset}",
                p.display()
            );
            let time = h.header_clock_unix_s;
            let start = (time as i64).div_euclid(1800) * 1800;
            let block = (time as i64).div_euclid(10) * 10;
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
    let mut z = 100f64;
    let mut out = vec![];
    while z <= 1400.0001 {
        let r = range(z, job.el);
        let nu = if z <= 400. {
            3
        } else {
            (3. + (z - 400.) * 7. / 1000.).round() as usize
        };
        let i = ((2. * r / C_KM + PW / 2. - (nu as f64 - 1.) * DT / 2. - job.front_us * 1e-6) / DT)
            .round();
        if i >= 1. && i + nu as f64 + (NL as f64) < NS as f64 {
            let t = job.front_us * 1e-6 + i * DT + (nu as f64 - 1.) * DT / 2. - PW / 2.;
            let rr = t * C_KM / 2.;
            out.push(Gate {
                altitude_km: altitude(rr, job.el),
                range_km: rr,
                index: i as usize,
                nu,
            });
        }
        if z == 1400. {
            break;
        }
        let step = if z <= 400. {
            15.
        } else {
            15. + (z - 400.) * 0.085
        };
        z = (z + step).min(1400.);
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
    if out.exists() {
        let old: Decoded = campaign::load(out)?;
        if old
            .blocks
            .first()
            .is_some_and(|b| !b.range_doppler_raw.is_empty())
        {
            return Ok(());
        }
    }

    let gates = gates(job);
    let m = gates.len() * NL;
    let mut groups = BTreeMap::<i64, Vec<&Part>>::new();
    for p in &job.parts {
        groups.entry(p.block).or_default().push(p);
    }
    let mut blocks = vec![];
    for (start, parts) in groups {
        let mut frames = vec![];
        for p in parts {
            let mut f = BufReader::new(fs::File::open(&p.file)?);
            f.seek(SeekFrom::Start(p.offset))?;
            for _ in 0..p.pulses {
                let fr =
                    crate::reader::read_frame(&mut f)?.ok_or_else(|| anyhow::anyhow!("EOF"))?;
                frames.push(fr.iq);
            }
        }
        if frames.len() < 64 {
            continue;
        }
        let audit = crate::lp480_screen::screen(&frames, NS);
        let kept = audit.keep.iter().filter(|x| **x).count();
        let mut sum = vec![C::default(); m];
        let mut count = vec![0usize; m];
        let mut noise = vec![C::default(); NL + 1];
        let mut pairs = vec![0usize; NL + 1];
        for (p, x) in frames.iter().enumerate() {
            if !audit.keep[p] {
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
        for (t, spectrum) in audit.spectra.iter().enumerate() {
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
            pulses: frames.len(),
            kept,
            sums: sum.iter().map(|x| [x.re, x.im]).collect(),
            counts: count,
            noise: noise.iter().map(|x| [x.re, x.im]).collect(),
            noise_pairs: pairs,
        });
    }
    let d=Decoded{averaged:None,schema:"syisr-lp480-fixed-support-v1".into(),job:job.clone(),gates,blocks,lags_us:(1..=NL).map(|l|l as f64*10.).collect(),rejection:"All-range 128-sample Hann Doppler spectra, 8-pulse averages; per-frequency temporal Q3+6IQR with narrow-feature/track support and broadband guard; reject bracketing pulses. Whole-pulse masks applied before any lag product. No decoded-amplitude outlier deletion.".into()};
    campaign::save(out, &d)?;
    eprintln!("PREPARED {} blocks={}", job.id, d.blocks.len());
    Ok(())
}
#[cfg(test)]
mod tests {
    use super::*;
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
            let before = Header::parse(&h)?.header_clock_unix_s;
            f.seek(SeekFrom::Start(p.offset))?;
            f.read_exact(&mut h)?;
            let observed = Header::parse(&h)?.header_clock_unix_s;
            f.seek(SeekFrom::Start(p.offset + p.pulses as u64 * FRAME as u64))?;
            f.read_exact(&mut h)?;
            let after = Header::parse(&h)?.header_clock_unix_s;
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
