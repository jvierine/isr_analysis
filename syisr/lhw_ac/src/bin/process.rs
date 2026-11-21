//! UTC indexing, cycle-safe screening and lag decoding. Raw voltages stay on cluster.
use anyhow::{ensure, Context, Result};
use chrono::Utc;
use hdf5::File as H5;
use memmap2::Mmap;
use num_complex::{Complex32 as C32, Complex64 as C};
use rayon::prelude::*;
use std::{collections::BTreeMap, fs, path::Path};
use syisr_lhw_ac::{phases, plan, reader::Header, DT, NL};
#[derive(Clone, Copy)]
struct Ref {
    file: u16,
    offset: u64,
    t: f64,
}
struct Cycle {
    az: u16,
    el: u16,
    refs: Vec<Ref>,
}
fn bytes(h: &H5, k: &str, v: &str) -> Result<()> {
    h.new_dataset_builder().with_data(v.as_bytes()).create(k)?;
    Ok(())
}
fn load_maps(files: &[String]) -> Result<Vec<Mmap>> {
    files
        .iter()
        .map(|p| {
            let f = fs::File::open(p)?;
            Ok(unsafe { Mmap::map(&f)? })
        })
        .collect()
}
fn index(raw: &str, prefix: &str, out: &str) -> Result<()> {
    let mut files: Vec<_> = fs::read_dir(raw)?
        .filter_map(|e| e.ok())
        .map(|e| e.path())
        .filter(|p| {
            p.file_name().unwrap().to_string_lossy().starts_with(prefix)
                && p.file_name().unwrap().to_string_lossy().contains(".dat")
        })
        .collect();
    files.sort_by_key(|p| {
        p.file_name()
            .unwrap()
            .to_string_lossy()
            .rsplit('-')
            .next()
            .unwrap()
            .parse::<usize>()
            .unwrap()
    });
    ensure!(!files.is_empty(), "no raw files");
    let mut current: BTreeMap<(u16, u16), Vec<Ref>> = BTreeMap::new();
    let mut cycles = vec![];
    let mut count = 0u64;
    let mut dropped = 0u64;
    let mut prev = None;
    let mut first = 0.;
    let mut last = 0.;
    let mut n = 0;
    let mut gate = 0;
    for (fi, p) in files.iter().enumerate() {
        let f = fs::File::open(p)?;
        let m = unsafe { Mmap::map(&f)? };
        let h = Header::parse(&m[..1024])?;
        n = h.samples;
        gate = h.gate_front_us;
        let stride = 1024 + (n + 1) * 8;
        let remainder = m.len() % stride;
        ensure!(
            remainder == 0 || fi + 1 == files.len(),
            "incomplete interior file"
        );
        for off in (0..m.len() / stride * stride).step_by(stride) {
            let h = Header::parse(&m[off..off + 1024])?;
            ensure!(
                h.waveform == 3
                    && h.sample_rate_hz == 100000.
                    && h.pulse_width_us == 640
                    && h.ipp_us == 16000
                    && h.big_code == 51
                    && h.samples == n
                    && h.gate_front_us == gate,
                "unsupported experiment change"
            );
            ensure!(
                h.azimuth_raw == h.tx_azimuth_raw && h.elevation_raw == h.tx_elevation_raw,
                "TX/RX mismatch"
            );
            let t = h.header_clock_unix_s - 28800.;
            if count == 0 {
                first = t;
            }
            if let Some(old) = prev {
                ensure!(t > old, "nonmonotonic raw clock");
                if t - old > 0.025 {
                    current.clear();
                }
            }
            prev = Some(t);
            last = t;
            count += 1;
            let key = (h.azimuth_raw, h.elevation_raw);
            let c = current.entry(key).or_default();
            let cid = h.little_code as usize;
            if cid == 0 {
                dropped += c.len() as u64;
                c.clear();
            }
            if cid != c.len() {
                dropped += 1;
                continue;
            }
            c.push(Ref {
                file: fi as u16,
                offset: off as u64,
                t,
            });
            if c.len() == 64 {
                cycles.push(Cycle {
                    az: key.0,
                    el: key.1,
                    refs: std::mem::take(c),
                });
            }
        }
        println!(
            "INDEX {} pulses={} complete_cycles={}",
            p.display(),
            count,
            cycles.len()
        );
    }
    let mut beams: Vec<_> = cycles.iter().map(|c| (c.az, c.el)).collect();
    beams.sort_by(|a, b| b.1.cmp(&a.1));
    beams.dedup();
    ensure!(beams.len() == 5, "expected five recorded directions");
    cycles.sort_by(|a, b| a.refs[0].t.total_cmp(&b.refs[0].t));
    let nc = cycles.len();
    let h = H5::create(out)?;
    let paths: Vec<_> = files
        .iter()
        .map(|p| p.to_string_lossy().to_string())
        .collect();
    bytes(&h, "files_json_utf8", &serde_json::to_string(&paths)?)?;
    bytes(&h,"metadata_json_utf8",&serde_json::json!({"schema":"syisr-lhw-ac-index-v1","clock":"Beijing header converted to UTC by subtracting28800s","raw_first_utc_s":first,"raw_last_utc_s":last,"raw_pulses":count,"discarded_startup_or_partial_pulses":dropped,"samples":n,"stored_complex_words":n+1,"gate_front_us":gate,"sample_rate_hz":100000,"pulse_us":640,"baud_us":20,"phase_set":51,"ipp_us":16000,"cycle":"64 code members gathered by actual beam and code index, interleaved across five beams"}).to_string())?;
    let times: Vec<_> = cycles
        .iter()
        .flat_map(|c| [c.refs[0].t, c.refs[63].t])
        .collect();
    h.new_dataset::<f64>()
        .shape([nc, 2])
        .create("cycle_time_utc_s")?
        .write_raw(&times)?;
    let bid: Vec<_> = cycles
        .iter()
        .map(|c| beams.iter().position(|b| *b == (c.az, c.el)).unwrap() as u8)
        .collect();
    h.new_dataset_builder()
        .with_data(&bid)
        .create("beam_index")?;
    let offsets: Vec<_> = cycles
        .iter()
        .flat_map(|c| c.refs.iter().map(|r| r.offset))
        .collect();
    let ids: Vec<_> = cycles
        .iter()
        .flat_map(|c| c.refs.iter().map(|r| r.file))
        .collect();
    h.new_dataset::<u64>()
        .shape([nc, 64])
        .deflate(4)
        .create("frame_offset")?
        .write_raw(&offsets)?;
    h.new_dataset::<u16>()
        .shape([nc, 64])
        .deflate(4)
        .create("file_index")?
        .write_raw(&ids)?;
    let angles: Vec<_> = beams
        .iter()
        .flat_map(|(a, e)| [*a as f64 * 360. / 65536., *e as f64 * 360. / 65536.])
        .collect();
    h.new_dataset::<f64>()
        .shape([5, 2])
        .create("beam_az_el_deg")?
        .write_raw(&angles)?;
    println!("INDEX COMPLETE {nc} cycles, {}", Utc::now());
    Ok(())
}
struct Index {
    files: Vec<String>,
    time: Vec<f64>,
    beam: Vec<u8>,
    offset: Vec<u64>,
    file: Vec<u16>,
    angles: Vec<f64>,
    n: usize,
    gate: f64,
    anchor: f64,
    blocks: usize,
}
impl Index {
    fn read(p: &str) -> Result<Self> {
        let h = H5::open(p)?;
        let meta: serde_json::Value =
            serde_json::from_slice(&h.dataset("metadata_json_utf8")?.read_raw::<u8>()?)?;
        let time = h.dataset("cycle_time_utc_s")?.read_raw::<f64>()?;
        let anchor = (time[0] / 300.).floor() * 300.;
        let end = time.chunks(2).map(|t| t[1]).fold(0f64, f64::max);
        Ok(Self {
            files: serde_json::from_slice(&h.dataset("files_json_utf8")?.read_raw::<u8>()?)?,
            time,
            beam: h.dataset("beam_index")?.read_raw()?,
            offset: h.dataset("frame_offset")?.read_raw()?,
            file: h.dataset("file_index")?.read_raw()?,
            angles: h.dataset("beam_az_el_deg")?.read_raw()?,
            n: meta["samples"].as_u64().unwrap() as usize,
            gate: meta["gate_front_us"].as_f64().unwrap(),
            anchor,
            blocks: ((end - anchor) / 300.).floor() as usize,
        })
    }
}
struct Product {
    acf: Vec<C>,
    raw: Vec<C>,
    counts: Vec<u32>,
    power: Vec<f64>,
    bad: Vec<u8>,
    times: Vec<f64>,
    nc: usize,
    candidates: usize,
    screened: usize,
    noise: Vec<C>,
}
fn beam_product(
    ix: &Index,
    maps: &[Mmap],
    codes: &[Vec<f64>],
    terms: &[Vec<syisr_lhw_ac::Term>],
    block: usize,
    b: usize,
    ng: usize,
) -> Result<Product> {
    let start = ix.anchor + block as f64 * 300.;
    let stop = start + 300.;
    let chosen: Vec<_> = (0..ix.beam.len())
        .filter(|&i| {
            ix.beam[i] as usize == b && ix.time[i * 2] >= start && ix.time[i * 2 + 1] < stop
        })
        .collect();
    let nc = chosen.len();
    ensure!(nc >= 2, "too few complete cycles");
    let n = ix.n;
    let np = nc * 64;
    let mut x = vec![C32::default(); np * n];
    for (j, &ci) in chosen.iter().enumerate() {
        for c in 0..64 {
            let r = ci * 64 + c;
            let off = ix.offset[r] as usize + 1024;
            let m = &maps[ix.file[r] as usize];
            for k in 0..n {
                let p = off + k * 8;
                let q = i32::from_le_bytes(m[p..p + 4].try_into().unwrap()) as f32;
                let i = i32::from_le_bytes(m[p + 4..p + 8].try_into().unwrap()) as f32;
                x[(j * 64 + c) * n + k] = C32::new(i, q);
            }
        }
    }
    let (mut mask, summary) = syisr_lhw_ac::coded_screen::screen(&x, np, n, codes, 4);
    // Persistent coherent range support is retained; only transient code-match masks.
    for k in 0..n {
        let hits = (0..np).filter(|&p| mask[p * n + k] & 2 != 0).count();
        if hits * 4 > np {
            for p in 0..np {
                mask[p * n + k] &= !2;
            }
        }
    }
    let mut noise = vec![C::default(); NL + 1];
    let mut noise_count = vec![0u64; NL + 1];
    for p in 0..np {
        for l in 0..=NL {
            for k in 1000..1150 {
                let i = p * n + k;
                if mask[i] == 0 && mask[i + l] == 0 {
                    let a = x[i];
                    let b = x[i + l];
                    noise[l] +=
                        C::new(b.re as f64, b.im as f64) * C::new(a.re as f64, -(a.im as f64));
                    noise_count[l] += 1;
                }
            }
        }
    }
    for l in 0..=NL {
        ensure!(noise_count[l] > 100, "far reference rejected");
        noise[l] /= noise_count[l] as f64;
    }
    let mut bad = vec![0u8; nc * ng];
    let mut counts = vec![nc as u32; ng];
    let mut power = vec![0.; ng];
    let need = ng * 2 + 64;
    ensure!(need + NL < n, "range support outside recording");
    let mut bank = vec![C::default(); 64 * NL * need];
    for cy in 0..nc {
        let mut support = vec![false; n];
        for c in 0..64 {
            for k in 0..n {
                support[k] |= mask[(cy * 64 + c) * n + k] != 0;
            }
        }
        for g in 0..ng {
            let k = g * 2;
            if support[k..k + 64].iter().any(|v| *v) {
                bad[cy * ng + g] = 1;
                counts[g] -= 1;
            } else {
                let mut p = 0.;
                for c in 0..64 {
                    for j in k..k + 64 {
                        p += x[(cy * 64 + c) * n + j].norm_sqr() as f64;
                    }
                }
                power[g] += p / (64 * 64) as f64;
            }
        }
        for c in 0..64 {
            let y = &x[(cy * 64 + c) * n..(cy * 64 + c + 1) * n];
            for l in 1..=NL {
                let dst = &mut bank[(c * NL + l - 1) * need..(c * NL + l) * need];
                for k in 0..need {
                    let a = y[k];
                    let b = y[k + l];
                    dst[k] +=
                        C::new(b.re as f64, b.im as f64) * C::new(a.re as f64, -(a.im as f64));
                }
            }
        }
    }
    let mut total = vec![C::default(); ng * (NL + 1)];
    for g in 0..ng {
        for l in 1..=NL {
            total[g * (NL + 1) + l] = terms[l - 1]
                .iter()
                .map(|t| bank[(t.c * NL + l - 1) * need + g * 2 + t.k] * t.w)
                .sum();
        }
    }
    let raw = total.clone();
    for cy in 0..nc {
        for g in 0..ng {
            if bad[cy * ng + g] == 0 {
                continue;
            }
            for l in 1..=NL {
                let mut q = C::default();
                for t in &terms[l - 1] {
                    let k = (cy * 64 + t.c) * n + g * 2 + t.k;
                    let a = x[k];
                    let b = x[k + l];
                    q += C::new(b.re as f64, b.im as f64)
                        * C::new(a.re as f64, -(a.im as f64))
                        * t.w;
                }
                total[g * (NL + 1) + l] -= q;
            }
        }
    }
    let mut un = raw;
    for g in 0..ng {
        for l in 1..=NL {
            let pedestal = noise[l] * terms[l - 1].iter().map(|t| t.w).sum::<f64>();
            un[g * (NL + 1) + l] = un[g * (NL + 1) + l] / nc as f64 - pedestal;
            total[g * (NL + 1) + l] = if counts[g] > 0 {
                total[g * (NL + 1) + l] / counts[g] as f64 - pedestal
            } else {
                C::new(f64::NAN, f64::NAN)
            };
        }
        un[g * (NL + 1)] = C::new(
            syisr_lhw_ac::zero_lag(un[g * (NL + 1) + 1], un[g * (NL + 1) + 2]),
            0.,
        );
        total[g * (NL + 1)] = C::new(
            syisr_lhw_ac::zero_lag(total[g * (NL + 1) + 1], total[g * (NL + 1) + 2]),
            0.,
        );
        power[g] = if counts[g] > 0 {
            power[g] / counts[g] as f64
        } else {
            f64::NAN
        };
    }
    Ok(Product {
        acf: total,
        raw: un,
        counts,
        power,
        bad,
        times: chosen
            .iter()
            .map(|&i| (ix.time[i * 2] + ix.time[i * 2 + 1]) / 2.)
            .collect(),
        nc,
        candidates: summary.candidates,
        screened: summary.masked,
        noise,
    })
}
fn worker(index: &str, phase: &str, out: &str) -> Result<()> {
    let ix = Index::read(index)?;
    let maps = load_maps(&ix.files)?;
    let codes = phases(Path::new(phase))?;
    let terms = plan(&codes);
    let rank = std::env::var("SLURM_PROCID")
        .unwrap_or("0".into())
        .parse::<usize>()?;
    let ranks = std::env::var("SLURM_NTASKS")
        .unwrap_or("1".into())
        .parse::<usize>()?;
    fs::create_dir_all(out)?;
    let ng = 267;
    for block in (rank..ix.blocks).step_by(ranks) {
        let path = Path::new(out).join(format!("{block:04}.h5"));
        if path.exists() {
            continue;
        }
        let results: Vec<Result<Product>> = (0..5)
            .into_par_iter()
            .map(|b| beam_product(&ix, &maps, &codes, &terms, block, b, ng))
            .collect();
        let tmp = path.with_extension("partial.h5");
        let h = H5::create(&tmp)?;
        h.new_dataset_builder()
            .with_data(&[
                ix.anchor + block as f64 * 300.,
                ix.anchor + (block + 1) as f64 * 300.,
            ])
            .create("interval_utc_s")?;
        let ranges: Vec<_> = (0..ng)
            .map(|g| (ix.gate + g as f64 * 20.) * 0.149896229)
            .collect();
        h.new_dataset_builder()
            .with_data(&ranges)
            .create("slant_range_km")?;
        let lag: Vec<_> = (0..=NL).map(|l| l as f64 * DT).collect();
        h.new_dataset_builder().with_data(&lag).create("lag_s")?;
        h.new_dataset::<f64>()
            .shape([5, 2])
            .create("beam_az_el_deg")?
            .write_raw(&ix.angles)?;
        for (b, p) in results.into_iter().enumerate() {
            let p = p?;
            let g = h.create_group(&format!("beam_{b}"))?;
            for (name, v) in [("acf_adu2", p.acf), ("unscreened_acf_adu2", p.raw)] {
                let flat: Vec<_> = v.iter().flat_map(|v| [v.re, v.im]).collect();
                g.new_dataset::<f64>()
                    .shape([ng, NL + 1, 2])
                    .deflate(4)
                    .create(name)?
                    .write_raw(&flat)?;
            }
            let nv: Vec<_> = p.noise.iter().flat_map(|v| [v.re, v.im]).collect();
            g.new_dataset::<f64>()
                .shape([NL + 1, 2])
                .create("far_reference_acf_adu2")?
                .write_raw(&nv)?;
            g.new_dataset_builder()
                .with_data(&p.counts)
                .create("accepted_cycles")?;
            g.new_dataset_builder()
                .with_data(&p.power)
                .create("received_support_power_adu2")?;
            g.new_dataset::<u8>()
                .shape([p.nc, ng])
                .deflate(4)
                .create("rejected_cycle_gate")?
                .write_raw(&p.bad)?;
            g.new_dataset_builder()
                .with_data(&p.times)
                .create("cycle_time_utc_s")?;
            g.new_dataset_builder()
                .with_data(&[p.candidates as u64, p.screened as u64])
                .create("screen_summary_counts")?;
        }
        bytes(&h,"metadata_json_utf8",&serde_json::json!({"schema":"syisr-lhw-gated-acf-v1","version":env!("CARGO_PKG_VERSION"),"frequency_sign":"R(tau)=x(t+tau)conj(x(t)); FFT yields positive f for positive IQ rotation; velocity sign uncalibrated","background":"sample lags1000..1149, about1600..1824km slant range, masks applied; subtract through exact decode coefficient sum, preserving signed estimates; far-reference approximation may contain weak plasma", "zero_lag":"local even-quadratic extrapolation (4 Re R10us - Re R20us)/3; signed; lag-dependent range-kernel approximation, not measured decoded zero lag","range":"nominal leading-edge gate delay667us plus20us native cell index; no inherited uncoded restart shift; half-baud kernels differ","screen":"full-band code-Doppler matched transients plus burst/clipping mask; common cycle/gate masks over all64 members and full640us support; persistent code-coherent supports retained; satellite candidates not independently identified"}).to_string())?;
        drop(h);
        fs::rename(tmp, path)?;
        println!(
            "BLOCK {block}/{} rank={rank} complete {}",
            ix.blocks,
            Utc::now()
        );
    }
    Ok(())
}
fn main() -> Result<()> {
    let a: Vec<_> = std::env::args().collect();
    match a.get(1).map(String::as_str) {
        Some("index") => index(&a[2], &a[3], &a[4]),
        Some("worker") => worker(&a[2], &a[3], &a[4]),
        _ => anyhow::bail!("process index RAW PREFIX INDEX.h5 | worker INDEX.h5 PHASE OUTPUT"),
    }
    .context("SYISR gated AC")
}
