use crate::reader::Header;
use anyhow::{ensure, Result};
use serde::{de::DeserializeOwned, Deserialize, Serialize};
use std::{
    fs,
    io::{BufReader, Read, Seek, SeekFrom},
    path::{Path, PathBuf},
};
#[derive(Clone, Debug, Serialize, Deserialize)]
pub struct Span {
    pub file: String,
    pub offset: u64,
    pub pulses: usize,
}
#[derive(Clone, Debug, Serialize, Deserialize)]
pub struct Dwell {
    pub id: usize,
    pub start: f64,
    pub end: f64,
    pub az: f64,
    pub el: f64,
    pub az_raw: u16,
    pub el_raw: u16,
    pub pulses: usize,
    pub first_code: u16,
    pub spans: Vec<Span>,
}
#[derive(Clone, Debug, Serialize, Deserialize)]
pub struct Job {
    pub id: String,
    pub start: f64,
    pub end: f64,
    pub az: f64,
    pub el: f64,
    pub dwells: Vec<usize>,
}
#[derive(Clone, Debug, Serialize, Deserialize)]
pub struct Quarantine {
    pub file: String,
    pub first_offset: u64,
    pub frames: usize,
    pub bytes: u64,
    pub first_clock: Option<f64>,
    pub last_clock: Option<f64>,
    pub reason: String,
}
#[derive(Clone, Debug, Serialize, Deserialize)]
pub struct Campaign {
    pub schema: String,
    pub raw_root: String,
    pub phase_table: String,
    pub l2: Option<String>,
    pub dwells: Vec<Dwell>,
    pub jobs: Vec<Job>,
    pub files: Vec<String>,
    pub quarantined: Vec<Quarantine>,
}
pub fn save<T: Serialize>(path: &Path, value: &T) -> Result<()> {
    if let Some(p) = path.parent() {
        fs::create_dir_all(p)?;
    }
    let tmp = path.with_extension("h5.partial");
    let f = hdf5::File::create(&tmp)?;
    f.new_dataset_builder()
        .with_data(&serde_json::to_vec(value)?)
        .deflate(4)
        .create("json_utf8")?;
    drop(f);
    fs::rename(tmp, path)?;
    Ok(())
}
pub fn load<T: DeserializeOwned>(path: &Path) -> Result<T> {
    let f = hdf5::File::open(path)?;
    Ok(serde_json::from_slice(
        &f.dataset("json_utf8")?.read_raw::<u8>()?,
    )?)
}
pub fn files(path: &Path, ext: &str) -> Result<Vec<PathBuf>> {
    let mut a = vec![];
    for e in fs::read_dir(path)? {
        let p = e?.path();
        if p.is_file() && p.file_name().unwrap().to_string_lossy().contains(ext) {
            a.push(p)
        }
    }
    a.sort();
    Ok(a)
}
// Resynchronize only after corrupt sentinels. A plausible complete1024-byte
// header with both64-bit framing constraints and valid fields is required.
pub(crate) fn resync(f: &mut BufReader<fs::File>, from: u64, bytes: u64) -> Result<Option<u64>> {
    let mut position = from;
    while position + 1024 <= bytes {
        f.seek(SeekFrom::Start(position))?;
        let take = ((bytes - position) as usize).min(1 << 20);
        let mut block = vec![0u8; take];
        f.read_exact(&mut block)?;
        for j in 0..=take - 1024 {
            if block[j..j + 4] == 0xaaaa5555u32.to_le_bytes()
                && block[j + 1020..j + 1024] == 0xaa5555aau32.to_le_bytes()
                && Header::parse(&block[j..j + 1024]).is_ok()
            {
                return Ok(Some(position + j as u64));
            }
        }
        position += (take - 1023) as u64;
    }
    Ok(None)
}
pub fn index(raw: &Path, phase: &Path, l2: Option<String>, out: &Path) -> Result<Campaign> {
    let mut source = files(raw, ".dat")?;
    ensure!(!source.is_empty(), "no raw files");
    // Numeric acquisition suffix, not lexical ordering (13,14,...,100).
    source.sort_by_key(|p| {
        p.file_name()
            .unwrap()
            .to_string_lossy()
            .rsplit('-')
            .next()
            .and_then(|s| s.parse::<u64>().ok())
            .unwrap_or(0)
    });
    let mut dwells: Vec<Dwell> = vec![];
    let mut prevtime = None;
    let mut prevcode = None;
    let mut quarantine_since_last = false;
    let mut quarantined: Vec<Quarantine> = vec![];
    for p in &source {
        let mut f = BufReader::with_capacity(1 << 20, fs::File::open(p)?);
        let bytes = fs::metadata(p)?.len();
        let mut offset = 0u64;
        while offset < bytes {
            if bytes - offset < 1024 {
                quarantined.push(Quarantine {
                    file: p.display().to_string(),
                    first_offset: offset,
                    frames: 0,
                    bytes: bytes - offset,
                    first_clock: None,
                    last_clock: None,
                    reason: "truncated final header; exact bytes excluded".into(),
                });
                break;
            }
            let mut h = [0u8; 1024];
            f.read_exact(&mut h)?;
            let sentinels =
                h[..4] == 0xaaaa5555u32.to_le_bytes() && h[1020..] == 0xaa5555aau32.to_le_bytes();
            if !sentinels {
                let next = resync(&mut f, offset + 1, bytes)?;
                let stop = next.unwrap_or(bytes);
                quarantined.push(Quarantine {
                    file: p.display().to_string(),
                    first_offset: offset,
                    frames: 0,
                    bytes: stop - offset,
                    first_clock: None,
                    last_clock: None,
                    reason:
                        "corrupt framing bytes; excluded until next independently validated header"
                            .into(),
                });
                quarantine_since_last = true;
                if let Some(next) = next {
                    offset = next;
                    f.seek(SeekFrom::Start(offset))?;
                    continue;
                } else {
                    break;
                }
            }
            // Good framing with an unsupported layout is still a hard error.
            let h = Header::parse(&h)?;
            if offset + 1024 + h.samples as u64 * 8 > bytes {
                quarantined.push(Quarantine {
                    file: p.display().to_string(),
                    first_offset: offset,
                    frames: 1,
                    bytes: bytes - offset,
                    first_clock: Some(h.header_clock_unix_s - 28800.),
                    last_clock: Some(h.header_clock_unix_s - 28800.),
                    reason: "truncated final IQ payload; excluded".into(),
                });
                break;
            }
            ensure!(
                h.waveform == 3
                    && h.big_code == 51
                    && h.sample_rate_hz == 1e5
                    && h.samples == 1334
                    && h.pulse_width_us == 640
                    && h.little_code < 64,
                "unsupported AC layout in {}",
                p.display()
            );
            // Main-site Tx/Rx pointing mismatch is not bistatic geometry: the site is shared.
            // Beam-pattern uncertainty remains an empirical-density/model limitation.
            let t = h.header_clock_unix_s - 28800.;
            if prevtime.is_some_and(|old| t <= old) {
                let file = p.display().to_string();
                let append = quarantined
                    .last()
                    .map(|q| {
                        q.file == file
                            && q.first_offset + q.bytes == offset
                            && q.reason.starts_with("non-increasing")
                    })
                    .unwrap_or(false);
                if append {
                    let q = quarantined.last_mut().unwrap();
                    q.frames += 1;
                    q.bytes += 1024 + 1334 * 8;
                    q.last_clock = Some(t);
                } else {
                    quarantined.push(Quarantine {
                        file,
                        first_offset: offset,
                        frames: 1,
                        bytes: 1024 + 1334 * 8,
                        first_clock: Some(t),
                        last_clock: Some(t),
                        reason: "non-increasing raw calendar timestamp; excluded, raw unchanged"
                            .into(),
                    });
                }
                quarantine_since_last = true;
                let skip = (h.samples * 8) as u64;
                offset += 1024 + skip;
                ensure!(offset <= bytes, "truncated quarantined frame");
                f.seek_relative(skip as i64)?;
                continue;
            }
            prevtime = Some(t);
            let fresh = quarantine_since_last
                || prevcode.is_some_and(|v: u16| h.little_code != (v + 1) % 64)
                || dwells
                    .last()
                    .map(|d| {
                        d.az_raw != h.azimuth_raw
                            || d.el_raw != h.elevation_raw
                            || t - d.end > 0.03
                            || (d.start / 1800.).floor() != (t / 1800.).floor()
                    })
                    .unwrap_or(true);
            quarantine_since_last = false;
            prevcode = Some(h.little_code);
            if fresh {
                dwells.push(Dwell {
                    id: dwells.len(),
                    start: t,
                    end: t,
                    az: h.azimuth_deg(),
                    el: h.elevation_deg(),
                    az_raw: h.azimuth_raw,
                    el_raw: h.elevation_raw,
                    pulses: 0,
                    first_code: h.little_code,
                    spans: vec![],
                });
            }
            let d = dwells.last_mut().unwrap();
            let name = p.to_string_lossy().to_string();
            if d.spans.last().map(|s| s.file.as_str()) != Some(name.as_str())
                || d.spans
                    .last()
                    .is_some_and(|s| s.offset + s.pulses as u64 * (1024 + 1334 * 8) != offset)
            {
                d.spans.push(Span {
                    file: name,
                    offset,
                    pulses: 0,
                });
            }
            d.spans.last_mut().unwrap().pulses += 1;
            d.pulses += 1;
            d.end = t;
            let skip = (h.samples * 8) as u64;
            offset += 1024 + skip;
            ensure!(offset <= bytes, "truncated frame");
            f.seek_relative(skip as i64)?;
        }
        eprintln!("indexed {}: {} dwells", p.display(), dwells.len());
    }
    let mut map = std::collections::BTreeMap::<(i64, u16, u16), Vec<usize>>::new();
    for d in &dwells {
        let trim = (64 - d.first_code as usize) % 64;
        if d.pulses < trim + 64 {
            continue;
        }
        let bin = (d.start / 1800.).floor() as i64;
        map.entry((bin, d.az_raw, d.el_raw)).or_default().push(d.id);
    }
    let jobs = map
        .into_iter()
        .map(|((b, az, el), ids)| Job {
            id: format!("{}_az{}_el{}", b * 1800, az, el),
            start: b as f64 * 1800.,
            end: b as f64 * 1800. + 1800.,
            az: az as f64 * 360. / 65536.,
            el: el as f64 * 360. / 65536.,
            dwells: ids,
        })
        .collect();
    let c = Campaign {
        schema: "syisr-full-ac-native-adaptive-v2".into(),
        raw_root: raw.display().to_string(),
        phase_table: phase.display().to_string(),
        l2,
        dwells,
        jobs,
        files: source.iter().map(|p| p.display().to_string()).collect(),
        quarantined,
    };
    save(out, &c)?;
    Ok(c)
}
pub fn altitude(range: f64, el: f64) -> f64 {
    let re = 6371.;
    (6371.05f64.powi(2) + range * range + 2. * 6371.05 * range * el.to_radians().sin()).sqrt() - re
}
pub fn range(height: f64, el: f64) -> f64 {
    let b = 6371.05 * el.to_radians().sin();
    -b + (b * b + (6371. + height).powi(2) - 6371.05f64.powi(2)).sqrt()
}
pub fn width(z: f64) -> f64 {
    if z <= 400. {
        15.
    } else {
        let end = 10. * (6771f64 / 8371.).powi(2);
        15. * (1. + (end - 1.) * (z - 400.) / 1600.) * ((6371. + z) / 6771.).powi(2)
    }
}
#[cfg(test)]
mod tests {
    use super::*;
    #[test]
    fn geometry_roundtrip() {
        for e in [30., 45., 60., 75., 90.] {
            for h in [100., 500., 1000.] {
                assert!((altitude(range(h, e), e) - h).abs() < 1e-8)
            }
        }
    }
    #[test]
    fn windows() {
        assert_eq!(width(400.), 15.);
        assert!((width(2000.) - 150.).abs() < 1e-10);
    }
}
#[cfg(test)]
mod corrupt_frame_tests {
    use super::*;
    fn header(code: u16, ticks: u16) -> Vec<u8> {
        let mut h = vec![0u8; 1024];
        h[..4].copy_from_slice(&0xaaaa5555u32.to_le_bytes());
        h[1020..].copy_from_slice(&0xaa5555aau32.to_le_bytes());
        h[12..18].copy_from_slice(&[10, 25, 7, 6, 0, 30]);
        h[18..20].copy_from_slice(&ticks.to_le_bytes());
        h[36] = 0x31;
        h[764] = 1;
        h[768..772].copy_from_slice(&1334u32.to_le_bytes());
        h[936..938].copy_from_slice(&51u16.to_le_bytes());
        h[942..944].copy_from_slice(&code.to_le_bytes());
        h[980..982].copy_from_slice(&21330u16.to_le_bytes());
        h[988..990].copy_from_slice(&640u16.to_le_bytes());
        h[858..860].copy_from_slice(&16384u16.to_le_bytes());
        h[870..872].copy_from_slice(&16384u16.to_le_bytes());
        h
    }
    fn temp(name: &str) -> PathBuf {
        let p = std::env::temp_dir().join(format!(
            "syisr-{name}-{}-{}",
            std::process::id(),
            std::time::SystemTime::now()
                .duration_since(std::time::UNIX_EPOCH)
                .unwrap()
                .as_nanos()
        ));
        fs::create_dir_all(&p).unwrap();
        p
    }
    #[test]
    fn missing_header_payload_is_quarantined_and_next_frame_recovered() {
        let root = temp("resync");
        let mut raw = header(0, 0);
        raw.extend(vec![0; 1334 * 8]);
        raw.extend(vec![0; 1334 * 8]);
        let next = raw.len() as u64;
        raw.extend(header(2, 1706));
        raw.extend(vec![0; 1334 * 8]);
        raw.extend([0u8; 17]);
        fs::write(root.join("test.dat-1"), raw).unwrap();
        let c = index(
            &root,
            &root.join("phase.bc"),
            None,
            &root.join("campaign.h5"),
        )
        .unwrap();
        assert_eq!(c.dwells.iter().map(|d| d.pulses).sum::<usize>(), 2);
        assert_eq!(c.dwells.last().unwrap().spans[0].offset, next);
        assert_eq!(c.quarantined.len(), 2);
        assert_eq!(c.quarantined[0].bytes, 1334 * 8);
        assert_eq!(c.quarantined[1].bytes, 17);
        fs::remove_dir_all(root).unwrap();
    }
    #[test]
    fn backwards_clock_does_not_bridge_a_valid_span() {
        let root = temp("clock");
        let mut raw = vec![];
        for (code, ticks) in [(0, 0), (1, 853), (2, 853), (3, 1706)] {
            raw.extend(header(code, ticks));
            raw.extend(vec![0; 1334 * 8]);
        }
        fs::write(root.join("test.dat-1"), raw).unwrap();
        let c = index(
            &root,
            &root.join("phase.bc"),
            None,
            &root.join("campaign.h5"),
        )
        .unwrap();
        assert_eq!(c.dwells.iter().map(|d| d.pulses).sum::<usize>(), 3);
        assert_eq!(c.quarantined.len(), 1);
        assert_eq!(c.quarantined[0].frames, 1);
        assert_eq!(
            c.dwells.last().unwrap().spans[0].offset,
            3 * (1024 + 1334 * 8)
        );
        fs::remove_dir_all(root).unwrap();
    }
}
