//! SYISR 1024-byte raw-echo header and signed int32 Q,I samples.
//! Layout reference: readDataFromRawEcho0_ts.m supplied on the recording disk.
//! No alternating-code sequence, clock offset or absolute calibration is inferred.
use anyhow::{bail, ensure, Result};
use chrono::NaiveDate;
use num_complex::Complex32;
use serde::{Deserialize, Serialize};
use std::io::Read;
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct Header {
    pub header_clock_unix_s: f64,
    pub sample_rate_hz: f64,
    pub waveform: u8,
    pub bandwidth_code: u8,
    pub radar_frequency_hz: f64,
    pub code_width_raw_us: u16,
    pub gate_front_us: u16,
    pub samples: usize,
    pub azimuth_raw: u16,
    pub elevation_raw: u16,
    pub tx_azimuth_raw: u16,
    pub tx_elevation_raw: u16,
    pub big_code: u16,
    pub little_code: u16,
    pub ipp_us: u16,
    pub pulse_width_us: u16,
    pub mode: u16,
}
impl Header {
    pub fn parse(b: &[u8]) -> Result<Self> {
        ensure!(b.len() == 1024, "header must be 1024 bytes");
        let u16at = |o| u16::from_le_bytes([b[o], b[o + 1]]);
        let u32at = |o| u32::from_le_bytes(b[o..o + 4].try_into().unwrap());
        ensure!(
            u32at(0) == 0xaaaa5555 && u32at(1020) == 0xaa5555aa,
            "invalid frame sentinels"
        );
        ensure!(b[764] == 1, "unsupported DBF output beam count {}", b[764]);
        let date = NaiveDate::from_ymd_opt(2000 + b[13] as i32, b[12] as u32, b[15] as u32)
            .ok_or_else(|| anyhow::anyhow!("invalid frame date"))?;
        let time = date
            .and_hms_opt(b[14] as u32, b[17] as u32, b[16] as u32)
            .ok_or_else(|| anyhow::anyhow!("invalid frame clock"))?;
        let fractional = u16at(18) as f64 * 25e-6;
        ensure!(fractional < 1., "invalid 25 us clock ticks");
        let sample_rate_hz = match b[36] & 15 {
            0 => 4e6,
            1 => 1e5,
            2 => 2e5,
            3 => 4e5,
            4 => 20e6,
            5 => 16e6,
            x => bail!("unknown sample-rate code {x}"),
        };
        let samples = u32at(768) as usize;
        ensure!((1..=2_000_000).contains(&samples), "invalid sample count");
        let waveform = b[36] >> 4;
        ensure!(
            [0, 1, 2, 3, 4, 7].contains(&waveform),
            "unknown waveform {waveform}"
        );
        let h = Self {
            header_clock_unix_s: time.and_utc().timestamp() as f64 + fractional,
            sample_rate_hz,
            waveform,
            bandwidth_code: b[37] >> 4,
            radar_frequency_hz: (430. + b[32] as f64 * 0.2) * 1e6,
            code_width_raw_us: u16at(668),
            gate_front_us: u16at(766),
            samples,
            azimuth_raw: u16at(856),
            elevation_raw: u16at(858),
            tx_azimuth_raw: u16at(868),
            tx_elevation_raw: u16at(870),
            big_code: u16at(936),
            little_code: u16at(942),
            ipp_us: u16at(980),
            pulse_width_us: u16at(988),
            mode: u16at(40),
        };
        ensure!(
            h.ipp_us > 0 && h.pulse_width_us > 0 && h.pulse_width_us < h.ipp_us,
            "invalid pulse timing"
        );
        Ok(h)
    }
    pub fn azimuth_deg(&self) -> f64 {
        self.azimuth_raw as f64 * 360. / 65536.
    }
    pub fn elevation_deg(&self) -> f64 {
        self.elevation_raw as f64 * 360. / 65536.
    }
}
pub struct Frame {
    pub header: Header,
    pub iq: Vec<Complex32>,
}
pub fn read_frame(reader: &mut impl Read) -> Result<Option<Frame>> {
    let mut h = [0u8; 1024];
    let n = reader.read(&mut h[..1])?;
    if n == 0 {
        return Ok(None);
    }
    reader.read_exact(&mut h[1..])?;
    let header = Header::parse(&h)?;
    let mut bytes = vec![0u8; header.samples * 8];
    reader.read_exact(&mut bytes)?;
    // This campaign stores one extra I/Q pair after the MATLAB sample count.
    // It is deliberately excluded, exactly as in the supplied MATLAB reader.
    let mut extra = [0u8; 8];
    reader.read_exact(&mut extra)?;
    let iq = bytes
        .chunks_exact(8)
        .map(|b| {
            let q = i32::from_le_bytes(b[..4].try_into().unwrap());
            let i = i32::from_le_bytes(b[4..].try_into().unwrap());
            Complex32::new(i as f32, q as f32)
        })
        .collect();
    Ok(Some(Frame { header, iq }))
}
#[cfg(test)]
mod tests {
    use super::*;
    fn h() -> Vec<u8> {
        let mut b = vec![0; 1024];
        b[..4].copy_from_slice(&0xaaaa5555u32.to_le_bytes());
        b[1020..].copy_from_slice(&0xaa5555aau32.to_le_bytes());
        b[12..18].copy_from_slice(&[10, 25, 0, 6, 0, 43]);
        b[36] = 0x31;
        b[37] = 0x20;
        b[764] = 1;
        b[768..772].copy_from_slice(&2u32.to_le_bytes());
        b[980..982].copy_from_slice(&21330u16.to_le_bytes());
        b[988..990].copy_from_slice(&640u16.to_le_bytes());
        b
    }
    #[test]
    fn signed_qi_and_bitfields() {
        let mut b = h();
        b.extend((-123i32).to_le_bytes());
        b.extend(456i32.to_le_bytes());
        b.extend(i32::MIN.to_le_bytes());
        b.extend(i32::MAX.to_le_bytes());
        b.extend([0u8; 8]);
        let f = read_frame(&mut b.as_slice()).unwrap().unwrap();
        assert_eq!(f.header.waveform, 3);
        assert_eq!(f.header.sample_rate_hz, 1e5);
        assert_eq!(f.iq[0], Complex32::new(456., -123.));
        assert!(f.iq[1].im < 0.);
        assert!(read_frame(&mut &[][..]).unwrap().is_none());
    }
    #[test]
    fn truncated_and_multibeam_fail() {
        let b = h();
        assert!(read_frame(&mut b.as_slice()).is_err());
        let mut b = h();
        b[764] = 4;
        assert!(Header::parse(&b).is_err());
    }
}
