//! Header-sampled primary-site archive survey; candidate bytes are not exposure.
use anyhow::{ensure, Result};
use rayon::prelude::*;
use std::{
    collections::BTreeSet,
    io::{Read, Seek, SeekFrom},
    path::Path,
};
use syisr_ac::{campaign, reader::Header};
fn key(h: &Header) -> serde_json::Value {
    serde_json::json!({"waveform":h.waveform,"sample_rate_hz":h.sample_rate_hz,"radar_frequency_hz":h.radar_frequency_hz,"pulse_us":h.pulse_width_us,"ipp_us":h.ipp_us,"front_us":h.gate_front_us,"samples":h.samples,"bandwidth_code":h.bandwidth_code,"mode":h.mode,"code_width_raw_us":h.code_width_raw_us})
}
fn header(f: &mut std::fs::File, offset: u64) -> Result<Header> {
    f.seek(SeekFrom::Start(offset))?;
    let mut b = [0; 1024];
    f.read_exact(&mut b)?;
    Header::parse(&b)
}
fn main() -> Result<()> {
    let a = std::env::args().collect::<Vec<_>>();
    ensure!(a.len() == 4, "survey L0_ROOT BASELINE_RAW OUTPUT_H5");
    let mut base = std::fs::File::open(&a[2])?;
    let expected = key(&header(&mut base, 0)?);
    let mut days = vec![];
    for year in std::fs::read_dir(&a[1])? {
        let year = year?.path();
        if !year
            .file_name()
            .unwrap()
            .to_string_lossy()
            .chars()
            .all(|c| c.is_ascii_digit())
            || !year.is_dir()
        {
            continue;
        }
        for day in std::fs::read_dir(year)? {
            let day = day?.path();
            let name = day.file_name().unwrap().to_string_lossy();
            if day.is_dir() && name.len() == 8 && name.chars().all(|c| c.is_ascii_digit()) {
                days.push(day);
            }
        }
    }
    days.sort();
    let rows=days.par_iter().map(|day|->Result<serde_json::Value>{let files=campaign::files(day,".dat")?;let mut matching=vec![];let mut nonmatching=0;let mut errors=vec![];let mut beams=BTreeSet::new();let mut bytes=0u64;
  for path in files.iter(){let n=std::fs::metadata(path)?.len();let test=(||->Result<bool>{let mut f=std::fs::File::open(path)?;let first=header(&mut f,0)?;if key(&first)!=expected{return Ok(false);}let frame=1024+first.samples as u64*8;let frames=n/frame;ensure!(frames>0,"no complete frame");for i in [0,frames/2,frames-1]{let h=header(&mut f,i*frame)?;ensure!(key(&h)==expected,"mode change at sampled header");beams.insert((h.azimuth_raw,h.elevation_raw));}Ok(true)})();
   match test{Ok(true)=>{bytes+=n;matching.push(path.to_string_lossy().to_string());},Ok(false)=>nonmatching+=1,Err(e)=>errors.push(serde_json::json!({"file":path,"error":e.to_string()}))}
  }
  eprintln!("SURVEY {} candidate_files={} bytes={}",day.display(),matching.len(),bytes);
  Ok(serde_json::json!({"acquisition_day":day.file_name().unwrap().to_string_lossy(),"total_files":files.len(),"candidate_files":matching,"candidate_bytes":bytes,"nonmatching_files":nonmatching,"sampled_beams_raw":beams,"errors":errors}))
 }).collect::<Result<Vec<_>>>()?;
    campaign::save(
        Path::new(&a[3]),
        &serde_json::json!({"baseline":expected,"rows":rows,"method":"Primary-site numeric year/day directories; first/middle/last fixed-frame headers per file. Candidate only: internal mode changes, corruption and duplicate files need full native validation; acquisition folder date is not UTC coverage. Bytes are not exposure or validated unique bytes."}),
    )?;
    Ok(())
}
