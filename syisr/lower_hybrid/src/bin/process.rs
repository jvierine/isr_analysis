//! Read-only mmap of the indexed recording; independent per-beam wall windows.
use anyhow::{ensure,Result};
use hdf5::{File,types::VarLenUnicode};
use memmap2::MmapOptions;
use num_complex::Complex32;
use rayon::prelude::*;
use std::{path::Path,str::FromStr,sync::Arc};
use syisr_lower_hybrid::*;

#[path="../../../src/reader.rs"]
mod reader;

fn attr(g:&hdf5::Group,name:&str,text:&str)->Result<()> {
    g.new_attr::<VarLenUnicode>().create(name)?.write_scalar(&VarLenUnicode::from_str(text)?)?;Ok(())
}

struct Spectrum {beam:usize,frame:usize,count:u32,first:f64,last:f64,psd:Vec<f64>,unscreened:Vec<f64>,mask:Vec<u8>,accepted:Vec<u32>,seconds_count:Vec<u32>}

fn main()->Result<()> {
    let a:Vec<_>=std::env::args().collect();ensure!(a.len()==3,"process INDEX.h5 OUTPUT.h5");
    rayon::ThreadPoolBuilder::new().num_threads(4).build_global()?;
    let index=File::open(&a[1])?;
    let paths:Vec<String>=if index.link_exists("compute_cache_paths") {index.dataset("compute_cache_paths")?.read_raw::<VarLenUnicode>()?.iter().map(|v|v.as_str().to_string()).collect()}else if index.link_exists("source_paths"){index.dataset("source_paths")?.read_raw::<VarLenUnicode>()?.iter().map(|v|v.as_str().to_string()).collect()}else{vec![index.attr("raw_path")?.read_scalar::<VarLenUnicode>()?.as_str().to_string()]};
    let sources:Vec<_>=paths.iter().map(std::fs::File::open).collect::<std::io::Result<_>>()?;
    let maps:Vec<_>=sources.iter().map(|source|unsafe{MmapOptions::new().map(source)}).collect::<std::io::Result<_>>()?;
    let stride=index.attr("record_stride_bytes")?.read_scalar::<u64>()? as usize;
    let samples=index.attr("sample_count")?.read_scalar::<u64>()? as usize;
    let expected=index.attr("raw_bytes")?.read_scalar::<u64>()?;
    ensure!(maps.iter().map(|m|m.len() as u64).sum::<u64>()==expected,"raw file sizes changed");
    let time=index.dataset("time_utc_s")?.read_raw::<f64>()?;
    let beam=index.dataset("beam_index")?.read_raw::<u8>()?;
    let raw_angles=index.dataset("beam_raw_angles")?.read_raw::<u16>()?;
    ensure!(time.len()==beam.len() && maps.iter().map(|m|m.len()/stride).sum::<usize>()==time.len(),"index/recording mismatch");
    let file_index=if index.link_exists("source_file_index"){index.dataset("source_file_index")?.read_raw::<u16>()?}else{vec![0;time.len()]};
    let record_index=if index.link_exists("source_record_index"){index.dataset("source_record_index")?.read_raw::<u64>()?}else{(0..time.len() as u64).collect()};
    let mut starts=Vec::new();let t0=time[0];let end=time[time.len()-1]+0.01;
    let window=300_f64.min(end-t0);ensure!(window>=1.,"recording too short");
    let mut s=t0;
    while s+window<=end {starts.push(s);s+=150.;}
    let last=end-window;
    if last-*starts.last().unwrap()>0.001 {starts.push(last);}
    let nframe=starts.len();let nsec=window.ceil() as usize;
    let range_starts:Vec<_>=(0..=samples-FAST_SAMPLES).step_by(RANGE_HOP).collect();
    let nr=range_starts.len();
    let ranges:Vec<_>=range_starts.iter().map(|j| {
        (533.+(*j as f64+(FAST_SAMPLES-1) as f64/2.)*10.-240.)*0.149896229
    }).collect();
    let frequency:Vec<_>=(0..FFT_BINS).map(|i|(i as f64-FFT_BINS as f64/2.)*SAMPLE_HZ/FFT_BINS as f64).collect();
    let w=hann();let mut planner=rustfft::FftPlanner::new();
    let fft:Arc<dyn rustfft::Fft<f32>>=planner.plan_fft_forward(FFT_BINS);
    let jobs:Vec<_>=(0..5).flat_map(|b|(0..nframe).map(move|f|(b,f))).collect();
    let results:Result<Vec<_>>=jobs.par_iter().map(|&(b,frame)|->Result<_>{
        let start=starts[frame];let stop=start+window;
        let mut seconds=vec![0.;nsec*nr*FFT_BINS];let mut seconds_count=vec![0_u32;nsec];
        let mut buffer=vec![Complex32::new(0.,0.);FFT_BINS];
        let mut scratch=vec![Complex32::new(0.,0.);fft.get_inplace_scratch_len()];
        let mut iq=vec![Complex32::new(0.,0.);samples];
        let mut count=0_u32;let mut first=f64::NAN;let mut last=f64::NAN;
        let lo=time.partition_point(|&t|t<start);let hi=time.partition_point(|&t|t<stop);
        for p in lo..hi {
            if beam[p] as usize!=b {continue;}
            let mmap=&maps[file_index[p] as usize];
            let off=record_index[p] as usize*stride;
            let h=reader::Header::parse(&mmap[off..off+1024])?;
            ensure!(h.waveform==4 && h.pulse_width_us==480 && h.sample_rate_hz==SAMPLE_HZ
                    && h.samples==samples && h.ipp_us==10000 && h.gate_front_us==533,"configuration changed");
            ensure!(h.azimuth_raw==raw_angles[2*b] && h.elevation_raw==raw_angles[2*b+1],"beam mismatch");
            for (j,z) in iq.iter_mut().enumerate() {
                let v=&mmap[off+1024+8*j..off+1024+8*j+8];
                let q=i32::from_le_bytes(v[..4].try_into().unwrap());
                let i=i32::from_le_bytes(v[4..].try_into().unwrap());
                *z=Complex32::new(i as f32,q as f32);
            }
            let sec=((time[p]-start).floor() as usize).min(nsec-1);seconds_count[sec]+=1;
            for (r,&gate) in range_starts.iter().enumerate() {
                add_periodogram(&iq[gate..gate+FAST_SAMPLES],&w,fft.as_ref(),&mut buffer,&mut scratch,
                                &mut seconds[(sec*nr+r)*FFT_BINS..(sec*nr+r+1)*FFT_BINS]);
            }
            if count==0 {first=time[p];} last=time[p];count+=1;
        }
        ensure!(count>0,"empty beam/window");
        let mask=transient_mask(&seconds,&seconds_count,nr);
        let mut psd=vec![0.;nr*FFT_BINS];let mut unscreened=psd.clone();let mut accepted=vec![0_u32;nr];
        for sec in 0..nsec {for r in 0..nr {
            if mask[sec*nr+r]==0 {accepted[r]+=seconds_count[sec];}
            for f in 0..FFT_BINS {let value=seconds[(sec*nr+r)*FFT_BINS+f];unscreened[r*FFT_BINS+f]+=value;
                if mask[sec*nr+r]==0 {psd[r*FFT_BINS+f]+=value;}
            }
        }}
        for r in 0..nr {ensure!(accepted[r]>0,"transient screen removed all exposure");
            for f in 0..FFT_BINS {psd[r*FFT_BINS+f]/=accepted[r] as f64;unscreened[r*FFT_BINS+f]/=count as f64;}
        }
        println!("beam {b}, window {frame}, {count} pulses");
        Ok(Spectrum{beam:b,frame,count,first,last,psd,unscreened,mask,accepted,seconds_count})
    }).collect();
    let results=results?;
    let out=File::create(&a[2])?;
    out.new_dataset_builder().with_data(&frequency).create("frequency_hz")?;
    out.new_dataset_builder().with_data(&ranges).create("slant_range_km")?;
    out.new_dataset_builder().with_data(&range_starts.iter().map(|&j|j as u64).collect::<Vec<_>>()).create("range_window_start_sample")?;
    out.new_dataset_builder().with_data(&w).create("fast_window_hann")?;
    out.new_dataset_builder().with_data(&starts).create("window_start_utc_s")?;
    out.new_dataset_builder().with_data(&starts.iter().map(|s|s+window).collect::<Vec<_>>()).create("window_end_utc_s")?;
    for key in ["raw_path","raw_sha256","clock_basis","geometry_model","geometry_epoch_utc","geometry_columns"] {
        attr(&out,key,index.attr(key)?.read_scalar::<VarLenUnicode>()?.as_str())?;
    }
    for name in ["source_paths","source_sha256"] {if index.link_exists(name){let values=index.dataset(name)?.read_raw::<VarLenUnicode>()?;out.new_dataset_builder().with_data(&values).create(name)?;}}
    attr(&out,"frequency_sign","I+iQ; FFT exp(-i2pi f t); positive phase rotation gives positive Doppler; physical velocity sign uncalibrated")?;
    attr(&out,"power_units","ADU^2/Hz; uncalibrated receiver units; raw noise retained; no noise subtraction or satellite screening")?;
    attr(&out,"normalization","divide each averaged range spectrum by sum over all 256 frequency bins times df; integral = 1 per gate")?;
    attr(&out,"range_reference","pulse-centred monostatic slant range: c/2*(gate_front + Hann centroid - pulse_width/2); receiver group delay uncalibrated")?;
    attr(&out,"range_response","480 us rectangular pulse convolved with 480 us Hann-squared receiver window; nominal full support ~144 km, ~72 km FWHM; overlapping 18 km gate spacing")?;
    attr(&out,"time_averaging","300 s wall-clock windows, 150 s steps; final full window anchored at recording end; interleaved beams averaged separately; shorter recording uses its complete available duration")?;
    attr(&out,"transient_screen","one-second same-beam PSD / same-window temporal median > 8; connected excess component >=3 bins, <=16 range gates; positive-excess cosine similarity >=0.7 to calculated rectangular-Tx/Hann-receiver range-Doppler ambiguity power; expand by four range gates and one second; exclude entire affected time/range exposures across all frequencies; candidate satellite morphology, not confirmed object identification")?;
    out.new_attr::<f64>().create("integration_window_s")?.write_scalar(&window)?;
    out.new_attr::<f64>().create("integration_step_s")?.write_scalar(&150.)?;
    attr(&out,"spectral_resolution","48 measured samples, 480 us duration, symmetric Hann; 256-point zero padding gives 390.625 Hz display spacing, not physical resolution; ENBW ~3.19 kHz")?;
    out.new_attr::<f64>().create("sample_rate_hz")?.write_scalar(&SAMPLE_HZ)?;
    out.new_attr::<f64>().create("frequency_bin_hz")?.write_scalar(&(SAMPLE_HZ/FFT_BINS as f64))?;
    let geom=index.dataset("beam_geometry")?.read_raw::<f64>()?;
    out.new_dataset::<f64>().shape((5,7)).create("beam_geometry")?.write_raw(&geom)?;
    for b in 0..5 {
        let g=out.create_group(&format!("beam_{b}"))?;
        let mut raw=vec![0.;nframe*nr*FFT_BINS];let mut norm=raw.clone();let mut power=vec![0.;nframe*nr];
        let mut unscreened=raw.clone();let mut masks=vec![0_u8;nframe*nsec*nr];let mut accepted=vec![0_u32;nframe*nr];let mut sec_counts=vec![0_u32;nframe*nsec];
        let mut counts=vec![0_u32;nframe];let mut first=vec![0.;nframe];let mut last=first.clone();
        for r in results.iter().filter(|r|r.beam==b) {
            let offset=r.frame*nr*FFT_BINS;raw[offset..offset+r.psd.len()].copy_from_slice(&r.psd);
            unscreened[offset..offset+r.psd.len()].copy_from_slice(&r.unscreened);
            masks[r.frame*nsec*nr..(r.frame+1)*nsec*nr].copy_from_slice(&r.mask);
            accepted[r.frame*nr..(r.frame+1)*nr].copy_from_slice(&r.accepted);
            sec_counts[r.frame*nsec..(r.frame+1)*nsec].copy_from_slice(&r.seconds_count);
            for j in 0..nr {
                let (v,area)=normalized(&r.psd[j*FFT_BINS..(j+1)*FFT_BINS])?;
                norm[offset+j*FFT_BINS..offset+(j+1)*FFT_BINS].copy_from_slice(&v);
                power[r.frame*nr+j]=area;
            }
            counts[r.frame]=r.count;first[r.frame]=r.first;last[r.frame]=r.last;
        }
        g.new_dataset::<f64>().shape((nframe,nr,FFT_BINS)).chunk((1,nr,FFT_BINS)).deflate(3).create("unscreened_psd_adu2_per_hz")?.write_raw(&unscreened)?;
        g.new_dataset::<u8>().shape((nframe,nsec,nr)).chunk((1,nsec,nr)).deflate(3).create("transient_rejected_time_range")?.write_raw(&masks)?;
        g.new_dataset::<u32>().shape((nframe,nr)).create("accepted_pulse_count")?.write_raw(&accepted)?;
        g.new_dataset::<u32>().shape((nframe,nsec)).create("one_second_pulse_count")?.write_raw(&sec_counts)?;
        g.new_dataset::<f64>().shape((nframe,nr,FFT_BINS)).chunk((1,nr,FFT_BINS)).deflate(3).create("psd_adu2_per_hz")?.write_raw(&raw)?;
        g.new_dataset::<f64>().shape((nframe,nr,FFT_BINS)).chunk((1,nr,FFT_BINS)).deflate(3).create("normalized_psd_per_hz")?.write_raw(&norm)?;
        g.new_dataset::<f64>().shape((nframe,nr)).create("integrated_power_adu2")?.write_raw(&power)?;
        g.new_dataset_builder().with_data(&counts).create("pulse_count")?;
        g.new_dataset_builder().with_data(&first).create("first_pulse_utc_s")?;
        g.new_dataset_builder().with_data(&last).create("last_pulse_utc_s")?;
        g.new_dataset_builder().with_data(&counts.iter().map(|&x|x as f64*0.01).collect::<Vec<_>>()).create("scheduled_beam_time_s")?;
    }
    println!("Saved {}: {nframe} windows per beam, {nr} range gates",Path::new(&a[2]).display());
    Ok(())
}
