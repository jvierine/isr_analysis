use anyhow::{ensure, Result};
use num_complex::Complex64 as C;
use rayon::prelude::*;
use std::{fs::File, io::BufReader};
use syisr_long_pulse::{gate, kernel, reader::read_frame, LAGS, NU, PULSE};
const NS: usize = 3782;

fn ds(f: &hdf5::File, name: &str, shape: &[usize], v: &[f64]) -> Result<()> {
    f.new_dataset::<f64>()
        .shape(shape)
        .create(name)?
        .write_raw(v)?;
    Ok(())
}
fn main() -> Result<()> {
    let a: Vec<_> = std::env::args().collect();
    ensure!(
        (3..=4).contains(&a.len()),
        "process input.dat output.h5 [--screen-bright]"
    );
    let screened = a.get(3).map(String::as_str) == Some("--screen-bright");
    ensure!(a.len() == 3 || screened, "unknown option");
    let mut r = BufReader::with_capacity(4 * 1024 * 1024, File::open(&a[1])?);
    let mut frames = Vec::new();
    let mut times = Vec::new();
    let mut first = None;
    while let Some(f) = read_frame(&mut r)? {
        let h = &f.header;
        ensure!(
            h.samples == NS
                && h.sample_rate_hz == 100000.
                && h.waveform == 4
                && h.pulse_width_us == 2000
                && h.ipp_us == 40000
                && h.gate_front_us == 2001,
            "unsupported layout: {h:?}"
        );
        ensure!(
            h.elevation_raw == 16384 && h.tx_elevation_raw == 16384,
            "not zenith"
        );
        if let Some(ref g) = first {
            let g: &syisr_long_pulse::reader::Header = g;
            ensure!(
                h.azimuth_raw == g.azimuth_raw
                    && h.radar_frequency_hz == g.radar_frequency_hz
                    && h.bandwidth_code == g.bandwidth_code,
                "configuration changed"
            );
        } else {
            first = Some(h.clone());
        }
        times.push(h.header_clock_unix_s - 28800.);
        frames.push(f.iq);
    }
    ensure!(
        frames.len() >= 500,
        "expected at least 500 pulses, got {}",
        frames.len()
    );
    ensure!(
        // Three headers differ by one 25-us clock tick; there are no lost IPPs.
        times.windows(2).all(|t| (t[1] - t[0] - 0.04).abs() < 26e-6),
        "clock gap"
    );
    let nb = frames.len().div_ceil(250);
    eprintln!(
        "Read {} pulses; accumulating {} lags in {} blocks",
        frames.len(),
        LAGS,
        nb
    );
    let (initial_keep, ratio) = if screened {
        syisr_long_pulse::screen::bright_mask(&frames, 60, 1660, 16, 64.)
    } else {
        (vec![true; frames.len()], vec![0.; frames.len()])
    };
    let audit = screened
        .then(|| syisr_long_pulse::iqr_screen::screen(&frames, syisr_long_pulse::iqr_screen::END));
    let mut keep: Vec<bool> = initial_keep
        .iter()
        .enumerate()
        .map(|(i, &v)| v && audit.as_ref().map(|a| a.keep[i]).unwrap_or(true))
        .collect();
    // Satellite tracks can remove a complete ten-second block. Exclude that
    // block rather than fail an otherwise usable half-hour integration.
    for block in keep.chunks_mut(250) {
        if block.iter().filter(|&&v| v).count() < 25 {
            block.fill(false);
        }
    }
    let counts: Vec<f64> = keep
        .chunks(250)
        .map(|b| b.iter().filter(|&&v| v).count() as f64)
        .collect();
    let effective_nb = counts.iter().filter(|&&n| n > 0.).count();
    ensure!(
        effective_nb >= 3,
        "fewer than three usable error blocks after satellite rejection"
    );
    let retained: f64 = counts.iter().sum();
    let weights: Vec<f64> = counts.iter().map(|n| n / retained).collect();
    eprintln!("Retained {retained} pulses of {}", frames.len());
    let block: Vec<Vec<C>> = frames
        .par_chunks(250)
        .enumerate()
        .map(|(b, pulses)| {
            if counts[b] == 0. {
                return vec![C::new(f64::NAN, f64::NAN); LAGS * NS];
            }
            let mut sums = vec![C::new(0., 0.); LAGS * NS];
            for (q, p) in pulses.iter().enumerate() {
                if !keep[b * 250 + q] {
                    continue;
                }
                for l in 0..LAGS {
                    let row = &mut sums[l * NS..(l + 1) * NS];
                    for k in 0..NS - l {
                        let x = p[k];
                        let y = p[k + l];
                        row[k].re += x.re as f64 * y.re as f64 + x.im as f64 * y.im as f64;
                        row[k].im += x.im as f64 * y.re as f64 - x.re as f64 * y.im as f64;
                    }
                }
            }
            sums.iter_mut().for_each(|v| *v /= counts[b]);
            sums
        })
        .collect();
    let indices: Vec<usize> = (LAGS - 1..NS - (LAGS - 1) - NU).step_by(5).collect();
    let nr = indices.len();
    // Last complete common-support gate, with exactly the same summation rule.
    let reference_i = NS - (LAGS - 1) - NU;
    let alt_i = reference_i - 250;
    let mut backgrounds = vec![C::new(f64::NAN, f64::NAN); nb * LAGS];
    let mut alternate = backgrounds.clone();
    let mut gated = vec![C::new(f64::NAN, f64::NAN); nb * nr * LAGS];
    let mut raw = vec![C::new(0., 0.); nr * LAGS];
    let mut native = vec![C::new(0., 0.); LAGS * NS];
    for b in 0..nb {
        if counts[b] == 0. {
            continue;
        }
        for l in 0..LAGS {
            let profile = &block[b][l * NS..(l + 1) * NS];
            let bg: C =
                profile[reference_i - l..reference_i + NU].iter().sum::<C>() / (l + NU) as f64;
            backgrounds[b * LAGS + l] = bg;
            alternate[b * LAGS + l] =
                profile[alt_i - l..alt_i + NU].iter().sum::<C>() / (l + NU) as f64;
            for k in 0..NS - l {
                native[l * NS + k] += profile[k] * weights[b];
            }
            for (ri, &i) in indices.iter().enumerate() {
                gated[(b * nr + ri) * LAGS + l] = gate(profile, i, l, bg);
                raw[ri * LAGS + l] += gate(profile, i, l, C::new(0., 0.)) * weights[b];
            }
        }
    }
    // No synthetic zero-valued measurements beyond the available pair indices.
    for l in 0..LAGS {
        for k in NS - l..NS {
            native[l * NS + k] = C::new(f64::NAN, f64::NAN);
        }
    }
    let mut mean = vec![C::new(0., 0.); nr * LAGS];
    let mut semr = vec![0.; nr * LAGS];
    let mut semi = semr.clone();
    for j in 0..nr * LAGS {
        mean[j] = (0..nb)
            .filter(|&b| weights[b] > 0.)
            .map(|b| gated[b * nr * LAGS + j] * weights[b])
            .sum::<C>();
        let leave: Vec<C> = (0..nb)
            .filter(|&b| weights[b] > 0.)
            .map(|b| (mean[j] - gated[b * nr * LAGS + j] * weights[b]) / (1. - weights[b]))
            .collect();
        let center = leave.iter().sum::<C>() / effective_nb as f64;
        semr[j] = (leave
            .iter()
            .map(|v| (v.re - center.re).powi(2))
            .sum::<f64>()
            * (effective_nb - 1) as f64
            / effective_nb as f64)
            .sqrt();
        semi[j] = (leave
            .iter()
            .map(|v| (v.im - center.im).powi(2))
            .sum::<f64>()
            * (effective_nb - 1) as f64
            / effective_nb as f64)
            .sqrt();
    }
    let normalized: Vec<f64> = (0..nr * LAGS)
        .map(|j| {
            let z = mean[j / LAGS * LAGS].re;
            if z != 0. {
                mean[j].re / z
            } else {
                f64::NAN
            }
        })
        .collect();
    let range: Vec<f64> = indices
        .iter()
        .map(|&i| (2001. + (i as f64 + (NU as f64 - PULSE as f64) / 2.) * 10.) * 0.149896229)
        .collect();
    let f = hdf5::File::create(&a[2])?;
    ds(&f, "time_utc_s", &[times.len()], &times)?;
    ds(
        &f,
        "pulse_keep",
        &[keep.len()],
        &keep
            .iter()
            .map(|&v| if v { 1. } else { 0. })
            .collect::<Vec<_>>(),
    )?;
    ds(&f, "pulse_max_guard_power_ratio", &[ratio.len()], &ratio)?;
    ds(&f, "block_retained_pulses", &[nb], &counts)?;
    ds(
        &f,
        "initial_pulse_keep",
        &[initial_keep.len()],
        &initial_keep
            .iter()
            .map(|&v| if v { 1. } else { 0. })
            .collect::<Vec<_>>(),
    )?;
    if let Some(a) = audit {
        for (name, v, shape) in [
            ("iqr_spectral_score", a.score, vec![a.nt, a.nw]),
            ("iqr_peak_frequency_hz", a.peak_hz, vec![a.nt, a.nw]),
            (
                "iqr_narrow_excess_fraction",
                a.narrow_fraction,
                vec![a.nt, a.nw],
            ),
            ("iqr_spectral_candidate", a.candidate, vec![a.nt, a.nw]),
            ("iqr_power_score", a.power_score, vec![a.nt, a.nw]),
            ("iqr_spectrum_median_adu2", a.median, vec![a.nw, 128]),
            ("iqr_spectrum_iqr_adu2", a.iqr, vec![a.nw, 128]),
            ("iqr_spectrum_q3_adu2", a.q3, vec![a.nw, 128]),
        ] {
            ds(
                &f,
                name,
                &shape,
                &v.iter().map(|&v| v as f64).collect::<Vec<_>>(),
            )?;
        }
    }
    ds(&f, "slant_range_km", &[nr], &range)?;
    ds(
        &f,
        "lag_us",
        &[LAGS],
        &(0..LAGS).map(|i| i as f64 * 10.).collect::<Vec<_>>(),
    )?;
    for (name, v, shape) in [
        ("acf", &mean, vec![nr, LAGS]),
        ("acf_before_background", &raw, vec![nr, LAGS]),
        ("block_acf", &gated, vec![nb, nr, LAGS]),
        ("background_acf", &backgrounds, vec![nb, LAGS]),
        ("alternate_background_acf", &alternate, vec![nb, LAGS]),
        ("native_lpm", &native, vec![LAGS, NS]),
    ] {
        ds(
            &f,
            &format!("{name}_real_adu2"),
            &shape,
            &v.iter().map(|v| v.re).collect::<Vec<_>>(),
        )?;
        ds(
            &f,
            &format!("{name}_imag_adu2"),
            &shape,
            &v.iter().map(|v| v.im).collect::<Vec<_>>(),
        )?;
    }
    ds(&f, "acf_real_sem_adu2", &[nr, LAGS], &semr)?;
    ds(&f, "acf_imag_sem_adu2", &[nr, LAGS], &semi)?;
    ds(&f, "normalized_acf_real", &[nr, LAGS], &normalized)?;
    ds(
        &f,
        "range_ambiguity_weights",
        &[LAGS, PULSE + NU - 1],
        &(0..LAGS).flat_map(kernel).collect::<Vec<_>>(),
    )?;
    ds(
        &f,
        "range_ambiguity_offset_km",
        &[PULSE + NU - 1],
        &(0..PULSE + NU - 1)
            .map(|i| (i as f64 - 109.5) * 1.49896229)
            .collect::<Vec<_>>(),
    )?;
    let bg_range =
        (2001. + (reference_i as f64 + (NU as f64 - PULSE as f64) / 2.) * 10.) * 0.149896229;
    let meta = serde_json::json!({"reader_reference":"readDataFromRawEcho0_ts.m", "implementation":"experiments/ulp2ms", "header":first,"header_clock_convention":"China civil time UTC+8, explicitly confirmed by data owner","selected_utc_start_s":times[0],"selected_utc_end_exclusive_s":times[times.len()-1]+0.04,"integration_seconds":frames.len() as f64*0.04,"range_definition":"monostatic slant range; pulse-corrected centroid; nominal rectangular transmit pulse; no measured receiver group delay correction","acf_convention":"x[k] conj(x[k+lag]); exp(+i 2pi f t) has negative ACF phase; physical velocity sign unverified","range_summation":"trapezoidal: late sample j=i..i+lag+nu-1; nu=21; divide by number of summed pairs then by fractional pulse overlap (200-lag)/200","ambiguity":"all lags have common centre and 329.7717 km extent; kernel SHAPES remain lag-dependent and are saved; receiver filter tails not included in rectangular kernels","background_reference_index":reference_i,"background_reference_slant_range_km":bg_range,"background_removal":"complex background from last complete gate, each 10-second block and lag, subtracted at every range BEFORE overlap correction and normalization; receiver signal transfer function not deconvolved","normalization":"real noise-subtracted ACF divided by real noise-subtracted zero lag at each range; signed values retained","error":"delete-one-10-second-block jackknife SEM; unequal retained-pulse counts weighted; not a full covariance or guaranteed independent samples","pulse_count":frames.len(),"outlier_rejection":screened,"retained_pulse_count":retained,"screen":"optional whole-IPP rejection: any 16-sample guard average >64 times same-range median over selected integration, samples 60..1660; or >12 times median and >0.92 single-tone power fraction across 48 samples for three successive pulses; plus/minus 2 pulse margin; candidates not orbital identifications","iqr_screen":"Hann 128 samples / hop 32, 8-pulse spectra; each range-frequency bin uses temporal Q3+6 IQR; positive excess concentrated >0.6 in three of eleven bins; neighboring time/range support within one Doppler bin; one time-block margin. Broad-power Q3+6 IQR branch retained.","absolute_calibration":false,"assumed_future_F_peak_ne_m3":1e12,"ion_fraction_fit_performed":false});
    f.new_attr_builder()
        .with_data(meta.to_string().as_str())
        .create("metadata_json")?;
    eprintln!(
        "Saved {nr} ranges, background centroid {bg_range:.2} km, UTC start {}",
        times[0]
    );
    Ok(())
}
