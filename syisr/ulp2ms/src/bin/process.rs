use anyhow::{ensure, Result};
use num_complex::Complex64 as C;
use rayon::prelude::*;
use std::io::BufReader;
use syisr_long_pulse::{gate, kernel, reader::read_frame, LAGS, NU, PULSE};
#[cfg(test)]
const NS: usize = 3782;

fn ds(f: &hdf5::File, name: &str, shape: &[usize], v: &[f64]) -> Result<()> {
    f.new_dataset::<f64>()
        .shape(shape)
        .create(name)?
        .write_raw(v)?;
    Ok(())
}

#[cfg(test)]
mod partial_tests {
    use super::*;
    #[test]
    fn exported_partial_products_match_direct_stable_pair_sums() {
        let counts = [2., 3., 4.];
        let voltage = |b: usize, k: usize| {
            C::new(
                ((b * 7 + k) as f64 * 0.13).sin(),
                ((b + k * 3) as f64 * 0.07).cos(),
            )
        };
        let block: Vec<Vec<C>> = (0..3)
            .map(|b| {
                (0..61 * NS)
                    .map(|j| {
                        let l = j / NS;
                        let k = j % NS;
                        voltage(b, k) * voltage(b, k + l).conj()
                    })
                    .collect()
            })
            .collect();
        let background = vec![C::new(0.1, -0.02); 3 * 61];
        let path =
            std::env::temp_dir().join(format!("syisr-partial-native-{}.h5", std::process::id()));
        partial_product(
            path.to_str().unwrap(),
            &block,
            &counts,
            &background,
            NS,
            2001.,
        )
        .unwrap();
        let h = hdf5::File::open(&path).unwrap();
        let re = h
            .dataset("block_acf_real_adu2")
            .unwrap()
            .read_raw::<f64>()
            .unwrap();
        let im = h
            .dataset("block_acf_imag_adu2")
            .unwrap()
            .read_raw::<f64>()
            .unwrap();
        for b in 0..3 {
            for g in [0, 7, 13] {
                let i: usize = 20 + 5 * g;
                for l in [0, 1, 30, 60] {
                    let lo = i.saturating_sub(l).max(30);
                    let hi = i + 21;
                    let expected = ((lo..hi)
                        .map(|k| voltage(b, k) * voltage(b, k + l).conj())
                        .sum::<C>()
                        / (hi - lo) as f64
                        - background[b * 61 + l])
                        * 200.
                        / (200 - l) as f64;
                    let j = (b * 14 + g) * 61 + l;
                    assert!((C::new(re[j], im[j]) - expected).norm() < 1e-14);
                }
            }
        }
        drop(h);
        std::fs::remove_file(path).unwrap();
    }
}
fn centered_product(sum: C, n: f64, a: C, b: C) -> C {
    (sum - n * a * b.conj()) / (n - 1.)
}
#[cfg(test)]
mod centering_tests {
    use super::*;
    #[test]
    fn complex_mean_removal_matches_explicit_centering_and_is_shift_invariant() {
        let a: Vec<C> = (0..29)
            .map(|j| C::new((j as f64 * 0.3).sin(), (j as f64 * 0.7).cos()))
            .collect();
        let b: Vec<C> = (0..29)
            .map(|j| C::new((j as f64 * 0.5).cos(), (j as f64 * 0.2).sin()))
            .collect();
        let n = a.len() as f64;
        let ma: C = a.iter().sum::<C>() / n;
        let mb: C = b.iter().sum::<C>() / n;
        let expected: C = a
            .iter()
            .zip(&b)
            .map(|(a, b)| (*a - ma) * (*b - mb).conj())
            .sum::<C>()
            / (n - 1.);
        for shift in [C::default(), C::new(100., -250.)] {
            let sum: C = a
                .iter()
                .zip(&b)
                .map(|(a, b)| (*a + shift) * (*b - shift).conj())
                .sum();
            assert!((centered_product(sum, n, ma + shift, mb - shift) - expected).norm() < 1e-9);
        }
    }
}
fn partial_product(
    path: &str,
    block: &[Vec<C>],
    counts: &[f64],
    background: &[C],
    ns: usize,
    gate_front_us: f64,
) -> Result<()> {
    let indices: Vec<usize> = (20..90).step_by(5).collect();
    let nr = indices.len();
    let nb = counts.len();
    let total: f64 = counts.iter().sum();
    let good: Vec<usize> = (0..nb).filter(|&b| counts[b] > 0.).collect();
    let mut values = vec![C::new(f64::NAN, f64::NAN); nb * nr * 61];
    for &b in &good {
        for (g, &i) in indices.iter().enumerate() {
            for l in 0..61 {
                let lo = i.saturating_sub(l).max(30);
                let hi = i + 21;
                let mean: C =
                    block[b][l * ns + lo..l * ns + hi].iter().sum::<C>() / (hi - lo) as f64;
                values[(b * nr + g) * 61 + l] =
                    (mean - background[b * 61 + l]) * 200. / (200 - l) as f64;
            }
        }
    }
    let mut mean = vec![];
    let mut sr = vec![];
    let mut si = vec![];
    for j in 0..nr * 61 {
        let y: C = good
            .iter()
            .map(|&b| values[b * nr * 61 + j] * counts[b] / total)
            .sum();
        let leave: Vec<C> = good
            .iter()
            .map(|&b| {
                let w = counts[b] / total;
                (y - values[b * nr * 61 + j] * w) / (1. - w)
            })
            .collect();
        let center: C = leave.iter().sum::<C>() / good.len() as f64;
        let factor = (good.len() - 1) as f64 / good.len() as f64;
        sr.push(
            (factor
                * leave
                    .iter()
                    .map(|v| (v.re - center.re).powi(2))
                    .sum::<f64>())
            .sqrt(),
        );
        si.push(
            (factor
                * leave
                    .iter()
                    .map(|v| (v.im - center.im).powi(2))
                    .sum::<f64>())
            .sqrt(),
        );
        mean.push(y);
    }
    let f = hdf5::File::create(path)?;
    for (name, data, shape) in [
        (
            "acf_real_adu2",
            mean.iter().map(|v| v.re).collect::<Vec<_>>(),
            vec![nr, 61],
        ),
        (
            "acf_imag_adu2",
            mean.iter().map(|v| v.im).collect(),
            vec![nr, 61],
        ),
        ("acf_real_sem_adu2", sr, vec![nr, 61]),
        ("acf_imag_sem_adu2", si, vec![nr, 61]),
        (
            "block_acf_real_adu2",
            values.iter().map(|v| v.re).collect(),
            vec![nb, nr, 61],
        ),
        (
            "block_acf_imag_adu2",
            values.iter().map(|v| v.im).collect(),
            vec![nb, nr, 61],
        ),
    ] {
        ds(&f, name, &shape, &data)?;
    }
    f.new_dataset_builder()
        .with_data(
            &indices
                .iter()
                .map(|&i| (gate_front_us + (i as f64 - 89.5) * 10.) * 0.149896229)
                .collect::<Vec<_>>(),
        )
        .create("slant_range_km")?;
    f.new_dataset_builder()
        .with_data(&indices.iter().map(|&i| i as u64).collect::<Vec<_>>())
        .create("native_gate_index")?;
    f.new_attr::<u64>()
        .create("first_stable_sample")?
        .write_scalar(&30)?;
    f.new_attr_builder().with_data("Native block products; stable early samples >=30; actual pair-count normalization; identical full-recording pulse mask and background to main product").create("estimator")?;
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
    let mut r = BufReader::with_capacity(4 * 1024 * 1024, syisr_long_pulse::reader::input(&a[1])?);
    let probe = read_frame(&mut r)?.ok_or_else(|| anyhow::anyhow!("empty raw file"))?;
    let layout = syisr_long_pulse::reader::LongPulseLayout::from_header(&probe.header)?;
    drop(r);
    let ns = layout.samples;
    let ipp = layout.ipp_s;
    let block_pulses = (10. / ipp).round() as usize;
    let screen_end: usize = std::env::var("SYISR_SCREEN_END_SAMPLE")
        .ok()
        .map(|v| v.parse())
        .transpose()?
        .unwrap_or(ns);
    ensure!(
        (256..=ns).contains(&screen_end),
        "unsupported screening extent"
    );
    let mut r = BufReader::with_capacity(4 * 1024 * 1024, syisr_long_pulse::reader::input(&a[1])?);
    let mut frames = Vec::new();
    let mut times = Vec::new();
    let mut first = None;
    while let Some(f) = read_frame(&mut r)? {
        let h = &f.header;
        ensure!(
            syisr_long_pulse::reader::LongPulseLayout::from_header(h)? == layout,
            "configuration changed"
        );
        ensure!(
            h.elevation_raw == 16384
                && (h.tx_elevation_raw == 16384 || (h.samples == 1194 && h.tx_elevation_raw == 0)),
            "not zenith"
        );
        if let Some(ref g) = first {
            let g: &syisr_long_pulse::reader::Header = g;
            ensure!(
                h.radar_frequency_hz == g.radar_frequency_hz
                    && h.bandwidth_code == g.bandwidth_code,
                "configuration changed"
            );
        } else {
            first = Some(h.clone());
        }
        times.push(h.header_clock_unix_s - 28800.);
        frames.push(f.iq);
    }
    let original_header_times = times.clone();
    if ns == 1194 && a[1].ends_with(".json") {
        use std::io::{Read, Seek, SeekFrom};
        let selection: serde_json::Value = serde_json::from_slice(&std::fs::read(&a[1])?)?;
        let mut verified = vec![];
        for segment in selection["segments"].as_array().unwrap() {
            let mut file = std::fs::File::open(segment["path"].as_str().unwrap())?;
            file.seek(SeekFrom::Start(
                segment["byte_offset"].as_u64().unwrap_or(0),
            ))?;
            let mut bytes = [0u8; 1024];
            file.read_exact(&mut bytes)?;
            let start =
                syisr_long_pulse::reader::Header::parse(&bytes)?.header_clock_unix_s - 28800.;
            for j in
                segment["first_frame"].as_u64().unwrap()..segment["stop_frame"].as_u64().unwrap()
            {
                verified.push(start + j as f64 * ipp);
            }
        }
        ensure!(
            verified.len() == times.len(),
            "indexed pulse timeline mismatch"
        );
        times = verified;
    }
    ensure!(
        frames.len() >= 500,
        "expected at least 500 pulses, got {}",
        frames.len()
    );
    ensure!(
        // Products are formed within each pulse. Short documented acquisition
        // gaps and clock resynchronizations do not require inventing pulses.
        times.windows(2).all(|t| (t[1] - t[0]).abs() < 60.),
        "clock gap"
    );
    let nb = frames.len().div_ceil(block_pulses);
    eprintln!(
        "Read {} pulses; accumulating {} lags in {} blocks",
        frames.len(),
        LAGS,
        nb
    );
    let (initial_keep, ratio) = if screened {
        syisr_long_pulse::screen::bright_mask(&frames, 60, screen_end, 16, 64.)
    } else {
        (vec![true; frames.len()], vec![0.; frames.len()])
    };
    let audit = screened.then(|| syisr_long_pulse::iqr_screen::screen(&frames, screen_end));
    let mut keep: Vec<bool> = initial_keep
        .iter()
        .enumerate()
        .map(|(i, &v)| v && audit.as_ref().map(|a| a.keep[i]).unwrap_or(true))
        .collect();
    // Satellite tracks can remove a complete ten-second block. Exclude that
    // block rather than fail an otherwise usable half-hour integration.
    for block in keep.chunks_mut(block_pulses) {
        if block.iter().filter(|&&v| v).count() < block_pulses / 10 {
            block.fill(false);
        }
    }
    let counts: Vec<f64> = keep
        .chunks(block_pulses)
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
    let remove_stationary = std::env::var("SYISR_REMOVE_STATIONARY_MEAN")
        .ok()
        .map(|s| s == "1")
        .unwrap_or(ns == 1194);
    let stationary: Vec<Vec<C>> = frames
        .par_chunks(block_pulses)
        .enumerate()
        .map(|(b, pulses)| {
            let mut mean = vec![C::default(); ns];
            if counts[b] > 0. {
                for (q, p) in pulses.iter().enumerate() {
                    if keep[b * block_pulses + q] {
                        for (k, v) in p.iter().enumerate() {
                            mean[k] += C::new(v.re as f64, v.im as f64) / counts[b];
                        }
                    }
                }
            }
            mean
        })
        .collect();
    let block: Vec<Vec<C>> = frames
        .par_chunks(block_pulses)
        .enumerate()
        .map(|(b, pulses)| {
            if counts[b] == 0. {
                return vec![C::new(f64::NAN, f64::NAN); LAGS * ns];
            }
            let mut sums = vec![C::new(0., 0.); LAGS * ns];
            for (q, p) in pulses.iter().enumerate() {
                if !keep[b * block_pulses + q] {
                    continue;
                }
                for l in 0..LAGS {
                    let row = &mut sums[l * ns..(l + 1) * ns];
                    for k in 0..ns - l {
                        let x = p[k];
                        let y = p[k + l];
                        row[k].re += x.re as f64 * y.re as f64 + x.im as f64 * y.im as f64;
                        row[k].im += x.im as f64 * y.re as f64 - x.re as f64 * y.im as f64;
                    }
                }
            }
            if remove_stationary {
                for l in 0..LAGS {
                    for k in 0..ns - l {
                        sums[l * ns + k] = centered_product(
                            sums[l * ns + k],
                            counts[b],
                            stationary[b][k],
                            stationary[b][k + l],
                        );
                    }
                }
            } else {
                sums.iter_mut().for_each(|v| *v /= counts[b]);
            }
            sums
        })
        .collect();
    let indices: Vec<usize> = (LAGS - 1..ns - (LAGS - 1) - NU).step_by(5).collect();
    let nr = indices.len();
    // Last complete common-support gate, with exactly the same summation rule.
    let reference_i = ns - (LAGS - 1) - NU;
    let alt_i = reference_i - 250;
    let mut backgrounds = vec![C::new(f64::NAN, f64::NAN); nb * LAGS];
    let mut alternate = backgrounds.clone();
    let mut gated = vec![C::new(f64::NAN, f64::NAN); nb * nr * LAGS];
    let mut raw = vec![C::new(0., 0.); nr * LAGS];
    let mut native = vec![C::new(0., 0.); LAGS * ns];
    for b in 0..nb {
        if counts[b] == 0. {
            continue;
        }
        for l in 0..LAGS {
            let profile = &block[b][l * ns..(l + 1) * ns];
            let bg: C =
                profile[reference_i - l..reference_i + NU].iter().sum::<C>() / (l + NU) as f64;
            backgrounds[b * LAGS + l] = bg;
            alternate[b * LAGS + l] =
                profile[alt_i - l..alt_i + NU].iter().sum::<C>() / (l + NU) as f64;
            for k in 0..ns - l {
                native[l * ns + k] += profile[k] * weights[b];
            }
            for (ri, &i) in indices.iter().enumerate() {
                gated[(b * nr + ri) * LAGS + l] = gate(profile, i, l, bg);
                raw[ri * LAGS + l] += gate(profile, i, l, C::new(0., 0.)) * weights[b];
            }
        }
    }
    if let Ok(path) = std::env::var("SYISR_PARTIAL_OUTPUT") {
        partial_product(
            &path,
            &block,
            &counts,
            &backgrounds,
            ns,
            layout.gate_front_us,
        )?;
    }
    // No synthetic zero-valued measurements beyond the available pair indices.
    for l in 0..LAGS {
        for k in ns - l..ns {
            native[l * ns + k] = C::new(f64::NAN, f64::NAN);
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
        .map(|&i| {
            (layout.gate_front_us + (i as f64 + (NU as f64 - PULSE as f64) / 2.) * 10.)
                * 0.149896229
        })
        .collect();
    let f = hdf5::File::create(&a[2])?;
    ds(&f, "time_utc_s", &[times.len()], &times)?;
    ds(
        &f,
        "original_header_time_utc_s",
        &[times.len()],
        &original_header_times,
    )?;
    if remove_stationary {
        ds(
            &f,
            "removed_stationary_voltage_real",
            &[nb, ns],
            &stationary
                .iter()
                .flatten()
                .map(|v| v.re)
                .collect::<Vec<_>>(),
        )?;
        ds(
            &f,
            "removed_stationary_voltage_imag",
            &[nb, ns],
            &stationary
                .iter()
                .flatten()
                .map(|v| v.im)
                .collect::<Vec<_>>(),
        )?;
    }
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
        ("native_lpm", &native, vec![LAGS, ns]),
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
    let bg_range = (layout.gate_front_us
        + (reference_i as f64 + (NU as f64 - PULSE as f64) / 2.) * 10.)
        * 0.149896229;
    let meta = serde_json::json!({"reader_reference":"readDataFromRawEcho0_ts.m", "implementation":"syisr/ulp2ms", "header":first,"header_clock_convention":"China civil time UTC+8, explicitly confirmed by data owner","selected_utc_start_s":times[0],"selected_utc_end_exclusive_s":times[times.len()-1]+ipp,"integration_seconds":frames.len() as f64*ipp,"range_definition":"monostatic slant range; pulse-corrected centroid; nominal rectangular transmit pulse; no measured receiver group delay correction","acf_convention":"x[k] conj(x[k+lag]); exp(+i 2pi f t) has negative ACF phase; physical velocity sign unverified","range_summation":"trapezoidal: late sample j=i..i+lag+nu-1; nu=21; divide by number of summed pairs then by fractional pulse overlap (200-lag)/200","ambiguity":"all lags have common centre and 329.7717 km extent; kernel SHAPES remain lag-dependent and are saved; receiver filter tails not included in rectangular kernels","background_reference_index":reference_i,"background_reference_slant_range_km":bg_range,"background_removal":"complex background from last complete gate, each 10-second block and lag, subtracted at every range BEFORE overlap correction and normalization; receiver signal transfer function not deconvolved","normalization":"real noise-subtracted ACF divided by real noise-subtracted zero lag at each range; signed values retained","error":"delete-one-10-second-block jackknife SEM; unequal retained-pulse counts weighted; not a full covariance or guaranteed independent samples","pulse_count":frames.len(),"outlier_rejection":screened,"retained_pulse_count":retained,"screen":"optional whole-IPP rejection: any 16-sample guard average >64 times same-range median over selected integration, samples 60..1660; or >12 times median and >0.92 single-tone power fraction across 48 samples for three successive pulses; plus/minus 2 pulse margin; candidates not orbital identifications","iqr_screen":"Hann 128 samples / hop 32, 8-pulse spectra; each range-frequency bin uses temporal Q3+6 IQR; positive excess concentrated >0.6 in three of eleven bins; neighboring time/range support within one Doppler bin; one time-block margin. Broad-power Q3+6 IQR branch retained.","absolute_calibration":false,"assumed_future_F_peak_ne_m3":1e12,"ion_fraction_fit_performed":false});
    let mut meta = meta;
    meta["screen_end_sample_exclusive"] = serde_json::json!(screen_end);
    meta["header_clock_reconstruction"] = serde_json::json!({"enabled":ns==1194 && a[1].ends_with(".json"),"corrected_header_count":times.iter().zip(&original_header_times).filter(|(a,b)|(*a-*b).abs()>26e-6).count(),"maximum_absolute_correction_seconds":times.iter().zip(&original_header_times).map(|(a,b)|(a-b).abs()).fold(0.,f64::max),"method":"2023 spans anchored to indexed first header with endpoint-verified frame count and50msIPP; original_header_time_utc_s preserves all raw timestamp errors; no fabricated voltage samples"});
    meta["timing_gaps"] = serde_json::json!({"nonmonotonic_header_intervals":times.windows(2).filter(|t|t[1]<=t[0]).count(),"count":times.windows(2).filter(|t|(t[1]-t[0]-ipp).abs()>26e-6).count(),"handling":"timestamps retained; no fabricated pulses; within-pulse lag products; nominal10s blocks are groups of retained acquisition slots; full timestamp vector saved","elapsed_seconds":times[times.len()-1]+ipp-times[0]});
    meta["stationary_mean_removal"] = serde_json::json!({"enabled":remove_stationary,"estimator":"per-native-sample complex mean over retained pulses in each nominal10s block; centered product sum divided by N-1; removed means saved; suppresses coherent stationary returns; Gaussian covariance approximation has finite-N correction"});
    meta["screen"]=serde_json::json!(format!("Whole-IPP bright/coherent and range-Doppler IQR rejection; native samples through {screen_end}; same thresholds as previous campaign; candidates not orbital identifications"));
    f.new_attr_builder()
        .with_data(meta.to_string().as_str())
        .create("metadata_json")?;
    eprintln!(
        "Saved {nr} ranges, background centroid {bg_range:.2} km, UTC start {}",
        times[0]
    );
    Ok(())
}
