//! Prepare measured lag products and errors in Rust; no inverse lag filtering.
use anyhow::{ensure, Result};
use num_complex::Complex64 as C;
use serde_json::{json, Value};
use std::{f64::consts::TAU, fs};
fn main() -> Result<()> {
    let a: Vec<_> = std::env::args().collect();
    ensure!(
        a.len() >= 3,
        "pair_masked.h5 output.json [receiver: full|taper] [minutes] [bin]"
    );
    let receiver = a.get(3).map(String::as_str).unwrap_or("taper");
    let minutes: usize = a.get(4).map(|s| s.parse()).transpose()?.unwrap_or(60);
    let bin: usize = a.get(5).map(|s| s.parse()).transpose()?.unwrap_or(0);
    let h = hdf5::File::open(&a[1])?;
    let meta: Value = serde_json::from_slice(&h.dataset("metadata_json_utf8")?.read_raw::<u8>()?)?;
    let first = meta["windows"][0]["utc_start_s"].as_f64().unwrap();
    let hour = (first / 3600.).floor() * 3600.;
    let lo = hour + (bin * minutes * 60) as f64;
    let hi = lo + (minutes * 60) as f64;
    let ids: Vec<_> = meta["windows"]
        .as_array()
        .unwrap()
        .iter()
        .enumerate()
        .filter_map(|(i, w)| {
            let t = w["utc_start_s"].as_f64().unwrap();
            (t >= lo && t < hi).then_some(i)
        })
        .collect();
    ensure!(!ids.is_empty(), "empty bin");
    let names = [
        "acf_real_adu2",
        "acf_imag_adu2",
        "standard_error_real_adu2",
        "standard_error_imag_adu2",
    ];
    let raw: Vec<Vec<f32>> = names
        .iter()
        .map(|n| h.dataset(n)?.read_raw::<f32>())
        .collect::<std::result::Result<_, _>>()?;
    let counts = if let Ok(d) = h.dataset("pair_count") {
        d.read_raw::<u32>()?
    } else {
        meta["windows"]
            .as_array()
            .unwrap()
            .iter()
            .flat_map(|w| {
                (0..41).flat_map(move |l| {
                    (0..1324).map(move |k| {
                        if k + l < 1324 {
                            w["pulse_count"].as_u64().unwrap() as u32
                        } else {
                            0
                        }
                    })
                })
            })
            .collect()
    };
    let n = 1324;
    let size = n * 41;
    let bg = 1200..1260;
    let mut values = vec![C::default(); size];
    let mut errors = vec![[0.; 2]; size];
    let mut pooled = vec![0.; size];
    let mut noise = vec![C::default(); 41];
    let mut noise_sem = vec![[0.; 2]; 41];
    let mut noisew = vec![0.; 41];
    let mut per_window = vec![];
    for &w in &ids {
        let mut nz = vec![C::default(); 41];
        let mut ns = vec![[0.; 2]; 41];
        for l in 0..41 {
            let nc: f64 = bg
                .clone()
                .map(|k| counts[w * size + l * n + k] as f64)
                .sum();
            ensure!(nc > 0., "no background");
            for k in bg.clone() {
                let j = w * size + l * n + k;
                let weight = counts[j] as f64 / nc;
                nz[l] += C::new(raw[0][j] as f64, raw[1][j] as f64) * weight;
                for q in 0..2 {
                    ns[l][q] += raw[q + 2][j] as f64 * weight;
                }
            }
            let c = meta["windows"][w]["pulse_count"].as_u64().unwrap() as f64;
            noise[l] += nz[l] * c;
            for q in 0..2 {
                noise_sem[l][q] += ns[l][q] * c;
            }
            noisew[l] += c;
        }
        per_window.push((w, nz, ns));
    }
    for l in 0..41 {
        noise[l] /= noisew[l];
        for q in 0..2 {
            noise_sem[l][q] /= noisew[l];
        }
    }
    for l in 0..41 {
        for k in 0..n - l {
            let i = l * n + k;
            let total: f64 = ids.iter().map(|&w| counts[w * size + i] as f64).sum();
            if total < 200. {
                continue;
            }
            pooled[i] = total;
            let mut mu = C::default();
            let mut w2 = 0.;
            let mut within = [0.; 2];
            for (w, nz, ns) in &per_window {
                let j = w * size + i;
                let weight = counts[j] as f64 / total;
                mu += (C::new(raw[0][j] as f64, raw[1][j] as f64) - nz[l]) * weight;
                w2 += weight * weight;
                for q in 0..2 {
                    within[q] += (weight * (raw[q + 2][j] as f64).hypot(ns[l][q])).powi(2);
                }
            }
            let mut between = [0.; 2];
            for (w, nz, _) in &per_window {
                let j = w * size + i;
                let weight = counts[j] as f64 / total;
                let d = C::new(raw[0][j] as f64, raw[1][j] as f64) - nz[l] - mu;
                between[0] += weight * d.re * d.re;
                between[1] += weight * d.im * d.im;
            }
            values[i] = mu;
            for q in 0..2 {
                errors[i][q] = within[q]
                    .max(between[q] * w2 / (1. - w2).max(1e-12))
                    .sqrt()
                    .max(noise[0].re * 1e-5);
            }
        }
    }
    ensure!(noise[0].re > 0., "invalid receiver noise");
    let mut rho: Vec<_> = noise.iter().map(|v| *v / noise[0].re).collect();
    let half = if receiver == "full" || receiver == "pedestal" {
        30isize
    } else {
        10
    };
    let pedestal = if receiver == "pedestal" {
        noise[20..41].iter().map(|v| v.re).sum::<f64>() / 21.
    } else {
        0.
    };
    if receiver == "pedestal" {
        rho = noise
            .iter()
            .map(|v| (*v - C::new(pedestal, 0.)) / (noise[0].re - pedestal))
            .collect();
    }
    if receiver != "full" && receiver != "pedestal" {
        for l in 1..41 {
            rho[l] *= if l <= 10 {
                0.5 + 0.5 * (std::f64::consts::PI * l as f64 / 11.).cos()
            } else {
                0.
            };
        }
    }
    let nn = 512;
    let psd: Vec<f64> = (0..nn)
        .map(|k| {
            1. + (1..41)
                .map(|l| {
                    2. * (rho[l] * C::from_polar(1., TAU * k as f64 * l as f64 / nn as f64)).re
                })
                .sum::<f64>()
        })
        .collect();
    ensure!(psd.iter().all(|v| *v > 0.), "receiver PSD not positive");
    let fir: Vec<_> = (-half..=half)
        .map(|j| {
            let z: C = psd
                .iter()
                .enumerate()
                .map(|(k, v)| {
                    C::from_polar(v.sqrt() / nn as f64, TAU * k as f64 * j as f64 / nn as f64)
                })
                .sum();
            json!([j as f64 * 1e-5, [z.re, z.im]])
        })
        .collect();
    let knots: Vec<_> = (0..10).map(|i| 275. + 150. * i as f64).collect();
    let edges: Vec<_> = (0..=56).map(|i| 237.5 + 25. * i as f64).collect();
    let mut products = vec![];
    let total_pulses: f64 = ids
        .iter()
        .map(|&w| meta["windows"][w]["pulse_count"].as_u64().unwrap() as f64)
        .sum();
    for l in 0..41 {
        for k in (100..1100).step_by(10) {
            let t = 767e-6 + k as f64 * 1e-5;
            let tau = l as f64 * 1e-5;
            let pad = half as f64 * 1e-5;
            if (t + tau - 480e-6 - pad) * 149896.229 < edges[0]
                || (t + pad) * 149896.229 > *edges.last().unwrap()
            {
                continue;
            }
            let i = l * n + k;
            if pooled[i] < 0.5 * total_pulses {
                continue;
            }
            let mut e = errors[i];
            if l == 0 {
                e[1] = 1e30;
            }
            products.push(json!({"sample":k,"lag":l,"value":[values[i].re,values[i].im],"sigma":e,"pair_count":pooled[i]}));
        }
    }
    let initial:Vec<_>=knots.iter().map(|r|{let k=((2.*r/299792.458+240e-6-767e-6)/1e-5).round()as usize;json!({"power_adu2":values[k].re.max(noise[0].re*0.005),"ti_k":1000.,"te_k":1500.,"velocity_m_s":0.,"h_fraction":0.2,"he_fraction":0.05,"o_fraction":0.75})}).collect();
    let prescribed: Vec<_> = knots
        .iter()
        .zip(&initial)
        .map(|(r, v)| {
            let weak = *r >= 1100. && v["power_adu2"].as_f64().unwrap() / noise[0].re < 0.04;
            [false, weak, weak, weak, weak, weak]
        })
        .collect();
    let p = json!({"schema":"syisr-hplus-coupled-v2","radar_frequency_hz":430.2e6,"shape_density_m3":1e11,"frequency_step_hz":500.,"frequency_limit_hz":50000.,"sample_rate_hz":1e5,"gate_front_s":767e-6,"pulse_width_s":480e-6,"knot_range_km":knots,"cell_edges_km":edges,"fir":fir,"products":products,"initial":initial,"prescribed_node_fields":prescribed,"background_basis":noise.iter().map(|z|vec![z.re/noise[0].re,z.im/noise[0].re]).collect::<Vec<_>>(),"background_sigma":noise_sem[0][0]*2.,"score_tolerance":0.003,"provenance":{"source":a[1],"time_start_unix_s":lo,"time_end_unix_s":hi,"beam_dwells":ids.len(),"pulse_count":total_pulses,"receiver":receiver,"removed_reference_pedestal_adu2":pedestal,"noise_power_adu2":noise[0].re,"screen":meta,"errors":"max propagated pair SEM and between-dwell SEM, diagonal; reference-gate SEM conservative","response":"individual lag-dependent range/receiver kernels, continuous range-coupled profiles; no equal-ambiguity filtering","lags":"all 0..400us, imaginary zero-lag has zero weight"}});
    fs::write(&a[2], serde_json::to_vec(&p)?)?;
    println!("{} products {} dwells", products.len(), ids.len());
    Ok(())
}
