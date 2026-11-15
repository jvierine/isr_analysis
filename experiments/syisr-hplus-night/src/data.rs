use crate::{retrieval::Observation, theory::Theory};
use anyhow::{ensure, Result};
use num_complex::Complex64 as C;
use serde_json::{json, Value};
use std::f64::consts::PI;
pub struct Archive {
    pub meta: Value,
    values: [Vec<f32>; 4],
}
pub struct Hour {
    pub hour: usize,
    pub count: f64,
    pub mean: [Vec<f64>; 2],
    pub sem: [Vec<f64>; 2],
}
impl Archive {
    pub fn load(p: &str) -> Result<Self> {
        let h = hdf5::File::open(p)?;
        let meta: Value =
            serde_json::from_slice(&h.dataset("metadata_json_utf8")?.read_raw::<u8>()?)?;
        let names = [
            "acf_real_adu2",
            "acf_imag_adu2",
            "standard_error_real_adu2",
            "standard_error_imag_adu2",
        ];
        let mut values: [Vec<f32>; 4] = std::array::from_fn(|_| vec![]);
        for (i, n) in names.iter().enumerate() {
            values[i] = h.dataset(n)?.read_raw::<f32>()?;
        }
        Ok(Self { meta, values })
    }
    pub fn indices(&self, hour: usize) -> Vec<usize> {
        self.meta["windows"]
            .as_array()
            .unwrap()
            .iter()
            .enumerate()
            .filter_map(|(j, w)| {
                let t = w["utc_start_s"].as_f64().unwrap();
                (((t as i64).rem_euclid(86400) / 3600) as usize == hour).then_some(j)
            })
            .collect()
    }
    pub fn hour(&self, hour: usize) -> Hour {
        self.hour_indices(hour, &self.indices(hour))
    }
    pub fn hour_indices(&self, hour: usize, indices: &[usize]) -> Hour {
        let n = 41 * 1324;
        let mut mean = [vec![0.; n], vec![0.; n]];
        let mut moments = [vec![0.; n], vec![0.; n]];
        let mut count = 0.;
        for &j in indices {
            let w = &self.meta["windows"][j];
            let t = w["utc_start_s"].as_f64().unwrap();
            let dh = chrono::DateTime::from_timestamp(t as i64, 0)
                .unwrap()
                .format("%H")
                .to_string()
                .parse::<usize>()
                .unwrap();
            if dh != hour {
                continue;
            }
            let c = w["pulse_count"].as_u64().unwrap() as f64;
            count += c;
            for q in 0..2 {
                for k in 0..n {
                    let y = self.values[q][j * n + k] as f64;
                    let se = self.values[q + 2][j * n + k] as f64;
                    mean[q][k] += c * y;
                    moments[q][k] += c * y * y + c * (c - 1.) * se * se;
                }
            }
        }
        for q in 0..2 {
            for k in 0..n {
                mean[q][k] /= count;
                moments[q][k] = ((moments[q][k] - count * mean[q][k].powi(2)).max(0.)
                    / ((count - 1.) * count))
                    .sqrt();
            }
        }
        Hour {
            hour,
            count,
            mean,
            sem: moments,
        }
    }
}
fn fir(noise: &[C]) -> (Vec<(f64, C)>, f64) {
    let n = 256;
    let psd: Vec<f64> = (0..n)
        .map(|k| {
            1. + (1..noise.len())
                .map(|l| {
                    2. * (noise[l] * C::from_polar(1., 2. * PI * k as f64 * l as f64 / n as f64)).re
                })
                .sum::<f64>()
        })
        .collect();
    let negative: f64 = psd.iter().filter(|v| **v < 0.).map(|v| -v).sum::<f64>()
        / psd.iter().map(|v| v.abs()).sum::<f64>();
    let root: Vec<_> = psd.iter().map(|v| v.max(1e-8).sqrt()).collect();
    let mut taps: Vec<(f64, C)> = (-30..=30)
        .map(|j| {
            (
                j as f64 * 1e-5,
                root.iter()
                    .enumerate()
                    .map(|(k, &v)| {
                        C::from_polar(v / n as f64, 2. * PI * k as f64 * j as f64 / n as f64)
                    })
                    .sum(),
            )
        })
        .collect();
    let norm = taps.iter().map(|(_, z)| z.norm_sqr()).sum::<f64>().sqrt();
    for (_, z) in &mut taps {
        *z /= norm;
    }
    (taps, negative)
}
impl Hour {
    pub fn observation(
        &self,
        height: f64,
        theory: &Theory,
        background: u8,
        lag_min: usize,
    ) -> Result<(Observation, Value)> {
        let n = 1324;
        let bg = match background {
            1 => 1200..1260,
            2 => 1100..1160,
            _ => 0..35,
        };
        let width = (bg.end - bg.start) as f64;
        let noise: Vec<C> = (0..41)
            .map(|l| {
                C::new(
                    bg.clone().map(|k| self.mean[0][l * n + k]).sum::<f64>() / width,
                    bg.clone().map(|k| self.mean[1][l * n + k]).sum::<f64>() / width,
                )
            })
            .collect();
        ensure!(noise[0].re > 0., "negative reference noise");
        let rho: Vec<_> = noise.iter().map(|z| *z / noise[0].re).collect();
        let (taps, negative) = fir(&rho);
        let noise_sem = bg.clone().map(|k| self.sem[0][k]).sum::<f64>() / width;
        // Fit a local exponential slope to the observed zero-lag echo. This is a
        // controlled LP-gradient approximation, not a deconvolved density profile.
        let mut xy = vec![];
        for k in 0..1200 {
            let h = (767e-6 + k as f64 * 1e-5 - 240e-6) * 299792.458 / 2. + 0.05;
            let p = self.mean[0][k] - noise[0].re;
            if (h - height).abs() < 125. && p > 5. * self.sem[0][k].hypot(noise_sem) {
                xy.push((h - height, p.ln()));
            }
        }
        let gamma = if xy.len() > 30 {
            let mx = xy.iter().map(|v| v.0).sum::<f64>() / xy.len() as f64;
            let my = xy.iter().map(|v| v.1).sum::<f64>() / xy.len() as f64;
            (xy.iter().map(|v| (v.0 - mx) * (v.1 - my)).sum::<f64>()
                / xy.iter().map(|v| (v.0 - mx).powi(2)).sum::<f64>())
            .clamp(-0.03, 0.03)
        } else {
            0.
        };
        let df = theory.f[1] - theory.f[0];
        let mut values = vec![];
        let mut sigma = vec![];
        let mut kernel = vec![];
        let mut nr = vec![];
        for l in lag_min..=40 {
            let tau = l as f64 * 1e-5;
            let time = 2. * (height - 0.05) / 299792.458 + 240e-6 - tau / 2.;
            let q = (time - 767e-6) / 1e-5;
            let center = q.floor() as isize;
            let frac = q - center as f64;
            ensure!(
                center >= 9 && center + 10 + (l as isize) < (n as isize),
                "range outside data"
            );
            let mut value = C::new(0., 0.);
            let mut se = [0.; 2];
            for d in -8..=8 {
                for (j, w) in [
                    (center + d, (1. - frac) / 17.),
                    (center + d + 1, frac / 17.),
                ] {
                    let i = l * n + j as usize;
                    value += C::new(self.mean[0][i], self.mean[1][i]) * w;
                    for z in 0..2 {
                        se[z] += w * self.sem[z][i];
                    }
                }
            }
            value -= noise[l];
            for z in 0..2 {
                let bse = bg.clone().map(|k| self.sem[z][l * n + k]).sum::<f64>() / width;
                se[z] = se[z].hypot(bse).max(noise_sem * 0.1);
            }
            values.push(value);
            sigma.push(se);
            nr.push(rho[l]);
            let mut weights = vec![(0., C::new(0., 0.)); 121];
            for &(a, ha) in &taps {
                for &(b, hb) in &taps {
                    let tt = tau + a - b;
                    let length = (480e-6 - tt.abs()).max(0.);
                    if length == 0. {
                        continue;
                    }
                    let d = gamma * 299792.458 * length / 4.;
                    let sinhc = if d.abs() < 1e-7 { 1. } else { d.sinh() / d };
                    let weight = ha
                        * hb.conj()
                        * (length / 480e-6)
                        * sinhc
                        * (-gamma * 299792.458 * (a + b) / 4.).exp();
                    let index = ((a - b) / 1e-5).round() as isize + 60;
                    weights[index as usize].0 = tt;
                    weights[index as usize].1 += weight;
                }
            }
            let response: Vec<C> = theory
                .f
                .iter()
                .enumerate()
                .map(|(i, &f)| {
                    weights
                        .iter()
                        .map(|(t, w)| *w * C::from_polar(1., -2. * PI * f * t))
                        .sum::<C>()
                        * df
                        * if i == 0 || i + 1 == theory.f.len() {
                            0.5
                        } else {
                            1.
                        }
                })
                .collect();
            kernel.push(response);
        }
        let obs = Observation {
            value: values,
            sigma,
            kernel,
            rho: nr,
            background_sigma: noise_sem * 2.,
            lag_min,
        };
        Ok((
            obs,
            json!({"height_km":height,"utc_hour":self.hour,"pulse_count":self.count,"background_gates":match background{1=>vec![1200,1260],2=>vec![1100,1160],_=>vec![0,35]},"noise_power_adu2":noise[0].re,"noise_mean_sem_adu2":noise_sem,"receiver_negative_psd_mass_fraction":negative,"receiver":"minimum zero-phase voltage spectral factor from 41 measured background lags; only inferred receiver PSD is projected nonnegative; signed data retained","lag_min":lag_min,"lag_max":40,"local_log_power_gradient_per_km":gamma,"gradient_fit_points":xy.len(),"range_response":"480 us rectangle plus inferred voltage FIR and local exponential power gradient; 17 receive bins average; nominal support about 98 km plus FIR tails","absolute_density_calibrated":false,"shape_assumption":"constant Ti/Te/composition within each LP footprint, local exponential raw-power gradient; all species share Ti"}),
        ))
    }
}
