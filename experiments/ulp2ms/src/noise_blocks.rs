//! One authoritative extraction of fitted range/lag block measurements.
use anyhow::{ensure, Result};
use nalgebra::DMatrix as M;
use num_complex::Complex64 as C;
use std::path::Path;
pub struct NoiseBlocks {
    pub zero_blocks: M<f64>,
    pub native: Vec<usize>,
    pub counts: Vec<f64>,
    pub total: f64,
    pub blocks: M<f64>,
    pub background: Vec<Vec<C>>,
    pub sigma: Vec<f64>,
    pub pairs: Vec<usize>,
    pub differences: M<f64>,
}
pub fn get(f: &hdf5::File, n: &str) -> Result<Vec<f64>> {
    Ok(f.dataset(n)?.read_raw::<f64>()?)
}
impl NoiseBlocks {
    pub fn load(root: &Path, t: usize) -> Result<Self> {
        let f = hdf5::File::open(root.join(format!("windows/{t:03}/profile.h5")))?;
        let m: serde_json::Value =
            serde_json::from_slice(&f.attr("metadata_json")?.read_raw::<u8>()?)?;
        let old = hdf5::File::open(m["provenance"]["lpm"].as_str().unwrap())?;
        let partial = hdf5::File::open(m["partial_products"]["source"].as_str().unwrap())?;
        let native = f
            .dataset("observed_native_gate_index")?
            .read_raw::<u64>()?
            .iter()
            .map(|&v| v as usize)
            .collect::<Vec<_>>();
        let counts = get(&old, "block_retained_pulses")?;
        let total = counts.iter().sum();
        let nb = counts.len();
        let ng = native.len();
        let nr = old.dataset("slant_range_km")?.size();
        let ne = partial.dataset("slant_range_km")?.size();
        let br = get(&old, "block_acf_real_adu2")?;
        let bi = get(&old, "block_acf_imag_adu2")?;
        let er = get(&partial, "block_acf_real_adu2")?;
        let ei = get(&partial, "block_acf_imag_adu2")?;
        let blocks = M::from_fn(ng * 120, nb, |row, b| {
            let i = native[row / 120];
            let l = (row / 2) % 60 + 1;
            if i < 90 {
                let j = (b * ne + (i - 20) / 5) * 61 + l;
                if row % 2 == 0 {
                    er[j]
                } else {
                    ei[j]
                }
            } else {
                let j = (b * nr + (i - 60) / 5) * 61 + l;
                if row % 2 == 0 {
                    br[j]
                } else {
                    bi[j]
                }
            }
        });
        let zero_blocks = M::from_fn(ng, nb, |g, b| {
            let i = native[g];
            if i < 90 {
                er[(b * ne + (i - 20) / 5) * 61]
            } else {
                br[(b * nr + (i - 60) / 5) * 61]
            }
        });
        let bg_r = get(&old, "background_acf_real_adu2")?;
        let bg_i = get(&old, "background_acf_imag_adu2")?;
        let background = (0..nb)
            .map(|b| {
                (0..61)
                    .map(|l| C::new(bg_r[b * 61 + l], bg_i[b * 61 + l]))
                    .collect()
            })
            .collect();
        let mut sigma = vec![0.; ng * 120];
        for (c, name) in ["real", "imag"].iter().enumerate() {
            let s = get(&f, &format!("sem_{name}_adu2"))?;
            let y = get(&f, &format!("observed_{name}_adu2"))?;
            for j in 0..ng * 60 {
                sigma[2 * j + c] = s[j];
                let mean = (0..nb)
                    .filter(|&b| counts[b] > 0.)
                    .map(|b| blocks[(2 * j + c, b)] * counts[b] / total)
                    .sum::<f64>();
                ensure!((mean - y[j]).abs() / s[j] < 1e-6, "block/fit mean mismatch");
            }
        }
        let pairs: Vec<usize> = (0..nb - 1)
            .step_by(2)
            .filter(|&b| counts[b] > 0. && counts[b + 1] > 0.)
            .collect();
        let differences = M::from_fn(ng * 120, pairs.len(), |i, j| {
            let b = pairs[j];
            (blocks[(i, b + 1)] - blocks[(i, b)]) / (1. / counts[b] + 1. / counts[b + 1]).sqrt()
        });
        Ok(Self {
            zero_blocks,
            native,
            counts,
            total,
            blocks,
            background,
            sigma,
            pairs,
            differences,
        })
    }
    pub fn voltage_noise(&self, pairs: &[usize]) -> Result<(Vec<C>, f64, f64)> {
        let mut mean = vec![C::default(); 61];
        let mut n = 0.;
        for &j in pairs {
            for b in [self.pairs[j], self.pairs[j] + 1] {
                n += self.counts[b];
                for l in 0..61 {
                    mean[l] += self.background[b][l] * self.counts[b];
                }
            }
        }
        for v in &mut mean {
            *v /= n;
        }
        let power = mean[0].re;
        ensure!(power > 0., "nonpositive reference noise power");
        let rho: Vec<C> = (0..61)
            .map(|l| {
                if l == 0 {
                    C::new(1., 0.)
                } else if l < 21 {
                    mean[l] / power * (0.5 + 0.5 * (std::f64::consts::PI * l as f64 / 21.).cos())
                } else {
                    C::default()
                }
            })
            .collect();
        let (fir, negative) = crate::fit_response::receiver_fir(&rho);
        ensure!(negative < 1e-3, "receiver PSD projection too large");
        let gamma = crate::structured_covariance::VoltageProducts::from_fir(&fir)?.gamma;
        Ok((gamma, power, negative))
    }
    pub fn local_voltage_acfs(&self, pairs: &[usize]) -> Vec<Vec<C>> {
        let mut n = 0.;
        let mut mean = vec![vec![C::default(); 61]; self.native.len()];
        for &j in pairs {
            for b in [self.pairs[j], self.pairs[j] + 1] {
                let w = self.counts[b];
                n += w;
                for g in 0..self.native.len() {
                    mean[g][0] +=
                        (C::new(self.zero_blocks[(g, b)], 0.) + self.background[b][0]) * w;
                    for l in 1..=60 {
                        let i = 2 * (g * 60 + l - 1);
                        let q = C::new(self.blocks[(i, b)], self.blocks[(i + 1, b)])
                            * (200 - l) as f64
                            / 200.
                            + self.background[b][l];
                        mean[g][l] += q * w;
                    }
                }
            }
        }
        for a in &mut mean {
            for v in a {
                *v /= n;
            }
        }
        mean
    }
}
pub fn difference_factors(z: &M<f64>, columns: &[usize], total: f64) -> M<f64> {
    let n = columns.len();
    let means: Vec<f64> = (0..z.nrows())
        .map(|i| columns.iter().map(|&j| z[(i, j)]).sum::<f64>() / n as f64)
        .collect();
    M::from_fn(z.nrows(), n, |i, j| {
        (z[(i, columns[j])] - means[i]) / ((n - 1) as f64 * total).sqrt()
    })
}
