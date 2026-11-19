use crate::{
    campaign::{save, Dwell},
    reader::read_frame,
    screen::{screen, Summary},
};
use anyhow::{ensure, Result};
use num_complex::{Complex32 as C32, Complex64 as C};
use serde::{Deserialize, Serialize};
use std::{
    fs,
    io::{BufReader, Seek, SeekFrom},
    path::Path,
};
pub const NL: usize = 62;
#[derive(Clone, Serialize, Deserialize)]
pub struct ProductDwell {
    pub schema: String,
    pub dwell: Dwell,
    pub pulses: usize,
    pub cycles: usize,
    pub cycle_time: Vec<f64>,
    pub heights: Vec<f64>,
    pub ranges: Vec<f64>,
    pub widths: Vec<f64>,
    pub altitude_edges_km: Vec<f64>,
    pub values: Vec<[f64; 2]>,
    pub valid: Vec<bool>,
    pub noise: Vec<[f64; 2]>,
    pub mask_summary: Summary,
    pub ground_samples: Vec<usize>,
    pub reference_ne: Vec<Option<f64>>,
    pub reference_ne_error: Vec<Option<f64>>,
}
#[derive(Clone)]
struct Coeff {
    p: usize,
    k: usize,
    w: f64,
}
pub fn phase(path: &Path) -> Result<Vec<Vec<f64>>> {
    let c: Vec<Vec<f64>> = fs::read_to_string(path)?
        .lines()
        .filter(|s| !s.trim().is_empty())
        .map(|s| {
            s.chars()
                .map(|c| match c {
                    '+' => Ok(1.),
                    '-' => Ok(-1.),
                    _ => anyhow::bail!("phase"),
                })
                .collect()
        })
        .collect::<Result<_>>()?;
    ensure!(
        c.len() == 64 && c.iter().all(|a| a.len() == 32),
        "64x32 phase table required"
    );
    Ok(c)
}
fn make_plan(codes: &[Vec<f64>], el: f64) -> (Vec<Vec<Coeff>>, Vec<crate::grid::GateBin>) {
    let n = 1334;
    let grid = crate::grid::build(el, crate::grid::maximum_altitude_km());
    let mut plans = vec![];
    for gate in &grid {
        for lag in 1..=NL {
            let mut coeff = vec![0f64; 64 * n];
            let branches: Vec<_> = [(0, lag / 2), (1, (lag + 1) / 2)]
                .into_iter()
                .filter(|(_, l)| *l > 0)
                .collect();
            for (a, l) in &branches {
                let factor =
                    (if lag % 2 == 1 { 2. } else { 1. }) / (branches.len() * 64 * (32 - l)) as f64;
                for b in 0..32 - l {
                    for (j, weight) in gate.native_cells.iter().zip(&gate.weights) {
                        let k = j * 2 + b * 2 + a;
                        for c in 0..64 {
                            coeff[c * n + k] += factor * weight * codes[c][b] * codes[c][b + l];
                        }
                    }
                }
            }
            plans.push(
                coeff
                    .into_iter()
                    .enumerate()
                    .filter(|(_, w)| w.abs() > 1e-14)
                    .map(|(j, w)| Coeff {
                        p: j / n,
                        k: j % n,
                        w,
                    })
                    .collect(),
            );
        }
    }
    (plans, grid)
}
type PlanData = (Vec<Vec<Coeff>>, Vec<crate::grid::GateBin>);
fn plan(codes: &[Vec<f64>], el: f64) -> std::sync::Arc<PlanData> {
    use std::collections::BTreeMap;
    use std::hash::{Hash, Hasher};
    use std::sync::{Arc, Mutex, OnceLock};
    static CACHE: OnceLock<Mutex<BTreeMap<(u64, u64), Arc<PlanData>>>> = OnceLock::new();
    let mut hash = std::collections::hash_map::DefaultHasher::new();
    for c in codes {
        for v in c {
            v.to_bits().hash(&mut hash);
        }
    }
    let key = (el.to_bits(), hash.finish());
    let mut cache = CACHE
        .get_or_init(|| Mutex::new(BTreeMap::new()))
        .lock()
        .unwrap();
    cache
        .entry(key)
        .or_insert_with(|| Arc::new(make_plan(codes, el)))
        .clone()
}
pub fn decode(d: &Dwell, codes: &[Vec<f64>], out: &Path, keepmask: bool) -> Result<ProductDwell> {
    let n = 1334;
    let mut frames = vec![];
    for s in &d.spans {
        let mut f = BufReader::new(fs::File::open(&s.file)?);
        f.seek(SeekFrom::Start(s.offset))?;
        for _ in 0..s.pulses {
            frames.push(read_frame(&mut f)?.ok_or_else(|| anyhow::anyhow!("EOF"))?);
        }
    }
    // Keep complete0..63 cycles only; partial file-edge dwell data is explicit.
    let first = frames
        .iter()
        .position(|f| f.header.little_code == 0)
        .ok_or_else(|| anyhow::anyhow!("no full cycle"))?;
    let np = (frames.len() - first) / 64 * 64;
    ensure!(np >= 64, "no complete code cycle");
    let frames = &frames[first..first + np];
    ensure!(
        frames
            .iter()
            .enumerate()
            .all(|(p, f)| f.header.little_code as usize == p % 64),
        "code phase sequence changed"
    );
    let x: Vec<C32> = frames.iter().flat_map(|f| f.iq.clone()).collect();
    let (mask, summary) = screen(&x, np, n, codes, 4);
    let nc = np / 64;
    let mut noise = vec![C::default(); NL + 1];
    for l in 0..=NL {
        let mut count = 0;
        for p in 0..np {
            for k in 1200..1250 {
                if mask[p * n + k] == 0 && mask[p * n + k + l] == 0 {
                    let a = x[p * n + k];
                    let b = x[p * n + k + l];
                    noise[l] +=
                        C::new(a.re as f64, a.im as f64) * C::new(b.re as f64, b.im as f64).conj();
                    count += 1;
                }
            }
        }
        ensure!(count > 100, "quiet reference rejected");
        noise[l] /= count as f64;
    }
    // Stationary-coherence detector uses independent first/second repeat halves.
    let mut ground = vec![];
    let mut means = vec![C::default(); 64 * n];
    let mut mean_count = vec![0usize; 64 * n];
    for c in 0..64 {
        for k in 0..n {
            for p in (c..np).step_by(64) {
                if mask[p * n + k] == 0 {
                    means[c * n + k] += C::new(x[p * n + k].re as f64, x[p * n + k].im as f64);
                    mean_count[c * n + k] += 1;
                }
            }
            if mean_count[c * n + k] > 0 {
                means[c * n + k] /= mean_count[c * n + k] as f64;
            }
        }
    }
    for k in 0..n {
        let (mut cross, mut variance) = (0., 0.);
        for c in 0..64 {
            let mut half = [C::default(); 2];
            let mut count = [0usize; 2];
            let mut energy = [0.; 2];
            for (p, q) in (c..np).step_by(64).enumerate() {
                let part = usize::from(p >= nc / 2);
                if mask[q * n + k] == 0 {
                    let v = C::new(x[q * n + k].re as f64, x[q * n + k].im as f64);
                    half[part] += v;
                    energy[part] += v.norm_sqr();
                    count[part] += 1;
                }
            }
            if count.iter().all(|n| *n >= 4) {
                let a = half[0] / count[0] as f64;
                let b = half[1] / count[1] as f64;
                cross += (a * b.conj()).re;
                let va = (energy[0] / count[0] as f64 - a.norm_sqr()).max(0.) / count[0] as f64;
                let vb = (energy[1] / count[1] as f64 - b.norm_sqr()).max(0.) / count[1] as f64;
                variance += va * vb / 2.;
            }
        }
        if cross > 8. * variance.sqrt().max(1e-20) {
            ground.push(k);
        }
    }
    // Three adjacent detections and ten-sample support margin.
    let original = ground.clone();
    ground.retain(|k| {
        original.binary_search(&k.saturating_sub(1)).is_ok()
            && original.binary_search(&(k + 1)).is_ok()
    });
    let mut support = vec![false; n];
    for &k in &ground {
        for j in k.saturating_sub(10)..(k + 11).min(n) {
            support[j] = true;
        }
    }
    let cached = plan(codes, d.el);
    let (plans, grid) = cached.as_ref();
    let ng = grid.len();
    let heights = grid.iter().map(|g| g.altitude_km).collect();
    let ranges = grid.iter().map(|g| g.slant_range_km).collect();
    let widths = grid
        .iter()
        .map(|g| g.upper_altitude_km - g.lower_altitude_km)
        .collect();
    let mut altitude_edges_km: Vec<_> = grid.iter().map(|g| g.lower_altitude_km).collect();
    altitude_edges_km.push(grid.last().unwrap().upper_altitude_km);
    let mut values = vec![[0.; 2]; nc * ng * NL];
    let mut valid = vec![false; values.len()];
    for cycle in 0..nc {
        for (gate, coeff) in plans.iter().enumerate() {
            let l = gate % NL + 1;
            let mut sum = C::default();
            let mut good = true;
            for a in coeff {
                let p = cycle * 64 + a.p;
                let i = p * n + a.k;
                let j = i + l;
                if mask[i] != 0 || mask[j] != 0 {
                    good = false;
                    continue;
                }
                let u = C::new(x[i].re as f64, x[i].im as f64);
                let v = C::new(x[j].re as f64, x[j].im as f64);
                let mut product = u * v.conj();
                if support[a.k] || support[a.k + l] {
                    let nn = mean_count[a.p * n + a.k].min(mean_count[a.p * n + a.k + l]);
                    if nn < 4 {
                        good = false;
                        continue;
                    }
                    product = (u - means[a.p * n + a.k])
                        * (v - means[a.p * n + a.k + l]).conj()
                        * nn as f64
                        / (nn - 1) as f64;
                }
                sum += (product - noise[l]) * a.w;
            }
            let idx = cycle * ng * NL + gate;
            values[idx] = [sum.re, sum.im];
            valid[idx] = good;
        }
    }
    let product = ProductDwell {
        schema: "syisr-ac-native-adaptive-dwell-v2".into(),
        dwell: d.clone(),
        pulses: np,
        cycles: nc,
        cycle_time: (0..nc)
            .map(|c| frames[c * 64].header.header_clock_unix_s - 28800.)
            .collect(),
        heights,
        ranges,
        widths,
        altitude_edges_km,
        values,
        valid,
        noise: noise.iter().map(|v| [v.re, v.im]).collect(),
        mask_summary: summary,
        ground_samples: ground,
        reference_ne: vec![None; ng],
        reference_ne_error: vec![None; ng],
    };
    save(out, &product)?;
    if keepmask {
        let f = hdf5::File::open_rw(out)?;
        f.new_dataset::<u8>()
            .shape((np, n))
            .deflate(4)
            .create("voltage_mask")?
            .write_raw(&mask)?;
    }
    Ok(product)
}

#[cfg(test)]
mod native_tests {
    use super::*;
    #[test]
    fn homogeneous_acf_and_background_cancel_on_native_and_wide_gates() {
        let c = phase(Path::new(concat!(
            env!("CARGO_MANIFEST_DIR"),
            "/m32-reference.bc"
        )))
        .unwrap();
        let p = plan(&c, 90.);
        let (plans, grid) = p.as_ref();
        assert!(grid.len() > 100);
        for g in [0, 40, grid.len() - 1] {
            for lag in [1, 2, 5, 20, 62] {
                let raw: Vec<f64> = (0..64)
                    .map(|code| {
                        (0..64 - lag)
                            .map(|s| c[code][s / 2] * c[code][(s + lag) / 2])
                            .sum()
                    })
                    .collect();
                let coeff = &plans[g * NL + lag - 1];
                let estimate: f64 = coeff.iter().map(|a| a.w * raw[a.p]).sum();
                let background: f64 = coeff.iter().map(|a| a.w).sum();
                assert!(
                    (estimate - 2.).abs() < 1e-10,
                    "gate{g} lag{lag}: {estimate}"
                );
                assert!(background.abs() < 1e-10);
            }
        }
    }
}
