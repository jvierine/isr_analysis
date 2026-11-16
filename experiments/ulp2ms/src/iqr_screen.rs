//! Median/IQR range-Doppler screening of narrow echoes before lag estimation.
//! Hann fast-time windows, independent incoherent pulse averaging, local spectral
//! shape and track continuity; no orbital identity is inferred from these flags.
use num_complex::Complex32 as C;
use rayon::prelude::*;
use rustfft::FftPlanner;
pub const NF: usize = 128;
pub const NP: usize = 8;
pub const HOP: usize = 32;
pub const START: usize = 30;
pub const END: usize = 1660;
pub struct Audit {
    pub keep: Vec<bool>,
    pub nt: usize,
    pub nw: usize,
    pub score: Vec<f32>,
    pub peak_hz: Vec<f32>,
    pub narrow_fraction: Vec<f32>,
    pub candidate: Vec<f32>,
    pub median: Vec<f32>,
    pub iqr: Vec<f32>,
    pub q3: Vec<f32>,
    pub power_score: Vec<f32>,
}
pub fn screen(frames: &[Vec<C>], end: usize) -> Audit {
    let nw = (end - START - NF) / HOP + 1;
    let nt = frames.len().div_ceil(NP);
    let hann: Vec<f32> = (0..NF)
        .map(|j| 0.5 - 0.5 * (std::f32::consts::TAU * j as f32 / (NF - 1) as f32).cos())
        .collect();
    let energy: f32 = hann.iter().map(|v| v * v).sum();
    let fft = FftPlanner::<f32>::new().plan_fft_forward(NF);
    let spectra: Vec<Vec<f32>> = frames
        .par_chunks(NP)
        .map(|pulses| {
            let mut out = vec![0.; nw * NF];
            let mut work = vec![C::default(); NF];
            for w in 0..nw {
                let k = START + w * HOP;
                for p in pulses {
                    for j in 0..NF {
                        work[j] = p[k + j] * hann[j];
                    }
                    fft.process(&mut work);
                    for j in 0..NF {
                        out[w * NF + j] += work[j].norm_sqr() / (energy * pulses.len() as f32);
                    }
                }
            }
            out
        })
        .collect();
    let quartiles: Vec<[f32; 3]> = (0..nw * NF)
        .into_par_iter()
        .map(|j| {
            let mut v: Vec<f32> = spectra.iter().map(|s| s[j]).collect();
            v.sort_unstable_by(f32::total_cmp);
            [v[nt / 4], v[nt / 2], v[3 * nt / 4]]
        })
        .collect();
    let median: Vec<f32> = quartiles.iter().map(|q| q[1]).collect();
    let q3: Vec<f32> = quartiles.iter().map(|q| q[2]).collect();
    let iqr: Vec<f32> = quartiles.iter().map(|q| (q[2] - q[0]).max(1e-20)).collect();
    let mut score = vec![0.; nt * nw];
    let mut peak_hz = score.clone();
    let mut narrow_fraction = score.clone();
    let mut candidate = score.clone();
    let mut peak = vec![0; nt * nw];
    let power_q: Vec<[f32; 2]> = (0..nw)
        .map(|w| {
            let mut v: Vec<f32> = spectra
                .iter()
                .map(|s| s[w * NF..(w + 1) * NF].iter().sum())
                .collect();
            v.sort_unstable_by(f32::total_cmp);
            [v[3 * nt / 4], (v[3 * nt / 4] - v[nt / 4]).max(1e-20)]
        })
        .collect();
    let mut power_score = score.clone();
    for t in 0..nt {
        for w in 0..nw {
            let idx = t * nw + w;
            let s = &spectra[t][w * NF..(w + 1) * NF];
            power_score[idx] = (s.iter().sum::<f32>() - power_q[w][0]) / power_q[w][1];
            let best = (0..NF)
                .max_by(|&i, &j| {
                    ((s[i] - q3[w * NF + i]) / iqr[w * NF + i])
                        .total_cmp(&((s[j] - q3[w * NF + j]) / iqr[w * NF + j]))
                })
                .unwrap();
            peak[idx] = best;
            score[idx] = (s[best] - q3[w * NF + best]) / iqr[w * NF + best];
            peak_hz[idx] = (if best <= NF / 2 {
                best as f32
            } else {
                best as f32 - NF as f32
            }) * 100000.
                / NF as f32;
            let excess = |d: isize| {
                let j = (best as isize + d).rem_euclid(NF as isize) as usize;
                (s[j] - median[w * NF + j]).max(0.)
            };
            narrow_fraction[idx] =
                (-1..=1).map(excess).sum::<f32>() / (-5..=5).map(excess).sum::<f32>().max(1e-20);
            if score[idx] > 6. && narrow_fraction[idx] > 0.6 {
                candidate[idx] = 1.;
            }
        }
    }
    let mut keep = vec![true; frames.len()];
    for t in 0..nt {
        for w in 0..nw {
            let i = t * nw + w;
            let mut supported = false;
            if candidate[i] > 0. {
                for dt in -1isize..=1 {
                    for dw in -1isize..=1 {
                        if dt == 0 && dw == 0 {
                            continue;
                        }
                        let tt = t as isize + dt;
                        let ww = w as isize + dw;
                        if tt < 0 || ww < 0 || tt >= nt as isize || ww >= nw as isize {
                            continue;
                        }
                        let j = tt as usize * nw + ww as usize;
                        let df = peak[i].abs_diff(peak[j]);
                        if candidate[j] > 0. && df.min(NF - df) <= 1 {
                            supported = true;
                        }
                    }
                }
            }
            // Broadband severe outliers use the same IQR reference. The range-Doppler
            // branch detects much weaker narrow echoes than this power-only branch.
            if supported || power_score[i] > 6. {
                for p in (t.saturating_sub(1) * NP)..((t + 2) * NP).min(frames.len()) {
                    keep[p] = false;
                }
            }
        }
    }
    Audit {
        keep,
        nt,
        nw,
        score,
        peak_hz,
        narrow_fraction,
        candidate,
        median,
        iqr,
        q3,
        power_score,
    }
}
#[cfg(test)]
mod tests {
    use super::*;
    #[test]
    fn weak_tone_is_rejected_without_removing_thermal_population() {
        let mut seed = 87192u64;
        let mut u = || {
            seed = seed.wrapping_mul(6364136223846793005).wrapping_add(1);
            ((seed >> 32) as f64 + 0.5) / 4294967296.
        };
        let mut x = vec![vec![C::default(); 512]; 1024];
        for p in &mut x {
            for v in p {
                let r = (-u().ln()).sqrt() as f32;
                *v = C::from_polar(r, (std::f64::consts::TAU * u()) as f32);
            }
        }
        let quiet = screen(&x, 500);
        assert!(quiet.keep.iter().filter(|&&v| !v).count() < 20);
        for p in 400..440 {
            for k in 100..400 {
                x[p][k] += C::from_polar(0.45, 0.31 * k as f32 + 0.7 * p as f32);
            }
        }
        let hit = screen(&x, 500);
        assert!(hit.keep[400..440].iter().filter(|&&v| !v).count() >= 32);
    }
}
