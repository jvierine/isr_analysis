//! Gated strong alternating-code ACFs for pulse-interleaved five-beam SYISR.
use anyhow::{ensure, Result};
use num_complex::Complex64 as C;
pub mod coded_screen;
#[path = "../../src/reader.rs"]
pub mod reader;
pub const NL: usize = 62;
pub const DT: f64 = 1e-5;
pub const NFFT: usize = 512;
pub fn phases(path: &std::path::Path) -> Result<Vec<Vec<f64>>> {
    let v: Vec<Vec<_>> = std::fs::read_to_string(path)?
        .lines()
        .filter(|s| !s.trim().is_empty())
        .map(|s| {
            s.trim()
                .chars()
                .map(|c| match c {
                    '+' => Ok(1.),
                    '-' => Ok(-1.),
                    _ => anyhow::bail!("invalid phase"),
                })
                .collect::<Result<_>>()
        })
        .collect::<Result<_>>()?;
    ensure!(
        v.len() == 64 && v.iter().all(|r| r.len() == 32),
        "expected 64x32 phase table"
    );
    Ok(v)
}
#[derive(Clone)]
pub struct Term {
    pub c: usize,
    pub k: usize,
    pub w: f64,
}
pub fn plan(codes: &[Vec<f64>]) -> Vec<Vec<Term>> {
    (1..=NL)
        .map(|lag| {
            let branches: Vec<_> = [(0, lag / 2), (1, (lag + 1) / 2)]
                .into_iter()
                .filter(|(_, l)| *l > 0)
                .collect();
            let mut terms = vec![];
            for (a, l) in &branches {
                let f =
                    if lag % 2 == 1 { 2. } else { 1. } / (branches.len() * 64 * (32 - l)) as f64;
                for b in 0..32 - l {
                    for c in 0..64 {
                        terms.push(Term {
                            c,
                            k: 2 * b + a,
                            w: f * codes[c][b] * codes[c][b + l],
                        });
                    }
                }
            }
            terms
        })
        .collect()
}
/// A local even-quadratic short-lag extrapolation. Stored signed; no floor.
pub fn zero_lag(r1: C, r2: C) -> f64 {
    (4. * r1.re - r2.re) / 3.
}
/// Positive lag uses x(t+tau)*conj(x(t)); positive IQ rotation -> positive f.
pub fn spectrum(r: &[C]) -> Vec<f64> {
    assert_eq!(r.len(), NL + 1);
    let mut a = vec![C::default(); NFFT];
    a[0] = C::new(r[0].re, 0.);
    for l in 1..=NL {
        let q = r[l] * (1. - l as f64 / (NL + 1) as f64);
        a[l] = q;
        a[NFFT - l] = q.conj();
    }
    rustfft::FftPlanner::<f64>::new()
        .plan_fft_forward(NFFT)
        .process(&mut a);
    (0..NFFT)
        .map(|i| a[(i + NFFT / 2) % NFFT].re * DT)
        .collect()
}
#[cfg(test)]
mod tests {
    use super::*;
    #[test]
    fn signed_tone_and_integral() {
        let f = 7812.5;
        let r: Vec<_> = (0..=NL)
            .map(|l| C::from_polar(3., std::f64::consts::TAU * f * l as f64 * DT))
            .collect();
        let s = spectrum(&r);
        let peak = s
            .iter()
            .enumerate()
            .max_by(|a, b| a.1.total_cmp(b.1))
            .unwrap()
            .0;
        assert_eq!(peak, 296);
        assert!((s.iter().sum::<f64>() / (DT * NFFT as f64) - 3.).abs() < 1e-10);
        let mut signed = r;
        signed[0] = C::new(-1., 0.);
        assert!(spectrum(&signed).iter().any(|v| *v < 0.));
    }
    #[test]
    fn quadratic_zero() {
        let a = 7.;
        let b = 2.;
        assert!((zero_lag(C::new(a - b, 0.), C::new(a - 4. * b, 0.)) - a).abs() < 1e-12);
    }
}

#[cfg(test)]
mod code_tests {
    use super::*;
    #[test]
    fn actual_lag_codes_are_orthogonal() {
        let c = phases(std::path::Path::new("../m32-reference.bc")).unwrap();
        for l in 1..32 {
            for a in 0..32 - l {
                for b in 0..32 - l {
                    let q: f64 = (0..64)
                        .map(|p| c[p][a] * c[p][a + l] * c[p][b] * c[p][b + l])
                        .sum();
                    assert_eq!(q, if a == b { 64. } else { 0. });
                }
            }
        }
    }
    #[test]
    fn common_cycle_masks_preserve_decode() {
        let c = phases(std::path::Path::new("../m32-reference.bc")).unwrap();
        let p = plan(&c);
        for (l, ts) in p.iter().enumerate() {
            let lag = l + 1;
            let mut total = 0.;
            let mut moment = 0.;
            for source in -70_i32..70 {
                let mut kernel = 0.;
                for t in ts {
                    let a = (t.k as i32 - source).div_euclid(2);
                    let b = (t.k as i32 + lag as i32 - source).div_euclid(2);
                    if (0..32).contains(&a) && (0..32).contains(&b) {
                        kernel += t.w * c[t.c][a as usize] * c[t.c][b as usize];
                    }
                }
                total += kernel;
                moment += source as f64 * kernel;
            }
            assert!((total - 2.).abs() < 1e-9, "lag{} response{}", lag, total);
            assert!(moment.abs() < 1e-9, "lag{} range centroid{}", lag, moment / total);
        }
    }
}

#[cfg(test)]
mod satellite_test {
    use super::*;
    use num_complex::Complex32;
    #[test]
    fn detects_injected_coded_doppler_echo() {
        let c = phases(std::path::Path::new("../m32-reference.bc")).unwrap();
        let n = 256;
        let np = 64;
        let mut state = 13579u64;
        let mut noise = || {
            state = state.wrapping_mul(6364136223846793005).wrapping_add(1);
            ((state >> 32) as u32 as f64 / u32::MAX as f64 - 0.5) as f32
        };
        let mut x: Vec<_> = (0..np * n)
            .map(|_| Complex32::new(noise(), noise()))
            .collect();
        for k in 0..64 {
            let v = C::from_polar(
                50. * c[16][k / 2],
                std::f64::consts::TAU * 17000. * k as f64 * DT,
            );
            x[16 * n + 80 + k] += Complex32::new(v.re as f32, v.im as f32);
        }
        let (m, s) = coded_screen::screen(&x, np, n, &c, 4);
        assert!(s.max_fraction > 0.95);
        assert!(s.candidates > 0);
        assert!(m[16 * n + 100] != 0);
        assert!(m[50 * n + 240] == 0);
    }
}
