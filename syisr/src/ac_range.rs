//! Range-resolved decoded AC response, including FIR cross-lags before decoding.
//! A native 20-us source cell is integrated as two exact 10-us subintervals.
use num_complex::Complex64 as C;
use rayon::prelude::*;
pub const FIRST: i32 = -40;
pub const LAST: i32 = 40;
pub fn fir(noise: &[[f64; 2]]) -> Vec<C> {
    let mut rho = vec![C::default(); 11];
    for l in 0..11 {
        rho[l] = C::new(noise[l][0], noise[l][1]) / noise[0][0]
            * (0.5 + 0.5 * (std::f64::consts::PI * l as f64 / 11.).cos());
    }
    let psd: Vec<_> = (0..512)
        .map(|j| {
            let f = j as f64 / 512.;
            (1. + (1..11)
                .map(|l| 2. * (rho[l] * C::from_polar(1., std::f64::consts::TAU * f * l as f64)).re)
                .sum::<f64>())
            .max(1e-8)
        })
        .collect();
    (-10..=10)
        .map(|t| {
            (0..512)
                .map(|j| {
                    C::from_polar(
                        psd[j].sqrt() / 512.,
                        std::f64::consts::TAU * j as f64 * t as f64 / 512.,
                    )
                })
                .sum()
        })
        .collect()
}
pub fn stencil(codes: &[Vec<f64>], taps: &[C], lags: usize, frequency: &[f64]) -> Vec<Vec<Vec<C>>> {
    assert!(taps.len() % 2 == 1);
    let half = (taps.len() / 2) as i32;
    let mut gram = vec![0.; 1024 * 1024];
    gram.par_chunks_mut(1024).enumerate().for_each(|(p, row)| {
        for (q, v) in row.iter_mut().enumerate() {
            *v = (0..64)
                .map(|c| codes[c][p / 32] * codes[c][p % 32] * codes[c][q / 32] * codes[c][q % 32])
                .sum::<f64>()
                / 64.;
        }
    });
    (1..=lags)
        .into_par_iter()
        .map(|lag| {
            let branches: Vec<_> = [(0, lag / 2), (1, (lag + 1) / 2)]
                .into_iter()
                .filter(|(_, l)| *l > 0)
                .collect();
            (FIRST..=LAST)
                .map(|d| {
                    let mut co = vec![C::default(); 4 * half as usize + 1];
                    for (a, l) in &branches {
                        let factor = (if lag % 2 == 1 { 2. } else { 1. })
                            / (branches.len() * (32 - l)) as f64;
                        for fa in -half..=half {
                            for fb in -half..=half {
                                let tap =
                                    taps[(fa + half) as usize] * taps[(fb + half) as usize].conj();
                                if tap.norm() < 1e-20 {
                                    continue;
                                }
                                let mut ov = 0.;
                                for b in 0..32 - l {
                                    let p = b * 32 + b + l;
                                    for sub in 0..2 {
                                        let i =
                                            (2 * b as i32 + *a - fa - 2 * d - sub).div_euclid(2);
                                        let j = (2 * b as i32 + *a + lag as i32 - fb - 2 * d - sub)
                                            .div_euclid(2);
                                        if (0..32).contains(&i) && (0..32).contains(&j) {
                                            ov += 0.5 * gram[p * 1024 + (i * 32 + j) as usize];
                                        }
                                    }
                                }
                                co[(fa - fb + 2 * half) as usize] += tap * factor * ov;
                            }
                        }
                    }
                    frequency
                        .iter()
                        .map(|f| {
                            co.iter()
                                .enumerate()
                                .map(|(k, c)| {
                                    *c * C::from_polar(
                                        1.,
                                        -std::f64::consts::TAU
                                            * f
                                            * (lag as i32 + k as i32 - 2 * half) as f64
                                            * 1e-5,
                                    )
                                })
                                .sum()
                        })
                        .collect()
                })
                .collect()
        })
        .collect()
}
#[cfg(test)]
mod tests {
    use super::*;
    #[test]
    fn finite_cell_matches_explicit_code_integral() {
        let codes = crate::decode::phase(std::path::Path::new(concat!(
            env!("CARGO_MANIFEST_DIR"),
            "/m32-reference.bc"
        )))
        .unwrap();
        let taps = vec![C::new(0.2, 0.1), C::new(0.7, 0.), C::new(0.2, -0.1)];
        let f = [7000.];
        let k = stencil(&codes, &taps, 3, &f);
        for lag in [2usize, 3] {
            let branches: Vec<_> = [(0, lag / 2), (1, (lag + 1) / 2)]
                .into_iter()
                .filter(|(_, l)| *l > 0)
                .collect();
            for d in [-2i32, 0, 2] {
                let mut direct = C::default();
                for (a, l) in &branches {
                    let fact = (if lag % 2 == 1 { 2. } else { 1. })
                        / (branches.len() * 64 * (32 - l)) as f64;
                    for b in 0..32 - l {
                        for code in &codes {
                            let v = crate::exact_response::coded_cell(
                                (2 * b + *a) as f64 * 1e-5,
                                lag as f64 * 1e-5,
                                2e-5,
                                &code.iter().map(|v| C::new(*v, 0.)).collect::<Vec<_>>(),
                                [(2 * d - 1) as f64 * 1e-5, (2 * d + 1) as f64 * 1e-5],
                                &taps
                                    .iter()
                                    .enumerate()
                                    .map(|(i, c)| ((i as i32 - 1) as f64 * 1e-5, *c))
                                    .collect::<Vec<_>>(),
                                &f,
                            )
                            .unwrap();
                            direct += v[0] * 32. * fact * code[b] * code[b + l];
                        }
                    }
                }
                assert!(
                    (k[lag - 1][(d - FIRST) as usize][0] - direct).norm() < 1e-10,
                    "lag{lag} offset{d}"
                );
            }
        }
    }
}
