//! Independent physical-time voltage replay for the range-lag proof memo.
//! This reference does not use the interval endpoints or weights of Ambiguity.
use crate::profile_response::{Ambiguity, DR};
use anyhow::Result;
use num_complex::Complex64 as C;
use serde::Serialize;
pub fn nominal_range(index: usize) -> f64 {
    (2001. + 10. * (index as f64 + (21. - 200.) / 2.)) * 0.149896229
}
#[derive(Serialize)]
pub struct Replay {
    pub receive_index: usize,
    pub lag_us: f64,
    pub range_km: f64,
    pub forward: [f64; 2],
    pub direct: [f64; 2],
    pub relative_error: f64,
}
/// Piecewise independent local scatter populations have different strengths
/// and positive/negative frequencies. The direct integrator forms each local
/// filtered voltage before taking its product; unrelated ranges are never
/// added coherently. Midpoint quadrature is intentionally independent of the
/// native ambiguity-bin boundaries and analytic 1/R^2 integration.
pub fn replay(
    fir: &[(f64, C)],
    lag_indices: &[usize],
    points_per_cell: usize,
) -> Result<Vec<Replay>> {
    let indices = [180, 320, 510];
    let ranges = indices.map(nominal_range);
    let edges = [225.4, 451.7, 668.3, 977.9, 1250.6];
    let amplitudes = [0.2, 2.1, 0.6, 1.3];
    let frequencies = [-8700., 1900., 7300., -3200.];
    let terms = Ambiguity::new(fir)?.terms(&ranges, &edges, lag_indices, true)?;
    let mut prediction = vec![C::default(); indices.len() * lag_indices.len()];
    for (cell, list) in terms.iter().enumerate() {
        for t in list {
            let tau = (t.time as f64 - 40.) * 1e-5;
            prediction[t.measurement] += t.weight
                * C::from_polar(
                    amplitudes[cell],
                    -std::f64::consts::TAU * frequencies[cell] * tau,
                );
        }
    }
    let mut results = vec![];
    for (g, &index) in indices.iter().enumerate() {
        for (li, &lag) in lag_indices.iter().enumerate() {
            let mut direct = C::default();
            for cell in 0..4 {
                let dz = (edges[cell + 1] - edges[cell]) / points_per_cell as f64;
                let omega = std::f64::consts::TAU * frequencies[cell];
                let taps = fir
                    .iter()
                    .map(|&(a, h)| (a, h * C::from_polar(1., -omega * a)))
                    .collect::<Vec<_>>();
                let phase = C::from_polar(1., -omega * lag as f64 * 1e-5);
                for k in 0..points_per_cell {
                    let r = edges[cell] + (k as f64 + 0.5) * dz;
                    let delay = 2. * r / 299792.458;
                    let mut products = C::default();
                    for q in 0..lag + 21 {
                        // Actual header receive time, selected early/late indices,
                        // and explicit +half-sample convention used by the model.
                        let t1 =
                            (2001. + 10. * (index as f64 - lag as f64 + q as f64 + 0.5)) * 1e-6;
                        let t2 = t1 + lag as f64 * 1e-5;
                        let mut v1 = C::default();
                        let mut v2 = C::default();
                        for &(a, h) in &taps {
                            let u = t1 - a - delay;
                            let v = t2 - a - delay;
                            if (0.0..0.002).contains(&u) {
                                v1 += h;
                            }
                            if (0.0..0.002).contains(&v) {
                                v2 += h;
                            }
                        }
                        products += phase * v1 * v2.conj();
                    }
                    direct += products * amplitudes[cell] * (1000. / r).powi(2) * dz
                        / DR
                        / ((200 - lag) * (lag + 21)) as f64;
                }
            }
            let forward = prediction[g * lag_indices.len() + li];
            let error = (forward - direct).norm() / forward.norm().max(1e-8);
            results.push(Replay {
                receive_index: index,
                lag_us: lag as f64 * 10.,
                range_km: ranges[g],
                forward: [forward.re, forward.im],
                direct: [direct.re, direct.im],
                relative_error: error,
            });
        }
    }
    Ok(results)
}
/// Exact moments of the unfiltered trapezoidal response, with geometry inside
/// each native bin. Returns gain relative to centre-only spreading, centroid,
/// and G(lag)/G(0) for the requested lags, at one centre range.
pub fn geometry_moments(r: f64, lags: &[usize]) -> Vec<(f64, f64, f64)> {
    let calc = |l: usize| {
        let weights = crate::kernel(l);
        let mut m0 = 0.;
        let mut m1 = 0.;
        for (j, &w) in weights.iter().enumerate() {
            let u = r + (j as f64 - 110.) * DR;
            let v = u + DR;
            m0 += w * (1. / u - 1. / v) / DR;
            m1 += w * (v / u).ln() / DR;
        }
        (r * r * m0, m1 / m0)
    };
    let reference = calc(0).0;
    lags.iter()
        .map(|&l| {
            let (g, c) = calc(l);
            (g, c, g / reference)
        })
        .collect()
}
#[cfg(test)]
mod tests {
    use super::*;
    #[test]
    fn stratified_complex_scene_matches_independent_physical_time_replay() {
        let fir = [
            (-1e-5, C::new(0.2, 0.1)),
            (0., C::new(0.7, 0.)),
            (1e-5, C::new(0.2, -0.1)),
        ];
        let result = replay(&fir, &[1, 17, 60], 16000).unwrap();
        for r in result {
            assert!(
                r.relative_error < 8e-4,
                "range {} lag {} error {}",
                r.range_km,
                r.lag_us,
                r.relative_error
            );
        }
    }
    #[test]
    fn normalized_acf_retains_a_lag_dependent_spreading_factor() {
        let moments = geometry_moments(600., &[0, 30, 60]);
        assert!((moments[0].2 - 1.).abs() < 1e-12);
        assert!(moments[0].0 > moments[2].0);
        assert!(moments[2].2 < 0.99);
        assert!(moments.iter().all(|v| v.1 < 600.));
    }
}
