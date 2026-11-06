//! Range-coupled rectangular-pulse response with an explicit voltage FIR.
use crate::Complex as C;
use anyhow::{ensure, Result};
/// Integrate the range-cell response for x(t) conj(x(t+lag)). `t`, `lag`,
/// width, delay edges and FIR offsets are all seconds. FIR offsets may be
/// negative for a centered, group-delay-corrected receiver response. The
/// caller supplies the measured/assumed FIR and its timing provenance.
/// Weights are normalized by pulse width, but contain no radar calibration.
pub fn rectangular_cell(
    t: f64,
    lag: f64,
    width: f64,
    edges: [f64; 2],
    fir: &[(f64, C)],
    frequency_hz: &[f64],
) -> Result<Vec<C>> {
    ensure!(
        t.is_finite()
            && lag.is_finite()
            && lag >= 0.
            && width.is_finite()
            && width > 0.
            && edges.iter().all(|x| x.is_finite())
            && edges[1] > edges[0],
        "invalid pulse/range timing"
    );
    ensure!(
        !fir.is_empty()
            && fir
                .iter()
                .all(|(t, z)| t.is_finite() && z.re.is_finite() && z.im.is_finite())
            && frequency_hz.iter().all(|x| x.is_finite()),
        "invalid FIR/frequency grid"
    );
    let mut out = vec![C::default(); frequency_hz.len()];
    for &(a, ha) in fir {
        for &(b, hb) in fir {
            let first = t - a;
            let second = t + lag - b;
            let length = (first.min(second).min(edges[1])
                - (first - width).max(second - width).max(edges[0]))
            .max(0.);
            if length == 0. {
                continue;
            }
            let weight = ha * hb.conj() * (length / width);
            let tau = lag + a - b;
            for (z, &f) in out.iter_mut().zip(frequency_hz) {
                *z += weight * C::from_polar(1., -std::f64::consts::TAU * f * tau);
            }
        }
    }
    Ok(out)
}
#[cfg(test)]
mod tests {
    use super::*;
    #[test]
    fn partition_matches_uniform_filtered_profile() {
        let fir = [
            (-1e-5, C::new(0.2, 0.1)),
            (0., C::new(0.7, 0.)),
            (1e-5, C::new(0.2, -0.1)),
        ];
        let f = [-12000., 0., 8000.];
        let width = 480e-6;
        let lag = 170e-6;
        let mut sum = vec![C::default(); 3];
        for i in 0..100 {
            for (z, v) in sum.iter_mut().zip(
                rectangular_cell(
                    1e-3,
                    lag,
                    width,
                    [i as f64 * 20e-6, (i + 1) as f64 * 20e-6],
                    &fir,
                    &f,
                )
                .unwrap(),
            ) {
                *z += v;
            }
        }
        for (i, &freq) in f.iter().enumerate() {
            let mut expected = C::default();
            for &(a, ha) in &fir {
                for &(b, hb) in &fir {
                    let tau = lag + a - b;
                    expected += ha
                        * hb.conj()
                        * (1. - tau.abs() / width).max(0.)
                        * C::from_polar(1., -std::f64::consts::TAU * freq * tau);
                }
            }
            assert!((sum[i] - expected).norm() < 1e-12);
        }
    }
    #[test]
    fn finite_cell_keeps_range_coupling() {
        let k = rectangular_cell(
            1e-3,
            100e-6,
            480e-6,
            [700e-6, 800e-6],
            &[(0., C::new(1., 0.))],
            &[0.],
        )
        .unwrap();
        assert!((k[0].re - 100. / 480.).abs() < 1e-12);
    }
}
