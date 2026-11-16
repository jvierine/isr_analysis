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

/// Piecewise-constant phase/amplitude code, with voltage filtering before
/// correlation. Unlike multiplying a decoded ACF by one filter response, this
/// retains cross-lag terms and the finite range/code boundaries. The scale is
/// normalized by total transmitted code energy (seconds times amplitude²).
pub fn coded_cell(
    t: f64,
    lag: f64,
    baud_s: f64,
    code: &[C],
    edges: [f64; 2],
    fir: &[(f64, C)],
    frequency_hz: &[f64],
) -> Result<Vec<C>> {
    ensure!(
        !code.is_empty()
            && baud_s.is_finite()
            && baud_s > 0.
            && code.iter().all(|z| z.re.is_finite() && z.im.is_finite()),
        "invalid transmitted code"
    );
    // Reuse timing/FIR validation without relying on its rectangular result.
    rectangular_cell(t, lag, baud_s * code.len() as f64, edges, fir, &[])?;
    ensure!(
        frequency_hz.iter().all(|f| f.is_finite()),
        "invalid frequency"
    );
    let energy = baud_s * code.iter().map(|z| z.norm_sqr()).sum::<f64>();
    ensure!(energy > 0., "zero transmitted energy");
    let mut out = vec![C::default(); frequency_hz.len()];
    for &(a, ha) in fir {
        for &(b, hb) in fir {
            let first = t - a;
            let second = t + lag - b;
            let mut overlap = C::default();
            for (u, &cu) in code.iter().enumerate() {
                let lo = (first - (u + 1) as f64 * baud_s).max(edges[0]);
                let hi = (first - u as f64 * baud_s).min(edges[1]);
                if hi <= lo {
                    continue;
                }
                // Only code intervals intersecting this receive interval contribute.
                let begin = ((second - hi) / baud_s).floor().max(0.) as usize;
                let end = ((second - lo) / baud_s).ceil().max(0.) as usize;
                for v in begin..end.min(code.len()) {
                    let length = (hi.min(second - v as f64 * baud_s)
                        - lo.max(second - (v + 1) as f64 * baud_s))
                    .max(0.);
                    overlap += cu * code[v].conj() * length;
                }
            }
            let weight = ha * hb.conj() * overlap / energy;
            let tau = lag + a - b;
            for (z, &f) in out.iter_mut().zip(frequency_hz) {
                *z += weight * C::from_polar(1., -std::f64::consts::TAU * f * tau);
            }
        }
    }
    Ok(out)
}
#[cfg(test)]
mod coded_tests {
    use super::*;
    #[test]
    fn constant_code_matches_rectangular_response() {
        let fir = [
            (-1e-5, C::new(0.2, 0.1)),
            (0., C::new(0.7, 0.)),
            (1e-5, C::new(0.2, -0.1)),
        ];
        let freq = [-12000., 0., 8000.];
        for lag in [0., 13e-6, 77e-6, 170e-6] {
            let coded = coded_cell(
                1e-3,
                lag,
                20e-6,
                &vec![C::new(1., 0.); 32],
                [500e-6, 900e-6],
                &fir,
                &freq,
            )
            .unwrap();
            let rect = rectangular_cell(1e-3, lag, 640e-6, [500e-6, 900e-6], &fir, &freq).unwrap();
            for (a, b) in coded.iter().zip(rect) {
                assert!((a - b).norm() < 1e-12);
            }
        }
    }
    #[test]
    fn coded_kernel_matches_direct_voltage_quadrature() {
        let code = [
            C::new(1., 0.),
            C::new(-1., 0.),
            C::new(0., 1.),
            C::new(1., 0.),
        ];
        let baud = 20e-6;
        let t = 123e-6;
        let lag = 13e-6;
        let edges = [43e-6, 104e-6];
        let fir = [
            (-3e-6, C::new(0.2, 0.1)),
            (0., C::new(0.7, 0.)),
            (4e-6, C::new(0.1, -0.1)),
        ];
        let f = 7000.;
        let got = coded_cell(t, lag, baud, &code, edges, &fir, &[f]).unwrap()[0];
        let samples = 61000;
        let step = (edges[1] - edges[0]) / samples as f64;
        let mut expected = C::default();
        let voltage = |time: f64, delay: f64| -> C {
            fir.iter()
                .map(|&(offset, h)| {
                    let q = time - offset - delay;
                    let index = (q / baud).floor() as isize;
                    let phase = if index >= 0 && (index as usize) < code.len() {
                        code[index as usize]
                    } else {
                        C::default()
                    };
                    h * phase * C::from_polar(1., std::f64::consts::TAU * f * (time - offset))
                })
                .sum()
        };
        for i in 0..samples {
            let delay = edges[0] + (i as f64 + 0.5) * step;
            expected += voltage(t, delay) * voltage(t + lag, delay).conj() * step / (4. * baud);
        }
        assert!((got - expected).norm() < 1e-9, "{got} vs {expected}");
    }
}
