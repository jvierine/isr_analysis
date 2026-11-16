//! Exact discrete trapezoidal product selection with a rectangular TX pulse and
//! explicit voltage FIR. Physical-cell integrals include 1/R^2 BEFORE summation.
//! FIR timing is the same centered, uncalibrated-phase assumption as the pilot.
use anyhow::{ensure, Result};
use num_complex::Complex64 as C;
pub const DR: f64 = 1.49896229;
pub const OFFSET: isize = -150;
pub const BINS: usize = 300;
pub const DIFFERENCES: usize = 81;
pub const TIMES: usize = 141; // -400 .. +1000 us, includes filter cross-lags
#[derive(Clone, Debug)]
pub struct Term {
    pub measurement: usize,
    pub time: usize,
    pub weight: C,
}
pub struct Ambiguity {
    pub weights: Vec<C>,
}
impl Ambiguity {
    pub fn new(fir: &[(f64, C)]) -> Result<Self> {
        Self::for_gate(fir, 60)
    }
    /// Only recorded pairs: early voltage index i-l+q must be nonnegative.
    pub fn for_gate(fir: &[(f64, C)], native_index: usize) -> Result<Self> {
        Self::for_gate_from(fir, native_index, 0)
    }
    pub fn for_gate_from(
        fir: &[(f64, C)],
        native_index: usize,
        first_sample: usize,
    ) -> Result<Self> {
        ensure!(!fir.is_empty(), "empty receiver FIR");
        let taps: Vec<_> = fir
            .iter()
            .map(|&(s, h)| {
                let k = (s / 1e-5).round() as isize;
                ensure!(
                    (s - k as f64 * 1e-5).abs() < 1e-10
                        && k.abs() <= 20
                        && h.re.is_finite()
                        && h.im.is_finite(),
                    "FIR must use native +/-20 sample offsets"
                );
                Ok((k, h))
            })
            .collect::<Result<_>>()?;
        let mut weights = vec![C::default(); 61 * DIFFERENCES * BINS];
        for l in 0usize..61 {
            let mut diff = vec![C::default(); DIFFERENCES * (BINS + 1)];
            let first = (l + first_sample).saturating_sub(native_index);
            ensure!(
                first < l + 21,
                "no recorded stable pairs at gate {native_index}"
            );
            let scale = 1. / ((200 - l) * (l + 21 - first)) as f64;
            for &(a, ha) in &taps {
                for &(b, hb) in &taps {
                    let d = (a - b + 40) as usize;
                    for q in first..l + 21 {
                        // Memo 58 derivation: t_n=t_front+(n+eta)*dt,
                        // r(i)=c/2*[t_front+(i+eta+(nu-1-M)/2)*dt].
                        // Subtracting 2*r(i)/c from the two filtered receive
                        // times gives e1=q-l-a+(M-nu+1)/2 and e2=q-b+90.
                        // The current absolute range convention uses eta=1/2;
                        // eta cancels from these RELATIVE endpoints. Hardware
                        // timing/filter phase still require calibration.
                        let e1 = q as isize - l as isize + 90 - a;
                        let e2 = q as isize + 90 - b;
                        let lo = e1.max(e2) - 200 - OFFSET;
                        let hi = e1.min(e2) - OFFSET;
                        if hi <= lo {
                            continue;
                        }
                        ensure!(
                            lo >= 0 && hi <= BINS as isize,
                            "response exceeds allocated support"
                        );
                        // 1/[L Q (1-l/M)] = 1/[(M-l) Q dR]. The dR
                        // denominator is applied by terms() during integration.
                        let w = ha * hb.conj() * scale;
                        diff[d * (BINS + 1) + lo as usize] += w;
                        diff[d * (BINS + 1) + hi as usize] -= w;
                    }
                }
            }
            for d in 0..DIFFERENCES {
                let mut acc = C::default();
                for s in 0..BINS {
                    acc += diff[d * (BINS + 1) + s];
                    weights[(l * DIFFERENCES + d) * BINS + s] = acc;
                }
            }
        }
        Ok(Self { weights })
    }
    /// Contributions grouped by physical range cell. Calibration reference range
    /// is 1000 km: weights contain (1000/R)^2, with the integral analytic in R.
    pub fn terms(
        &self,
        ranges: &[f64],
        cell_edges: &[f64],
        lags: &[usize],
        spreading: bool,
    ) -> Result<Vec<Vec<Term>>> {
        self.terms_with_floor(ranges, cell_edges, lags, spreading, false)
    }
    /// Optional lower boundary has zero ionospheric scatter below the model floor.
    pub fn terms_with_floor(
        &self,
        ranges: &[f64],
        cell_edges: &[f64],
        lags: &[usize],
        spreading: bool,
        vacuum_below: bool,
    ) -> Result<Vec<Vec<Term>>> {
        ensure!(
            cell_edges.len() > 1
                && cell_edges[0] > 0.
                && cell_edges.windows(2).all(|z| z[1] > z[0]),
            "invalid physical range cells"
        );
        ensure!(lags.iter().all(|&l| l <= 60), "unsupported lag");
        let nc = cell_edges.len() - 1;
        let mut cells = vec![vec![]; nc];
        for (g, &r) in ranges.iter().enumerate() {
            for (li, &l) in lags.iter().enumerate() {
                let mut accum = vec![C::default(); nc * DIFFERENCES];
                for s in 0..BINS {
                    let mut lo = r + (OFFSET + s as isize) as f64 * DR;
                    let hi = lo + DR;
                    let nonzero = (0..DIFFERENCES)
                        .any(|d| self.weights[(l * DIFFERENCES + d) * BINS + s].norm() > 1e-14);
                    if !nonzero {
                        continue;
                    }
                    if vacuum_below {
                        if hi <= cell_edges[0] {
                            continue;
                        }
                        lo = lo.max(cell_edges[0]);
                    }
                    ensure!(
                        lo >= cell_edges[0] - 1e-8 && hi <= cell_edges[nc] + 1e-8,
                        "physical grid truncates response at {lo}..{hi} km"
                    );
                    let first = cell_edges
                        .partition_point(|&z| z <= lo)
                        .saturating_sub(1)
                        .min(nc - 1);
                    for c in first..nc {
                        if cell_edges[c] >= hi {
                            break;
                        }
                        let a = lo.max(cell_edges[c]);
                        let b = hi.min(cell_edges[c + 1]);
                        if b <= a {
                            continue;
                        }
                        let geo = if spreading {
                            1e6 * (1. / a - 1. / b) / DR
                        } else {
                            (b - a) / DR
                        };
                        for d in 0..DIFFERENCES {
                            accum[c * DIFFERENCES + d] +=
                                self.weights[(l * DIFFERENCES + d) * BINS + s] * geo;
                        }
                    }
                }
                for c in 0..nc {
                    for d in 0..DIFFERENCES {
                        let weight = accum[c * DIFFERENCES + d];
                        if weight.norm() > 1e-13 {
                            cells[c].push(Term {
                                measurement: g * lags.len() + li,
                                time: l + d,
                                weight,
                            });
                        }
                    }
                }
            }
        }
        Ok(cells)
    }
}
#[cfg(test)]
mod tests {
    #[test]
    fn partial_support_matches_direct_voltage_pairs_and_complete_limit() {
        use super::*;
        use crate::ambiguity_audit::{direct_point_with_partial, nominal_range, point_response};
        let fir = [
            (-1e-5, C::new(0.1, 0.02)),
            (0., C::new(0.95, 0.)),
            (1e-5, C::new(0.1, -0.02)),
        ];
        for i in [20usize, 40, 60, 85, 90] {
            let a = Ambiguity::for_gate_from(&fir, i, 30).unwrap();
            for l in [1usize, 30, 60] {
                for r in [91.123, 157.321, 205.543, 310.765] {
                    for k in l as isize - 2..=l as isize + 2 {
                        let want = crate::ambiguity_audit::direct_point_from_sample(
                            i,
                            l,
                            r,
                            k,
                            &fir,
                            Some(30),
                        );
                        let got = point_response(&a, nominal_range(i), l, r, k, true);
                        assert!((want - got).norm() < 1e-12);
                    }
                }
            }
        }
        assert_eq!(
            Ambiguity::for_gate_from(&fir, 90, 30).unwrap().weights,
            Ambiguity::new(&fir).unwrap().weights
        );
        for i in [0usize, 5, 20, 40, 60] {
            let a = Ambiguity::for_gate(&fir, i).unwrap();
            for l in [1usize, 20, 40, 60] {
                for r in [91.123, 157.321, 205.543, 310.765] {
                    for k in l as isize - 2..=l as isize + 2 {
                        let got = point_response(&a, nominal_range(i), l, r, k, true);
                        let want = direct_point_with_partial(i, l, r, k, &fir, true);
                        assert!(
                            (got - want).norm() < 1e-12,
                            "i={i}, l={l}, R={r}, intrinsic={k}: {got} vs {want}"
                        );
                    }
                }
            }
        }
        assert_eq!(
            Ambiguity::new(&fir).unwrap().weights,
            Ambiguity::for_gate(&fir, 60).unwrap().weights
        );
        for i in [0usize, 20, 40] {
            let a = Ambiguity::for_gate(&[(0., C::new(1., 0.))], i).unwrap();
            for l in 0..61 {
                let area: f64 = (0..BINS)
                    .map(|s| a.weights[(l * DIFFERENCES + 40) * BINS + s].re)
                    .sum();
                assert!((area - 1.).abs() < 1e-12);
            }
        }
    }
    use super::*;
    #[test]
    fn delta_fir_matches_stored_range_kernels_and_centres() {
        let a = Ambiguity::new(&[(0., C::new(1., 0.))]).unwrap();
        for l in 0..61 {
            let w = &a.weights[(l * 81 + 40) * BINS..(l * 81 + 41) * BINS];
            assert!((w.iter().map(|v| v.re).sum::<f64>() - 1.).abs() < 1e-12);
            let centroid: f64 = w
                .iter()
                .enumerate()
                .map(|(s, v)| v.re * (OFFSET as f64 + s as f64 + 0.5))
                .sum();
            assert!(centroid.abs() < 1e-10);
            for (j, v) in crate::kernel(l).iter().enumerate() {
                assert!((w[j + 40].re - v).abs() < 1e-12);
            }
        }
    }
    #[test]
    fn inverse_square_is_inside_integral_and_partition_invariant() {
        let a = Ambiguity::new(&[(0., C::new(1., 0.))]).unwrap();
        let coarse = a.terms(&[600.], &[350., 850.], &[0, 30, 60], true).unwrap();
        let fine = a
            .terms(
                &[600.],
                &(0..=100).map(|j| 350. + 5. * j as f64).collect::<Vec<_>>(),
                &[0, 30, 60],
                true,
            )
            .unwrap();
        for li in 0..3 {
            let c: C = coarse
                .iter()
                .flatten()
                .filter(|t| t.measurement == li)
                .map(|t| t.weight)
                .sum();
            let f: C = fine
                .iter()
                .flatten()
                .filter(|t| t.measurement == li)
                .map(|t| t.weight)
                .sum();
            assert!((c - f).norm() < 1e-12);
            // Jensen: averaging 1/R^2 differs from evaluating it at the centre.
            assert!(c.re > 1e6 / 600f64.powi(2) * 1.03);
        }
    }
    #[test]
    fn filtered_response_matches_direct_voltage_range_quadrature() {
        let fir = [
            (-1e-5, C::new(0.2, 0.1)),
            (0., C::new(0.7, 0.)),
            (1e-5, C::new(0.2, -0.1)),
        ];
        let a = Ambiguity::new(&fir).unwrap();
        let freq = 7300.;
        let l = 17;
        let terms = a.terms(&[700.], &[450., 950.], &[l], true).unwrap();
        let got: C = terms
            .iter()
            .flatten()
            .map(|t| {
                t.weight
                    * C::from_polar(
                        1.,
                        -std::f64::consts::TAU * freq * (t.time as f64 - 40.) * 1e-5,
                    )
            })
            .sum();
        let mut direct = C::default();
        let n = 100000;
        let dz = 500. / n as f64;
        for j in 0..n {
            let r = 450. + (j as f64 + 0.5) * dz;
            for q in 0..l + 21 {
                // Receive times expressed in sample units relative to the same centre.
                let t = q as f64 - l as f64 + 90.;
                let voltage = |time: f64| -> C {
                    fir.iter()
                        .map(|&(dt, h)| {
                            let u = time - dt / 1e-5 - (r - 700.) / DR;
                            if u >= 0. && u < 200. {
                                h * C::from_polar(
                                    1.,
                                    std::f64::consts::TAU * freq * (time * 1e-5 - dt),
                                )
                            } else {
                                C::default()
                            }
                        })
                        .sum()
                };
                direct += voltage(t) * voltage(t + l as f64).conj() * dz / DR * 1e6
                    / r.powi(2)
                    / ((200 - l) * (l + 21)) as f64;
            }
        }
        assert!((got - direct).norm() < 5e-5, "{got} versus {direct}");
    }
}
