//! BEATA 32-baud, two-samples-per-baud tail decoder geometry.
//! Derived from alter_dec/__user__ft_deco/__user__filter. Codes and input
//! covariances are explicit arguments; this module assumes no notch setting.
use crate::{noise::ProductTerm, Complex as C};
use anyhow::{ensure, Result};
#[derive(Clone, Debug)]
pub struct LagBlock {
    pub address: usize,
    pub lag_samples: usize,
    pub code_lag: usize,
    pub leading_zeros: usize,
    pub gates: usize,
}
pub fn blocks() -> Vec<LagBlock> {
    let mut out = Vec::new();
    let mut address = 480;
    for lag in 1..=41 {
        let below = lag / 2;
        let code_lags: Vec<_> = if lag % 2 == 0 {
            vec![below]
        } else if below == 0 {
            vec![1]
        } else {
            vec![below, below + 1]
        };
        for code_lag in code_lags {
            let gates = 432 - lag;
            out.push(LagBlock {
                address,
                lag_samples: lag,
                code_lag,
                leading_zeros: 2 * (31 - code_lag),
                gates,
            });
            address += gates;
        }
    }
    assert_eq!(address, 25531);
    out
}
impl LagBlock {
    /// q = sum c_b c_(b+L) z_(k+lag) conj(z_k), including the partial-code tail.
    /// The positive imaginary convention follows the online decoder, not its
    /// commented-out conjugate convention. No zero samples become observations.
    pub fn terms(&self, code: &[f64], gate: usize) -> Result<Vec<ProductTerm>> {
        ensure!(
            code.len() == 32 && code.iter().all(|x| x.abs() == 1.),
            "expected 32 binary code signs"
        );
        ensure!(gate < self.gates, "gate outside lag block");
        let mut terms = Vec::new();
        for b in 0..32 - self.code_lag {
            let padded = gate + 2 * b;
            if padded < self.leading_zeros {
                continue;
            }
            let k = padded - self.leading_zeros;
            ensure!(
                k + self.lag_samples < 432,
                "decoder reaches beyond received samples"
            );
            terms.push(ProductTerm {
                first: k + self.lag_samples,
                second: k,
                weight: C::new(code[b] * code[b + self.code_lag], 0.),
            });
        }
        ensure!(!terms.is_empty(), "empty measured gate");
        Ok(terms)
    }
}
#[cfg(test)]
mod tests {
    use super::*;
    #[test]
    fn full_layout_contains_all_recorded_lags_without_overlap() {
        let b = blocks();
        assert_eq!(b.len(), 61);
        assert_eq!(b[0].address, 480);
        assert_eq!(b[1].address, 911);
        let mut next = 480;
        for g in &b {
            assert_eq!(g.address, next);
            next += g.gates;
        }
        assert_eq!(next, 25531);
    }
    #[test]
    fn short_lag_tail_has_expected_number_of_products() {
        let g = &blocks()[0];
        let c = vec![1.; 32];
        for i in 0..431 {
            assert_eq!(g.terms(&c, i).unwrap().len(), (i / 2 + 1).min(31));
        }
    }
    #[test]
    fn every_lag_keeps_partial_gates_and_uses_valid_samples() {
        let c = (0..32)
            .map(|i| if i % 3 == 0 { -1. } else { 1. })
            .collect::<Vec<_>>();
        for g in blocks() {
            for i in 0..g.gates {
                let t = g.terms(&c, i).unwrap();
                for q in t {
                    assert_eq!(q.first - q.second, g.lag_samples);
                    assert!(q.first < 432);
                }
            }
        }
    }
}

/// Ideal rectangular-baud forward mean for a short-lag dump whose scattering
/// may change during its 64 code pulses. Columns are [time_bin, source_gate].
/// Signed off-range responses must survive: they cancel only for stationary
/// scattering. This is NOT the receiver-filtered production operator.
#[derive(Clone, Debug)]
pub struct ShortCycleResponse {
    pub source_gates: usize,
    pub time_bins: usize,
    pub rows: Vec<Vec<(usize, f64)>>,
}
impl ShortCycleResponse {
    pub fn rectangular(
        codes: &[Vec<f64>],
        observed_gates: usize,
        source_gates: usize,
        time_bins: usize,
    ) -> Result<Self> {
        ensure!(
            codes.len() == 64
                && observed_gates > 0
                && observed_gates <= 431
                && source_gates >= observed_gates
                && time_bins > 0
                && 64 % time_bins == 0,
            "unsupported short-cycle geometry"
        );
        let decoder = &blocks()[0];
        let mut rows = Vec::new();
        for r in 0..observed_gates {
            let mut row = std::collections::BTreeMap::<usize, f64>::new();
            for (c, code) in codes.iter().enumerate() {
                let terms = decoder.terms(code, r)?;
                let norm = (64 * terms.len()) as f64;
                let time = c * time_bins / 64;
                for term in terms {
                    for j in 0..63 {
                        let s = term.second as isize + 61 - j as isize;
                        if s >= 0 && s < source_gates as isize {
                            *row.entry(time * source_gates + s as usize).or_default() +=
                                term.weight.re * code[j / 2] * code[(j + 1) / 2] / norm;
                        }
                    }
                }
            }
            rows.push(row.into_iter().filter(|(_, w)| w.abs() > 1e-14).collect());
        }
        Ok(Self {
            source_gates,
            time_bins,
            rows,
        })
    }
    pub fn apply(&self, brightness: &[f64]) -> Result<Vec<f64>> {
        ensure!(
            brightness.len() == self.source_gates * self.time_bins
                && brightness.iter().all(|v| v.is_finite()),
            "invalid within-cycle brightness"
        );
        Ok(self
            .rows
            .iter()
            .map(|r| r.iter().map(|(i, w)| brightness[*i] * w).sum())
            .collect())
    }
    /// Adjoint for fitting, preserving the signed decoder weights.
    pub fn adjoint(&self, residual: &[f64]) -> Result<Vec<f64>> {
        ensure!(residual.len() == self.rows.len(), "invalid residual shape");
        let mut out = vec![0.; self.source_gates * self.time_bins];
        for (row, &e) in self.rows.iter().zip(residual) {
            for &(j, w) in row {
                out[j] += w * e;
            }
        }
        Ok(out)
    }
}
#[cfg(test)]
mod cycle_tests {
    use super::*;
    #[test]
    fn sparse_within_cycle_operator_matches_voltage_covariance_mean() {
        let codes: Vec<Vec<f64>> = (0..64)
            .map(|c| {
                (0..32)
                    .map(|b| {
                        if ((c + 3 * b) * (b + 1)) % 7 < 3 {
                            -1.
                        } else {
                            1.
                        }
                    })
                    .collect()
            })
            .collect();
        let n = 16;
        let source = 80;
        let bins = 8;
        let operator = ShortCycleResponse::rectangular(&codes, n, source, bins).unwrap();
        let x: Vec<_> = (0..bins * source)
            .map(|i| 0.1 + ((i * 7) % 19) as f64 / 3.)
            .collect();
        let actual = operator.apply(&x).unwrap();
        let mut expected = vec![0.; n];
        for (c, code) in codes.iter().enumerate() {
            let tx: Vec<_> = code.iter().flat_map(|&v| [C::new(v, 0.); 2]).collect();
            let mut covariance = nalgebra::DMatrix::zeros(n + 1, n + 1);
            for s in 0..source {
                let brightness = x[(c * bins / 64) * source + s];
                let acf: Vec<_> = (0..n + 1)
                    .map(|k| C::new(brightness / 0.9 * 0.9f64.powi(k as i32), 0.))
                    .collect();
                crate::voltage::add_range_cell(&mut covariance, &tx, s as isize - 61, &acf)
                    .unwrap();
            }
            for r in 0..n {
                let terms = blocks()[0].terms(code, r).unwrap();
                let norm = (64 * terms.len()) as f64;
                for t in terms {
                    expected[r] += (t.weight * covariance[(t.first, t.second)]).re / norm;
                }
            }
        }
        for (a, b) in actual.iter().zip(expected) {
            assert!((a - b).abs() < 1e-10);
        }
        let y: Vec<_> = (0..n).map(|i| i as f64 - 7.).collect();
        let lhs: f64 = actual.iter().zip(&y).map(|(a, b)| a * b).sum();
        let rhs: f64 = x
            .iter()
            .zip(operator.adjoint(&y).unwrap())
            .map(|(a, b)| a * b)
            .sum();
        assert!((lhs - rhs).abs() < 1e-9);
    }
}
