//! Clamped cubic B-splines: model knots are independent of measurement gates.
use serde::{Deserialize, Serialize};
#[derive(Clone, Serialize, Deserialize)]
pub struct Spline {
    pub knots: Vec<f64>,
}
impl Spline {
    pub fn adaptive() -> Self {
        let mut breaks = vec![100.];
        let mut z = 100f64;
        while z < 1000. {
            let step = if z <= 400. {
                15.
            } else {
                15. + 135. * ((z - 400.) / 1600.)
            };
            z = (z + step).min(1000.);
            breaks.push(z);
        }
        let mut knots = vec![100.; 4];
        knots.extend(breaks[1..breaks.len() - 1].iter());
        knots.extend([1000.; 4]);
        Self { knots }
    }
    pub fn count(&self) -> usize {
        self.knots.len() - 4
    }
    pub fn controls(&self) -> Vec<f64> {
        (0..self.count())
            .map(|i| (self.knots[i + 1] + self.knots[i + 2] + self.knots[i + 3]) / 3.)
            .collect()
    }
    pub fn weights(&self, z: f64) -> Vec<(usize, f64)> {
        let n = self.count();
        let z = z.clamp(self.knots[3], self.knots[n]);
        if z == self.knots[n] {
            return vec![(n - 1, 1.)];
        }
        let mut b: Vec<f64> = self
            .knots
            .windows(2)
            .map(|t| if t[0] <= z && z < t[1] { 1. } else { 0. })
            .collect();
        for d in 1..=3 {
            for i in 0..b.len() - d {
                let a = self.knots[i + d] - self.knots[i];
                let c = self.knots[i + d + 1] - self.knots[i + 1];
                b[i] = (if a > 0. {
                    (z - self.knots[i]) / a * b[i]
                } else {
                    0.
                }) + (if c > 0. {
                    (self.knots[i + d + 1] - z) / c * b[i + 1]
                } else {
                    0.
                });
            }
        }
        (0..n)
            .filter_map(|i| if b[i] > 1e-15 { Some((i, b[i])) } else { None })
            .collect()
    }
}
pub fn h_weight(z: f64) -> f64 {
    let x = ((z - 500.) / 30.).clamp(0., 1.);
    x * x * x * (10. + x * (-15. + 6. * x))
}
#[cfg(test)]
mod tests {
    use super::*;
    #[test]
    fn partition_positive_and_linear_precision() {
        let s = Spline::adaptive();
        let c = s.controls();
        for i in 1000..10001 {
            let z = i as f64 / 10.;
            let w = s.weights(z);
            assert!(w.len() <= 4);
            assert!(w.iter().all(|(_, v)| *v >= 0.));
            assert!((w.iter().map(|(_, v)| v).sum::<f64>() - 1.).abs() < 1e-12);
            assert!((w.iter().map(|(i, v)| c[*i] * v).sum::<f64>() - z).abs() < 1e-9);
        }
    }
    #[test]
    fn increasing_step_and_no_topside_molecular_component() {
        let s = Spline::adaptive();
        assert!(s.knots.windows(2).all(|v| v[1] >= v[0]));
        assert_eq!(h_weight(499.), 0.);
        assert_eq!(h_weight(530.), 1.);
    }
}
