//! Adapted from syisr/src/h_transition.rs at 0dbc990; same parameter bounds and shape.
//! Two-parameter H/O transition in an IRI-temperature diffusive-height coordinate.
use serde::{Deserialize, Serialize};
pub const CENTER_MIN: f64 = 550.;
pub const CENTER_MAX: f64 = 2000.;
pub const WIDTH_MIN: f64 = 20.;
pub const WIDTH_MAX: f64 = 300.;
#[derive(Clone, Serialize, Deserialize)]
pub struct Shape {
    pub altitude_km: Vec<f64>,
    pub scale_height_km: Vec<f64>,
    pub source: String,
    #[serde(default)]
    pub exponential_scale_km: Option<f64>,
}
impl Shape {
    pub fn validate(&self) -> anyhow::Result<()> {
        if let Some(l) = self.exponential_scale_km {
            anyhow::ensure!(
                l.is_finite() && l >= 500. && l <= 20000.,
                "exponential scale outside500..20000km"
            );
        }
        anyhow::ensure!(
            self.altitude_km.len() >= 2
                && self.altitude_km.len() == self.scale_height_km.len()
                && self
                    .altitude_km
                    .windows(2)
                    .all(|z| z[0].is_finite() && z[1] > z[0])
                && self
                    .scale_height_km
                    .iter()
                    .all(|h| h.is_finite() && *h > 0.)
                && self.scale_height_km.windows(2).all(|h| h[1] >= h[0]),
            "invalid monotone IRI scale-height table"
        );
        Ok(())
    }
    fn segment(&self, z: f64) -> usize {
        self.altitude_km
            .partition_point(|x| *x <= z)
            .saturating_sub(1)
            .min(self.altitude_km.len() - 2)
    }
    pub fn height(&self, z: f64) -> (f64, f64) {
        if z < self.altitude_km[0] {
            return (self.scale_height_km[0], 0.);
        }
        let i = self.segment(z);
        let slope = (self.scale_height_km[i + 1] - self.scale_height_km[i])
            / (self.altitude_km[i + 1] - self.altitude_km[i]);
        (
            self.scale_height_km[i] + slope * (z - self.altitude_km[i]),
            slope,
        )
    }
    /// Integral dz/H(z), exact for each piecewise-linear scale-height segment.
    pub fn coordinate(&self, z: f64) -> f64 {
        let first = self.altitude_km[0];
        if z <= first {
            return (z - first) / self.scale_height_km[0];
        }
        let mut sum = 0.;
        let mut a = first;
        while a < z {
            let i = self.segment(a + 1e-9);
            let b = if a >= *self.altitude_km.last().unwrap() {
                z
            } else {
                z.min(self.altitude_km[i + 1])
            };
            let (h, s) = self.height(a);
            sum += if s.abs() < 1e-12 {
                (b - a) / h
            } else {
                (s * (b - a) / h).ln_1p() / s
            };
            a = b;
        }
        sum
    }
    pub fn parameters(q: [f64; 2]) -> (f64, f64) {
        (
            CENTER_MIN + (CENTER_MAX - CENTER_MIN) * q[0],
            WIDTH_MIN * (q[1] * (WIDTH_MAX / WIDTH_MIN).ln()).exp(),
        )
    }
    pub fn encode(center: f64, width: f64) -> [f64; 2] {
        [
            (center - CENTER_MIN) / (CENTER_MAX - CENTER_MIN),
            (width / WIDTH_MIN).ln() / (WIDTH_MAX / WIDTH_MIN).ln(),
        ]
        .map(|v| v.clamp(0., 1.))
    }
    /// H fraction before the prescribed 500--530km onset; derivatives in bounded optimizer coordinates.
    pub fn evaluate(&self, z: f64, q: [f64; 2]) -> (f64, [f64; 2]) {
        let (center, width) = Self::parameters(q);
        let (h, s) = self.height(center);
        let delta = self.coordinate(z) - self.coordinate(center);
        let (u, dc) = if let Some(l) = self.exponential_scale_km {
            let d = z - center;
            (-l * (-d / l).exp_m1() / width, -(-d / l).exp() / width)
        } else {
            (delta * h / width, (-1. + delta * s) / width)
        };
        let f = if u >= 0. {
            1. / (1. + (-u).exp())
        } else {
            let e = u.exp();
            e / (1. + e)
        };
        let df = f * (1. - f);
        (
            f,
            [
                df * dc * (CENTER_MAX - CENTER_MIN),
                -df * u * (WIDTH_MAX / WIDTH_MIN).ln(),
            ],
        )
    }
    pub fn quantile_altitude(&self, q: [f64; 2], fraction: f64) -> Option<f64> {
        let mut a = 0.;
        let mut b = 10000.;
        if self.evaluate(a, q).0 * h_weight(a) > fraction
            || self.evaluate(b, q).0 * h_weight(b) < fraction
        {
            return None;
        }
        for _ in 0..70 {
            let mid = (a + b) / 2.;
            if self.evaluate(mid, q).0 * h_weight(mid) < fraction {
                a = mid;
            } else {
                b = mid;
            }
        }
        Some((a + b) / 2.)
    }
}
#[derive(Clone, Serialize, Deserialize)]
pub struct Transition {
    pub shape: Shape,
    pub normalized_parameters: [f64; 2],
    pub crossing50_altitude_km: f64,
    pub local_scale_width_km: f64,
    pub altitude10_km: Option<f64>,
    pub altitude90_km: Option<f64>,
    pub width10_to90_km: Option<f64>,
    pub center_bounds_km: [f64; 2],
    pub width_bounds_km: [f64; 2],
    pub model: String,
}
impl Transition {
    pub fn new(shape: Shape, q: [f64; 2]) -> Self {
        let (c, w) = Shape::parameters(q);
        let a = shape.quantile_altitude(q, 0.1);
        let b = shape.quantile_altitude(q, 0.9);
        let model = if let Some(l) = shape.exponential_scale_km {
            format!("H=sigmoid((L/w)*(1-exp(-(z-h50)/L))); exponential local scale height w*exp((z-h50)/L), L={l}km fixed; C2 onset500..530km; h50 crossing exact; upper asymptote sigmoid(L/w), not forced to1")
        } else {
            "IRI-height sigmoid with fixed monotone reference".into()
        };
        Self {
            shape,
            normalized_parameters: q,
            crossing50_altitude_km: c,
            local_scale_width_km: w,
            altitude10_km: a,
            altitude90_km: b,
            width10_to90_km: a.zip(b).map(|(a, b)| b - a),
            center_bounds_km: [CENTER_MIN, CENTER_MAX],
            width_bounds_km: [WIDTH_MIN, WIDTH_MAX],
            model,
        }
    }
}
#[cfg(test)]
mod tests {
    use super::*;
    fn shape() -> Shape {
        Shape {
            altitude_km: vec![500., 1000., 2000.],
            scale_height_km: vec![60., 90., 150.],
            source: "test".into(),
            exponential_scale_km: None,
        }
    }
    #[test]
    fn crossing_width_and_analytic_derivatives() {
        let s = shape();
        s.validate().unwrap();
        let q = Shape::encode(875., 90.);
        assert!((s.evaluate(875., q).0 - 0.5).abs() < 1e-12);
        for z in [550., 730., 1200., 1600.] {
            let (_, d) = s.evaluate(z, q);
            for k in 0..2 {
                let mut a = q;
                let mut b = q;
                a[k] += 1e-6;
                b[k] -= 1e-6;
                let fd = (s.evaluate(z, a).0 - s.evaluate(z, b).0) / 2e-6;
                assert!((fd - d[k]).abs() < 1e-7);
            }
        }
        let dt = 0.01;
        let slope = (s.evaluate(875. + dt, q).0 - s.evaluate(875. - dt, q).0) / (2. * dt);
        assert!((slope - 1. / 360.).abs() < 1e-9);
    }
    #[test]
    fn monotone_slowing_skew_and_quantiles() {
        let s = shape();
        let q = Shape::encode(900., 100.);
        let t = Transition::new(s.clone(), q);
        assert!(t.altitude90_km.unwrap() - 900. > 900. - t.altitude10_km.unwrap());
        let mut last = 0.;
        for z in 100..5000 {
            let f = s.evaluate(z as f64, q).0;
            assert!(f >= last && f <= 1.);
            last = f;
        }
        assert!((s.evaluate(t.altitude10_km.unwrap(), q).0 - 0.1).abs() < 1e-12);
    }
}

/// Fraction of the non-H plasma in a Gaussian He envelope.
/// q: peak (0--0.5), center (550--1600km), log sigma (50--500km).
pub fn helium(z: f64, q: [f64; 3]) -> (f64, [f64; 3]) {
    let a = 0.5 * q[0];
    let c = 550. + 1050. * q[1];
    let sigma = 50. * (q[2] * 10f64.ln()).exp();
    let u = (z - c) / sigma;
    let e = (-0.5 * u * u).exp();
    let f = a * e;
    (f, [0.5 * e, f * u / sigma * 1050., f * u * u * 10f64.ln()])
}
pub fn helium_encode(amplitude: f64, center: f64, sigma: f64) -> [f64; 3] {
    [
        (amplitude / 0.5).clamp(0., 1.),
        ((center - 550.) / 1050.).clamp(0., 1.),
        ((sigma / 50.).ln() / 10f64.ln()).clamp(0., 1.),
    ]
}
#[derive(Clone, Serialize, Deserialize)]
pub struct HeliumLayer {
    pub peak_non_h_fraction: f64,
    pub center_km: f64,
    pub sigma_km: f64,
    pub normalized_parameters: [f64; 3],
    pub amplitude_bounds: [f64; 2],
    pub center_bounds_km: [f64; 2],
    pub sigma_bounds_km: [f64; 2],
    pub model: String,
}
impl HeliumLayer {
    pub fn new(q: [f64; 3]) -> Self {
        Self{peak_non_h_fraction:0.5*q[0],center_km:550.+1050.*q[1],sigma_km:50.*(q[2]*10f64.ln()).exp(),normalized_parameters:q,amplitude_bounds:[0.,0.5],center_bounds_km:[550.,1600.],sigma_bounds_km:[50.,500.],model:"fHe=(1-fH)*A*exp[-0.5*((z-center)/sigma)^2]*prescribed_C2_onset500--530km; fO=1-fH-fHe".into()}
    }
}
#[cfg(test)]
mod helium_tests {
    use super::*;
    #[test]
    fn helium_derivatives_and_composition() {
        let q = helium_encode(0.1, 850., 150.);
        for z in [550., 730., 1100., 1500.] {
            let (f, d) = helium(z, q);
            for k in 0..3 {
                let mut a = q;
                let mut b = q;
                a[k] += 1e-6;
                b[k] -= 1e-6;
                assert!(((helium(z, a).0 - helium(z, b).0) / 2e-6 - d[k]).abs() < 1e-8);
            }
            for h in [0., 0.5, 0.99] {
                let he = (1. - h) * f;
                assert!(h + he <= 1.);
            }
        }
    }
}

/// Prescribed C2 onset, identical to the alternating-code implementation.
pub fn h_weight(z: f64) -> f64 {
    let x = ((z - 500.) / 30.).clamp(0., 1.);
    x * x * x * (10. + x * (-15. + 6. * x))
}
/// Global shape coordinates occupy existing H slots 0/1 and He slots 2/3/4.
pub const COLUMNS: [usize; 5] = [4, 10, 17, 23, 29];
impl Shape {
    /// Return H and He/(1-H), plus their derivatives in the five global coordinates.
    pub fn composition(&self, altitude: f64, x: &[f64]) -> ([f64; 2], [[f64; 5]; 2]) {
        let (h, dh) = self.evaluate(altitude, [x[4], x[10]]);
        let (he, dhe) = helium(altitude, [x[17], x[23], x[29]]);
        let w = h_weight(altitude);
        (
            [h * w, he * w],
            [
                [dh[0] * w, dh[1] * w, 0., 0., 0.],
                [0., 0., dhe[0] * w, dhe[1] * w, dhe[2] * w],
            ],
        )
    }
    /// Propagate full conditional parameter covariance to physical H and He.
    /// Nonzero derivatives touching an unidentified/active-bound direction remain NaN.
    pub fn fractions(
        &self,
        altitude: f64,
        x: &[f64],
        covariance: Option<&[f64]>,
    ) -> ([f64; 3], [f64; 2]) {
        let (p, d) = self.composition(altitude, x);
        let dh = d[0];
        let dhe: Vec<_> = (0..5)
            .map(|i| (1. - p[0]) * d[1][i] - p[1] * d[0][i])
            .collect();
        let variance = |g: &[f64]| {
            let mut v = 0.;
            for a in 0..5 {
                for b in 0..5 {
                    if g[a] * g[b] != 0. {
                        v += g[a]
                            * g[b]
                            * covariance
                                .map(|c| c[COLUMNS[a] * x.len() + COLUMNS[b]])
                                .unwrap_or(f64::NAN);
                    }
                }
            }
            v.sqrt()
        };
        (
            [p[0], (1. - p[0]) * p[1], (1. - p[0]) * (1. - p[1])],
            [variance(&dh), variance(&dhe)],
        )
    }
}
pub fn from_metadata(m: &serde_json::Value) -> anyhow::Result<Option<Shape>> {
    m.get("composition_shape")
        .filter(|v| !v.is_null())
        .map(|v| {
            let shape: Shape = serde_json::from_value(v.clone())?;
            shape.validate()?;
            Ok(shape)
        })
        .transpose()
}

#[cfg(test)]
mod propagation_tests {
    use super::*;
    #[test]
    fn onset_simplex_and_full_cross_covariance() {
        let shape = Shape {
            altitude_km: vec![500., 1000., 2000.],
            scale_height_km: vec![60., 90., 150.],
            source: "test".into(),
            exponential_scale_km: None,
        };
        let mut x = vec![0.; 30];
        let h = Shape::encode(900., 100.);
        let he = helium_encode(0.15, 800., 160.);
        for (i, v) in COLUMNS.into_iter().zip([h[0], h[1], he[0], he[1], he[2]]) {
            x[i] = v;
        }
        let mut cov = vec![f64::NAN; 900];
        for (j, &a) in COLUMNS.iter().enumerate() {
            for (k, &b) in COLUMNS.iter().enumerate() {
                cov[a * 30 + b] = if j == k { 0.001 } else { 0.0002 };
            }
        }
        for z in [
            200., 499., 500., 515., 530., 750., 950., 2000., 5000., 5500.,
        ] {
            let (f, e) = shape.fractions(z, &x, Some(&cov));
            assert!((f.iter().sum::<f64>() - 1.).abs() < 1e-12);
            assert!(f.iter().all(|v| *v >= 0. && *v <= 1.));
            if z <= 500. {
                assert_eq!(f, [0., 0., 1.]);
                assert_eq!(e, [0., 0.]);
            }
            let mut d = [[0.; 5]; 2];
            for (j, &col) in COLUMNS.iter().enumerate() {
                let mut a = x.clone();
                let mut b = x.clone();
                a[col] += 1e-6;
                b[col] -= 1e-6;
                let fa = shape.fractions(z, &a, None).0;
                let fb = shape.fractions(z, &b, None).0;
                for k in 0..2 {
                    d[k][j] = (fa[k] - fb[k]) / 2e-6;
                }
            }
            for k in 0..2 {
                let mut var = 0.;
                for a in 0..5 {
                    for b in 0..5 {
                        var += d[k][a] * d[k][b] * cov[COLUMNS[a] * 30 + COLUMNS[b]];
                    }
                }
                assert!((var.sqrt() - e[k]).abs() < 1e-9);
            }
        }
        cov[4 * 30 + 4] = f64::NAN;
        assert!(shape.fractions(900., &x, Some(&cov)).1[0].is_nan());
    }
}

#[cfg(test)]
mod exponential_tests {
    use super::*;
    #[test]
    fn exponential_scale_sigmoid_slope_and_derivatives() {
        let s = Shape {
            altitude_km: vec![500., 2000.],
            scale_height_km: vec![100., 100.],
            source: "test".into(),
            exponential_scale_km: Some(1500.),
        };
        s.validate().unwrap();
        let q = Shape::encode(900., 100.);
        assert!((s.evaluate(900., q).0 - 0.5).abs() < 1e-12);
        let slope = (s.evaluate(900.001, q).0 - s.evaluate(899.999, q).0) / 0.002;
        assert!((slope - 1. / 400.).abs() < 1e-10);
        let mut last = 0.;
        for z in 500..5500 {
            let (f, d) = s.evaluate(z as f64, q);
            assert!(f >= last && f <= 1.);
            last = f;
            for k in 0..2 {
                let mut a = q;
                let mut b = q;
                a[k] += 1e-6;
                b[k] -= 1e-6;
                assert!(
                    ((s.evaluate(z as f64, a).0 - s.evaluate(z as f64, b).0) / 2e-6 - d[k]).abs()
                        < 1e-7
                );
            }
        }
        assert!(s.evaluate(1400., q).0 - 0.5 < 0.5 - s.evaluate(400., q).0);
    }
}
