// Adapted from the previous SYISR H+/He+/O+ Rust theory; geometry supplied
// from IGRF-14 for this observation date, not the old hardcoded 2023 epoch.
//! Zenith topside Maxwellian mixture: exact magnetized electron response;
//! ions use the large-k*rho unmagnetized limit, checked separately.
use anyhow::{ensure, Result};
use errorfunctions::ComplexErrorFunctions;
use num_complex::Complex64 as C;
use rayon::prelude::*;
use std::f64::consts::PI;
const KB: f64 = 1.380649e-23;
const ME: f64 = 9.1093837139e-31;
const AMU: f64 = 1.66053906892e-27;
const E: f64 = 1.602176634e-19;
const EPS: f64 = 8.8541878188e-12;
pub struct Theory {
    pub f: Vec<f64>,
    pub k: f64,
    pub b: f64,
    pub alpha: f64,
    pub h_temperature_ratio: f64,
    pub electron_magnetized: bool,
    pub include_electron_fluctuations: bool,
    eg: Vec<Vec<C>>,
    temps: Vec<f64>,
    cache_step_hz: f64,
    cache_half: usize,
}
fn scaled_i(n: usize, x: f64) -> f64 {
    let mut term = 1.;
    for k in 1..=n {
        term *= x / (2. * k as f64);
    }
    let mut sum = term;
    for m in 1..300 {
        term *= x * x / (4. * m as f64 * (m + n) as f64);
        sum += term;
        if term.abs() < sum.abs() * 1e-16 {
            break;
        }
    }
    sum * (-x).exp()
}
fn electron_g(k: f64, b: f64, alpha: f64, t: f64, f: f64) -> C {
    let v = (KB * t / ME).sqrt();
    let kp = k * alpha.to_radians().sin();
    let kt = k * alpha.to_radians().cos();
    let gyro = E * b / ME;
    let bb = (kt * v / gyro).powi(2);
    let a = 2_f64.sqrt() * kp * v;
    (-40..=40)
        .map(|n: i32| {
            scaled_i(n.unsigned_abs() as usize, bb) * PI.sqrt() / a
                * C::new((2. * PI * f - n as f64 * gyro) / a, 0.).w()
        })
        .sum()
}
impl Theory {
    pub fn at_field(b: f64, alpha: f64, df: f64) -> Result<Self> {
        ensure!(alpha > 3., "collisionless electron response is not validated within 3 degrees of perpendicular B (aspect={alpha})");
        let k = 4. * PI * 430.2e6 / 299792458.;
        let temps: Vec<_> = (0..=200)
            .map(|j| 150. * (20000_f64 / 150.).powf(j as f64 / 200.))
            .collect();
        let cache_step_hz = if alpha < 15. { 100. } else { 500. };
        let cache_half = (60000_f64 / cache_step_hz) as usize;
        let eg = temps
            .par_iter()
            .map(|&t| {
                let v = (KB * t / ME).sqrt();
                let gyro = E * b * 1e-9 / ME;
                let a = 2_f64.sqrt() * k * alpha.to_radians().sin() * v;
                let bb = (k * alpha.to_radians().cos() * v / gyro).powi(2);
                let weights: Vec<_> = (-40_i32..=40)
                    .map(|n| (n, scaled_i(n.unsigned_abs() as usize, bb)))
                    .collect();
                (-(cache_half as i32)..=cache_half as i32)
                    .map(|j| {
                        weights
                            .iter()
                            .filter(|(_, w)| *w > 1e-20)
                            .map(|(n, w)| {
                                w * PI.sqrt() / a
                                    * C::new(
                                        (2. * PI * j as f64 * cache_step_hz - *n as f64 * gyro) / a,
                                        0.,
                                    )
                                    .w()
                            })
                            .sum()
                    })
                    .collect()
            })
            .collect();
        let n = (50000. / df).round() as i32;
        Ok(Self {
            f: (-n..=n).map(|j| j as f64 * df).collect(),
            k,
            b: b * 1e-9,
            alpha,
            h_temperature_ratio: 1.,
            electron_magnetized: true,
            include_electron_fluctuations: true,
            eg,
            temps,
            cache_step_hz,
            cache_half,
        })
    }
    fn g(&self, t: f64, f: f64) -> C {
        let qt = ((t / 150.).ln() / (20000_f64 / 150.).ln() * 200.).clamp(0., 199.999999);
        let j = qt.floor() as usize;
        let u = qt - j as f64;
        let qf = (f / self.cache_step_hz + self.cache_half as f64)
            .clamp(0., (2 * self.cache_half) as f64 - 1e-6);
        let i = qf.floor() as usize;
        let v = qf - i as f64;
        let at = |j: usize| self.eg[j][i] * (1. - v) + self.eg[j][i + 1] * v;
        at(j) * (1. - u) + at(j + 1) * u
    }
    pub fn spectrum(
        &self,
        ti: f64,
        te: f64,
        velocity: f64,
        h: f64,
        he_other: f64,
        ne: f64,
    ) -> Vec<f64> {
        self.spectrum_with_mass(ti, te, velocity, h, he_other, ne).0
    }
    /// Normalized spectrum and its finite-band unnormalized integral. The latter
    /// retains the temperature/composition dependence of scattered power per
    /// electron for joint density-profile inversion. Legacy shape fits are unchanged.
    pub fn spectrum_with_mass(
        &self,
        ti: f64,
        te: f64,
        velocity: f64,
        h: f64,
        he_other: f64,
        ne: f64,
    ) -> (Vec<f64>, f64) {
        let fractions = [h, (1. - h) * he_other, (1. - h) * (1. - he_other)];
        let alpha = ne * E * E / (EPS * KB * self.k * self.k);
        let mut s: Vec<f64> = self
            .f
            .iter()
            .map(|&f| {
                let omega = 2. * PI * f - self.k * velocity;
                let ge = if self.electron_magnetized {
                    self.g(te, omega / (2. * PI))
                } else {
                    let a = self.k * (2. * KB * te / ME).sqrt();
                    PI.sqrt() / a * C::new(omega / a, 0.).w()
                };
                let ce = (C::new(1., 0.) + C::new(0., omega) * ge) * (alpha / te);
                let (mut ci, mut fi) = (C::new(0., 0.), 0.);
                for (mass, fraction) in [1., 4., 16.].iter().zip(fractions) {
                    let temperature = if *mass == 1. {
                        ti * self.h_temperature_ratio
                    } else {
                        ti
                    };
                    let a = self.k * (2. * KB * temperature / (mass * AMU)).sqrt();
                    let gi = PI.sqrt() / a * C::new(omega / a, 0.).w();
                    ci += (C::new(1., 0.) + C::new(0., omega) * gi)
                        * (alpha * fraction / temperature);
                    fi += fraction * 2. * gi.re;
                }
                let eps = C::new(1., 0.) + ce + ci;
                ((C::new(1., 0.) + ci).norm_sqr()
                    * 2.
                    * ge.re
                    * (if self.include_electron_fluctuations {
                        1.
                    } else {
                        0.
                    })
                    + ce.norm_sqr() * fi)
                    / eps.norm_sqr()
            })
            .collect();
        let df = self.f[1] - self.f[0];
        let sum: f64 = s
            .iter()
            .enumerate()
            .map(|(i, v)| v * df * if i == 0 || i + 1 == s.len() { 0.5 } else { 1. })
            .sum();
        for v in &mut s {
            *v /= sum;
        }
        (s, sum)
    }
    pub fn validate(&self) -> Result<serde_json::Value> {
        let mut max = 0_f64;
        for t in [300., 1000., 4000., 12000.] {
            for f in [-40000., -10000., 0., 12500., 45000.] {
                let a = self.g(t, f);
                let b = electron_g(self.k, self.b, self.alpha, t, f);
                max = max.max((a - b).norm() / b.norm());
            }
        }
        ensure!(max < 0.002, "electron cache interpolation too coarse {max}");
        let ion_error = self.ion_limit_check();
        ensure!(
            ion_error < 0.003,
            "ion large-k*rho approximation too inaccurate {ion_error}"
        );
        Ok(
            serde_json::json!({"max_ion_G_absolute_error_over_G0":ion_error,"magnetic_offset_from_perpendicular_deg":self.alpha,"field_t":self.b,"max_electron_cache_relative_error":max,"temperature_nodes":self.temps.len(),"ion_limit":"unmagnetized large k*rho; only zenith topside in this pilot","collision_model":"collisionless topside approximation; no fitted collision rate"}),
        )
    }
    fn ion_limit_check(&self) -> f64 {
        let mut max = 0_f64;
        for mass in [1., 4., 16.] {
            for temp in [300., 1000., 3000.] {
                let v = (KB * temp / (mass * AMU)).sqrt();
                let gyro = E * self.b / (mass * AMU);
                let kp = self.k * self.alpha.to_radians().sin();
                let kt = self.k * self.alpha.to_radians().cos();
                let b = (kt * v / gyro).powi(2);
                let steps = 4000;
                let dt = 12. / (kp * v) / steps as f64;
                let a = 2_f64.sqrt() * self.k * v;
                let g0 = PI.sqrt() / a;
                for f in [0., 1000., 5000., 15000., 30000.] {
                    let omega = 2. * PI * f;
                    let mut g = C::new(0., 0.);
                    for j in 0..=steps {
                        let t = j as f64 * dt;
                        let w = if j == 0 || j == steps {
                            1.
                        } else if j % 2 == 0 {
                            2.
                        } else {
                            4.
                        };
                        g += C::from_polar(
                            w * (-0.5 * (kp * v * t).powi(2) - b * (1. - (gyro * t).cos())).exp(),
                            omega * t,
                        );
                    }
                    g *= dt / 3.;
                    let unmag = PI.sqrt() / a * C::new(omega / a, 0.).w();
                    max = max.max((g - unmag).norm() / g0);
                }
            }
        }
        max
    }
}
