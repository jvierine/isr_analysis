// Magnetized number-conserving BGK thermal response and collisionless SAMI2-PE hot response.
// Full complex hot susceptibility is supplied by cache_complex.inc.rs.
use anyhow::{ensure, Result};
use errorfunctions::ComplexErrorFunctions;
use num_complex::Complex64 as C;
use rayon::prelude::*;
use serde_json::json;
use std::{f64::consts::PI, fs};
const E: f64 = 1.602176634e-19;
const ME: f64 = 9.1093837139e-31;
const KB: f64 = 1.380649e-23;
const EPS: f64 = 8.8541878188e-12;
const LIGHT: f64 = 299792458.;
const ETA: f64 = 1e-4;
const DF: f64 = 250.;
const FMIN: f64 = 0.1e6;
const NF: usize = 23600;
#[derive(Clone)]
struct Core {
    ne: f64,
    te: f64,
    b: f64,
    kp: f64,
    kt: f64,
    k: f64,
    nu: f64,
    weights: Vec<(i32, f64)>,
}
impl Core {
    fn new(ne: f64, te: f64, b: f64, alpha: f64, radar: f64) -> Self {
        let k = 4. * PI * radar / LIGHT;
        let kp = k * alpha.to_radians().sin();
        let kt = k * alpha.to_radians().cos();
        let gyro = E * b / ME;
        let v = (KB * te / ME).sqrt();
        let bb = (kt * v / gyro).powi(2);
        let weights = (-32_i32..=32)
            .map(|n| (n, scaled_in(n.unsigned_abs() as usize, bb)))
            .collect();
        let ld = (EPS * KB * te / (ne * E * E)).sqrt();
        // Same scalar Coulomb-rate estimate as the recovered Longley reproduction.
        // The BGK operator remains a surrogate, not a full Coulomb collision operator.
        let nu = (PI / 9.).sqrt() * ne * E.powi(4)
            / (4. * 2_f64.sqrt() * PI * EPS.powi(2) * ME.powi(2) * v.powi(3))
            * (4. * PI * EPS * KB * te * ld / (E * E)).ln();
        let multiplier = std::env::var("SYISR_NU_MULTIPLIER")
            .ok()
            .map(|v| v.parse::<f64>().expect("invalid collision multiplier"))
            .unwrap_or(1.);
        assert!(multiplier > 0.);
        Self {
            ne,
            te,
            b,
            kp,
            kt,
            k,
            nu: nu * multiplier,
            weights,
        }
    }
    fn response(&self, f: f64) -> (C, f64) {
        let w = 2. * PI * f;
        let a = 2_f64.sqrt() * self.kp * (KB * self.te / ME).sqrt();
        let gyro = E * self.b / ME;
        let mut j = C::new(0., 0.);
        for &(n, weight) in &self.weights {
            if weight > 1e-20 {
                j += weight * PI.sqrt() / a * C::new((w - n as f64 * gyro) / a, self.nu / a).w();
            }
        }
        j /= C::new(1., 0.) - self.nu * j;
        let aa = self.ne * E * E / (EPS * KB * self.te * self.k * self.k);
        let chi = aa * (C::new(1., 0.) + C::new(0., w) * j);
        (C::new(1., 0.) + chi, j.re)
    }
    fn roots(&self, step: f64) -> Vec<f64> {
        let mut roots = vec![];
        let mut lo = FMIN;
        let stop = FMIN + NF as f64 * DF + 0.2e6;
        let mut last = self.response(lo).0.re;
        while lo < stop {
            let hi = lo + step;
            let next = self.response(hi).0.re;
            // Only positive-slope, weakly damped resonances have positive mode energy.
            if last < 0. && next >= 0. {
                let (mut l, mut r) = (lo, hi);
                for _ in 0..44 {
                    let m = (l + r) / 2.;
                    if self.response(m).0.re < 0. {
                        l = m;
                    } else {
                        r = m;
                    }
                }
                roots.push((l + r) / 2.);
            }
            lo = hi;
            last = next;
        }
        roots
    }
}
fn scaled_in(n: usize, x: f64) -> f64 {
    let mut term = 1.;
    for k in 1..=n {
        term *= x / 2. / k as f64;
    }
    let mut sum = term;
    for m in 1..200 {
        term *= x * x / 4. / (m * (m + n)) as f64;
        sum += term;
        if term.abs() < sum.abs() * 1e-16 {
            break;
        }
    }
    sum * (-x).exp()
}
fn beam_location(r: f64) -> (f64, f64, f64) {
    let re = 6371.;
    let el = 60_f64.to_radians();
    let az = 358_f64.to_radians();
    let lat0 = 18.3_f64.to_radians();
    let lon0 = 109.6_f64.to_radians();
    let h = (re * re + r * r + 2. * re * r * el.sin()).sqrt() - re;
    let d = (r * el.cos()).atan2(re + r * el.sin());
    let lat = (lat0.sin() * d.cos() + lat0.cos() * d.sin() * az.cos()).asin();
    let lon = lon0 + (az.sin() * d.sin() * lat0.cos()).atan2(d.cos() - lat0.sin() * lat.sin());
    (h, lat.to_degrees(), lon.to_degrees())
}
fn range_for_altitude(h: f64) -> f64 {
    let re = 6371.;
    let s = 60_f64.to_radians().sin();
    -(re * s) + (re * re * s * s + 2. * re * h + h * h).sqrt()
}
fn neutral_collision(r: f64, te: f64) -> (f64, f64, f64, f64) {
    let (h, lat, lon) = beam_location(r);
    let input = rmsis::Input::consistent(
        260,
        81120.,
        h,
        lat,
        lon,
        149.,
        140.4,
        rmsis::MagneticActivity::Daily { ap: 14. },
    );
    let n = rmsis::Model::new().gtd7(&input);
    let en = 2.33e-17 * n.molecular_nitrogen * te * (1. - 1.21e-4 * te)
        + 1.82e-16 * n.molecular_oxygen * te.sqrt() * (1. + 0.036 * te.sqrt())
        + 8.9e-17 * n.atomic_oxygen * te.sqrt() * (1. + 5.7e-4 * te);
    assert!(en >= 0. && en.is_finite());
    (
        en,
        n.molecular_nitrogen,
        n.molecular_oxygen,
        n.atomic_oxygen,
    )
}
