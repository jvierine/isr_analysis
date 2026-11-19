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
    /// Fixed O+, NO+, O2+ fractions within the heavy-ion population.
    pub heavy_ion_fractions: [f64; 4],
    pub electron_magnetized: bool,
    pub include_electron_fluctuations: bool,
    near_perpendicular: Option<(f64, f64)>,
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
    pub fn new(height: f64, df: f64) -> Result<Self> {
        let bvec = crate::geomagnetic::igrf14(2025.76, 18.3492, 109.6222, height)?;
        let b = bvec.iter().map(|v| v * v).sum::<f64>().sqrt();
        Self::at_field(b, (bvec[2].abs() / b).asin().to_degrees(), df)
    }
    /// Monostatic ray, azimuth clockwise from north, elevation above horizon.
    /// Spherical-Earth ray geometry matches the archived range/altitude convention.
    pub fn from_beam(range_km: f64, azimuth_deg: f64, elevation_deg: f64, df: f64) -> Result<Self> {
        let (b, alpha) = Self::beam_field(range_km, azimuth_deg, elevation_deg)?;
        Self::at_field(b, alpha, df)
    }
    pub fn beam_field(range_km: f64, azimuth_deg: f64, elevation_deg: f64) -> Result<(f64, f64)> {
        Self::beam_field_at_year(range_km, azimuth_deg, elevation_deg, 2025.76)
    }
    pub fn from_beam_at_year(
        range_km: f64,
        azimuth_deg: f64,
        elevation_deg: f64,
        df: f64,
        year: f64,
    ) -> Result<Self> {
        let (b, a) = Self::beam_field_at_year(range_km, azimuth_deg, elevation_deg, year)?;
        Self::at_field(b, a, df)
    }
    pub fn beam_field_at_year(
        range_km: f64,
        azimuth_deg: f64,
        elevation_deg: f64,
        year: f64,
    ) -> Result<(f64, f64)> {
        let lat = 18.3492_f64.to_radians();
        let lon = 109.6222_f64.to_radians();
        let az = azimuth_deg.to_radians();
        let el = elevation_deg.to_radians();
        let up = [lat.cos() * lon.cos(), lat.cos() * lon.sin(), lat.sin()];
        let north = [-lat.sin() * lon.cos(), -lat.sin() * lon.sin(), lat.cos()];
        let east = [-lon.sin(), lon.cos(), 0.];
        let ray: Vec<_> = (0..3)
            .map(|i| el.sin() * up[i] + el.cos() * (az.cos() * north[i] + az.sin() * east[i]))
            .collect();
        let pos: Vec<_> = (0..3)
            .map(|i| 6371.05 * up[i] + range_km * ray[i])
            .collect();
        let radius = pos.iter().map(|v| v * v).sum::<f64>().sqrt();
        let plat = (pos[2] / radius).asin();
        let plon = pos[1].atan2(pos[0]);
        let bvec =
            crate::geomagnetic::igrf14(year, plat.to_degrees(), plon.to_degrees(), radius - 6371.)?;
        let b = bvec.iter().map(|v| v * v).sum::<f64>().sqrt();
        let n = [
            -plat.sin() * plon.cos(),
            -plat.sin() * plon.sin(),
            plat.cos(),
        ];
        let e = [-plon.sin(), plon.cos(), 0.];
        let d = [
            -plat.cos() * plon.cos(),
            -plat.cos() * plon.sin(),
            -plat.sin(),
        ];
        let projection = (0..3)
            .map(|i| ray[i] * (bvec[0] * n[i] + bvec[1] * e[i] + bvec[2] * d[i]))
            .sum::<f64>();
        Ok((b, (projection.abs() / b).clamp(0., 1.).asin().to_degrees()))
    }
    /// Explicit finite-beam BGK response; ordinary collisionless guards are unchanged.
    pub fn near_perpendicular_from_beam(
        range: f64,
        az: f64,
        el: f64,
        year: f64,
        frequency: Vec<f64>,
        nu: f64,
        width: f64,
    ) -> Result<Self> {
        ensure!(
            nu > 0. && width > 0.,
            "explicit positive BGK rate and beam width required"
        );
        let (b, alpha) = Self::beam_field_at_year(range, az, el, year)?;
        Ok(Self {
            f: frequency,
            k: 4. * PI * 430.2e6 / 299792458.,
            b: b * 1e-9,
            alpha,
            h_temperature_ratio: 1.,
            heavy_ion_fractions: [1., 0., 0., 0.],
            electron_magnetized: true,
            include_electron_fluctuations: true,
            near_perpendicular: Some((nu, width)),
            eg: vec![],
            temps: vec![],
            cache_step_hz: 1.,
            cache_half: 0,
        })
    }
    fn direct_channels(
        &self,
        ti: f64,
        te: f64,
        v: f64,
        h: f64,
        he: f64,
        ne: f64,
        gain: f64,
        range: f64,
        hw: f64,
        derivatives: bool,
    ) -> Vec<Vec<f64>> {
        let (nu, width) = self.near_perpendicular.unwrap_or((0., 0.));
        crate::lower_hybrid::physical_channels(
            self.b,
            self.alpha,
            nu,
            width,
            &self.f,
            ti,
            te,
            v,
            h,
            he,
            ne,
            gain,
            range,
            hw,
            self.heavy_ion_fractions,
            self.h_temperature_ratio,
            self.include_electron_fluctuations,
            derivatives,
        )
    }
    fn at_field(b: f64, alpha: f64, df: f64) -> Result<Self> {
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
            heavy_ion_fractions: [1., 0., 0., 0.],
            electron_magnetized: true,
            include_electron_fluctuations: true,
            near_perpendicular: None,
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
    pub fn spectrum_with_mass(
        &self,
        ti: f64,
        te: f64,
        velocity: f64,
        h: f64,
        he_other: f64,
        ne: f64,
    ) -> (Vec<f64>, f64) {
        if self.electron_magnetized
            && (self.near_perpendicular.is_some()
                || std::env::var_os("SYISR_EXACT_MAGNETIZED_RESPONSE").is_some())
        {
            let mut raw = self
                .direct_channels(ti, te, velocity, h, he_other, ne, 1., 1., 1., false)
                .remove(0);
            for x in &mut raw {
                *x /= ne;
            }
            let mass = raw
                .windows(2)
                .zip(self.f.windows(2))
                .map(|(s, f)| (s[0] + s[1]) * (f[1] - f[0]) / 2.)
                .sum::<f64>();
            for x in &mut raw {
                *x /= mass;
            }
            return (raw, mass);
        }
        let heavy = (1. - h) * (1. - he_other);
        let fractions = [
            h,
            (1. - h) * he_other,
            heavy * self.heavy_ion_fractions[0],
            heavy * self.heavy_ion_fractions[1],
            heavy * self.heavy_ion_fractions[2],
            heavy * self.heavy_ion_fractions[3],
        ];
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
                for (mass, fraction) in [1., 4., 16., 30., 32., 14.].iter().zip(fractions) {
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

    /// Forward-mode AD of the physical H/He/O spectrum in six local coordinates.
    /// Custom primitives differentiate Faddeeva w and the exact cached electron response.
    /// Inputs h/he_other are physical fractions after the prescribed altitude taper;
    /// the last two derivative seeds include that same taper.
    pub fn physical_channels_ad(
        &self,
        ti: f64,
        te: f64,
        velocity: f64,
        h: f64,
        he_other: f64,
        ne: f64,
        gain: f64,
        range: f64,
        hweight: f64,
    ) -> Vec<Vec<f64>> {
        if self.electron_magnetized
            && (self.near_perpendicular.is_some()
                || std::env::var_os("SYISR_EXACT_MAGNETIZED_RESPONSE").is_some())
        {
            return self.direct_channels(
                ti, te, velocity, h, he_other, ne, gain, range, hweight, true,
            );
        }
        use crate::autodiff::{D, Z};
        let ti = D::seed(ti, 1, ti);
        let te = D::seed(te, 2, te);
        let velocity = D::seed(velocity, 3, 1000.);
        let h = D::seed(h, 4, hweight);
        let he = D::seed(he_other, 5, hweight);
        let ne = D::seed(ne, 0, ne);
        let one = D::constant(1.);
        let z1 = Z::constant(C::new(1., 0.));
        let alpha = ne * (E * E / (EPS * KB * self.k * self.k));
        let scale = ne / (gain * range * range);
        let heavy = (one - h) * (one - he);
        let fractions = [
            h,
            (one - h) * he,
            heavy * self.heavy_ion_fractions[0],
            heavy * self.heavy_ion_fractions[1],
            heavy * self.heavy_ion_fractions[2],
            heavy * self.heavy_ion_fractions[3],
        ];
        let ef = if self.include_electron_fluctuations {
            1.
        } else {
            0.
        };
        let mut out = vec![vec![0.; self.f.len()]; 7];
        for (k, &f) in self.f.iter().enumerate() {
            let omega = D::constant(2. * PI * f) - velocity * self.k;
            let ge = if self.electron_magnetized {
                let (g, gt, gf) = self.g_derivatives(te.v, omega.v / (2. * PI));
                Z {
                    v: g,
                    d: std::array::from_fn(|i| {
                        gt * (te.d[i] / te.v) + gf * (omega.d[i] / (2. * PI))
                    }),
                }
            } else {
                ad_g((te * (2. * KB / ME)).sqrt() * self.k, omega)
            };
            let ce = (z1 + Z::i(omega) * ge) * (alpha / te);
            let mut ci = Z::constant(C::default());
            let mut fi = D::constant(0.);
            for (mass, fraction) in [1., 4., 16., 30., 32., 14.].into_iter().zip(fractions) {
                let temp = ti
                    * if mass == 1. {
                        self.h_temperature_ratio
                    } else {
                        1.
                    };
                let gi = ad_g((temp * (2. * KB / (mass * AMU))).sqrt() * self.k, omega);
                ci = ci + (z1 + Z::i(omega) * gi) * (alpha * fraction / temp);
                fi = fi + fraction * gi.re() * 2.;
            }
            let eps = z1 + ce + ci;
            let raw =
                ((z1 + ci).norm_sqr() * ge.re() * (2. * ef) + ce.norm_sqr() * fi) / eps.norm_sqr();
            let physical = scale * raw;
            out[0][k] = physical.v;
            for i in 0..6 {
                out[i + 1][k] = physical.d[i];
            }
        }
        out
    }

    /// Physical density-scaled spectrum and transformed derivatives in one pass.
    /// Channels: value, logNe, logTi, logTe, velocity/1000, H coefficient, He=0.
    /// No normalization approximation: Ne*mass*normalized_s == Ne*raw_s.
    pub fn physical_channels(
        &self,
        ti: f64,
        te: f64,
        velocity: f64,
        h: f64,
        ne: f64,
        gain: f64,
        range: f64,
        hweight: f64,
    ) -> Vec<Vec<f64>> {
        let mut out = vec![vec![0.; self.f.len()]; 7];
        let alpha = ne * E * E / (EPS * KB * self.k * self.k);
        let scale = ne / (gain * range * range);
        let dw = -self.k * 1000.;
        let ef = if self.include_electron_fluctuations {
            1.
        } else {
            0.
        };
        for (k, &f) in self.f.iter().enumerate() {
            let omega = 2. * PI * f - self.k * velocity;
            let (ge, get, gev) = if self.electron_magnetized {
                let (g, gt, gf) = self.g_derivatives(te, omega / (2. * PI));
                (g, gt, gf * (dw / (2. * PI)))
            } else {
                let a = self.k * (2. * KB * te / ME).sqrt();
                let z = omega / a;
                let w = C::new(z, 0.).w();
                let wp = -2. * z * w + C::new(0., 2. / PI.sqrt());
                let g = PI.sqrt() / a * w;
                (
                    g,
                    -0.5 * g - 0.5 * z * PI.sqrt() / a * wp,
                    PI.sqrt() / (a * a) * wp * dw,
                )
            };
            let ce = (C::new(1., 0.) + C::new(0., omega) * ge) * (alpha / te);
            let mut dc = [C::default(); 5];
            dc[0] = ce;
            dc[2] = C::new(0., omega) * get * (alpha / te) - ce;
            dc[3] = (C::new(0., dw) * ge + C::new(0., omega) * gev) * (alpha / te);
            let mut ci = C::default();
            let mut fi = 0.;
            let mut di = [C::default(); 5];
            let mut df = [0.; 5];
            for (mass, fraction, sign) in [(1., h, 1.), (16., 1. - h, -1.)] {
                let temp = ti
                    * if mass == 1. {
                        self.h_temperature_ratio
                    } else {
                        1.
                    };
                let a = self.k * (2. * KB * temp / (mass * AMU)).sqrt();
                let z = omega / a;
                let w = C::new(z, 0.).w();
                let wp = -2. * z * w + C::new(0., 2. / PI.sqrt());
                let gi = PI.sqrt() / a * w;
                let git = -0.5 * gi - 0.5 * z * PI.sqrt() / a * wp;
                let giv = PI.sqrt() / (a * a) * wp * dw;
                let chi = (C::new(1., 0.) + C::new(0., omega) * gi) * (alpha / temp);
                ci += chi * fraction;
                fi += fraction * 2. * gi.re;
                di[1] += (C::new(0., omega) * git * (alpha / temp) - chi) * fraction;
                di[3] +=
                    (C::new(0., dw) * gi + C::new(0., omega) * giv) * (alpha * fraction / temp);
                di[4] += chi * (sign * hweight);
                df[1] += 2. * fraction * git.re;
                df[3] += 2. * fraction * giv.re;
                df[4] += 2. * gi.re * sign * hweight;
            }
            di[0] = ci;
            let one = C::new(1., 0.) + ci;
            let eps = one + ce;
            let den = eps.norm_sqr();
            let numerator = one.norm_sqr() * 2. * ge.re * ef + ce.norm_sqr() * fi;
            let raw = numerator / den;
            out[0][k] = scale * raw;
            for field in 0..5 {
                let dg = match field {
                    2 => get,
                    3 => gev,
                    _ => C::default(),
                };
                let dn = 4. * (one.conj() * di[field]).re * ge.re * ef
                    + one.norm_sqr() * 2. * dg.re * ef
                    + 2. * (ce.conj() * dc[field]).re * fi
                    + ce.norm_sqr() * df[field];
                let dd = 2. * (eps.conj() * (di[field] + dc[field])).re;
                out[field + 1][k] =
                    scale * ((dn - raw * dd) / den + if field == 0 { raw } else { 0. });
            }
        }
        out
    }
    /// Derivatives of exactly the same bilinear cache used by g(): logTe and Hz.
    fn g_derivatives(&self, t: f64, f: f64) -> (C, C, C) {
        let qt = ((t / 150.).ln() / (20000_f64 / 150.).ln() * 200.).clamp(0., 199.999999);
        let j = qt.floor() as usize;
        let u = qt - j as f64;
        let qf = (f / self.cache_step_hz + self.cache_half as f64)
            .clamp(0., (2 * self.cache_half) as f64 - 1e-6);
        let i = qf.floor() as usize;
        let v = qf - i as f64;
        let at = |j: usize| self.eg[j][i] * (1. - v) + self.eg[j][i + 1] * v;
        let g = at(j) * (1. - u) + at(j + 1) * u;
        let mut gt = at(j + 1) - at(j);
        if u < 1e-9 && j > 0 {
            gt = (gt + at(j) - at(j - 1)) * 0.5;
        }
        gt *= 200. / (20000_f64 / 150.).ln();
        let row = |i: usize| self.eg[j][i] * (1. - u) + self.eg[j + 1][i] * u;
        let mut gf = row(i + 1) - row(i);
        if v < 1e-9 && i > 0 {
            gf = (gf + row(i) - row(i - 1)) * 0.5;
        }
        gf /= self.cache_step_hz;
        (g, gt, gf)
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

#[cfg(test)]
mod beam_tests {
    use super::*;
    #[test]
    fn zenith_geometry_matches_original_field() {
        for r in [250., 725., 1600.] {
            let (b, alpha) = Theory::beam_field(r, 357.99, 90.).unwrap();
            let v = crate::geomagnetic::igrf14(2025.76, 18.3492, 109.6222, r + 0.05).unwrap();
            let m = v.iter().map(|x| x * x).sum::<f64>().sqrt();
            assert!((b - m).abs() / m < 1e-10);
            assert!((alpha - (v[2].abs() / m).asin().to_degrees()).abs() < 1e-9);
        }
    }
    #[test]
    fn near_perpendicular_guard_and_cache_accuracy() {
        assert!(Theory::from_beam(250., 0., 60., 500.).is_err());
        for alpha in [3.1, 12., 25.] {
            let t = Theory::at_field(35000., alpha, 500.).unwrap();
            for temperature in [505., 1010., 2020., 4040.] {
                for frequency in [173., 1373., 5373., 12373., 20373.] {
                    let exact = electron_g(t.k, t.b, alpha, temperature, frequency);
                    let error = (t.g(temperature, frequency) - exact).norm() / exact.norm();
                    assert!(
                        error < 0.005,
                        "alpha={alpha} T={temperature} f={frequency} error={error}"
                    );
                }
            }
        }
    }
}

#[cfg(test)]
mod physical_gradient_tests {
    use super::*;
    #[test]
    fn analytic_channels_match_original_spectrum_and_finite_differences() {
        let t = Theory::new(500., 500.).unwrap();
        for (ti, te, v, h, ne) in [
            (500.123, 900.123, 37.123, 0.03, 1e10),
            (1200.123, 2400.123, -231.12, 0.4, 5e11),
            (3000.123, 8000.123, 517.23, 0.9, 1e12),
            (1000., 1500., 0., 0.2, 1e11),
        ] {
            let hw = 0.8;
            let channels = t.physical_channels(ti, te, v, h, ne, 700., 600., hw);
            let calc = |p: [f64; 5]| {
                let (s, m) = t.spectrum_with_mass(p[1], p[2], p[3], p[4], 0., p[0]);
                s.iter()
                    .map(|s| s * m * p[0] / (700. * 600. * 600.))
                    .collect::<Vec<_>>()
            };
            let q = [ne, ti, te, v, h];
            let reference = calc(q);
            let norm = reference.iter().map(|v| v * v).sum::<f64>().sqrt();
            let err = reference
                .iter()
                .zip(&channels[0])
                .map(|(a, b)| (a - b).powi(2))
                .sum::<f64>()
                .sqrt()
                / norm;
            assert!(err < 1e-12, "spectrum{err}");
            for f in 0..5 {
                let e = 1e-6;
                let mut a = q;
                let mut b = q;
                match f {
                    0 | 1 | 2 => {
                        a[f] *= f64::exp(e);
                        b[f] *= f64::exp(-e)
                    }
                    3 => {
                        a[f] += 1000. * e;
                        b[f] -= 1000. * e
                    }
                    _ => {
                        a[f] += hw * e;
                        b[f] -= hw * e
                    }
                }
                let va = calc(a);
                let vb = calc(b);
                let d: Vec<_> = va.iter().zip(vb).map(|(a, b)| (a - b) / (2. * e)).collect();
                let scale = d.iter().map(|v| v * v).sum::<f64>().sqrt().max(1e-10);
                let err = d
                    .iter()
                    .zip(&channels[f + 1])
                    .map(|(a, b)| (a - b).powi(2))
                    .sum::<f64>()
                    .sqrt()
                    / scale;
                assert!(err < 3e-4, "field{f},err{err},Te{te},v{v}");
            }
        }
    }
}

// The Faddeeva identity w'(z)=-2zw(z)+2i/sqrt(pi) supplies the custom AD rule.
fn ad_g(a: crate::autodiff::D, omega: crate::autodiff::D) -> crate::autodiff::Z {
    use crate::autodiff::Z;
    let z = omega / a;
    let w = C::new(z.v, 0.).w();
    let wp = -2. * z.v * w + C::new(0., 2. / PI.sqrt());
    Z {
        v: w,
        d: z.d.map(|d| wp * d),
    } * (crate::autodiff::D::constant(PI.sqrt()) / a)
}

#[cfg(test)]
mod forward_ad_tests {
    use super::*;
    #[test]
    fn dual_spectrum_and_all_six_derivatives_match_independent_differences() {
        let mut t = Theory::new(850., 500.).unwrap();
        t.heavy_ion_fractions = [0.2, 0.5, 0.2, 0.1];
        for magnetized in [true, false] {
            t.electron_magnetized = magnetized;
            for ef in [true, false] {
                t.include_electron_fluctuations = ef;
                for (ti, te, v, h, he, ne, hw) in [
                    (500.123, 900.123, 37.123, 0.03, 0.07, 1e10, 1.),
                    (1200.123, 2400.123, -231.12, 0.4, 0.2, 5e11, 0.8),
                    (3000.123, 8000.123, 517.23, 0.9, 0.5, 1e12, 1.),
                    (1000., 1500., 0., 0.2, 0., 1e11, 1.),
                    (800., 1700., -13., 0., 0., 1e11, 0.),
                ] {
                    let channels = t.physical_channels_ad(ti, te, v, h, he, ne, 700., 850., hw);
                    let q = [ne, ti, te, v, h, he];
                    let calc = |p: [f64; 6]| {
                        let (s, m) = t.spectrum_with_mass(p[1], p[2], p[3], p[4], p[5], p[0]);
                        s.iter()
                            .map(|v| v * m * p[0] / (700. * 850. * 850.))
                            .collect::<Vec<_>>()
                    };
                    let r = calc(q);
                    let norm = r.iter().map(|v| v * v).sum::<f64>().sqrt();
                    let err = r
                        .iter()
                        .zip(&channels[0])
                        .map(|(a, b)| (a - b).powi(2))
                        .sum::<f64>()
                        .sqrt()
                        / norm;
                    assert!(err < 2e-12, "primal {err}");
                    for k in 0..6 {
                        let eps = 1e-6;
                        let mut a = q;
                        let mut b = q;
                        match k {
                            0..=2 => {
                                a[k] *= f64::exp(eps);
                                b[k] *= f64::exp(-eps)
                            }
                            3 => {
                                a[k] += 1000. * eps;
                                b[k] -= 1000. * eps
                            }
                            _ => {
                                a[k] += hw * eps;
                                b[k] -= hw * eps
                            }
                        }
                        let a = calc(a);
                        let b = calc(b);
                        let fd: Vec<_> =
                            a.iter().zip(b).map(|(a, b)| (a - b) / (2. * eps)).collect();
                        let scale = fd
                            .iter()
                            .map(|v| v * v)
                            .sum::<f64>()
                            .sqrt()
                            .max(norm * 1e-8);
                        let err = fd
                            .iter()
                            .zip(&channels[k + 1])
                            .map(|(a, b)| (a - b).powi(2))
                            .sum::<f64>()
                            .sqrt()
                            / scale;
                        assert!(
                            err < 3e-4,
                            "field{k} error{err} magnetized{magnetized} ef{ef} h{h} he{he}"
                        );
                    }
                }
            }
        }
    }
}

#[cfg(test)]
mod direct_response_tests {
    use super::*;
    #[test]
    fn smooth_direct_response_agrees_with_cache_and_finite_differences() {
        let mut t = Theory::new(700., 500.).unwrap();
        t.heavy_ion_fractions = [0.6, 0.2, 0.1, 0.1];
        let input = [1e12, 1100., 1800., 23., 0.2, 0.05];
        let eval = |x: [f64; 6], d| {
            t.direct_channels(x[1], x[2], x[3], x[4], x[5], x[0], 43., 700., 1., d)
        };
        let a = eval(input, true);
        let old = t.physical_channels_ad(
            input[1], input[2], input[3], input[4], input[5], input[0], 43., 700., 1.,
        );
        let peak = a[0].iter().copied().fold(0., f64::max);
        assert!(
            a[0].iter()
                .zip(&old[0])
                .map(|(a, b)| (a - b).abs())
                .fold(0., f64::max)
                / peak
                < 0.003
        );
        for k in 0..6 {
            let mut p = input;
            let mut m = input;
            let eps = 1e-5f64;
            if k < 3 {
                p[k] *= eps.exp();
                m[k] *= (-eps).exp();
            } else {
                let step = eps * if k == 3 { 1000. } else { 1. };
                p[k] += step;
                m[k] -= step;
            }
            let pp = eval(p, false);
            let mm = eval(m, false);
            let fd: Vec<_> = pp[0]
                .iter()
                .zip(&mm[0])
                .map(|(a, b)| (a - b) / (2. * eps))
                .collect();
            let denominator = fd.iter().map(|x| x.abs()).fold(0., f64::max);
            let error = fd
                .iter()
                .zip(&a[k + 1])
                .map(|(f, d)| (f - d).abs())
                .fold(0., f64::max)
                / denominator;
            assert!(error < 2e-5, "field{k} error{error}");
        }
    }
}
