//! Gaussian voltage fourth moments for exact partial ACF sums and a common reference.
//! No independent-noise assumption between gates sharing products or the reference.
use anyhow::{ensure, Result};
use nalgebra::DMatrix as M;
use num_complex::Complex64 as C;
use rustfft::FftPlanner;
const MAX: isize = 40;
struct Prefix {
    values: Vec<C>,
    sum: C,
}
impl Prefix {
    fn new(k: &[C]) -> Self {
        let mut s = C::default();
        let mut f = C::default();
        let mut values = vec![];
        for &v in k {
            s += v;
            f += s;
            values.push(f);
        }
        Self { values, sum: s }
    }
    fn at(&self, x: isize) -> C {
        if x < -MAX {
            C::default()
        } else if x > MAX {
            self.values[80] + self.sum * (x - MAX) as f64
        } else {
            self.values[(x + MAX) as usize]
        }
    }
    fn rectangle(&self, a: isize, b: isize, c: isize, d: isize) -> C {
        if a - d + 1 > MAX || b - c - 1 < -MAX {
            return C::default();
        }
        self.at(b - 1 - c) - self.at(a - 1 - c) - self.at(b - 1 - d) + self.at(a - 1 - d)
    }
}
pub struct VoltageProducts {
    pub gamma: Vec<C>,
    c: Vec<Prefix>,
    p: Vec<Prefix>,
}
impl VoltageProducts {
    pub fn from_fir(fir: &[(f64, C)]) -> Result<Self> {
        let mut h = vec![C::default(); 41];
        for &(t, v) in fir {
            let j = (t / 1e-5).round() as isize + 20;
            ensure!((0..41).contains(&j), "FIR outside supported offsets");
            h[j as usize] = v;
        }
        let gamma: Vec<C> = (0..=40)
            .map(|d| (0..41 - d).map(|i| h[i] * h[i + d].conj()).sum())
            .collect();
        Self::new(gamma)
    }
    pub fn new(gamma: Vec<C>) -> Result<Self> {
        ensure!(
            gamma.len() == 41
                && gamma[0].re > 0.
                && gamma.iter().all(|v| v.re.is_finite() && v.im.is_finite()),
            "invalid voltage correlation"
        );
        let rho = |d: isize| {
            if d.abs() > MAX {
                C::default()
            } else if d < 0 {
                gamma[-d as usize].conj()
            } else {
                gamma[d as usize]
            }
        };
        let mut c = vec![];
        let mut p = vec![];
        for l in 1isize..=60 {
            for m in 1isize..=60 {
                c.push(Prefix::new(
                    &(-MAX..=MAX)
                        .map(|d| rho(-d) * rho(d + l - m))
                        .collect::<Vec<_>>(),
                ));
                p.push(Prefix::new(
                    &(-MAX..=MAX)
                        .map(|d| rho(m - d) * rho(l + d))
                        .collect::<Vec<_>>(),
                ));
            }
        }
        Ok(Self { gamma, c, p })
    }
    /// Returns RR, RI, IR, II for one pulse. Both products include actual-pair
    /// averaging and the existing 200/(200-l) signal normalization.
    pub fn covariance(&self, i: usize, l: usize, j: usize, m: usize, first: usize) -> [f64; 4] {
        let a = i.saturating_sub(l).max(first) as isize;
        let b = (i + 21) as isize;
        let c = j.saturating_sub(m).max(first) as isize;
        let d = (j + 21) as isize;
        assert!(b > a && d > c && l > 0 && m > 0 && l <= 60 && m <= 60);
        let k = (l - 1) * 60 + m - 1;
        let scale =
            200. / (200 - l) as f64 / (b - a) as f64 * 200. / (200 - m) as f64 / (d - c) as f64;
        let v = self.c[k].rectangle(a, b, c, d) * scale;
        let p = self.p[k].rectangle(a, b, c, d) * scale;
        [
            0.5 * (v + p).re,
            0.5 * (p.im - v.im),
            0.5 * (p.im + v.im),
            0.5 * (v - p).re,
        ]
    }
    pub fn local_matrix(&self, native: &[usize], first: usize) -> M<f64> {
        let p = native.len() * 120;
        let mut out = M::zeros(p, p);
        for g in 0..native.len() {
            for h in g..native.len() {
                if native[h].saturating_sub(native[g]) > 120 {
                    continue;
                }
                for l in 1..=60 {
                    for m in 1..=60 {
                        if g == h && m < l {
                            continue;
                        }
                        let i = 2 * (g * 60 + l - 1);
                        let j = 2 * (h * 60 + m - 1);
                        let v = self.covariance(native[g], l, native[h], m, first);
                        out[(i, j)] = v[0];
                        out[(j, i)] = v[0];
                        out[(i + 1, j + 1)] = v[3];
                        out[(j + 1, i + 1)] = v[3];
                        if i == j {
                            out[(i, j + 1)] = (v[1] + v[2]) * 0.5;
                            out[(j + 1, i)] = out[(i, j + 1)];
                        } else {
                            out[(i, j + 1)] = v[1];
                            out[(j + 1, i)] = v[1];
                            out[(i + 1, j)] = v[2];
                            out[(j, i + 1)] = v[2];
                        }
                    }
                }
            }
        }
        out
    }
}
pub struct StructuredModel {
    pub voltage_band: Option<VoltageBand>,
    pub gamma: Vec<C>,
    pub local_scale: Vec<f64>,
    pub reference_scale: f64,
    pub sigma: Vec<f64>,
    pub clamped: usize,
    pub normalized: M<f64>,
}
impl StructuredModel {
    /// sigma is the normalizing SEM used by the likelihood. post_variance is
    /// estimated independently for the target covariance (same units, ADU^4).
    pub fn build(
        gamma: Vec<C>,
        native: &[usize],
        post_variance: &[f64],
        sigma: &[f64],
        reference_scale: f64,
    ) -> Result<Self> {
        let v = VoltageProducts::new(gamma.clone())?;
        let local = v.local_matrix(native, 30);
        Self::calibrate(gamma, local, post_variance, sigma, reference_scale, None)
    }
    pub fn with_self_noise(
        gamma: Vec<C>,
        native: &[usize],
        local_acf: &[Vec<C>],
        noise_power: f64,
        post_variance: &[f64],
        sigma: &[f64],
        reference_scale: f64,
    ) -> Result<Self> {
        let band = VoltageBand::from_local_acfs(&gamma, native, local_acf, noise_power)?;
        let local = band.product_matrix(native);
        Self::calibrate(
            gamma,
            local,
            post_variance,
            sigma,
            reference_scale,
            Some(band),
        )
    }
    fn calibrate(
        gamma: Vec<C>,
        mut local: M<f64>,
        post_variance: &[f64],
        sigma: &[f64],
        reference_scale: f64,
        voltage_band: Option<VoltageBand>,
    ) -> Result<Self> {
        let v = VoltageProducts::new(gamma.clone())?;
        let reference = v.local_matrix(&[3701], 30);
        let n = local.nrows();
        ensure!(
            post_variance.len() == n && sigma.len() == n,
            "structured covariance dimensions"
        );
        let mut clamped = 0;
        let mut local_scale = vec![0.; n];
        for i in 0..n {
            let b = reference[(i % 120, i % 120)] * reference_scale;
            let target = post_variance[i] - b;
            let floor = 0.02 * post_variance[i];
            if target < floor {
                clamped += 1;
            }
            ensure!(
                local[(i, i)] > 0. && sigma[i] > 0.,
                "nonpositive local variance"
            );
            local_scale[i] = (target.max(floor) / local[(i, i)]).sqrt() / sigma[i];
        }
        for j in 0..n {
            for i in 0..n {
                local[(i, j)] = local[(i, j)] * local_scale[i] * local_scale[j]
                    + reference[(i % 120, j % 120)] * reference_scale / (sigma[i] * sigma[j]);
            }
        }
        // Tiny, declared numerical floor, not a fitted noise inflation factor.
        for i in 0..n {
            local[(i, i)] += 1e-8;
        }
        Ok(Self {
            voltage_band,
            gamma,
            local_scale,
            reference_scale,
            sigma: sigma.to_vec(),
            clamped,
            normalized: local,
        })
    }
    pub fn reconstruct(
        gamma: Vec<C>,
        native: &[usize],
        local_scale: &[f64],
        sigma: &[f64],
        reference_scale: f64,
    ) -> Result<M<f64>> {
        let v = VoltageProducts::new(gamma)?;
        let mut c = v.local_matrix(native, 30);
        let r = v.local_matrix(&[3701], 30);
        let n = c.nrows();
        ensure!(
            local_scale.len() == n && sigma.len() == n,
            "structured scale dimensions"
        );
        for j in 0..n {
            for i in 0..n {
                c[(i, j)] = c[(i, j)] * local_scale[i] * local_scale[j]
                    + r[(i % 120, j % 120)] * reference_scale / (sigma[i] * sigma[j]);
            }
        }
        for i in 0..n {
            c[(i, i)] += 1e-8;
        }
        Ok(c)
    }
}

/// PSD nonstationary voltage covariance: stationary receiver noise plus smooth
/// mixtures of independently stationary plasma components. Signal spectra are
/// trained from local voltage ACFs; no fitted plasma residual enters the model.
pub struct VoltageBand {
    pub first: usize,
    pub last: usize,
    pub values: Vec<C>,
    pub signal_projection_fraction: Vec<f64>,
    pub signed_signal_power: Vec<f64>,
}
impl VoltageBand {
    pub fn from_local_acfs(
        noise: &[C],
        native: &[usize],
        acfs: &[Vec<C>],
        power: f64,
    ) -> Result<Self> {
        ensure!(
            native.len() == acfs.len() && power > 0.,
            "voltage model dimensions"
        );
        let nfft = 256;
        let mut plan = FftPlanner::<f64>::new();
        let inverse = plan.plan_fft_inverse(nfft);
        let forward = plan.plan_fft_forward(nfft);
        let tap: Vec<f64> = (0..61)
            .map(|i| (std::f64::consts::PI * (i as f64 + 0.5) / 61.).sin())
            .collect();
        let norm = tap.iter().map(|v| v * v).sum::<f64>();
        let taper: Vec<f64> = (0..61)
            .map(|d| (0..61 - d).map(|i| tap[i] * tap[i + d]).sum::<f64>() / norm)
            .collect();
        let mut signals = vec![];
        let mut signed_signal_power = vec![];
        let mut signal_projection_fraction = vec![];
        for a in acfs {
            let signed = a[0].re / power - 1.;
            signed_signal_power.push(signed);
            if signed <= 0. {
                signals.push(vec![C::default(); 61]);
                signal_projection_fraction.push(0.);
                continue;
            }
            let mut spec = vec![C::default(); nfft];
            spec[0] = C::new(signed, 0.);
            for d in 1..=60 {
                let g = a[d] / power
                    - if d < noise.len() {
                        noise[d]
                    } else {
                        C::default()
                    };
                spec[d] = g;
                spec[nfft - d] = g.conj();
            }
            inverse.process(&mut spec);
            let negative = spec.iter().map(|v| (-v.re).max(0.)).sum::<f64>();
            let positive = spec.iter().map(|v| v.re.max(0.)).sum::<f64>();
            signal_projection_fraction.push(negative / (positive + negative).max(1e-30));
            for s in &mut spec {
                *s = C::new(
                    s.re.max(0.) * signed * nfft as f64 / positive.max(1e-30),
                    0.,
                );
            }
            forward.process(&mut spec);
            let mut g: Vec<C> = (0..61).map(|d| spec[d] / nfft as f64 * taper[d]).collect();
            g[0] = C::new(signed, 0.);
            signals.push(g);
        }
        let first = 30;
        let last = *native.last().unwrap() + 80;
        let centers: Vec<f64> = native.iter().map(|&v| v as f64 + 10.).collect();
        let powers: Vec<f64> = signed_signal_power.iter().map(|v| v.max(0.)).collect();
        let mut weights = vec![];
        for k in first..=last {
            let mut w: Vec<f64> = centers
                .iter()
                .map(|&c| (-((k as f64 - c) / 60.).powi(2) / 4.).exp())
                .collect();
            let s = w.iter().map(|v| v * v).sum::<f64>().sqrt();
            for v in &mut w {
                *v /= s;
            }
            let j = centers
                .partition_point(|&c| c <= k as f64)
                .saturating_sub(1)
                .min(centers.len() - 2);
            let u = ((k as f64 - centers[j]) / (centers[j + 1] - centers[j])).clamp(0., 1.);
            let target = (1. - u) * powers[j] + u * powers[j + 1];
            let mixed = w.iter().zip(&powers).map(|(w, p)| w * w * p).sum::<f64>();
            let scale = if mixed > 0. {
                (target / mixed).sqrt()
            } else {
                0.
            };
            for v in &mut w {
                *v *= scale;
            }
            weights.push(w);
        }
        let mut values = vec![C::default(); (last - first + 1) * 61];
        for k in first..=last {
            for d in 0..=60 {
                if k + d > last {
                    continue;
                }
                let sky: C = (0..native.len())
                    .map(|j| signals[j][d] * weights[k - first][j] * weights[k + d - first][j])
                    .sum();
                values[(k - first) * 61 + d] = sky
                    + if d < noise.len() {
                        noise[d]
                    } else {
                        C::default()
                    };
            }
        }
        Ok(Self {
            first,
            last,
            values,
            signal_projection_fraction,
            signed_signal_power,
        })
    }
    #[inline]
    pub fn at(&self, a: isize, b: isize) -> C {
        if a < self.first as isize
            || b < self.first as isize
            || a > self.last as isize
            || b > self.last as isize
            || (a - b).abs() > 60
        {
            return C::default();
        }
        if a <= b {
            self.values[(a as usize - self.first) * 61 + (b - a) as usize]
        } else {
            self.values[(b as usize - self.first) * 61 + (a - b) as usize].conj()
        }
    }
    pub fn product_matrix(&self, native: &[usize]) -> M<f64> {
        let n = native.len() * 120;
        let mut out = M::zeros(n, n);
        let first = self.first as isize;
        let end = *native.last().unwrap() as isize + 21;
        let na = (end - first) as usize;
        let stride = na + 1;
        let mut pc = vec![C::default(); 121 * stride];
        let mut pp = pc.clone();
        for l in 1isize..=60 {
            for m in l..=60 {
                for d in -60isize..=60 {
                    let ic = d + l - m;
                    let usec = ic.abs() <= 60;
                    let usep = (m - d).abs() <= 60 && (l + d).abs() <= 60;
                    if !usec && !usep {
                        continue;
                    }
                    let o = (d + 60) as usize * stride;
                    pc[o] = C::default();
                    pp[o] = C::default();
                    let mut c = C::default();
                    let mut p = C::default();
                    for a in first..end {
                        let b = a - d;
                        if usec {
                            c += self.at(a, b) * self.at(b + m, a + l);
                            pc[o + (a - first) as usize + 1] = c;
                        }
                        if usep {
                            p += self.at(a, b + m) * self.at(b, a + l);
                            pp[o + (a - first) as usize + 1] = p;
                        }
                    }
                }
                for g in 0..native.len() {
                    for h in 0..native.len() {
                        if l == m && h < g {
                            continue;
                        }
                        let a = (native[g] as isize - l).max(first);
                        let b = native[g] as isize + 21;
                        let c = (native[h] as isize - m).max(first);
                        let e = native[h] as isize + 21;
                        let dmin = (a - e + 1).max(-60);
                        let dmax = (b - c - 1).min(60);
                        if dmin > dmax {
                            continue;
                        }
                        let mut vc = C::default();
                        let mut vp = C::default();
                        for d in dmin..=dmax {
                            let lo = a.max(c + d);
                            let hi = b.min(e + d);
                            if hi <= lo {
                                continue;
                            }
                            let o = (d + 60) as usize * stride;
                            let low = (lo - first) as usize;
                            let high = (hi - first) as usize;
                            if (d + l - m).abs() <= 60 {
                                vc += pc[o + high] - pc[o + low];
                            }
                            if (m - d).abs() <= 60 && (l + d).abs() <= 60 {
                                vp += pp[o + high] - pp[o + low];
                            }
                        }
                        let scale = 200. / (200 - l) as f64 / (b - a) as f64 * 200.
                            / (200 - m) as f64
                            / (e - c) as f64;
                        vc *= scale;
                        vp *= scale;
                        let i = 2 * (g * 60 + l as usize - 1);
                        let j = 2 * (h * 60 + m as usize - 1);
                        let rr = (vc + vp).re * 0.5;
                        let ii = (vc - vp).re * 0.5;
                        let ri = (vp.im - vc.im) * 0.5;
                        let ir = (vp.im + vc.im) * 0.5;
                        out[(i, j)] = rr;
                        out[(j, i)] = rr;
                        out[(i + 1, j + 1)] = ii;
                        out[(j + 1, i + 1)] = ii;
                        if i == j {
                            out[(i, j + 1)] = vp.im * 0.5;
                            out[(j + 1, i)] = out[(i, j + 1)];
                        } else {
                            out[(i, j + 1)] = ri;
                            out[(j + 1, i)] = ri;
                            out[(i + 1, j)] = ir;
                            out[(j, i + 1)] = ir;
                        }
                    }
                }
            }
        }
        out
    }
}
#[cfg(test)]
mod tests {
    use super::*;
    #[test]
    fn nonstationary_prefix_matches_direct_wick_sum() {
        let native = [20usize, 40, 90];
        let first = 30;
        let last = 170;
        let mut values = vec![C::default(); (last - first + 1) * 61];
        for a in first..=last {
            for d in 0..=60 {
                let amp = |k: usize| 1. + k as f64 / 100.;
                values[(a - first) * 61 + d] =
                    C::from_polar(amp(a) * amp(a + d) * 0.8f64.powi(d as i32), 0.13 * d as f64);
            }
        }
        let band = VoltageBand {
            first,
            last,
            values,
            signal_projection_fraction: vec![],
            signed_signal_power: vec![],
        };
        let matrix = band.product_matrix(&native);
        for (g, l, h, m) in [(0, 1, 1, 2), (0, 60, 2, 45), (2, 40, 1, 15), (1, 30, 1, 30)] {
            let mut vc = C::default();
            let mut vp = C::default();
            let lo_a = native[g].saturating_sub(l).max(first);
            let lo_b = native[h].saturating_sub(m).max(first);
            for a in lo_a..native[g] + 21 {
                for b in lo_b..native[h] + 21 {
                    let (a, b) = (a as isize, b as isize);
                    vc += band.at(a, b) * band.at(b + m as isize, a + l as isize);
                    vp += band.at(a, b + m as isize) * band.at(b, a + l as isize);
                }
            }
            let scale = 200. / (200 - l) as f64 / (native[g] + 21 - lo_a) as f64 * 200.
                / (200 - m) as f64
                / (native[h] + 21 - lo_b) as f64;
            let expected = [
                (vc + vp).re * 0.5,
                (vp.im - vc.im) * 0.5,
                (vp.im + vc.im) * 0.5,
                (vc - vp).re * 0.5,
            ];
            let i = 2 * (g * 60 + l - 1);
            let j = 2 * (h * 60 + m - 1);
            for (v, e) in [
                matrix[(i, j)],
                matrix[(i, j + 1)],
                matrix[(i + 1, j)],
                matrix[(i + 1, j + 1)],
            ]
            .iter()
            .zip(expected)
            {
                assert!(
                    (v - e * scale).abs() < 1e-10,
                    "{g} {l} {h} {m}: {v} {}",
                    e * scale
                );
            }
        }
    }
    #[test]
    fn white_noise_common_reference_is_half_correlated_at_large_separation() {
        let mut gamma = vec![C::default(); 41];
        gamma[0] = C::new(1., 0.);
        let v = VoltageProducts::new(gamma.clone()).unwrap();
        let native = [100usize, 400];
        let local = v.local_matrix(&native, 30);
        let var: Vec<f64> = (0..240).map(|i| 2. * local[(i, i)]).collect();
        let sigma: Vec<f64> = var.iter().map(|v| v.sqrt()).collect();
        let m = StructuredModel::build(gamma, &native, &var, &sigma, 1.).unwrap();
        assert!((m.normalized[(0, 120)] - 0.5).abs() < 1e-12);
        assert!(m.normalized[(0, 122)].abs() < 1e-12);
        assert!(m.clamped == 0);
    }
    #[test]
    fn prefix_matches_direct_gaussian_pair_sum() {
        let fir = [
            (-1e-5, C::new(0.1, 0.03)),
            (0., C::new(0.95, 0.)),
            (1e-5, C::new(0.1, -0.03)),
        ];
        let v = VoltageProducts::from_fir(&fir).unwrap();
        let rho = |d: isize| {
            if d.abs() > 40 {
                C::default()
            } else if d < 0 {
                v.gamma[-d as usize].conj()
            } else {
                v.gamma[d as usize]
            }
        };
        for (i, l, j, m) in [
            (20usize, 1usize, 40usize, 2usize),
            (20, 60, 55, 45),
            (90, 40, 125, 15),
            (200, 10, 400, 20),
        ] {
            let mut c = C::default();
            let mut p = C::default();
            let a = i.saturating_sub(l).max(30);
            let b = j.saturating_sub(m).max(30);
            for x in a..i + 21 {
                for y in b..j + 21 {
                    let d = x as isize - y as isize;
                    c += rho(-d) * rho(d + l as isize - m as isize);
                    p += rho(m as isize - d) * rho(l as isize + d);
                }
            }
            let s = 200. / (200 - l) as f64 / (i + 21 - a) as f64 * 200.
                / (200 - m) as f64
                / (j + 21 - b) as f64;
            let want = [
                (c + p).re * 0.5 * s,
                (p.im - c.im) * 0.5 * s,
                (p.im + c.im) * 0.5 * s,
                (c - p).re * 0.5 * s,
            ];
            let got = v.covariance(i, l, j, m, 30);
            for k in 0..4 {
                assert!((want[k] - got[k]).abs() < 1e-12);
            }
        }
    }
    #[test]
    fn gaussian_fourth_moments_match_simulated_complex_voltages() {
        let h = [C::new(0.8, 0.), C::new(0.3, 0.2), C::new(-0.1, 0.1)];
        let v = VoltageProducts::from_fir(
            &h.iter()
                .enumerate()
                .map(|(i, &h)| ((i as f64 - 1.) * 1e-5, h))
                .collect::<Vec<_>>(),
        )
        .unwrap();
        let want = v.covariance(20, 1, 40, 2, 30);
        let mut seed = 9127831u64;
        let mut normal = || {
            let mut uniform = || {
                seed ^= seed << 13;
                seed ^= seed >> 7;
                seed ^= seed << 17;
                ((seed >> 11) as f64 + 0.5) / (1u64 << 53) as f64
            };
            let a = uniform();
            let b = uniform();
            (-2. * a.ln()).sqrt() * (std::f64::consts::TAU * b).cos()
        };
        let n = 60000;
        let mut mean = [0.; 4];
        let mut cross = [0.; 4];
        for _ in 0..n {
            let u: Vec<C> = (0..170)
                .map(|_| C::new(normal(), normal()) / 2f64.sqrt())
                .collect();
            let x: Vec<C> = (0..160)
                .map(|i| (0..3).map(|k| h[k] * u[i + 3 - k]).sum())
                .collect();
            let q = |i: usize, l: usize| {
                let lo = i.saturating_sub(l).max(30);
                (lo..i + 21).map(|k| x[k] * x[k + l].conj()).sum::<C>() / (i + 21 - lo) as f64
                    * 200.
                    / (200 - l) as f64
            };
            let a = q(20, 1);
            let b = q(40, 2);
            for (k, z) in [a.re, a.im, b.re, b.im].iter().enumerate() {
                mean[k] += z / n as f64;
            }
            for (k, z) in [a.re * b.re, a.re * b.im, a.im * b.re, a.im * b.im]
                .iter()
                .enumerate()
            {
                cross[k] += z / n as f64;
            }
        }
        let estimate = [
            cross[0] - mean[0] * mean[2],
            cross[1] - mean[0] * mean[3],
            cross[2] - mean[1] * mean[2],
            cross[3] - mean[1] * mean[3],
        ];
        for i in 0..4 {
            assert!(
                (estimate[i] - want[i]).abs() < 0.001,
                "entry {i}: {} vs {}",
                estimate[i],
                want[i]
            );
        }
    }
}
