//! Conservative nonthermal-echo screening before incoherent lag products.
//! Flags are satellite/RFI candidates, not an object classification. Raw is unchanged.
use num_complex::Complex32 as C;
use serde::{Deserialize, Serialize};
#[derive(Clone, Serialize, Deserialize)]
pub struct Settings {
    pub range_bin: usize,
    pub slow_pulses: usize,
    pub bright_ratio: f32,
    pub coherent_ratio: f32,
    pub slow_coherence: f32,
    pub fast_ratio: f32,
    pub fast_coherence: f32,
    pub range_margin: usize,
    pub pulse_margin: usize,
}
impl Default for Settings {
    fn default() -> Self {
        Self {
            range_bin: 8,
            slow_pulses: 8,
            bright_ratio: 64.,
            coherent_ratio: 4.,
            slow_coherence: 0.75,
            fast_ratio: 12.,
            fast_coherence: 0.92,
            range_margin: 16,
            pulse_margin: 2,
        }
    }
}
pub struct Screen {
    pub mask: Vec<u8>,
    pub reference: Vec<f32>,
    pub power_ratio: Vec<f32>,
    pub slow_coherence: Vec<f32>,
    pub flagged_cells: [usize; 3],
    pub range_bins: usize,
    pub time_bins: usize,
    pub masked_samples: usize,
}
fn median(x: &mut [f32]) -> f32 {
    let n = x.len();
    x.select_nth_unstable_by(n / 2, |a, b| a.total_cmp(b));
    x[n / 2]
}
fn fast_coherence(x: &[C]) -> f32 {
    let r: C = x.windows(2).map(|a| a[1] * a[0].conj()).sum();
    let phase = libm::atan2f(r.im, r.re);
    let step = C::from_polar(1., -phase);
    let mut oscillator = C::new(1., 0.);
    let mut sum = C::default();
    let mut energy = 0.;
    for &z in x {
        sum += z * oscillator;
        energy += z.norm_sqr();
        oscillator *= step;
    }
    sum.norm_sqr() / (x.len() as f32 * energy).max(1e-30)
}
pub fn screen(x: &[C], pulses: usize, n: usize, s: &Settings) -> Screen {
    assert_eq!(x.len(), pulses * n);
    assert!(pulses > 0 && s.range_bin > 0 && s.slow_pulses > 1);
    let nb = n / s.range_bin;
    let nt = (pulses + s.slow_pulses - 1) / s.slow_pulses;
    let mut power = vec![0f32; pulses * nb];
    for p in 0..pulses {
        for b in 0..nb {
            power[p * nb + b] = x[p * n + b * s.range_bin..p * n + (b + 1) * s.range_bin]
                .iter()
                .map(|z| z.norm_sqr())
                .sum::<f32>()
                / s.range_bin as f32;
        }
    }
    let mut reference = vec![0.; nb];
    let mut v = vec![0.; pulses];
    for b in 0..nb {
        for p in 0..pulses {
            v[p] = power[p * nb + b];
        }
        reference[b] = median(&mut v).max(1e-20);
    }
    let mut mask = vec![0u8; pulses * n];
    let mut flags = [0usize; 3];
    let mut candidate = vec![false; pulses * nb];
    // Very bright, localized power can be removed without a coherent-phase assumption.
    for p in 0..pulses {
        for b in 0..nb {
            let ratio = power[p * nb + b] / reference[b];
            if ratio > s.bright_ratio {
                flags[0] += 1;
                for k in b * s.range_bin..(b + 1) * s.range_bin {
                    mask[p * n + k] |= 1;
                }
            }
            // A point-like fast-time sinusoid helps when the transmitter phase is not
            // coherent across IPPs. Require three successive pulses below before accepting.
            if ratio > s.fast_ratio {
                let k = (b * s.range_bin).saturating_sub(20).min(n - 48);
                candidate[p * nb + b] =
                    fast_coherence(&x[p * n + k..p * n + k + 48]) > s.fast_coherence;
            }
        }
    }
    for p in 1..pulses.saturating_sub(1) {
        for b in 0..nb {
            if candidate[(p - 1) * nb + b] && candidate[p * nb + b] && candidate[(p + 1) * nb + b] {
                flags[2] += 1;
                for q in p - 1..=p + 1 {
                    for k in b * s.range_bin..(b + 1) * s.range_bin {
                        mask[q * n + k] |= 4;
                    }
                }
            }
        }
    }
    let mut ratios = vec![0.; nt * nb];
    let mut coherence = vec![0.; nt * nb];
    for t in 0..nt {
        let a = t * s.slow_pulses;
        let z = (a + s.slow_pulses).min(pulses);
        for b in 0..nb {
            let ratio =
                (a..z).map(|p| power[p * nb + b]).sum::<f32>() / ((z - a) as f32 * reference[b]);
            ratios[t * nb + b] = ratio;
            if z - a < 4 {
                continue;
            }
            let mut cross = C::default();
            let mut ea = 0.;
            let mut eb = 0.;
            for p in a..z - 1 {
                for k in b * s.range_bin..(b + 1) * s.range_bin {
                    let u = x[p * n + k];
                    let v = x[(p + 1) * n + k];
                    cross += u * v.conj();
                    ea += u.norm_sqr();
                    eb += v.norm_sqr();
                }
            }
            let rho = cross.norm() / (ea * eb).sqrt().max(1e-20);
            coherence[t * nb + b] = rho;
            if ratio > s.coherent_ratio && rho > s.slow_coherence {
                flags[1] += 1;
                for p in a..z {
                    for k in b * s.range_bin..(b + 1) * s.range_bin {
                        mask[p * n + k] |= 2;
                    }
                }
            }
        }
    }
    // Dilation is applied to voltage-sample masks, not to an inverted ACF.
    let original = mask.clone();
    for p in 0..pulses {
        let mut k = 0;
        while k < n {
            if original[p * n + k] == 0 {
                k += 1;
                continue;
            }
            let start = k;
            let mut reason = 0;
            while k < n && original[p * n + k] != 0 {
                reason |= original[p * n + k];
                k += 1;
            }
            for q in p.saturating_sub(s.pulse_margin)..=(p + s.pulse_margin).min(pulses - 1) {
                for j in start.saturating_sub(s.range_margin)..(k + s.range_margin).min(n) {
                    mask[q * n + j] |= reason;
                }
            }
        }
    }
    let masked_samples = mask.iter().filter(|&&v| v != 0).count();
    Screen {
        mask,
        reference,
        power_ratio: ratios,
        slow_coherence: coherence,
        flagged_cells: flags,
        range_bins: nb,
        time_bins: nt,
        masked_samples,
    }
}
#[cfg(test)]
mod tests {
    use super::*;
    struct Rng(u64);
    impl Rng {
        fn uniform(&mut self) -> f32 {
            self.0 = self.0.wrapping_mul(6364136223846793005).wrapping_add(1);
            (((self.0 >> 32) as u32) as f64 / 4294967296.) as f32
        }
        fn complex(&mut self) -> C {
            let r = (-self.uniform().max(1e-9).ln()).sqrt();
            C::from_polar(r, std::f32::consts::TAU * self.uniform())
        }
    }
    fn thermal(p: usize, n: usize, rho: f32) -> Vec<C> {
        let mut rng = Rng(491375);
        let mut x = vec![C::default(); p * n];
        for i in 0..p {
            let mut prev = rng.complex();
            for k in 0..n {
                prev = rho * prev + (1. - rho * rho).sqrt() * rng.complex();
                let amp = 1. + 8. * (-((k as f32 - 100.) / 40.).powi(2)).exp();
                x[i * n + k] = prev * amp;
            }
        }
        x
    }
    #[test]
    fn colored_thermal_scatter_is_preserved() {
        for rho in [0., 0.7, 0.98] {
            let p = 2048;
            let n = 256;
            let x = thermal(p, n, rho);
            let got = screen(&x, p, n, &Settings::default());
            assert!(
                (got.masked_samples as f64 / (p * n) as f64) < 0.002,
                "rho={rho} masked={}",
                got.masked_samples
            );
        }
    }
    #[test]
    fn injected_moving_coherent_echo_is_masked() {
        let p = 2048;
        let n = 256;
        let mut x = thermal(p, n, 0.6);
        let mut affected = vec![false; p * n];
        for i in 700..900 {
            let k0 = 150 + (i - 700) / 20;
            for k in k0..k0 + 48 {
                x[i * n + k] += C::from_polar(12., 0.7 * i as f32 + 0.43 * k as f32);
                affected[i * n + k] = true;
            }
        }
        let got = screen(&x, p, n, &Settings::default());
        let covered = affected
            .iter()
            .zip(&got.mask)
            .filter(|&(a, b)| *a && *b != 0)
            .count();
        assert!(
            covered as f64 / 9600. > 0.95,
            "coverage {}",
            covered as f64 / 9600.
        );
        let outside = affected
            .iter()
            .zip(&got.mask)
            .filter(|&(a, b)| !*a && *b != 0)
            .count();
        assert!((outside as f64 / ((p * n) - 9600) as f64) < 0.04);
    }
    #[test]
    fn bright_incoherent_burst_is_masked() {
        let p = 256;
        let n = 256;
        let mut x = thermal(p, n, 0.);
        for k in 180..195 {
            x[100 * n + k] *= 100.;
        }
        let got = screen(&x, p, n, &Settings::default());
        assert!((180..195).all(|k| got.mask[100 * n + k] != 0));
    }
}
