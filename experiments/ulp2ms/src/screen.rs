//! Conservative whole-IPP bright-echo guard; raw voltages remain unchanged.
use num_complex::Complex32;
fn tone_fraction(x: &[Complex32]) -> f64 {
    use num_complex::Complex64 as C;
    let z: Vec<C> = x.iter().map(|v| C::new(v.re as f64, v.im as f64)).collect();
    let cross: C = z.windows(2).map(|v| v[1] * v[0].conj()).sum();
    let step = C::from_polar(1., -cross.arg());
    let mut phase = C::new(1., 0.);
    let mut sum = C::new(0., 0.);
    let mut power = 0.;
    for v in z {
        sum += v * phase;
        power += v.norm_sqr();
        phase *= step;
    }
    sum.norm_sqr() / (x.len() as f64 * power).max(1e-30)
}
pub fn bright_mask(
    frames: &[Vec<Complex32>],
    start: usize,
    end: usize,
    width: usize,
    threshold: f64,
) -> (Vec<bool>, Vec<f64>) {
    let bins = (end - start) / width;
    let mut ratio = vec![0f64; frames.len()];
    let mut coherent = vec![false; frames.len()];
    for b in 0..bins {
        let k = start + b * width;
        let p: Vec<f64> = frames
            .iter()
            .map(|f| {
                f[k..k + width]
                    .iter()
                    .map(|z| (z.re as f64).powi(2) + (z.im as f64).powi(2))
                    .sum::<f64>()
                    / width as f64
            })
            .collect();
        let mut sorted = p.clone();
        let n = sorted.len();
        sorted.select_nth_unstable_by(n / 2, |a, b| a.total_cmp(b));
        let median = sorted[n / 2];
        if median > 0. {
            let candidate: Vec<bool> = (0..p.len())
                .map(|i| {
                    let a = k.saturating_sub(16).min(frames[i].len() - 48);
                    p[i] / median > 12. && tone_fraction(&frames[i][a..a + 48]) > 0.92
                })
                .collect();
            for i in 0..p.len() {
                ratio[i] = ratio[i].max(p[i] / median);
                if i > 0 && i + 1 < p.len() && candidate[i - 1] && candidate[i] && candidate[i + 1]
                {
                    coherent[i] = true;
                }
            }
        }
    }
    let mut keep = vec![true; frames.len()];
    for i in 0..frames.len() {
        if ratio[i] > threshold || coherent[i] {
            for k in i.saturating_sub(2)..=(i + 2).min(frames.len() - 1) {
                keep[k] = false;
            }
        }
    }
    (keep, ratio)
}
#[cfg(test)]
mod tests {
    use super::*;
    #[test]
    fn isolated_echo_is_removed_with_pulse_margin() {
        let mut f = vec![vec![Complex32::new(1., 0.); 64]; 101];
        f[50][32] = Complex32::new(100., 0.);
        let (k, _) = bright_mask(&f, 16, 64, 16, 64.);
        assert_eq!(k.iter().filter(|&&v| !v).count(), 5);
        assert!(!k[48] && !k[52]);
    }
    #[test]
    fn stationary_range_structure_is_preserved() {
        let mut f = vec![vec![Complex32::new(1., 0.); 64]; 101];
        for p in &mut f {
            p[32] = Complex32::new(100., 0.);
        }
        let (k, _) = bright_mask(&f, 16, 64, 16, 64.);
        assert!(k.iter().all(|&v| v));
    }
    #[test]
    fn thermal_noise_preserved_and_weaker_persistent_tone_rejected() {
        let mut seed = 91237u64;
        let mut uniform = || {
            seed = seed.wrapping_mul(6364136223846793005).wrapping_add(1);
            ((seed >> 32) as f64 + 0.5) / 4294967296.
        };
        let mut frames = vec![vec![Complex32::default(); 256]; 1024];
        for row in &mut frames {
            for v in row {
                let radius = (-uniform().ln()).sqrt() as f32;
                *v = Complex32::from_polar(radius, (std::f64::consts::TAU * uniform()) as f32);
            }
        }
        let (keep, _) = bright_mask(&frames, 32, 240, 16, 64.);
        assert!(keep.iter().all(|&v| v));
        for p in 500..510 {
            for k in 64..192 {
                frames[p][k] += Complex32::from_polar(5., 0.3 * k as f32);
            }
        }
        let (keep, ratio) = bright_mask(&frames, 32, 240, 16, 64.);
        assert!(ratio[500..510].iter().all(|&v| v < 64.));
        assert!(keep[500..510].iter().all(|&v| !v));
    }
}
