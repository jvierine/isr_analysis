//! Instrument-independent within-pulse spectra. Drivers supply decoded I+iQ,
//! UTC pulse times, beam grouping and the physical experiment configuration.
//! This does not perform a slow-time Doppler FFT or decode coded experiments.
use anyhow::{ensure, Result};
use num_complex::Complex32;
use rustfft::{Fft, FftPlanner};
use std::sync::Arc;

pub const VERSION: &str = env!("CARGO_PKG_VERSION");

pub fn hann(samples: usize) -> Result<Vec<f32>> {
    ensure!(
        samples >= 3,
        "Symmetric Hann window requires at least three samples"
    );
    Ok((0..samples)
        .map(|i| {
            (0.5 - 0.5 * (2. * std::f64::consts::PI * i as f64 / (samples - 1) as f64).cos()) as f32
        })
        .collect())
}

/// Shifted two-sided density, |FFT(w*x)|²/(fs sum(w²)), in input units²/Hz.
/// A positive phase rotation of I+iQ appears at positive FFT frequency.
#[derive(Clone)]
pub struct PeriodogramPlan {
    sample_rate_hz: f64,
    weights: Vec<f32>,
    fft: Arc<dyn Fft<f32>>,
    scale: f64,
}
pub struct Workspace {
    buffer: Vec<Complex32>,
    scratch: Vec<Complex32>,
}
impl PeriodogramPlan {
    pub fn new(sample_rate_hz: f64, weights: Vec<f32>, fft_bins: usize) -> Result<Self> {
        ensure!(
            sample_rate_hz.is_finite() && sample_rate_hz > 0.,
            "Invalid sample rate"
        );
        ensure!(
            !weights.is_empty() && fft_bins >= weights.len(),
            "FFT cannot truncate the measured window"
        );
        ensure!(weights.iter().all(|w| w.is_finite()), "Nonfinite taper");
        let energy = weights.iter().map(|w| (*w as f64).powi(2)).sum::<f64>();
        ensure!(energy > 0., "Zero-energy taper");
        let fft = FftPlanner::new().plan_fft_forward(fft_bins);
        Ok(Self {
            sample_rate_hz,
            weights,
            fft,
            scale: 1. / (sample_rate_hz * energy),
        })
    }
    pub fn workspace(&self) -> Workspace {
        Workspace {
            buffer: vec![Complex32::new(0., 0.); self.fft.len()],
            scratch: vec![Complex32::new(0., 0.); self.fft.get_inplace_scratch_len()],
        }
    }
    pub fn weights(&self) -> &[f32] {
        &self.weights
    }
    pub fn bins(&self) -> usize {
        self.fft.len()
    }
    pub fn bin_width_hz(&self) -> f64 {
        self.sample_rate_hz / self.bins() as f64
    }
    pub fn frequency_hz(&self) -> Vec<f64> {
        (0..self.bins())
            .map(|i| (i as f64 - (self.bins() / 2) as f64) * self.bin_width_hz())
            .collect()
    }
    pub fn add(&self, iq: &[Complex32], work: &mut Workspace, sum: &mut [f64]) -> Result<()> {
        ensure!(
            iq.len() == self.weights.len()
                && sum.len() == self.bins()
                && work.buffer.len() == self.bins()
                && work.scratch.len() >= self.fft.get_inplace_scratch_len(),
            "Periodogram shape mismatch"
        );
        ensure!(
            iq.iter().all(|x| x.re.is_finite() && x.im.is_finite()),
            "Nonfinite IQ"
        );
        work.buffer.fill(Complex32::new(0., 0.));
        for ((b, x), w) in work.buffer.iter_mut().zip(iq).zip(&self.weights) {
            *b = *x * *w;
        }
        self.fft
            .process_with_scratch(&mut work.buffer, &mut work.scratch);
        let n = sum.len();
        for (i, s) in sum.iter_mut().enumerate() {
            let z = work.buffer[(i + (n + 1) / 2) % n];
            *s += (z.re as f64 * z.re as f64 + z.im as f64 * z.im as f64) * self.scale;
        }
        Ok(())
    }
}

/// Overlapping fast-time gates. Gate spacing is not physical range resolution.
#[derive(Clone)]
pub struct RangeDopplerPlan {
    spectrum: PeriodogramPlan,
    gate_starts: Vec<usize>,
    samples: usize,
}
impl RangeDopplerPlan {
    pub fn spectrum(&self) -> &PeriodogramPlan {
        &self.spectrum
    }
    pub fn gate_starts(&self) -> &[usize] {
        &self.gate_starts
    }
    pub fn new(samples: usize, hop_samples: usize, spectrum: PeriodogramPlan) -> Result<Self> {
        ensure!(
            hop_samples > 0 && samples >= spectrum.weights.len(),
            "Invalid range gate layout"
        );
        let gate_starts = (0..=samples - spectrum.weights.len())
            .step_by(hop_samples)
            .collect();
        Ok(Self {
            spectrum,
            gate_starts,
            samples,
        })
    }
    /// Adds one decoded pulse to a [range, frequency] sum. The driver decides
    /// beam assignment, exposure rejection and time integration boundaries.
    pub fn add_pulse(&self, iq: &[Complex32], work: &mut Workspace, sum: &mut [f64]) -> Result<()> {
        ensure!(
            iq.len() == self.samples && sum.len() == self.gate_starts.len() * self.spectrum.bins(),
            "Range-Doppler shape mismatch"
        );
        for (&gate, row) in self
            .gate_starts
            .iter()
            .zip(sum.chunks_mut(self.spectrum.bins()))
        {
            self.spectrum
                .add(&iq[gate..gate + self.spectrum.weights.len()], work, row)?;
        }
        Ok(())
    }
    /// Monostatic slant range at the taper's power centroid. Delay is measured
    /// from the transmit leading edge; subtract half the transmit duration.
    /// Receiver delay/calibration must be supplied explicitly by the driver.
    pub fn slant_range_km(
        &self,
        gate_front_us: f64,
        transmit_us: f64,
        receiver_delay_us: f64,
    ) -> Result<Vec<f64>> {
        ensure!(
            gate_front_us.is_finite()
                && transmit_us.is_finite()
                && transmit_us >= 0.
                && receiver_delay_us.is_finite(),
            "Invalid timing"
        );
        let energy = self
            .spectrum
            .weights
            .iter()
            .map(|w| (*w as f64).powi(2))
            .sum::<f64>();
        let centroid = self
            .spectrum
            .weights
            .iter()
            .enumerate()
            .map(|(i, w)| i as f64 * (*w as f64).powi(2))
            .sum::<f64>()
            / energy;
        Ok(self
            .gate_starts
            .iter()
            .map(|&j| {
                (gate_front_us + (j as f64 + centroid) * 1e6 / self.spectrum.sample_rate_hz
                    - transmit_us / 2.
                    - receiver_delay_us)
                    * 0.149896229
            })
            .collect())
    }
}

/// Full-band positive-PSD normalization. Background-subtracted signed spectra
/// should use `normalize_signed_l1` instead of dividing by a near-zero sum.
pub fn normalize_positive(spectrum: &[f64], bin_hz: f64) -> Result<(Vec<f64>, f64)> {
    ensure!(
        !spectrum.is_empty()
            && bin_hz.is_finite()
            && bin_hz > 0.
            && spectrum.iter().all(|x| x.is_finite() && *x >= 0.),
        "Invalid positive PSD"
    );
    let area = spectrum.iter().sum::<f64>() * bin_hz;
    ensure!(area.is_finite() && area > 0., "Invalid integrated PSD");
    Ok((spectrum.iter().map(|x| x / area).collect(), area))
}
pub fn normalize_signed_l1(spectrum: &[f64], bin_hz: f64) -> Result<Vec<f64>> {
    ensure!(
        !spectrum.is_empty()
            && bin_hz.is_finite()
            && bin_hz > 0.
            && spectrum.iter().all(|x| x.is_finite()),
        "Invalid signed spectrum"
    );
    let area = spectrum.iter().map(|x| x.abs()).sum::<f64>() * bin_hz;
    ensure!(area.is_finite(), "Nonfinite integrated magnitude");
    Ok(spectrum
        .iter()
        .map(|x| if area > 0. { x / area } else { 0. })
        .collect())
}

/// Fixed wall-clock windows with an end-anchored final window. The last pulse
/// occupies `pulse_period_s`; a short recording uses its available duration.
pub fn integration_windows(
    times_utc_s: &[f64],
    duration_s: f64,
    step_s: f64,
    pulse_period_s: f64,
) -> Result<(Vec<f64>, f64)> {
    ensure!(
        !times_utc_s.is_empty()
            && times_utc_s.iter().all(|t| t.is_finite())
            && times_utc_s.windows(2).all(|t| t[1] >= t[0]),
        "Pulse times must be finite and ordered"
    );
    ensure!(
        [duration_s, step_s, pulse_period_s]
            .iter()
            .all(|t| t.is_finite() && *t > 0.),
        "Invalid integration timing"
    );
    let t0 = times_utc_s[0];
    let end = times_utc_s[times_utc_s.len() - 1] + pulse_period_s;
    let window = duration_s.min(end - t0);
    let mut starts = Vec::new();
    let mut start = t0;
    ensure!(
        window > 0. && t0 + step_s > t0,
        "Integration timing below timestamp precision"
    );
    while start + window <= end {
        starts.push(start);
        let next = start + step_s;
        ensure!(next > start, "Integration step below timestamp precision");
        start = next;
    }
    let last = end - window;
    if starts.is_empty() || last - *starts.last().unwrap() > 0.001 {
        starts.push(last);
    }
    Ok((starts, window))
}

#[cfg(test)]
mod tests {
    use super::*;
    #[test]
    fn signed_tones_parseval_and_different_experiments() {
        for (fs, n, nfft) in [(100000., 48, 256), (200000., 64, 128), (90000., 33, 99)] {
            let plan = PeriodogramPlan::new(fs, hann(n).unwrap(), nfft).unwrap();
            for bin in [-12, 0, 12] {
                let iq: Vec<_> = (0..n)
                    .map(|j| {
                        Complex32::from_polar(
                            3.,
                            2. * std::f32::consts::PI * bin as f32 * j as f32 / nfft as f32,
                        )
                    })
                    .collect();
                let mut sum = vec![0.; nfft];
                plan.add(&iq, &mut plan.workspace(), &mut sum).unwrap();
                let peak = sum
                    .iter()
                    .enumerate()
                    .max_by(|a, b| a.1.total_cmp(b.1))
                    .unwrap()
                    .0;
                assert_eq!(peak, (bin + nfft as i32 / 2) as usize);
                let (normalized, area) = normalize_positive(&sum, plan.bin_width_hz()).unwrap();
                assert!((area - 9.).abs() < 1e-5);
                assert!((normalized.iter().sum::<f64>() * plan.bin_width_hz() - 1.).abs() < 1e-12);
            }
        }
    }
    #[test]
    fn gate_layout_timing_and_signed_power() {
        let plan = RangeDopplerPlan::new(
            100,
            12,
            PeriodogramPlan::new(1e5, hann(48).unwrap(), 256).unwrap(),
        )
        .unwrap();
        assert_eq!(plan.gate_starts, vec![0, 12, 24, 36, 48]);
        let r = plan.slant_range_km(533., 480., 160.).unwrap();
        assert!((r[0] - (533. + 235. - 240. - 160.) * 0.149896229).abs() < 1e-10);
        let v = normalize_signed_l1(&[-2., 1.], 10.).unwrap();
        assert!(v[0] < 0.);
        assert!((v.iter().map(|x| x.abs()).sum::<f64>() * 10. - 1.).abs() < 1e-12);
        assert_eq!(normalize_signed_l1(&[0., 0.], 1.).unwrap(), vec![0., 0.]);
    }
    #[test]
    fn wall_clock_gaps_and_short_recordings() {
        let (starts, w) = integration_windows(&[0., 1., 2., 8., 9.], 5., 3., 1.).unwrap();
        assert_eq!(starts, vec![0., 3., 5.]);
        assert_eq!(w, 5.);
        assert_eq!(
            integration_windows(&[4., 4.1], 300., 150., 0.1).unwrap().0,
            vec![4.]
        );
        assert!(integration_windows(&[2., 1.], 5., 3., 1.).is_err());
        assert!(PeriodogramPlan::new(0., vec![1.], 16).is_err());
        assert!(PeriodogramPlan::new(1e5, vec![1.; 32], 16).is_err());
        assert!(normalize_positive(&[-1., 2.], 1.).is_err());
    }
}
