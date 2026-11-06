//! Measured-waveform diagnostics. These identify candidates, not independently
//! known transmit timing: a repeated ground echo can also resemble a pulse.
use anyhow::{ensure, Result};
use num_complex::{Complex32, Complex64};
use serde::{Deserialize, Serialize};
#[derive(Clone, Debug, Serialize, Deserialize)]
pub struct WaveformStatistics {
    pub energy: f64,
    pub carrier_offset_hz: f64,
    pub derotated_coherence: f64,
    pub clipped_samples: usize,
    pub waveform_class: String,
}
pub fn waveform_statistics(z: &[Complex32], sample_rate_hz: f64) -> Result<WaveformStatistics> {
    ensure!(
        z.len() > 2 && sample_rate_hz.is_finite() && sample_rate_hz > 0.,
        "invalid waveform sampling"
    );
    ensure!(
        z.iter().all(|v| v.re.is_finite() && v.im.is_finite()),
        "invalid waveform voltage"
    );
    let energy = z.iter().map(|v| v.norm_sqr() as f64).sum::<f64>();
    ensure!(energy > 0., "empty waveform");
    let cross = z
        .windows(2)
        .map(|w| {
            let a = w[1];
            let b = w[0];
            Complex64::new(a.re as f64, a.im as f64) * Complex64::new(b.re as f64, -b.im as f64)
        })
        .sum::<Complex64>();
    let omega = cross.arg();
    let step = Complex64::from_polar(1., -omega);
    let mut phase = Complex64::new(1., 0.);
    let mut sum = Complex64::default();
    let mut amplitude = 0.;
    for v in z {
        sum += Complex64::new(v.re as f64, v.im as f64) * phase;
        amplitude += v.norm() as f64;
        phase *= step;
    }
    let coherence = sum.norm() / amplitude;
    Ok(WaveformStatistics {
        energy,
        carrier_offset_hz: omega * sample_rate_hz / std::f64::consts::TAU,
        derotated_coherence: coherence,
        clipped_samples: z
            .iter()
            .filter(|v| v.re.abs() >= 32767. || v.im.abs() >= 32767.)
            .count(),
        waveform_class: if coherence > 0.9 {
            "uncoded candidate"
        } else if coherence < 0.5 {
            "coded candidate"
        } else {
            "uncertain"
        }
        .into(),
    })
}
/// Median across IPPs after within-bin averaging. Unlike a folded mean, one
/// exceptionally bright satellite or heating burst cannot set pulse timing.
pub fn median_fold(z: &[Complex32], ipp: usize, bin: usize) -> Result<Vec<f32>> {
    ensure!(
        ipp > 0 && bin > 0 && ipp % bin == 0 && z.len() / ipp >= 3,
        "invalid folded-envelope layout"
    );
    let pulses = z.len() / ipp;
    let bins = ipp / bin;
    let mut powers = vec![vec![0_f32; pulses]; bins];
    for (p, echo) in z.chunks_exact(ipp).enumerate() {
        for (i, chunk) in echo.chunks_exact(bin).enumerate() {
            powers[i][p] =
                (chunk.iter().map(|v| v.norm_sqr() as f64).sum::<f64>() / bin as f64) as f32;
        }
    }
    Ok(powers
        .into_iter()
        .map(|mut p| {
            let middle = p.len() / 2;
            *p.select_nth_unstable_by(middle, f32::total_cmp).1
        })
        .collect())
}
#[derive(Clone, Debug, Serialize, Deserialize)]
pub struct PulseCandidate {
    pub first_bin: usize,
    pub width_bins: usize,
    pub peak_power: f32,
    pub nominal_width_compatible: bool,
}
pub fn pulse_candidates(
    envelope: &[f32],
    nominal_width_bins: usize,
) -> Result<Vec<PulseCandidate>> {
    ensure!(
        envelope.len() > 3
            && nominal_width_bins > 0
            && envelope.iter().all(|v| v.is_finite() && *v >= 0.),
        "invalid pulse envelope"
    );
    let mut sorted = envelope.to_vec();
    sorted.sort_by(f32::total_cmp);
    let median = sorted[sorted.len() / 2];
    let mut deviations: Vec<_> = sorted.iter().map(|v| (*v - median).abs()).collect();
    deviations.sort_by(f32::total_cmp);
    let mad = deviations[deviations.len() / 2];
    let threshold = (20. * median)
        .max(median + 15. * mad)
        .max(sorted.last().unwrap() * 1e-6);
    let mask: Vec<_> = envelope.iter().map(|v| *v > threshold).collect();
    let anchor = mask
        .iter()
        .position(|v| !*v)
        .ok_or_else(|| anyhow::anyhow!("no off-pulse support; timing is ambiguous"))?;
    let mut candidates = vec![];
    let mut start = None;
    let mut width = 0;
    let mut peak = 0_f32;
    for step in 1..=envelope.len() {
        let i = (anchor + step) % envelope.len();
        if mask[i] {
            if start.is_none() {
                start = Some(i);
            }
            width += 1;
            peak = peak.max(envelope[i]);
        } else if let Some(first) = start.take() {
            candidates.push(PulseCandidate {
                first_bin: first,
                width_bins: width,
                peak_power: peak,
                nominal_width_compatible: width.abs_diff(nominal_width_bins)
                    <= nominal_width_bins / 5,
            });
            width = 0;
            peak = 0.;
        }
    }
    Ok(candidates)
}
#[cfg(test)]
mod tests {
    use super::*;
    #[test]
    fn carrier_offset_does_not_turn_an_uncoded_pulse_into_a_code() {
        let fs = 25e6;
        let omega = 0.12;
        let plain: Vec<_> = (0..10000)
            .map(|i| Complex32::from_polar(8., omega * i as f32))
            .collect();
        let a = waveform_statistics(&plain, fs).unwrap();
        assert!(a.derotated_coherence > 0.9999);
        assert!((a.carrier_offset_hz - omega as f64 * fs / std::f64::consts::TAU).abs() < 5.);
        let coded: Vec<_> = plain
            .iter()
            .enumerate()
            .map(|(i, v)| if i / 500 % 2 == 0 { *v } else { -*v })
            .collect();
        assert!(waveform_statistics(&coded, fs).unwrap().derotated_coherence < 0.01);
    }
    #[test]
    fn rare_bright_echo_does_not_move_the_folded_pulse() {
        let mut z = vec![Complex32::new(1., 0.); 10 * 100];
        for p in 0..10 {
            for i in 30..40 {
                z[p * 100 + i] = Complex32::new(100., 0.);
            }
        }
        for i in 70..80 {
            z[i] = Complex32::new(10000., 0.);
        }
        let folded = median_fold(&z, 100, 1).unwrap();
        let candidates = pulse_candidates(&folded, 10).unwrap();
        assert_eq!(candidates.len(), 1);
        assert_eq!(candidates[0].first_bin, 30);
    }
    #[test]
    fn repeated_ground_echo_is_reported_as_an_ambiguity() {
        let mut p = vec![1.; 100];
        p[20..30].fill(100.);
        p[60..70].fill(1000.);
        let c = pulse_candidates(&p, 10).unwrap();
        assert_eq!(c.len(), 2);
        assert!(c.iter().all(|c| c.nominal_width_compatible));
    }
}
