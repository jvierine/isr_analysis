//! Within-pulse spectra for a 480-us uncoded pulse, with explicit normalization.
use anyhow::{ensure, Result};
use num_complex::Complex32;
use rustfft::Fft;

pub const SAMPLE_HZ: f64 = 100000.;
pub const FAST_SAMPLES: usize = 48;
pub const FFT_BINS: usize = 256;
pub const RANGE_HOP: usize = 12;

pub fn hann() -> Vec<f32> {
    (0..FAST_SAMPLES).map(|i|(0.5-0.5*(2.*std::f64::consts::PI*i as f64/(FAST_SAMPLES-1) as f64).cos()) as f32).collect()
}

/// Add the two-sided |FFT(w*x)|^2/(fs*sum(w^2)) periodogram. The input
/// convention I+iQ gives a positive-frequency tone at positive FFT frequency.
pub fn add_periodogram(iq: &[Complex32], w: &[f32], fft: &dyn Fft<f32>,
    buffer: &mut [Complex32], scratch: &mut [Complex32], sum: &mut [f64]) {
    assert_eq!(iq.len(),w.len());assert_eq!(sum.len(),buffer.len());
    buffer.fill(Complex32::new(0.,0.));
    for ((b,x),weight) in buffer.iter_mut().zip(iq).zip(w) {*b=*x**weight;}
    fft.process_with_scratch(buffer,scratch);
    let scale=1./(SAMPLE_HZ*w.iter().map(|x|(*x as f64).powi(2)).sum::<f64>());
    let n=sum.len();
    for (i,s) in sum.iter_mut().enumerate() {
        let z=buffer[(i+n/2)%n];
        *s+=(z.re as f64*z.re as f64+z.im as f64*z.im as f64)*scale;
    }
}

pub fn normalized(spectrum: &[f64]) -> Result<(Vec<f64>,f64)> {
    ensure!(spectrum.len()==FFT_BINS && spectrum.iter().all(|x|x.is_finite() && *x>=0.),"invalid PSD");
    let area=spectrum.iter().sum::<f64>()*SAMPLE_HZ/FFT_BINS as f64;
    ensure!(area>0.,"zero integrated PSD");
    Ok((spectrum.iter().map(|x|x/area).collect(),area))
}

#[cfg(test)]
mod tests {
    use super::*;
    #[test]
    fn signed_tones_and_parseval_after_zero_padding() {
        let mut planner=rustfft::FftPlanner::new();let fft=planner.plan_fft_forward(FFT_BINS);
        let w=hann();
        for bin in [-32_i32,0,32] {
            let iq:Vec<_>=(0..FAST_SAMPLES).map(|j|Complex32::from_polar(3.,2.*std::f32::consts::PI*bin as f32*j as f32/FFT_BINS as f32)).collect();
            let mut result=vec![0.;FFT_BINS];let mut b=vec![Complex32::new(0.,0.);FFT_BINS];
            let mut scratch=vec![Complex32::new(0.,0.);fft.get_inplace_scratch_len()];
            add_periodogram(&iq,&w,fft.as_ref(),&mut b,&mut scratch,&mut result);
            let peak=result.iter().enumerate().max_by(|a,b|a.1.total_cmp(b.1)).unwrap().0;
            assert_eq!(peak,(bin+128) as usize);
            let (norm,area)=normalized(&result).unwrap();
            assert!((area-9.).abs()<1e-5);
            assert!((norm.iter().sum::<f64>()*SAMPLE_HZ/FFT_BINS as f64-1.).abs()<1e-12);
        }
    }
}
