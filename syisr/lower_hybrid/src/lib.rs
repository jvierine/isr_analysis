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

/// Conservative transient morphology screen on one-second, same-beam PSDs.
/// Exclude whole time/range samples, never selected frequency bins. The
/// unscreened estimate and the exact exposure mask must accompany the result.
pub fn transient_mask(seconds:&[f64], counts:&[u32], nr:usize, protected_gate:Option<usize>)->Vec<u8> {
    let nt=counts.len();let plane=nr*FFT_BINS;assert_eq!(seconds.len(),nt*plane);
    // Point-target ambiguity power for the actual 480-us rectangular transmit
    // pulse and 48-sample Hann receiver window, including range truncation.
    let template=pulse_ambiguity_template();
    let mut reference=vec![0.;plane];let mut work=Vec::with_capacity(nt);
    for j in 0..plane {
        work.clear();for t in 0..nt {if counts[t]>0 {work.push(seconds[t*plane+j]/counts[t] as f64);}}
        let k=work.len()/2;reference[j]=*work.select_nth_unstable_by(k,f64::total_cmp).1;
    }
    let mut mask=vec![0_u8;nt*nr];
    for t in 0..nt {
        if counts[t]==0 {continue;}
        let mut hot=vec![false;plane];
        for j in 0..plane {hot[j]=seconds[t*plane+j]/counts[t] as f64>8.*reference[j].max(1e-30);}
        // Connected components describe a localized long-pulse range/Doppler
        // blip. Broad common interference is deliberately not classified here.
        for seed in 0..plane {
            if !hot[seed] {continue;}hot[seed]=false;
            let mut queue=vec![seed];let(mut r0,mut r1,mut f0,mut f1)=(nr,0,FFT_BINS,0);let mut pos=0;
            while pos<queue.len() {
                let j=queue[pos];pos+=1;let r=j/FFT_BINS;let f=j%FFT_BINS;
                r0=r0.min(r);r1=r1.max(r);f0=f0.min(f);f1=f1.max(f);
                for (rr,ff) in [(r.wrapping_sub(1),f),(r+1,f),(r,f.wrapping_sub(1)),(r,f+1)] {
                    if rr<nr && ff<FFT_BINS {let k=rr*FFT_BINS+ff;if hot[k]{hot[k]=false;queue.push(k);}}
                }
            }
            if queue.len()<3 || r1-r0+1>16 {continue;}
            let excess=|r:usize,f:usize|(seconds[t*plane+r*FFT_BINS+f]/counts[t] as f64-reference[r*FFT_BINS+f]).max(0.);
            let peak=*queue.iter().max_by(|&&a,&&b|excess(a/FFT_BINS,a%FFT_BINS).total_cmp(&excess(b/FFT_BINS,b%FFT_BINS))).unwrap();
            let (rp,fp)=(peak/FFT_BINS,peak%FFT_BINS);
            // A narrow central ion-line near perpendicular B is a desired
            // physical signal, not sufficient evidence for a satellite.
            if protected_gate.is_some_and(|gate|rp.abs_diff(gate)<=8) && fp.abs_diff(FFT_BINS/2)<=13 {continue;}
            let mut best=0_f64;
            for centre in rp.saturating_sub(2)..=(rp+2).min(nr-1) {
                let(mut dot,mut aa,mut bb)=(0.,0.,0.);
                for dr in -8_isize..=8 {for df in -32_isize..=32 {
                    let r=centre as isize+dr;let f=fp as isize+df;
                    if r<0 || r>=nr as isize || f<0 || f>=FFT_BINS as isize {continue;}
                    let x=excess(r as usize,f as usize);let y=template[((dr+8)*65+df+32) as usize];
                    dot+=x*y;aa+=x*x;bb+=y*y;
                }}
                if aa>0. && bb>0. {best=best.max(dot/(aa*bb).sqrt());}
            }
            if best<0.7 {continue;}
            // 480-us Tx + tapered 480-us receive response has ~144-km
            // support: guard four 18-km gate centres on each side.
            for tt in t.saturating_sub(1)..=(t+1).min(nt-1) {
                for rr in r0.saturating_sub(4)..=(r1+4).min(nr-1) {mask[tt*nr+rr]=1;}
            }
        }
    }
    mask
}

pub fn pulse_ambiguity_template()->Vec<f64> {
    let w=hann();let mut template=Vec::with_capacity(17*65);
    for dr in -8..=8 {for df in -32..=32 {
        let delay=-5.-dr as f64*120.;let mut re=0.;let mut im=0.;
        for (j,&weight) in w.iter().enumerate() {let time=j as f64*10.;
            if time>=delay && time<delay+480. {let phase=2.*std::f64::consts::PI*df as f64*j as f64/256.;re+=weight as f64*phase.cos();im+=weight as f64*phase.sin();}
        }
        template.push(re*re+im*im);
    }}
    template
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
    #[test]
    fn localized_blip_excludes_whole_range_exposure_but_preserves_steady_line() {
        let (nt,nr)=(60,30);let plane=nr*FFT_BINS;let mut data=vec![20.;nt*plane];
        for t in 0..nt {for r in 0..nr {for f in 120..125 {data[t*plane+r*FFT_BINS+f]=200.;}}}
        let template=pulse_ambiguity_template();
        for dr in -8_isize..=8 {for df in -32_isize..=32 {
            data[20*plane+(12_isize+dr) as usize*FFT_BINS+(153_isize+df) as usize]+=2000.*template[((dr+8)*65+df+32) as usize];
        }}
        let mask=transient_mask(&data,&vec![20;nt],nr,Some(12));
        assert!(mask[19*nr+6..19*nr+18].iter().all(|m|*m==1));
        assert!(mask[..18*nr].iter().all(|m|*m==0));
        assert!(mask[22*nr..].iter().all(|m|*m==0));
        // A persistent narrow plasma feature is part of the temporal reference.
        let clean=transient_mask(&vec![20.;nt*plane],&vec![20;nt],nr,Some(12));
        assert!(clean.iter().all(|m|*m==0));
        let mut common=vec![20.;nt*plane];
        for r in 0..nr {for f in 150..157 {common[20*plane+r*FFT_BINS+f]=2000.;}}
        assert!(transient_mask(&common,&vec![20;nt],nr,Some(12)).iter().all(|m|*m==0));
        let mut central=vec![20.;nt*plane];
        for dr in -8_isize..=8 {for df in -32_isize..=32 {
            central[20*plane+(12_isize+dr) as usize*FFT_BINS+(128_isize+df) as usize]+=2000.*template[((dr+8)*65+df+32) as usize];
        }}
        assert!(transient_mask(&central,&vec![20;nt],nr,Some(12)).iter().all(|m|*m==0));
    }
}
