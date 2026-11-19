//! Raw, gap-preserving fast-time spectra; diagnostic only, no notch or data edits.
use anyhow::{ensure, Result};
use rustfft::FftPlanner;
use serde_json::json;
use std::{
    collections::BTreeSet,
    fs::File,
    io::{BufReader, Seek, SeekFrom},
    path::Path,
};
use syisr_ac::{
    campaign::{self, Campaign},
    reader::read_frame,
    Complex as C,
};
fn spectrum(x: &[C], fft: &std::sync::Arc<dyn rustfft::Fft<f64>>) -> (Vec<C>, f64) {
    let n = x.len();
    let mut e = 0.;
    let mut z: Vec<_> = x
        .iter()
        .enumerate()
        .map(|(j, v)| {
            let w = 0.5 - 0.5 * (std::f64::consts::TAU * j as f64 / (n - 1) as f64).cos();
            e += w * w;
            *v * w
        })
        .collect();
    fft.process(&mut z);
    (z, e * 1e5)
}
fn median(x: &mut Vec<f64>) -> f64 {
    x.sort_by(f64::total_cmp);
    x[x.len() / 2]
}
fn main() -> Result<()> {
    let a: Vec<_> = std::env::args().collect();
    ensure!(a.len() == 3, "syisr_rfi_audit CAMPAIGN_ROOT OUT_H5");
    let root = Path::new(&a[1]);
    let c: Campaign = campaign::load(&root.join("campaign.h5"))?;
    let mut chosen = BTreeSet::new();
    let mut planner = FftPlanner::<f64>::new();
    let ff = planner.plan_fft_forward(1024);
    let fq = planner.plan_fft_forward(256);
    let slow = planner.plan_fft_forward(128);
    let mut rows = vec![];
    for d in &c.dwells {
        if d.pulses < 128 {
            continue;
        }
        let key = (d.az_raw, d.el_raw, (d.start / 3600.).floor() as i64);
        if !chosen.insert(key) {
            continue;
        }
        let mut frames = vec![];
        for span in &d.spans {
            let mut f = BufReader::new(File::open(&span.file)?);
            f.seek(SeekFrom::Start(span.offset))?;
            for _ in 0..span.pulses.min((128 - frames.len()) as usize) {
                let frame = read_frame(&mut f)?.ok_or_else(|| anyhow::anyhow!("EOF"))?;
                ensure!(
                    frame.header.samples == 1334 && frame.header.sample_rate_hz == 1e5,
                    "unexpected sample layout"
                );
                frames.push(
                    frame
                        .iq
                        .iter()
                        .map(|v| C::new(v.re as f64, v.im as f64))
                        .collect::<Vec<_>>(),
                );
            }
            if frames.len() == 128 {
                break;
            }
        }
        ensure!(frames.len() == 128, "dwell pulse count");
        let mut pf = vec![0.; 1024];
        let mut pq = vec![0.; 256];
        let mut complexq = vec![];
        let mut dc = C::default();
        let mut energy = 0.;
        for x in &frames {
            let (z, n) = spectrum(&x[128..1152], &ff);
            for (i, v) in z.iter().enumerate() {
                pf[(i + 512) % 1024] += v.norm_sqr() / n / 128.;
            }
            let (z, n) = spectrum(&x[1078..1334], &fq);
            for (i, v) in z.iter().enumerate() {
                pq[(i + 128) % 256] += v.norm_sqr() / n / 128.;
            }
            complexq.push(z);
            for v in &x[1078..1334] {
                dc += *v;
                energy += v.norm_sqr();
            }
        }
        let mut ratios = vec![1.; 256];
        for k in 8usize..248 {
            let mut reference: Vec<_> = (k - 8..=k + 8)
                .filter(|j| (*j).abs_diff(k) > 2)
                .map(|j| pq[j])
                .collect();
            ratios[k] = pq[k] / median(&mut reference).max(1e-30);
        }
        let peak = (8..248)
            .max_by(|a, b| ratios[*a].total_cmp(&ratios[*b]))
            .unwrap();
        let raw = (peak + 128) % 256;
        let mut amplitude: Vec<_> = complexq.iter().map(|v| v[raw]).collect();
        let sum = amplitude.iter().map(|v| v.norm_sqr()).sum::<f64>();
        slow.process(&mut amplitude);
        let max =
            amplitude.iter().map(|v| v.norm_sqr()).fold(0f64, f64::max) / (128. * sum).max(1e-30);
        dc /= 128. * 256.;
        let dc_fraction = dc.norm_sqr() / (energy / (128. * 256.)).max(1e-30);
        rows.push(json!({"dwell":d.id,"start_utc_s":d.start,"azimuth_deg":d.az,"elevation_deg":d.el,"pulses":128,"full_psd":pf,"quiet_psd":pq,"quiet_peak_offset_hz":(peak as f64-128.)*1e5/256.,"quiet_peak_local_ratio":ratios[peak],"quiet_peak_slow_coherence":max,"quiet_mean_iq":[dc.re,dc.im],"quiet_dc_fraction":dc_fraction}));
        eprintln!(
            "RFI_SAMPLE {} peak{:.1}Hz ratio{:.2} coherence{:.3}",
            d.id,
            (peak as f64 - 128.) * 1e5 / 256.,
            ratios[peak],
            max
        );
    }
    campaign::save(
        Path::new(&a[2]),
        &json!({"schema":"syisr-raw-rfi-audit-v1","radar_frequency_hz":430.2e6,"sample_rate_hz":1e5,"full_frequency_hz":(0..1024).map(|i|(i as f64-512.)*1e5/1024.).collect::<Vec<_>>(),"quiet_frequency_hz":(0..256).map(|i|(i as f64-128.)*1e5/256.).collect::<Vec<_>>(),"psd_units":"ADU^2/Hz, complex two-sided Hann periodograms, no mean subtraction","full_sample_interval":[128,1152],"quiet_sample_interval":[1078,1334],"frequency_sign":"FFT of raw I+iQ; RF sideband orientation not independently calibrated","selection":"first >=128pulse dwell per recorded beam per UTC hour;128 contiguous pulses, gaps never concatenated","screening":"raw unscreened spectra; coherent narrow-line persistence across quiet gates and beams is a diagnostic, not a proved interference classification","rows":rows}),
    )?;
    Ok(())
}
