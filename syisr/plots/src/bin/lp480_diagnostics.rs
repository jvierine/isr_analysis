//! Production QA: retained far-range spectra and unmodified signed data residuals.
use anyhow::{Result, ensure};
use chrono::{DateTime, Utc};
use rustplotlib::*;
use std::{
    collections::{BTreeMap, BTreeSet},
    path::Path,
};
use syisr_ac::{campaign, lp480::Decoded, lp480_fit::WindowFit};

fn standardized(y: f64, m: f64, s: f64) -> Result<f64> {
    ensure!(
        y.is_finite() && m.is_finite() && s.is_finite() && s > 0.,
        "invalid residual or assigned sigma"
    );
    Ok((m - y) / s)
}
fn figure(out: &Path) -> Figure {
    let mut f = Figure::new(IntendedUse::ArticleWide);
    f.width_mm = 205.;
    f.height_mm = 240.;
    f.font_pt = 9.;
    f.latex.cache = out.join("latex-cache");
    f
}
fn main() -> Result<()> {
    let a = std::env::args().collect::<Vec<_>>();
    ensure!(a.len() == 4, "lp480_diagnostics INPUT FULL_FITS OUTPUT");
    let input = Path::new(&a[1]);
    let out = Path::new(&a[3]);
    std::fs::create_dir_all(out)?;
    for name in ["DIAGNOSTIC_PLOTS_COMPLETE", "DIAGNOSTIC_PLOTS_PARTIAL"] {
        let p = out.join(name);
        if p.exists() {
            std::fs::remove_file(p)?;
        }
    }
    let mut files = std::fs::read_dir(&a[2])?
        .map(|e| e.map(|e| e.path()))
        .collect::<std::io::Result<Vec<_>>>()?;
    files.sort();
    let mut spectra = BTreeMap::<(i64, i64), Vec<(i64, Vec<Vec<f64>>)>>::new();
    let mut rows = vec![];
    let mut profiles = 0;
    let mut fit_times = BTreeSet::new();
    let mut input_times = BTreeSet::new();
    for path in files {
        if path.extension().is_none_or(|x| x != "h5")
            || path
                .file_stem()
                .and_then(|v| v.to_str())
                .and_then(|v| v.parse::<i64>().ok())
                .is_none()
        {
            continue;
        }
        let w: WindowFit = campaign::load(&path)?;
        fit_times.insert(w.start_utc_s);
        for b in &w.beams {
            let d: Decoded = campaign::load(&input.join("products").join(format!("{}.h5", b.id)))?;
            let n = b.observed.len();
            ensure!(
                n > 0
                    && [
                        b.predicted.len(),
                        b.assigned_sigma.len(),
                        b.measurement_gate.len(),
                        b.measurement_lag.len()
                    ]
                    .iter()
                    .all(|x| *x == n),
                "missing full residual arrays; compact snapshots cannot certify QA"
            );
            let ng = d.gates.len();
            let nl = d.lags_us.len();
            ensure!(ng > 1 && nl > 1, "empty gating");
            let mut fields = [vec![f64::NAN; ng * nl], vec![f64::NAN; ng * nl]];
            let mut sums = vec![[0.; 2]; nl];
            let mut squares = sums.clone();
            let mut count = vec![0usize; nl];
            for i in 0..n {
                let g = b.measurement_gate[i];
                let l = b.measurement_lag[i]
                    .checked_sub(1)
                    .ok_or_else(|| anyhow::anyhow!("zero lag is not a fitted LP480 product"))?;
                ensure!(g < ng && l < nl, "gate/lag index outside source grid");
                count[l] += 1;
                for c in 0..2 {
                    let r =
                        standardized(b.observed[i][c], b.predicted[i][c], b.assigned_sigma[i][c])?;
                    fields[c][g * nl + l] = r;
                    sums[l][c] += r;
                    squares[l][c] += r * r;
                }
            }
            let z = d.gates.iter().map(|g| g.altitude_km).collect::<Vec<_>>();
            let edge = |v: &[f64]| {
                let mut e = vec![v[0] - (v[1] - v[0]) / 2.];
                e.extend(v.windows(2).map(|p| (p[0] + p[1]) / 2.));
                e.push(v[v.len() - 1] + (v[v.len() - 1] - v[v.len() - 2]) / 2.);
                e
            };
            let x = edge(&d.lags_us);
            let y = edge(&z);
            let date = DateTime::<Utc>::from_timestamp(w.start_utc_s, 0).unwrap();
            let mut f = figure(out);
            f.height_mm = 175.;
            let scale = f.scale(Scale::new(
                [-5., 5.],
                Normalization::Linear,
                ColorMap::diverging(),
            )?);
            f.labels.push(Label::new(
                format!(
                    "SYISR; {}; azimuth {:.2}$^\\circ$, elevation {:.2}$^\\circ$",
                    date.format("%d %b %Y %H:%M UTC"),
                    b.azimuth_deg,
                    b.elevation_deg
                ),
                102.,
                5.,
            ));
            for c in 0..2 {
                let mut ax = Axes::new(
                    [x[0], *x.last().unwrap()],
                    [y[0], *y.last().unwrap()],
                    if c == 1 { "Lag ($\\mu$s)" } else { "" },
                    "Altitude (km)",
                );
                ax.show_x_axis = c == 1;
                let rect = Rect::new(23., 22. + c as f64 * 68., 145., 49.);
                let mut p = Panel::new(rect, ax);
                p.pcolormesh(x.clone(), y.clone(), fields[c].clone(), scale)?;
                f.panels.push(p);
                f.colorbars.push(Colorbar::new(
                    Rect::new(174., rect.y, 3., rect.height),
                    scale,
                    "(Model--data) / assigned $\\sigma$",
                    vec![-5., 0., 5.],
                ));
                f.labels.push(Label::new(
                    if c == 0 {
                        "Real ACF residual"
                    } else {
                        "Imaginary ACF residual"
                    },
                    95.,
                    rect.y - 4.,
                ));
            }
            f.labels.push(Label::new("Signed data residuals before robust weighting; no prior penalties; blank cells are unmeasured",102.,160.));
            let stem = format!(
                "residual-{}-az{}-el{}",
                w.start_utc_s,
                (b.azimuth_deg * 100.).round() as i64,
                (b.elevation_deg * 100.).round() as i64
            );
            f.save(out.join(format!("{stem}.png")))?;
            rows.push(serde_json::json!({"time_utc_s":w.start_utc_s,"beam_id":b.id,"status":w.status,"lag_us":d.lags_us,"count":count,"sum_normalized_residual":sums,"sum_squared_normalized_residual":squares,"display_exceedances_abs5":fields.iter().flatten().filter(|v|v.is_finite()&&v.abs()>5.).count()}));
            profiles += 1;
        }
    }
    // Background diagnostics include every recorded look, including beams that
    // the current ion-line theory cannot fit, and times with no accepted fit.
    for path in std::fs::read_dir(input.join("products"))? {
        let path = path?.path();
        if path.extension().is_none_or(|x| x != "h5") {
            continue;
        }
        let d: Decoded = campaign::load(&path)?;
        input_times.insert(d.job.start);
        let av = d
            .averaged
            .as_ref()
            .ok_or_else(|| anyhow::anyhow!("missing retained spectra"))?;
        let nf = 128;
        let nw = av.range_doppler_raw.len() / nf;
        ensure!(
            nw >= 4 && av.range_doppler_raw.len() % nf == 0 && av.range_doppler_pulses > 0,
            "missing measured background spectra"
        );
        spectra
            .entry((
                (d.job.az * 100.).round() as i64,
                (d.job.el * 100.).round() as i64,
            ))
            .or_default()
            .push((
                d.job.start,
                (nw - 4..nw)
                    .map(|i| av.range_doppler_raw[i * nf..(i + 1) * nf].to_vec())
                    .collect(),
            ));
    }
    ensure!(profiles > 0, "no full fitted profiles");
    for ((az, el), mut data) in spectra {
        data.sort_by_key(|v| v.0);
        let start = data[0].0.div_euclid(86400) * 86400;
        let nt = ((data.last().unwrap().0 - start) / 1800 + 1) as usize;
        let nf = 128;
        let x = (0..=nt).map(|i| i as f64 / 2.).collect::<Vec<_>>();
        let y = (0..=nf)
            .map(|i| (i as f64 - 64.5) * 100. / 128.)
            .collect::<Vec<_>>();
        let mut f = figure(out);
        let mut powers = data
            .iter()
            .flat_map(|(_, a)| a.iter().flatten())
            .filter(|v| **v > 0. && v.is_finite())
            .map(|v| 10. * v.log10())
            .collect::<Vec<_>>();
        ensure!(!powers.is_empty(), "no positive noise PSD");
        powers.sort_by(f64::total_cmp);
        let lo = powers[0].floor();
        let hi = powers.last().unwrap().ceil().max(lo + 1.);
        let absolute = f.scale(Scale::new(
            [lo, hi],
            Normalization::Linear,
            ColorMap::storm(),
        )?);
        let relative = f.scale(Scale::new(
            [-3., 6.],
            Normalization::Linear,
            ColorMap::storm(),
        )?);
        f.labels.push(Label::new(
            format!(
                "SYISR background; {}; azimuth {:.2}$^\\circ$, elevation {:.2}$^\\circ$",
                DateTime::<Utc>::from_timestamp(start, 0)
                    .unwrap()
                    .format("%d %b %Y"),
                az as f64 / 100.,
                el as f64 / 100.
            ),
            102.,
            5.,
        ));
        for win in 0..4 {
            for col in 0..2 {
                let mut values = vec![f64::NAN; nf * nt];
                for (t, a) in &data {
                    let j = ((*t - start) / 1800) as usize;
                    let mut sorted = a[win].clone();
                    sorted.sort_by(f64::total_cmp);
                    let median = (sorted[63] + sorted[64]) / 2.;
                    for k in 0..nf {
                        let v = a[win][k];
                        if v > 0. && v.is_finite() && (col == 0 || median > 0.) {
                            values[k * nt + j] =
                                10. * (v / if col == 0 { 1. } else { median }).log10();
                        }
                    }
                }
                let mut ax = Axes::new(
                    [0., nt as f64 / 2.],
                    [-50., 50.],
                    if win == 3 {
                        if nt > 48 { "UTC date" } else { "UTC hour" }
                    } else {
                        ""
                    },
                    if col == 0 { "Frequency (kHz)" } else { "" },
                );
                ax.show_x_axis = win == 3;
                if nt > 48 {
                    ax.xticks = (0..=nt / 48).map(|d| d as f64 * 24.).collect();
                    ax.xticklabels = Some(
                        ax.xticks
                            .iter()
                            .map(|h| {
                                DateTime::<Utc>::from_timestamp(start + (*h * 3600.) as i64, 0)
                                    .unwrap()
                                    .format("%d/%m")
                                    .to_string()
                            })
                            .collect(),
                    );
                }
                let rect = Rect::new(18. + col as f64 * 99., 25. + win as f64 * 48., 69., 32.);
                let mut p = Panel::new(rect, ax);
                let scale = if col == 0 { absolute } else { relative };
                p.pcolormesh(x.clone(), y.clone(), values, scale)?;
                f.panels.push(p);
                f.colorbars.push(Colorbar::new(
                    Rect::new(rect.x + 72., rect.y, 2., rect.height),
                    scale,
                    if col == 0 {
                        "dB ADU$^2$ / FFT bin"
                    } else {
                        "dB / band median"
                    },
                    if col == 0 {
                        vec![lo, (lo + hi) / 2., hi]
                    } else {
                        vec![-3., 0., 3., 6.]
                    },
                ));
                f.labels.push(Label::new(
                    format!("Receive window {}", win + 1),
                    rect.x + 34.,
                    rect.y - 4.,
                ));
            }
        }
        f.labels.push(Label::new("Final four 128-sample Hann windows; retained pulses; no background subtraction or spectral notch",102.,228.));
        f.save(out.join(format!("background-az{az}-el{el}.png")))?;
        campaign::save(
            &out.join(format!("background-az{az}-el{el}.h5")),
            &serde_json::json!({"rows":data,"frequency_bin_khz":100./128.,"frequency_sign":"recorded I/Q FFT convention; not a verified velocity sign","units":"ADU^2 per FFT bin; Hann periodogram","note":"Measured end-of-record spectra, not a guarantee of plasma-free noise. Original powers preserved; nonpositive log-display cells blank."}),
        )?;
    }
    campaign::save(
        &out.join("residual-summary.h5"),
        &serde_json::json!({"rows":rows,"normalization":"(model-data)/assigned diagonal sigma, before Huber weights; not independent normal deviates or calibrated parameter uncertainty","lags":"All measured ion-line fit lags, including short lags; no zero-lag normalization","display_limits":[-5.,5.]}),
    )?;
    let missing = input_times
        .difference(&fit_times)
        .copied()
        .collect::<Vec<_>>();
    campaign::save(
        &out.join("coverage.h5"),
        &serde_json::json!({"measured_times":input_times,"fitted_times":fit_times,"times_missing_residual_plots":missing,"note":"Unsupported beams remain identified in original fit metadata; spectra include all recorded beams."}),
    )?;
    std::fs::write(
        out.join(if missing.is_empty() {
            "DIAGNOSTIC_PLOTS_COMPLETE"
        } else {
            "DIAGNOSTIC_PLOTS_PARTIAL"
        }),
        format!("{profiles} measured beam profiles\n"),
    )?;
    Ok(())
}
#[cfg(test)]
mod tests {
    use super::*;
    #[test]
    fn signed_standardization_rejects_invalid_errors() {
        assert_eq!(standardized(-2., 1., 0.5).unwrap(), 6.);
        assert_eq!(standardized(2., -1., 0.5).unwrap(), -6.);
        assert!(standardized(1., 2., 0.).is_err());
        assert!(standardized(1., 2., f64::NAN).is_err());
    }
}
