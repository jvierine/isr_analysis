//! Progressive time-altitude figures from campaign HDF5. Pending/rejected cells
//! stay blank; raw fits and flags remain in the scientific product.
use anyhow::Result;
use rustplotlib::*;
fn main() -> Result<()> {
    let a: Vec<_> = std::env::args().collect();
    let show_all = a.get(3).map(|s| s == "--show-all").unwrap_or(false);
    let h = hdf5::File::open(&a[1])?;
    let no_he = h
        .attr("helium_fixed_zero")
        .ok()
        .and_then(|a| a.read_scalar::<u8>().ok())
        .unwrap_or(0)
        != 0;
    let joint = h.attr("analysis_type").is_ok();
    let fitted_maximum = h
        .attr("display_max_slant_range_km")
        .ok()
        .and_then(|a| a.read_scalar::<f64>().ok())
        .unwrap_or(1450.);
    let maximum: f64 = std::env::var("SYISR_PLOT_MAX_KM")
        .ok()
        .map(|v| v.parse())
        .transpose()?
        .unwrap_or(fitted_maximum)
        .min(fitted_maximum);
    let get = |n: &str| h.dataset(n).unwrap().read_raw::<f64>().unwrap();
    let edges = get("time_edges_utc_s");
    let z = get("altitude_km");
    let done = h.dataset("window_complete")?.read_raw::<u8>()?;
    let flags = h.dataset("quality_flags")?.read_raw::<u8>()?;
    let density_valid = h
        .dataset("density_valid")
        .ok()
        .map(|d| d.read_raw::<u8>().unwrap());
    let nt = done.len();
    let end = done.iter().rposition(|&v| v > 0).unwrap() + 1;
    let count = done.iter().filter(|&&v| v > 0).count();
    let base = (edges[0] / 86400.).floor() * 86400.;
    let x: Vec<f64> = edges[..=end].iter().map(|t| (t - base) / 3600.).collect();
    let ze: Vec<f64> = (0..=z.len())
        .map(|i| {
            if i == 0 {
                z[0] - (z[1] - z[0]) / 2.
            } else if i == z.len() {
                z[i - 1] + (z[i - 1] - z[i - 2]) / 2.
            } else {
                (z[i - 1] + z[i]) / 2.
            }
        })
        .collect();
    let start = chrono::DateTime::from_timestamp(edges[0] as i64, 0).unwrap();
    let finish = chrono::DateTime::from_timestamp(edges[end] as i64, 0).unwrap();
    let mut fig = Figure::new(IntendedUse::ArticleWide);
    fig.width_mm = 230.;
    fig.height_mm = if no_he { 235. } else { 283. };
    fig.raster_dpi = 160.;
    fig.labels.push(Label::new(
        if no_he {
            "SYISR: 2 ms pulse, robust O/H profile inversion"
        } else if h.attr("temporal_method").is_ok() {
            "SYISR: 2 ms pulse, temporally regularized profile inversion"
        } else if h.attr("composition_model").is_ok() {
            "SYISR: 2 ms pulse, sigmoid H and Gaussian He layer"
        } else if maximum > 2000. {
            "SYISR: 2 ms uncoded pulse, joint full-profile inversion"
        } else if h.attr("partial_pulse_extension").is_ok() {
            "SYISR: partial-pulse full-profile inversion, 30 min, 10--600 microseconds"
        } else if joint {
            "SYISR: joint full-profile inversion, 30 minutes, 10--600 microsecond fits"
        } else {
            "SYISR: 30 minute integrations, satellite rejection, 10--600 microsecond fits"
        },
        115.,
        5.,
    ));
    fig.labels.push(Label::new(
        format!(
            "{} to {} UTC; {count}/{nt} windows complete",
            start.format("%d %b %Y %H:%M"),
            finish.format("%d %b %H:%M")
        ),
        115.,
        12.,
    ));
    for (row, key, label, limit, ticks, log, error) in [
        (
            0,
            "ne_m3",
            r"$\log_{10} n_e$ (m$^{-3}$)",
            [if maximum > 2000. { 7. } else { 9. }, 13.],
            if maximum > 2000. {
                vec![7., 8., 9., 10., 11., 12., 13.]
            } else {
                vec![9., 10., 11., 12., 13.]
            },
            true,
            None,
        ),
        (
            1,
            "te_k",
            if maximum > 2000. {
                r"$T_e$ (kK)"
            } else {
                r"$T_e$ (K)"
            },
            if maximum > 2000. {
                [0.3, 8.]
            } else {
                [300., 8000.]
            },
            if maximum > 2000. {
                vec![0.3, 2., 4., 6., 8.]
            } else {
                vec![300., 2000., 4000., 6000., 8000.]
            },
            false,
            Some(("sigma_te_k", 0.5)),
        ),
        (
            2,
            "ti_k",
            if maximum > 2000. {
                r"$T_i$ (kK)"
            } else {
                r"$T_i$ (K)"
            },
            if maximum > 2000. {
                [0.3, 4.]
            } else {
                [300., 4000.]
            },
            if maximum > 2000. {
                vec![0.3, 1., 2., 3., 4.]
            } else {
                vec![300., 1000., 2000., 3000., 4000.]
            },
            false,
            Some(("sigma_ti_k", 0.5)),
        ),
        (
            3,
            "h_fraction",
            r"H$^+$ fraction",
            [0., 1.],
            vec![0., 0.25, 0.5, 0.75, 1.],
            false,
            Some(("sigma_h", 0.15)),
        ),
        (
            4,
            "he_fraction",
            r"He$^+$ fraction",
            [0., if maximum > 2000. { 1. } else { 0.3 }],
            if maximum > 2000. {
                vec![0., 0.25, 0.5, 0.75, 1.]
            } else {
                vec![0., 0.1, 0.2, 0.3]
            },
            false,
            Some(("sigma_he", 0.15)),
        ),
    ] {
        if no_he && row == 4 {
            continue;
        }
        let values = get(key);
        let errors = error.map(|(n, _)| get(n));
        let mut data = Vec::with_capacity(z.len() * end);
        for j in 0..z.len() {
            for t in 0..end {
                let i = j * nt + t;
                let mut v = values[i];
                if done[t] == 0
                    || (!show_all
                        && if row == 0 {
                            density_valid
                                .as_ref()
                                .map(|v| v[i] == 0)
                                .unwrap_or(flags[i] != 0)
                        } else {
                            flags[i] != 0
                        })
                {
                    v = f64::NAN;
                }
                if let Some((_, threshold)) = error.filter(|_| !show_all) {
                    let err = errors.as_ref().unwrap()[i];
                    if !err.is_finite()
                        || if row == 1 || row == 2 {
                            err > threshold * values[i]
                        } else {
                            err > threshold
                        }
                    {
                        v = f64::NAN;
                    }
                }
                if log {
                    v = v.log10();
                }
                if maximum > 2000. && (row == 1 || row == 2) {
                    v /= 1000.;
                }
                data.push(v);
            }
        }
        let sc = fig.scale(Scale::new(limit, Normalization::Linear, ColorMap::turbo())?);
        let y = 25. + row as f64 * 48.;
        let mut ax = Axes::new(
            [x[0], *x.last().unwrap()],
            if joint {
                [
                    (z[0] - 0.05).max(200.),
                    (z[z.len() - 1] - 0.05).min(maximum),
                ]
            } else {
                [ze[0], *ze.last().unwrap()]
            },
            if row == (if no_he { 3 } else { 4 }) {
                "UTC (day/hour)"
            } else {
                ""
            },
            "Altitude (km)",
        );
        let span = x[end] - x[0];
        let step = if span > 12. {
            3.
        } else if span > 6. {
            2.
        } else {
            1.
        };
        ax.xticks = (0..=48)
            .map(|i| i as f64 * step)
            .filter(|&v| v >= x[0] && v <= x[end])
            .collect();
        ax.xticklabels = Some(
            ax.xticks
                .iter()
                .map(|v| {
                    chrono::DateTime::from_timestamp((base + v * 3600.) as i64, 0)
                        .unwrap()
                        .format("%d/%H")
                        .to_string()
                })
                .collect(),
        );
        ax.yticks = if maximum <= 2500. && maximum > 2000. {
            vec![500., 1000., 1500., 2000., 2500.]
        } else if maximum > 2000. {
            vec![1000., 2000., 3000., 4000., 5000.]
        } else {
            vec![200., 500., 800., 1100., 1400.]
        };
        let mut p = Panel::new(Rect::new(23., y, 164., 34.), ax);
        p.push(Layer::Pcolormesh(Field::new(
            x.clone(),
            ze.clone(),
            data,
            sc,
        )?));
        // Requested display convention only; preserve fitted fractions in HDF5.
        // A separate cell boundary keeps the transition exactly at 500 km.
        if row == 3 {
            p.push(Layer::Pcolormesh(Field::new(
                x.clone(),
                vec![200., 500.],
                done[..end]
                    .iter()
                    .map(|&v| if v != 0 { 0. } else { f64::NAN })
                    .collect(),
                sc,
            )?));
        }
        fig.panels.push(p);
        let mut bar = Colorbar::new(Rect::new(193., y, 3., 34.), sc, label, ticks);
        bar.horizontal = false;
        fig.colorbars.push(bar);
    }
    fig.labels.push(Label::new(
        if no_he {
            r"He$^+$ fixed to zero; H$^+$ below 500 km displayed as zero"
        } else {
            r"H$^+$ below 500 km displayed as zero"
        },
        115.,
        if no_he { 219. } else { 267. },
    ));
    fig.labels.push(Label::new(
        if show_all {
            "All fitted values shown; weak and bound-limited estimates retained"
        } else if joint {
            "Blank: weak/uncertain or bound-limited; regularized profile reconstruction"
        } else {
            "Blank: pending or weak/rejected fit; broad overlapping range responses"
        },
        115.,
        if no_he { 225. } else { 273. },
    ));
    fig.labels.push(Label::new(
        r"Density scale assumes noon F-peak $10^{12}$ m$^{-3}$; no absolute calibration",
        115.,
        if no_he { 231. } else { 279. },
    ));
    for ext in ["png", "pdf", "svg"] {
        fig.save(format!("{}.{}", a[2], ext))?;
    }
    Ok(())
}
