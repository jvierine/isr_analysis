//! Progressive time-altitude figures from campaign HDF5. Pending/rejected cells
//! stay blank; raw fits and flags remain in the scientific product.
use anyhow::Result;
use rustplotlib::*;
fn main() -> Result<()> {
    let a: Vec<_> = std::env::args().collect();
    let h = hdf5::File::open(&a[1])?;
    let joint = h.attr("analysis_type").is_ok();
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
    fig.height_mm = 283.;
    fig.raster_dpi = 160.;
    fig.labels.push(Label::new(
        if joint {
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
            start.format("%d %b %H:%M"),
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
            [9., 12.],
            vec![9., 10., 11., 12.],
            true,
            None,
        ),
        (
            1,
            "te_k",
            r"$T_e$ (K)",
            [300., 8000.],
            vec![300., 2000., 4000., 6000., 8000.],
            false,
            Some(("sigma_te_k", 0.5)),
        ),
        (
            2,
            "ti_k",
            r"$T_i$ (K)",
            [300., 4500.],
            vec![300., 1500., 3000., 4500.],
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
            [0., 0.3],
            vec![0., 0.1, 0.2, 0.3],
            false,
            Some(("sigma_he", 0.15)),
        ),
    ] {
        let values = get(key);
        let errors = error.map(|(n, _)| get(n));
        let mut data = Vec::with_capacity(z.len() * end);
        for j in 0..z.len() {
            for t in 0..end {
                let i = j * nt + t;
                let mut v = values[i];
                if done[t] == 0
                    || if row == 0 {
                        density_valid
                            .as_ref()
                            .map(|v| v[i] == 0)
                            .unwrap_or(flags[i] != 0)
                    } else {
                        flags[i] != 0
                    }
                {
                    v = f64::NAN;
                }
                if let Some((_, threshold)) = error {
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
                data.push(v);
            }
        }
        let sc = fig.scale(Scale::new(limit, Normalization::Linear, ColorMap::turbo())?);
        let y = 25. + row as f64 * 48.;
        let mut ax = Axes::new(
            [x[0], *x.last().unwrap()],
            if joint {
                [450., 1450.]
            } else {
                [ze[0], *ze.last().unwrap()]
            },
            if row == 4 { "UTC (day/hour)" } else { "" },
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
        ax.yticks = vec![400., 700., 1000., 1300.];
        let mut p = Panel::new(Rect::new(23., y, 164., 34.), ax);
        p.push(Layer::Pcolormesh(Field::new(
            x.clone(),
            ze.clone(),
            data,
            sc,
        )?));
        fig.panels.push(p);
        let mut bar = Colorbar::new(Rect::new(193., y, 3., 34.), sc, label, ticks);
        bar.horizontal = false;
        fig.colorbars.push(bar);
    }
    fig.labels.push(Label::new(
        if joint {
            "Blank: weak/uncertain or bound-limited; regularized profiles at model knots"
        } else {
            "Blank: pending or weak/rejected fit; broad overlapping range responses"
        },
        115.,
        273.,
    ));
    fig.labels.push(Label::new(
        r"Density scale assumes noon F-peak $10^{12}$ m$^{-3}$; no absolute calibration",
        115.,
        279.,
    ));
    for ext in ["png", "pdf", "svg"] {
        fig.save(format!("{}.{}", a[2], ext))?;
    }
    Ok(())
}
