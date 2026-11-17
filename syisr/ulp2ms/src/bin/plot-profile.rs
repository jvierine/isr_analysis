//! Profile and data-space diagnostics; only scientific annotations are rendered.
use anyhow::{ensure, Result};
use rustplotlib::*;
fn main() -> Result<()> {
    let a = std::env::args().collect::<Vec<_>>();
    ensure!(
        a.len() == 4,
        "plot-profile joint-profile.h5 old-fits.h5 output-prefix"
    );
    let h = hdf5::File::open(&a[1])?;
    let get = |n: &str| h.dataset(n).unwrap().read_raw::<f64>().unwrap();
    let z = get("altitude_km");
    let m: serde_json::Value = serde_json::from_slice(&h.attr("metadata_json")?.read_raw::<u8>()?)?;
    let old = hdf5::File::open(&a[2])?;
    let mut seeds = vec![];
    for n in old.member_names()? {
        if n.starts_with("range_") {
            let mm: serde_json::Value =
                serde_json::from_slice(&old.group(&n)?.attr("metadata_json")?.read_raw::<u8>()?)?;
            seeds.push(mm);
        }
    }
    seeds.sort_by(|a, b| {
        a["slant_range_km"]
            .as_f64()
            .unwrap()
            .total_cmp(&b["slant_range_km"].as_f64().unwrap())
    });
    let mut f = Figure::new(IntendedUse::ArticleWide);
    f.width_mm = 230.;
    f.height_mm = 188.;
    f.raster_dpi = 180.;
    let time = get("time_utc_s");
    let start = chrono::DateTime::from_timestamp(time[0] as i64, 0).unwrap();
    f.labels
        .push(Label::new("SYISR: joint full-profile inversion", 115., 5.));
    f.labels.push(Label::new(
        format!(
            "{} UTC; 2 ms pulse; fits use 10--600 microseconds",
            start.format("%d %b %Y %H:%M")
        ),
        115.,
        12.,
    ));
    for (i, key, label, limits, ticks) in [
        (
            0,
            "ne_m3",
            r"$\log_{10} n_e$ (m$^{-3}$)",
            [9., 12.2],
            vec![9., 10., 11., 12.],
        ),
        (
            1,
            "te_k",
            r"$T_e$ (K)",
            [300., 6500.],
            vec![1000., 3000., 5000.],
        ),
        (
            2,
            "ti_k",
            r"$T_i$ (K)",
            [300., 4500.],
            vec![1000., 2000., 3000., 4000.],
        ),
        (
            3,
            "h_fraction",
            r"H$^+$ fraction",
            [0., 1.],
            vec![0., 0.25, 0.5, 0.75, 1.],
        ),
        (
            4,
            "he_fraction",
            r"He$^+$ fraction",
            [0., 0.2],
            vec![0., 0.05, 0.1, 0.15, 0.2],
        ),
        (
            5,
            "o_fraction",
            r"O$^+$ fraction",
            [0., 1.],
            vec![0., 0.25, 0.5, 0.75, 1.],
        ),
    ] {
        let values = get(key);
        let transform = |v: f64| if key == "ne_m3" { v.log10() } else { v };
        let mut axes = Axes::new(
            limits,
            [400., 1450.],
            label,
            if i % 3 == 0 { "Altitude (km)" } else { "" },
        );
        axes.xticks = ticks;
        axes.yticks = vec![400., 600., 800., 1000., 1200., 1400.];
        let mut p = Panel::new(
            Rect::new(
                18. + (i % 3) as f64 * 74.,
                27. + (i / 3) as f64 * 75.,
                55.,
                54.,
            ),
            axes,
        );
        let points = z
            .iter()
            .zip(values)
            .filter(|(z, _)| **z >= 400. && **z <= 1450.)
            .map(|(&z, v)| [transform(v), z])
            .collect::<Vec<_>>();
        p.push(Layer::Line(Line::new(points, "#17608e")));
        let prior = seeds
            .iter()
            .filter_map(|m| {
                let r = m["slant_range_km"].as_f64()?;
                if r < 400. || r > 1450. {
                    return None;
                }
                let value = if key == "ne_m3" {
                    m["density_shape_assumption_m3"].as_f64()?
                } else {
                    m["fit"][key].as_f64()?
                };
                Some([transform(value), r + 0.05])
            })
            .collect::<Vec<_>>();
        p.push(Layer::Line(Line::new(prior, "#bc6235")));
        f.panels.push(p);
    }
    f.labels.push(Label::new(
        "Blue: joint physical profile; orange: earlier broad-volume fits",
        115.,
        174.,
    ));
    f.labels.push(Label::new(
        "Assumed noon peak density; smoothing and background systematics remain",
        115.,
        181.,
    ));
    for ext in ["pdf", "png", "svg"] {
        f.save(format!("{}.{ext}", a[3]))?;
    }
    // Show held data-space predictions and whitened residuals, without multiplying
    // the observations by R^2 or deconvolving them.
    let range = get("observed_slant_range_km");
    let lag = get("lag_us");
    let obs = get("observed_real_adu2");
    let model = get("model_real_adu2");
    let sigma = get("sem_real_adu2");
    let mut f = Figure::new(IntendedUse::ArticleWide);
    f.width_mm = 230.;
    f.height_mm = 122.;
    f.raster_dpi = 180.;
    f.labels.push(Label::new(
        "Joint profile: signed real ACF and residual",
        115.,
        5.,
    ));
    let nr = range.len();
    let nl = lag.len();
    let xe = (0..=nl).map(|j| 5. + 10. * j as f64).collect::<Vec<_>>();
    let ye = (0..=nr)
        .map(|j| {
            if j == 0 {
                range[0] - (range[1] - range[0]) / 2.
            } else if j == nr {
                range[nr - 1] + (range[nr - 1] - range[nr - 2]) / 2.
            } else {
                (range[j - 1] + range[j]) / 2.
            }
        })
        .collect::<Vec<_>>();
    for (i, title) in [
        (0, "Measured / row maximum"),
        (1, "Model / same maximum"),
        (2, "Residual / block SEM"),
    ] {
        let mut values = vec![];
        for r in 0..nr {
            let scale = obs[r * nl..(r + 1) * nl]
                .iter()
                .map(|x| x.abs())
                .fold(0., f64::max)
                .max(1.);
            for l in 0..nl {
                let j = r * nl + l;
                values.push(if i == 0 {
                    obs[j] / scale
                } else if i == 1 {
                    model[j] / scale
                } else {
                    (obs[j] - model[j]) / sigma[j]
                });
            }
        }
        let sc = f.scale(Scale::new(
            if i == 2 { [-4., 4.] } else { [-1., 1.] },
            Normalization::Linear,
            ColorMap::diverging(),
        )?);
        let mut axes = Axes::new(
            [5., 605.],
            [ye[0], *ye.last().unwrap()],
            r"Lag ($\mu$s)",
            if i == 0 { "Slant range (km)" } else { "" },
        );
        axes.xticks = vec![100., 300., 500.];
        axes.yticks = vec![500., 800., 1100., 1400.];
        let mut p = Panel::new(Rect::new(19. + i as f64 * 74., 22., 58., 62.), axes);
        p.push(Layer::Pcolormesh(Field::new(
            xe.clone(),
            ye.clone(),
            values,
            sc,
        )?));
        f.panels.push(p);
        f.labels.push(Label::new(title, 48. + i as f64 * 74., 15.));
        let mut cb = Colorbar::new(
            Rect::new(19. + i as f64 * 74., 94., 58., 2.),
            sc,
            "",
            if i == 2 {
                vec![-4., 0., 4.]
            } else {
                vec![-1., 0., 1.]
            },
        );
        cb.horizontal = true;
        f.colorbars.push(cb);
    }
    f.labels.push(Label::new(
        format!(
            "{}; data chi-square {:.1}; penalty {:.1}",
            m["fit"]["status"].as_str().unwrap(),
            m["fit"]["data_chi_square"].as_f64().unwrap(),
            m["fit"]["penalty"].as_f64().unwrap()
        ),
        115.,
        116.,
    ));
    for ext in ["pdf", "png", "svg"] {
        f.save(format!("{}-residual.{ext}", a[3]))?;
    }
    Ok(())
}
