//! Single-step measured/predicted complex ACF diagnostic in the actual data space.
use anyhow::{ensure, Result};
use rustplotlib::*;
fn main() -> Result<()> {
    let a: Vec<_> = std::env::args().collect();
    ensure!(a.len() == 3, "plot-step-acfs profile.h5 output-prefix");
    let f = hdf5::File::open(&a[1])?;
    let get = |n: &str| f.dataset(n).unwrap().read_raw::<f64>().unwrap();
    let ranges = get("observed_slant_range_km");
    let lag = get("lag_us");
    let time = get("time_utc_s");
    let m: serde_json::Value = serde_json::from_slice(&f.attr("metadata_json")?.read_raw::<u8>()?)?;
    let start = chrono::DateTime::from_timestamp(time[0] as i64, 0).unwrap();
    let end = chrono::DateTime::from_timestamp(time[1] as i64, 0).unwrap();
    let mut fig = Figure::new(IntendedUse::ArticleWide);
    fig.width_mm = 230.;
    fig.height_mm = 215.;
    fig.raster_dpi = 160.;
    fig.labels.push(Label::new(
        "SYISR: one-step complex ACF validation",
        115.,
        5.,
    ));
    fig.labels.push(Label::new(
        format!(
            "{}, {}--{} UTC",
            start.format("%d %b %Y"),
            start.format("%H:%M"),
            end.format("%H:%M")
        ),
        115.,
        12.,
    ));
    let re = get("observed_real_adu2");
    let im = get("observed_imag_adu2");
    let mr = get("model_real_adu2");
    let mi = get("model_imag_adu2");
    let sr = get("sem_real_adu2");
    let si = get("sem_imag_adu2");
    for (row, target) in [450., 600., 750.].into_iter().enumerate() {
        let g = ranges
            .iter()
            .enumerate()
            .min_by(|a, b| (a.1 - target).abs().total_cmp(&(b.1 - target).abs()))
            .unwrap()
            .0;
        let norm = re[60 * g];
        for (col, (data, model, sigma, label, limits)) in [
            (&re, &mr, &sr, "Real ACF", [-0.4, 1.05]),
            (&im, &mi, &si, "Imaginary ACF", [-0.05, 0.05]),
        ]
        .into_iter()
        .enumerate()
        {
            let mut ax = Axes::new(
                [0., 600.],
                limits,
                if row == 2 { r"Lag ($\mu$s)" } else { "" },
                label,
            );
            ax.xticks = vec![0., 100., 200., 300., 400., 500., 600.];
            ax.yticks = if col == 0 {
                vec![-0.4, -0.2, 0., 0.5, 1.]
            } else {
                vec![-0.04, -0.02, 0., 0.02, 0.04]
            };
            let rect = Rect::new(23. + 114. * col as f64, 30. + 53. * row as f64, 86., 37.);
            let mut p = Panel::new(rect, ax);
            for (l, &x) in lag.iter().enumerate() {
                let j = 60 * g + l;
                p.push(Layer::Line(Line::new(
                    vec![
                        [x, (data[j] - sigma[j]) / norm],
                        [x, (data[j] + sigma[j]) / norm],
                    ],
                    "#b9d7e5",
                )));
            }
            p.push(Layer::Line(Line::new(
                lag.iter()
                    .enumerate()
                    .map(|(l, &x)| [x, data[60 * g + l] / norm])
                    .collect(),
                "#17608e",
            )));
            p.push(Layer::Line(Line::new(
                lag.iter()
                    .enumerate()
                    .map(|(l, &x)| [x, model[60 * g + l] / norm])
                    .collect(),
                "#b43b2e",
            )));
            fig.panels.push(p);
            fig.labels.push(Label::new(
                format!("{:.0} km slant range", ranges[g]),
                66. + 114. * col as f64,
                26. + 53. * row as f64,
            ));
        }
    }
    let nfit = m["fitted_parameter_count"].as_u64().unwrap() as f64;
    let red = m["fit"]["data_chi_square"].as_f64().unwrap() / (ranges.len() as f64 * 119. - nfit);
    fig.labels.push(Label::new(
        "Blue: measured; red: joint model; pale bars: marginal standard errors",
        115.,
        192.,
    ));
    fig.labels.push(Label::new(
        format!("ACF / measured real 10-us lag; joint reduced chi-square {red:.2}"),
        115.,
        199.,
    ));
    fig.labels.push(Label::new(
        if m["helium_fixed_zero"].as_bool() == Some(true) {
            "Nominal flat signal response; robust temporal continuity; He fixed zero"
        } else if m["temporal_prior"]["product"].is_string() {
            "Nominal flat signal response; temporal continuity included"
        } else {
            "Nominal flat signal response; no temporal prior"
        },
        115.,
        206.,
    ));
    for ext in ["png", "pdf", "svg"] {
        fig.save(format!("{}.{ext}", a[2]))?;
    }
    Ok(())
}
