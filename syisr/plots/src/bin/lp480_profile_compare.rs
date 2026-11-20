//! Compare evaluated fits on the identical altitude grid and physical data.
use anyhow::{Result, ensure};
use rustplotlib::*;
use syisr_ac::{campaign, lp480_fit::WindowFit};
fn main() -> Result<()> {
    let args: Vec<_> = std::env::args().collect();
    ensure!(args.len() == 4, "OLD_H5 NEW_H5 OUTPUT_PNG");
    let old: WindowFit = campaign::load(std::path::Path::new(&args[1]))?;
    let new: WindowFit = campaign::load(std::path::Path::new(&args[2]))?;
    ensure!(old.start_utc_s == new.start_utc_s, "different time windows");
    let mut figure = Figure::new(IntendedUse::ArticleWide);
    figure.width_mm = 200.;
    figure.height_mm = 253.;
    figure.font_pt = 10.;
    let timestamp = chrono::DateTime::<chrono::Utc>::from_timestamp(new.start_utc_s, 0).unwrap();
    figure.labels.push(Label::new(
        format!(
            "SYISR: {}; profile regularization comparison",
            timestamp.format("%d %B %Y %H:%M UTC")
        ),
        100.,
        5.,
    ));
    let a = old
        .beams
        .iter()
        .max_by(|a, b| a.elevation_deg.total_cmp(&b.elevation_deg))
        .unwrap();
    let b = new
        .beams
        .iter()
        .max_by(|a, b| a.elevation_deg.total_cmp(&b.elevation_deg))
        .unwrap();
    let top = *b.altitude_km.last().unwrap();
    for (field, label, limit, ticks) in [
        (
            0,
            r"$\log_{10} N_e$ (m$^{-3}$)",
            [9., 13.],
            vec![9., 11., 13.],
        ),
        (1, r"$T_i$ (K)", [150., 6000.], vec![1000., 3000., 5000.]),
        (2, r"$T_e$ (K)", [150., 12000.], vec![2000., 6000., 10000.]),
        (3, r"$v_i$ (m/s)", [-200., 200.], vec![-200., 0., 200.]),
        (4, r"H$^+$ fraction", [0., 1.], vec![0., 0.5, 1.]),
    ] {
        let mut axes = Axes::new(
            limit,
            [100., top],
            label,
            if field % 2 == 0 { "Altitude (km)" } else { "" },
        );
        axes.xticks = ticks;
        axes.yticks = vec![100., 500., 1000., 1400., 1800.];
        let mut panel = Panel::new(
            Rect::new(
                20. + 90. * (field % 2) as f64,
                18. + 73. * (field / 2) as f64,
                67.,
                55.,
            ),
            axes,
        );
        for (beam, color) in [(a, "#888888"), (b, "#0072B2")] {
            let values = [
                &beam.ne_m3,
                &beam.ti_k,
                &beam.te_k,
                &beam.velocity_m_s,
                &beam.h_fraction,
            ][field];
            let points = beam
                .altitude_km
                .iter()
                .zip(values)
                .map(|(z, p)| [if field == 0 { p.log10() } else { *p }, *z])
                .collect();
            panel.push(Layer::Line(Line::new(points, color)));
        }
        figure.panels.push(panel);
    }
    figure.labels.push(Label::new(
        "Gray: previous fit; blue: regularized refit",
        100.,
        242.,
    ));
    figure.save(&args[3])?;
    Ok(())
}
