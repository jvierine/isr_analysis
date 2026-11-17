use anyhow::Result;
use rustplotlib::*;
fn main() -> Result<()> {
    let a: Vec<_> = std::env::args().collect();
    let h = hdf5::File::open(&a[1])?;
    let time = h
        .dataset("time_utc_s")
        .ok()
        .map(|d| d.read_raw::<f64>().unwrap());
    let interval = if let Some(t) = time {
        format!(
            "{}--{} UTC; 2 ms pulse; 10 microsecond lags",
            chrono::DateTime::from_timestamp(t[0] as i64, 0)
                .unwrap()
                .format("%d %B %Y, %H:%M"),
            chrono::DateTime::from_timestamp(t[1] as i64, 0)
                .unwrap()
                .format("%H:%M")
        )
    } else {
        "SYISR integration; 2 ms pulse; 10 microsecond lags".into()
    };
    let mut names: Vec<_> = h
        .member_names()?
        .into_iter()
        .filter(|n| n.starts_with("range_"))
        .collect();
    names.sort_by_key(|n| n.trim_start_matches("range_").parse::<u32>().unwrap());
    let mut fig = Figure::new(IntendedUse::ArticleWide);
    fig.width_mm = 230.;
    fig.height_mm = 230.;
    fig.raster_dpi = 180.;
    fig.labels.push(Label::new(
        r"SYISR: preliminary H$^+$/He$^+$/O$^+$ fits",
        115.,
        5.,
    ));
    fig.labels.push(Label::new(interval, 115., 11.));
    fig.labels.push(Label::new(
        "Real ACF: blue measured, red best fit; pale bars: block uncertainty",
        115.,
        19.,
    ));
    fig.labels.push(Label::new(
        "Imaginary ACF: gray measured, green dashed best fit",
        115.,
        25.,
    ));
    for (j, name) in names.iter().enumerate() {
        let g = h.group(name)?;
        let m: serde_json::Value =
            serde_json::from_slice(&g.attr("metadata_json")?.read_raw::<u8>()?)?;
        let fit = &m["fit"];
        let get = |n: &str| g.dataset(n).unwrap().read_raw::<f64>().unwrap();
        let x = get("lag_us");
        let yr = get("observed_real_adu2");
        let yi = get("observed_imag_adu2");
        let mr = get("model_real_adu2");
        let mi = get("model_imag_adu2");
        let se = get("sigma_real_adu2");
        let norm = m["held_out_zero_lag_observed_adu2"]
            .as_f64()
            .unwrap_or(yr[0]);
        let col = j % 2;
        let row = j / 2;
        let px = 23. + 112. * col as f64;
        let py = 48. + 97. * row as f64;
        let title = format!(
            r"{:.0} km slant range; $T_i={:.0}$ K, $T_e={:.0}$ K",
            m["slant_range_km"].as_f64().unwrap(),
            fit["ti_k"].as_f64().unwrap(),
            fit["te_k"].as_f64().unwrap()
        );
        fig.labels.push(Label::new(title, px + 45., py - 13.));
        let title = format!(
            r"H$^+$ {:.1}\%, He$^+$ {:.1}\%, O$^+$ {:.1}\%; $\chi^2_\nu={:.1}$",
            100. * fit["h_fraction"].as_f64().unwrap(),
            100. * fit["he_fraction"].as_f64().unwrap(),
            100. * fit["o_fraction"].as_f64().unwrap(),
            fit["chi_square"].as_f64().unwrap() / fit["dof"].as_u64().unwrap() as f64
        );
        fig.labels.push(Label::new(title, px + 45., py - 7.));
        let mut ax = Axes::new(
            [0., 600.],
            [-0.3, 1.1],
            r"Lag ($\mu$s)",
            "ACF / measured zero lag",
        );
        ax.xticks = vec![0., 100., 200., 300., 400., 500., 600.];
        ax.yticks = vec![-0.2, 0., 0.5, 1.];
        let mut panel = Panel::new(Rect::new(px, py, 90., 55.), ax);
        let pts: Vec<_> = x
            .iter()
            .enumerate()
            .map(|(k, &x)| [x, yr[k] / norm])
            .collect();
        let bounds: Vec<_> = yr
            .iter()
            .zip(&se)
            .map(|(&v, &s)| [(v - s) / norm, (v + s) / norm])
            .collect();
        panel.errorbars(&pts, None, Some(&bounds), "#9ab8d1", 0.5, 1.5, 0.65)?;
        for (y, c, dash, width) in [
            (&yr, "#17608e", false, 0.8),
            (&mr, "#b33a2f", false, 1.5),
            (&yi, "#737373", false, 0.7),
            (&mi, "#25804a", true, 1.2),
        ] {
            let mut line = Line::new(
                x.iter()
                    .enumerate()
                    .map(|(k, &x)| [x, y[k] / norm])
                    .collect(),
                c,
            );
            line.width_pt = width;
            if dash {
                line.dash = Some("1.3 0.7".into());
            }
            panel.push(Layer::Line(line));
        }
        fig.panels.push(panel);
    }
    fig.labels.push(Label::new(
        "Fitted lags 10--600 microseconds; free background; provisional F-peak density",
        115.,
        224.,
    ));
    for ext in ["png", "pdf", "svg"] {
        fig.save(format!("{}.{}", a[2], ext))?;
    }
    Ok(())
}
