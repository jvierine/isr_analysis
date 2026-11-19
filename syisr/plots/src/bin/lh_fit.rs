use anyhow::{Result, ensure};
use chrono::{DateTime, Utc};
use rustplotlib::*;
use std::path::Path;
use syisr_ac::campaign;
fn main() -> Result<()> {
    let a = std::env::args().collect::<Vec<_>>();
    ensure!(a.len() == 3, "FIT_H5 OUTPUT_DIR");
    let v: serde_json::Value = campaign::load(Path::new(&a[1]))?;
    let out = Path::new(&a[2]);
    std::fs::create_dir_all(out)?;
    let mut f = Figure::new(IntendedUse::ArticleWide);
    f.width_mm = 180.;
    f.height_mm = 173.;
    f.font_pt = 10.;
    f.raster_dpi = 220.;
    f.latex.cache = out.join("latex-cache");
    let t = DateTime::<Utc>::from_timestamp(v["utc_start"].as_i64().unwrap() + 28800, 0).unwrap();
    f.labels.push(Label::new(
        format!(
            "SYISR lower-hybrid beam; {} China local time",
            t.format("%d %B %Y, %H:%M")
        ),
        90.,
        5.,
    ));
    f.labels.push(Label::new(
        format!(
            "Az 358$^\\circ$, El 60$^\\circ$; altitude {:.0} km, slant range {:.0} km",
            v["altitude_km"].as_f64().unwrap(),
            v["slant_range_km"].as_f64().unwrap()
        ),
        90.,
        12.,
    ));
    let lags = v["lags"].as_array().unwrap();
    for c in 0..2 {
        let data: Vec<_> = lags
            .iter()
            .map(|l| {
                [
                    l["lag_us"].as_f64().unwrap(),
                    l["observed"][c].as_f64().unwrap() / 1e4,
                ]
            })
            .collect();
        let errors: Vec<_> = lags
            .iter()
            .map(|l| l["assigned_sigma"][c].as_f64().unwrap() / 1e4)
            .collect();
        let prediction: Vec<_> = lags
            .iter()
            .map(|l| {
                [
                    l["lag_us"].as_f64().unwrap(),
                    l["predicted"][c].as_f64().unwrap() / 1e4,
                ]
            })
            .collect();
        let lo = data
            .iter()
            .zip(&errors)
            .map(|(p, e)| p[1] - e)
            .chain(prediction.iter().map(|p| p[1]))
            .fold(0., f64::min);
        let hi = data
            .iter()
            .zip(&errors)
            .map(|(p, e)| p[1] + e)
            .chain(prediction.iter().map(|p| p[1]))
            .fold(0., f64::max);
        let span = (hi - lo).max(0.1);
        let mut ax = Axes::new(
            [0., 480.],
            [lo - span * 0.08, hi + span * 0.08],
            if c == 1 { "Lag ($\\mu$s)" } else { "" },
            if c == 0 {
                r"Real ACF ($10^4$ ADU$^2$)"
            } else {
                r"Imaginary ACF ($10^4$ ADU$^2$)"
            },
        );
        ax.xticks = vec![0., 100., 200., 300., 400.];
        let base = 10f64.powf((span / 4.).log10().floor());
        let step = [1., 2., 5., 10.]
            .into_iter()
            .map(|m| m * base)
            .find(|x| *x >= span / 4.)
            .unwrap();
        ax.yticks = ((lo / step).ceil() as i64..=(hi / step).floor() as i64)
            .map(|i| i as f64 * step)
            .collect();
        if c == 0 {
            ax.show_x_axis = false;
        }
        let mut p = Panel::new(Rect::new(25., 25. + c as f64 * 57., 142., 41.), ax);
        let line = Line::new(prediction, "#d95f02");
        if c == 0 {
            p.legend = Some(Legend::new(vec![line.legend("Magnetized finite-beam fit")]));
        }
        p.push(Layer::Line(line));
        let error_pairs = data
            .iter()
            .zip(&errors)
            .map(|(p, e)| [p[1] - e, p[1] + e])
            .collect::<Vec<_>>();
        p.errorbars(&data, None, Some(&error_pairs), "#334e68", 0.5, 1., 0.5)?;
        p.markers(
            &mut f,
            &data,
            &MarkerStyle {
                shape: MarkerShape::Circle,
                size_pt: 1.8,
                color: [0.2, 0.3, 0.4],
            },
        )?;
        f.panels.push(p);
    }
    f.labels.push(Label::new(
        format!(
            "$N_e={:.2}\\times10^{{12}}$ m$^{{-3}}$; $T_i={:.0}$ K; $T_e={:.0}$ K; $v_i={:.1}$ m/s",
            v["ne_m3"].as_f64().unwrap() / 1e12,
            v["ti_k"].as_f64().unwrap(),
            v["te_k"].as_f64().unwrap(),
            v["velocity_m_s"].as_f64().unwrap()
        ),
        90.,
        143.,
    ));
    f.labels.push(Label::new(format!("Finite 480 $\\mu$s pulse; effective BGK $\\nu_e={:.0}$ s$^{{-1}}$; fitted two-way width {:.2}$^\\circ$",v["effective_BGK_rate_s"].as_f64().unwrap(),v["two_way_aspect_fwhm_deg"].as_f64().unwrap()),90.,151.));
    f.labels.push(Label::new(
        "IRI empirical density calibration; gated pilot; no parameter errors",
        90.,
        160.,
    ));
    f.save(out.join("lower-hybrid-gated-fit.png"))?;
    f.save(out.join("lower-hybrid-gated-fit.pdf"))?;
    Ok(())
}
