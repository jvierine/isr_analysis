//! Direct H+ response and drift/wind reference comparisons; no difference panels.
use anyhow::{Result, ensure};
use rustplotlib::*;
use serde_json::Value;
use std::{collections::BTreeMap, path::Path};
fn median(v: &[f64]) -> f64 {
    let mut a = v.to_vec();
    a.sort_by(f64::total_cmp);
    a[a.len() / 2]
}
fn axes(x: &[f64], y: [f64; 2], bottom: bool, label: &str) -> Axes {
    let mut a = Axes::new(
        [x[0], *x.last().unwrap()],
        y,
        if bottom {
            "6--7 October 2025; China local time (UTC+8)"
        } else {
            ""
        },
        label,
    );
    a.xticks = vec![8., 12., 16., 20., 24., 28., 32.];
    a.xticklabels = Some(
        [
            "08:00", "12:00", "16:00", "20:00", "00:00", "04:00", "08:00",
        ]
        .iter()
        .map(|s| s.to_string())
        .collect(),
    );
    if !bottom {
        a.show_x_axis = false;
    }
    a
}
fn main() -> Result<()> {
    let a: Vec<_> = std::env::args().collect();
    ensure!(a.len() == 4, "RESPONSE_HDF DRIVER_HDF OUTPUT");
    let out = Path::new(&a[3]);
    std::fs::create_dir_all(out)?;
    let h = hdf5::File::open(&a[1])?;
    let time = h.dataset("start_utc_s")?.read_raw::<i64>()?;
    let zs = h.dataset("altitude_km")?.read_raw::<f64>()?;
    let mut x: Vec<_> = time
        .iter()
        .map(|t| (*t as f64 + 28800. - 1759708800.) / 3600.)
        .collect();
    x.push(x.last().unwrap() + 0.5);
    let mut z: Vec<_> = zs.iter().map(|v| v - 5.).collect();
    z.push(zs.last().unwrap() + 5.);
    let names = ["fit", "baseline", "measured", "measured_reversed"];
    let labels = [
        "SYISR AC: shared fitted H$^+$ fraction",
        "SAMI2: Fejer--Scherliess drift; HWM14 winds",
        "SAMI2: measured drift, toward-positive LOS; HWM14 winds",
        "SAMI2: measured drift, reversed-sign sensitivity; HWM14 winds",
    ];
    let colors = ["#111111", "#0072B2", "#D55E00", "#009E73"];
    let mut f = Figure::new(IntendedUse::ArticleWide);
    f.width_mm = 205.;
    f.height_mm = 250.;
    f.font_pt = 10.;
    f.raster_dpi = 240.;
    f.latex.cache = out.join("latex-cache");
    let sc = f.scale(Scale::new(
        [0., 1.],
        Normalization::Linear,
        ColorMap::turbo(),
    )?);
    for q in 0..4 {
        let yy = 18. + q as f64 * 53.;
        let mut ax = axes(&x, [500., 1300.], q == 3, "Altitude (km)");
        ax.yticks = vec![500., 700., 1000., 1300.];
        let mut p = Panel::new(Rect::new(22., yy, 150., 40.), ax);
        p.pcolormesh(
            x.clone(),
            z.clone(),
            h.dataset(&format!("{}_h_fraction", names[q]))?
                .read_raw::<f64>()?,
            sc,
        )?;
        f.panels.push(p);
        f.labels.push(Label::new(labels[q], 97., yy - 4.));
        f.colorbars
            .push(Colorbar::new(Rect::new(175., yy, 3., 40.), sc, "", vec![]));
        for (v, y) in [(0., yy + 40.), (0.5, yy + 20.), (1., yy)] {
            f.labels.push(Label::new(format!("{v:.1}"), 185., y));
        }
        let mut l = Label::new("H$^+$ fraction", 200., yy + 20.);
        l.rotation = -90.;
        f.labels.push(l);
    }
    f.labels.push(Label::new(
        "Matched 30-minute windows; LOS positive toward radar",
        103.,
        243.,
    ));
    f.save(out.join("hplus-response.png"))?;
    f.save(out.join("hplus-response.pdf"))?;
    let mut f = Figure::new(IntendedUse::ArticleWide);
    f.width_mm = 205.;
    f.height_mm = 100.;
    f.font_pt = 10.;
    f.raster_dpi = 240.;
    f.latex.cache = out.join("latex-cache");
    let mut p = Panel::new(
        Rect::new(22., 24., 170., 48.),
        axes(&x, [500., 1300.], true, "H$^+$ 50\\% height (km)"),
    );
    for q in 0..4 {
        let values = h
            .dataset(&format!("{}_h50_km", names[q]))?
            .read_raw::<f64>()?;
        let points = values
            .iter()
            .enumerate()
            .map(|(t, v)| [(x[t] + x[t + 1]) / 2., *v])
            .collect();
        p.push(Layer::Line(Line::new(points, colors[q])));
        let mut l = Label::new(
            [
                "AC fit",
                "Fejer--Scherliess",
                "Measured drift",
                "Reversed sign",
            ][q],
            34. + q as f64 * 46.,
            11.,
        );
        l.color = colors[q].into();
        f.labels.push(l);
    }
    f.panels.push(p);
    f.labels.push(Label::new(
        "Only crossings inside 500--1300 km are shown",
        103.,
        96.,
    ));
    f.save(out.join("hplus-transition.png"))?;
    let d: Value = syisr_ac::campaign::load(Path::new(&a[2]))?;
    let rr = d["rows"].as_array().unwrap();
    let mut groups: BTreeMap<i64, Vec<&Value>> = BTreeMap::new();
    for r in rr {
        if r["altitude_km"].as_f64().unwrap() >= 300. && r["altitude_km"].as_f64().unwrap() <= 500.
        {
            groups
                .entry(r["start_utc_s"].as_i64().unwrap())
                .or_default()
                .push(r);
        }
    }
    let mut f = Figure::new(IntendedUse::ArticleWide);
    f.width_mm = 205.;
    f.height_mm = 215.;
    f.font_pt = 10.;
    f.raster_dpi = 240.;
    f.latex.cache = out.join("latex-cache");
    let titles = [
        "Perpendicular meridional plasma drift vs. Fejer--Scherliess",
        "Geographic east: ion velocity and HWM14 neutral wind",
        "Geographic north: ion velocity and HWM14 neutral wind",
        "Along B: ion velocity and projected HWM14 neutral wind",
    ];
    for q in 0..4 {
        let yy = 18. + q as f64 * 44.;
        let limit = if q == 1 { [-250., 150.] } else { [-150., 150.] };
        let mut p = Panel::new(
            Rect::new(22., yy, 170., 30.),
            axes(&x, limit, q == 3, "Velocity (m/s)"),
        );
        for model in 0..2 {
            let mut points = vec![];
            for (t, rows) in &groups {
                let vals: Vec<f64> = rows
                    .iter()
                    .filter_map(|r| {
                        if model == 0 && r["accepted"] != true {
                            return None;
                        }
                        if model == 0 {
                            Some(match q {
                                0 => r["measured_magnetic_m_s"][1].as_f64()?,
                                1 => r["measured_geographic_m_s"][0].as_f64()?,
                                2 => r["measured_geographic_m_s"][1].as_f64()?,
                                _ => r["measured_magnetic_m_s"][2].as_f64()?,
                            })
                        } else {
                            r[[
                                "fejer_scherliess_local_meridional_m_s",
                                "hwm_east_m_s",
                                "hwm_north_m_s",
                                "hwm_along_igrf_b_m_s",
                            ][q]]
                                .as_f64()
                        }
                    })
                    .collect();
                points.push([
                    (*t as f64 + 900. + 28800. - 1759708800.) / 3600.,
                    if vals.len() >= 10 {
                        median(&vals)
                    } else {
                        f64::NAN
                    },
                ]);
            }
            p.push(Layer::Line(Line::new(
                points,
                if model == 0 { colors[2] } else { colors[1] },
            )));
        }
        f.panels.push(p);
        f.labels.push(Label::new(titles[q], 107., yy - 4.));
    }
    for (text, color, pos) in [
        ("Measured ion velocity", colors[2], 63.),
        ("Empirical model reference", colors[1], 155.),
    ] {
        let mut l = Label::new(text, pos, 202.);
        l.color = color.into();
        f.labels.push(l);
    }
    f.labels.push(Label::new(
        "300--500 km medians; ion/wind quantities differ; LOS positive toward radar",
        103.,
        210.,
    ));
    f.save(out.join("drift-wind-reference.png"))?;
    f.save(out.join("drift-wind-reference.pdf"))?;
    Ok(())
}
