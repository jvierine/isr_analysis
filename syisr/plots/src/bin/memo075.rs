//! Article-size rendering of the existing AC and empirical reference curves for HPLA memo 75.
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
 let a:Vec<_>=std::env::args().collect(); ensure!(a.len()==3,"DRIVERS OUTPUT");
 let out=Path::new(&a[2]);std::fs::create_dir_all(out)?;
    let d: Value = syisr_ac::campaign::load(Path::new(&a[1]))?;
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
    let x: Vec<_> = groups.keys().map(|t| (*t as f64 + 900. + 28800. - 1759708800.)/3600.).collect();
    let colors = ["#111111", "#0072B2", "#D55E00"];
    let mut f = ArticleLayout::new(179.324,10.95)?.figure(1., 204.)?;
    f.raster_dpi=300.; f.latex.cache=out.join("latex-cache");
    let titles = [
        "Perpendicular meridional plasma drift vs. Fejer--Scherliess",
        "Geographic east: ion velocity and HWM14 neutral wind",
        "Geographic north: ion velocity and HWM14 neutral wind",
        "Along B: ion velocity and projected HWM14 neutral wind",
    ];
    for q in 0..4 {
        let yy = 16. + q as f64 * 41.;
        let limit = if q == 1 { [-250., 150.] } else { [-150., 150.] };
        let mut p = Panel::new(
            Rect::new(19., yy, 153., 27.),
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
        f.labels.push(Label::new(titles[q], 95., yy - 4.));
    }
    for (text, color, pos) in [
        ("Measured ion velocity", colors[2], 53.),
        ("Empirical model reference", colors[1], 137.),
    ] {
        let mut l = Label::new(text, pos, 188.);
        l.color = color.into();
        f.labels.push(l);
    }
    f.labels.push(Label::new(
        "300--500 km medians; positive LOS toward radar",
        95.,
        199.,
    ));
    f.save_for_article(out.join("reference.pdf"), "figures/memo075/reference.pdf")?;
    Ok(())
}
