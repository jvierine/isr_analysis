//! Provisional shared-composition pilot, with independent beam plasma fields.
use anyhow::{ensure, Result};
use rustplotlib::*;
use serde_json::Value;
use std::path::Path;
fn main() -> Result<()> {
    let a: Vec<_> = std::env::args().collect();
    ensure!(a.len() == 3, "PROFILE_ROOT OUTPUT");
    let root = Path::new(&a[1]);
    let out = Path::new(&a[2]);
    std::fs::create_dir_all(out)?;
    let files = syisr_ac::campaign::files(&root.join("joint-fits"), "h5")?;
    let profiles: Vec<Value> = files
        .iter()
        .map(|p| syisr_ac::campaign::load(p))
        .collect::<Result<_>>()?;
    ensure!(!profiles.is_empty(), "no profiles");
    let tr: syisr_ac::h_transition::Transition =
        serde_json::from_value(profiles[0]["h_transition"].clone())?;
    let he: syisr_ac::h_transition::HeliumLayer =
        serde_json::from_value(profiles[0]["helium_layer"].clone())?;
    let colors = [
        "#0072B2", "#D55E00", "#009E73", "#CC79A7", "#E69F00", "#56B4E9", "#7B3294", "#A65628",
        "#555555",
    ];
    let keys = [
        "ne_m3",
        "ti_k",
        "te_k",
        "velocity_m_s",
        "h_fraction",
        "he_fraction",
    ];
    let labels = [
        r"$N_e$ (m$^{-3}$)",
        r"$T_i$ (K)",
        r"$T_e$ (K)",
        r"$v_i$ (m/s)",
        r"H$^+$ fraction",
        r"He$^+$ fraction",
    ];
    let limits = [
        [1e9, 1e13],
        [150., 6000.],
        [150., 12000.],
        [-2000., 2000.],
        [0., 1.],
        [0., 0.5],
    ];
    let mut f = Figure::new(IntendedUse::ArticleWide);
    f.width_mm = 190.;
    f.height_mm = 180.;
    f.font_pt = 9.;
    f.latex.cache = out.join("latex-cache");
    for k in 0..6 {
        let mut ax = Axes::new(
            limits[k],
            [100., 1300.],
            labels[k],
            if k % 3 == 0 { "Altitude (km)" } else { "" },
        );
        ax.yticks = vec![100., 500., 900., 1300.];
        ax.xticks = match k {
            0 => vec![1e9, 1e11, 1e13],
            1 => vec![150., 3000., 6000.],
            2 => vec![150., 6000., 12000.],
            3 => vec![-2000., 0., 2000.],
            _ => vec![0., limits[k][1] / 2., limits[k][1]],
        };
        if k == 0 {
            ax.set_xscale(AxisScale::Log10)?;
        }
        let mut p = Panel::new(
            Rect::new(
                16. + 61. * (k % 3) as f64,
                21. + 68. * (k / 3) as f64,
                43.,
                49.,
            ),
            ax,
        );
        if k < 4 {
            for (i, v) in profiles.iter().enumerate() {
                let pts = v["gates"]
                    .as_array()
                    .unwrap()
                    .iter()
                    .filter_map(|g| {
                        let z = g["altitude_km"].as_f64()?;
                        let x = if k == 0 {
                            g["ne_m3"].as_f64()?
                        } else {
                            g["node"][keys[k]].as_f64()?
                        };
                        Some([x, z])
                    })
                    .collect();
                p.push(Layer::Line(Line::new(pts, colors[i % colors.len()])));
            }
        } else {
            let pts = (100..=1300)
                .map(|z| {
                    let z = z as f64;
                    let h = tr.shape.evaluate(z, tr.normalized_parameters).0
                        * syisr_ac::spline::h_weight(z);
                    let x = if k == 4 {
                        h
                    } else {
                        (1. - h)
                            * syisr_ac::h_transition::helium(z, he.normalized_parameters).0
                            * syisr_ac::spline::h_weight(z)
                    };
                    [x, z]
                })
                .collect();
            p.push(Layer::Line(Line::new(pts, "black")));
        }
        f.panels.push(p);
    }
    let start = profiles[0]["job"]["start"].as_f64().unwrap() as i64;
    let end = profiles[0]["job"]["end"].as_f64().unwrap() as i64;
    let s = chrono::DateTime::<chrono::Utc>::from_timestamp(start + 28800, 0).unwrap();
    let e = chrono::DateTime::<chrono::Utc>::from_timestamp(end + 28800, 0).unwrap();
    f.labels.push(Label::new(
        format!(
            "SYISR: {}–{} local (UTC+8)",
            s.format("%-d %B %Y, %H:%M"),
            e.format("%H:%M")
        ),
        95.,
        4.,
    ));
    f.labels.push(Label::new(
        "Provisional: iteration limit; temporal regularization not applied yet",
        95.,
        11.,
    ));
    for (i, v) in profiles.iter().enumerate() {
        let az = v["job"]["az"].as_f64().unwrap();
        let el = v["job"]["el"].as_f64().unwrap();
        let mut l = Label::new(
            format!("Az {az:.0} degrees, El {el:.0} degrees"),
            34. + 61. * (i % 3) as f64,
            153. + 7. * (i / 3) as f64,
        );
        l.color = colors[i % colors.len()].into();
        f.labels.push(l);
    }
    f.save(out.join("shared-pilot.png"))?;
    f.save(out.join("shared-pilot.pdf"))?;
    Ok(())
}
