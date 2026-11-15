use anyhow::Result;
use rustplotlib::*;
use serde_json::Value;
use std::{fs, path::Path};
fn main() -> Result<()> {
    let a: Vec<_> = std::env::args().collect();
    let p: Value = serde_json::from_slice(&fs::read(&a[1])?)?;
    let out = Path::new(&a[2]);
    fs::create_dir_all(out)?;
    let heights: Vec<_> = p["altitude_km"]
        .as_array()
        .unwrap()
        .iter()
        .map(|v| v.as_f64().unwrap())
        .collect();
    let minutes=p["integration_s"].as_f64().unwrap()/60.;
    let nt=p["h_fraction"].as_array().unwrap().len();
    let time_edges:Vec<_>=(0..=nt).map(|i|15.+i as f64*minutes/60.).collect();
    let mut edges = vec![heights[0] - (heights[1] - heights[0]) / 2.];
    for w in heights.windows(2) {
        edges.push((w[0] + w[1]) / 2.);
    }
    edges.push(
        heights.last().unwrap() + (heights[heights.len() - 1] - heights[heights.len() - 2]) / 2.,
    );
    let mut fig = Figure::new(IntendedUse::ArticleWide);
    fig.width_mm = 190.;
    fig.height_mm = 175.;
    fig.font_pt = 9.;
    fig.latex.cache = out.join("latex-cache");
    for q in 0..2 {
        let scale = fig.scale(Scale::new(
            [0., 1.],
            Normalization::Linear,
            ColorMap::plasma(),
        )?);
        let mut ax = Axes::new(
            [15., 21.],
            [350., 1700.],
            "UTC on 2023-11-15",
            "Altitude (km)",
        );
        ax.xticks = (15..=21).map(|x| x as f64).collect();
        ax.xticklabels = Some((15..=21).map(|x| format!("{x}:00")).collect());
        ax.yticks = vec![400., 600., 800., 1000., 1200., 1400., 1600.];
        let rect = Rect::new(22., 13. + q as f64 * 74., 137., 52.);
        let mut panel = Panel::new(rect, ax);
        let mut values = vec![];
        for z in 0..heights.len() {
            for t in 0..nt {
                let flag = p["quality_flag"][t][z].as_u64().unwrap();
                let v = if q == 0 {
                    p["h_fraction"][t][z].as_f64().unwrap()
                } else {
                    p["h_fraction_ci_upper"][t][z].as_f64().unwrap()
                        - p["h_fraction_ci_lower"][t][z].as_f64().unwrap()
                };
                values.push(if flag == 2 { f64::NAN } else { v });
            }
        }
        panel.pcolormesh(
            time_edges.clone(),
            edges.clone(),
            values,
            scale,
        )?;
        for z in 0..heights.len() {
            for t in 0..nt {
                if p["quality_flag"][t][z] == 1 {
                    panel.markers(
                        &mut fig,
                        &[[15.+(t as f64+0.5)*minutes/60., heights[z]]],
                        &MarkerStyle {
                            shape: MarkerShape::Circle,
                            size_pt: 3.2,
                            color: [1., 1., 1.],
                        },
                    )?;
                }
            }
        }
        fig.panels.push(panel);
        fig.colorbars.push(Colorbar::new(
            Rect::new(163., rect.y, 3., rect.height),
            scale,
            if q == 0 {
                r"$H^+$ fraction"
            } else {
                "Conditional interval width"
            },
            vec![0., 0.25, 0.5, 0.75, 1.],
        ));
        fig.labels.push(Label::new(
            if q == 0 {
                "SYISR nighttime zenith: fitted hydrogen-ion fraction"
            } else {
                "Uncertainty: conditional profile interval width"
            },
            95.,
            rect.y - 7.,
        ));
    }
    fig.labels.push(Label::new(
        format!("{minutes:.0}-minute integrations; ~100 km footprint. White dots: conditional support."),
        95.,
        160.,
    ));
    fig.labels.push(Label::new(
        "Blank: failed or boundary fit. Broad intervals: weak information.",
        95.,
        167.,
    ));
    fig.save(out.join(a.get(3).map(String::as_str).unwrap_or("hplus-time-altitude.png")))?;
    Ok(())
}
