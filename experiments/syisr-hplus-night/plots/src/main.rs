use anyhow::Result;
use rustplotlib::*;
use serde_json::Value;
use std::{fs, path::Path};
fn array(v: &Value) -> Vec<f64> {
    v.as_array()
        .unwrap()
        .iter()
        .map(|v| v.as_f64().unwrap())
        .collect()
}
fn main() -> Result<()> {
    let a: Vec<_> = std::env::args().collect();
    let p: Value = serde_json::from_slice(&fs::read(&a[1])?)?;
    let b: Value = serde_json::from_slice(&fs::read(&a[2])?)?;
    let out = Path::new(&a[3]);
    fs::create_dir_all(out)?;
    let row = p["results"]
        .as_array()
        .unwrap()
        .iter()
        .find(|x| x["metadata"]["height_km"] == 700.)
        .unwrap();
    let null = row["profile_likelihood"]
        .as_array()
        .unwrap()
        .iter()
        .find(|x| x["h_fraction"] == 0.)
        .unwrap();
    let norm = row["best"]["signal_adu2"].as_f64().unwrap();
    let mut fig = Figure::new(IntendedUse::ArticleWide);
    fig.width_mm = 190.;
    fig.height_mm = 165.;
    fig.font_pt = 9.;
    fig.latex.cache = out.join("latex-cache");
    for q in 0..2 {
        let y = array(
            &row[if q == 0 {
                "measured_real"
            } else {
                "measured_imag"
            }],
        );
        let sem = array(&row[if q == 0 { "sem_real" } else { "sem_imag" }]);
        let mut ax = Axes::new(
            [0., 400.],
            [-0.2, 1.1],
            r"Lag ($\mu$s)",
            if q == 0 {
                "ACF / fitted ion-line power"
            } else {
                ""
            },
        );
        ax.xticks = vec![0., 100., 200., 300., 400.];
        ax.yticks = vec![0., 0.5, 1.];
        let rect = Rect::new(19. + q as f64 * 94., 13., 68., 42.);
        let mut panel = Panel::new(rect, ax);
        let points: Vec<_> = y
            .iter()
            .enumerate()
            .map(|(l, &v)| [l as f64 * 10., v / norm])
            .collect();
        let bounds: Vec<_> = y
            .iter()
            .zip(&sem)
            .map(|(y, s)| [(y - s) / norm, (y + s) / norm])
            .collect();
        panel.errorbars(&points, None, Some(&bounds), "#333333", 0.55, 2., 0.75)?;
        panel.markers(
            &mut fig,
            &points,
            &MarkerStyle {
                shape: MarkerShape::Circle,
                size_pt: 2.5,
                color: [0.2, 0.2, 0.2],
            },
        )?;
        let mut low = vec![];
        let mut high = vec![];
        for l in 0..41 {
            let mut v: Vec<_> = b["resamples"]
                .as_array()
                .unwrap()
                .iter()
                .map(|x| x["fit"]["predicted"][l][q].as_f64().unwrap() / norm)
                .collect();
            v.sort_by(f64::total_cmp);
            low.push([l as f64 * 10., v[5]]);
            high.push([l as f64 * 10., v[194]]);
        }
        let mut polygon = low;
        polygon.extend(high.into_iter().rev());
        let mut band = Line::new(polygon, "#146d9a");
        band.width_pt = 0.;
        band.fill = Some("#146d9a".into());
        band.fill_opacity = 0.3;
        panel.push(Layer::Line(band));
        for (f, color, dashed) in [(&row["best"], "#146d9a", false), (null, "#c66a21", true)] {
            let mut line = Line::new(
                f["predicted"]
                    .as_array()
                    .unwrap()
                    .iter()
                    .enumerate()
                    .map(|(l, v)| [l as f64 * 10., v[q].as_f64().unwrap() / norm])
                    .collect(),
                color,
            );
            line.width_pt = 1.;
            if dashed {
                line.dash = Some("3,2".into());
            }
            panel.push(Layer::Line(line));
        }
        fig.panels.push(panel);
        fig.labels.push(Label::new(
            if q == 0 {
                "Real ACF -- 700 km"
            } else {
                "Imaginary ACF -- same vertical scale"
            },
            rect.x + rect.width / 2.,
            6.,
        ));
    }
    let mut ax = Axes::new(
        [0., 0.6],
        [0., 100.],
        r"$H^+$ number fraction",
        r"Conditional $\Delta\chi^2$",
    );
    ax.xticks = vec![0., 0.2, 0.4, 0.6];
    ax.yticks = vec![0., 25., 50., 75., 100.];
    let rect = Rect::new(19., 84., 68., 42.);
    let mut panel = Panel::new(rect, ax);
    let base = row["best"]["chi_square"].as_f64().unwrap();
    let mut pts: Vec<_> = row["profile_likelihood"]
        .as_array()
        .unwrap()
        .iter()
        .map(|f| {
            [
                f["h_fraction"].as_f64().unwrap(),
                (f["chi_square"].as_f64().unwrap() - base).max(0.),
            ]
        })
        .collect();
    pts.push([row["best"]["h_fraction"].as_f64().unwrap(), 0.]);
    pts.sort_by(|a, b| a[0].total_cmp(&b[0]));
    panel.push(Layer::Line(Line::new(pts, "#146d9a")));
    let mut threshold = Line::new(vec![[0., 3.84], [0.6, 3.84]], "#888888");
    threshold.dash = Some("2,2".into());
    panel.push(Layer::Line(threshold));
    fig.panels.push(panel);
    fig.labels
        .push(Label::new("H fraction profile likelihood", 53., 76.));
    let mut ax = Axes::new(
        [16., 17.],
        [0., 0.5],
        "UTC hour on 2023-11-15",
        r"$H^+$ number fraction",
    );
    ax.xticks = vec![16., 16.25, 16.5, 16.75, 17.];
    ax.xticklabels = Some(
        ["16:00", "16:15", "16:30", "16:45", "17:00"]
            .map(str::to_string)
            .to_vec(),
    );
    ax.yticks = vec![0., 0.1, 0.2, 0.3, 0.4, 0.5];
    let rect = Rect::new(113., 84., 68., 42.);
    let mut panel = Panel::new(rect, ax);
    let t0 = 1700006400_f64;
    let points: Vec<_> = b["individual_integrations"]
        .as_array()
        .unwrap()
        .iter()
        .map(|v| {
            [
                (v["utc_start_s"].as_f64().unwrap() - t0) / 3600.,
                v["best"]["h_fraction"].as_f64().unwrap(),
            ]
        })
        .collect();
    panel.markers(
        &mut fig,
        &points,
        &MarkerStyle {
            shape: MarkerShape::Circle,
            size_pt: 3.2,
            color: [0.08, 0.43, 0.6],
        },
    )?;
    let ci = array(&b["bootstrap_h_fraction_percentiles"]);
    let mut band = Line::new(
        vec![
            [16., ci[0]],
            [17., ci[0]],
            [17., ci[2]],
            [16., ci[2]],
            [16., ci[0]],
        ],
        "#146d9a",
    );
    band.width_pt = 0.;
    band.fill = Some("#146d9a".into());
    band.fill_opacity = 0.3;
    panel.push(Layer::Line(band));
    panel.push(Layer::Line(Line::new(
        vec![
            [16., row["best"]["h_fraction"].as_f64().unwrap()],
            [17., row["best"]["h_fraction"].as_f64().unwrap()],
        ],
        "#146d9a",
    )));
    fig.panels.push(panel);
    fig.labels.push(Label::new(
        "Separate dwells; hourly resampling band",
        147.,
        76.,
    ));
    fig.labels.push(Label::new(
        "SYISR zenith, 16:00--17:00 UTC; long-pulse footprint about 100 km",
        95.,
        145.,
    ));
    fig.labels.push(Label::new(
        "Black: measurements; blue: H/He/O fit and resampling band; orange dashed: H=0",
        95.,
        152.,
    ));
    fig.labels.push(Label::new("19,849 retained pulses; receiver, Debye density and species-temperature assumptions remain conditional",95.,159.));
    fig.save(out.join("hplus-proof.png"))?;
    let mut fig = Figure::new(IntendedUse::ArticleWide);
    fig.width_mm = 177.;
    fig.height_mm = 130.;
    fig.font_pt = 9.;
    fig.latex.cache = out.join("latex-cache");
    let mut ax = Axes::new(
        [0., 1.],
        [350., 1650.],
        r"$H^+$ number fraction",
        "Altitude (km)",
    );
    ax.xticks = vec![0., 0.2, 0.4, 0.6, 0.8, 1.];
    ax.yticks = vec![400., 600., 800., 1000., 1200., 1400., 1600.];
    let rect = Rect::new(20., 12., 64., 87.);
    let mut panel = Panel::new(rect, ax);
    for v in p["results"].as_array().unwrap() {
        let f = &v["best"];
        let h = v["metadata"]["height_km"].as_f64().unwrap();
        if f["ti_k"].as_f64().unwrap() < 310.
            || f["te_k"].as_f64().unwrap() < 310.
            || f["chi_square"].as_f64().unwrap() / f["dof"].as_f64().unwrap() > 2.
        {
            continue;
        }
        let chi = f["chi_square"].as_f64().unwrap();
        let hbest = f["h_fraction"].as_f64().unwrap();
        let mut q: Vec<_> = v["profile_likelihood"]
            .as_array()
            .unwrap()
            .iter()
            .map(|f| {
                (
                    f["h_fraction"].as_f64().unwrap(),
                    f["chi_square"].as_f64().unwrap() - chi,
                )
            })
            .collect();
        q.push((hbest, 0.));
        q.sort_by(|a, b| a.0.total_cmp(&b.0));
        let mut lo = hbest;
        let mut hi = hbest;
        for p in q.windows(2) {
            if p[0].1 <= 3.84 {
                lo = lo.min(p[0].0);
                hi = hi.max(p[0].0);
            }
            if (p[0].1 - 3.84) * (p[1].1 - 3.84) < 0. {
                let t = (3.84 - p[0].1) / (p[1].1 - p[0].1);
                let x = p[0].0 + t * (p[1].0 - p[0].0);
                lo = lo.min(x);
                hi = hi.max(x);
            }
            if p[1].1 <= 3.84 {
                lo = lo.min(p[1].0);
                hi = hi.max(p[1].0);
            }
        }
        let good = hi - lo < 0.5 && lo > 0.;
        let x = if good { hbest } else { 0.5 };
        panel.errorbars(
            &[[x, h]],
            Some(&[[lo, hi]]),
            None,
            if good { "#146d9a" } else { "#aaaaaa" },
            0.8,
            3.,
            1.,
        )?;
        if good {
            panel.markers(
                &mut fig,
                &[[x, h]],
                &MarkerStyle {
                    shape: MarkerShape::Circle,
                    size_pt: 3.5,
                    color: [0.08, 0.43, 0.6],
                },
            )?;
        }
    }
    fig.panels.push(panel);
    fig.labels
        .push(Label::new("Conditional composition constraints", 52., 5.));
    let mut ax = Axes::new(
        [16., 17.],
        [0., 2200.],
        "UTC hour on 2023-11-15",
        "Temperature (K)",
    );
    ax.xticks = vec![16., 16.25, 16.5, 16.75, 17.];
    ax.xticklabels = Some(
        ["16:00", "16:15", "16:30", "16:45", "17:00"]
            .map(str::to_string)
            .to_vec(),
    );
    ax.yticks = vec![0., 500., 1000., 1500., 2000.];
    let rect = Rect::new(105., 12., 64., 87.);
    let mut panel = Panel::new(rect, ax);
    for (k, color) in [("ti_k", [0.08, 0.43, 0.6]), ("te_k", [0.78, 0.42, 0.13])] {
        let pts: Vec<_> = b["individual_integrations"]
            .as_array()
            .unwrap()
            .iter()
            .map(|v| {
                [
                    (v["utc_start_s"].as_f64().unwrap() - 1700006400.) / 3600.,
                    v["best"][k].as_f64().unwrap(),
                ]
            })
            .collect();
        panel.markers(
            &mut fig,
            &pts,
            &MarkerStyle {
                shape: MarkerShape::Circle,
                size_pt: 3.2,
                color,
            },
        )?;
    }
    fig.panels.push(panel);
    fig.labels
        .push(Label::new("700 km: Ti (blue), Te (orange)", 137., 5.));
    fig.labels.push(Label::new(
        "Grey bars: weak constraints or upper limits. Boundary-temperature and poor fits omitted.",
        88.,
        116.,
    ));
    fig.labels.push(Label::new(
        "SYISR zenith, 2023-11-15 16:00--17:00 UTC; pulse/range response about 100 km",
        88.,
        124.,
    ));
    fig.save(out.join("hplus-altitude.png"))?;
    Ok(())
}
