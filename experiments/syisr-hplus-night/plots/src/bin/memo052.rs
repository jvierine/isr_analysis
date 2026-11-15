use anyhow::Result;
use rustplotlib::*;
use serde_json::Value;
use std::{fs, path::Path};
fn val(v: &Value, k: &str) -> f64 {
    v[k].as_f64().unwrap()
}
fn line(p: &mut Panel, points: Vec<[f64; 2]>, color: &str, dash: bool) {
    let mut l = Line::new(points, color);
    l.width_pt = 1.;
    if dash {
        l.dash = Some("3,2".into());
    }
    p.push(Layer::Line(l));
}
fn figure(out: &Path, h: f64) -> Figure {
    let mut f = Figure::new(IntendedUse::ArticleWide);
    f.width_mm = 185.;
    f.height_mm = h;
    f.font_pt = 9.;
    f.latex.cache = out.join("latex-cache");
    f
}
fn save(f: &Figure, out: &Path, n: &str) -> Result<()> {
    f.save(out.join(format!("{n}.pdf")))?;
    f.save(out.join(format!("{n}.png")))?;
    Ok(())
}
fn main() -> Result<()> {
    let a: Vec<_> = std::env::args().collect();
    let d: Value = serde_json::from_slice(&fs::read(&a[1])?)?;
    let out = Path::new(&a[2]);
    fs::create_dir_all(out)?;
    let primary = &d["primary"];
    let nodes = primary["best"]["nodes"].as_array().unwrap();
    let knots: Vec<_> = primary["knots"]
        .as_array()
        .unwrap()
        .iter()
        .map(|x| x.as_f64().unwrap() + 0.05)
        .collect();
    let mut f = figure(out, 90.);
    let mut ax = Axes::new(
        [0., 1.],
        [250., 1100.],
        r"$H^+$ number fraction",
        "Altitude (km)",
    );
    ax.xticks = vec![0., 0.25, 0.5, 0.75, 1.];
    ax.yticks = vec![300., 500., 700., 900., 1100.];
    let mut p = Panel::new(Rect::new(20., 10., 60., 49.), ax);
    for (s, c, ds) in [
        ("", "#146d9a", false),
        ("-full", "#c66a21", true),
        ("-pedestal", "#367e50", true),
    ] {
        let n = d["controls"][s]["nodes"].as_array().unwrap();
        line(
            &mut p,
            (0..6)
                .map(|k| [val(&n[k], "h_fraction"), knots[k]])
                .collect(),
            c,
            ds,
        );
    }
    for k in [3, 4, 5] {
        let h = val(&nodes[k], "h_fraction");
        let chi = val(&primary["best"], "chi_square");
        let mut pts: Vec<_> = d["profiles"]
            .as_array()
            .unwrap()
            .iter()
            .filter(|r| r["knot"] == k)
            .map(|r| [val(r, "h_fraction"), val(r, "chi_square") - chi])
            .collect();
        pts.push([h, 0.]);
        pts.sort_by(|a, b| a[0].total_cmp(&b[0]));
        let mut bounds = [h, h];
        for w in pts.windows(2) {
            for v in w {
                if v[1] <= 3.84 {
                    bounds[0] = bounds[0].min(v[0]);
                    bounds[1] = bounds[1].max(v[0]);
                }
            }
            if (w[0][1] - 3.84) * (w[1][1] - 3.84) < 0. {
                let x = w[0][0] + (w[1][0] - w[0][0]) * (3.84 - w[0][1]) / (w[1][1] - w[0][1]);
                bounds[0] = bounds[0].min(x);
                bounds[1] = bounds[1].max(x);
            }
        }
        p.errorbars(
            &[[h, knots[k]]],
            Some(&[bounds]),
            None,
            "#146d9a",
            0.7,
            2.,
            1.,
        )?;
    }
    f.panels.push(p);
    f.labels.push(Label::new("Range-coupled profiles", 50., 4.));
    let mut ax = Axes::new(
        [0., 1.],
        [0., 25.],
        r"Fixed $H^+$ fraction",
        r"Profile $\Delta Q$",
    );
    ax.xticks = vec![0., 0.25, 0.5, 0.75, 1.];
    ax.yticks = vec![0., 5., 10., 15., 20., 25.];
    let mut p = Panel::new(Rect::new(114., 10., 60., 49.), ax);
    for (k, c) in [(3, "#146d9a"), (4, "#c66a21"), (5, "#367e50")] {
        let mut points: Vec<_> = d["profiles"]
            .as_array()
            .unwrap()
            .iter()
            .filter(|r| r["knot"] == k)
            .map(|r| {
                [
                    val(r, "h_fraction"),
                    (val(r, "chi_square") - val(&primary["best"], "chi_square")).max(0.),
                ]
            })
            .collect();
        points.push([val(&nodes[k], "h_fraction"), 0.]);
        points.sort_by(|a, b| a[0].total_cmp(&b[0]));
        line(&mut p, points, c, false);
    }
    line(&mut p, vec![[0., 3.84], [1., 3.84]], "#888888", true);
    f.panels.push(p);
    f.labels
        .push(Label::new("Refit all nuisance parameters", 144., 4.));
    for (text, x, y, color) in [
        ("Blue: short response", 48., 72., "#146d9a"),
        ("Orange: full reference", 48., 77., "#c66a21"),
        ("Green: pedestal separated", 48., 82., "#367e50"),
        ("Blue: 725 km", 143., 72., "#146d9a"),
        ("Orange: 875 km", 143., 77., "#c66a21"),
        ("Green: 1025 km", 143., 82., "#367e50"),
    ] {
        let mut l = Label::new(text, x, y + 3.);
        l.color = color.into();
        f.labels.push(l);
    }
    save(&f, out, "likelihood")?;
    let mut f = figure(out, 122.);
    let mut ax = Axes::new(
        [-50., 50.],
        [0., 1.15],
        "Offset from range centre (km)",
        "Ideal overlap",
    );
    ax.xticks = vec![-40., -20., 0., 20., 40.];
    ax.yticks = vec![0., 0.5, 1.];
    let mut p = Panel::new(Rect::new(20., 10., 60., 31.), ax);
    for (l, c) in [
        (0, "#146d9a"),
        (100, "#c66a21"),
        (300, "#367e50"),
        (400, "#a43d8a"),
    ] {
        let half = 299792.458 * (480. - l as f64) * 1e-6 / 4.;
        line(
            &mut p,
            vec![
                [-50., 0.],
                [-half, 0.],
                [-half, 1.],
                [half, 1.],
                [half, 0.],
                [50., 0.],
            ],
            c,
            false,
        );
    }
    f.panels.push(p);
    f.labels
        .push(Label::new("Same centre, unequal widths", 50., 4.));
    let mut ax = Axes::new(
        [100., 400.],
        [0., 80.],
        r"Lag ($\mu$s)",
        r"Reference ($\mathrm{ADU}^2$)",
    );
    ax.xticks = vec![100., 200., 300., 400.];
    ax.yticks = vec![0., 20., 40., 60., 80.];
    let mut p = Panel::new(Rect::new(114., 10., 60., 31.), ax);
    let n0 = val(&d, "noise_power_adu2");
    let noise = d["noise_acf"].as_array().unwrap();
    line(
        &mut p,
        (10..41)
            .map(|i| [i as f64 * 10., noise[i][0].as_f64().unwrap() * n0])
            .collect(),
        "#146d9a",
        false,
    );
    line(
        &mut p,
        vec![
            [100., val(&d, "pedestal_adu2")],
            [400., val(&d, "pedestal_adu2")],
        ],
        "#c66a21",
        true,
    );
    f.panels.push(p);
    f.labels
        .push(Label::new("Far-range long-lag pedestal", 144., 4.));
    let products = primary["products"].as_array().unwrap();
    let predicted = primary["best"]["predicted"].as_array().unwrap();
    let mut selected = vec![];
    for l in 0..41 {
        let k = (2. * 725. / 299792.458 + 240e-6 - l as f64 * 5e-6 - 767e-6) / 1e-5;
        let i = products
            .iter()
            .enumerate()
            .filter(|(_, p)| p["lag"] == l)
            .min_by(|(_, a), (_, b)| {
                (val(a, "sample") - k)
                    .abs()
                    .total_cmp(&(val(b, "sample") - k).abs())
            })
            .unwrap()
            .0;
        selected.push(i);
    }
    let norm = predicted[selected[0]][0].as_f64().unwrap();
    for q in 0..2 {
        let mut ax = Axes::new(
            [0., 400.],
            [-0.15, 1.1],
            r"Lag ($\mu$s)",
            if q == 0 {
                "Product / fitted zero lag"
            } else {
                ""
            },
        );
        ax.xticks = vec![0., 100., 200., 300., 400.];
        ax.yticks = vec![0., 0.5, 1.];
        let mut p = Panel::new(Rect::new(20. + 94. * q as f64, 72., 60., 31.), ax);
        let points: Vec<_> = selected
            .iter()
            .enumerate()
            .map(|(l, &i)| {
                [
                    l as f64 * 10.,
                    products[i]["value"][q].as_f64().unwrap() / norm,
                ]
            })
            .collect();
        let errors: Vec<_> = selected
            .iter()
            .map(|&i| {
                let y = products[i]["value"][q].as_f64().unwrap() / norm;
                let e = if q == 1 && products[i]["lag"] == 0 {
                    0.
                } else {
                    products[i]["sigma"][q].as_f64().unwrap() / norm
                };
                [y - e, y + e]
            })
            .collect();
        p.errorbars(&points, None, Some(&errors), "#555555", 0.5, 1.2, 0.8)?;
        p.markers(
            &mut f,
            &points,
            &MarkerStyle {
                shape: MarkerShape::Circle,
                size_pt: 2.2,
                color: [0.25, 0.25, 0.25],
            },
        )?;
        line(
            &mut p,
            selected
                .iter()
                .enumerate()
                .map(|(l, &i)| [l as f64 * 10., predicted[i][q].as_f64().unwrap() / norm])
                .collect(),
            "#146d9a",
            false,
        );
        f.panels.push(p);
        f.labels.push(Label::new(
            if q == 0 {
                "725 km: real products"
            } else {
                "725 km: imaginary products"
            },
            50. + 94. * q as f64,
            66.,
        ));
    }
    f.labels.push(Label::new(
        r"Lags: 0 (blue), 100 (orange), 300 (green), 400 $\mu$s (purple)",
        92.,
        53.,
    ));
    f.labels.push(Label::new(
        "Black: measured products and SEM. Blue: range-coupled prediction.",
        92.,
        117.,
    ));
    save(&f, out, "response")?;
    let mut f = figure(out, 110.);
    for q in 0..2 {
        let scale = f.scale(Scale::new(
            [0., 1.],
            Normalization::Linear,
            ColorMap::plasma(),
        )?);
        let mut ax = Axes::new(
            [15., 21.],
            [200., 1700.],
            "UTC on 2023-11-15",
            "Altitude (km)",
        );
        ax.xticks = (15..=21).map(|i| i as f64).collect();
        ax.xticklabels = Some((15..=21).map(|i| format!("{i}:00")).collect());
        ax.yticks = vec![300., 600., 900., 1200., 1500.];
        let rect = Rect::new(20., 10. + q as f64 * 52., 139., 32.);
        let mut p = Panel::new(rect, ax);
        let mut bg = Line::new(
            vec![
                [15., 200.],
                [21., 200.],
                [21., 1700.],
                [15., 1700.],
                [15., 200.],
            ],
            "#e5e5e5",
        );
        bg.width_pt = 0.;
        bg.fill = Some("#e5e5e5".into());
        p.push(Layer::Line(bg));
        let mut z = vec![];
        for k in 0..10 {
            for h in 0..6 {
                let n = &d["night"][h];
                z.push(if n["quality_rejected"][k].as_bool().unwrap() {
                    f64::NAN
                } else {
                    val(
                        &n["best"]["nodes"][k],
                        if q == 0 { "h_fraction" } else { "he_fraction" },
                    )
                });
            }
        }
        p.pcolormesh(
            (15..=21).map(|i| i as f64).collect(),
            (0..=10).map(|i| 200. + i as f64 * 150.).collect(),
            z,
            scale,
        )?;
        f.panels.push(p);
        f.colorbars.push(Colorbar::new(
            Rect::new(163., rect.y, 3., 32.),
            scale,
            if q == 0 {
                r"$H^+$ fraction"
            } else {
                r"$He^+$ fraction"
            },
            vec![0., 0.25, 0.5, 0.75, 1.],
        ));
        f.labels.push(Label::new(
            if q == 0 {
                "Range-coupled H+ best profiles"
            } else {
                "Jointly fitted He+ best profiles"
            },
            92.,
            rect.y - 6.,
        ));
    }
    save(&f, out, "night")?;
    Ok(())
}
