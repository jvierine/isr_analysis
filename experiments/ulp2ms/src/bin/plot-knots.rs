//! Memo 61: actual noon-reference knots and exact transformed interpolation.
use anyhow::{ensure, Result};
use rustplotlib::*;
use syisr_long_pulse::profile::interpolation;
fn main() -> Result<()> {
    let a: Vec<String> = std::env::args().collect();
    ensure!(a.len() == 3, "plot-knots profile.h5 output-prefix");
    let h = hdf5::File::open(&a[1])?;
    let get = |n: &str| h.dataset(n).unwrap().read_raw::<f64>().unwrap();
    let z = get("altitude_km");
    let m: serde_json::Value = serde_json::from_slice(&h.attr("metadata_json")?.read_raw::<u8>()?)?;
    let x: Vec<f64> = m["fit"]["transformed_profile"]
        .as_array()
        .unwrap()
        .iter()
        .map(|v| v.as_f64().unwrap())
        .collect();
    ensure!(x.len() == 6 * z.len(), "profile dimensions");
    let mut f = Figure::new(IntendedUse::ArticleWide);
    f.width_mm = 230.;
    f.height_mm = 218.;
    f.raster_dpi = 170.;
    f.labels.push(Label::new(
        "SYISR: profile basis and fitted knots",
        115.,
        5.,
    ));
    f.labels.push(Label::new(
        "18 Dec 2025, 04:00--04:30 UTC; 100 km knot spacing",
        115.,
        12.,
    ));
    let colors = ["#17608e", "#b55327", "#287c53", "#9c438a", "#9a760b"];
    for p in 0..6 {
        let (yl, lab, ticks) = match p {
            0 => ([0., 1.1], "Hat basis weight", vec![0., 0.5, 1.]),
            1 => (
                [9., 12.1],
                r"$\log_{10} n_e$ (m$^{-3}$)",
                vec![9., 10., 11., 12.],
            ),
            2 => ([0., 4500.], r"$T_i$ (K)", vec![0., 1500., 3000., 4500.]),
            3 => ([0., 6000.], r"$T_e$ (K)", vec![0., 2000., 4000., 6000.]),
            4 => ([0., 0.75], r"H$^+$ fraction", vec![0., 0.25, 0.5, 0.75]),
            _ => ([0., 0.06], r"He$^+$ fraction", vec![0., 0.02, 0.04, 0.06]),
        };
        let mut axes = Axes::new([200., 1700.], yl, "Altitude (km)", lab);
        axes.xticks = vec![200., 500., 800., 1100., 1400., 1700.];
        axes.yticks = ticks;
        let mut panel = Panel::new(
            Rect::new(
                21. + (p % 2) as f64 * 114.,
                25. + (p / 2) as f64 * 59.,
                88.,
                42.,
            ),
            axes,
        );
        // Mark every actual model knot with a faint vertical guide.
        for &zz in &z {
            let mut l = Line::new(vec![[zz, yl[0]], [zz, yl[1]]], "#dddddd");
            l.width_pt = 0.45;
            panel.push(Layer::Line(l));
        }
        if p == 0 {
            for k in [3usize, 4, 5, 6, 7] {
                panel.push(Layer::Line(Line::new(
                    vec![[z[k - 1], 0.], [z[k], 1.], [z[k + 1], 0.]],
                    colors[(k - 3) % 5],
                )));
            }
        } else {
            let value = |q: &[f64]| match p {
                1 => 11. + q[0] / std::f64::consts::LN_10,
                2 => 1000. * q[1].exp(),
                3 => 1000. * q[2].exp(),
                4 => q[4],
                _ => (1. - q[4]) * q[5],
            };
            let mut points = vec![];
            for j in 0..=1500 {
                let zz = z[0] + j as f64;
                let weights = interpolation(&z, zz)?;
                let mut q = [0.; 6];
                for (k, w) in weights {
                    for v in 0..6 {
                        q[v] += w * x[6 * k + v];
                    }
                }
                points.push([zz, value(&q)]);
            }
            // Grey sections are guard regions, not independently supported retrievals.
            for (lo, hi, color) in [
                (200., 500., "#aaaaaa"),
                (500., 1400., "#17608e"),
                (1400., 1701., "#aaaaaa"),
            ] {
                panel.push(Layer::Line(Line::new(
                    points
                        .iter()
                        .copied()
                        .filter(|v| v[0] >= lo && v[0] <= hi)
                        .collect(),
                    color,
                )));
            }
            let pts: Vec<_> = z
                .iter()
                .enumerate()
                .map(|(k, &zz)| [zz, value(&x[6 * k..6 * k + 6])])
                .collect();
            panel.markers(
                &mut f,
                &pts,
                &MarkerStyle {
                    shape: MarkerShape::Circle,
                    size_pt: 3.5,
                    color: [0.09, 0.38, 0.56],
                },
            )?;
        }
        f.panels.push(panel);
    }
    f.labels.push(Label::new(
        "Dots: fitted knot values; curves: transformed interpolation",
        115.,
        204.,
    ));
    f.labels.push(Label::new(
        "Grey: guard regions; knot spacing does not specify measurement resolution",
        115.,
        211.,
    ));
    for ext in ["png", "pdf", "svg"] {
        f.save(format!("{}.{ext}", a[2]))?;
    }
    Ok(())
}
