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
    let composition = syisr_long_pulse::composition::from_metadata(&m)?;
    let maximum = m["requested_observation_max_slant_range_km"]
        .as_f64()
        .unwrap_or(1400.);
    let extended = maximum > 2000.;
    let plot_max: f64 = std::env::var("SYISR_PLOT_MAX_KM")
        .ok()
        .map(|v| v.parse())
        .transpose()?
        .unwrap_or(if extended { 5500. } else { 1700. });
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
    let times = get("time_utc_s");
    let start = chrono::DateTime::from_timestamp(times[0] as i64, 0).unwrap();
    let end = chrono::DateTime::from_timestamp(times[1] as i64, 0).unwrap();
    f.labels.push(Label::new(
        format!(
            "{}, {}--{} UTC; nonuniform profile knots",
            start.format("%d %b %Y"),
            start.format("%H:%M"),
            end.format("%H:%M")
        ),
        115.,
        12.,
    ));
    let colors = ["#17608e", "#b55327", "#287c53", "#9c438a", "#9a760b"];
    for p in 0..6 {
        let (yl, lab, ticks) = match p {
            0 => ([0., 1.1], "Hat basis weight", vec![0., 0.5, 1.]),
            1 => (
                [if extended { 7. } else { 9. }, 13.],
                r"$\log_{10} n_e$ (m$^{-3}$)",
                if extended {
                    vec![7., 9., 11., 13.]
                } else {
                    vec![9., 10., 11., 12., 13.]
                },
            ),
            2 => (
                [0., 4000.],
                r"$T_i$ (K)",
                vec![0., 1000., 2000., 3000., 4000.],
            ),
            3 => (
                [0., 8000.],
                r"$T_e$ (K)",
                vec![0., 2000., 4000., 6000., 8000.],
            ),
            4 => ([0., 1.], r"H$^+$ fraction", vec![0., 0.25, 0.5, 0.75, 1.]),
            _ => (
                [0., if extended { 1. } else { 0.06 }],
                r"He$^+$ fraction",
                if extended {
                    vec![0., 0.25, 0.5, 0.75, 1.]
                } else {
                    vec![0., 0.02, 0.04, 0.06]
                },
            ),
        };
        let mut axes = Axes::new([200., plot_max], yl, "Altitude (km)", lab);
        axes.xticks = if plot_max == 2500. {
            vec![500., 1000., 1500., 2000., 2500.]
        } else if extended {
            vec![500., 1500., 2500., 3500., 4500., 5500.]
        } else {
            vec![200., 500., 800., 1100., 1400., 1700.]
        };
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
            let basis: Vec<usize> = if extended {
                (12..z.len() - 1).collect()
            } else {
                vec![3, 4, 5, 6, 7]
            };
            for k in basis {
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
            for j in 0..=(z[z.len() - 1] - z[0]).round() as usize {
                let zz = z[0] + j as f64;
                let weights = interpolation(&z, zz)?;
                let mut q = [0.; 6];
                for (k, w) in weights {
                    for v in 0..6 {
                        q[v] += w * x[6 * k + v];
                    }
                }
                if let Some(join) = m["bottomside_density"]["join_slant_range_km"].as_f64() {
                    let k = z
                        .iter()
                        .position(|&v| (v - join - 0.05).abs() < 1e-8)
                        .unwrap();
                    if let Some(d) = syisr_long_pulse::profile::bottomside_mapping(
                        &z,
                        k,
                        m["bottomside_density"]["scale_height_km"].as_f64().unwrap(),
                        zz,
                    ) {
                        q[0] = d.offset + d.weights.iter().map(|&(k, w)| w * x[6 * k]).sum::<f64>();
                    }
                }
                if let Some(shape) = &composition {
                    let (c, _) = shape.composition(zz, &x);
                    q[4] = c[0];
                    q[5] = c[1];
                }
                points.push([zz, value(&q)]);
            }
            // Grey sections are guard regions, not independently supported retrievals.
            let lower = if m["partial_products"].is_object() {
                200.
            } else {
                500.
            };
            for (lo, hi, color) in [
                (80., lower, "#aaaaaa"),
                (lower, maximum + 0.05, "#17608e"),
                (maximum + 0.05, z[z.len() - 1] + 1., "#aaaaaa"),
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
            if composition.is_none() || p < 4 {
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
        }
        f.panels.push(panel);
    }
    f.labels.push(Label::new(
        if composition.is_some() {
            "Dots: plasma knots; composition: fitted sigmoid and Gaussian layer"
        } else {
            "Dots: fitted knots; curves: interpolation; includes bound-limited fits"
        },
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
