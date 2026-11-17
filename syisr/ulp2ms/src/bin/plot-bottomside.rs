//! Actual forward-fit boundary sensitivity; prescribed shapes distinguished visually.
use anyhow::{ensure, Result};
use rustplotlib::*;
use syisr_long_pulse::profile::{bottomside_mapping, interpolation};
struct P {
    z: Vec<f64>,
    x: Vec<f64>,
    bottom: Option<(f64, f64)>,
    chi: f64,
    status: String,
}
fn read(path: &str) -> Result<P> {
    let f = hdf5::File::open(path)?;
    let m: serde_json::Value = serde_json::from_slice(&f.attr("metadata_json")?.read_raw::<u8>()?)?;
    Ok(P {
        z: f.dataset("slant_range_km")?.read_raw::<f64>()?,
        x: m["fit"]["transformed_profile"]
            .as_array()
            .unwrap()
            .iter()
            .map(|v| v.as_f64().unwrap())
            .collect(),
        bottom: m["bottomside_density"]["join_slant_range_km"]
            .as_f64()
            .map(|r| {
                (
                    r,
                    m["bottomside_density"]["scale_height_km"].as_f64().unwrap(),
                )
            }),
        chi: m["fit"]["data_chi_square"].as_f64().unwrap(),
        status: m["fit"]["status"].as_str().unwrap().into(),
    })
}
fn state(p: &P, r: f64) -> Result<[f64; 6]> {
    let mut q = [0.; 6];
    for (k, w) in interpolation(&p.z, r.max(p.z[0]))? {
        for j in 0..6 {
            q[j] += w * p.x[6 * k + j];
        }
    }
    if let Some((peak, h)) = p.bottom {
        let k = p.z.iter().position(|&v| v == peak).unwrap();
        if let Some(m) = bottomside_mapping(&p.z, k, h, r) {
            q[0] = m.offset + m.weights.iter().map(|&(j, w)| w * p.x[6 * j]).sum::<f64>();
        }
    }
    Ok(q)
}
fn main() -> Result<()> {
    let a: Vec<String> = std::env::args().collect();
    ensure!(
        a.len() == 6,
        "plot-bottomside baseline H50 H70 H100 output-prefix"
    );
    let profiles = a[1..5]
        .iter()
        .map(|s| read(s))
        .collect::<Result<Vec<_>>>()?;
    ensure!(
        profiles.iter().all(|p| p.status == "converged"),
        "nonconverged fit"
    );
    let colors = ["#888888", "#9857a0", "#17608e", "#23845b"];
    let mut f = Figure::new(IntendedUse::ArticleWide);
    f.width_mm = 230.;
    f.height_mm = 130.;
    f.raster_dpi = 180.;
    f.labels.push(Label::new(
        "SYISR: F2 bottomside boundary sensitivity",
        115.,
        5.,
    ));
    f.labels.push(Label::new(
        "18 Dec 2025, 04:00--04:30 UTC; fixed gain and identical screened ACFs",
        115.,
        12.,
    ));
    for panel in 0..3 {
        let (xl, yl, lab, xt, yt) = match panel {
            0 => (
                [100., 800.],
                [0., 1.6],
                r"$n_e$ ($10^{12}$ m$^{-3}$)",
                vec![100., 300., 500., 700.],
                vec![0., 0.5, 1., 1.5],
            ),
            1 => (
                [200., 1400.],
                [8., 12.2],
                r"$\log_{10} n_e$ (m$^{-3}$)",
                vec![200., 600., 1000., 1400.],
                vec![8., 9., 10., 11., 12.],
            ),
            _ => (
                [500., 1400.],
                [0., 0.8],
                r"H$^+$ fraction",
                vec![500., 800., 1100., 1400.],
                vec![0., 0.2, 0.4, 0.6, 0.8],
            ),
        };
        let mut ax = Axes::new(xl, yl, "Altitude (km)", lab);
        ax.xticks = xt;
        ax.yticks = yt;
        let mut p = Panel::new(Rect::new(20. + panel as f64 * 74., 27., 55., 65.), ax);
        for (i, profile) in profiles.iter().enumerate() {
            for bottom in [true, false] {
                let pts = (xl[0] as usize..=xl[1] as usize)
                    .filter(|&r| {
                        let r = r as f64;
                        if i == 0 && r < 200. {
                            return false;
                        }
                        (r < profile.bottom.map(|v| v.0).unwrap_or(400.)) == bottom
                    })
                    .map(|r| {
                        let q = state(profile, r as f64)?;
                        Ok([
                            r as f64 + 0.05,
                            match panel {
                                0 => 0.1 * q[0].exp(),
                                1 => 11. + q[0] / std::f64::consts::LN_10,
                                _ => q[4],
                            },
                        ])
                    })
                    .collect::<Result<Vec<_>>>()?;
                let mut line = Line::new(pts, colors[i]);
                if bottom && i > 0 {
                    line.dash = Some("1.5 1".into());
                }
                p.push(Layer::Line(line));
            }
        }
        f.panels.push(p);
    }
    f.labels.push(Label::new(
        format!(
            "Grey: free knots; purple / blue / green: H = {:.0} / {:.0} / {:.0} km",
            profiles[1].bottom.unwrap().1,
            profiles[2].bottom.unwrap().1,
            profiles[3].bottom.unwrap().1
        ),
        115.,
        108.,
    ));
    f.labels.push(Label::new(
        "Dashed: prescribed bottomside; solid: fitted upper profiles with a smooth join",
        115.,
        115.,
    ));
    f.labels.push(Label::new(
        format!("Prescribed joins: {:.0} / {:.0} / {:.0} km; density scale provisional; E/F1 layers omitted",profiles[1].bottom.unwrap().0,profiles[2].bottom.unwrap().0,profiles[3].bottom.unwrap().0),
        115.,
        122.,
    ));
    for ext in ["png", "pdf", "svg"] {
        f.save(format!("{}.{ext}", a[5]))?;
    }
    let base = &profiles[0];
    let mut summary = vec![];
    for p in &profiles {
        let mut maxh = 0f64;
        let mut maxne = 0f64;
        for r in (500..=1400).step_by(25) {
            let x = state(base, r as f64)?;
            let y = state(p, r as f64)?;
            maxh = maxh.max(100. * (y[4] - x[4]).abs());
            maxne = maxne.max(100. * ((y[0] - x[0]).exp() - 1.).abs());
        }
        summary.push(serde_json::json!({"H_km":p.bottom.map(|v|v.1),"data_chi_square":p.chi,"delta_data_chi_square":p.chi-base.chi,"max_H_change_percentage_points_500_1400":maxh,"max_ne_change_percent_500_1400":maxne}));
    }
    std::fs::write(
        format!("{}.json", a[5]),
        serde_json::to_vec_pretty(&summary)?,
    )?;
    println!("{}", serde_json::to_string_pretty(&summary)?);
    Ok(())
}
