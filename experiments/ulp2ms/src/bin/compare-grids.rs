//! Compare unchanged screened observations under uniform/nonuniform profile bases.
use anyhow::{ensure, Result};
use rustplotlib::*;
use std::path::Path;
use syisr_long_pulse::profile::interpolation;
struct Profile {
    bottomside: Option<(usize, f64)>,
    z: Vec<f64>,
    x: Vec<f64>,
    chi: f64,
}
fn read(root: &str, t: usize) -> Result<Profile> {
    let f = hdf5::File::open(Path::new(root).join(format!("windows/{t:03}/profile.h5")))?;
    let m: serde_json::Value = serde_json::from_slice(&f.attr("metadata_json")?.read_raw::<u8>()?)?;
    ensure!(
        m["fit"]["status"] == "converged",
        "window {t} not converged"
    );
    Ok(Profile {
        bottomside: m["bottomside_density"]["join_slant_range_km"]
            .as_f64()
            .map(|join| {
                let z = f
                    .dataset("slant_range_km")
                    .unwrap()
                    .read_raw::<f64>()
                    .unwrap();
                (
                    z.iter().position(|v| (*v - join).abs() < 1e-8).unwrap(),
                    m["bottomside_density"]["scale_height_km"].as_f64().unwrap(),
                )
            }),
        z: f.dataset("slant_range_km")?.read_raw::<f64>()?,
        x: m["fit"]["transformed_profile"]
            .as_array()
            .unwrap()
            .iter()
            .map(|v| v.as_f64().unwrap())
            .collect(),
        chi: m["fit"]["data_chi_square"].as_f64().unwrap(),
    })
}
fn physical(p: &Profile, r: f64) -> Result<[f64; 6]> {
    let mut q = [0.; 6];
    for (k, w) in interpolation(&p.z, r)? {
        for j in 0..6 {
            q[j] += w * p.x[6 * k + j];
        }
    }
    if let Some((k, h)) = p.bottomside {
        if let Some(d) = syisr_long_pulse::profile::bottomside_mapping(&p.z, k, h, r) {
            q[0] = d.offset + d.weights.iter().map(|&(j, w)| w * p.x[6 * j]).sum::<f64>();
        }
    }
    Ok([
        1e11 * q[0].exp(),
        1000. * q[1].exp(),
        1000. * q[2].exp(),
        q[4],
        (1. - q[4]) * q[5],
        (1. - q[4]) * (1. - q[5]),
    ])
}
fn main() -> Result<()> {
    let a: Vec<String> = std::env::args().collect();
    ensure!(
        a.len() == 4,
        "compare-grids uniform-root nonuniform-root output-prefix"
    );
    let ranges: Vec<f64> = (500..=1400).step_by(25).map(|v| v as f64).collect();
    let mut diffs = vec![];
    let mut delta_chi = vec![];
    for t in 0..48 {
        let old = read(&a[1], t)?;
        let new = read(&a[2], t)?;
        delta_chi.push(new.chi - old.chi);
        for &r in &ranges {
            let x = physical(&old, r)?;
            let y = physical(&new, r)?;
            for k in 0..6 {
                diffs.push(if k < 3 {
                    100. * (y[k] / x[k] - 1.)
                } else {
                    100. * (y[k] - x[k])
                });
            }
        }
    }
    let out = hdf5::File::create(format!("{}.h5", a[3]))?;
    out.new_dataset::<f64>()
        .shape((48, ranges.len(), 6))
        .create("nonuniform_minus_uniform")?
        .write_raw(&diffs)?;
    out.new_dataset_builder()
        .with_data(&ranges)
        .create("slant_range_km")?;
    out.new_dataset_builder()
        .with_data(&delta_chi)
        .create("delta_data_chi_square")?;
    out.new_attr_builder().with_data("ne, Ti, Te relative percent; H, He, O percentage-point differences; all converged fits, unmasked comparison, 500..1400 km; no precision claim").create("difference_units_and_scope")?;
    let mut summary = serde_json::Map::new();
    for (k, name) in [
        "ne_relative_percent",
        "ti_relative_percent",
        "te_relative_percent",
        "h_percentage_points",
        "he_percentage_points",
        "o_percentage_points",
    ]
    .iter()
    .enumerate()
    {
        let mut v: Vec<f64> = diffs.chunks(6).map(|x| x[k].abs()).collect();
        v.sort_by(f64::total_cmp);
        summary.insert(name.to_string(),serde_json::json!({"median_absolute":v[v.len()/2],"p95_absolute":v[(v.len() as f64*0.95) as usize],"max_absolute":v[v.len()-1]}));
    }
    delta_chi.sort_by(f64::total_cmp);
    summary.insert(
        "delta_data_chi_square".into(),
        serde_json::json!({"min":delta_chi[0],"median":delta_chi[24],"max":delta_chi[47]}),
    );
    summary.insert("scope".into(),serde_json::json!("All 48 fitted windows and 500..1400 km, including weakly constrained estimates; differences are sensitivity to grid choice, not uncertainty intervals."));
    std::fs::write(
        format!("{}.json", a[3]),
        serde_json::to_vec_pretty(&summary)?,
    )?;
    let mut f = Figure::new(IntendedUse::ArticleWide);
    f.width_mm = 230.;
    f.height_mm = 190.;
    f.raster_dpi = 170.;
    f.labels.push(Label::new(
        "SYISR full-profile inversion: knot-grid comparison",
        115.,
        5.,
    ));
    f.labels.push(Label::new(
        "18 Dec 2025, 04:00--04:30 UTC; fixed gain and identical screened ACFs",
        115.,
        12.,
    ));
    let old = read(&a[1], 24)?;
    let new = read(&a[2], 24)?;
    for (k, label, lim, ticks) in [
        (
            0,
            r"$\log_{10} n_e$ (m$^{-3}$)",
            [9., 12.],
            vec![9., 10., 11., 12.],
        ),
        (1, r"$T_i$ (K)", [300., 4000.], vec![1000., 2000., 3000.]),
        (2, r"$T_e$ (K)", [300., 5000.], vec![1000., 2500., 4000.]),
        (
            3,
            r"H$^+$ fraction",
            [0., 0.8],
            vec![0., 0.2, 0.4, 0.6, 0.8],
        ),
        (
            4,
            r"He$^+$ fraction",
            [0., 0.08],
            vec![0., 0.02, 0.04, 0.06, 0.08],
        ),
        (
            5,
            r"O$^+$ fraction",
            [0., 1.],
            vec![0., 0.25, 0.5, 0.75, 1.],
        ),
    ] {
        let mut ax = Axes::new(
            lim,
            [500., 1400.],
            label,
            if k % 3 == 0 { "Altitude (km)" } else { "" },
        );
        ax.xticks = ticks;
        ax.yticks = vec![500., 800., 1100., 1400.];
        let mut panel = Panel::new(
            Rect::new(
                20. + (k % 3) as f64 * 74.,
                27. + (k / 3) as f64 * 74.,
                55.,
                54.,
            ),
            ax,
        );
        for (p, col) in [(&old, "#b55327"), (&new, "#17608e")] {
            let transform = |v: f64| if k == 0 { v.log10() } else { v };
            let pts = (500..=1400)
                .map(|r| {
                    let r = r as f64;
                    Ok([transform(physical(p, r)?[k]), r + 0.05])
                })
                .collect::<Result<Vec<_>>>()?;
            panel.push(Layer::Line(Line::new(pts, col)));
            if col == "#17608e" {
                let pts =
                    p.z.iter()
                        .filter(|&&r| r >= 500. && r <= 1400.)
                        .map(|&r| Ok([transform(physical(p, r)?[k]), r + 0.05]))
                        .collect::<Result<Vec<_>>>()?;
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
        "Blue + dots: nonuniform knots; orange: uniform 100 km knots",
        115.,
        174.,
    ));
    f.labels.push(Label::new(
        "Identical range--lag response and smoothing rule; He+ remains weakly constrained",
        115.,
        181.,
    ));
    for ext in ["png", "pdf", "svg"] {
        f.save(format!("{}.{ext}", a[3]))?;
    }
    println!("{}", serde_json::to_string_pretty(&summary)?);
    Ok(())
}
