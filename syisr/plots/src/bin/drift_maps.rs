//! Regional velocity and beam-consistency maps from fitted LOS velocities.
use anyhow::{Result, ensure};
use rustplotlib::*;
use serde_json::Value;
use std::{collections::BTreeSet, path::Path};
fn main() -> Result<()> {
    let a: Vec<_> = std::env::args().collect();
    ensure!(a.len() == 3, "DRIFT.h5 OUTPUT");
    let v: Value = syisr_ac::campaign::load(Path::new(&a[1]))?;
    let out = Path::new(&a[2]);
    std::fs::create_dir_all(out)?;
    let r = v["rows"].as_array().unwrap();
    let times: Vec<_> = r
        .iter()
        .map(|v| v["start_utc_s"].as_i64().unwrap())
        .collect::<BTreeSet<_>>()
        .into_iter()
        .collect();
    let zs: Vec<_> = r
        .iter()
        .map(|v| v["altitude_km"].as_i64().unwrap())
        .collect::<BTreeSet<_>>()
        .into_iter()
        .collect();
    let nt = times.len();
    let nz = zs.len();
    let mut x: Vec<_> = times
        .iter()
        .map(|t| (*t as f64 + 28800. - 1759708800.) / 3600.)
        .collect();
    x.push(x.last().unwrap() + 0.5);
    let mut z: Vec<_> = zs.iter().map(|v| *v as f64 - 5.).collect();
    z.push(*zs.last().unwrap() as f64 + 5.);
    let hf = hdf5::File::create(out.join("drift-maps.h5"))?;
    hf.new_dataset_builder()
        .with_data(&times)
        .create("start_utc_s")?;
    hf.new_dataset_builder()
        .with_data(&zs)
        .create("altitude_km")?;
    for (kind, masked) in [
        ("magnetic", false),
        ("magnetic", true),
        ("geographic", false),
        ("diagnostics", false),
    ] {
        let mut fig = Figure::new(IntendedUse::ArticleWide);
        fig.width_mm = 205.;
        fig.height_mm = 198.;
        fig.font_pt = 10.;
        fig.raster_dpi = 220.;
        fig.latex.cache = out.join("latex-cache");
        let labels = if kind == "magnetic" {
            [
                "Magnetic east (m/s)",
                "Meridional perpendicular (m/s)",
                "Along magnetic field (m/s)",
            ]
        } else if kind == "geographic" {
            [
                "Geographic east (m/s)",
                "Geographic north (m/s)",
                "Geographic up (m/s)",
            ]
        } else {
            [
                "LOS residual RMS (m/s)",
                "Geometry condition number",
                "Leave-one-beam-out error (m/s)",
            ]
        };
        for q in 0..3 {
            let mut values = vec![f64::NAN; nt * nz];
            for row in r {
                let t = times
                    .binary_search(&row["start_utc_s"].as_i64().unwrap())
                    .unwrap();
                let h = zs
                    .binary_search(&row["altitude_km"].as_i64().unwrap())
                    .unwrap();
                let fit = &row[if kind == "diagnostics" {
                    "magnetic"
                } else {
                    kind
                }];
                if masked && fit["consistency_ok"] != true {
                    continue;
                }
                values[h * nt + t] = if kind == "diagnostics" {
                    fit[[
                        "residual_rms_m_s",
                        "condition_number",
                        "maximum_leave_one_out_error_m_s",
                    ][q]]
                        .as_f64()
                        .unwrap_or(f64::NAN)
                } else {
                    fit["velocity_m_s"][q].as_f64().unwrap_or(f64::NAN)
                };
            }
            let key = format!("{kind}{}_{}", if masked { "_accepted" } else { "" }, q);
            hf.new_dataset::<f64>()
                .shape((nz, nt))
                .create(key.as_str())?
                .write_raw(&values)?;
            let lim = if kind == "diagnostics" {
                [[0., 50.], [1., 15.], [0., 75.]][q]
            } else {
                [-100., 100.]
            };
            let sc = fig.scale(Scale::new(
                lim,
                Normalization::Linear,
                if kind == "diagnostics" {
                    ColorMap::turbo()
                } else {
                    ColorMap::diverging()
                },
            )?);
            let mut ax = Axes::new(
                [x[0], *x.last().unwrap()],
                [250., 1000.],
                if q == 2 {
                    "6--7 October 2025; China local time (UTC+8)"
                } else {
                    ""
                },
                "Altitude (km)",
            );
            ax.yticks = vec![300., 500., 700., 1000.];
            ax.xticks = vec![8., 12., 16., 20., 24., 28., 32.];
            ax.xticklabels = Some(
                vec![
                    "08:00", "12:00", "16:00", "20:00", "00:00", "04:00", "08:00",
                ]
                .iter()
                .map(|s| s.to_string())
                .collect(),
            );
            if q < 2 {
                ax.show_x_axis = false;
            }
            let yy = 20. + q as f64 * 51.;
            let mut panel = Panel::new(Rect::new(21., yy, 146., 41.), ax);
            panel.pcolormesh(x.clone(), z.clone(), values, sc)?;
            fig.panels.push(panel);
            fig.colorbars
                .push(Colorbar::new(Rect::new(170., yy, 3., 41.), sc, "", vec![]));
            for val in [lim[0], (lim[0] + lim[1]) / 2., lim[1]] {
                let y = yy + 41. * (lim[1] - val) / (lim[1] - lim[0]);
                fig.labels.push(Label::new(format!("{val:.0}"), 180., y));
            }
            let mut lab = Label::new(labels[q], 198., yy + 20.5);
            lab.rotation = -90.;
            fig.labels.push(lab);
        }
        fig.labels.push(Label::new(
            if kind == "diagnostics" {
                "SYISR multibeam velocity consistency"
            } else if masked {
                "SYISR regional drift: beam-consistent estimates"
            } else {
                "SYISR regional drift: unmasked estimates"
            },
            103.,
            7.,
        ));
        fig.labels.push(Label::new(
            if kind == "diagnostics" {
                "Common-flow assumption tested against individual beam velocities"
            } else {
                "AC software polarity retained; absolute physical sign remains unverified"
            },
            103.,
            190.,
        ));
        fig.save(out.join(format!(
            "{kind}{}.png",
            if masked { "-accepted" } else { "" }
        )))?;
    }
    Ok(())
}
