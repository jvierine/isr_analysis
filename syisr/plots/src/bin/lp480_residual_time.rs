//! Time-altitude QA of each short lag. Source subsets retain signed measurements.
use anyhow::{Result, ensure};
use chrono::{DateTime, Utc};
use rustplotlib::*;
use serde_json::Value;
use std::path::Path;
fn main() -> Result<()> {
    let a = std::env::args().collect::<Vec<_>>();
    ensure!(a.len() == 3, "lp480_residual_time INPUT_H5 OUTPUT");
    let out = Path::new(&a[2]);
    std::fs::create_dir_all(out)?;
    let input: Value = syisr_ac::campaign::load(Path::new(&a[1]))?;
    let mut rows = input["rows"]
        .as_array()
        .ok_or_else(|| anyhow::anyhow!("missing rows"))?
        .iter()
        .collect::<Vec<_>>();
    rows.sort_by_key(|v| v["time_utc_s"].as_i64().unwrap());
    ensure!(!rows.is_empty(), "no measured residuals");
    let first = rows[0]["time_utc_s"].as_i64().unwrap().div_euclid(86400) * 86400;
    let last = rows.last().unwrap()["time_utc_s"].as_i64().unwrap();
    let nt = ((last - first) / 1800 + 1) as usize;
    let z = rows[0]["altitude_km"]
        .as_array()
        .unwrap()
        .iter()
        .map(|v| v.as_f64().unwrap())
        .collect::<Vec<_>>();
    let nz = z.len();
    ensure!(nz > 1, "empty gates");
    let mut values = vec![vec![f64::NAN; nt * nz]; 20];
    let mut counts = vec![0usize; nt];
    for row in &rows {
        let time = row["time_utc_s"].as_i64().unwrap();
        ensure!((time - first) % 1800 == 0, "non-half-hour time");
        let t = ((time - first) / 1800) as usize;
        let zz = row["altitude_km"].as_array().unwrap();
        ensure!(
            zz.len() == nz
                && zz
                    .iter()
                    .zip(&z)
                    .all(|(v, z)| (v.as_f64().unwrap() - z).abs() < 1e-6),
            "source gate grid changed: do not interpolate diagnostic residuals"
        );
        let gates = row["gate"].as_array().unwrap();
        for i in 0..gates.len() {
            let g = gates[i].as_u64().unwrap() as usize;
            let l = row["lag"][i].as_u64().unwrap() as usize;
            ensure!(g < nz && (1..=10).contains(&l), "invalid short-lag index");
            for c in 0..2 {
                let y = row["observed"][i][c].as_f64().unwrap();
                let m = row["predicted"][i][c].as_f64().unwrap();
                let s = row["sigma"][i][c].as_f64().unwrap();
                ensure!(
                    y.is_finite() && m.is_finite() && s.is_finite() && s > 0.,
                    "invalid assigned error"
                );
                let cell = &mut values[(l - 1) * 2 + c][g * nt + t];
                ensure!(cell.is_nan(), "duplicate time/gate/lag");
                *cell = (m - y) / s;
            }
            counts[t] += 1;
        }
    }
    let x = (0..=nt).map(|i| i as f64 / 48.).collect::<Vec<_>>();
    let mut edges = vec![z[0] - (z[1] - z[0]) / 2.];
    edges.extend(z.windows(2).map(|p| (p[0] + p[1]) / 2.));
    edges.push(z[nz - 1] + (z[nz - 1] - z[nz - 2]) / 2.);
    let mut f = Figure::new(IntendedUse::ArticleWide);
    f.width_mm = 210.;
    f.height_mm = 447.;
    f.font_pt = 9.;
    f.raster_dpi = 150.;
    f.latex.cache = out.join("latex-cache");
    let date = |t| {
        DateTime::<Utc>::from_timestamp(t, 0)
            .unwrap()
            .format("%d %b %Y")
            .to_string()
    };
    f.labels.push(Label::new(
        format!(
            "SYISR; azimuth {:.2}$^\\circ$, elevation {:.2}$^\\circ$; {}--{}",
            rows[0]["azimuth_deg"].as_f64().unwrap(),
            rows[0]["elevation_deg"].as_f64().unwrap(),
            date(first),
            date(last)
        ),
        105.,
        5.,
    ));
    f.labels
        .push(Label::new("Real normalized residual", 58., 13.));
    f.labels
        .push(Label::new("Imaginary normalized residual", 154., 13.));
    let scale = f.scale(Scale::new(
        [-5., 5.],
        Normalization::Linear,
        ColorMap::diverging(),
    )?);
    for l in 0..10 {
        for c in 0..2 {
            let mut ax = Axes::new(
                [0., nt as f64 / 48.],
                [100., 1800.],
                if l == 9 { "UTC date" } else { "" },
                if c == 0 { "Altitude (km)" } else { "" },
            );
            ax.yticks = vec![100., 700., 1400., 1800.];
            ax.show_x_axis = l == 9;
            let days = nt / 48;
            ax.xticks = (0..=days).step_by(days.div_ceil(6).max(1)).map(|d| d as f64).collect();
            ax.xticklabels = Some(
                ax.xticks
                    .iter()
                    .map(|d| {
                        DateTime::<Utc>::from_timestamp(first + (*d * 86400.) as i64, 0)
                            .unwrap()
                            .format("%d/%m")
                            .to_string()
                    })
                    .collect(),
            );
            let rect = Rect::new(20. + c as f64 * 98., 25. + l as f64 * 39., 72., 29.);
            let mut p = Panel::new(rect, ax);
            p.pcolormesh(x.clone(), edges.clone(), values[l * 2 + c].clone(), scale)?;
            f.panels.push(p);
            f.colorbars.push(Colorbar::new(
                Rect::new(rect.x + 75., rect.y, 2., rect.height),
                scale,
                "",
                vec![-5., 0., 5.],
            ));
            f.labels.push(Label::new(
                format!("{} $\\mu$s", (l + 1) * 10),
                rect.x + 36.,
                rect.y - 3.,
            ));
        }
    }
    f.labels.push(Label::new("(Model--data) / assigned $\\sigma$; before robust weighting; no time or altitude averaging",105.,429.));
    f.labels.push(Label::new("Blank cells are unmeasured; assigned errors are working weights, not independent calibrated errors",105.,437.));
    f.save(out.join("first10-residual-time.png"))?;
    f.save(out.join("first10-residual-time.pdf"))?;
    syisr_ac::campaign::save(
        &out.join("first10-residual-time.h5"),
        &serde_json::json!({"start_utc_s":first,"step_s":1800,"altitude_km":z,"lag_us":[10,20,30,40,50,60,70,80,90,100],"normalized_residual_lag_component_altitude_time":values,"valid_products_per_time":counts,"display_limits":[-5.,5.],"normalization":"(model-data)/assigned diagonal sigma, before robust weighting; no averaging; null=missing","provenance":input["provenance"]}),
    )?;
    Ok(())
}
