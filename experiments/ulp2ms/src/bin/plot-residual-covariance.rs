//! Scientific residual/covariance figures, Rust/rustplotlib only.
use anyhow::{ensure, Result};
use rustplotlib::*;
use std::{fs, path::Path};
fn median(v: &[f64]) -> f64 {
    let mut v: Vec<_> = v.iter().copied().filter(|x| x.is_finite()).collect();
    v.sort_by(f64::total_cmp);
    if v.len() % 2 == 0 {
        0.5 * (v[v.len() / 2 - 1] + v[v.len() / 2])
    } else {
        v[v.len() / 2]
    }
}
fn save(f: &Figure, root: &Path, name: &str) -> Result<()> {
    for ext in ["png", "pdf", "svg"] {
        f.save(root.join(format!("{name}.{ext}")))?;
    }
    Ok(())
}
fn main() -> Result<()> {
    let a: Vec<String> = std::env::args().collect();
    ensure!(a.len() == 2, "plot-residual-covariance audit-root");
    let root = Path::new(&a[1]);
    let h = hdf5::File::open(root.join("audit.h5"))?;
    let get = |n: &str| h.dataset(n).unwrap().read_raw::<f64>().unwrap();
    let ks = get("correlation_kernel");
    let shape = h.dataset("correlation_kernel")?.shape();
    let nb = shape[3];
    let nl = shape[4];
    let mut fig = Figure::new(IntendedUse::ArticleWide);
    fig.width_mm = 236.;
    fig.height_mm = 215.;
    fig.raster_dpi = 180.;
    fig.labels.push(Label::new(
        "SYISR: measured range--lag noise correlations",
        115.,
        5.,
    ));
    let sc = fig.scale(Scale::new(
        [-1., 1.],
        Normalization::Linear,
        ColorMap::diverging(),
    )?);
    for m in 0..3 {
        for c in 0..2 {
            let mut data = vec![];
            for j in 0..nb * nl {
                data.push(median(
                    &(0..48)
                        .map(|t| ks[((t * 3 + m) * 2 + c) * nb * nl + j])
                        .collect::<Vec<_>>(),
                ));
            }
            let mut ax = Axes::new(
                [-120., 120.],
                [0., 300.],
                if m == 2 {
                    r"Lag separation ($\mu$s)"
                } else {
                    ""
                },
                if c == 0 { "Gate separation (km)" } else { "" },
            );
            ax.xticks = vec![-100., 0., 100.];
            ax.yticks = vec![0., 100., 200., 300.];
            let mut p = Panel::new(
                Rect::new(22. + c as f64 * 104., 26. + m as f64 * 55., 80., 40.),
                ax,
            );
            let xe = (0..=nl).map(|j| -125. + 10. * j as f64).collect();
            let ye = (0..=nb).map(|j| -12.5 + 25. * j as f64).collect();
            p.push(Layer::Pcolormesh(Field::new(xe, ye, data, sc)?));
            fig.panels.push(p);
            let label = match m {
                0 => "Full blocks, background subtracted",
                1 => "Full blocks, before subtraction",
                _ => "Disjoint adjacent-block differences",
            };
            fig.labels.push(Label::new(
                format!("{label}: {}", if c == 0 { "real" } else { "imaginary" }),
                62. + c as f64 * 104.,
                18. + m as f64 * 55.,
            ));
        }
    }
    let mut bar = Colorbar::new(
        Rect::new(214., 26., 3., 150.),
        sc,
        "Correlation",
        vec![-1., -0.5, 0., 0.5, 1.],
    );
    bar.horizontal = false;
    fig.colorbars.push(bar);
    fig.labels.push(Label::new(
        "Median of 48 windows; disjoint pairs contain two successive 10-second blocks",
        115.,
        198.,
    ));
    fig.labels.push(Label::new(
        "Gate separations within 6 km of each 25-km bin; white denotes zero, not masking",
        115.,
        205.,
    ));
    save(&fig, root, "range-lag-correlations")?;
    let r = get("nominal_gate_slant_range_km");
    let nr = r.len();
    let all = get("standardized_residual");
    let mean = get("mean_standardized_residual");
    let mut ye = vec![r[0] - (r[1] - r[0]) / 2.];
    ye.extend(r.windows(2).map(|v| (v[0] + v[1]) / 2.));
    ye.push(r[nr - 1] + (r[nr - 1] - r[nr - 2]) / 2.);
    let mut f = Figure::new(IntendedUse::ArticleWide);
    f.width_mm = 236.;
    f.height_mm = 218.;
    f.raster_dpi = 180.;
    f.labels.push(Label::new(
        "SYISR: full-profile ACF fit residuals",
        115.,
        5.,
    ));
    for row in 0..2 {
        let limit = if row == 0 { 5. } else { 2. };
        let sc = f.scale(Scale::new(
            [-limit, limit],
            Normalization::Linear,
            ColorMap::diverging(),
        )?);
        for c in 0..2 {
            let v: Vec<f64> = (0..nr * 60)
                .map(|j| {
                    if row == 0 {
                        all[(24 * nr * 60 + j) * 2 + c]
                    } else {
                        mean[j * 2 + c]
                    }
                })
                .collect();
            let mut ax = Axes::new(
                [5., 605.],
                [ye[0], ye[nr]],
                r"Lag ($\mu$s)",
                if c == 0 {
                    "Nominal gate range (km)"
                } else {
                    ""
                },
            );
            ax.xticks = vec![100., 300., 500.];
            ax.yticks = vec![200., 500., 1000., 1400.];
            let mut p = Panel::new(
                Rect::new(23. + c as f64 * 103., 26. + row as f64 * 84., 80., 62.),
                ax,
            );
            p.push(Layer::Pcolormesh(Field::new(
                (0..=60).map(|j| 5. + 10. * j as f64).collect(),
                ye.clone(),
                v,
                sc,
            )?));
            f.panels.push(p);
            f.labels.push(Label::new(
                format!(
                    "{}: {}",
                    if row == 0 {
                        "Noon reference"
                    } else {
                        "Mean of 48 fitted windows"
                    },
                    if c == 0 { "real" } else { "imaginary" }
                ),
                63. + c as f64 * 103.,
                18. + row as f64 * 84.,
            ));
        }
        let mut b = Colorbar::new(
            Rect::new(214., 26. + row as f64 * 84., 3., 62.),
            sc,
            r"$(\mathrm{data}-\mathrm{model})/\mathrm{SEM}$",
            vec![-limit, 0., limit],
        );
        b.horizontal = false;
        f.colorbars.push(b);
    }
    f.labels.push(Label::new(
        "Reference: 18 Dec 2025, 04:00--04:30 UTC; signed residuals retained",
        115.,
        199.,
    ));
    f.labels.push(Label::new(
        "White denotes zero; no quality masks; colour limits saturate",
        115.,
        207.,
    ));
    save(&f, root, "fit-residuals")?;
    let mut summary: serde_json::Value =
        serde_json::from_slice(&fs::read(root.join("summary.json"))?)?;
    let per = summary["windows"].as_array().unwrap();
    let mut med = serde_json::Map::new();
    for (key, value) in per[0].as_object().unwrap() {
        if value.is_number() && key != "window" {
            let v: Vec<f64> = per.iter().filter_map(|v| v[key].as_f64()).collect();
            med.insert(key.clone(), serde_json::json!(median(&v)));
        }
    }
    for j in 0..2 {
        med.insert(
            format!("heldout_adjacent_lag_half_{}", j + 1),
            serde_json::json!(median(
                &per.iter()
                    .map(|v| v["heldout_difference_adjacent_lag_correlation"][j]
                        .as_f64()
                        .unwrap())
                    .collect::<Vec<_>>()
            )),
        );
    }
    for j in 0..3 {
        med.insert(
            format!("detrended_block_time_{}s", [10, 20, 60][j]),
            serde_json::json!(median(
                &per.iter()
                    .map(
                        |v| v["detrended_block_temporal_correlation_10_20_60_seconds"][j]
                            .as_f64()
                            .unwrap()
                    )
                    .collect::<Vec<_>>()
            )),
        );
    }
    for (c, name) in ["real", "imag"].iter().enumerate() {
        for (b, d) in [(1, 25), (2, 50), (4, 100), (12, 300)] {
            med.insert(
                format!("before_subtraction_range{d}km_{name}"),
                serde_json::json!(median(
                    &(0..48)
                        .map(|t| ks[((t * 3 + 1) * 2 + c) * nb * nl + b * nl + 12])
                        .collect::<Vec<_>>()
                )),
            );
        }
    }
    let mut ratio = get("difference_to_saved_variance_ratio");
    ratio.sort_by(f64::total_cmp);
    summary["variance_ratio_quantiles"] = serde_json::json!({"p05":ratio[ratio.len()/20],"median":ratio[ratio.len()/2],"p95":ratio[ratio.len()*19/20]});
    summary["campaign_medians"] = serde_json::Value::Object(med);
    fs::write(
        root.join("summary.json"),
        serde_json::to_vec_pretty(&summary)?,
    )?;
    println!(
        "{}",
        serde_json::to_string_pretty(&summary["campaign_medians"])?
    );
    Ok(())
}
