//! Compare externally simulated composition, IRI and accepted nominal fits.
use anyhow::{ensure, Result};
use rustplotlib::*;
use serde_json::{json, Value};
use std::{collections::BTreeMap, fs, path::Path};
fn median(v: &mut [f64]) -> Option<f64> {
    if v.is_empty() {
        None
    } else {
        v.sort_by(f64::total_cmp);
        Some(v[v.len() / 2])
    }
}
fn main() -> Result<()> {
    let a: Vec<_> = std::env::args().collect();
    ensure!(
        a.len() == 5,
        "IRI_COMPARISON_HDF MODEL_HDF MODEL_LABEL OUTPUT"
    );
    let iri = hdf5::File::open(&a[1])?;
    let model = hdf5::File::open(&a[2])?;
    let out = Path::new(&a[4]);
    fs::create_dir_all(out)?;
    let meta: Value =
        serde_json::from_slice(&iri.dataset("metadata_json_utf8")?.read_raw::<u8>()?)?;
    let mmeta: Value =
        serde_json::from_slice(&model.dataset("metadata_json_utf8")?.read_raw::<u8>()?)?;
    let q = iri.dataset("iri_query_points")?.read_raw::<f64>()?;
    let mq = model.dataset("query_points")?.read_raw::<f64>()?;
    ensure!(
        q.len() == mq.len() && q.iter().zip(mq).all(|(a, b)| (*a - b).abs() < 1e-9),
        "model query ordering or geometry differs"
    );
    let frac = model.dataset("ion_fraction")?.read_raw::<f64>()?;
    ensure!(frac.len() / 7 == q.len() / 6, "species dimensions");
    let ids: Vec<_> = meta["profile_ids"]
        .as_array()
        .unwrap()
        .iter()
        .map(|v| v.as_str().unwrap())
        .collect();
    let starts: Vec<f64> = ids
        .iter()
        .map(|id| id.split('_').next().unwrap().parse().unwrap())
        .collect();
    let first = starts.iter().copied().min_by(f64::total_cmp).unwrap();
    let mut values: BTreeMap<String, Vec<Vec<f64>>> = BTreeMap::new();
    for name in iri.member_names()? {
        if !name.starts_with("beam-") {
            continue;
        }
        let g = iri.group(&name)?;
        let size = g.dataset("fit_h_fraction")?.size();
        values.insert(name, vec![vec![f64::NAN; size]; 2]);
    }
    for (i, p) in q.chunks_exact(6).enumerate() {
        let pi = p[1] as usize;
        let gate = p[2] as usize;
        let fields: Vec<_> = ids[pi].split('_').collect();
        let beam = format!("beam-{}-{}", fields[1], fields[2]);
        let g = iri.group(&beam)?;
        let shape = g.dataset("fit_h_fraction")?.shape();
        let time = ((starts[pi] - first) / 1800.).round() as usize;
        let idx = gate * shape[1] + time;
        for s in 0..2 {
            values.get_mut(&beam).unwrap()[s][idx] = frac[7 * i + s];
        }
    }
    let cube = hdf5::File::create(out.join("comparison.h5"))?;
    let mut summaries = vec![];
    for (beam, refs) in &values {
        let g = iri.group(beam)?;
        let x = g.dataset("china_local_hours")?.read_raw::<f64>()?;
        let edges = g.dataset("altitude_edges_km")?.read_raw::<f64>()?;
        let shape = g.dataset("fit_h_fraction")?.shape();
        let (ng, nt) = (shape[0], shape[1]);
        let fits = [
            g.dataset("fit_h_fraction")?.read_raw::<f64>()?,
            g.dataset("fit_he_fraction")?.read_raw::<f64>()?,
        ];
        let iref = [
            g.dataset("iri_h_fraction")?.read_raw::<f64>()?,
            g.dataset("iri_he_fraction")?.read_raw::<f64>()?,
        ];
        let mut diffs = vec![vec![f64::NAN; ng * nt]; 2];
        let mut stats = [vec![], vec![]];
        for s in 0..2 {
            for k in 0..ng {
                for t in 0..nt {
                    let i = k * nt + t;
                    diffs[s][i] = fits[s][i] - refs[s][i];
                    if edges[k] >= 1000. && diffs[s][i].is_finite() {
                        stats[s].push(diffs[s][i]);
                    }
                }
            }
        }
        let group = cube.create_group(beam)?;
        group
            .new_dataset_builder()
            .with_data(&edges)
            .create("altitude_edges_km")?;
        group
            .new_dataset_builder()
            .with_data(&x)
            .create("china_local_hours")?;
        for (name, v) in [
            ("fit_h", &fits[0]),
            ("fit_he", &fits[1]),
            ("iri_h", &iref[0]),
            ("iri_he", &iref[1]),
            ("model_h", &refs[0]),
            ("model_he", &refs[1]),
            ("fit_minus_model_h", &diffs[0]),
            ("fit_minus_model_he", &diffs[1]),
        ] {
            group
                .new_dataset::<f64>()
                .shape((ng, nt))
                .deflate(4)
                .create(name)?
                .write_raw(v)?;
        }
        let az: u16 = beam
            .split('-')
            .nth(1)
            .unwrap()
            .trim_start_matches("az")
            .parse()?;
        let el: u16 = beam
            .split('-')
            .nth(2)
            .unwrap()
            .trim_start_matches("el")
            .parse()?;
        let mut fig = Figure::new(IntendedUse::ArticleWide);
        fig.width_mm = 210.;
        fig.height_mm = 258.;
        fig.raster_dpi = 400.;
        fig.latex.cache = out.join("latex-cache");
        for row in 0..4 {
            for col in 0..2 {
                let lim = if col == 0 { 1. } else { 0.5 };
                let scale = fig.scale(Scale::new(
                    if row == 3 { [-lim, lim] } else { [0., lim] },
                    Normalization::Linear,
                    if row == 3 {
                        ColorMap::diverging()
                    } else {
                        ColorMap::turbo()
                    },
                )?);
                let mut ax = Axes::new(
                    [x[0], *x.last().unwrap()],
                    [500., 1300.],
                    if row == 3 { "Local time (UTC+8)" } else { "" },
                    if col == 0 { "Altitude (km)" } else { "" },
                );
                ax.yticks = vec![500., 700., 1000., 1300.];
                ax.xticks = (0..=7)
                    .map(|k| 8. + 4. * k as f64)
                    .filter(|h| *h <= *x.last().unwrap())
                    .collect();
                ax.xticklabels = Some(
                    ax.xticks
                        .iter()
                        .map(|h| format!("{:02}:00", (*h as i32) % 24))
                        .collect(),
                );
                if row < 3 {
                    ax.show_x_axis = false;
                }
                let rect = Rect::new(20. + 98. * col as f64, 23. + 53. * row as f64, 66., 38.);
                let mut p = Panel::new(rect, ax);
                let v = match row {
                    0 => &fits[col],
                    1 => &iref[col],
                    2 => &refs[col],
                    _ => &diffs[col],
                };
                p.pcolormesh(x.clone(), edges.clone(), v.clone(), scale)?;
                fig.panels.push(p);
                let mut b = Colorbar::new(
                    Rect::new(rect.x + 69., rect.y, 2.5, rect.height),
                    scale,
                    "",
                    vec![],
                );
                b.horizontal = false;
                fig.colorbars.push(b);
                for (v, y) in [
                    (if row == 3 { -lim } else { 0. }, rect.y + rect.height),
                    (lim, rect.y),
                ] {
                    fig.labels.push(Label::new(
                        if col == 0 {
                            format!("{v:.0}")
                        } else {
                            format!("{v:.1}")
                        },
                        rect.x + 78.,
                        y,
                    ));
                }
                if row == 3 {
                    fig.labels
                        .push(Label::new("0", rect.x + 78., rect.y + rect.height / 2.));
                }
            }
        }
        fig.labels.push(Label::new("H$^+$ fraction", 53., 7.));
        fig.labels.push(Label::new("He$^+$ fraction", 151., 7.));
        for (row, label) in [
            "Fitted".into(),
            "IRI TBT-2015".into(),
            a[3].clone(),
            format!("Fitted minus {}", a[3]),
        ]
        .iter()
        .enumerate()
        {
            fig.labels
                .push(Label::new(label.clone(), 105., 18. + 53. * row as f64));
        }
        fig.labels.push(Label::new(
            format!(
                "Azimuth {:.0} degrees, elevation {:.0} degrees; 6–7 October 2025",
                az as f64 * 360. / 65536.,
                el as f64 * 360. / 65536.
            ),
            105.,
            247.,
        ));
        fig.save(out.join(format!("{beam}.png")))?;
        summaries.push(json!({"beam":beam,"1000_to1300km_median_fit_minus_model_H_He":[median(&mut stats[0]),median(&mut stats[1])]}));
    }
    let result = json!({"reference":mmeta,"fit_selection":meta["fit_selection"],"model_label":a[3],"summary":summaries});
    cube.new_dataset_builder()
        .with_data(&serde_json::to_vec(&result)?)
        .create("metadata_json_utf8")?;
    fs::write(
        out.join("summary.json"),
        serde_json::to_vec_pretty(&result)?,
    )?;
    println!("Model comparison rendered {} beams", values.len());
    Ok(())
}
