//! Direct absolute-fraction comparison; Sanya AC fit plus five model rows.
use anyhow::{ensure, Result};
use rustplotlib::*;
use serde_json::{json, Value};
use std::{collections::BTreeMap, fs, path::Path};
fn median(v: &mut Vec<f64>) -> Option<f64> {
    v.retain(|x| x.is_finite());
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
        a.len() == 7,
        "IRI_CUBE SAMI2_HDF TTS_HDF DY_HDF GCPM_HDF OUTPUT"
    );
    let iri = hdf5::File::open(&a[1])?;
    let out = Path::new(&a[6]);
    fs::create_dir_all(out)?;
    let q = iri.dataset("iri_query_points")?.read_raw::<f64>()?;
    let meta: Value =
        serde_json::from_slice(&iri.dataset("metadata_json_utf8")?.read_raw::<u8>()?)?;
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
    let names = [
        "Sanya AC fit",
        "SAMI2",
        "TBT-2015",
        "TTS-2003",
        "DY-1985",
        "GCPM v2.4",
    ];
    let mut models = vec![];
    let mut modelmeta = vec![];
    for file in &a[2..6] {
        let f = hdf5::File::open(file)?;
        let p = f.dataset("query_points")?.read_raw::<f64>()?;
        ensure!(
            p.len() == q.len() && p.iter().zip(&q).all(|(a, b)| (*a - b).abs() < 1e-9),
            "unmatched model coordinates"
        );
        models.push(f.dataset("ion_fraction")?.read_raw::<f64>()?);
        modelmeta.push(serde_json::from_slice::<Value>(
            &f.dataset("metadata_json_utf8")?.read_raw::<u8>()?,
        )?);
    }
    let mut beams: BTreeMap<String, Vec<Vec<f64>>> = BTreeMap::new();
    for name in iri.member_names()? {
        if name.starts_with("beam-") {
            let g = iri.group(&name)?;
            let n = g.dataset("fit_h_fraction")?.size();
            let mut d = vec![vec![f64::NAN; n]; 12];
            d[0] = g.dataset("fit_h_fraction")?.read_raw()?;
            d[1] = g.dataset("fit_he_fraction")?.read_raw()?;
            d[4] = g.dataset("iri_h_fraction")?.read_raw()?;
            d[5] = g.dataset("iri_he_fraction")?.read_raw()?;
            beams.insert(name, d);
        }
    }
    for (i, p) in q.chunks_exact(6).enumerate() {
        let pi = p[1] as usize;
        let gate = p[2] as usize;
        let pieces: Vec<_> = ids[pi].split('_').collect();
        let beam = format!("beam-{}-{}", pieces[1], pieces[2]);
        let ng = iri.group(&beam)?.dataset("fit_h_fraction")?.shape();
        let nt = ng[1];
        let ti = ((starts[pi] - first) / 1800.).round() as usize;
        let idx = gate * nt + ti;
        for (m, row) in [(0, 1), (1, 3), (2, 4), (3, 5)] {
            for s in 0..2 {
                beams.get_mut(&beam).unwrap()[2 * row + s][idx] = models[m][7 * i + s];
            }
        }
    }
    let cube = hdf5::File::create(out.join("five-model-comparison.h5"))?;
    let mut summaries = vec![];
    for (beam, data) in &beams {
        let g = iri.group(beam)?;
        let x = g.dataset("china_local_hours")?.read_raw::<f64>()?;
        let edges = g.dataset("altitude_edges_km")?.read_raw::<f64>()?;
        let accepted = g.dataset("accepted_profile")?.read_raw::<u8>()?;
        let nt = x.len() - 1;
        let ng = edges.len() - 1;
        let group = cube.create_group(beam)?;
        group
            .new_dataset_builder()
            .with_data(&edges)
            .create("altitude_edges_km")?;
        group
            .new_dataset_builder()
            .with_data(&x)
            .create("china_local_hours")?;
        group
            .new_dataset_builder()
            .with_data(&accepted)
            .create("accepted_profile")?;
        let mut stats = vec![];
        for row in 0..6 {
            for s in 0..2 {
                let field = format!(
                    "{}_{}",
                    ["sanya", "sami2", "tbt2015", "tts2003", "dy1985", "gcpm"][row],
                    if s == 0 { "h_fraction" } else { "he_fraction" }
                );
                group
                    .new_dataset::<f64>()
                    .shape((ng, nt))
                    .deflate(4)
                    .create(field.as_str())?
                    .write_raw(&data[2 * row + s])?;
            }
            let mut med = vec![];
            for (low, high) in [(800., 1000.), (1000., 1301.)] {
                let mut vals = [vec![], vec![]];
                for k in 0..ng {
                    let z = (edges[k] + edges[k + 1]) / 2.;
                    if z < low || z >= high {
                        continue;
                    }
                    for t in 0..nt {
                        if accepted[t] == 0 {
                            continue;
                        }
                        for s in 0..2 {
                            vals[s].push(data[2 * row + s][k * nt + t]);
                        }
                    }
                }
                med.push(json!({"altitude_band_km":[low,high],"median_H_He":[median(&mut vals[0]),median(&mut vals[1])]}));
            }
            stats.push(json!({"model":names[row],"bands":med}));
        }
        let ar: f64 = beam
            .split('-')
            .nth(1)
            .unwrap()
            .trim_start_matches("az")
            .parse::<f64>()?
            * 360.
            / 65536.;
        let er: f64 = beam
            .split('-')
            .nth(2)
            .unwrap()
            .trim_start_matches("el")
            .parse::<f64>()?
            * 360.
            / 65536.;
        let mut fig = ArticleLayout::new(179.324, 10.95)?.figure(1., 232.)?;
        fig.raster_dpi = 500.;
        fig.latex.cache = out.join("latex-cache");
        for row in 0..6 {
            for col in 0..2 {
                let scale = fig.scale(Scale::new(
                    if col == 0 { [0., 1.] } else { [0., 0.5] },
                    Normalization::Linear,
                    ColorMap::turbo(),
                )?);
                let mut ax = Axes::new(
                    [x[0], *x.last().unwrap()],
                    [500., 1300.],
                    if row == 5 { "Local time (UTC+8)" } else { "" },
                    if col == 0 { "Altitude (km)" } else { "" },
                );
                ax.yticks = vec![500., 700., 1000., 1300.];
                ax.xticks = vec![8., 16., 24., 32.];
                ax.xticklabels = Some(vec![
                    "08:00".into(),
                    "16:00".into(),
                    "00:00".into(),
                    "08:00".into(),
                ]);
                if row < 5 {
                    ax.show_x_axis = false;
                }
                let rect = Rect::new(18. + 82. * col as f64, 32. + 32. * row as f64, 56., 22.);
                let mut p = Panel::new(rect, ax);
                p.pcolormesh(x.clone(), edges.clone(), data[2 * row + col].clone(), scale)?;
                fig.panels.push(p);
                let mut bar =
                    Colorbar::new(Rect::new(rect.x + 59., rect.y, 2., 22.), scale, "", vec![]);
                bar.horizontal = false;
                fig.colorbars.push(bar);
                for (v, y) in [
                    (0., rect.y + rect.height),
                    (if col == 0 { 1. } else { 0.5 }, rect.y),
                ] {
                    fig.labels.push(Label::new(
                        if col == 0 {
                            format!("{v:.0}")
                        } else {
                            format!("{v:.1}")
                        },
                        rect.x + 68.,
                        y,
                    ));
                }
            }
            fig.labels.push(Label::new(
                if row == 1 {
                    "SAMI2: Fejer–Scherliess electric drifts; HWM14 neutral winds"
                } else {
                    names[row]
                },
                89.662,
                27. + 32. * row as f64,
            ));
        }
        fig.labels.push(Label::new(
            format!("Sanya: azimuth {ar:.0} degrees, elevation {er:.0} degrees; 6–7 October 2025"),
            89.662,
            4.,
        ));
        fig.labels.push(Label::new("H$^+$ fraction", 46., 21.));
        fig.labels.push(Label::new("He$^+$ fraction", 128., 21.));
        fig.save(out.join(format!("{beam}.png")))?;
        fig.save_for_article(
            out.join(format!("{beam}.pdf")),
            &format!("figures/memo071/{beam}.pdf"),
        )?;
        summaries.push(json!({"beam":beam,"accepted_profiles":accepted.iter().filter(|x|**x==1).count(),"time_bins":nt,"fraction_statistics":stats}));
    }
    let meta = json!({"row_order":names,"reference_details":modelmeta,"fit_selection":meta["fit_selection"],"reference_TBT2015":meta["reference"],"DY1985_note":"original DY model domain extends to1000km; upper values are the IRI implementation's extension","summary":summaries});
    cube.new_dataset_builder()
        .with_data(&serde_json::to_vec(&meta)?)
        .create("metadata_json_utf8")?;
    fs::write(out.join("summary.json"), serde_json::to_vec_pretty(&meta)?)?;
    println!(
        "Absolute-fraction comparison ready for {} beams",
        beams.len()
    );
    Ok(())
}
