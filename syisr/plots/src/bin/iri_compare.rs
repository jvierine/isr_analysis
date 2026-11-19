//! Direct IRI2020/TBT2015 composition at each fitted gate location and midpoint.
use anyhow::{ensure, Result};
use chrono::{DateTime, Datelike, Timelike, Utc};
use rustplotlib::*;
use serde_json::{json, Value};
use std::{
    collections::BTreeMap,
    fs,
    io::Write,
    path::Path,
    process::{Command, Stdio},
};
#[derive(Clone)]
struct Profile {
    v: Value,
    az: f64,
    el: f64,
    id: String,
    start: f64,
    end: f64,
    valid: bool,
    iri: Vec<[f64; 4]>,
}
fn median(v: &mut [f64]) -> Option<f64> {
    if v.is_empty() {
        return None;
    }
    v.sort_by(f64::total_cmp);
    Some(v[v.len() / 2])
}
fn main() -> Result<()> {
    let a: Vec<_> = std::env::args().collect();
    ensure!(
        a.len() == 5,
        "FIT_ROOT IRI_BATCH_DRIVER IRI_DATA_DIRECTORY OUTPUT"
    );
    let root = Path::new(&a[1]);
    let out = Path::new(&a[4]);
    fs::create_dir_all(out)?;
    let c: syisr_ac::campaign::Campaign = syisr_ac::campaign::load(&root.join("campaign.h5"))?;
    let mut ps = vec![];
    for e in fs::read_dir(root.join("joint-fits"))? {
        let p = e?.path();
        if p.extension().is_none_or(|s| s != "h5") {
            continue;
        }
        let v: Value = syisr_ac::campaign::load(&p)?;
        let j = &v["job"];
        let status = v["nominal_fit"]["status"].as_str().unwrap_or("");
        let valid = status.starts_with("converged")
            && v["nominal_fit"]["normalized_gradient"]
                .as_f64()
                .unwrap_or(f64::INFINITY)
                < 0.05;
        ps.push(Profile {
            az: j["az"].as_f64().unwrap(),
            el: j["el"].as_f64().unwrap(),
            start: j["start"].as_f64().unwrap(),
            end: j["end"].as_f64().unwrap(),
            id: j["id"].as_str().unwrap().into(),
            valid,
            iri: vec![],
            v,
        });
    }
    ps.sort_by(|a, b| a.id.cmp(&b.id));
    let input = out.join("iri-query.tmp");
    let output = out.join("iri-output.tmp");
    let mut f = fs::File::create(&input)?;
    let mut queries = vec![];
    let mut query_points = vec![];
    for (pidx, p) in ps.iter_mut().enumerate() {
        let gates = p.v["gates"].as_array().unwrap();
        p.iri = vec![[f64::NAN; 4]; gates.len()];
        let t = DateTime::<Utc>::from_timestamp(((p.start + p.end) / 2.) as i64, 0).unwrap();
        let hour = t.hour() as f64 + t.minute() as f64 / 60. + t.second() as f64 / 3600.;
        let lat = 18.3492f64.to_radians();
        let lon = 109.6222f64.to_radians();
        let az = p.az.to_radians();
        let el = p.el.to_radians();
        let up = [lat.cos() * lon.cos(), lat.cos() * lon.sin(), lat.sin()];
        let north = [-lat.sin() * lon.cos(), -lat.sin() * lon.sin(), lat.cos()];
        let east = [-lon.sin(), lon.cos(), 0.];
        for (gidx, g) in gates.iter().enumerate() {
            let z = g["altitude_km"].as_f64().unwrap();
            if z < 500. || z > 1300. {
                continue;
            }
            let r = g["range_km"].as_f64().unwrap();
            let pos: Vec<_> = (0..3)
                .map(|i| {
                    6371.05 * up[i]
                        + r * (el.sin() * up[i]
                            + el.cos() * (az.cos() * north[i] + az.sin() * east[i]))
                })
                .collect();
            let glat = (pos[2] / pos.iter().map(|x| x * x).sum::<f64>().sqrt())
                .asin()
                .to_degrees();
            let glon = pos[1].atan2(pos[0]).to_degrees();
            writeln!(
                f,
                "{} {} {} {:.10} {:.10} {:.10} {:.10}",
                t.year(),
                t.month(),
                t.day(),
                hour,
                glat,
                glon,
                z
            )?;
            queries.push((pidx, gidx, z));
            query_points.extend([
                (p.start + p.end) / 2.,
                pidx as f64,
                gidx as f64,
                glat,
                glon,
                z,
            ]);
        }
    }
    drop(f);
    let status = Command::new(fs::canonicalize(&a[2])?)
        .current_dir(&a[3])
        .stdin(Stdio::from(fs::File::open(&input)?))
        .stdout(Stdio::from(fs::File::create(&output)?))
        .status()?;
    ensure!(status.success(), "IRI batch failed");
    let text = fs::read_to_string(&output)?;
    let rows: Vec<_> = text.lines().filter(|s| !s.trim().is_empty()).collect();
    ensure!(
        rows.len() == queries.len(),
        "IRI output length {} differs from {} queries",
        rows.len(),
        queries.len()
    );
    for ((p, g, z), line) in queries.iter().zip(rows) {
        let row: Vec<f64> = line
            .split_whitespace()
            .map(str::parse)
            .collect::<std::result::Result<_, _>>()?;
        ensure!(
            row.len() == 7
                && (row[0] - z).abs() < 0.02
                && row[2..6]
                    .iter()
                    .all(|x| x.is_finite() && (0.0..=1.0).contains(x)),
            "invalid IRI fractions"
        );
        ensure!(
            (row[2..6].iter().sum::<f64>() - 1.).abs() < 0.025,
            "IRI species closure"
        );
        ps[*p].iri[*g] = [row[2], row[3], row[4], row[5]];
    }
    let first = c
        .jobs
        .iter()
        .map(|j| j.start)
        .min_by(f64::total_cmp)
        .unwrap();
    let end = c.jobs.iter().map(|j| j.end).max_by(f64::total_cmp).unwrap();
    let nt = ((end - first) / 1800.).round() as usize;
    let midnight = ((first + 28800.) / 86400.).floor() * 86400. - 28800.;
    let x: Vec<_> = (0..=nt)
        .map(|i| (first + 1800. * i as f64 - midnight) / 3600.)
        .collect();
    let mut beams = BTreeMap::new();
    for p in &ps {
        beams.insert(
            (
                (p.az * 65536. / 360.).round() as u16,
                (p.el * 65536. / 360.).round() as u16,
            ),
            (p.az, p.el),
        );
    }
    let cube = hdf5::File::create(out.join("iri-comparison.h5"))?;
    cube.new_dataset::<f64>()
        .shape((queries.len(), 6))
        .deflate(4)
        .create("iri_query_points")?
        .write_raw(&query_points)?;
    let mut summary = vec![];
    for ((ar, er), (az, el)) in beams {
        let chosen: Vec<_> = ps
            .iter()
            .filter(|p| (p.az - az).abs() < 1e-3 && (p.el - el).abs() < 1e-3)
            .collect();
        let ng = chosen[0].iri.len();
        let edges: Vec<f64> = chosen[0].v["altitude_edges_km"]
            .as_array()
            .unwrap()
            .iter()
            .map(|z| z.as_f64().unwrap())
            .collect();
        let name = format!("beam-az{ar}-el{er}");
        let group = cube.create_group(&name)?;
        let mut data = vec![vec![f64::NAN; ng * nt]; 6];
        let mut species = vec![vec![f64::NAN; ng * nt]; 2];
        let mut raw = vec![vec![f64::NAN; ng * nt]; 2];
        let mut accepted = vec![0u8; nt];
        let mut diffs = [vec![], vec![]];
        for p in &chosen {
            let ti = ((p.start - first) / 1800.).round() as usize;
            if ti >= nt {
                continue;
            }
            accepted[ti] = p.valid as u8;
            for (k, g) in p.v["gates"].as_array().unwrap().iter().enumerate() {
                let idx = k * nt + ti;
                for q in 0..2 {
                    raw[q][idx] = g["node"][if q == 0 { "h_fraction" } else { "he_fraction" }]
                        .as_f64()
                        .unwrap();
                    data[2 + q][idx] = p.iri[k][q];
                    species[q][idx] = p.iri[k][2 + q];
                    if p.valid {
                        data[q][idx] = g["node"][if q == 0 { "h_fraction" } else { "he_fraction" }]
                            .as_f64()
                            .unwrap();
                        data[4 + q][idx] = data[q][idx] - data[2 + q][idx];
                        if data[4 + q][idx].is_finite() {
                            diffs[q].push(data[4 + q][idx]);
                        }
                    }
                }
            }
        }
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
        for (q, key) in [
            "fit_h_fraction",
            "fit_he_fraction",
            "iri_h_fraction",
            "iri_he_fraction",
            "fit_minus_iri_h",
            "fit_minus_iri_he",
        ]
        .iter()
        .enumerate()
        {
            group
                .new_dataset::<f64>()
                .shape((ng, nt))
                .deflate(4)
                .create(*key)?
                .write_raw(&data[q])?;
        }
        for (q, key) in ["raw_fit_h_fraction", "raw_fit_he_fraction"]
            .iter()
            .enumerate()
        {
            group
                .new_dataset::<f64>()
                .shape((ng, nt))
                .deflate(4)
                .create(*key)?
                .write_raw(&raw[q])?;
        }
        for (q, key) in ["iri_o_fraction", "iri_n_fraction"].iter().enumerate() {
            group
                .new_dataset::<f64>()
                .shape((ng, nt))
                .deflate(4)
                .create(*key)?
                .write_raw(&species[q])?;
        }
        let mut fig = Figure::new(IntendedUse::ArticleWide);
        fig.width_mm = 210.;
        fig.height_mm = 203.;
        fig.raster_dpi = 400.;
        fig.latex.cache = out.join("latex-cache");
        for row in 0..3 {
            for col in 0..2 {
                let q = 2 * row + col;
                let limit = if col == 0 { 1. } else { 0.5 };
                let scale = fig.scale(Scale::new(
                    if row == 2 {
                        [-limit, limit]
                    } else {
                        [0., limit]
                    },
                    Normalization::Linear,
                    if row == 2 {
                        ColorMap::diverging()
                    } else {
                        ColorMap::turbo()
                    },
                )?);
                let mut ax = Axes::new(
                    [x[0], *x.last().unwrap()],
                    [500., 1300.],
                    if row == 2 { "Local time (UTC+8)" } else { "" },
                    if col == 0 { "Altitude (km)" } else { "" },
                );
                ax.yticks = vec![500., 700., 1000., 1300.];
                ax.xticks = (0..=7)
                    .map(|k| 8. + 4. * k as f64)
                    .filter(|t| *t <= *x.last().unwrap())
                    .collect();
                ax.xticklabels = Some(
                    ax.xticks
                        .iter()
                        .map(|h| format!("{:02}:00", (*h as i32) % 24))
                        .collect(),
                );
                if row < 2 {
                    ax.show_x_axis = false;
                }
                let rect = Rect::new(20. + 98. * col as f64, 23. + 53. * row as f64, 66., 38.);
                let mut panel = Panel::new(rect, ax);
                panel.pcolormesh(x.clone(), edges.clone(), data[q].clone(), scale)?;
                fig.panels.push(panel);
                let mut cb = Colorbar::new(
                    Rect::new(rect.x + 69., rect.y, 2.5, rect.height),
                    scale,
                    "",
                    vec![],
                );
                cb.horizontal = false;
                fig.colorbars.push(cb);
                for (v, y) in [
                    (if row == 2 { -limit } else { 0. }, rect.y + rect.height),
                    (limit, rect.y),
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
                if row == 2 {
                    fig.labels
                        .push(Label::new("0", rect.x + 78., rect.y + rect.height / 2.));
                }
            }
        }
        fig.labels.push(Label::new("H$^+$ fraction", 53., 7.));
        fig.labels.push(Label::new("He$^+$ fraction", 151., 7.));
        for (row, label) in ["Fitted", "IRI TBT-2015", "Fitted minus IRI"]
            .iter()
            .enumerate()
        {
            fig.labels
                .push(Label::new(*label, 105., 18. + 53. * row as f64));
        }
        fig.labels.push(Label::new(
            format!("Azimuth {az:.0} degrees, elevation {el:.0} degrees; 6–7 October 2025"),
            105.,
            191.,
        ));
        fig.save(out.join(format!("{name}.png")))?;
        let mut bias = [None, None];
        let mut mae = [None, None];
        for q in 0..2 {
            let mut abs: Vec<_> = diffs[q].iter().map(|d| d.abs()).collect();
            mae[q] = median(&mut abs);
            bias[q] = median(&mut diffs[q]);
        }
        summary.push(json!({"beam":name,"accepted_profiles":accepted.iter().filter(|x|**x==1).count(),"profiles":chosen.len(),"median_fit_minus_iri_H_He":bias,"median_absolute_difference_H_He":mae}));
    }
    let meta = json!({"reference":"IRI2020 official2025-09-25 release; topside composition TBT2015; JF6=false JF22=true; fractions of total ion density, including IRI N+; no three-species renormalization","geometry":"Each gate center's nominal spherical beam location, each30-minute integration midpoint in UTC","fit_selection":"status starts converged and normalized gradient<0.05; nonconverged raw fit products preserved separately","query_count":queries.len(),"iri_query_point_columns":["midpoint_utc_s","profile_index","gate_index","latitude_deg","longitude_deg","altitude_km"],"profile_ids":ps.iter().map(|p|p.id.clone()).collect::<Vec<_>>(),"summary":summary});
    cube.new_dataset_builder()
        .with_data(&serde_json::to_vec(&meta)?)
        .create("metadata_json_utf8")?;
    fs::write(out.join("summary.json"), serde_json::to_vec_pretty(&meta)?)?;
    fs::remove_file(input)?;
    fs::remove_file(output)?;
    println!("{}", serde_json::to_string(&meta)?);
    Ok(())
}
