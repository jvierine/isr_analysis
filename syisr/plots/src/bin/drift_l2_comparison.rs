//! Same-beam, same-window vector comparisons and distinct empirical references.
use anyhow::{Result, ensure};
use rustplotlib::*;
use serde_json::{Value, json};
use std::{
    collections::{BTreeMap, BTreeSet},
    path::Path,
};
fn median(a: &[f64]) -> f64 {
    let mut v = a.to_vec();
    v.sort_by(f64::total_cmp);
    v[v.len() / 2]
}
fn axes(x: &[f64], y: [f64; 2], bottom: bool, label: &str) -> Axes {
    let mut a = Axes::new(
        [x[0], *x.last().unwrap()],
        y,
        if bottom {
            "China local time (UTC+8)"
        } else {
            ""
        },
        label,
    );
    a.xticks = vec![8., 16., 24., 32.];
    a.xticklabels = Some(
        ["08:00", "16:00", "00:00", "08:00"]
            .iter()
            .map(|v| v.to_string())
            .collect(),
    );
    if !bottom {
        a.show_x_axis = false;
    }
    a
}
fn main() -> Result<()> {
    let a: Vec<_> = std::env::args().collect();
    ensure!(a.len() == 5, "AC L2 MODEL_DRIVERS OUTPUT");
    let ac: Value = syisr_ac::campaign::load(Path::new(&a[1]))?;
    let l2: Value = syisr_ac::campaign::load(Path::new(&a[2]))?;
    let driver: Value = syisr_ac::campaign::load(Path::new(&a[3]))?;
    let out = Path::new(&a[4]);
    std::fs::create_dir_all(out)?;
    let maps: Vec<BTreeMap<(i64, i64), &Value>> = [&ac, &l2]
        .iter()
        .map(|v| {
            v["rows"]
                .as_array()
                .unwrap()
                .iter()
                .map(|r| {
                    (
                        (
                            r["start_utc_s"].as_i64().unwrap(),
                            r["altitude_km"].as_f64().unwrap() as i64,
                        ),
                        r,
                    )
                })
                .collect()
        })
        .collect();
    let times: Vec<_> = maps[0]
        .keys()
        .map(|k| k.0)
        .collect::<BTreeSet<_>>()
        .into_iter()
        .collect();
    let heights: Vec<_> = maps[0]
        .keys()
        .map(|k| k.1)
        .collect::<BTreeSet<_>>()
        .into_iter()
        .collect();
    let (nt, nz) = (times.len(), heights.len());
    let mut x: Vec<_> = times
        .iter()
        .map(|t| (*t as f64 + 28800. - 1759708800.) / 3600.)
        .collect();
    x.push(x.last().unwrap() + 0.5);
    let mut z: Vec<_> = heights.iter().map(|v| *v as f64 - 5.).collect();
    z.push(*heights.last().unwrap() as f64 + 5.);
    let mut stats = vec![];
    for q in 0..3 {
        let pairs: Vec<_> = maps[0]
            .iter()
            .filter_map(|(k, r)| {
                let l = maps[1].get(k)?;
                if !(300..=500).contains(&k.1)
                    || r["magnetic"]["consistency_ok"] != true
                    || l["magnetic"]["consistency_ok"] != true
                {
                    return None;
                }
                Some([
                    r["magnetic"]["velocity_m_s"][q].as_f64()?,
                    l["magnetic"]["velocity_m_s"][q].as_f64()?,
                ])
            })
            .collect();
        let n = pairs.len() as f64;
        let means = [
            pairs.iter().map(|v| v[0]).sum::<f64>() / n,
            pairs.iter().map(|v| v[1]).sum::<f64>() / n,
        ];
        let corr = pairs
            .iter()
            .map(|v| (v[0] - means[0]) * (v[1] - means[1]))
            .sum::<f64>()
            / (pairs.iter().map(|v| (v[0] - means[0]).powi(2)).sum::<f64>()
                * pairs.iter().map(|v| (v[1] - means[1]).powi(2)).sum::<f64>())
            .sqrt();
        let d: Vec<_> = pairs.iter().map(|v| v[0] - v[1]).collect();
        stats.push(json!({"component":(["magnetic_east","perpendicular_meridional","parallel_B"][q]),"pairs":pairs.len(),"correlation":corr,"difference_median_m_s":median(&d),"difference_rms_m_s":(d.iter().map(|v|v*v).sum::<f64>()/n).sqrt()}));
    }
    syisr_ac::campaign::save(
        &out.join("comparison-statistics.h5"),
        &json!({"statistics":stats,"qualification":"300-500km cells accepted by both regional inversions; descriptivestatistics overcorrelated time/altitudesamples,notindependent confidenceestimates. LOSpositivetowardradar forbothACandL2."}),
    )?;
    for masked in [false, true] {
        let mut f = Figure::new(IntendedUse::ArticleWide);
        f.width_mm = 230.;
        f.height_mm = 190.;
        f.font_pt = 10.;
        f.raster_dpi = 240.;
        f.latex.cache = out.join("latex-cache");
        let scale = f.scale(Scale::new(
            [-100., 100.],
            Normalization::Linear,
            ColorMap::diverging(),
        )?);
        for q in 0..3 {
            for source in 0..2 {
                let mut values = vec![f64::NAN; nt * nz];
                for (k, r) in &maps[source] {
                    let Some(other) = maps[1 - source].get(k) else {
                        continue;
                    };
                    if masked
                        && (r["magnetic"]["consistency_ok"] != true
                            || other["magnetic"]["consistency_ok"] != true)
                    {
                        continue;
                    }
                    let t = times.binary_search(&k.0).unwrap();
                    let h = heights.binary_search(&k.1).unwrap();
                    values[h * nt + t] = r["magnetic"]["velocity_m_s"][q].as_f64().unwrap();
                }
                let yy = 20. + q as f64 * 48.;
                let mut ax = axes(
                    &x,
                    [250., 1000.],
                    q == 2,
                    if source == 0 { "Altitude (km)" } else { "" },
                );
                ax.yticks = vec![300., 500., 700., 1000.];
                let mut p = Panel::new(Rect::new(20. + source as f64 * 92., yy, 81., 37.), ax);
                p.pcolormesh(x.clone(), z.clone(), values, scale)?;
                f.panels.push(p);
            }
            let yy = 20. + q as f64 * 48.;
            f.colorbars.push(Colorbar::new(
                Rect::new(196., yy, 3., 37.),
                scale,
                "",
                vec![],
            ));
            for (v, y) in [(-100., yy + 37.), (0., yy + 18.5), (100., yy)] {
                f.labels.push(Label::new(format!("{v:.0}"), 208., y));
            }
            let mut l = Label::new(
                [
                    "Magnetic east (m/s)",
                    "Perpendicular meridional (m/s)",
                    "Along B (m/s)",
                ][q],
                224.,
                yy + 18.5,
            );
            l.rotation = -90.;
            f.labels.push(l);
        }
        f.labels.push(Label::new("Our AC fits", 60., 8.));
        f.labels.push(Label::new(
            "Supplied L2 LOS: reconstructed vectors",
            153.,
            8.,
        ));
        f.labels.push(Label::new(
            "6--7 October 2025; matched beams and 30-minute windows",
            115.,
            173.,
        ));
        f.labels.push(Label::new(
            if masked {
                "Only cells passing both beam-consistency checks; LOS positive toward radar"
            } else {
                "All matched estimates; LOS positive toward radar"
            },
            115.,
            182.,
        ));
        f.save(out.join(if masked {
            "ac-l2-accepted.png"
        } else {
            "ac-l2-all.png"
        }))?;
    }
    let mut refs = BTreeMap::new();
    for r in driver["rows"].as_array().unwrap() {
        refs.insert(
            (
                r["start_utc_s"].as_i64().unwrap(),
                r["altitude_km"].as_f64().unwrap() as i64,
            ),
            r,
        );
    }
    let mut f = Figure::new(IntendedUse::ArticleWide);
    f.width_mm = 205.;
    f.height_mm = 222.;
    f.font_pt = 10.;
    f.raster_dpi = 240.;
    f.latex.cache = out.join("latex-cache");
    let colors = ["#D55E00", "#009E73", "#0072B2"];
    let titles = [
        "Perpendicular meridional ion drift vs. Fejer--Scherliess",
        "Geographic east: ion velocities and HWM14 neutral wind",
        "Geographic north: ion velocities and HWM14 neutral wind",
        "Along B: ion velocities and projected HWM14 wind",
    ];
    for q in 0..4 {
        let yy = 18. + q as f64 * 44.;
        let mut p = Panel::new(
            Rect::new(22., yy, 170., 30.),
            axes(&x, [-150., 200.], q == 3, "Velocity (m/s)"),
        );
        for source in 0..3 {
            let mut pts = vec![];
            for time in &times {
                let mut vals = vec![];
                for z in (300..=500).step_by(10) {
                    let k = (*time, z);
                    let (Some(aa), Some(ll)) = (maps[0].get(&k), maps[1].get(&k)) else {
                        continue;
                    };
                    if aa["magnetic"]["consistency_ok"] != true
                        || ll["magnetic"]["consistency_ok"] != true
                    {
                        continue;
                    }
                    let value = if source < 2 {
                        let r = maps[source][&k];
                        if q == 0 {
                            r["magnetic"]["velocity_m_s"][1].as_f64()
                        } else if q == 3 {
                            r["magnetic"]["velocity_m_s"][2].as_f64()
                        } else {
                            r["geographic"]["velocity_m_s"][q - 1].as_f64()
                        }
                    } else {
                        refs.get(&k).and_then(|r| {
                            r[[
                                "fejer_scherliess_local_meridional_m_s",
                                "hwm_east_m_s",
                                "hwm_north_m_s",
                                "hwm_along_igrf_b_m_s",
                            ][q]]
                                .as_f64()
                        })
                    };
                    if let Some(v) = value {
                        vals.push(v);
                    }
                }
                pts.push([
                    (*time as f64 + 900. + 28800. - 1759708800.) / 3600.,
                    if vals.len() >= 5 {
                        median(&vals)
                    } else {
                        f64::NAN
                    },
                ]);
            }
            p.push(Layer::Line(Line::new(pts, colors[source])));
        }
        f.panels.push(p);
        f.labels.push(Label::new(titles[q], 106., yy - 4.));
    }
    for source in 0..3 {
        let mut l = Label::new(
            ["Our AC", "L2 reconstructed", "Empirical reference"][source],
            42. + source as f64 * 64.,
            202.,
        );
        l.color = colors[source].into();
        f.labels.push(l);
    }
    f.labels.push(Label::new(
        "Common accepted 300--500 km heights; ion/wind quantities differ",
        103.,
        210.,
    ));
    f.labels.push(Label::new(
        "6--7 October 2025; LOS positive toward radar",
        103.,
        217.,
    ));
    f.save(out.join("ac-l2-model-references.png"))?;
    f.save(out.join("ac-l2-model-references.pdf"))?;
    println!("{}", serde_json::to_string_pretty(&stats)?);
    Ok(())
}
