//! Publication-size figures for HPLA memo74. Paths and build provenance remain in metadata/comments.
use anyhow::{Result, ensure};
use rustplotlib::*;
use serde_json::Value;
use std::{
    collections::{BTreeMap, BTreeSet},
    path::Path,
};
fn median(a: &[f64]) -> f64 {
    let mut v = a.to_vec();
    v.sort_by(f64::total_cmp);
    v[v.len() / 2]
}
fn axes(x: &[f64], y: [f64; 2], last: bool, label: &str) -> Axes {
    let mut a = Axes::new(
        [x[0], *x.last().unwrap()],
        y,
        if last { "China local time (UTC+8)" } else { "" },
        label,
    );
    a.xticks = vec![8., 12., 16., 20., 24., 28., 32.];
    a.xticklabels = Some(
        ["08", "12", "16", "20", "00", "04", "08"]
            .iter()
            .map(|v| v.to_string())
            .collect(),
    );
    if !last {
        a.show_x_axis = false;
    }
    a
}
fn main() -> Result<()> {
    let a: Vec<_> = std::env::args().collect();
    ensure!(a.len() == 6, "AC L2 DRIVERS RESPONSE OUTPUT");
    let ac: Value = syisr_ac::campaign::load(Path::new(&a[1]))?;
    let l2: Value = syisr_ac::campaign::load(Path::new(&a[2]))?;
    let driver: Value = syisr_ac::campaign::load(Path::new(&a[3]))?;
    let out = Path::new(&a[5]);
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
    let mut x: Vec<_> = times
        .iter()
        .map(|t| (*t as f64 + 28800. - 1759708800.) / 3600.)
        .collect();
    x.push(x.last().unwrap() + 0.5);
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
    let colors = ["#D55E00", "#009E73", "#0072B2"];
    let mut f = ArticleLayout::new(179.324, 10.95)?.figure(1., 207.)?;
    f.raster_dpi = 400.;
    f.latex.cache = out.join("latex-cache");
    let titles = [
        "Perpendicular meridional ion drift and Fejer--Scherliess",
        "Geographic east: ion velocities and HWM14 wind",
        "Geographic north: ion velocities and HWM14 wind",
        "Along B: ion velocities and projected HWM14 wind",
    ];
    for q in 0..4 {
        let yy = 16. + q as f64 * 41.;
        let mut p = Panel::new(
            Rect::new(19., yy, 153., 27.),
            axes(&x, [-100., 175.], q == 3, "Velocity (m/s)"),
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
                        match q {
                            0 => r["magnetic"]["velocity_m_s"][1].as_f64(),
                            3 => r["magnetic"]["velocity_m_s"][2].as_f64(),
                            _ => r["geographic"]["velocity_m_s"][q - 1].as_f64(),
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
        f.labels.push(Label::new(titles[q], 95., yy - 4.));
    }
    for source in 0..3 {
        let mut l = Label::new(
            ["Our AC", "L2 reconstruction", "Empirical reference"][source],
            30. + source as f64 * 57.,
            192.,
        );
        l.color = colors[source].into();
        f.labels.push(l);
    }
    f.labels.push(Label::new(
        "6--7 October 2025; common accepted heights, 300--500 km",
        90.,
        204.,
    ));
    f.save(out.join("drivers.png"))?;
    f.save_for_article(out.join("drivers.pdf"), "figures/memo074/drivers.pdf")?;
    let h = hdf5::File::open(&a[4])?;
    let mut f = ArticleLayout::new(179.324, 10.95)?.figure(1., 91.)?;
    f.raster_dpi = 400.;
    f.latex.cache = out.join("latex-cache");
    let mut p = Panel::new(
        Rect::new(22., 18., 150., 49.),
        axes(&x, [500., 1300.], true, "H$^+$ 50\\% altitude (km)"),
    );
    for (q, (name, color)) in [
        ("fit", "#111111"),
        ("baseline", "#0072B2"),
        ("measured", "#D55E00"),
    ]
    .iter()
    .enumerate()
    {
        let v = h.dataset(&format!("{name}_h50_km"))?.read_raw::<f64>()?;
        let pts = v
            .iter()
            .enumerate()
            .map(|(t, v)| [(x[t] + x[t + 1]) / 2., *v])
            .collect();
        p.push(Layer::Line(Line::new(pts, *color)));
        let mut l = Label::new(
            ["AC fit", "Fejer--Scherliess", "Measured forcing"][q],
            30. + q as f64 * 57.,
            7.,
        );
        l.color = (*color).into();
        f.labels.push(l);
    }
    f.panels.push(p);
    f.labels.push(Label::new(
        "6--7 October 2025; crossings outside 500--1300 km omitted",
        90.,
        88.,
    ));
    f.save(out.join("h50.png"))?;
    f.save_for_article(out.join("h50.pdf"), "figures/memo074/h50.pdf")?;
    Ok(())
}
