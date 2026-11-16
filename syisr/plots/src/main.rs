//! Native/adaptive gate-edge campaign maps; no interpolation or display rebinning.
use anyhow::{Result, ensure};
use chrono::{DateTime, Utc};
use rustplotlib::*;
use serde_json::json;
use std::{collections::BTreeMap, fs, path::Path};
fn main() -> Result<()> {
    let a: Vec<_> = std::env::args().collect();
    ensure!(a.len() == 3, "campaign-root plot-output");
    let root = Path::new(&a[1]);
    let out = Path::new(&a[2]);
    fs::create_dir_all(out)?;
    let c: syisr_ac::campaign::Campaign = syisr_ac::campaign::load(&root.join("campaign.h5"))?;
    let mut profiles = vec![];
    if root.join("fits").exists() {
        for e in fs::read_dir(root.join("fits"))? {
            let p = e?.path();
            if p.extension().is_some_and(|e| e == "h5") {
                if let Ok(v) = syisr_ac::campaign::load::<syisr_ac::fitting::Profile>(&p) {
                    if v.schema == "syisr-ac-native-adaptive-profile-v2" {
                        profiles.push(v);
                    }
                }
            }
        }
    }
    if profiles.is_empty() {
        fs::write(
            out.join("status.json"),
            serde_json::to_vec_pretty(
                &json!({"completed_profiles":0,"expected_profiles":c.jobs.len(),"state":"waiting for native/adaptive fits; no legacy data displayed"}),
            )?,
        )?;
        return Ok(());
    }
    let first = c
        .jobs
        .iter()
        .map(|j| j.start)
        .min_by(f64::total_cmp)
        .unwrap();
    let latest = profiles
        .iter()
        .map(|p| p.job.end)
        .max_by(f64::total_cmp)
        .unwrap();
    let nt = ((latest - first) / 1800.).round() as usize;
    let midnight = ((first + 28800.) / 86400.).floor() * 86400. - 28800.;
    let x: Vec<_> = (0..=nt)
        .map(|i| (first + 1800. * i as f64 - midnight) / 3600.)
        .collect();
    let mut beams = BTreeMap::new();
    for j in &c.jobs {
        let key = (
            (j.az * 65536. / 360.).round() as u16,
            (j.el * 65536. / 360.).round() as u16,
        );
        beams.insert(key, (j.az, j.el));
    }
    let cube = hdf5::File::create(out.join("progress.h5"))?;
    let mut metadata = vec![];
    for ((azraw, elraw), (az, el)) in beams {
        let grid = syisr_ac::grid::build(el, 1000.);
        let ng = grid.len();
        let mut edges: Vec<_> = grid.iter().map(|g| g.lower_altitude_km).collect();
        edges.push(grid.last().unwrap().upper_altitude_km);
        let mut data = vec![vec![f64::NAN; ng * nt]; 4];
        let mut quality = vec![1u8; ng * nt];
        let mut errors = vec![vec![f64::NAN; ng * nt]; 8];
        let mut complete = 0;
        let mut last_exposure = 0f64;
        for p in profiles
            .iter()
            .filter(|p| (p.job.az - az).abs() < 0.001 && (p.job.el - el).abs() < 0.001)
        {
            ensure!(
                p.gates.len() == ng && p.altitude_edges_km.len() == edges.len() && p.altitude_edges_km.iter().zip(&edges).all(|(a,b)| (a-b).abs() < 1e-8),
                "grid mismatch; refuse plotting false resolution"
            );
            let t = ((p.job.start - first) / 1800.).round() as usize;
            if t >= nt {
                continue;
            }
            complete += 1;
            for d in &p.source_dwells {
                last_exposure = last_exposure.max(c.dwells[*d].end);
            }
            for (k, g) in p.gates.iter().enumerate() {
                let idx = k * nt + t;
                if let Some(n) = &g.node {
                    data[0][idx] = g.ne_m3.unwrap_or(f64::NAN);
                    data[1][idx] = n.ti_k;
                    data[2][idx] = n.te_k;
                    data[3][idx] = n.velocity_m_s;
                    quality[idx] = u8::from(!g.flags.is_empty());
                    for f in 0..4 {
                        errors[f * 2][idx] = g.lower95[f].unwrap_or(f64::NAN);
                        errors[f * 2 + 1][idx] = g.upper95[f].unwrap_or(f64::NAN);
                    }
                }
            }
        }
        let name = format!("beam-az{azraw}-el{elraw}");
        let group = cube.create_group(&name)?;
        group
            .new_dataset_builder()
            .with_data(&edges)
            .create("altitude_edges_km")?;
        group
            .new_dataset_builder()
            .with_data(&x)
            .create("local_hours")?;
        group
            .new_dataset::<u8>()
            .shape((ng, nt))
            .create("quality_flag")?
            .write_raw(&quality)?;
        let mut fig = Figure::new(IntendedUse::ArticleWide);
        fig.width_mm = 190.;
        fig.height_mm = 270.;
        fig.latex.cache = out.join("latex-cache");
        fig.raster_dpi = 600.;
        let fields = [
            (
                "ne_m3",
                "Electron density (m$^{-3}$), empirical",
                [1e10, 1e12],
                Normalization::Log,
            ),
            (
                "ti_k",
                "Ion temperature (K)",
                [150., 3000.],
                Normalization::Linear,
            ),
            (
                "te_k",
                "Electron temperature (K)",
                [150., 6000.],
                Normalization::Linear,
            ),
            (
                "vi_m_s",
                "Spectral velocity (m/s)",
                [-1000., 1000.],
                Normalization::Linear,
            ),
        ];
        for (q, (key, label, lim, norm)) in fields.iter().enumerate() {
            group
                .new_dataset::<f64>()
                .shape((ng, nt))
                .deflate(4)
                .create(*key)?
                .write_raw(&data[q])?;
            for k in 0..2 {
                group
                    .new_dataset::<f64>()
                    .shape((ng, nt))
                    .deflate(4)
                    .create(format!("{key}_{}95", if k == 0 { "lower" } else { "upper" }).as_str())?
                    .write_raw(&errors[q * 2 + k])?;
            }
            let scale = fig.scale(Scale::new(*lim, *norm, ColorMap::turbo())?);
            let mut ax = Axes::new(
                [x[0], *x.last().unwrap()],
                [100., 1000.],
                if q == 3 {
                    "China local time (UTC+8)"
                } else {
                    ""
                },
                "Altitude (km)",
            );
            let interval = if x.last().unwrap() - x[0] < 2. {
                0.5
            } else {
                4.
            };
            let mut tick = (x[0] / interval).ceil() * interval;
            while tick <= *x.last().unwrap() + 1e-8 {
                ax.xticks.push(tick);
                tick += interval;
            }
            ax.xticklabels = Some(
                ax.xticks
                    .iter()
                    .map(|h| {
                        let ts = (midnight + h * 3600. + 28800.) as i64;
                        DateTime::<Utc>::from_timestamp(ts, 0)
                            .unwrap()
                            .format("%H:%M")
                            .to_string()
                    })
                    .collect(),
            );
            ax.yticks = vec![100., 300., 500., 700., 1000.];
            let rect = Rect::new(23., 22. + q as f64 * 58., 130., 36.);
            let mut panel = Panel::new(rect, ax);
            panel.pcolormesh(x.clone(), edges.clone(), data[q].clone(), scale)?;
            fig.panels.push(panel);
            let mut bar = Colorbar::new(
                Rect::new(25., rect.y + 40., 126., 2.),
                scale,
                *label,
                vec![lim[0], lim[1]],
            );
            bar.horizontal = true;
            fig.colorbars.push(bar);
        }
        fig.labels.push(Label::new(format!("SYISR: azimuth {az:.1} degrees, elevation {el:.1} degrees; {complete} completed windows"),95.,7.));
        fig.labels.push(Label::new("Native baud gates below 400 km; increasingly wider gates above; diagnostic fits retained",95.,262.));
        fig.save(out.join(format!("{name}.png")))?;
        metadata.push(json!({"beam":name,"azimuth_deg":az,"elevation_deg":el,"completed_profiles":complete,"latest_exposure_utc_s":last_exposure,"native_or_adaptive_gates":ng,"altitude_edges_km":edges,"image":format!("{name}.png")}));
    }
    fs::write(
        out.join("status.json"),
        serde_json::to_vec_pretty(
            &json!({"schema":"native-adaptive-progress-v2","completed_profiles":profiles.len(),"expected_profiles":c.jobs.len(),"latest_window_utc_s":latest,"first_window_utc_s":first,"beams":metadata,"colormap":"turbo","display":"actual gate edges; no interpolation or rebinning"}),
        )?,
    )?;
    Ok(())
}
