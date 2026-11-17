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
                    if v.schema == "syisr-ac-native-adaptive-profile-v2" || v.schema == "syisr-ac-joint-spline-profile-v3" {
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
    let campaign_end=c.jobs.iter().map(|j|j.end).max_by(f64::total_cmp).unwrap();
    let nt = ((campaign_end - first) / 1800.).round() as usize;
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
        if std::env::var("SYISR_PLOT_BEAM").ok().is_some_and(|wanted|wanted!=format!("beam-az{azraw}-el{elraw}")){continue;}
        let top=profiles.iter().filter(|p|(p.job.az-az).abs()<0.001 && (p.job.el-el).abs()<0.001).filter_map(|p|p.altitude_edges_km.last().copied()).fold(1000.,f64::max);
        let top=(top/100.).round()*100.;
        let grid = syisr_ac::grid::build(el, top);
        let ng = grid.len();
        let mut edges: Vec<_> = grid.iter().map(|g| g.lower_altitude_km).collect();
        edges.push(grid.last().unwrap().upper_altitude_km);
        let mut data = vec![vec![f64::NAN; ng * nt]; 5];
        let mut quality = vec![1u8; ng * nt];
        let mut errors = vec![vec![f64::NAN; ng * nt]; 10];
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
                    data[4][idx] = if g.altitude_km <= 500. {0.} else {n.h_fraction};
                    quality[idx] = u8::from(!g.flags.is_empty());
                    for f in 0..5 {
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
        fig.width_mm = 210.;
        fig.height_mm = 265.;
        fig.latex.cache = out.join("latex-cache");
        fig.raster_dpi = 600.;
        let fields = [
            (
                "ne_m3",
                r"$N_e$ (m$^{-3}$)",
                [1e10, 1e13],
                Normalization::Log,
            ),
            (
                "ti_k",
                r"$T_i$ (K)",
                [150., 3000.],
                Normalization::Linear,
            ),
            (
                "te_k",
                r"$T_e$ (K)",
                [150., 6000.],
                Normalization::Linear,
            ),
            (
                "vi_m_s",
                r"$v_i$ (m/s)",
                [-100., 100.],
                Normalization::Linear,
            ),
            ("h_fraction", r"H$^+$ fraction", [0.,1.], Normalization::Linear),
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
            let scale = fig.scale(Scale::new(*lim, *norm, if *key=="vi_m_s" {ColorMap::diverging()}else{ColorMap::turbo()})?);
            let mut ax = Axes::new(
                [x[0], *x.last().unwrap()],
                [100., top],
                if q == 4 {
                    "China local time (UTC+8)"
                } else {
                    ""
                },
                "Altitude (km)",
            );
            let interval = if x.last().unwrap() - x[0] < 2. {
                0.5
            } else if x.last().unwrap() - x[0] < 8. {
                1.
            } else {
                4.
            };
            ax.xticks.clear();
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
            if top>1100. {ax.yticks.push(1300.);}
            if q < 4 { ax.show_x_axis = false; }
            let rect = Rect::new(23., 20. + q as f64 * 44., 137., 36.);
            let mut panel = Panel::new(rect, ax);
            let mut shown=data[q].clone();
            let threshold=std::env::var("SYISR_MAX_95_HALF_WIDTH").ok().map(|s|s.split(',').map(|v|v.parse::<f64>().expect("five numeric uncertainty thresholds")).collect::<Vec<_>>());
            let mut mask=vec![0u8;shown.len()];
            if let Some(t)=threshold {
                ensure!(t.len()==5 && t.iter().all(|v|v.is_finite() && *v>0.),"five positive thresholds: relative Ne,Ti,Te; absolute vi(m/s),H fraction");
                for (i,v) in shown.iter_mut().enumerate() {
                    if q==4 && grid[i/nt].altitude_km<=500. {continue;}
                    let half=(errors[2*q+1][i]-errors[2*q][i])*0.5;
                    let width=if q<3 {half/v.abs()} else {half};
                    if !width.is_finite() || width<0. || width>t[q] {*v=f64::NAN;mask[i]=1;}
                }
            }
            group.new_dataset::<u8>().shape((ng,nt)).create(format!("{key}_uncertainty_mask").as_str())?.write_raw(&mask)?;
            panel.pcolormesh(x.clone(), edges.clone(), shown, scale)?;
            fig.panels.push(panel);
            let mut bar = Colorbar::new(
                Rect::new(164., rect.y, 2.5, 36.),
                scale,
                "",
                vec![],
            );
            bar.horizontal = false;
            fig.colorbars.push(bar);
            for (v,y) in [(lim[0],rect.y+rect.height),(lim[1],rect.y)] { let txt=if q==0 {format!("$10^{{{:.0}}}$",v.log10())}else{format!("{v:.0}")};fig.labels.push(Label::new(txt,178.,y)); }
            if *key=="vi_m_s" {fig.labels.push(Label::new("0",178.,rect.y+rect.height/2.));}
            let mut lab=Label::new(*label,193.,rect.y+rect.height/2.);lab.rotation=-90.;fig.labels.push(lab);
        }
        fig.labels.push(Label::new(format!("SYISR: azimuth {az:.1} degrees, elevation {el:.1} degrees; {complete} completed windows"),105.,7.));
        let joint=profiles.iter().all(|p|p.schema=="syisr-ac-joint-spline-profile-v3");
        fig.labels.push(Label::new(if joint && std::env::var_os("SYISR_NOMINAL_ONLY").is_some() {"Nominal joint spline profiles; joint bootstrap uncertainty refits pending"} else if joint {"Joint cubic-spline profiles; native source cells integrated through each gate response"}else{"Native baud gates below 400 km; increasingly wider gates above; diagnostic fits retained"},105.,254.));
        fig.save(out.join(format!("{name}.png")))?;
        metadata.push(json!({"beam":name,"azimuth_deg":az,"elevation_deg":el,"completed_profiles":complete,"latest_exposure_utc_s":last_exposure,"native_or_adaptive_gates":ng,"altitude_edges_km":edges,"image":format!("{name}.png")}));
    }
    fs::write(
        out.join("status.json"),
        serde_json::to_vec_pretty(
            &json!({"schema":"native-adaptive-progress-v2","completed_profiles":profiles.len(),"expected_profiles":c.jobs.len(),"latest_window_utc_s":latest,"first_window_utc_s":first,"beams":metadata,"colormaps":{"vi_m_s":"diverging","other_parameters":"turbo"},"display":"actual gate edges; no interpolation or rebinning"}),
        )?,
    )?;
    Ok(())
}
