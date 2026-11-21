//! Multibeam EPB views; dated geometry and data are HDF5 inputs.
use anyhow::{Result, ensure};
use rustplotlib::*;
use std::path::Path;
fn v(h: &hdf5::Group, k: &str) -> Result<Vec<f64>> {
    Ok(h.dataset(k)?.read_raw::<f64>()?)
}
fn dot(a: &[f64], b: &[f64]) -> f64 {
    a.iter().zip(b).map(|(a, b)| a * b).sum()
}
fn color(i: usize) -> String {
    let c = ColorMap::storm().sample(0.2 + 0.65 * i as f64 / 10.);
    format!("#{:02x}{:02x}{:02x}", c[0], c[1], c[2])
}
fn line(p: &mut Panel, points: Vec<[f64; 2]>, c: &str, width: f64, opacity: f64) {
    let mut l = Line::new(points, c);
    l.width_pt = width;
    l.opacity = opacity;
    p.push(Layer::Line(l));
}
fn hull(mut pts: Vec<[f64; 2]>) -> Vec<[f64; 2]> {
    pts.sort_by(|a, b| a[0].total_cmp(&b[0]).then(a[1].total_cmp(&b[1])));
    pts.dedup();
    let cross = |a: [f64; 2], b: [f64; 2], c: [f64; 2]| {
        (b[0] - a[0]) * (c[1] - a[1]) - (b[1] - a[1]) * (c[0] - a[0])
    };
    let mut lower = Vec::new();
    for &p in &pts {
        while lower.len() >= 2 && cross(lower[lower.len() - 2], lower[lower.len() - 1], p) <= 0. {
            lower.pop();
        }
        lower.push(p);
    }
    let mut upper = Vec::new();
    for &p in pts.iter().rev() {
        while upper.len() >= 2 && cross(upper[upper.len() - 2], upper[upper.len() - 1], p) <= 0. {
            upper.pop();
        }
        upper.push(p);
    }
    lower.pop();
    upper.pop();
    lower.extend(upper);
    if let Some(p) = lower.first().copied() {
        lower.push(p);
    }
    lower
}
fn geom(root: &Path, out: &Path) -> Result<()> {
    let h = hdf5::File::open(root.join("geometry.h5"))?;
    let r = v(&h, "reference_ecef_km")?;
    let basis = v(&h, "basis_e1_e2_B_ecef")?;
    let site = v(&h, "site_lat_lon_height")?;
    let station = syisr_ac::lfm_geometry::ecef(site[0], site[1], site[2]);
    let mut names: Vec<_> = h
        .member_names()?
        .into_iter()
        .filter(|k| k.starts_with("beam_"))
        .collect();
    names.sort();
    let tube = v(&h.group("hypothetical")?, "field_aligned_surface_ecef_km")?;
    let project = |p: &[f64], mode: usize| -> [f64; 2] {
        let d: Vec<f64> = p.iter().zip(&r).map(|(a, b)| a - b).collect();
        if mode == 0 || mode == 4 {
            [dot(&d, &basis[..3]), dot(&d, &basis[3..6])]
        } else if mode == 2 {
            [dot(&d, &basis[..3]), dot(&d, &basis[6..9])]
        } else {
            let pos = syisr_ac::lfm_geometry::enu(nal(p));
            if mode == 1 {
                [pos.x, pos.y]
            } else {
                [pos.x - 0.5 * pos.y, pos.z + 0.25 * pos.y]
            }
        }
    };
    for (mode, name, title, xlab, ylab, xlim, ylim) in [
        (
            0,
            "geometry_along_B",
            "View along B",
            "Cross-field east (km)",
            "Cross-field upward (km)",
            [-600., 600.],
            [-450., 650.],
        ),
        (
            1,
            "geometry_top",
            "View from above",
            "East of radar (km)",
            "North of radar (km)",
            [-600., 600.],
            [-700., 700.],
        ),
        (
            2,
            "geometry_perp_B_above",
            "View perpendicular to B from above",
            "Cross-field east (km)",
            "Distance along B (km)",
            [-600., 600.],
            [-900., 900.],
        ),
        (
            3,
            "geometry_oblique",
            "Oblique view",
            "East - 0.5 North (km)",
            "Up + 0.25 North (km)",
            [-700., 700.],
            [-150., 1200.],
        ),
        (
            4,
            "geometry_mapped",
            "Field-line mapping to the reference plane",
            "Mapped cross-field east (km)",
            "Mapped cross-field upward (km)",
            [-600., 600.],
            [-450., 650.],
        ),
    ] {
        let mut f = Figure::new(IntendedUse::ArticleWide);
        f.width_mm = 175.;
        f.height_mm = 162.;
        f.raster_dpi = 400.;
        f.latex.cache = out.join("latex-cache");
        let mut ax = Axes::new(xlim, ylim, xlab, ylab);
        ax.xticks = vec![-300., 0., 300.];
        ax.yticks = if mode == 1 {
            vec![-400., 0., 400.]
        } else if mode == 2 {
            vec![-600., 0., 600.]
        } else if mode == 3 {
            vec![0., 400., 800.]
        } else {
            vec![-300., 0., 300.]
        };
        let mut p = Panel::new(Rect::new(24., 14., 132., 125.), ax);
        for (i, beam) in names.iter().enumerate() {
            let g = h.group(beam)?;
            let xyz = v(&g, "ecef_km")?;
            let geo = v(&g, "geodetic_lat_lon_alt")?;
            let mapped = v(&g, "field_line_reference_plane_km")?;
            let pts: Vec<_> = xyz
                .chunks_exact(3)
                .enumerate()
                .filter(|(j, _)| geo[3 * j + 2] >= 150. && geo[3 * j + 2] <= 900.)
                .map(|(j, q)| {
                    if mode == 4 {
                        [mapped[3 * j], mapped[3 * j + 1]]
                    } else {
                        project(q, mode)
                    }
                })
                .collect();
            if mode != 4 && g.link_exists("oneway_half_power_edges_ecef_km") {
                let edges = v(&g, "oneway_half_power_edges_ecef_km")?;
                let mut cloud = Vec::new();
                for j in 0..geo.len() / 3 {
                    if geo[3 * j + 2] < 150. || geo[3 * j + 2] > 900. {
                        continue;
                    }
                    for a in 0..17 {
                        cloud.push(project(
                            &edges[(j * 17 + a) * 3..(j * 17 + a + 1) * 3],
                            mode,
                        ));
                    }
                }
                let mut envelope = Line::new(hull(cloud), color(i));
                envelope.fill = Some(color(i));
                envelope.fill_opacity = 0.12;
                envelope.width_pt = 0.35;
                p.push(Layer::Line(envelope));
            }
            line(&mut p, pts.clone(), &color(i), 1.2, 1.);
            if g.link_exists("aspect_deviation_from_perpendicular_deg") {
                let aspect = v(&g, "aspect_deviation_from_perpendicular_deg")?;
                let risky: Vec<_> = xyz
                    .chunks_exact(3)
                    .enumerate()
                    .filter(|(j, _)| geo[3 * j + 2] >= 150. && geo[3 * j + 2] <= 900.)
                    .map(|(j, q)| {
                        if aspect[j].abs() < 2. {
                            if mode == 4 {
                                [mapped[3 * j], mapped[3 * j + 1]]
                            } else {
                                project(q, mode)
                            }
                        } else {
                            [f64::NAN; 2]
                        }
                    })
                    .collect();
                let mut l = Line::new(risky, "#111111");
                l.width_pt = 1.8;
                l.dash = Some("2 1".into());
                p.push(Layer::Line(l));
            }

            if !pts.is_empty() {
                let index = pts.len() * 3 / 4;
                let point = pts[index];
                if mode == 1 {
                    // Separate near-meridional beams with explicit leader labels.
                    let target = match i + 1 {
                        9 => [95., 570.],
                        10 => [95., 300.],
                        11 => [95., 65.],
                        5 => [95., -265.],
                        4 => [95., -560.],
                        _ => [
                            point[0] + if point[0] < 0. { -35. } else { 35. },
                            point[1] + 15.,
                        ],
                    };
                    line(&mut p, vec![point, target], "#555555", 0.4, 0.8);
                    p.label_at(format!("{}", i + 1), target);
                } else {
                    p.label_at(format!("{}", i + 1), [point[0] + 8., point[1] + 8.]);
                }
            }
        }
        // Illustrative field-aligned wire surface. Never a measured inversion.
        if mode == 0 || mode == 4 {
            let pts = (0..=128)
                .map(|j| {
                    let a = j as f64 * 2. * std::f64::consts::PI / 128.;
                    [15. * a.cos(), 35. * a.sin()]
                })
                .collect();
            line(&mut p, pts, "#111111", 1.8, 1.);
        } else if mode == 1 {
            // Trace a central IGRF field line through the 400 km reference point.
            let mut traced = Vec::new();
            for sign in [-1., 1.] {
                let mut q = nal(&r);
                for k in 0..=100 {
                    if k > 0 || sign < 0. {
                        traced.push((sign * k as f64 * 5., project(q.as_slice(), mode)));
                    }
                    let step = sign * 5.;
                    let k1 = syisr_ac::lfm_geometry::bunit(q)?;
                    let k2 = syisr_ac::lfm_geometry::bunit(q + k1 * step / 2.)?;
                    let k3 = syisr_ac::lfm_geometry::bunit(q + k2 * step / 2.)?;
                    let k4 = syisr_ac::lfm_geometry::bunit(q + k3 * step)?;
                    q += (k1 + 2. * k2 + 2. * k3 + k4) * step / 6.;
                }
            }
            traced.sort_by(|a, b| a.0.total_cmp(&b.0));
            let points: Vec<[f64; 2]> = traced
                .into_iter()
                .map(|(_, q)| [q[0] - 520., q[1]])
                .collect();
            let mid = points.len() / 2;
            let tip = points[mid];
            let d = [
                points[mid + 1][0] - points[mid - 1][0],
                points[mid + 1][1] - points[mid - 1][1],
            ];
            let norm = d[0].hypot(d[1]);
            let d = [d[0] / norm, d[1] / norm];
            line(&mut p, points, "#111111", 1.6, 1.);
            line(
                &mut p,
                vec![
                    [
                        tip[0] - d[0] * 45. - d[1] * 15.,
                        tip[1] - d[1] * 45. + d[0] * 15.,
                    ],
                    tip,
                    [
                        tip[0] - d[0] * 45. + d[1] * 15.,
                        tip[1] - d[1] * 45. - d[0] * 15.,
                    ],
                ],
                "#111111",
                1.6,
                1.,
            );
            p.label_at("$\\vec{B}$", [-450., 85.]);
        } else {
            for j in (0..17).step_by(2) {
                let points = tube[j * 201 * 3..(j + 1) * 201 * 3]
                    .chunks_exact(3)
                    .map(|q| project(q, mode))
                    .collect();
                line(&mut p, points, "#222222", 0.7, 0.65);
            }
            for k in [0, 50, 100, 150, 200] {
                let pts = (0..17)
                    .map(|j| project(&tube[(j * 201 + k) * 3..(j * 201 + k + 1) * 3], mode))
                    .collect();
                line(&mut p, pts, "#111111", 0.7, 0.8);
            }
        }
        let radar = project(station.as_slice(), mode);
        if mode != 1
            && radar[0] >= xlim[0]
            && radar[0] <= xlim[1]
            && radar[1] >= ylim[0]
            && radar[1] <= ylim[1]
        {
            p.label_at("Radar", [radar[0] - 50., radar[1] - 45.]);
        }
        f.panels.push(p);
        f.labels.push(Label::new(title, 88., 7.));
        if std::env::var_os("SYISR_MEMO_FIGURES").is_some() {
            f.labels.clear();
        }
        f.save(out.join(format!("{name}.png")))?;
        if std::env::var_os("SYISR_MEMO_FIGURES").is_some() {
            f.labels.clear();
        }
        f.save(out.join(format!("{name}.pdf")))?;
    }
    Ok(())
}
fn nal(p: &[f64]) -> nalgebra::Vector3<f64> {
    nalgebra::Vector3::new(p[0], p[1], p[2])
}
fn main() -> Result<()> {
    let a: Vec<_> = std::env::args().collect();
    ensure!(a.len() >= 3, "ROOT OUTPUT [diagnostics-only]");
    let root = Path::new(&a[1]);
    let out = Path::new(&a[2]);
    std::fs::create_dir_all(out)?;
    if a.len() == 3 {
        geom(root, out)?;
        beam_views(root, out, false)?;
    } else if a[3] == "geometry-only" {
        geom(root, out)?;
        return Ok(());
    } else if a[3] == "atlas" {
        geom(root, out)?;
        beam_views(root, out, true)?;
        return Ok(());
    } else if a[3] == "memo" {
        geom(root, out)?;
    }
    cavity_views(root, out)?;
    aspect_view(root, out)?;
    drift_view(root, out)
}
struct BeamMap {
    id: String,
    az: f64,
    el: f64,
    x: Vec<f64>,
    z: Vec<f64>,
    values: Vec<f64>,
    nt: usize,
    nz: usize,
}
fn beam_maps(root: &Path) -> Result<Vec<BeamMap>> {
    let mut paths: Vec<_> = std::fs::read_dir(root.join("beams"))?
        .filter_map(|e| e.ok().map(|e| e.path()))
        .filter(|p| p.is_dir())
        .collect();
    paths.sort();
    let mut maps = Vec::new();
    for path in paths {
        let h = hdf5::File::open(path.join("display.h5"))?;
        let az = h.attr("beam_azimuth_deg")?.read_scalar()?;
        let el = h.attr("beam_elevation_deg")?.read_scalar()?;
        let te = v(&h, "time_edges_unix_s")?;
        let ze = v(&h, "altitude_edges_km")?;
        let r = v(&h, "range_km")?;
        let sig = v(&h, "median_signal")?;
        let day = (te[0] / 86400.).floor() * 86400.;
        let t0 = te.iter().position(|t| (*t - day) / 3600. >= 11.75).unwrap();
        let t1 = te
            .iter()
            .rposition(|t| (*t - day) / 3600. <= 13.75)
            .unwrap();
        let k0 = ze.iter().position(|z| *z >= 150.).unwrap();
        let k1 = ze.iter().rposition(|z| *z <= 900.).unwrap();
        let nt = t1 - t0;
        let nz = k1 - k0;
        let x = te[t0..=t1].iter().map(|t| (t - day) / 3600.).collect();
        let z = ze[k0..=k1].to_vec();
        let mut values = vec![0.; nt * nz];
        for k in 0..nz {
            for j in 0..nt {
                let p = sig[(j + t0) * r.len() + k + k0] * r[k + k0].powi(2);
                values[k * nt + j] = if p.is_finite() {
                    if p > 0. { 10. * p.log10() } else { 78. }
                } else {
                    f64::NAN
                };
            }
        }
        maps.push(BeamMap {
            id: path.file_name().unwrap().to_string_lossy().to_string(),
            az,
            el,
            x,
            z,
            values,
            nt,
            nz,
        });
    }
    Ok(maps)
}
fn beam_panel(
    m: &BeamMap,
    rect: Rect,
    scale: usize,
    showx: bool,
    showy: bool,
    dpi: f64,
) -> Result<Panel> {
    ensure!(
        (rect.width * dpi / 25.4).floor() >= m.nt as f64
            && (rect.height * dpi / 25.4).floor() >= m.nz as f64,
        "Beam raster undersamples {}",
        m.id
    );
    let mut ax = Axes::new(
        [11.75, 13.75],
        [150., 900.],
        if showx {
            "UTC (Beijing = UTC + 8 h)"
        } else {
            ""
        },
        if showy { "Nominal altitude (km)" } else { "" },
    );
    ax.xticks = vec![12., 12.5, 13., 13.5];
    ax.xticklabels = Some(vec![
        "12:00".into(),
        "12:30".into(),
        "13:00".into(),
        "13:30".into(),
    ]);
    ax.yticks = vec![200., 400., 600., 800.];
    if !showx {
        ax.show_x_axis = false;
    }
    if !showy {
        ax.show_y_axis = false;
    }
    let mut p = Panel::new(rect, ax);
    p.pcolormesh(m.x.clone(), m.z.clone(), m.values.clone(), scale)?;
    Ok(p)
}
fn beam_views(root: &Path, out: &Path, atlas_only: bool) -> Result<()> {
    let maps = beam_maps(root)?;
    let sc = Scale::new([78., 105.], Normalization::Linear, ColorMap::storm())?;
    if !atlas_only {
        for (i, m) in maps.iter().enumerate() {
            let mut f = Figure::new(IntendedUse::ArticleWide);
            f.width_mm = 155.;
            f.height_mm = 163.;
            f.raster_dpi = 600.;
            f.latex.cache = out.join("latex-cache");
            let scale = f.scale(sc.clone());
            f.panels.push(beam_panel(
                m,
                Rect::new(23., 14., 105., 125.),
                scale,
                true,
                true,
                600.,
            )?);
            f.labels.push(Label::new(
                format!(
                    "Beam {}: az {:.0}$^\\circ$, el {:.0}$^\\circ$",
                    i + 1,
                    m.az,
                    m.el
                ),
                76.,
                7.,
            ));
            f.colorbars.push(Colorbar::new(
                Rect::new(133., 14., 3., 125.),
                scale,
                "dB (arbitrary)",
                vec![78., 85., 95., 105.],
            ));
            f.save(out.join(format!("{}_rti.png", m.id)))?;
            f.save(out.join(format!("{}_rti.pdf", m.id)))?;
        }
    }
    let mut f = Figure::new(IntendedUse::ArticleWide);
    f.width_mm = 355.;
    f.height_mm = 555.;
    f.font_pt = 9.;
    f.raster_dpi = 600.;
    f.latex.cache = out.join("latex-cache");
    let scale = f.scale(sc);
    for (i, m) in maps.iter().enumerate() {
        // Geographic placement matches the overhead diagram: north up/east right.
        let (row, col) = match i + 1 {
            8 => (0, 0),
            9 => (0, 1),
            1 => (0, 2),
            10 => (1, 1),
            7 => (2, 0),
            11 => (2, 1),
            2 => (2, 2),
            5 => (3, 1),
            6 => (4, 0),
            4 => (4, 1),
            3 => (4, 2),
            _ => unreachable!(),
        };
        let rect = Rect::new(23. + col as f64 * 100., 14. + row as f64 * 105., 100., 105.);
        let mut panel = beam_panel(m, rect, scale, row == 4, col == 0, 600.)?;
        panel.axes.as_mut().unwrap().xlabel = String::new();
        panel.axes.as_mut().unwrap().ylabel = String::new();
        panel
            .label_at(
                format!("{}: ${:.0}^\\circ$, ${:.0}^\\circ$", i + 1, m.az, m.el),
                [12.75, 865.],
            )
            .color = "#ffffff".into();
        f.panels.push(panel);
    }
    f.colorbars.push(Colorbar::new(
        Rect::new(329., 14., 3., 525.),
        scale,
        "dB (arbitrary)",
        vec![78., 85., 95., 105.],
    ));
    f.labels
        .push(Label::new("UTC (Beijing = UTC + 8 h)", 173., 551.));
    let mut label = Label::new("Nominal altitude (km)", 5., 275.);
    label.rotation = -90.;
    f.labels.push(label);
    f.save(out.join("all_beams.png"))?;
    f.save(out.join("all_beams.pdf"))?;
    let mut f = Figure::new(IntendedUse::ArticleWide);
    f.width_mm = 155.;
    f.height_mm = 450.;
    f.raster_dpi = 600.;
    f.latex.cache = out.join("latex-cache");
    let scale = f.scale(Scale::new(
        [78., 105.],
        Normalization::Linear,
        ColorMap::storm(),
    )?);
    for (row, id) in [10, 11, 5, 4].into_iter().enumerate() {
        let m = &maps[id - 1];
        let mut p = beam_panel(
            m,
            Rect::new(23., 10. + row as f64 * 105., 100., 105.),
            scale,
            row == 3,
            true,
            600.,
        )?;
        p.axes.as_mut().unwrap().xlabel = String::new();
        p.axes.as_mut().unwrap().ylabel = String::new();
        p.label_at(
            format!("{}: ${:.0}^\\circ$, ${:.0}^\\circ$", id, m.az, m.el),
            [12.75, 865.],
        )
        .color = "#ffffff".into();
        f.panels.push(p);
    }
    f.colorbars.push(Colorbar::new(
        Rect::new(129., 10., 3., 420.),
        scale,
        "dB (arbitrary)",
        vec![78., 85., 95., 105.],
    ));
    f.labels
        .push(Label::new("UTC (Beijing = UTC + 8 h)", 73., 445.));
    let mut label = Label::new("Nominal altitude (km)", 5., 220.);
    label.rotation = -90.;
    f.labels.push(label);
    f.save(out.join("meridional_rtis.pdf"))?;
    f.save(out.join("meridional_rtis.png"))?;
    Ok(())
}
fn cavity_views(root: &Path, out: &Path) -> Result<()> {
    let path = root.join("cavity/all_beams.h5");
    if !path.exists() {
        return Ok(());
    }
    let h = hdf5::File::open(path)?;
    let x = v(&h, "x_km")?;
    let y = v(&h, "y_km")?;
    let d = v(&h, "contrast")?;
    let support = v(&h, "support_weight")?;
    let maxsupport = support.iter().copied().fold(0., f64::max);
    let supported: Vec<usize> = support
        .iter()
        .enumerate()
        .filter(|(_, w)| **w >= 0.005 * maxsupport)
        .map(|(i, _)| i)
        .collect();
    ensure!(!supported.is_empty(), "no supported reconstruction nodes");
    let xlim = [
        supported
            .iter()
            .map(|k| x[k % x.len()])
            .fold(f64::INFINITY, f64::min)
            - 30.,
        supported
            .iter()
            .map(|k| x[k % x.len()])
            .fold(f64::NEG_INFINITY, f64::max)
            + 30.,
    ];
    let ylim = [
        supported
            .iter()
            .map(|k| y[k / x.len()])
            .fold(f64::INFINITY, f64::min)
            - 30.,
        supported
            .iter()
            .map(|k| y[k / x.len()])
            .fold(f64::NEG_INFINITY, f64::max)
            + 30.,
    ];

    let dx = x[1] - x[0];
    let dy = y[1] - y[0];
    let mut xe: Vec<_> = x.iter().map(|v| v - dx / 2.).collect();
    xe.push(x.last().unwrap() + dx / 2.);
    let mut ye: Vec<_> = y.iter().map(|v| v - dy / 2.).collect();
    ye.push(y.last().unwrap() + dy / 2.);
    for (name, kind) in [("cavity_cross_section", 0), ("cavity_support", 1)] {
        let mut f = Figure::new(IntendedUse::ArticleWide);
        f.width_mm = 178.;
        f.height_mm = 151.;
        f.raster_dpi = 400.;
        f.latex.cache = out.join("latex-cache");
        let values: Vec<f64> = d
            .iter()
            .enumerate()
            .map(|(k, v)| {
                if support[k] >= 0.005 * maxsupport {
                    if kind == 0 {
                        1. + v
                    } else {
                        support[k].log10()
                    }
                } else {
                    f64::NAN
                }
            })
            .collect();
        let scale = f.scale(Scale::new(
            if kind == 0 { [0., 1.5] } else { [0., 4.] },
            Normalization::Linear,
            ColorMap::storm(),
        )?);
        let mut ax = Axes::new(
            xlim,
            ylim,
            "Mapped cross-field east (km)",
            "Mapped cross-field upward (km)",
        );
        ax.xticks = ((xlim[0] / 250.).ceil() as i32..=(xlim[1] / 250.).floor() as i32)
            .map(|i| i as f64 * 250.)
            .collect();
        ax.yticks = ((ylim[0] / 200.).ceil() as i32..=(ylim[1] / 200.).floor() as i32)
            .map(|i| i as f64 * 200.)
            .collect();
        let mut p = Panel::new(Rect::new(25., 14., 125., 111.), ax);
        p.pcolormesh(xe.clone(), ye.clone(), values, scale)?;
        f.panels.push(p);
        f.labels.push(Label::new(
            if kind == 0 {
                "Field-aligned reconstruction at 12:45 UTC"
            } else {
                "Measurement support in the moving frame"
            },
            89.,
            7.,
        ));
        f.colorbars.push(Colorbar::new(
            Rect::new(156., 14., 3., 111.),
            scale,
            if kind == 0 {
                "Relative density proxy"
            } else {
                "$\\log_{10}$ weight"
            },
            if kind == 0 {
                vec![0., 0.5, 1., 1.5]
            } else {
                vec![0., 2., 4.]
            },
        ));
        if std::env::var_os("SYISR_MEMO_FIGURES").is_some() {
            f.labels.clear();
        }
        f.save(out.join(format!("{name}.png")))?;
        if std::env::var_os("SYISR_MEMO_FIGURES").is_some() {
            f.labels.clear();
        }
        f.save(out.join(format!("{name}.pdf")))?;
    }
    let val = hdf5::File::open(root.join("cavity/validation.h5"))?;
    let metrics = v(&val, "whole_beam_holdout_diagnostics")?;
    let mut f = Figure::new(IntendedUse::ArticleWide);
    f.width_mm = 178.;
    f.height_mm = 95.;
    f.raster_dpi = 400.;
    f.latex.cache = out.join("latex-cache");
    let mut ax = Axes::new(
        [0.5, 11.5],
        [0., 1.2],
        "Withheld beam",
        "Prediction RMSE (relative units)",
    );
    ax.xticks = (1..=11).map(|i| i as f64).collect();
    ax.yticks = vec![0., 0.4, 0.8, 1.2];
    let mut p = Panel::new(Rect::new(22., 19., 146., 53.), ax);
    line(
        &mut p,
        metrics
            .chunks_exact(7)
            .enumerate()
            .map(|(i, m)| [(i + 1) as f64, (m[3] / m[4]).sqrt()])
            .collect(),
        "#111111",
        1.5,
        1.,
    );
    let mut base = Line::new(
        metrics
            .chunks_exact(7)
            .enumerate()
            .map(|(i, m)| [(i + 1) as f64, (m[5] / m[4]).sqrt()])
            .collect(),
        "#888888",
    );
    base.dash = Some("2 1".into());
    p.push(Layer::Line(base));
    f.panels.push(p);
    f.labels.push(Label::new(
        "Solid: field-aligned fit; dashed: quiet reference",
        89.,
        8.,
    ));
    if std::env::var_os("SYISR_MEMO_FIGURES").is_some() {
        f.labels.clear();
    }
    f.save(out.join("cavity_holdout.png"))?;
    if std::env::var_os("SYISR_MEMO_FIGURES").is_some() {
        f.labels.clear();
    }
    f.save(out.join("cavity_holdout.pdf"))?;
    let volume = root.join("cavity/volume.h5");
    if volume.exists() {
        let vol = hdf5::File::open(volume)?;
        let pos = v(&vol, "enu_km")?;
        let labels = v(&vol, "field_line_labels_km")?;
        let ratios = v(&vol, "relative_density_proxy")?;
        let s = v(&vol, "distance_along_B_km")?;
        for (name, oblique) in [("cavity_3d_top", false), ("cavity_3d_oblique", true)] {
            let mut f = Figure::new(IntendedUse::ArticleWide);
            f.width_mm = 178.;
            f.height_mm = 160.;
            f.raster_dpi = 400.;
            f.latex.cache = out.join("latex-cache");
            let scale = Scale::new([0., 1.], Normalization::Linear, ColorMap::storm())?;
            let projected: Vec<[f64; 2]> = pos
                .chunks_exact(3)
                .map(|q| {
                    if oblique {
                        [q[0] - 0.5 * q[1], q[2] + 0.25 * q[1]]
                    } else {
                        [q[0], q[1]]
                    }
                })
                .collect();
            ensure!(!projected.is_empty(), "empty lifted volume");
            let xlim = [
                projected.iter().map(|q| q[0]).fold(f64::INFINITY, f64::min) - 40.,
                projected
                    .iter()
                    .map(|q| q[0])
                    .fold(f64::NEG_INFINITY, f64::max)
                    + 40.,
            ];
            let ylim = [
                projected.iter().map(|q| q[1]).fold(f64::INFINITY, f64::min) - 40.,
                projected
                    .iter()
                    .map(|q| q[1])
                    .fold(f64::NEG_INFINITY, f64::max)
                    + 40.,
            ];
            let mut ax = Axes::new(
                xlim,
                ylim,
                if oblique {
                    "East - 0.5 North (km)"
                } else {
                    "East of radar (km)"
                },
                if oblique {
                    "Up + 0.25 North (km)"
                } else {
                    "North of radar (km)"
                },
            );
            ax.xticks = ((xlim[0] / 200.).ceil() as i32..=(xlim[1] / 200.).floor() as i32)
                .map(|i| i as f64 * 200.)
                .collect();
            ax.yticks = ((ylim[0] / 200.).ceil() as i32..=(ylim[1] / 200.).floor() as i32)
                .map(|i| i as f64 * 200.)
                .collect();
            let mut p = Panel::new(Rect::new(25., 14., 125., 121.), ax);
            let mut begin = 0;
            while begin < ratios.len() {
                let mut end = begin + 1;
                while end < ratios.len()
                    && labels[2 * end] == labels[2 * begin]
                    && labels[2 * end + 1] == labels[2 * begin + 1]
                {
                    end += 1;
                }
                let color = scale.map(ratios[begin]).unwrap();
                let c = format!("#{:02x}{:02x}{:02x}", color[0], color[1], color[2]);
                let mut order: Vec<usize> = (begin..end).collect();
                order.sort_by(|a, b| s[*a].total_cmp(&s[*b]));
                let pts = order
                    .into_iter()
                    .map(|i| {
                        let q = &pos[3 * i..3 * i + 3];
                        if oblique {
                            [q[0] - 0.5 * q[1], q[2] + 0.25 * q[1]]
                        } else {
                            [q[0], q[1]]
                        }
                    })
                    .collect();
                line(&mut p, pts, &c, 2., 0.5);
                begin = end;
            }
            let n = p.layers.len();
            p.rasterize_lines(0..n, 400.)?;
            f.panels.push(p);
            let scale = f.scale(scale);
            f.colorbars.push(Colorbar::new(
                Rect::new(156., 14., 3., 121.),
                scale,
                "Relative density proxy",
                vec![0., 0.5, 1.],
            ));
            f.labels.push(Label::new(
                "Field-aligned lift of measured cavity contrast",
                89.,
                7.,
            ));
            if std::env::var_os("SYISR_MEMO_FIGURES").is_some() {
                f.labels.clear();
            }
            f.save(out.join(format!("{name}.png")))?;
            if std::env::var_os("SYISR_MEMO_FIGURES").is_some() {
                f.labels.clear();
            }
            f.save(out.join(format!("{name}.pdf")))?;
        }
    }
    Ok(())
}
fn aspect_view(root: &Path, out: &Path) -> Result<()> {
    let h = hdf5::File::open(root.join("geometry.h5"))?;
    let mut names: Vec<_> = h
        .member_names()?
        .into_iter()
        .filter(|s| s.starts_with("beam_"))
        .collect();
    names.sort();
    let mut f = Figure::new(IntendedUse::ArticleWide);
    f.width_mm = 178.;
    f.height_mm = 125.;
    f.raster_dpi = 400.;
    f.latex.cache = out.join("latex-cache");
    let mut ax = Axes::new(
        [0., 70.],
        [150., 900.],
        "Deviation from perpendicular to B (deg)",
        "Nominal altitude (km)",
    );
    ax.xticks = vec![0., 2., 5., 15., 30., 45., 60.];
    ax.yticks = vec![200., 400., 600.];
    let mut p = Panel::new(Rect::new(24., 14., 142., 87.), ax);
    for (i, name) in names.iter().enumerate() {
        let g = h.group(name)?;
        let z = v(&g, "geodetic_lat_lon_alt")?;
        let angle = v(&g, "aspect_deviation_from_perpendicular_deg")?;
        let pts: Vec<_> = angle
            .iter()
            .enumerate()
            .filter(|(j, _)| z[3 * j + 2] >= 180. && z[3 * j + 2] <= 650.)
            .map(|(j, a)| [a.abs(), z[3 * j + 2]])
            .collect();
        line(&mut p, pts.clone(), &color(i), 1.2, 1.);
        let target = 220. + i as f64 * 35.;
        let nearest = pts
            .iter()
            .min_by(|a, b| (a[1] - target).abs().total_cmp(&(b[1] - target).abs()));
        if let Some(point) = nearest {
            p.label_at(format!("{}", i + 1), [point[0] + 1., point[1]]);
        }
    }
    for cut in [2., 5.] {
        let mut l = Line::new(vec![[cut, 180.], [cut, 650.]], "#888888");
        l.dash = Some("2 1".into());
        l.width_pt = 0.6;
        p.push(Layer::Line(l));
    }
    f.panels.push(p);
    f.labels.push(Label::new(
        "Magnetic aspect along the eleven beams",
        89.,
        7.,
    ));
    if std::env::var_os("SYISR_MEMO_FIGURES").is_some() {
        f.labels.clear();
    }
    f.save(out.join("beam_aspect.png"))?;
    if std::env::var_os("SYISR_MEMO_FIGURES").is_some() {
        f.labels.clear();
    }
    f.save(out.join("beam_aspect.pdf"))?;
    Ok(())
}
fn drift_view(root: &Path, out: &Path) -> Result<()> {
    let path = root.join("cavity/selection.h5");
    if !path.exists() {
        return Ok(());
    }
    let h = hdf5::File::open(path)?;
    let config = v(&h, "config_vx_vy_lambda")?;
    let score = v(&h, "whole_beam_holdout_rmse")?;
    let best = v(&h, "selected_config")?;
    let mut vx: Vec<f64> = config.chunks_exact(3).map(|v| v[0] * 1000.).collect();
    vx.sort_by(f64::total_cmp);
    vx.dedup();
    let mut vy: Vec<f64> = config.chunks_exact(3).map(|v| v[1] * 1000.).collect();
    vy.sort_by(f64::total_cmp);
    vy.dedup();
    let mut values = vec![f64::INFINITY; vx.len() * vy.len()];
    for (i, c) in config.chunks_exact(3).enumerate() {
        let x = vx
            .iter()
            .position(|v| (*v - c[0] * 1000.).abs() < 1e-6)
            .unwrap();
        let y = vy
            .iter()
            .position(|v| (*v - c[1] * 1000.).abs() < 1e-6)
            .unwrap();
        values[y * vx.len() + x] = values[y * vx.len() + x].min(score[i]);
    }
    let step = vx[1] - vx[0];
    let mut xe: Vec<f64> = vx.iter().map(|v| v - step / 2.).collect();
    xe.push(vx.last().unwrap() + step / 2.);
    let step = vy[1] - vy[0];
    let mut ye: Vec<f64> = vy.iter().map(|v| v - step / 2.).collect();
    ye.push(vy.last().unwrap() + step / 2.);
    let low = values.iter().copied().fold(f64::INFINITY, f64::min);
    let high = values.iter().copied().fold(f64::NEG_INFINITY, f64::max);
    let mut f = Figure::new(IntendedUse::ArticleWide);
    f.width_mm = 178.;
    f.height_mm = 95.;
    f.raster_dpi = 400.;
    f.latex.cache = out.join("latex-cache");
    let scale = f.scale(Scale::new(
        [low, high],
        Normalization::Linear,
        ColorMap::storm(),
    )?);
    let mut ax = Axes::new(
        [xe[0], *xe.last().unwrap()],
        [ye[0], *ye.last().unwrap()],
        "Eastward pattern speed (m/s)",
        "Cross-field upward speed (m/s)",
    );
    ax.xticks = vx.clone();
    ax.yticks = vy.clone();
    let mut p = Panel::new(Rect::new(24., 14., 123., 57.), ax);
    p.pcolormesh(xe, ye, values, scale)?;
    let sx = best[0] * 1000.;
    let sy = best[1] * 1000.;
    line(
        &mut p,
        vec![[sx - 4., sy], [sx + 4., sy]],
        "#ffffff",
        1.6,
        1.,
    );
    line(
        &mut p,
        vec![[sx, sy - 4.], [sx, sy + 4.]],
        "#ffffff",
        1.6,
        1.,
    );
    f.panels.push(p);
    f.colorbars.push(Colorbar::new(
        Rect::new(154., 14., 3., 57.),
        scale,
        "Holdout RMSE",
        vec![(low * 100.).ceil() / 100., (high * 100.).floor() / 100.],
    ));
    f.labels.push(Label::new(
        "Joint drift and morphology: whole-beam prediction",
        89.,
        7.,
    ));
    if std::env::var_os("SYISR_MEMO_FIGURES").is_some() {
        f.labels.clear();
    }
    f.save(out.join("cavity_drift_scan.png"))?;
    if std::env::var_os("SYISR_MEMO_FIGURES").is_some() {
        f.labels.clear();
    }
    f.save(out.join("cavity_drift_scan.pdf"))?;
    Ok(())
}
