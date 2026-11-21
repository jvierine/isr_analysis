//! Multibeam EPB views; dated geometry and data are HDF5 inputs.
use anyhow::{ensure, Result};
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
            [-400., 400.],
            [-350., 350.],
        ),
        (
            1,
            "geometry_top",
            "View from above",
            "East of radar (km)",
            "North of radar (km)",
            [-400., 400.],
            [-500., 500.],
        ),
        (
            2,
            "geometry_perp_B_above",
            "View perpendicular to B from above",
            "Cross-field east (km)",
            "Distance along B (km)",
            [-400., 400.],
            [-650., 650.],
        ),
        (
            3,
            "geometry_oblique",
            "Oblique view",
            "East - 0.5 North (km)",
            "Up + 0.25 North (km)",
            [-500., 500.],
            [-150., 950.],
        ),
        (
            4,
            "geometry_mapped",
            "Field-line mapping to the reference plane",
            "Mapped cross-field east (km)",
            "Mapped cross-field upward (km)",
            [-400., 400.],
            [-350., 350.],
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
                .filter(|(j, _)| geo[3 * j + 2] >= 180. && geo[3 * j + 2] <= 650.)
                .map(|(j, q)| {
                    if mode == 4 {
                        [mapped[3 * j], mapped[3 * j + 1]]
                    } else {
                        project(q, mode)
                    }
                })
                .collect();
            line(&mut p, pts.clone(), &color(i), 1.2, 1.);
            if !pts.is_empty() {
                let index = pts.len() * 3 / 4;
                let point = pts[index];
                p.label_at(format!("{}", i + 1), [point[0] + 8., point[1] + 8.]);
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
        if radar[0] >= xlim[0] && radar[0] <= xlim[1] && radar[1] >= ylim[0] && radar[1] <= ylim[1]
        {
            p.label_at("Radar", [radar[0], radar[1] - 10.]);
        }
        f.panels.push(p);
        f.labels.push(Label::new(title, 88., 7.));
        f.save(out.join(format!("{name}.png")))?;
        f.save(out.join(format!("{name}.pdf")))?;
    }
    Ok(())
}
fn nal(p: &[f64]) -> nalgebra::Vector3<f64> {
    nalgebra::Vector3::new(p[0], p[1], p[2])
}
fn main() -> Result<()> {
    let a: Vec<_> = std::env::args().collect();
    ensure!(a.len() == 3, "ROOT OUTPUT");
    let root = Path::new(&a[1]);
    let out = Path::new(&a[2]);
    std::fs::create_dir_all(out)?;
    geom(root, out)?;
    beam_views(root, out)
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
        let t0 = te
            .iter()
            .position(|t| (*t - day) / 3600. >= 12. + 5. / 60.)
            .unwrap();
        let t1 = te
            .iter()
            .rposition(|t| (*t - day) / 3600. <= 13. + 20. / 60.)
            .unwrap();
        let k0 = ze.iter().position(|z| *z >= 180.).unwrap();
        let k1 = ze.iter().rposition(|z| *z <= 650.).unwrap();
        let nt = t1 - t0;
        let nz = k1 - k0;
        let x = te[t0..=t1].iter().map(|t| (t - day) / 3600.).collect();
        let z = ze[k0..=k1].to_vec();
        let mut values = vec![0.; nt * nz];
        for k in 0..nz {
            for j in 0..nt {
                let p = sig[(j + t0) * r.len() + k + k0] * r[k + k0].powi(2);
                values[k * nt + j] = if p.is_finite() {
                    if p > 0. {
                        10. * p.log10()
                    } else {
                        78.
                    }
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
        [12. + 5. / 60., 13. + 20. / 60.],
        [180., 650.],
        if showx {
            "UTC (Beijing = UTC + 8 h)"
        } else {
            ""
        },
        if showy { "Nominal altitude (km)" } else { "" },
    );
    ax.xticks = vec![12.25, 12.5, 12.75, 13., 13.25];
    ax.xticklabels = Some(vec![
        "12:15".into(),
        "12:30".into(),
        "12:45".into(),
        "13:00".into(),
        "13:15".into(),
    ]);
    ax.yticks = vec![200., 300., 400., 500., 600.];
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
fn beam_views(root: &Path, out: &Path) -> Result<()> {
    let maps = beam_maps(root)?;
    let sc = Scale::new([78., 105.], Normalization::Linear, ColorMap::storm())?;
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
    let mut f = Figure::new(IntendedUse::ArticleWide);
    f.width_mm = 379.;
    f.height_mm = 505.;
    f.font_pt = 9.;
    f.raster_dpi = 360.;
    f.latex.cache = out.join("latex-cache");
    let scale = f.scale(sc);
    for (i, m) in maps.iter().enumerate() {
        let row = i / 3;
        let col = i % 3;
        let rect = Rect::new(23. + col as f64 * 110., 14. + row as f64 * 120., 100., 105.);
        f.labels.push(Label::new(
            format!(
                "{}: az {:.0}$^\\circ$, el {:.0}$^\\circ$",
                i + 1,
                m.az,
                m.el
            ),
            rect.x + 50.,
            rect.y - 4.,
        ));
        f.panels
            .push(beam_panel(m, rect, scale, row == 3, col == 0, 360.)?);
    }
    f.colorbars.push(Colorbar::new(
        Rect::new(352., 14., 3., 465.),
        scale,
        "dB (arbitrary)",
        vec![78., 85., 95., 105.],
    ));
    f.save(out.join("all_beams.png"))?;
    f.save(out.join("all_beams.pdf"))?;
    Ok(())
}
