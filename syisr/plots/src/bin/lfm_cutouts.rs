//! Independent meridian cross sections and field-line marker overlays on native RTIs.
use anyhow::{ensure, Result};
use rustplotlib::*;
use std::path::Path;
fn data(g: &hdf5::Group, n: &str) -> Result<Vec<f64>> {
    Ok(g.dataset(n)?.read_raw::<f64>()?)
}
fn edges(v: &[f64]) -> Vec<f64> {
    let d = v[1] - v[0];
    (0..=v.len())
        .map(|i| v[0] - d / 2. + i as f64 * d)
        .collect()
}
fn mark(f: &mut Figure, p: &mut Panel, id: &str, xy: [f64; 2], offset: [f64; 2]) -> Result<()> {
    p.markers(
        f,
        &[xy],
        &MarkerStyle {
            shape: MarkerShape::Circle,
            size_pt: 6.,
            color: [1., 1., 1.],
        },
    )?;
    p.markers(
        f,
        &[xy],
        &MarkerStyle {
            shape: MarkerShape::Circle,
            size_pt: 3.,
            color: [0., 0., 0.],
        },
    )?;
    if id.is_empty() {
        return Ok(());
    }
    let l = p.label_at(id, [xy[0] + offset[0], xy[1] + offset[1]]);
    l.background = Some("#ffffff".into());
    l.padding_mm = 0.35;
    Ok(())
}
fn main() -> Result<()> {
    let a: Vec<_> = std::env::args().collect();
    let raw = hdf5::File::open(&a[1])?;
    let h = hdf5::File::open(if a.len() > 3 { &a[3] } else { &a[1] })?;
    let out = Path::new(&a[2]);
    std::fs::create_dir_all(out)?;
    let x = data(&h, "x_km")?;
    let y = data(&h, "y_km")?;
    for (name, prefix) in [("west", "W"), ("central", "C"), ("east", "E")] {
        let g = h.group(name)?;
        let q = data(&g, "ratio")?;
        let support = g.dataset("display_support")?.read_raw::<bool>()?;
        let marks = data(&g, "markers_uv_km")?;
        let mut f = Figure::new(IntendedUse::ArticleWide);
        f.width_mm = 195.;
        f.height_mm = 170.;
        f.raster_dpi = 400.;
        f.latex.cache = out.join("latex-cache");
        let scale = f.scale(Scale::new(
            [0., 1.],
            Normalization::Linear,
            ColorMap::storm(),
        )?);
        let mut ax = Axes::new(
            [-180., 240.],
            [-150., 210.],
            "Moving perpendicular east (km)",
            "Perpendicular upward (km)",
        );
        ax.xticks = vec![-100., 0., 100., 200.];
        ax.yticks = vec![-100., 0., 100., 200.];
        let mut p = Panel::new(Rect::new(26., 16., 140., 120.), ax);
        p.pcolormesh(
            edges(&x),
            edges(&y),
            q.iter()
                .zip(support)
                .map(|(q, s)| if s { *q } else { f64::NAN })
                .collect(),
            scale,
        )?;
        p.label_at(
            format!("{}: looking along $+\\vec{{B}}$", name),
            [30., 190.],
        )
        .background = Some("#ffffff".into());
        for j in 0..5 {
            mark(
                &mut f,
                &mut p,
                &format!("{prefix}{}", j + 1),
                [marks[2 * j], marks[2 * j + 1]],
                [15., 10.],
            )?;
        }
        f.panels.push(p);
        f.colorbars.push(Colorbar::new(
            Rect::new(170., 16., 3., 120.),
            scale,
            "Relative echo",
            vec![0., 0.25, 0.5, 0.75, 1.],
        ));
        f.save(out.join(format!("{name}_cutout.pdf")))?;
        f.save(out.join(format!("{name}_cutout.png")))?;
        let beams = g.dataset("beam_ids")?.read_raw::<i64>()?;
        let crossing = data(&g, "marker_crossings")?;
        for (part, bs) in beams.chunks(3).enumerate() {
            for altitude in [false, true] {
                let mut f = Figure::new(IntendedUse::ArticleWide);
                f.width_mm = 330.;
                f.height_mm = 190.;
                f.raster_dpi = 600.;
                f.latex.cache = out.join("latex-cache");
                let scale = f.scale(Scale::new(
                    [78., 105.],
                    Normalization::Linear,
                    ColorMap::storm(),
                )?);
                for (col, b) in bs.iter().enumerate() {
                    let bg = raw.group(&format!("rtis/beam_{b:02}"))?;
                    let te = data(&bg, "time_edges_unix_s")?;
                    let ranges = data(&bg, "range_km")?;
                    let azel = data(&bg, "az_el_deg")?;
                    let z = if altitude {
                        data(&bg, "altitude_edges_km")?
                    } else {
                        edges(&ranges)
                    };
                    let s = bg.dataset("signed_signal")?.read_raw::<f32>()?;
                    let nt = te.len() - 1;
                    let nr = ranges.len();
                    let t: Vec<_> = te.iter().map(|t| (t - 1772323200.) / 3600.).collect();
                    let mut vals = vec![0.; nt * nr];
                    for r in 0..nr {
                        for j in 0..nt {
                            let v = s[j * nr + r] as f64 * ranges[r].powi(2);
                            vals[r * nt + j] = if !v.is_finite() {
                                f64::NAN
                            } else if v <= 0. {
                                78.
                            } else {
                                10. * v.log10()
                            };
                        }
                    }
                    let top = if altitude { 900. } else { 1100. };
                    let mut ax = Axes::new(
                        [11.75, 13.75],
                        [150., top],
                        "UTC",
                        if col == 0 {
                            if altitude {
                                "Geodetic altitude (km)"
                            } else {
                                "Slant range (km)"
                            }
                        } else {
                            ""
                        },
                    );
                    ax.xticks = vec![12., 12.5, 13., 13.5];
                    ax.xticklabels = Some(vec![
                        "12:00".into(),
                        "12:30".into(),
                        "13:00".into(),
                        "13:30".into(),
                    ]);
                    ax.yticks = vec![200., 400., 600., 800.];
                    if !altitude {
                        ax.yticks.push(1000.);
                    }
                    if col > 0 {
                        ax.show_y_axis = false;
                    }
                    let rect = Rect::new(24. + 94. * col as f64, 12., 94., 150.);
                    ensure!(
                        (rect.width * f.raster_dpi / 25.4) >= nt as f64
                            && (rect.height * f.raster_dpi / 25.4) >= nr as f64,
                        "RTI raster undersampling"
                    );
                    let mut p = Panel::new(rect, ax);
                    p.pcolormesh(t, z, vals, scale)?;
                    let l = p.label_at(
                        format!("{}: ${:.0}^\\circ$, ${:.0}^\\circ$", b, azel[0], azel[1]),
                        [12.75, top - 35.],
                    );
                    l.color = "#ffffff".into();
                    l.background = Some("#333333".into());
                    for row in crossing.chunks_exact(10).filter(|r| r[0] as i64 == *b) {
                        let tx = (row[2] - 1772323200.) / 3600.;
                        let zz = if altitude { row[4] } else { row[3] };
                        let st = row[6] / 3600.;
                        let sz = if altitude { row[8] } else { row[7] };
                        for points in [
                            vec![[tx - st, zz], [tx + st, zz]],
                            vec![[tx, zz - sz], [tx, zz + sz]],
                        ] {
                            let mut l = Line::new(points, "#ffffff");
                            l.width_pt = 1.2;
                            p.push(Layer::Line(l));
                        }
                        mark(&mut f, &mut p, "", [tx, zz], [0.07, 18.])?;
                    }
                    f.panels.push(p);
                }
                f.colorbars.push(Colorbar::new(
                    Rect::new(310., 12., 3., 150.),
                    scale,
                    "Range-squared signal (dB, arbitrary)",
                    vec![78., 91., 105.],
                ));
                let mode = if altitude { "altitude" } else { "range" };
                f.save(out.join(format!("{name}_rtis_{}_{mode}.pdf", part + 1)))?;
                f.save(out.join(format!("{name}_rtis_{}_{mode}.png", part + 1)))?;
            }
        }
    }
    Ok(())
}
