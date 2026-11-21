//! Resolution comparison with explicit data-cell-to-image-pixel guarantees.
//! Same storm scale and physical framing for every candidate; no interpolation.
use anyhow::{ensure, Result};
use image::{Rgb, RgbImage};
use rustplotlib::*;
use std::path::Path;
struct Map {
    key: String,
    time: f64,
    range: f64,
    x: Vec<f64>,
    z: Vec<f64>,
    v: Vec<f64>,
    nt: usize,
    nz: usize,
}
fn read(root: &Path, seconds: i32, group: i32) -> Result<Map> {
    let key = format!("t{seconds:02}_g{group:02}");
    let h = hdf5::File::open(root.join(format!("{key}.h5")))?;
    let te = h.dataset("time_edges_unix_s")?.read_raw::<f64>()?;
    let day = (te[0] / 86400.).floor() * 86400.;
    let x: Vec<f64> = te.iter().map(|t| (t - day) / 3600.).collect();
    let ze = h.dataset("altitude_edges_km")?.read_raw::<f64>()?;
    let lo = ze.windows(2).position(|w| w[0] >= 180.).unwrap();
    let hi = ze.iter().rposition(|z| *z <= 650.).unwrap();
    let z = ze[lo..=hi].to_vec();
    let nt = x.len() - 1;
    let nz = z.len() - 1;
    let sig = h.dataset("median_signal")?.read_raw::<f64>()?;
    let ranges = h.dataset("range_km")?.read_raw::<f64>()?;
    let original_nz = ranges.len();
    let mut v = vec![0.; nt * nz];
    for k in 0..nz {
        for j in 0..nt {
            let p = sig[j * original_nz + k + lo] * ranges[k + lo].powi(2);
            v[k * nt + j] = if p.is_finite() {
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
    Ok(Map {
        key,
        time: seconds as f64,
        range: h.attr("range_bin_km")?.read_scalar()?,
        x,
        z,
        v,
        nt,
        nz,
    })
}
fn panel(m: &Map, rect: Rect, scale: usize, xlabels: bool, ylabels: bool) -> Result<Panel> {
    let mut ax = Axes::new(
        [12. + 20. / 60., 13. + 10. / 60.],
        [180., 650.],
        if xlabels {
            "UTC (Beijing = UTC + 8 h)"
        } else {
            ""
        },
        if ylabels { "Nominal altitude (km)" } else { "" },
    );
    ax.xticks = vec![12.5, 12.75, 13.];
    ax.xticklabels = Some(vec!["12:30".into(), "12:45".into(), "13:00".into()]);
    ax.yticks = vec![200., 300., 400., 500., 600.];
    if !xlabels {
        ax.show_x_axis = false;
    }
    if !ylabels {
        ax.show_y_axis = false;
    }
    let mut p = Panel::new(rect, ax);
    p.pcolormesh(m.x.clone(), m.z.clone(), m.v.clone(), scale)?;
    Ok(p)
}
fn guarantee(m: &Map, rect: &Rect, dpi: f64) -> Result<(u32, u32)> {
    let w = (rect.width * dpi / 25.4).floor() as u32;
    let h = (rect.height * dpi / 25.4).floor() as u32;
    ensure!(
        w >= m.nt as u32 && h >= m.nz as u32,
        "image undersamples data: {} {w}x{h}, {}x{} cells",
        m.key,
        m.nt,
        m.nz
    );
    Ok((w, h))
}
fn main() -> Result<()> {
    let a: Vec<_> = std::env::args().collect();
    ensure!(a.len() == 3, "SWEEP_ROOT OUTPUT");
    let root = Path::new(&a[1]);
    let out = Path::new(&a[2]);
    std::fs::create_dir_all(out)?;
    let mut maps = Vec::new();
    for seconds in [5, 15, 30, 60] {
        for group in [1, 2, 4, 8] {
            maps.push(read(root, seconds, group)?);
        }
    }
    let scale = Scale::new([78., 105.], Normalization::Linear, ColorMap::storm())?;
    let mut metadata = String::new();
    for m in &maps {
        // Borderless cell map: exactly 2x2 output pixels per data cell, no resampling.
        let mut im = RgbImage::new((m.nt * 2) as u32, (m.nz * 2) as u32);
        for k in 0..m.nz {
            for j in 0..m.nt {
                let rgba = scale.map(m.v[k * m.nt + j]).unwrap_or([255, 255, 255, 255]);
                let rgb = Rgb([rgba[0], rgba[1], rgba[2]]);
                for dy in 0..2 {
                    for dx in 0..2 {
                        im.put_pixel((j * 2 + dx) as u32, ((m.nz - 1 - k) * 2 + dy) as u32, rgb);
                    }
                }
            }
        }
        im.save(out.join(format!("{}_cells.png", m.key)))?;
        let mut f = Figure::new(IntendedUse::ArticleWide);
        f.width_mm = 150.;
        f.height_mm = 182.;
        f.raster_dpi = 600.;
        f.latex.cache = out.join("latex-cache");
        let s = f.scale(scale.clone());
        let rect = Rect::new(23., 14., 99., 145.);
        let (w, h) = guarantee(m, &rect, f.raster_dpi)?;
        f.panels.push(panel(m, rect, s, true, true)?);
        f.labels.push(Label::new(
            format!("{:.0} s $\\times$ {:.3} km", m.time, m.range),
            72.,
            7.,
        ));
        f.colorbars.push(Colorbar::new(
            Rect::new(127., 14., 3., 145.),
            s,
            "dB (arbitrary)",
            vec![78., 85., 95., 105.],
        ));
        f.save(out.join(format!("{}.png", m.key)))?;
        f.save(out.join(format!("{}.pdf", m.key)))?;
        metadata.push_str(&format!(
            "{} cells={}x{} labelled_plot={}x{} pixels cell_map={}x{} pixels\n",
            m.key,
            m.nt,
            m.nz,
            w,
            h,
            m.nt * 2,
            m.nz * 2
        ));
    }
    // Contact sheet also preserves at least one raster pixel per finest data cell.
    let mut f = Figure::new(IntendedUse::ArticleWide);
    f.width_mm = 385.;
    f.height_mm = 502.;
    f.font_pt = 9.;
    f.raster_dpi = 360.;
    f.latex.cache = out.join("latex-cache");
    let s = f.scale(scale);
    for (i, m) in maps.iter().enumerate() {
        let row = i / 4;
        let col = i % 4;
        let rect = Rect::new(23. + col as f64 * 85., 14. + row as f64 * 119., 75., 105.);
        guarantee(m, &rect, f.raster_dpi)?;
        f.labels.push(Label::new(
            format!("{:.0} s $\\times$ {:.3} km", m.time, m.range),
            rect.x + rect.width / 2.,
            rect.y - 4.,
        ));
        f.panels.push(panel(m, rect, s, row == 3, col == 0)?);
    }
    f.colorbars.push(Colorbar::new(
        Rect::new(362., 14., 3., 462.),
        s,
        "dB (arbitrary)",
        vec![78., 85., 95., 105.],
    ));
    f.save(out.join("resolution_matrix.png"))?;
    f.save(out.join("resolution_matrix.pdf"))?;
    std::fs::write(out.join("pixel_support.txt"), metadata)?;
    println!("Rendered 16 resolutions with native-cell support preserved");
    Ok(())
}
