//! Forward RTIs on observation cells. Rendering never interpolates absent gates.
use anyhow::{ensure, Result};
use rustplotlib::*;
use std::path::Path;
fn edges(v: &[f64]) -> Vec<f64> {
    let d = v[1] - v[0];
    (0..=v.len())
        .map(|i| v[0] - d / 2. + i as f64 * d)
        .collect()
}
fn main() -> Result<()> {
    let args: Vec<_> = std::env::args().collect();
    let h = hdf5::File::open(&args[1])?;
    let out = Path::new(&args[2]);
    std::fs::create_dir_all(out)?;
    let obs = h.dataset("observations")?.read_raw::<f64>()?;
    let fit = h.dataset("depletion_ratio")?.read_raw::<f64>()?;
    let hold = h.dataset("heldout_ratio")?.read_raw::<f64>()?;
    for beam in 1..=11 {
        let ids: Vec<_> = obs
            .chunks_exact(34)
            .enumerate()
            .filter(|(_, p)| p[1] as usize == beam - 1)
            .map(|(i, _)| i)
            .collect();
        let mut t: Vec<_> = ids.iter().map(|i| obs[i * 34]).collect();
        t.sort_by(f64::total_cmp);
        t.dedup();
        let mut r: Vec<_> = ids.iter().map(|i| obs[i * 34 + 2]).collect();
        r.sort_by(f64::total_cmp);
        r.dedup();
        let dt = t
            .windows(2)
            .map(|v| v[1] - v[0])
            .fold(f64::INFINITY, f64::min);
        let dr = r
            .windows(2)
            .map(|v| v[1] - v[0])
            .fold(f64::INFINITY, f64::min);
        let nt = ((t[t.len() - 1] - t[0]) / dt).round() as usize + 1;
        let nr = ((r[r.len() - 1] - r[0]) / dr).round() as usize + 1;
        let times: Vec<_> = (0..nt)
            .map(|i| (t[0] + i as f64 * dt - 1772323200.) / 3600.)
            .collect();
        let ranges: Vec<_> = (0..nr).map(|i| r[0] + i as f64 * dr).collect();
        let mut f = Figure::new(IntendedUse::ArticleWide);
        f.width_mm = 330.;
        f.height_mm = 125.;
        f.raster_dpi = 400.;
        f.latex.cache = out.join("latex-cache");
        let scale = f.scale(Scale::new(
            [0., 1.2],
            Normalization::Linear,
            ColorMap::storm(),
        )?);
        for col in 0..3 {
            let mut values = vec![f64::NAN; nt * nr];
            for i in &ids {
                let p = &obs[i * 34..i * 34 + 34];
                let it = ((p[0] - t[0]) / dt).round() as usize;
                let ir = ((p[2] - r[0]) / dr).round() as usize;
                values[ir * nt + it] = match col {
                    0 => p[6],
                    1 => fit[*i],
                    _ => hold[*i],
                };
            }
            let mut ax = Axes::new(
                [11.75, 13.75],
                [200., 700.],
                "UTC",
                if col == 0 { "Slant range (km)" } else { "" },
            );
            ax.xticks = vec![12., 12.5, 13., 13.5];
            ax.xticklabels = Some(vec![
                "12:00".into(),
                "12:30".into(),
                "13:00".into(),
                "13:30".into(),
            ]);
            ax.yticks = vec![200.,300.,400.,500.,600.,700.];
            if col > 0 {
                ax.show_y_axis = false;
            }
            let rect = Rect::new(24. + col as f64 * 94., 12., 94., 88.);
            ensure!(
                rect.width * f.raster_dpi / 25.4 >= nt as f64
                    && rect.height * f.raster_dpi / 25.4 >= nr as f64,
                "raster undersampling"
            );
            let mut p = Panel::new(rect, ax);
            p.pcolormesh(edges(&times), edges(&ranges), values, scale)?;
            let title = match col {
                0 => "Observed",
                1 => "Depletion model",
                _ => "Withheld beam prediction",
            };
            p.label_at(format!("{beam}: {title}"), [12.75, 680.]).color = "#111111".into();
            f.panels.push(p);
        }
        f.colorbars.push(Colorbar::new(
            Rect::new(310., 12., 3., 88.),
            scale,
            "Echo / quiet reference",
            vec![0., 0.3, 0.6, 0.9, 1.2],
        ));
        f.save(out.join(format!("beam_{beam:02}.pdf")))?;
        f.save(out.join(format!("beam_{beam:02}.png")))?;
    }
    Ok(())
}
