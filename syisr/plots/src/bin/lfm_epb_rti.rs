//! Standalone zenith LFM RTI; provenance is confined to HDF5 and code.
use anyhow::{ensure, Result};
use rustplotlib::*;
use std::path::Path;
fn main() -> Result<()> {
    let a: Vec<_> = std::env::args().collect();
    ensure!(a.len() == 3, "RTI_H5 OUTPUT_DIR");
    let h = hdf5::File::open(&a[1])?;
    let out = Path::new(&a[2]);
    std::fs::create_dir_all(out)?;
    let t = h.dataset("time_edges_unix_s")?.read_raw::<f64>()?;
    let day = (t[0] / 86400.).floor() * 86400.;
    let x: Vec<f64> = t.iter().map(|v| (v - day) / 3600.).collect();
    let z = h.dataset("altitude_edges_km")?.read_raw::<f64>()?;
    let nt = x.len() - 1;
    let nz = z.len() - 1;
    let total = h.dataset("median_power")?.read_raw::<f64>()?;
    let noise = h.dataset("noise_power")?.read_raw::<f64>()?;
    let signal = h.dataset("median_signal")?.read_raw::<f64>()?;
    // Signed linear contrast keeps depletion cores visible, including negative
    // estimates; mask only ranges whose temporal reference is too weak.
    let mut relative = vec![f64::NAN; nt * nz];
    let mut nn = noise.clone();
    nn.sort_by(f64::total_cmp);
    for k in 0..nz {
        let mut col: Vec<f64> = (0..nt)
            .map(|j| signal[j * nz + k])
            .filter(|v| v.is_finite())
            .collect();
        if col.is_empty() {
            continue;
        }
        col.sort_by(f64::total_cmp);
        let baseline = (col[(col.len() - 1) / 2] + col[col.len() / 2]) / 2.;
        if baseline > 0.1 * nn[nn.len() / 2] {
            for j in 0..nt {
                relative[j * nz + k] = signal[j * nz + k] / baseline;
            }
        }
    }
    let r = h.dataset("range_km")?.read_raw::<f64>()?;
    let proxy: Vec<f64> = signal
        .iter()
        .enumerate()
        .map(|(i, v)| {
            if *v > 0.05 * noise[i / nz] {
                10. * (v * r[i % nz].powi(2)).log10()
            } else {
                f64::NAN
            }
        })
        .collect();
    let plim = [85., 110.];
    for (name, top, bounds, step) in [
        ("zenith_rti_full", 1680., [x[0], *x.last().unwrap()], 1.),
        ("zenith_rti_f_region", 900., [x[0], *x.last().unwrap()], 1.),
        ("zenith_rti_event", 800., [11.75, 13.25], 0.25),
    ] {
        let mut f = Figure::new(IntendedUse::ArticleWide);
        f.width_mm = 210.;
        f.height_mm = 210.;
        f.raster_dpi = 180.;
        f.latex.cache = out.join("latex-cache");
        f.labels
            .push(Label::new("SYISR zenith LFM --- 1 March 2026", 100., 7.));
        for q in 0..3 {
            let lim = if q == 0 {
                [-0.5, 6.]
            } else if q == 1 {
                plim
            } else {
                [0., 2.]
            };
            let scale = f.scale(Scale::new(
                lim,
                Normalization::Linear,
                if q == 2 {
                    ColorMap::diverging()
                } else {
                    ColorMap::turbo()
                },
            )?);
            let mut ax = Axes::new(
                bounds,
                [100., top],
                if q == 2 {
                    "Time (UTC; Beijing time = UTC + 8 h)"
                } else {
                    ""
                },
                "Nominal altitude (km)",
            );
            ax.xticks = ((bounds[0] / step).ceil() as i32..=(bounds[1] / step).floor() as i32)
                .map(|v| v as f64 * step)
                .collect();
            ax.xticklabels = Some(
                ax.xticks
                    .iter()
                    .map(|v| format!("{:02}:{:02}", *v as i32, (v.fract() * 60.).round() as i32))
                    .collect(),
            );
            ax.yticks = if top > 1000. {
                vec![200., 600., 1000., 1400.]
            } else {
                vec![200., 400., 600., 800.]
            };
            if q < 2 {
                ax.show_x_axis = false;
            }
            let mut values = vec![f64::NAN; nt * nz];
            for j in 0..nt {
                for k in 0..nz {
                    values[k * nt + j] = match q {
                        0 => 10. * (total[j * nz + k] / noise[j]).log10(),
                        1 => proxy[j * nz + k],
                        _ => relative[j * nz + k],
                    };
                }
            }
            let y = 19. + q as f64 * 61.;
            let mut p = Panel::new(Rect::new(23., y, 145., 48.), ax);
            p.pcolormesh(x.clone(), z.clone(), values, scale)?;
            f.panels.push(p);
            f.labels.push(Label::new(
                [
                    "Median pulse power / estimated noise",
                    "Range-squared signal (SNR $>0.05$)",
                    "Signal / temporal median",
                ][q],
                95.,
                y - 3.,
            ));
            let ticks = if q == 0 {
                vec![0., 2., 4., 6.]
            } else if q == 2 {
                vec![0., 0.5, 1., 1.5, 2.]
            } else {
                vec![
                    plim[0].ceil(),
                    ((plim[0] + plim[1]) / 2.).round(),
                    plim[1].floor(),
                ]
            };
            f.colorbars.push(Colorbar::new(
                Rect::new(174., y, 3., 48.),
                scale,
                if q == 1 {
                    "dB (arbitrary)"
                } else if q == 2 {
                    "Ratio"
                } else {
                    "dB"
                },
                ticks,
            ));
        }
        f.save(out.join(format!("{name}.png")))?;
        f.save(out.join(format!("{name}.pdf")))?;
    }
    Ok(())
}
