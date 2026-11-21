//! Single-panel zenith LFM RTI. Scientific provenance stays in code/HDF5.
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
    let r = h.dataset("range_km")?.read_raw::<f64>()?;
    let signal = h.dataset("median_signal")?.read_raw::<f64>()?;
    let nt = x.len() - 1;
    let nz = z.len() - 1;
    let lim = [78., 105.];
    let mut values = vec![f64::NAN; nt * nz];
    for j in 0..nt {
        for k in 0..nz {
            let p = signal[j * nz + k] * r[k].powi(2);
            // Nonpositive estimates map to the display minimum, not missing/white.
            // Scientific signed signal is neither altered nor clipped in HDF5.
            values[k * nt + j] = if p.is_finite() {
                if p > 0. {
                    10. * p.log10()
                } else {
                    lim[0]
                }
            } else {
                f64::NAN
            };
        }
    }
    for (name, bottom, top, bounds, step) in [
        (
            "zenith_rti_full",
            100.,
            1680.,
            [x[0], *x.last().unwrap()],
            1.,
        ),
        (
            "zenith_rti_f_region",
            100.,
            900.,
            [x[0], *x.last().unwrap()],
            1.,
        ),
        ("zenith_rti_event", 100., 800., [11.75, 13.25], 0.25),
    ] {
        let mut f = Figure::new(IntendedUse::ArticleWide);
        f.width_mm = 210.;
        f.height_mm = 100.;
        f.raster_dpi = 360.;
        f.latex.cache = out.join("latex-cache");
        let scale = f.scale(Scale::new(lim, Normalization::Linear, ColorMap::storm())?);
        let mut ax = Axes::new(
            bounds,
            [bottom, top],
            "1 March 2026, UTC (Beijing time = UTC + 8 h)",
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
            vec![200., 600., 1000., 1400., 1600.]
        } else if bottom >= 200. {
            (2..=6)
                .map(|i| i as f64 * 100.)
                .filter(|v| *v >= bottom && *v <= top)
                .collect()
        } else {
            vec![200., 400., 600., 800.]
        };
        let mut p = Panel::new(Rect::new(23., 14., 145., 65.), ax);
        p.pcolormesh(x.clone(), z.clone(), values.clone(), scale)?;
        f.panels.push(p);
        f.labels.push(Label::new("Range-squared signal", 95., 8.));
        f.colorbars.push(Colorbar::new(
            Rect::new(174., 14., 3., 65.),
            scale,
            "dB (arbitrary)",
            vec![78., 85., 95., 105.],
        ));
        f.save(out.join(format!("{name}.png")))?;
        f.save(out.join(format!("{name}.pdf")))?;
    }
    Ok(())
}
