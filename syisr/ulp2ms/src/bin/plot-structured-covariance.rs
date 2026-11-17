//! Structured covariance validation, rendered entirely with rustplotlib.
use anyhow::{ensure, Result};
use rustplotlib::*;
use std::path::Path;
fn median(mut v: Vec<f64>) -> f64 {
    v.sort_by(f64::total_cmp);
    let n = v.len();
    (v[(n - 1) / 2] + v[n / 2]) * 0.5
}
fn main() -> Result<()> {
    let a: Vec<String> = std::env::args().collect();
    ensure!(a.len() == 2, "plot-structured-covariance validation-root");
    let root = Path::new(&a[1]);
    let h = hdf5::File::open(root.join("validation.h5"))?;
    let ds = h.dataset("correlation_kernels")?;
    let n = ds.shape()[0];
    let k = ds.read_raw::<f64>()?;
    let mut f = Figure::new(IntendedUse::ArticleWide);
    f.width_mm = 236.;
    f.height_mm = 210.;
    f.raster_dpi = 180.;
    f.labels.push(Label::new(
        "SYISR: testing the range--lag covariance",
        115.,
        5.,
    ));
    let sc = f.scale(Scale::new(
        [-1., 1.],
        Normalization::Linear,
        ColorMap::diverging(),
    )?);
    for m in 0..3 {
        for c in 0..2 {
            let data = (0..325)
                .map(|j| median((0..n).map(|t| k[((t * 3 + m) * 2 + c) * 325 + j]).collect()))
                .collect();
            let ax = Axes::new(
                [-120., 120.],
                [0., 300.],
                if m == 2 {
                    r"Lag separation ($\mu$s)"
                } else {
                    ""
                },
                if c == 0 { "Gate separation (km)" } else { "" },
            );
            let mut p = Panel::new(
                Rect::new(22. + c as f64 * 104., 26. + m as f64 * 55., 80., 40.),
                ax,
            );
            p.push(Layer::Pcolormesh(Field::new(
                (0..=25).map(|i| -125. + i as f64 * 10.).collect(),
                (0..=13).map(|i| -12.5 + i as f64 * 25.).collect(),
                data,
                sc,
            )?));
            f.panels.push(p);
            f.labels.push(Label::new(
                format!(
                    "{}: {}",
                    [
                        "Held-out measurements",
                        "Gaussian voltage model",
                        "Model + empirical correction"
                    ][m],
                    if c == 0 { "real" } else { "imaginary" }
                ),
                62. + c as f64 * 104.,
                18. + m as f64 * 55.,
            ));
        }
    }
    let mut bar = Colorbar::new(
        Rect::new(214., 26., 3., 150.),
        sc,
        "Correlation",
        vec![-1., -0.5, 0., 0.5, 1.],
    );
    bar.horizontal = false;
    f.colorbars.push(bar);
    f.labels.push(Label::new(
        format!("Median of {n} windows; training and validation use separate ten-second blocks"),
        115.,
        194.,
    ));
    for ext in ["png", "pdf", "svg"] {
        f.save(root.join(format!("structured-covariance-validation.{ext}")))?;
    }
    Ok(())
}
