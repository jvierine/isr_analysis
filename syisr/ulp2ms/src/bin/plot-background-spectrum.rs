//! Spectrum diagnostic for one integration; implementation provenance is metadata only.
use anyhow::{ensure, Result};
use rustplotlib::*;
fn main() -> Result<()> {
    let a: Vec<_> = std::env::args().collect();
    ensure!(
        a.len() == 3,
        "plot-background-spectrum spectrum.h5 output-prefix"
    );
    let f = hdf5::File::open(&a[1])?;
    let get = |n: &str| f.dataset(n).unwrap().read_raw::<f64>().unwrap();
    let x = get("frequency_hz");
    let raw = get("raw_psd_adu2_hz");
    let centered = get("centered_psd_adu2_hz");
    let mut sorted = centered.clone();
    sorted.sort_by(f64::total_cmp);
    let scale = sorted[sorted.len() / 2];
    let mut fig = Figure::new(IntendedUse::ArticleWide);
    fig.width_mm = 190.;
    fig.height_mm = 105.;
    fig.raster_dpi = 160.;
    fig.labels.push(Label::new(
        "Background voltage spectrum: 21 Aug 2023, 14:30--15:00 UTC",
        95.,
        5.,
    ));
    let mut ax = Axes::new(
        [-15., 15.],
        [-8., 30.],
        "Baseband frequency (kHz)",
        "PSD / median background (dB)",
    );
    ax.xticks = vec![-15., -10., -5., 0., 5., 10., 15.];
    ax.yticks = vec![-5., 0., 10., 20., 30.];
    let mut p = Panel::new(Rect::new(25., 17., 150., 62.), ax);
    for (v, color) in [(&raw, "#999999"), (&centered, "#17608e")] {
        p.push(Layer::Line(Line::new(
            x.iter()
                .zip(v)
                .map(|(x, y)| [*x / 1000., 10. * (y / scale).log10()])
                .collect(),
            color,
        )));
    }
    fig.panels.push(p);
    fig.labels.push(Label::new(
        "Grey: original voltages; blue: stationary complex mean removed",
        95.,
        94.,
    ));
    fig.labels.push(Label::new(
        "2048 retained pulses; 256-sample Hann spectra; frequency resolution 391 Hz",
        95.,
        101.,
    ));
    for ext in ["png", "pdf", "svg"] {
        fig.save(format!("{}.{ext}", a[2]))?;
    }
    Ok(())
}
