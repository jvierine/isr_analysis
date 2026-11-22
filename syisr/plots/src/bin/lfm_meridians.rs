//! Observed drift and independent meridian sections; Rustplotlib default style.
use anyhow::Result;
use rustplotlib::*;
use std::path::Path;
fn main() -> Result<()> {
    let a: Vec<_> = std::env::args().collect();
    let root = Path::new(&a[1]);
    let out = Path::new(&a[2]);
    std::fs::create_dir_all(out)?;
    let h = hdf5::File::open(root.join("drift_profile.h5"))?;
    let bounds = h.dataset("height_bounds_km")?.read_raw::<f64>()?;
    let speed = h.dataset("best_speed_m_s")?.read_raw::<f64>()?;
    let q = h.dataset("stability_quantiles_m_s")?.read_raw::<f64>()?;
    let pooled = h.dataset("pooled_speed_m_s")?.read_scalar::<f64>()?;
    let qualified = h.dataset("qualified")?.read_raw::<bool>()?;
    let mut f = Figure::new(IntendedUse::ArticleWide);
    f.width_mm = 170.;
    f.height_mm = 150.;
    f.latex.cache = out.join("latex-cache");
    let mut p = Panel::new(
        Rect::new(25., 12., 125., 115.),
        Axes::new(
            [40., 300.],
            [250., 700.],
            "Eastward pattern speed (m/s)",
            "Altitude (km)",
        ),
    );
    for i in 0..speed.len() {
        let z = (bounds[2 * i] + bounds[2 * i + 1]) / 2.;
        p.push(Layer::Line(Line::new(
            vec![[q[3 * i], z], [q[3 * i + 2], z]],
            "#888888",
        )));
    }
    p.push(Layer::Line(Line::series(
        speed
            .iter()
            .enumerate()
            .map(|(i, v)| {
                if qualified[i] {
                    [*v, (bounds[2 * i] + bounds[2 * i + 1]) / 2.]
                } else {
                    [f64::NAN; 2]
                }
            })
            .collect(),
        0,
    )));
    let weak: Vec<_> = speed
        .iter()
        .enumerate()
        .filter(|(i, v)| !qualified[*i] && v.is_finite())
        .map(|(i, v)| [*v, (bounds[2 * i] + bounds[2 * i + 1]) / 2.])
        .collect();
    p.markers(
        &mut f,
        &weak,
        &MarkerStyle {
            shape: MarkerShape::Cross,
            size_pt: 5.,
            color: [0.5, 0.5, 0.5],
        },
    )?;
    let mut l = Line::new(vec![[pooled, 250.], [pooled, 700.]], "#111111");
    l.dash = Some("3 2".into());
    p.push(Layer::Line(l));
    f.panels.push(p);
    f.save(out.join("altitude_drift.pdf"))?;
    f.save(out.join("altitude_drift.png"))?;
    let h = hdf5::File::open(root.join("sections.h5"))?;
    let x = h.dataset("x_km")?.read_raw::<f64>()?;
    let y = h.dataset("y_km")?.read_raw::<f64>()?;
    let mut f = Figure::new(IntendedUse::ArticleWide);
    f.width_mm = 330.;
    f.height_mm = 150.;
    f.raster_dpi = 400.;
    f.latex.cache = out.join("latex-cache");
    let scale = f.scale(Scale::new(
        [0., 1.5],
        Normalization::Linear,
        ColorMap::storm(),
    )?);
    let xe: Vec<_> = (0..=x.len()).map(|i| x[0] - 5. + i as f64 * 10.).collect();
    let ye: Vec<_> = (0..=y.len()).map(|i| y[0] - 5. + i as f64 * 10.).collect();
    for (i, name) in ["west", "central", "east"].iter().enumerate() {
        let g = h.group(name)?;
        let d = g.dataset("contrast")?.read_raw::<f64>()?;
        let w = g.dataset("support_weight")?.read_raw::<f64>()?;
        let max = w.iter().copied().fold(0., f64::max);
        let values = d
            .iter()
            .zip(w)
            .map(|(d, w)| if w > max * 0.005 { 1. + d } else { f64::NAN })
            .collect();
        let mut ax = Axes::new(
            [-800., 800.],
            [-250., 350.],
            "Moving cross-field east (km)",
            if i == 0 {
                "Cross-field upward (km)"
            } else {
                ""
            },
        );
        if i > 0 {
            ax.show_y_axis = false;
        }
        let mut p = Panel::new(Rect::new(24. + i as f64 * 95., 14., 95., 108.), ax);
        p.pcolormesh(xe.clone(), ye.clone(), values, scale)?;
        p.label_at(*name, [-560., 310.]).color = "#171717".into();
        f.panels.push(p);
    }
    f.colorbars.push(Colorbar::new(
        Rect::new(313., 14., 3., 108.),
        scale,
        "Relative echo proxy",
        vec![0., 0.5, 1., 1.5],
    ));
    f.save(out.join("three_sections.pdf"))?;
    f.save(out.join("three_sections.png"))?;
    Ok(())
}
