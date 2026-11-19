//! H+ model comparison at native Beamer inclusion size; no difference panels.
use anyhow::{ensure, Result};
use rustplotlib::*;
use std::path::Path;
fn main() -> Result<()> {
    let a: Vec<_> = std::env::args().collect();
    ensure!(a.len() == 3, "COMPARISON_HDF OUTPUT");
    let out = Path::new(&a[2]);
    std::fs::create_dir_all(out)?;
    let h = hdf5::File::open(&a[1])?;
    let g = h.group("beam-az65171-el16384")?;
    let x = g.dataset("china_local_hours")?.read_raw::<f64>()?;
    let y = g.dataset("altitude_edges_km")?.read_raw::<f64>()?;
    let mut f = ArticleLayout::new(144., 9.)?.figure(1., 69.)?;
    f.raster_dpi = 500.;
    f.latex.cache = out.join("latex-cache");
    let scale = f.scale(Scale::new(
        [0., 1.],
        Normalization::Linear,
        ColorMap::turbo(),
    )?);
    let fields = ["sanya", "sami2", "tbt2015", "tts2003", "dy1985", "gcpm"];
    let titles = [
        "Sanya AC fit",
        r"\shortstack{SAMI2: Fejer--Scherliess drifts\\HWM14 neutral winds}",
        "TBT-2015",
        "TTS-2003",
        "DY-1985",
        "GCPM v2.4",
    ];
    for i in 0..6 {
        let r = i / 3;
        let c = i % 3;
        let left = 14. + 41. * c as f64;
        let top = 11. + 29. * r as f64;
        let mut ax = Axes::new(
            [x[0], *x.last().unwrap()],
            [500., 1300.],
            if r == 1 { "Local time (UTC+8)" } else { "" },
            if c == 0 { "Altitude (km)" } else { "" },
        );
        ax.yticks = vec![500., 900., 1300.];
        ax.xticks = vec![8., 20., 32.];
        ax.xticklabels = Some(vec!["08:00".into(), "20:00".into(), "08:00".into()]);
        if r == 0 {
            ax.show_x_axis = false;
        }
        let mut p = Panel::new(Rect::new(left, top, 32., 18.), ax);
        p.pcolormesh(
            x.clone(),
            y.clone(),
            g.dataset(&format!("{}_h_fraction", fields[i]))?
                .read_raw::<f64>()?,
            scale,
        )?;
        f.panels.push(p);
        f.labels.push(Label::new(titles[i], left + 16., top - 5.));
    }
    f.colorbars.push(Colorbar::new(
        Rect::new(135., 11., 2., 47.),
        scale,
        "",
        vec![0., 0.5, 1.],
    ));
    f.save(out.join("hplus_models.png"))?;
    f.save_for_article(
        out.join("hplus_models.pdf"),
        "figures/memo050/hplus_models.pdf",
    )?;
    Ok(())
}
