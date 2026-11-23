//! HPLA89: identical support and axes for original/refined finite-beam fits.
use anyhow::Result;
mod storm;
use rustplotlib::*;
use std::path::Path;
fn main() -> Result<()> {
    let args: Vec<_> = std::env::args().collect();
    let root = Path::new(&args[1]);
    let out = Path::new(&args[2]);
    std::fs::create_dir_all(out)?;
    let xe: Vec<_> = (0..=701).map(|i| -701. + 2. * i as f64).collect();
    let ye: Vec<_> = (0..=351).map(|i| -301. + 2. * i as f64).collect();
    for name in ["west", "central", "east"] {
        let old = hdf5::File::open(root.join(format!("editor_seed/{name}.h5")))?;
        let new = hdf5::File::open(root.join(format!("editor_refinement/{name}.h5")))?;
        let s = old.dataset("support_weight")?.read_raw::<f64>()?;
        let threshold = 0.005 * s.iter().copied().fold(0., f64::max);
        let mut fig = ArticleLayout::new(179.324, 10.95)?.figure(1., 80.)?;
        fig.raster_dpi = 500.;
        fig.latex.cache = out.join("latex-cache");
        let scale = fig.scale(Scale::new(
            [0., 1.],
            Normalization::Linear,
            storm::palette()?,
        )?);
        for (j, (h, label)) in [(&old, "Previous"), (&new, "Refined")].iter().enumerate() {
            let q = h.dataset("contrast")?.read_raw::<f64>()?;
            let mut ax = Axes::new(
                [-260., 340.],
                [-210., 250.],
                "$u$ (km)",
                if j == 0 { "$v$ (km)" } else { "" },
            );
            ax.xticks = vec![-200., 0., 200.];
            ax.yticks = vec![-200., 0., 200.];
            let mut panel = Panel::new(Rect::new(18. + j as f64 * 76., 7., 65., 49.833333), ax);
            panel.pcolormesh(
                xe.clone(),
                ye.clone(),
                q.iter()
                    .zip(&s)
                    .map(|(q, s)| if *s > threshold { 1. + q } else { f64::NAN })
                    .collect(),
                scale,
            )?;
            panel.label_at(*label, [-180., 205.]).background = Some("#ffffff".into());
            fig.panels.push(panel);
        }
        fig.colorbars.push(Colorbar::new(
            Rect::new(161., 7., 3., 49.833333),
            scale,
            "Relative echo",
            vec![0., 0.5, 1.],
        ));
        fig.save_for_article(
            out.join(format!("{name}.pdf")),
            &format!("figures/memo089/{name}.pdf"),
        )?;
    }
    Ok(())
}
