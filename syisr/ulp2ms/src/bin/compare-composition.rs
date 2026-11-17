//! Compare knot-composition and global sigmoid/Gaussian fits on the same products.
use anyhow::{ensure, Result};
use rustplotlib::*;
use syisr_long_pulse::{composition, profile::interpolation};
struct Profile {
    z: Vec<f64>,
    x: Vec<f64>,
    shape: Option<composition::Shape>,
    time: Vec<f64>,
}
impl Profile {
    fn load(path: &str) -> Result<Self> {
        let f = hdf5::File::open(path)?;
        let m: serde_json::Value =
            serde_json::from_slice(&f.attr("metadata_json")?.read_raw::<u8>()?)?;
        ensure!(
            m["fit"]["status"] == "converged",
            "comparison requires converged fits"
        );
        Ok(Self {
            z: f.dataset("slant_range_km")?.read_raw()?,
            x: serde_json::from_value(m["fit"]["transformed_profile"].clone())?,
            shape: composition::from_metadata(&m)?,
            time: f.dataset("time_utc_s")?.read_raw()?,
        })
    }
    fn sample(&self, altitude: f64) -> Result<[f64; 6]> {
        let w = interpolation(&self.z, altitude - 0.05)?;
        let mut p = [0.; 6];
        for (k, v) in w {
            for j in 0..6 {
                p[j] += v * self.x[6 * k + j];
            }
        }
        if let Some(s) = &self.shape {
            let (c, _) = s.composition(altitude, &self.x);
            p[4] = c[0];
            p[5] = c[1];
        }
        Ok([
            11. + p[0] / std::f64::consts::LN_10,
            1000. * p[1].exp(),
            1000. * p[2].exp(),
            1000. * p[3],
            p[4],
            (1. - p[4]) * p[5],
        ])
    }
}
fn main() -> Result<()> {
    let a: Vec<_> = std::env::args().collect();
    ensure!(
        a.len() == 4,
        "compare-composition knot-profile shape-profile output-prefix"
    );
    let old = Profile::load(&a[1])?;
    let new = Profile::load(&a[2])?;
    ensure!(
        old.time == new.time && old.shape.is_none() && new.shape.is_some(),
        "incompatible comparison"
    );
    let mut fig = Figure::new(IntendedUse::ArticleWide);
    fig.width_mm = 230.;
    fig.height_mm = 215.;
    fig.raster_dpi = 160.;
    let start = chrono::DateTime::from_timestamp(new.time[0] as i64, 0).unwrap();
    let end = chrono::DateTime::from_timestamp(new.time[1] as i64, 0).unwrap();
    fig.labels.push(Label::new(
        "SYISR: 2 ms uncoded pulse, composition-model comparison",
        115.,
        5.,
    ));
    fig.labels.push(Label::new(
        format!(
            "{}, {}--{} UTC",
            start.format("%d %b %Y"),
            start.format("%H:%M"),
            end.format("%H:%M")
        ),
        115.,
        12.,
    ));
    for (panel, (field, label, lim, ticks)) in [
        (
            0,
            r"$\log_{10} n_e$ (m$^{-3}$)",
            [7., 13.],
            vec![7., 9., 11., 13.],
        ),
        (
            2,
            r"$T_e$ (K)",
            [0., 8000.],
            vec![0., 2000., 4000., 6000., 8000.],
        ),
        (
            1,
            r"$T_i$ (K)",
            [0., 4000.],
            vec![0., 1000., 2000., 3000., 4000.],
        ),
        (
            3,
            r"Spectral velocity (m s$^{-1}$)",
            [-300., 300.],
            vec![-300., 0., 300.],
        ),
        (
            4,
            r"H$^+$ fraction",
            [0., 1.],
            vec![0., 0.25, 0.5, 0.75, 1.],
        ),
        (
            5,
            r"He$^+$ fraction",
            [0., 0.5],
            vec![0., 0.1, 0.2, 0.3, 0.4, 0.5],
        ),
    ]
    .into_iter()
    .enumerate()
    {
        let mut axes = Axes::new([200.05, 2500.], lim, "Altitude (km)", label);
        axes.xticks = vec![500., 1000., 1500., 2000., 2500.];
        axes.yticks = ticks;
        let mut p = Panel::new(
            Rect::new(
                23. + 115. * (panel % 2) as f64,
                25. + 57. * (panel / 2) as f64,
                86.,
                39.,
            ),
            axes,
        );
        for (profile, color) in [(&old, "#999999"), (&new, "#17608e")] {
            let points = (0..=459)
                .map(|i| {
                    let z = 200.05 + 5. * i as f64;
                    Ok([z, profile.sample(z)?[field]])
                })
                .collect::<Result<Vec<_>>>()?;
            p.push(Layer::Line(Line::new(points, color)));
        }
        fig.panels.push(p);
    }
    fig.labels.push(Label::new(
        "Grey: independent composition knots; blue: sigmoid H and Gaussian He",
        115.,
        198.,
    ));
    fig.labels.push(Label::new(
        "Same measured products and full covariance; conditional model comparison",
        115.,
        205.,
    ));
    for ext in ["png", "pdf", "svg"] {
        fig.save(format!("{}.{ext}", a[3]))?;
    }
    Ok(())
}
