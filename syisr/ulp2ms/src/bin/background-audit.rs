//! Read-only audit of subtraction and common-reference errors; no new background
//! model is applied to the scientific ACF product.
use anyhow::Result;
use rustplotlib::*;
fn main() -> Result<()> {
    let a: Vec<_> = std::env::args().collect();
    let h = hdf5::File::open(&a[1])?;
    let get = |name: &str| h.dataset(name).unwrap().read_raw::<f64>().unwrap();
    let r = get("slant_range_km");
    let lag = get("lag_us");
    let n = lag.len();
    let nr = r.len();
    let block = get("block_acf_real_adu2");
    let bg = get("background_acf_real_adu2");
    let mut w = get("block_retained_pulses");
    let total: f64 = w.iter().sum();
    w.iter_mut().for_each(|v| *v /= total);
    let reference: Vec<f64> = (0..n)
        .map(|l| {
            (0..w.len())
                .filter(|&b| w[b] > 0.)
                .map(|b| w[b] * bg[b * n + l])
                .sum()
        })
        .collect();
    let power = get("acf_before_background_real_adu2");
    let quiet: Vec<usize> = (0..nr).filter(|&i| r[i] >= 3000. && r[i] < 5000.).collect();
    let quiet_acf: Vec<f64> = (0..n)
        .map(|l| {
            (0..w.len())
                .filter(|&b| w[b] > 0.)
                .map(|b| {
                    w[b] * (quiet
                        .iter()
                        .map(|&i| {
                            block[(b * nr + i) * n + l] * (200 - l) as f64 / 200. + bg[b * n + l]
                        })
                        .sum::<f64>()
                        / quiet.len() as f64)
                })
                .sum()
        })
        .collect();
    let mut f = Figure::new(IntendedUse::ArticleWide);
    f.width_mm = 210.;
    f.height_mm = 171.;
    f.raster_dpi = 180.;
    f.labels.push(Label::new(
        "SYISR background audit: subtraction is not uniform across range",
        105.,
        5.,
    ));
    f.labels.push(Label::new(
        "Measured raw background power and correlation; screened integration",
        105.,
        11.,
    ));
    let mut ax = Axes::new(
        [1600., 5750.],
        [7050., 7500.],
        "Slant range (km)",
        r"Raw zero-lag power (ADU$^2$)",
    );
    ax.xticks = vec![2000., 3000., 4000., 5000.];
    ax.yticks = vec![7100., 7200., 7300., 7400., 7500.];
    let mut p = Panel::new(Rect::new(25., 22., 175., 48.), ax);
    p.push(Layer::Line(Line::new(
        (0..nr)
            .filter(|&i| r[i] >= 1600.)
            .map(|i| [r[i], power[i * n]])
            .collect(),
        "#145981",
    )));
    p.push(Layer::Line(Line::new(
        vec![[1600., reference[0]], [5750., reference[0]]],
        "#b14b31",
    )));
    f.panels.push(p);
    for (t, c, x) in [
        ("Blue: measured range profile", "#145981", 70.),
        ("Red: subtracted high-range reference", "#b14b31", 160.),
    ] {
        let mut label = Label::new(t, x, 86.);
        label.color = c.into();
        f.labels.push(label);
    }
    let mut ax = Axes::new(
        [100., 600.],
        [-0.4, 0.4],
        r"Lag ($\mu$s)",
        r"Real ACF / reference power (\%)",
    );
    ax.xticks = vec![100., 200., 300., 400., 500., 600.];
    ax.yticks = vec![-0.4, -0.2, 0., 0.2, 0.4];
    let mut p = Panel::new(Rect::new(25., 101., 175., 43.), ax);
    for (series, c) in [(&reference, "#be8b27"), (&quiet_acf, "#145981")] {
        p.push(Layer::Line(Line::new(
            (10..n)
                .map(|l| [lag[l], 100. * series[l] / reference[0]])
                .collect(),
            c,
        )));
    }
    let difference: Vec<_> = (10..n)
        .map(|l| [lag[l], 100. * (quiet_acf[l] - reference[l]) / reference[0]])
        .collect();
    p.push(Layer::Line(Line::new(difference, "#b14b31")));
    f.panels.push(p);
    for (t, c, x) in [
        ("Gold: highest gate", "#be8b27", 50.),
        ("Blue: 3000--5000 km average", "#145981", 111.),
        ("Red: their difference", "#b14b31", 174.),
    ] {
        let mut label = Label::new(t, x, 162.);
        label.color = c.into();
        f.labels.push(label);
    }
    for ext in ["png", "pdf"] {
        f.save(format!("{}.{}", a[2], ext))?;
    }
    let out = hdf5::File::create(format!("{}.h5", a[2]))?;
    for (name, data) in [
        ("lag_us", lag),
        ("reference_real_adu2", reference),
        ("quiet_band_real_adu2", quiet_acf),
        ("slant_range_km", r),
        ("raw_zero_lag_adu2", (0..nr).map(|i| power[i * n]).collect()),
    ] {
        out.new_dataset_builder().with_data(&data).create(name)?;
    }
    Ok(())
}
