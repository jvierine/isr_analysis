//! Highest complete receive-gate background ACF, with retained-pulse weighting.
//! This is a noise autocorrelation measurement, not a voltage impulse response.
use anyhow::Result;
use rustplotlib::*;
fn main() -> Result<()> {
    let a: Vec<_> = std::env::args().collect();
    let h = hdf5::File::open(&a[1])?;
    let get = |s: &str| h.dataset(s).unwrap().read_raw::<f64>().unwrap();
    let lag = get("lag_us");
    let n = lag.len();
    let weights = get("block_retained_pulses");
    let nb = weights.len();
    let total: f64 = weights.iter().sum();
    let w: Vec<_> = weights.iter().map(|v| v / total).collect();
    let re = get("background_acf_real_adu2");
    let im = get("background_acf_imag_adu2");
    let average = |v: &[f64]| {
        (0..n)
            .map(|l| {
                (0..nb)
                    .filter(|&b| w[b] > 0.)
                    .map(|b| w[b] * v[b * n + l])
                    .sum::<f64>()
            })
            .collect::<Vec<_>>()
    };
    let real = average(&re);
    let imag = average(&im);
    let zero = real[0];
    let sem = |v: &[f64], mean: &[f64]| {
        (0..n)
            .map(|l| {
                let leave: Vec<_> = (0..nb)
                    .filter(|&b| w[b] > 0.)
                    .map(|b| (mean[l] - w[b] * v[b * n + l]) / (1. - w[b]))
                    .collect();
                let count = leave.len();
                let center = leave.iter().sum::<f64>() / count as f64;
                (leave.iter().map(|v| (v - center).powi(2)).sum::<f64>() * (count - 1) as f64
                    / count as f64)
                    .sqrt()
            })
            .collect::<Vec<_>>()
    };
    let sr = sem(&re, &real);
    let si = sem(&im, &imag);
    let metadata: serde_json::Value =
        serde_json::from_slice(&h.attr("metadata_json")?.read_raw::<u8>()?)?;
    let range = metadata["background_reference_slant_range_km"]
        .as_f64()
        .unwrap();
    let time = get("time_utc_s");
    let start = chrono::DateTime::from_timestamp(time[0] as i64, 0).unwrap();
    let end = chrono::DateTime::from_timestamp((time[time.len() - 1] + 0.04) as i64, 0).unwrap();
    let mut f = Figure::new(IntendedUse::ArticleWide);
    f.width_mm = 210.;
    f.height_mm = 174.;
    f.raster_dpi = 180.;
    f.labels.push(Label::new(
        format!(
            "SYISR highest complete range gate: {:.1} km slant range",
            range
        ),
        105.,
        5.,
    ));
    f.labels.push(Label::new(
        format!(
            "{}--{} UTC; {:.0} retained pulses",
            start.format("%d %B %Y, %H:%M"),
            end.format("%H:%M"),
            total
        ),
        105.,
        12.,
    ));
    for (row, xlim, ylim, yticks, title) in [
        (
            0,
            [0., 150.],
            [-0.05, 1.05],
            vec![0., 0.5, 1.],
            "Full normalized background ACF",
        ),
        (
            1,
            [10., 200.],
            [-0.045, 0.045],
            vec![-0.04, -0.02, 0., 0.02, 0.04],
            "Finite-lag response (expanded scale)",
        ),
    ] {
        let y = 30. + row as f64 * 70.;
        let mut axes = Axes::new(
            xlim,
            ylim,
            r"Lag ($\mu$s)",
            r"$R_N(\tau)/\mathrm{Re}\,R_N(0)$",
        );
        axes.xticks = if row == 0 {
            vec![0., 25., 50., 75., 100., 125., 150.]
        } else {
            vec![10., 50., 100., 150., 200.]
        };
        axes.yticks = yticks;
        let mut p = Panel::new(Rect::new(25., y, 176., 45.), axes);
        for (v, s, c) in [(&real, &sr, "#17608e"), (&imag, &si, "#bb4933")] {
            let ids: Vec<_> = (0..n)
                .filter(|&l| lag[l] >= xlim[0] && lag[l] <= xlim[1])
                .collect();
            let points: Vec<_> = ids.iter().map(|&l| [lag[l], v[l] / zero]).collect();
            let bounds: Vec<_> = ids
                .iter()
                .map(|&l| [(v[l] - s[l]) / zero, (v[l] + s[l]) / zero])
                .collect();
            p.errorbars(&points, None, Some(&bounds), c, 0.6, 2., 0.5)?;
            p.push(Layer::Line(Line::new(points, c)));
        }
        f.panels.push(p);
        f.labels.push(Label::new(title, 113., y - 6.));
    }
    for (t, c, x) in [
        ("Blue: real", "#17608e", 48.),
        ("Red: imaginary", "#bb4933", 102.),
        ("Bars: block jackknife SEM", "#555555", 166.),
    ] {
        let mut label = Label::new(t, x, 163.);
        label.color = c.into();
        f.labels.push(label);
    }
    for ext in ["png", "pdf", "svg"] {
        f.save(format!("{}.{}", a[2], ext))?;
    }
    let out = hdf5::File::create(format!("{}.h5", a[2]))?;
    for (name, data) in [
        ("lag_us", lag),
        ("background_real_adu2", real),
        ("background_imag_adu2", imag),
        ("background_real_sem_adu2", sr),
        ("background_imag_sem_adu2", si),
    ] {
        out.new_dataset_builder().with_data(&data).create(name)?;
    }
    out.new_attr::<f64>()
        .create("zero_lag_power_adu2")?
        .write_scalar(&zero)?;
    out.new_attr::<f64>()
        .create("slant_range_km")?
        .write_scalar(&range)?;
    Ok(())
}
