use anyhow::Result;
use rustplotlib::*;
fn main() -> Result<()> {
    let a: Vec<_> = std::env::args().collect();
    let h = hdf5::File::open(&a[1])?;
    let get = |s: &str| h.dataset(s).unwrap().read_raw::<f64>().unwrap();
    let lag = get("lag_us");
    let n = lag.len();
    let bg = get("background_acf_real_adu2");
    let bi = get("background_acf_imag_adu2");
    let alt = get("alternate_background_acf_real_adu2");
    let weights = get("block_retained_pulses");
    let total: f64 = weights.iter().sum();
    let avg = |v: &[f64], l: usize| {
        (0..weights.len())
            .filter(|&b| weights[b] > 0.)
            .map(|b| weights[b] * v[b * n + l])
            .sum::<f64>()
            / total
    };
    let z = avg(&bg, 0);
    let mut f = Figure::new(IntendedUse::ArticleWide);
    f.width_mm = 210.;
    f.height_mm = 175.;
    f.raster_dpi = 180.;
    f.labels.push(Label::new(
        "SYISR: background and range-response diagnostics",
        105.,
        5.,
    ));
    let mut ax = Axes::new(
        [0., 150.],
        [-0.2, 1.1],
        r"Lag ($\mu$s)",
        "Background / zero lag",
    );
    ax.xticks = vec![0., 50., 100., 150.];
    ax.yticks = vec![0., 0.5, 1.];
    let mut p = Panel::new(Rect::new(22., 18., 76., 48.), ax);
    for (v, c) in [(&bg, "#144c78"), (&bi, "#ad3e36"), (&alt, "#e5a62b")] {
        p.push(Layer::Line(Line::new(
            lag.iter()
                .enumerate()
                .map(|(l, &x)| [x, avg(v, l) / z])
                .collect(),
            c,
        )));
    }
    f.panels.push(p);
    for (t, c, y) in [
        ("Highest gate: real", "#144c78", 83.),
        ("Highest gate: imaginary", "#ad3e36", 89.),
        ("Second reference: real", "#e5a62b", 95.),
    ] {
        let mut label = Label::new(t, 60., y);
        label.color = c.into();
        f.labels.push(label);
    }
    let offsets = get("range_ambiguity_offset_km");
    let k = get("range_ambiguity_weights");
    let nk = offsets.len();
    let mut ax = Axes::new(
        [-180., 180.],
        [0., 0.008],
        "Range offset (km)",
        "Unit-area weight",
    );
    ax.xticks = vec![-150., 0., 150.];
    ax.yticks = vec![0., 0.004, 0.008];
    let mut p = Panel::new(Rect::new(125., 18., 76., 48.), ax);
    for (l, c) in [(0, "#144c78"), (30, "#e5a62b"), (60, "#ad3e36")] {
        p.push(Layer::Line(Line::new(
            offsets
                .iter()
                .enumerate()
                .map(|(i, &x)| [x, k[l * nk + i]])
                .collect(),
            c,
        )));
    }
    f.panels.push(p);
    f.labels
        .push(Label::new("0, 300 and 600 microsecond lags", 163., 83.));
    f.labels
        .push(Label::new("Common extent; different weighting", 163., 89.));
    let r = get("slant_range_km");
    let acf = get("normalized_acf_real");
    let mut ax = Axes::new(
        [0., 600.],
        [-0.6, 1.1],
        r"Lag ($\mu$s)",
        "Normalized real ACF",
    );
    ax.xticks = vec![0., 100., 200., 300., 400., 500., 600.];
    ax.yticks = vec![-0.5, 0., 0.5, 1.];
    let mut p = Panel::new(Rect::new(22., 118., 179., 37.), ax);
    for (rr, c) in [
        (400., "#144c78"),
        (700., "#e5a62b"),
        (1000., "#ad3e36"),
        (1500., "#6f4c91"),
    ] {
        let i = (0..r.len())
            .min_by(|&i, &j| (r[i] - rr).abs().total_cmp(&(r[j] - rr).abs()))
            .unwrap();
        p.push(Layer::Line(Line::new(
            lag.iter()
                .enumerate()
                .map(|(l, &x)| [x, acf[i * n + l]])
                .collect(),
            c,
        )));
    }
    f.panels.push(p);
    f.labels.push(Label::new(
        "Slant ranges: 400 km (blue), 700 km (gold), 1000 km (red), 1500 km (purple)",
        110.,
        110.,
    ));
    for ext in ["png", "pdf"] {
        f.save(format!("{}.{}", a[2], ext))?;
    }
    Ok(())
}
