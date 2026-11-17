use anyhow::Result;
use rustplotlib::*;
fn main() -> Result<()> {
    let a: Vec<_> = std::env::args().collect();
    let h = hdf5::File::open(&a[1])?;
    let time = h.dataset("time_utc_s")?.read_raw::<f64>()?;
    let first = chrono::DateTime::from_timestamp(time[0] as i64, 0).unwrap();
    let last = chrono::DateTime::from_timestamp((time[time.len() - 1] + 0.04) as i64, 0).unwrap();
    let minutes = time.len() as f64 * 0.04 / 60.;
    let full_r = h.dataset("slant_range_km")?.read_raw::<f64>()?;
    let l = h.dataset("lag_us")?.read_raw::<f64>()?;
    let full_acf = h.dataset("normalized_acf_real")?.read_raw::<f64>()?;
    let full_power = h.dataset("acf_real_adu2")?.read_raw::<f64>()?;
    let full_sem = h.dataset("acf_real_sem_adu2")?.read_raw::<f64>()?;
    let n = l.len();
    // Exclude receive-gate recovery: the longest lag must start after sample 29.
    let start = full_r.iter().position(|&r| r >= 300.).unwrap();
    let r = &full_r[start..];
    let acf = &full_acf[start * n..];
    let power = &full_power[start * n..];
    let sem = &full_sem[start * n..];
    let retained = h
        .dataset("block_retained_pulses")
        .ok()
        .map(|d| d.read_raw::<f64>().unwrap().iter().sum::<f64>())
        .unwrap_or(15000.);
    let upper = a.get(3).map(|s| s.parse::<f64>().unwrap()).unwrap_or(1500.);
    let dr = r[1] - r[0];
    let mut f = Figure::new(IntendedUse::ArticleWide);
    f.width_mm = 220.;
    f.height_mm = 166.;
    f.raster_dpi = 180.;
    f.labels.push(Label::new(
        if retained < time.len() as f64 {
            "SYISR: satellite-screened lag profile, 2 ms uncoded pulse"
        } else {
            "SYISR: unscreened lag profile, 2 ms uncoded pulse"
        },
        110.,
        5.,
    ));
    f.labels.push(Label::new(
        format!(
            "{:.0} minute window; {:.0} pulses retained; 10 microsecond lag spacing",
            minutes, retained
        ),
        110.,
        17.,
    ));
    f.labels.push(Label::new(
        format!(
            "{}--{} UTC ({}--{} Beijing time)",
            first.format("%d %B %Y, %H:%M"),
            last.format("%H:%M"),
            (first + chrono::Duration::hours(8)).format("%H:%M"),
            (last + chrono::Duration::hours(8)).format("%H:%M")
        ),
        110.,
        11.,
    ));
    let sc = f.scale(Scale::new(
        [-1., 1.],
        Normalization::Linear,
        ColorMap::diverging(),
    )?);
    let rend = r.iter().take_while(|&&v| v < upper).count();
    let edges: Vec<_> = (0..=rend).map(|i| r[0] - dr / 2. + i as f64 * dr).collect();
    let le: Vec<_> = (0..=n).map(|i| -5. + i as f64 * 10.).collect();
    let mut ax = Axes::new(
        [0., 600.],
        [edges[0], *edges.last().unwrap()],
        r"Lag ($\mu$s)",
        "Slant range (km)",
    );
    ax.xticks = vec![0., 100., 200., 300., 400., 500., 600.];
    ax.yticks = [300., 500., 700., 1000., 1250., 1500., 2000., 2500.]
        .into_iter()
        .filter(|&v| v <= upper)
        .collect();
    let mut p = Panel::new(Rect::new(23., 23., 130., 103.), ax);
    p.push(Layer::Pcolormesh(Field::new(
        le,
        edges,
        acf[..rend * n].to_vec(),
        sc,
    )?));
    f.panels.push(p);
    let mut bar = Colorbar::new(
        Rect::new(23., 140., 130., 2.5),
        sc,
        r"$\mathrm{Re}\,R(\tau)/\mathrm{Re}\,R(0)$",
        vec![-1., -0.5, 0., 0.5, 1.],
    );
    bar.horizontal = true;
    f.colorbars.push(bar);
    let mx = (0..rend).map(|i| power[i * n]).fold(0f64, f64::max);
    let mut ax = Axes::new(
        [-0.05, 1.05],
        [r[0] - dr / 2., r[rend - 1] + dr / 2.],
        "Relative power",
        "",
    );
    ax.xticks = vec![0., 0.5, 1.];
    ax.yticks = vec![];
    let mut p = Panel::new(Rect::new(168., 23., 40., 103.), ax);
    p.push(Layer::Line(Line::new(
        (0..rend).map(|i| [power[i * n] / mx, r[i]]).collect(),
        "#144c78",
    )));
    for sign in [-1., 1.] {
        p.push(Layer::Line(Line::new(
            (0..rend)
                .map(|i| [(power[i * n] + sign * sem[i * n]) / mx, r[i]])
                .collect(),
            "#b9cbd7",
        )));
    }
    f.panels.push(p);
    f.labels.push(Label::new("Zero lag", 188., 18.));
    f.labels.push(Label::new(
        "Background ACF subtracted at every lag; common 330 km range extent",
        110.,
        162.,
    ));
    for ext in ["png", "pdf", "svg"] {
        f.save(format!("{}.{}", a[2], ext))?;
    }
    Ok(())
}
