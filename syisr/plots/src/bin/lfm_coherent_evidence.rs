//! Conditional calibration evidence. Script provenance stays in source/HDF5 only.
use anyhow::Result;
use rustplotlib::*;
use std::path::Path;
fn v(h: &hdf5::Group, k: &str) -> Result<Vec<f64>> {
    Ok(h.dataset(k)?.read_raw::<f64>()?)
}
fn line(p: &mut Panel, pts: Vec<[f64; 2]>, color: &str, dash: bool) {
    let mut l = Line::new(pts, color);
    l.width_pt = 1.3;
    if dash {
        l.dash = Some("3 2".into());
    }
    p.push(Layer::Line(l));
}
fn main() -> Result<()> {
    let a: Vec<_> = std::env::args().collect();
    let root = Path::new(&a[1]);
    let out = Path::new(&a[2]);
    std::fs::create_dir_all(out)?;
    let h = hdf5::File::open(root.join("apparent_density.h5"))?;
    let g = h.group("beam_358.0_60.0")?;
    let time = v(&g, "time_unix_s")?;
    let z = v(&g, "altitude_km")?;
    let ne = v(&g, "apparent_Ne_m3")?;
    let model = v(&g, "IRI_Ne_m3")?;
    let ratio = v(&g, "median_mean_ratio")?;
    let snr = v(&g, "signal_noise_ratio")?;
    let x: Vec<_> = time.iter().map(|t| (t - 1772323200.) / 3600.).collect();
    let nx = x.len();
    let nz = z.len();
    let mut f = Figure::new(IntendedUse::ArticleWide);
    f.width_mm = 195.;
    f.height_mm = 153.;
    f.raster_dpi = 600.;
    f.font_pt = 8.;
    f.latex.cache = out.join("latex-cache");
    let palette = ColorMap::storm();
    let sc = f.scale(Scale::new([10., 17.], Normalization::Linear, palette)?);
    let mut ax = Axes::new(
        [300., 450.],
        [20., 60.],
        "Altitude (km)",
        "Calibration coefficient",
    );
    ax.xticks = vec![300., 350., 400., 450.];
    ax.yticks = vec![20., 30., 40., 50., 60.];
    let mut p = Panel::new(Rect::new(20., 8., 66., 55.), ax);
    let samples = v(&h, "zenith_calibration_samples")?;
    let tfirst = samples[0];
    for (group, c) in [(true, "#267782"), (false, "#a72860")] {
        line(
            &mut p,
            samples
                .chunks_exact(7)
                .filter(|q| (q[0] == tfirst) == group)
                .map(|q| [q[2], q[6]])
                .collect(),
            c,
            false,
        );
    }
    line(
        &mut p,
        vec![[300., 36.3693947], [450., 36.3693947]],
        "#111111",
        true,
    );
    p.label_at("(a)", [308., 57.]);
    f.panels.push(p);
    let mut ax = Axes::new([11.75, 13.75], [200., 650.], "UTC", "Altitude (km)");
    ax.xticks = vec![12., 12.5, 13., 13.5];
    ax.xticklabels = Some(vec![
        "12:00".into(),
        "12:30".into(),
        "13:00".into(),
        "13:30".into(),
    ]);
    ax.yticks = vec![200., 300., 400., 500., 600.];
    let mut p = Panel::new(Rect::new(111., 8., 63., 55.), ax);
    let xe: Vec<_> = (0..=nx)
        .map(|j| x[0] - 1. / 120. + j as f64 / 60.)
        .collect();
    let ze: Vec<_> = (0..=nz)
        .map(|j| {
            if j == 0 {
                z[0] - (z[1] - z[0]) / 2.
            } else if j == nz {
                z[nz - 1] + (z[nz - 1] - z[nz - 2]) / 2.
            } else {
                (z[j - 1] + z[j]) / 2.
            }
        })
        .collect();
    let vals = (0..nz)
        .flat_map(|k| (0..nx).map(move |j| (j, k)))
        .map(|(j, k)| {
            let q = ne[j * nz + k];
            if q > 0. { q.log10().max(10.) } else { 10. }
        })
        .collect();
    p.pcolormesh(xe, ze, vals, sc)?;
    p.label_at("(b)", [11.85, 620.]);
    f.panels.push(p);
    f.colorbars.push(Colorbar::new(
        Rect::new(177., 8., 2., 55.),
        sc,
        "$\\log_{10} N_{e,app}$ (m$^{-3}$)",
        vec![10., 12., 14., 16.],
    ));
    let mut ax = Axes::new(
        [11.75, 13.75],
        [11.5, 17.5],
        "UTC",
        "$\\log_{10} N_{e,app}$ (m$^{-3}$)",
    );
    ax.xticks = vec![12., 12.5, 13., 13.5];
    ax.xticklabels = Some(vec![
        "12:00".into(),
        "12:30".into(),
        "13:00".into(),
        "13:30".into(),
    ]);
    ax.yticks = vec![12., 13., 14., 15., 16., 17.];
    let mut p = Panel::new(Rect::new(20., 85., 66., 51.), ax);
    let peaks: Vec<_> = (0..nx)
        .map(|j| {
            let q = (0..nz)
                .filter(|k| {
                    z[*k] >= 250.
                        && z[*k] <= 650.
                        && ratio[j * nz + *k] > 0.5
                        && snr[j * nz + *k] > 100.
                })
                .map(|k| ne[j * nz + k])
                .fold(0., f64::max);
            if q > 0. { q.log10() } else { f64::NAN }
        })
        .collect();
    line(
        &mut p,
        x.iter().zip(&peaks).map(|(a, b)| [*a, *b]).collect(),
        "#a72860",
        false,
    );
    line(
        &mut p,
        x.iter().zip(&peaks).map(|(a, b)| [*a, *b - 1.]).collect(),
        "#a72860",
        true,
    );
    let crit = h
        .dataset("critical_plasma_density_at432MHz_m3")?
        .read_scalar::<f64>()?
        .log10();
    line(&mut p, vec![[11.75, crit], [13.75, crit]], "#111111", true);
    line(
        &mut p,
        (0..nx)
            .map(|j| {
                [
                    x[j],
                    model[j * nz..(j + 1) * nz]
                        .iter()
                        .copied()
                        .fold(0., f64::max)
                        .log10(),
                ]
            })
            .collect(),
        "#267782",
        false,
    );
    p.label_at("(c)", [11.86, 17.1]);
    f.panels.push(p);
    let mut ax = Axes::new([11.75, 13.75], [0.8, 1.2], "UTC", "Noise / quiet noise");
    ax.xticks = vec![12., 12.5, 13., 13.5];
    ax.xticklabels = Some(vec![
        "12:00".into(),
        "12:30".into(),
        "13:00".into(),
        "13:30".into(),
    ]);
    ax.yticks = vec![0.8, 0.9, 1., 1.1, 1.2];
    let mut p = Panel::new(Rect::new(111., 85., 63., 51.), ax);
    for (name, c) in [
        ("beam_358.0_60.0", "#a72860"),
        ("beam_358.0_90.0", "#267782"),
    ] {
        let q = hdf5::File::open(root.join("mean_power").join(format!("{name}.h5")))?;
        let n = v(&q, "noise_power")?;
        let mut quiet: Vec<_> = n
            .iter()
            .zip(&time)
            .filter(|(_, t)| **t < 1772364600.)
            .map(|(n, _)| *n)
            .collect();
        quiet.sort_by(f64::total_cmp);
        let base = quiet[quiet.len() / 2];
        line(
            &mut p,
            x.iter().zip(n).map(|(t, n)| [*t, n / base]).collect(),
            c,
            false,
        );
    }
    line(&mut p, vec![[11.75, 1.], [13.75, 1.]], "#777777", true);
    p.label_at("(d)", [11.86, 1.17]);
    f.panels.push(p);
    f.save(out.join("coherent_evidence.pdf"))?;
    f.save(out.join("coherent_evidence.png"))?;
    Ok(())
}
