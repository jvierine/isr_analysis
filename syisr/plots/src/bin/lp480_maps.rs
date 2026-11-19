//! Plasma/composition maps and signed background-subtracted range-Doppler frames.
use anyhow::{ensure, Result};
use chrono::{DateTime, Utc};
use rustplotlib::*;
use std::{collections::BTreeSet, path::Path};
use syisr_ac::{
    campaign,
    lp480::{self, Decoded, Manifest},
    lp480_fit::{Background, WindowFit},
};
fn main() -> Result<()> {
    let a: Vec<_> = std::env::args().collect();
    ensure!(a.len() == 4, "maps|video ROOT OUTPUT");
    let root = Path::new(&a[2]);
    let out = Path::new(&a[3]);
    std::fs::create_dir_all(out)?;
    if a[1] == "maps" {
        maps(root, out)
    } else {
        video(root, out)
    }
}
fn maps(root: &Path, out: &Path) -> Result<()> {
    let mut windows = vec![];
    for p in std::fs::read_dir(root)? {
        let p = p?.path();
        if p.extension().is_some_and(|x| x == "h5") && !p.to_string_lossy().contains("checkpoint") {
            if let Ok(w) = campaign::load::<WindowFit>(&p) {
                windows.push(w);
            }
        }
    }
    windows.sort_by_key(|w| w.start_utc_s);
    ensure!(!windows.is_empty(), "no completed fit windows");
    let times: Vec<_> = windows.iter().map(|w| w.start_utc_s).collect();
    let first = times[0];
    let day = (first + 28800).div_euclid(86400) * 86400;
    let x0 = (first + 28800 - day) as f64 / 3600.;
    let nt = ((times[times.len() - 1] - first) / 1800 + 1) as usize;
    let x: Vec<_> = (0..=nt).map(|i| x0 + i as f64 * 0.5).collect();
    let z: Vec<_> = (0..=261).map(|i| 97.5 + i as f64 * 5.).collect();
    let mut beams: BTreeSet<_> = windows
        .iter()
        .flat_map(|w| {
            w.beams.iter().map(|b| {
                (
                    (b.azimuth_deg * 100.).round() as i64,
                    (b.elevation_deg * 100.).round() as i64,
                )
            })
        })
        .collect();
    for w in &windows {
        if let Some(list) = w.metadata["unsupported_beams"].as_array() {
            for b in list {
                beams.insert((
                    (b["azimuth_deg"].as_f64().unwrap() * 100.).round() as i64,
                    (b["elevation_deg"].as_f64().unwrap() * 100.).round() as i64,
                ));
            }
        }
    }
    let first_date = DateTime::<Utc>::from_timestamp(first + 28800, 0).unwrap();
    let last_date =
        DateTime::<Utc>::from_timestamp(times[times.len() - 1] + 1800 + 28800, 0).unwrap();
    let date = if first_date.date_naive() == last_date.date_naive() {
        first_date.format("%d %B %Y").to_string()
    } else {
        format!(
            "{}--{}",
            first_date.format("%d %B"),
            last_date.format("%d %B %Y")
        )
    };
    let strict = windows.iter().filter(|w| w.status == "converged").count();
    let stable = windows
        .iter()
        .filter(|w| w.status == "data_stable_gradient_pending")
        .count();
    let unresolved = windows.len() - strict - stable;
    for (az, el) in beams {
        let mut f = Figure::new(IntendedUse::ArticleWide);
        f.width_mm = 205.;
        f.height_mm = 270.;
        f.font_pt = 10.;
        f.raster_dpi = 220.;
        f.latex.cache = out.join("latex-cache");
        f.labels.push(Label::new(
            format!(
                "SYISR 480 $\\mu$s; azimuth {:.2}$^\\circ$, elevation {:.2}$^\\circ$",
                az as f64 / 100.,
                el as f64 / 100.
            ),
            103.,
            5.,
        ));
        f.labels.push(Label::new(
            format!("{date}; 30-minute windows; shared H$^+$, He$^+=0$"),
            103.,
            12.,
        ));
        for q in 0..5 {
            let mut values = vec![f64::NAN; nt * 261];
            for w in &windows {
                let t = ((w.start_utc_s - first) / 1800) as usize;
                if let Some(b) = w.beams.iter().find(|b| {
                    (b.azimuth_deg * 100.).round() as i64 == az
                        && (b.elevation_deg * 100.).round() as i64 == el
                }) {
                    let v = match q {
                        0 => &b.ne_m3,
                        1 => &b.ti_k,
                        2 => &b.te_k,
                        3 => &b.velocity_m_s,
                        _ => &b.h_fraction,
                    };
                    for (zi, a) in v.iter().enumerate() {
                        values[zi * nt + t] = *a;
                    }
                }
            }
            let scale = f.scale(Scale::new(
                [
                    [1e9, 1e13],
                    [150., 3000.],
                    [150., 6000.],
                    [-100., 100.],
                    [0., 1.],
                ][q],
                if q == 0 {
                    Normalization::Log
                } else {
                    Normalization::Linear
                },
                if q == 3 {
                    ColorMap::diverging()
                } else {
                    ColorMap::turbo()
                },
            )?);
            let yy = 22. + q as f64 * 45.;
            let mut ax = Axes::new(
                [x[0], *x.last().unwrap()],
                [100., 1400.],
                if q == 4 {
                    "China local time (UTC+8)"
                } else {
                    ""
                },
                "Altitude (km)",
            );
            ax.yticks = vec![100., 400., 700., 1000., 1400.];
            ax.xticks = ((x[0] / 4.).ceil() as i64..=(*x.last().unwrap() / 4.).floor() as i64)
                .map(|i| i as f64 * 4.)
                .collect();
            ax.xticklabels = Some(
                ax.xticks
                    .iter()
                    .map(|v| format!("{:02}:00", (*v as i64).rem_euclid(24)))
                    .collect(),
            );
            if q < 4 {
                ax.show_x_axis = false;
            }
            let mut p = Panel::new(Rect::new(20., yy, 150., 35.), ax);
            p.pcolormesh(x.clone(), z.clone(), values, scale)?;
            f.panels.push(p);
            f.colorbars.push(Colorbar::new(
                Rect::new(175., yy, 3., 35.),
                scale,
                [
                    r"$N_e$ (m$^{-3}$)",
                    r"$T_i$ (K)",
                    r"$T_e$ (K)",
                    r"$v_i$ (m/s)",
                    r"H$^+$ fraction",
                ][q],
                match q {
                    0 => vec![],
                    1 => vec![150., 1000., 2000., 3000.],
                    2 => vec![150., 2000., 4000., 6000.],
                    3 => vec![-100., 0., 100.],
                    _ => vec![0., 0.5, 1.],
                },
            ));
            if q == 0 {
                for exponent in 9..=13 {
                    f.labels.push(Label::new(
                        format!("$10^{{{exponent}}}$"),
                        184.,
                        yy + 35. * (1. - (exponent - 9) as f64 / 4.),
                    ));
                }
            }
        }
        f.labels.push(Label::new(
            format!("IRI calibration; {strict} converged, {stable} data-stable, {unresolved} unresolved; no uncertainties"),
            103.,
            261.,
        ));
        if !windows.iter().any(|w| {
            w.beams.iter().any(|b| {
                (b.azimuth_deg * 100.).round() as i64 == az
                    && (b.elevation_deg * 100.).round() as i64 == el
            })
        }) {
            f.labels.push(Label::new("Unfitted: near-perpendicular magnetic aspect; current collisionless model unsupported",103.,254.));
        }
        let name = format!("beam-az{az}-el{el}");
        f.save(out.join(format!("{name}.png")))?;
        f.save(out.join(format!("{name}.pdf")))?;
    }
    Ok(())
}
fn video(root: &Path, out: &Path) -> Result<()> {
    let m: Manifest = campaign::load(&root.join("index.h5"))?;
    let bg: Background = campaign::load(&root.join("background.h5"))?;
    let times: BTreeSet<_> = m.jobs.iter().map(|j| j.start).collect();
    let beams: BTreeSet<_> = m
        .jobs
        .iter()
        .map(|j| ((j.az * 100.).round() as i64, (j.el * 100.).round() as i64))
        .collect();
    let beams: Vec<_> = beams.into_iter().collect();
    let nf = syisr_ac::lp480_screen::NF;
    let nw = (lp480::NS - syisr_ac::lp480_screen::START - nf) / syisr_ac::lp480_screen::HOP + 1;
    let hann: Vec<_> = (0..nf)
        .map(|i| 0.5 - 0.5 * (std::f64::consts::TAU * i as f64 / (nf - 1) as f64).cos())
        .collect();
    let energy = hann.iter().map(|v| v * v).sum::<f64>();
    let frequency: Vec<_> = (0..nf)
        .map(|i| (i as f64 - nf as f64 / 2.) * 100000. / nf as f64)
        .collect();
    let mut noise = vec![0.; nf];
    for (i, fr) in frequency.iter().enumerate() {
        noise[i] = bg.acf[0][0];
        for l in 1..=lp480::NL {
            let w = (0..nf - l).map(|j| hann[j] * hann[j + l]).sum::<f64>() / energy;
            let v = num_complex::Complex64::new(bg.acf[l][0], bg.acf[l][1])
                * num_complex::Complex64::from_polar(
                    1.,
                    std::f64::consts::TAU * fr * l as f64 * lp480::DT,
                );
            noise[i] += 2. * w * v.re;
        }
    }
    let audit = hdf5::File::create(out.join("range-doppler.h5"))?;
    audit
        .new_dataset_builder()
        .with_data(&frequency)
        .create("frequency_hz")?;
    audit
        .new_dataset_builder()
        .with_data(&noise)
        .create("background_power")?;
    for (fi, time) in times.iter().enumerate() {
        let mut f = Figure::new(IntendedUse::ArticleWide);
        f.width_mm = 300.;
        f.height_mm = 270.;
        f.font_pt = 9.;
        f.raster_dpi = 130.;
        f.latex.cache = out.join("latex-cache");
        let t = DateTime::<Utc>::from_timestamp(*time + 28800, 0).unwrap();
        f.labels.push(Label::new(
            format!(
                "SYISR 480 $\\mu$s; {} China local time (UTC+8)",
                t.format("%d %B %Y, %H:%M")
            ),
            150.,
            5.,
        ));
        let scale = f.scale(Scale::new(
            [-0.02, 0.12],
            Normalization::Asinh { width: 0.003 },
            ColorMap::inferno(),
        )?);
        let group = audit.create_group(&time.to_string())?;
        for (bi, (az, el)) in beams.iter().enumerate() {
            let row = bi / 3;
            let col = bi % 3;
            let rect = Rect::new(17. + col as f64 * 96., 20. + row as f64 * 57., 78., 40.);
            let mut ax = Axes::new(
                [-25., 25.],
                [100., 2200.],
                if row == 3 {
                    "Frequency offset (kHz)"
                } else {
                    ""
                },
                if col == 0 { "Slant range (km)" } else { "" },
            );
            ax.xticks = vec![-20., 0., 20.];
            ax.yticks = vec![500., 1000., 1500., 2000.];
            let mut p = Panel::new(rect, ax);
            if let Some(job) = m.jobs.iter().find(|j| {
                j.start == *time
                    && (j.az * 100.).round() as i64 == *az
                    && (j.el * 100.).round() as i64 == *el
            }) {
                let d: Decoded =
                    campaign::load(&root.join("products").join(format!("{}.h5", job.id)))?;
                let mut raw = vec![0.; nw * nf];
                let mut n = 0.;
                for b in &d.blocks {
                    if b.range_doppler_raw.len() == raw.len() {
                        let count = b.range_doppler_pulses as f64;
                        n += count;
                        for (j, v) in raw.iter_mut().enumerate() {
                            *v += count * b.range_doppler_raw[j];
                        }
                    }
                }
                if let Some(a) = &d.averaged {
                    raw = a.range_doppler_raw.clone();
                    n = a.range_doppler_pulses as f64;
                    for v in &mut raw {
                        *v *= n;
                    }
                }
                if n > 0. {
                    raw.iter_mut().for_each(|v| *v /= n);
                    let mut signed = vec![0.; raw.len()];
                    for w in 0..nw {
                        let band = raw[w * nf..(w + 1) * nf].iter().sum::<f64>();
                        for j in 0..nf {
                            signed[w * nf + j] = if band > 0. {
                                (raw[w * nf + j] - noise[j]) / band
                            } else {
                                f64::NAN
                            };
                        }
                    }
                    let rr: Vec<_> = (0..=nw)
                        .map(|w| {
                            (job.front_us * 1e-6
                                + (syisr_ac::lp480_screen::START as f64
                                    + (w as f64 - 0.5) * syisr_ac::lp480_screen::HOP as f64
                                    + (nf as f64 - 1.) / 2.)
                                    * lp480::DT
                                - lp480::PW / 2.)
                                * lp480::C_KM
                                / 2.
                        })
                        .collect();
                    let xx: Vec<_> = (0..=nf)
                        .map(|j| (-0.5 * nf as f64 + j as f64 - 0.5) * 100. / nf as f64)
                        .collect();
                    p.pcolormesh(xx, rr.clone(), signed.clone(), scale)?;
                    let g = group.create_group(&format!("az{az}-el{el}"))?;
                    g.new_dataset::<f64>()
                        .shape((nw, nf))
                        .deflate(4)
                        .create("raw_power")?
                        .write_raw(&raw)?;
                    g.new_dataset::<f64>()
                        .shape((nw, nf))
                        .deflate(4)
                        .create("signed_power_over_raw_band_power")?
                        .write_raw(&signed)?;
                    g.new_dataset_builder()
                        .with_data(&rr)
                        .create("range_edges_km")?;
                }
            }
            f.panels.push(p);
            f.labels.push(Label::new(
                format!(
                    "Az {:.0}$^\\circ$, El {:.0}$^\\circ$",
                    *az as f64 / 100.,
                    *el as f64 / 100.
                ),
                rect.x + rect.width / 2.,
                rect.y - 3.,
            ));
        }
        f.colorbars.push(Colorbar::new(
            Rect::new(115., 250., 110., 3.),
            scale,
            "Signed power / raw band power",
            vec![-0.02, 0., 0.01, 0.03, 0.12],
        ));
        f.colorbars.last_mut().unwrap().horizontal = true;
        f.save(out.join(format!("frame-{fi:04}.png")))?;
    }
    Ok(())
}
