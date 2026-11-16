//! Numerical audit and explanatory figures for the range-lag ambiguity proof.
//! Provenance is confined to this source and numerical-product metadata.
use anyhow::{ensure, Result};
use num_complex::Complex64 as C;
use rustplotlib::*;
use syisr_long_pulse::{
    ambiguity_audit::{geometry_moments, replay},
    profile_response::DR,
};
fn main() -> Result<()> {
    let a = std::env::args().collect::<Vec<_>>();
    ensure!(
        a.len() == 3,
        "ambiguity-proof reference-profile.h5 output-prefix"
    );
    let source = hdf5::File::open(&a[1])?;
    let get = |n: &str| source.dataset(n).unwrap().read_raw::<f64>().unwrap();
    let offsets = get("receiver_fir_offset_s");
    let re = get("receiver_fir_real");
    let im = get("receiver_fir_imag");
    let fir = offsets
        .iter()
        .enumerate()
        .map(|(j, &t)| (t, C::new(re[j], im[j])))
        .collect::<Vec<_>>();
    let simple = [
        (-1e-5, C::new(0.2, 0.1)),
        (0., C::new(0.7, 0.)),
        (1e-5, C::new(0.2, -0.1)),
    ];
    let simple_result = replay(&simple, &[1, 17, 60], 24000)?;
    let inferred_result = replay(&fir, &[1, 17, 60], 24000)?;
    let max_simple = simple_result
        .iter()
        .map(|r| r.relative_error)
        .fold(0., f64::max);
    let max_inferred = inferred_result
        .iter()
        .map(|r| r.relative_error)
        .fold(0., f64::max);
    ensure!(
        max_simple < 8e-4 && max_inferred < 8e-4,
        "physical-time replay mismatch"
    );
    let report = serde_json::json!({"simple_complex_fir":{"max_relative_error":max_simple,"cases":simple_result},"inferred_noon_fir":{"max_relative_error":max_inferred,"cases":inferred_result},"direct_quadrature_points_per_cell":24000,"scene":{"edges_slant_range_km":[225.4,451.7,668.3,977.9,1250.6],"relative_power":[0.2,2.1,0.6,1.3],"frequency_hz":[-8700.,1900.,7300.,-3200.]},"assumptions":"rectangular 2 ms TX, same supplied FIR, half-sample timing convention, independent spatial populations; this is a computational check, not instrument calibration","provenance":{"reference_profile":a[1]}});
    std::fs::write(
        format!("{}-audit.json", a[2]),
        serde_json::to_vec_pretty(&report)?,
    )?;
    let h = hdf5::File::create(format!("{}.h5", a[2]))?;
    let lags = (0..61).collect::<Vec<_>>();
    let ranges = [400., 600., 1000.];
    h.new_dataset_builder()
        .with_data(&lags.iter().map(|&l| l as f64 * 10.).collect::<Vec<_>>())
        .create("lag_us")?;
    h.new_dataset_builder()
        .with_data(&ranges)
        .create("centre_slant_range_km")?;
    let moments = ranges
        .iter()
        .map(|&r| geometry_moments(r, &lags))
        .collect::<Vec<_>>();
    for (name, part) in [
        ("gain_relative_to_centre", 0),
        ("geometric_centroid_km", 1),
        ("normalized_acf_geometry_ratio", 2),
    ] {
        let values = moments
            .iter()
            .flat_map(|r| {
                r.iter().map(|v| match part {
                    0 => v.0,
                    1 => v.1,
                    _ => v.2,
                })
            })
            .collect::<Vec<_>>();
        h.new_dataset::<f64>()
            .shape((3, 61))
            .create(name)?
            .write_raw(&values)?;
    }
    h.new_attr_builder()
        .with_data(serde_json::to_string(&report)?.as_bytes())
        .create("metadata_json")?;
    let mut f = Figure::new(IntendedUse::ArticleWide);
    f.width_mm = 230.;
    f.height_mm = 185.;
    f.raster_dpi = 180.;
    f.labels.push(Label::new(
        "Range weighting changes both lag response and effective height",
        115.,
        5.,
    ));
    let colours = ["#17608e", "#d28126", "#af3f46"];
    for col in 0..2 {
        let mut axes = Axes::new(
            [420., 780.],
            [0., if col == 0 { 0.006 } else { 0.008 }],
            "True slant range (km)",
            if col == 0 {
                r"$K_\ell(R)$ (km$^{-1}$)"
            } else {
                r"$K_\ell(R)(600/R)^2$ (km$^{-1}$)"
            },
        );
        axes.xticks = vec![450., 600., 750.];
        axes.yticks = if col == 0 {
            vec![0., 0.003, 0.006]
        } else {
            vec![0., 0.004, 0.008]
        };
        let mut p = Panel::new(Rect::new(24. + col as f64 * 115., 23., 82., 48.), axes);
        for (j, l) in [0, 30, 60].iter().enumerate() {
            let points = syisr_long_pulse::kernel(*l)
                .iter()
                .enumerate()
                .map(|(s, &w)| {
                    let r = 600. + (s as f64 - 109.5) * DR;
                    [r, w / DR * if col == 0 { 1. } else { (600. / r).powi(2) }]
                })
                .collect::<Vec<_>>();
            p.push(Layer::Line(Line::new(points, colours[j])));
        }
        f.panels.push(p);
        f.labels.push(Label::new(
            if col == 0 {
                "Common support, different lag shapes"
            } else {
                "Geometry applied inside each range weight"
            },
            65. + col as f64 * 115.,
            16.,
        ));
    }
    for (j, t) in ["0 microseconds", "300 microseconds", "600 microseconds"]
        .iter()
        .enumerate()
    {
        let mut label = Label::new(*t, 45. + j as f64 * 70., 83.);
        label.color = colours[j].into();
        f.labels.push(label);
    }
    for col in 0..2 {
        let mut axes = Axes::new(
            [0., 600.],
            if col == 0 { [0.92, 1.01] } else { [-50., 0.] },
            r"Measured lag ($\mu$s)",
            if col == 0 {
                r"$G_\ell/G_0$"
            } else {
                "Centroid shift (km)"
            },
        );
        axes.xticks = vec![0., 200., 400., 600.];
        axes.yticks = if col == 0 {
            vec![0.94, 0.97, 1.]
        } else {
            vec![-40., -20., 0.]
        };
        let mut p = Panel::new(Rect::new(24. + col as f64 * 115., 108., 82., 48.), axes);
        for (j, row) in moments.iter().enumerate() {
            let points = row
                .iter()
                .enumerate()
                .map(|(l, v)| [l as f64 * 10., if col == 0 { v.2 } else { v.1 - ranges[j] }])
                .collect::<Vec<_>>();
            p.push(Layer::Line(Line::new(points, colours[j])));
        }
        f.panels.push(p);
        f.labels.push(Label::new(
            if col == 0 {
                "Change in normalized ACF from geometry alone"
            } else {
                "Geometric weighting moves the effective centre"
            },
            65. + col as f64 * 115.,
            100.,
        ));
    }
    for (j, t) in ["400 km centre", "600 km centre", "1000 km centre"]
        .iter()
        .enumerate()
    {
        let mut label = Label::new(*t, 45. + j as f64 * 70., 168.);
        label.color = colours[j].into();
        f.labels.push(label);
    }
    f.labels.push(Label::new(
        "Uniform plasma; rectangular 2 ms pulse; receiver filter omitted in these panels",
        115.,
        179.,
    ));
    for ext in ["pdf", "png", "svg"] {
        f.save(format!("{}.{ext}", a[2]))?;
    }
    println!(
        "{}",
        serde_json::json!({"max_complex_fir_relative_replay_error":max_simple,"max_inferred_fir_relative_replay_error":max_inferred,"geometry_at_lags_0_300_600_us":ranges.iter().map(|&r|serde_json::json!({"range_km":r,"gain_centroid_ratio":geometry_moments(r,&[0,30,60])})).collect::<Vec<_>>() })
    );
    Ok(())
}
