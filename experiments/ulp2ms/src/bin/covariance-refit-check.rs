//! Compare identical-data noon fits under diagonal and empirical correlated errors.
use anyhow::{ensure, Result};
use nalgebra::DMatrix as M;
use rustplotlib::*;
use std::{fs, path::Path};
use syisr_long_pulse::{measurement_covariance::Whitening, profile::interpolation};
struct P {
    observed_range: Vec<f64>,
    z: Vec<f64>,
    x: Vec<f64>,
    residual: Vec<f64>,
    diagonal_chi: f64,
    weighted_chi: f64,
    declared: f64,
}
fn read(path: &Path) -> Result<P> {
    let h = hdf5::File::open(path)?;
    let m: serde_json::Value = serde_json::from_slice(&h.attr("metadata_json")?.read_raw::<u8>()?)?;
    ensure!(m["fit"]["status"] == "converged", "refit not converged");
    let get = |n: &str| h.dataset(n).unwrap().read_raw::<f64>().unwrap();
    let z = get("slant_range_km");
    let x = m["fit"]["transformed_profile"]
        .as_array()
        .unwrap()
        .iter()
        .map(|v| v.as_f64().unwrap())
        .collect();
    let nr = h.dataset("observed_real_adu2")?.size();
    let mut residual = vec![0.; 2 * nr];
    for (c, name) in ["real", "imag"].iter().enumerate() {
        let obs = get(&format!("observed_{name}_adu2"));
        let model = get(&format!("model_{name}_adu2"));
        let sigma = get(&format!("sem_{name}_adu2"));
        for i in 0..nr {
            residual[2 * i + c] = (obs[i] - model[i]) / sigma[i];
        }
    }
    let diagonal_chi = residual.iter().map(|v| v * v).sum::<f64>();
    let weighted_chi = if let Ok(d) = h.dataset("measurement_covariance_normalized_modes") {
        let s = d.shape();
        let u = M::from_row_slice(s[0], s[1], &d.read_raw::<f64>()?);
        let a = h
            .attr("measurement_covariance_shrinkage")?
            .read_scalar::<f64>()?;
        Whitening::new(&u, a)?
            .apply(&M::from_column_slice(2 * nr, 1, &residual))
            .norm_squared()
    } else {
        diagonal_chi
    };
    let declared = m["fit"]["data_chi_square"].as_f64().unwrap();
    ensure!(
        (weighted_chi - declared).abs() / declared < 1e-10,
        "refit covariance cost replay"
    );
    Ok(P {
        observed_range: get("observed_slant_range_km"),
        z,
        x,
        residual,
        diagonal_chi,
        weighted_chi,
        declared,
    })
}
fn state(p: &P, r: f64) -> Result<[f64; 6]> {
    let mut q = [0.; 6];
    for (k, w) in interpolation(&p.z, r)? {
        for j in 0..6 {
            q[j] += w * p.x[6 * k + j];
        }
    }
    Ok([
        1e11 * q[0].exp(),
        1000. * q[1].exp(),
        1000. * q[2].exp(),
        q[4],
        (1. - q[4]) * q[5],
        (1. - q[4]) * (1. - q[5]),
    ])
}
fn main() -> Result<()> {
    let a: Vec<String> = std::env::args().collect();
    ensure!(
        a.len() == 3,
        "covariance-refit-check original-noon-profile audit-root"
    );
    let root = Path::new(&a[2]);
    let cases = [
        read(Path::new(&a[1]))?,
        read(&root.join("refits/noon-alpha0.7.h5"))?,
        read(&root.join("refits/noon-alpha0.5.h5"))?,
        read(&root.join("refits/noon-diagonal-rough0.1.h5"))?,
        read(&root.join("refits/noon-correlated-rough0.1.h5"))?,
    ];
    let mut summary = vec![];
    for (i, p) in cases.iter().enumerate() {
        let mut max = [0f64; 5];
        let mut med = vec![vec![]; 5];
        for r in (500..=1400).step_by(25) {
            let old = state(&cases[0], r as f64)?;
            let new = state(p, r as f64)?;
            for k in 0..5 {
                let d = if k < 3 {
                    100. * (new[k] / old[k] - 1.)
                } else {
                    100. * (new[k] - old[k])
                };
                max[k] = max[k].max(d.abs());
                med[k].push(d.abs());
            }
        }
        let mut rows = serde_json::Map::new();
        for (k, name) in [
            "ne_relative_percent",
            "Ti_relative_percent",
            "Te_relative_percent",
            "H_percentage_points",
            "He_percentage_points",
        ]
        .iter()
        .enumerate()
        {
            med[k].sort_by(f64::total_cmp);
            rows.insert(name.to_string(),serde_json::json!({"median_abs_change":med[k][med[k].len()/2],"max_abs_change":max[k]}));
        }
        let mode = [
            "diagonal",
            "alpha0.7",
            "alpha0.5",
            "diagonal_rough0.1",
            "alpha0.7_rough0.1",
        ][i];
        summary.push(serde_json::json!({"mode":mode,"diagonal_replay_chi_square":p.diagonal_chi,"own_weighted_chi_square":p.weighted_chi,"cost_replay_relative_error":(p.weighted_chi-p.declared).abs()/p.declared,"changes_500_1400km":rows}));
        for (label, lo, hi) in [
            ("lower", 0., 450.),
            ("middle", 450., 900.),
            ("upper", 900., 2000.),
        ] {
            for (c, part) in ["real", "imag"].iter().enumerate() {
                let v: Vec<f64> = p
                    .observed_range
                    .iter()
                    .enumerate()
                    .filter(|(_, r)| **r >= lo && **r < hi)
                    .flat_map(|(g, _)| (0..60).map(move |l| p.residual[2 * (g * 60 + l) + c]))
                    .collect();
                summary.last_mut().unwrap()[format!("{label}_residual_rms_{part}_SEM")] = serde_json::json!(
                    (v.iter().map(|v| v * v).sum::<f64>() / v.len() as f64).sqrt()
                );
            }
        }
    }
    fs::write(
        root.join("refit-comparison.json"),
        serde_json::to_vec_pretty(&summary)?,
    )?;
    let mut f = Figure::new(IntendedUse::ArticleWide);
    f.width_mm = 230.;
    f.height_mm = 187.;
    f.raster_dpi = 180.;
    f.labels.push(Label::new(
        "SYISR: sensitivity to range--lag covariance weighting",
        115.,
        5.,
    ));
    f.labels.push(Label::new(
        "18 Dec 2025, 04:00--04:30 UTC; same data, gain, knots and smoothing",
        115.,
        12.,
    ));
    for k in 0..6 {
        let (xl, yl, xlab, ylab, xt, yt) = match k {
            0 => (
                [200., 1400.],
                [9., 12.3],
                "Altitude (km)",
                r"$\log_{10} n_e$ (m$^{-3}$)",
                vec![200., 600., 1000., 1400.],
                vec![9., 10., 11., 12.],
            ),
            1 => (
                [200., 1400.],
                [300., 4000.],
                "Altitude (km)",
                r"$T_i$ (K)",
                vec![200., 600., 1000., 1400.],
                vec![1000., 2000., 3000.],
            ),
            2 => (
                [200., 1400.],
                [300., 5000.],
                "Altitude (km)",
                r"$T_e$ (K)",
                vec![200., 600., 1000., 1400.],
                vec![1000., 2500., 4000.],
            ),
            3 => (
                [200., 1400.],
                [0., 0.8],
                "Altitude (km)",
                r"H$^+$ fraction",
                vec![200., 600., 1000., 1400.],
                vec![0., 0.4, 0.8],
            ),
            4 => (
                [200., 1400.],
                [0., 0.08],
                "Altitude (km)",
                r"He$^+$ fraction",
                vec![200., 600., 1000., 1400.],
                vec![0., 0.04, 0.08],
            ),
            _ => {
                let bound = cases
                    .iter()
                    .flat_map(|p| (0..60).map(|l| p.residual[2 * l].abs()))
                    .fold(1f64, f64::max)
                    .ceil();
                (
                    [10., 600.],
                    [-bound, bound],
                    r"Lag ($\mu$s)",
                    "Real residual / SEM",
                    vec![100., 300., 500.],
                    vec![-bound, 0., bound],
                )
            }
        };
        let mut ax = Axes::new(xl, yl, xlab, ylab);
        ax.xticks = xt;
        ax.yticks = yt;
        let mut panel = Panel::new(
            Rect::new(
                20. + (k % 3) as f64 * 74.,
                27. + (k / 3) as f64 * 73.,
                55.,
                52.,
            ),
            ax,
        );
        for (i, p) in cases.iter().take(3).enumerate() {
            let points = if k < 5 {
                (200..=1400)
                    .map(|r| {
                        let v = state(p, r as f64)?[k];
                        Ok([r as f64 + 0.05, if k == 0 { v.log10() } else { v }])
                    })
                    .collect::<Result<Vec<_>>>()?
            } else {
                (0..60)
                    .map(|l| [10. + 10. * l as f64, p.residual[2 * l]])
                    .collect()
            };
            panel.push(Layer::Line(Line::new(
                points,
                ["#888888", "#17608e", "#ba6b24"][i],
            )));
        }
        f.panels.push(panel);
    }
    f.labels.push(Label::new(
        "Grey: diagonal; blue / orange: empirical covariance with alpha = 0.7 / 0.5",
        115.,
        172.,
    ));
    f.labels.push(Label::new(
        "Last panel: lowest observed gate; diagnostic refits, not a new campaign",
        115.,
        179.,
    ));
    for ext in ["png", "pdf", "svg"] {
        f.save(root.join(format!("covariance-refit-comparison.{ext}")))?;
    }
    println!("{}", serde_json::to_string_pretty(&summary)?);
    Ok(())
}
