use anyhow::Result;
use serde_json::{json, Value};
use std::{fs, str::FromStr};
fn ci(r: &Value) -> [f64; 2] {
    let b = &r["best"];
    let h = b["h_fraction"].as_f64().unwrap();
    let chi = b["chi_square"].as_f64().unwrap();
    let mut p: Vec<_> = r["profile_likelihood"]
        .as_array()
        .unwrap()
        .iter()
        .map(|x| {
            (
                x["h_fraction"].as_f64().unwrap(),
                x["chi_square"].as_f64().unwrap() - chi,
            )
        })
        .collect();
    p.push((h, 0.));
    p.sort_by(|a, b| a.0.total_cmp(&b.0));
    let (mut lo, mut hi) = (h, h);
    for w in p.windows(2) {
        for &(h, c) in w {
            if c <= 3.84 {
                lo = lo.min(h);
                hi = hi.max(h);
            }
        }
        if (w[0].1 - 3.84) * (w[1].1 - 3.84) < 0. {
            let h = w[0].0 + (w[1].0 - w[0].0) * (3.84 - w[0].1) / (w[1].1 - w[0].1);
            lo = lo.min(h);
            hi = hi.max(h);
        }
    }
    [lo, hi]
}
fn units(d: &hdf5::Dataset, u: &str) -> Result<()> {
    d.new_attr::<hdf5::types::VarLenUnicode>()
        .create("units")?
        .write_scalar(&hdf5::types::VarLenUnicode::from_str(u)?)?;
    Ok(())
}
fn main() -> Result<()> {
    let a: Vec<_> = std::env::args().collect();
    let root = &a[1];
    let p: Value = serde_json::from_slice(&fs::read(format!("{root}/clean-profile/fits.json"))?)?;
    let b: Value = serde_json::from_slice(&fs::read(format!("{root}/clean-stability.json"))?)?;
    let file = hdf5::File::open(format!("{root}/clean-16.h5"))?;
    let source: Value =
        serde_json::from_slice(&file.dataset("metadata_json_utf8")?.read_raw::<u8>()?)?;
    let out = hdf5::File::create(format!("{root}/syisr-hplus-night-zenith.h5"))?;
    let rows = p["results"].as_array().unwrap();
    let flag: Vec<u8> = rows
        .iter()
        .map(|r| {
            let f = &r["best"];
            let c = ci(r);
            if f["ti_k"].as_f64().unwrap() < 310.
                || f["te_k"].as_f64().unwrap() < 310.
                || f["te_k"].as_f64().unwrap() > 11500.
                || f["chi_square"].as_f64().unwrap() / f["dof"].as_f64().unwrap() > 2.
            {
                2
            } else if c[0] > 0. && c[1] - c[0] < 0.35 {
                1
            } else if c[0] == 0. {
                3
            } else {
                0
            }
        })
        .collect();
    for (name, key, u) in [
        ("h_fraction", "h_fraction", "1"),
        ("he_fraction", "he_fraction", "1"),
        ("o_fraction", "o_fraction", "1"),
        ("ti_k", "ti_k", "K"),
        ("te_k", "te_k", "K"),
        ("spectral_velocity_m_s", "spectral_velocity_m_s", "m s^-1"),
        ("ion_line_power_adu2", "signal_adu2", "ADU^2"),
    ] {
        let v: Vec<f32> = rows
            .iter()
            .map(|r| r["best"][key].as_f64().unwrap() as f32)
            .collect();
        let ds = out
            .new_dataset::<f32>()
            .shape([1, rows.len()])
            .create(name)?;
        ds.write_raw(&v)?;
        units(&ds, u)?;
    }
    let v: Vec<f32> = rows
        .iter()
        .zip(&flag)
        .map(|(r, &f)| {
            if f == 1 {
                r["best"]["h_fraction"].as_f64().unwrap() as f32
            } else {
                f32::NAN
            }
        })
        .collect();
    let ds = out
        .new_dataset::<f32>()
        .shape([1, rows.len()])
        .create("h_fraction_conditionally_supported")?;
    ds.write_raw(&v)?;
    units(&ds, "1")?;
    let ds = out
        .new_dataset::<u8>()
        .shape([1, rows.len()])
        .create("quality_flag")?;
    ds.write_raw(&flag)?;
    for (name, k) in [("h_fraction_ci_lower", 0), ("h_fraction_ci_upper", 1)] {
        let v: Vec<_> = rows.iter().map(|r| ci(r)[k] as f32).collect();
        let ds = out
            .new_dataset::<f32>()
            .shape([1, rows.len()])
            .create(name)?;
        ds.write_raw(&v)?;
        units(&ds, "1")?;
    }
    for (name, u, v) in [
        (
            "altitude_km",
            "km",
            rows.iter()
                .map(|r| r["metadata"]["height_km"].as_f64().unwrap())
                .collect::<Vec<_>>(),
        ),
        ("lag_s", "s", (0..41).map(|i| i as f64 * 1e-5).collect()),
        (
            "time_start_unix_s",
            "s since 1970-01-01T00:00:00Z",
            vec![1700064000.],
        ),
        (
            "time_end_unix_s",
            "s since 1970-01-01T00:00:00Z",
            vec![1700067600.],
        ),
    ] {
        let ds = out.new_dataset_builder().with_data(&v).create(name)?;
        units(&ds, u)?;
    }
    for (key, name) in [
        ("measured_real", "acf_real_adu2"),
        ("measured_imag", "acf_imag_adu2"),
        ("sem_real", "acf_real_sem_adu2"),
        ("sem_imag", "acf_imag_sem_adu2"),
    ] {
        let v: Vec<_> = rows
            .iter()
            .flat_map(|r| {
                r[key].as_array().unwrap().iter().enumerate().map(|(i, v)| {
                    if key == "sem_imag" && i == 0 {
                        0.
                    } else {
                        v.as_f64().unwrap() as f32
                    }
                })
            })
            .collect();
        let ds = out
            .new_dataset::<f32>()
            .shape([1, rows.len(), 41])
            .shuffle()
            .deflate(4)
            .create(name)?;
        ds.write_raw(&v)?;
        units(&ds, "ADU^2")?;
    }
    for q in 0..2 {
        for null in [false, true] {
            let v: Vec<_> = rows
                .iter()
                .flat_map(|r| {
                    let f = if null {
                        r["profile_likelihood"]
                            .as_array()
                            .unwrap()
                            .iter()
                            .find(|x| x["h_fraction"] == 0.)
                            .unwrap()
                    } else {
                        &r["best"]
                    };
                    f["predicted"]
                        .as_array()
                        .unwrap()
                        .iter()
                        .map(|x| x[q].as_f64().unwrap() as f32)
                        .collect::<Vec<_>>()
                })
                .collect();
            let name = format!(
                "{}_model_acf_{}_adu2",
                if null { "no_h" } else { "fitted" },
                if q == 0 { "real" } else { "imag" }
            );
            let ds = out
                .new_dataset::<f32>()
                .shape([1, rows.len(), 41])
                .shuffle()
                .deflate(4)
                .create(name.as_str())?;
            ds.write_raw(&v)?;
            units(&ds, "ADU^2")?;
        }
    }
    let meta = json!({"schema":"ruisdap-syisr-hplus-zenith-poc-v1","producer":"Independent Rust fit of SYISR uncoded LP lag products; supplied L2 and IRI composition not used","fraction_definition":"nH+/(nH+ + nHe+ + nO+); each species singly charged","quality_flag_values":{"0":"weak/ambiguous conditional constraint","1":"conditional H+ support in fitted thermal/receiver model","2":"temperature boundary or poor-fit exclusion","3":"conditional upper limit, H=0 allowed"},"confidence":"profile CI uses diagonal conservative SEM, delta chi square 3.84; 700 km additionally has whole-dwell resampling stability","hplus_700km_block_interval":b["bootstrap_h_fraction_percentiles"],"source_screen":source,"processing_assumptions":p["limitations"],"absolute_electron_density_calibrated":false,"shape_density_m3":1e11,"nuisance_amplitude":"ion_line_power_adu2 is native band-power scale, not Ne","range_resolution":"roughly 100 km pulse + receiver + 17-bin averaging footprint, not independent 1.5 km gates","velocity_sign":"spectral sign only; physical toward/away sign not independently calibrated","science_status":"proof of concept; not full-profile composition inversion or final validated geocorona retrieval"});
    out.new_dataset_builder()
        .with_data(&serde_json::to_vec_pretty(&meta)?)
        .create("metadata_json_utf8")?;
    println!(
        "wrote named float32 datasets for {} heights; {} conditional H+ cells",
        rows.len(),
        flag.iter().filter(|&&f| f == 1).count()
    );
    Ok(())
}
