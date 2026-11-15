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
    let minutes: usize = a.get(2).map(|s| s.parse()).transpose()?.unwrap_or(60);
    anyhow::ensure!(
        minutes > 0 && 60 % minutes == 0,
        "integration must divide hour"
    );
    let nt = 6 * 60 / minutes;
    let suffix = if minutes == 60 {
        String::new()
    } else {
        format!("-{minutes}min")
    };
    let mut results = vec![];
    let mut screens = vec![];
    for h in 15..=20 {
        let dir = if minutes != 60 {
            format!("{root}/timefit-{minutes}/{h}")
        } else if h == 16 {
            format!("{root}/clean-profile")
        } else {
            format!("{root}/clean-hourly/{h}")
        };
        let p: Value = serde_json::from_slice(&fs::read(format!("{dir}/fits.json"))?)?;
        for rows in p["results"].as_array().unwrap().chunks(11) {
            results.push(rows.to_vec());
        }
        let f = hdf5::File::open(format!("{root}/clean-{h}.h5"))?;
        let m: Value = serde_json::from_slice(&f.dataset("metadata_json_utf8")?.read_raw::<u8>()?)?;
        screens.push(json!({"hour_utc":h,"raw_selected_pulses":m["raw_selected_pulses"],"excluded_pulses":m["excluded_pulses"],"screen":m["screen"]}));
    }
    let heights: Vec<f64> = results[0]
        .iter()
        .map(|r| r["metadata"]["height_km"].as_f64().unwrap())
        .collect();
    let n = heights.len();
    let mut flags = vec![];
    let mut lower = vec![];
    let mut upper = vec![];
    for rows in &results {
        for r in rows {
            let f = &r["best"];
            let c = ci(r);
            let boundary = f["status"] != "converged"
                || f["ti_k"].as_f64().unwrap() > 4900.
                || f["ti_k"].as_f64().unwrap() < 310.
                || f["te_k"].as_f64().unwrap() < 310.
                || f["te_k"].as_f64().unwrap() > 11500.
                || f["chi_square"].as_f64().unwrap() / f["dof"].as_f64().unwrap() > 2.;
            flags.push(if boundary {
                2_u8
            } else if c[0] > 0. && c[1] - c[0] < 0.35 {
                1
            } else if c[0] == 0. {
                3
            } else {
                0
            });
            lower.push(c[0] as f32);
            upper.push(c[1] as f32);
        }
    }
    let path = format!("{root}/syisr-hplus-time-altitude{suffix}.h5");
    let out = hdf5::File::create(&path)?;
    let mut product = json!({"schema":"syisr-hplus-time-altitude-v1","date_utc":"2023-11-15","integration_s":minutes*60,"altitude_km":heights,"screening":screens,"quality_flag":flags.chunks(n).collect::<Vec<_>>(),"h_fraction_ci_lower":lower.chunks(n).collect::<Vec<_>>(),"h_fraction_ci_upper":upper.chunks(n).collect::<Vec<_>>(),"limitations":["Local ~100 km pulse footprint, not full-profile composition inversion","Conditional diagonal-weight profile intervals; receiver and species-temperature systematics not included","All measured lags retained; inferred receiver response; absolute density not calibrated","Quality 0 weak, 1 conditional support, 2 boundary/poor, 3 H=0 allowed; never interpret an unconstrained best fit as detection"]});
    for (key, u) in [
        ("h_fraction", "1"),
        ("he_fraction", "1"),
        ("o_fraction", "1"),
        ("ti_k", "K"),
        ("te_k", "K"),
        ("signal_adu2", "ADU^2"),
    ] {
        let v: Vec<f32> = results
            .iter()
            .flatten()
            .map(|r| r["best"][key].as_f64().unwrap() as f32)
            .collect();
        let d = out.new_dataset::<f32>().shape([nt, n]).create(key)?;
        d.write_raw(&v)?;
        units(&d, u)?;
        product[key] = json!(v.chunks(n).collect::<Vec<_>>());
    }
    for (name, v) in [
        ("h_fraction_ci_lower", &lower),
        ("h_fraction_ci_upper", &upper),
    ] {
        let d = out.new_dataset::<f32>().shape([nt, n]).create(name)?;
        d.write_raw(v)?;
        units(&d, "1")?;
    }
    let d = out
        .new_dataset::<u8>()
        .shape([nt, n])
        .create("quality_flag")?;
    d.write_raw(&flags)?;
    let v: Vec<_> = results
        .iter()
        .flatten()
        .zip(&flags)
        .map(|(r, &f)| {
            if f == 1 {
                r["best"]["h_fraction"].as_f64().unwrap() as f32
            } else {
                f32::NAN
            }
        })
        .collect();
    let d = out
        .new_dataset::<f32>()
        .shape([nt, n])
        .create("h_fraction_conditionally_supported")?;
    d.write_raw(&v)?;
    units(&d, "1")?;
    for (name, u, v) in [
        ("altitude_km", "km", heights),
        (
            "time_start_unix_s",
            "s since 1970-01-01T00:00:00Z",
            (0..nt)
                .map(|i| 1700060400. + (i * minutes * 60) as f64)
                .collect(),
        ),
        (
            "time_end_unix_s",
            "s since 1970-01-01T00:00:00Z",
            (1..=nt)
                .map(|i| 1700060400. + (i * minutes * 60) as f64)
                .collect(),
        ),
    ] {
        let d = out.new_dataset_builder().with_data(&v).create(name)?;
        units(&d, u)?;
    }
    out.new_dataset_builder()
        .with_data(&serde_json::to_vec(&product)?)
        .create("metadata_json_utf8")?;
    fs::write(
        format!("{root}/time-altitude{suffix}.json"),
        serde_json::to_vec(&product)?,
    )?;
    println!(
        "{} time-altitude cells; {} conditional-support cells",
        flags.len(),
        flags.iter().filter(|&&f| f == 1).count()
    );
    Ok(())
}
