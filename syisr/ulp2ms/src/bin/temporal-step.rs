//! HDF5 interface for random-walk prediction and RTS smoothing.
use anyhow::{ensure, Result};
use nalgebra::{DMatrix as M, DVector as V};
use syisr_long_pulse::temporal;
fn meta(f: &hdf5::File) -> Result<serde_json::Value> {
    Ok(serde_json::from_slice(
        &f.attr("metadata_json")?.read_raw::<u8>()?,
    )?)
}
fn state(path: &str) -> Result<(Vec<usize>, V<f64>, M<f64>)> {
    let f = hdf5::File::open(path)?;
    if f.dataset("columns").is_ok() {
        let c = f
            .dataset("columns")?
            .read_raw::<u64>()?
            .into_iter()
            .map(|v| v as usize)
            .collect::<Vec<_>>();
        let n = c.len();
        return Ok((
            c,
            V::from_vec(f.dataset("mean")?.read_raw()?),
            M::from_row_slice(n, n, &f.dataset("covariance")?.read_raw::<f64>()?),
        ));
    }
    let m = meta(&f)?;
    let c: Vec<usize> = serde_json::from_value(m["fit"]["filter_columns"].clone())?;
    let x: Vec<f64> = serde_json::from_value(m["fit"]["transformed_profile"].clone())?;
    let p: Vec<f64> = serde_json::from_value(m["fit"]["filter_covariance"].clone())?;
    let n = c.len();
    Ok((
        c.clone(),
        V::from_iterator(n, c.iter().map(|&i| x[i])),
        M::from_column_slice(n, n, &p),
    ))
}
fn save(
    path: &str,
    c: &[usize],
    x: &V<f64>,
    p: &M<f64>,
    metadata: serde_json::Value,
) -> Result<()> {
    let f = hdf5::File::create(path)?;
    f.new_dataset_builder()
        .with_data(&c.iter().map(|v| *v as u64).collect::<Vec<_>>())
        .create("columns")?;
    f.new_dataset_builder()
        .with_data(x.as_slice())
        .create("mean")?;
    let row: Vec<_> = (0..p.nrows())
        .flat_map(|i| (0..p.ncols()).map(move |j| p[(i, j)]))
        .collect();
    f.new_dataset::<f64>()
        .shape(p.shape())
        .create("covariance")?
        .write_raw(&row)?;
    f.new_attr_builder()
        .with_data(serde_json::to_string(&metadata)?.as_bytes())
        .create("metadata_json")?;
    Ok(())
}
fn main() -> Result<()> {
    let a: Vec<_> = std::env::args().collect();
    ensure!(a.len()>=4,"temporal-step predict profile output dt_seconds [process_sigma_multiplier] | diffuse profile output | smooth filtered next-predicted next-smoothed output | copy-state profile output");
    match a[1].as_str() {
        "predict" => {
            let (c, x, p) = state(&a[2])?;
            let dt = a[4].parse()?;
            let strength = a.get(5).map(|s| s.parse()).transpose()?.unwrap_or(1.);
            let (x, p) = temporal::predict(&x, &p, &c, dt, strength)?;
            save(
                &a[3],
                &c,
                &x,
                &p,
                serde_json::json!({"source":a[2],"dt_seconds":dt,"process_sigma_multiplier":strength,"process_sigma_per_30min":c.iter().map(|&i|temporal::process_sigma(i)).collect::<Vec<_>>()}),
            )?;
        }
        "diffuse" => {
            let (c, x, _) = state(&a[2])?;
            let p = M::from_diagonal(&V::from_iterator(
                c.len(),
                c.iter()
                    .map(|&i| (10. * temporal::process_sigma(i)).powi(2)),
            ));
            save(
                &a[3],
                &c,
                &x,
                &p,
                serde_json::json!({"initialization":a[2],"weak_initial_covariance":"100 times nominal30min process covariance"}),
            )?;
        }
        "copy-state" => {
            let (c, x, p) = state(&a[2])?;
            save(
                &a[3],
                &c,
                &x,
                &p,
                serde_json::json!({"filtered_source":a[2]}),
            )?;
        }
        "smooth" => {
            ensure!(
                a.len() == 6,
                "smooth filtered next-predicted next-smoothed output"
            );
            let (c, x, p) = state(&a[2])?;
            let (cp, xp, pp) = state(&a[3])?;
            let (cn, xn, pn) = state(&a[4])?;
            ensure!(c == cp && c == cn, "RTS state mapping changed");
            let (xs, ps) = temporal::smooth(&x, &p, &xp, &pp, &xn, &pn)?;
            save(
                &a[5],
                &c,
                &xs,
                &ps,
                serde_json::json!({"filtered_source":a[2],"next_prediction":a[3],"next_smoothed":a[4],"method":"RTS random-walk smoother, no repeated likelihood; local Gaussian approximation; physical bounds must be checked"}),
            )?;
        }
        _ => anyhow::bail!("unknown temporal operation"),
    }
    Ok(())
}
