//! Fixed measurement covariance from unfitted block products, never fit residuals.
use anyhow::{ensure, Result};
use nalgebra::DMatrix as M;
use std::{fs, path::Path};
use syisr_long_pulse::{
    measurement_covariance::DenseWhitening,
    noise_blocks::{difference_factors, NoiseBlocks},
    structured_covariance::StructuredModel,
};
fn matrix(f: &hdf5::File, name: &str, m: &M<f64>) -> Result<()> {
    let row: Vec<f64> = (0..m.nrows())
        .flat_map(|i| (0..m.ncols()).map(move |j| m[(i, j)]))
        .collect();
    f.new_dataset::<f64>()
        .shape((m.nrows(), m.ncols()))
        .chunk((120.min(m.nrows()), 120.min(m.ncols())))
        .deflate(3)
        .create(name)?
        .write_raw(&row)?;
    Ok(())
}
fn main() -> Result<()> {
    let a: Vec<String> = std::env::args().collect();
    ensure!(
        a.len() == 4 || a.len() == 5,
        "prepare-structured-covariance profile-root validation-root output-root [window]"
    );
    let source = Path::new(&a[1]);
    let validation = Path::new(&a[2]);
    let root = Path::new(&a[3]);
    let rank: usize = std::env::var("OMPI_COMM_WORLD_RANK")
        .unwrap_or("0".into())
        .parse()?;
    let size: usize = std::env::var("OMPI_COMM_WORLD_SIZE")
        .unwrap_or("1".into())
        .parse()?;
    let windows: Vec<usize> = if a.len() == 5 {
        a[4].split(',')
            .map(str::parse)
            .collect::<std::result::Result<Vec<_>, _>>()?
    } else {
        {
            let p: serde_json::Value =
                serde_json::from_slice(&fs::read(source.join("plan.json"))?)?;
            (0..p["windows"].as_array().unwrap().len()).collect()
        }
    };
    for t in windows.into_iter().skip(rank).step_by(size) {
        let dir = root.join(format!("windows/{t:03}"));
        fs::create_dir_all(&dir)?;
        let path = dir.join("covariance.h5");
        if path.exists() {
            continue;
        }
        let cv: serde_json::Value =
            serde_json::from_slice(&fs::read(validation.join(format!("window-{t:03}.json")))?)?;
        let alpha = cv["selected_alpha"].as_f64().unwrap();
        ensure!(cv["selected_score_gain_vs_diagonal"].as_f64().unwrap()>0.
            && cv["selected_heldout_energy"].as_f64().unwrap()<2.,
            "window {t} covariance prediction failed; investigate screening/nonstationarity before fitting");
        let nugget = cv["diagonal_nugget"].as_f64().unwrap();
        let b = NoiseBlocks::load(source, t)?;
        let all: Vec<usize> = (0..b.pairs.len()).collect();
        ensure!(all.len() >= 10, "insufficient independent block pairs");
        let factors = difference_factors(&b.differences, &all, b.total);
        let variance: Vec<f64> = (0..factors.nrows())
            .map(|i| factors.row(i).norm_squared())
            .collect();
        let (gamma, power, negative) = b.voltage_noise(&all)?;
        let acfs = b.local_voltage_acfs(&all);
        let mut model = StructuredModel::with_self_noise(
            gamma,
            &b.native,
            &acfs,
            power,
            &variance,
            &b.sigma,
            power.powi(2) / b.total,
        )?;
        model.normalized *= 1. - nugget;
        for i in 0..b.sigma.len() {
            model.normalized[(i, i)] += nugget * variance[i] / b.sigma[i].powi(2);
        }
        let mut empirical = factors;
        for i in 0..b.sigma.len() {
            empirical.row_mut(i).scale_mut(1. / b.sigma[i]);
        }
        // Matrix factors completely specify the likelihood independently of its builder.
        let dense = DenseWhitening::new(model.normalized)?;
        let tmp = dir.join("covariance.tmp.h5");
        let h = hdf5::File::create(&tmp)?;
        matrix(&h, "structured_cholesky_lower", &dense.lower)?;
        matrix(&h, "normalized_empirical_modes", &empirical)?;
        h.new_dataset_builder()
            .with_data(&b.sigma)
            .create("measurement_sigma")?;
        let mean: Vec<f64> = (0..b.blocks.nrows())
            .map(|i| {
                (0..b.counts.len())
                    .filter(|&j| b.counts[j] > 0.)
                    .map(|j| b.blocks[(i, j)] * b.counts[j] / b.total)
                    .sum()
            })
            .collect();
        h.new_dataset_builder()
            .with_data(&mean)
            .create("measurement_mean")?;
        h.new_dataset_builder()
            .with_data(&b.native.iter().map(|&v| v as u64).collect::<Vec<_>>())
            .create("observed_native_gate_index")?;
        h.new_attr::<f64>()
            .create("structured_fraction")?
            .write_scalar(&alpha)?;
        h.new_attr::<f64>()
            .create("diagonal_nugget")?
            .write_scalar(&nugget)?;
        let metadata = serde_json::json!({"mode":"structured Gaussian voltage fourth moments plus empirical joint covariance","formula":"C = D [alpha L L^T + (1-alpha) U U^T] D; D is measurement_sigma","stacking":"range, lag 10..600 us, real then imaginary","training":"all disjoint adjacent-block differences, pulse-count normalized; no fitted residuals","structured_target":"nonstationary plasma self-noise, voltage filtering, exact partial-pair sums, shared reference; 10% diagonal component selected by prior development checks","reference_cross_terms":"zero in the compact voltage model; observation and reference voltage supports are separated by more than the modeled correlation length","fixed_during_fit":true,"nuisance":"background amplitudes must be profiled jointly after whitening","source_measurements_root":source,"window":t,"validation":cv,"receiver_negative_psd_mass":negative,"clamped_local_variances":model.clamped,"limitations":"conditional approximate Gaussian covariance; covariance-estimation uncertainty and common instrumental/model bias excluded"});
        let mut metadata = metadata;
        metadata.as_object_mut().unwrap().remove("source_profile");
        metadata["source_measurements_root"] = serde_json::json!(source);
        metadata["window"] = serde_json::json!(t);
        h.new_attr_builder()
            .with_data(serde_json::to_string(&metadata)?.as_bytes())
            .create("metadata_json")?;
        h.flush()?;
        drop(h);
        fs::rename(tmp, path)?;
        eprintln!("window {t}: covariance prepared, structured fraction {alpha}");
    }
    Ok(())
}
