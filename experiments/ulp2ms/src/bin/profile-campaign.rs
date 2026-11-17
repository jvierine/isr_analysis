//! MPI distribution of independent full-profile windows; local files only.
use anyhow::{ensure, Result};
use std::{
    fs,
    path::Path,
    process::{Command, Stdio},
};
fn metadata(path: &Path) -> Result<serde_json::Value> {
    let f = hdf5::File::open(path)?;
    Ok(serde_json::from_slice(
        &f.attr("metadata_json")?.read_raw::<u8>()?,
    )?)
}
fn main() -> Result<()> {
    let a = std::env::args().collect::<Vec<_>>();
    ensure!(
        a.len() == 5 || a.len() == 6,
        "profile-campaign run|collect old-campaign new-root noon-profile [knot-spacing-or-list]"
    );
    let old = Path::new(&a[2]);
    let root = Path::new(&a[3]);
    let grid_spec = a.get(5).map(String::as_str).unwrap_or("100");
    let z = syisr_long_pulse::profile::knot_grid(grid_spec)?;
    let observed_max: f64 = std::env::var("SYISR_OBSERVED_MAX_KM")
        .ok()
        .map(|v| v.parse())
        .transpose()?
        .unwrap_or(1450.);
    let iterations = std::env::var("SYISR_PROFILE_ITERATIONS").unwrap_or("200".into());
    let reference = metadata(Path::new(&a[4]))?;
    let calibration = reference["fit"]["calibration_adu2"].as_f64().unwrap();
    ensure!(
        reference["fit"]["status"] == "converged",
        "reference calibration did not converge"
    );
    let plan: serde_json::Value = serde_json::from_slice(&fs::read(old.join("plan.json"))?)?;
    let windows = plan["windows"].as_array().unwrap();
    fs::create_dir_all(root.join("windows"))?;
    if a[1] == "run" {
        let rank = std::env::var("OMPI_COMM_WORLD_RANK")
            .unwrap_or("0".into())
            .parse::<usize>()?;
        let size = std::env::var("OMPI_COMM_WORLD_SIZE")
            .unwrap_or("1".into())
            .parse::<usize>()?;
        let exe = std::env::current_exe()?.with_file_name("profile-fit");
        let mut failures = vec![];
        let limit: usize = std::env::var("SYISR_WINDOW_LIMIT")
            .ok()
            .map(|v| v.parse())
            .transpose()?
            .unwrap_or(windows.len());
        ensure!(limit <= windows.len(), "invalid window limit");
        let start: usize = std::env::var("SYISR_WINDOW_START")
            .ok()
            .map(|v| v.parse())
            .transpose()?
            .unwrap_or(0);
        ensure!(start <= limit, "invalid first window");
        for i in (start + rank..limit).step_by(size) {
            let source = old.join(format!("windows/{i:03}"));
            let dest = root.join(format!("windows/{i:03}"));
            fs::create_dir_all(&dest)?;
            let product = dest.join("profile.h5");
            if product.exists()
                && hdf5::File::open(&product)
                    .map(|f| f.dataset("receiver_fir_real").is_ok())
                    .unwrap_or(false)
                && metadata(&product)
                    .map(|m| {
                        m["fit"]["status"] == "converged"
                            && m["composition_shape"].is_object()
                                == std::env::var("SYISR_COMPOSITION_SHAPE_ROOT").is_ok()
                            && m["provenance"]["lpm"].as_str() == source.join("lpm.h5").to_str()
                            && m["measurement_covariance"]["product"].is_string()
                                == std::env::var("SYISR_COVARIANCE_ROOT").is_ok()
                            && m["requested_observation_max_slant_range_km"]
                                .as_f64()
                                .unwrap_or(1450.)
                                == observed_max
                    })
                    .unwrap_or(false)
                && hdf5::File::open(&product)?
                    .dataset("slant_range_km")?
                    .read_raw::<f64>()?
                    == z
            {
                continue;
            }
            let mut previous = None;
            if product.exists() {
                let backup = dest.join(format!(
                    "previous-{}.h5",
                    chrono::Utc::now().timestamp_millis()
                ));
                fs::rename(&product, &backup)?;
                previous = Some(backup);
            }
            let log = fs::File::create(dest.join("run.log"))?;
            let mut command = Command::new(&exe);
            if let Ok(p) = std::env::var("SYISR_PARTIAL_ROOT") {
                command.env(
                    "SYISR_PARTIAL_PRODUCT",
                    Path::new(&p).join(format!("windows/{i:03}/partial.h5")),
                );
            }
            if let Ok(p) = std::env::var("SYISR_INITIAL_ROOT") {
                command.env(
                    "SYISR_INITIAL_PROFILE",
                    Path::new(&p).join(format!("windows/{i:03}/profile.h5")),
                );
            }
            if let Some(p) = previous {
                command.env("SYISR_INITIAL_PROFILE", p);
            }
            if let Ok(p) = std::env::var("SYISR_COVARIANCE_ROOT") {
                command.env(
                    "SYISR_STRUCTURED_COVARIANCE",
                    Path::new(&p).join(format!("windows/{i:03}/covariance.h5")),
                );
            }
            if let Ok(p) = std::env::var("SYISR_COMPOSITION_SHAPE_ROOT") {
                command.env(
                    "SYISR_COMPOSITION_SHAPE",
                    Path::new(&p).join(format!("{i:03}.h5")),
                );
            }
            let status = command
                .arg(source.join("lpm.h5"))
                .arg(source.join("fits.h5"))
                .arg(&product)
                .args([grid_spec, "1", &calibration.to_string(), &iterations, "25"])
                .env(
                    "RAYON_NUM_THREADS",
                    std::env::var("SYISR_PROFILE_THREADS").unwrap_or("1".into()),
                )
                .stdout(Stdio::from(log.try_clone()?))
                .stderr(Stdio::from(log))
                .status()?;
            let good = status.success()
                && metadata(&product)
                    .map(|m| m["fit"]["status"] == "converged")
                    .unwrap_or(false);
            if !good {
                failures.push(i);
            }
            eprintln!(
                "rank {rank}: window {i} {}",
                if good { "converged" } else { "needs review" }
            );
        }
        fs::write(
            root.join(format!("rank-{rank:02}.json")),
            serde_json::to_vec_pretty(
                &serde_json::json!({"rank":rank,"failures":failures,"calibration":calibration}),
            )?,
        )?;
        // Do not abort other MPI ranks when this rank finishes with a failed
        // window. The saved rank reports and campaign collector retain failures.
        if !failures.is_empty() {
            eprintln!("rank {rank}: nonconverged profile windows {failures:?}");
        }
    } else if a[1] == "collect" {
        let oldh = hdf5::File::open(old.join("campaign.h5"))?;
        let oldz = oldh
            .dataset("slant_range_km")
            .ok()
            .map(|d| d.read_raw::<f64>())
            .transpose()?
            .unwrap_or_default();
        let density = oldh
            .dataset("density_valid")
            .ok()
            .map(|d| d.read_raw::<u8>())
            .transpose()?
            .unwrap_or_default();
        let nt = windows.len();
        let n = z.len() * nt;
        let keys = [
            "ne_m3",
            "ti_k",
            "te_k",
            "h_fraction",
            "he_fraction",
            "o_fraction",
            "sigma_h",
            "sigma_he",
            "sigma_ti_k",
            "sigma_te_k",
            "reduced_chi2",
            "sigma_log_ne",
            "density_resolution_diagonal",
        ];
        let mut arrays = vec![vec![f64::NAN; n]; keys.len()];
        let mut valid = vec![0u8; n];
        let mut flags = vec![255u8; n];
        let mut done = vec![0u8; nt];
        let mut nonconverged = vec![];
        let mut structured_windows = 0usize;
        let mut composition_windows = 0usize;
        for t in 0..nt {
            let path = root.join(format!("windows/{t:03}/profile.h5"));
            if !path.exists() {
                continue;
            }
            let f = hdf5::File::open(&path)?;
            let m = metadata(&path)?;
            let fit = &m["fit"];
            let composition = syisr_long_pulse::composition::from_metadata(&m)?;
            if composition.is_some() {
                composition_windows += 1;
            }
            if m["measurement_covariance"]["product"].is_string() {
                structured_windows += 1;
            }
            ensure!(
                m["requested_observation_max_slant_range_km"]
                    .as_f64()
                    .unwrap_or(1450.)
                    == observed_max,
                "window {t} observation coverage differs from requested campaign"
            );
            let converged = fit["status"] == "converged";
            if !converged {
                nonconverged.push(t);
            }
            done[t] = 1;
            let zz = f.dataset("slant_range_km")?.read_raw::<f64>()?;
            ensure!(zz == z, "inconsistent campaign profile knots");
            for (k, key) in keys[..6].iter().enumerate() {
                let v = f.dataset(key)?.read_raw::<f64>()?;
                for j in 0..z.len() {
                    arrays[k][j * nt + t] = v[j];
                }
            }
            let cov = f
                .dataset("conditional_transformed_covariance")
                .ok()
                .map(|d| d.read_raw::<f64>().unwrap());
            let res = f
                .dataset("conditional_resolution_matrix")
                .ok()
                .map(|d| d.read_raw::<f64>().unwrap());
            let np = z.len() * 6;
            let p = fit["transformed_profile"].as_array().unwrap();
            let data_chi = fit["data_chi_square"].as_f64().unwrap();
            let ng = f.dataset("observed_slant_range_km")?.size();
            let nfit = m["fitted_parameter_count"].as_u64().unwrap_or(np as u64) as f64;
            let red = data_chi / ((ng * 120 - ng) as f64 - nfit).max(1.);
            let x: Vec<f64> = p.iter().map(|v| v.as_f64().unwrap()).collect();
            for j in 0..z.len() {
                let i = j * nt + t;
                let diag = |k: usize| {
                    cov.as_ref()
                        .map(|c| c[(6 * j + k) * np + 6 * j + k])
                        .unwrap_or(f64::NAN)
                };
                let q = p[6 * j + 5].as_f64().unwrap();
                let hh = arrays[3][i];
                let cross = cov
                    .as_ref()
                    .map(|c| c[(6 * j + 4) * np + 6 * j + 5])
                    .unwrap_or(f64::NAN);
                arrays[6][i] = diag(4).sqrt();
                arrays[7][i] = (q * q * diag(4) + (1. - hh).powi(2) * diag(5)
                    - 2. * q * (1. - hh) * cross)
                    .sqrt();
                if let Some(shape) = &composition {
                    let (_, errors) = shape.fractions(z[j] + 0.05, &x, cov.as_deref());
                    arrays[6][i] = errors[0];
                    arrays[7][i] = errors[1];
                }
                arrays[8][i] = arrays[1][i] * diag(1).sqrt();
                arrays[9][i] = arrays[2][i] * diag(2).sqrt();
                arrays[10][i] = red;
                let nearest = oldz
                    .iter()
                    .enumerate()
                    .min_by(|a, b| (a.1 - z[j]).abs().total_cmp(&(b.1 - z[j]).abs()))
                    .map(|v| v.0)
                    .unwrap_or(0);
                let lower = m["partial_products"].is_object();
                let maximum = m["requested_observation_max_slant_range_km"]
                    .as_f64()
                    .unwrap_or(1450.);
                let extended = maximum > 1450.;
                let support = z[j] >= if lower { 200. } else { 450. }
                    && z[j] <= maximum
                    && (if lower && z[j] < 450. {
                        true
                    } else if extended {
                        // The legacy campaign has no density screening above 1450 km.
                        // Use the new fit's uncertainty/resolution, never its last old gate.
                        true
                    } else {
                        density[nearest * nt + t] > 0
                    });
                let leverage = res.as_ref().map(|c| c[(6 * j) * np + 6 * j]).unwrap_or(0.);
                arrays[11][i] = diag(0).sqrt();
                arrays[12][i] = leverage;
                valid[i] = u8::from(
                    converged && red <= 3. && support && diag(0).sqrt() < 0.5 && leverage > 0.1,
                );
                flags[i] = if !converged {
                    1
                } else if red > 3. {
                    2
                } else if !support {
                    8
                } else {
                    0
                };
                if (arrays[1][i] - 300.).abs() < 1.
                    || (arrays[1][i] - 5000.).abs() < 1.
                    || (arrays[2][i] - 300.).abs() < 1.
                    || (arrays[2][i] - 12000.).abs() < 1.
                {
                    flags[i] |= 4;
                }
                // Density's spectral power conversion also depends on temperatures.
                // A small conditional density error cannot validate a bound-limited fit.
                if flags[i] != 0 {
                    valid[i] = 0;
                }
                // Residual backgrounds are observation-range quantities, not physical knot fields.
            }
        }
        let out = hdf5::File::create(root.join("campaign.h5"))?;
        if composition_windows > 0 {
            ensure!(
                composition_windows == done.iter().filter(|v| **v > 0).count(),
                "mixed composition modes in one campaign"
            );
            out.new_attr_builder().with_data("IRI-height sigmoid H and bounded Gaussian He envelope; five joint global parameters; physical zero below500km; conditional errors under fixed shape model").create("composition_model")?;
        }
        out.new_attr::<f64>()
            .create("display_max_slant_range_km")?
            .write_scalar(&observed_max)?;
        out.new_attr_builder()
            .with_data("joint_full_profile")
            .create("analysis_type")?;
        if z[0] < 200. {
            out.new_attr_builder().with_data("Partial-pulse observations; lower physical nuisance guard layers; retrieved profiles shown from 200 km").create("partial_pulse_extension")?;
        }
        out.new_attr_builder()
            .with_data(a[2].as_str())
            .create("source_campaign")?;
        out.new_attr_builder()
            .with_data(a[4].as_str())
            .create("noon_reference_profile")?;
        for (k, key) in keys.iter().enumerate() {
            out.new_dataset::<f64>()
                .shape((z.len(), nt))
                .create(*key)?
                .write_raw(&arrays[k])?;
        }
        out.new_dataset_builder()
            .with_data(&z)
            .create("slant_range_km")?;
        out.new_dataset_builder()
            .with_data(&z.iter().map(|v| v + 0.05).collect::<Vec<_>>())
            .create("altitude_km")?;
        for name in ["time_edges_utc_s", "time_centres_utc_s"] {
            if let Ok(d) = oldh.dataset(name) {
                out.new_dataset_builder()
                    .with_data(&d.read_raw::<f64>()?)
                    .create(name)?;
            }
        }
        out.new_dataset_builder()
            .with_data(&done)
            .create("window_complete")?;
        out.new_dataset::<u8>()
            .shape((z.len(), nt))
            .create("density_valid")?
            .write_raw(&valid)?;
        out.new_dataset::<u8>()
            .shape((z.len(), nt))
            .create("quality_flags")?
            .write_raw(&flags)?;
        out.new_attr_builder().with_data("Joint profile, exact lag/range FIR ambiguity and in-integral R^-2; inferred fixed gain from assumed noon peak 1e12 m^-3; not absolute calibration. Density includes spectral power dependence on temperatures/composition.").create("density")?;
        out.new_attr_builder()
            .with_data(grid_spec)
            .create("knot_grid_specification")?;
        let covariance_note = if structured_windows > 0 {
            ensure!(
                structured_windows == done.iter().filter(|&&v| v > 0).count(),
                "mixed covariance modes in campaign"
            );
            "Fixed structured Gaussian plus empirical full range/lag/real-imag covariance, including a shared reference; covariance-estimation and instrumental systematics remain. Full-range pulse screening was repeated before these fits."
        } else {
            "Diagonal weights omit correlated errors. The original pulse screen did not inspect the entire extended topside."
        };
        out.new_attr_builder()
            .with_data(covariance_note)
            .create("measurement_covariance")?;
        out.new_attr_builder().with_data("Knots given by slant_range_km; 25 km quadrature; curvature regularization. Display excludes guards and temperature-bound fits; density additionally requires conditional log-density error <0.5 and resolution diagonal >0.1. Bound fraction errors withheld. These conditional cuts do not establish physical resolution or eliminate receiver/background/model bias.").create("limitations")?;
        fs::write(
            root.join("status.json"),
            serde_json::to_vec_pretty(
                &serde_json::json!({"completed_windows":done.iter().filter(|&&v|v>0).count(),"total_windows":nt,"nonconverged_windows":nonconverged,"calibration":calibration}),
            )?,
        )?;
        println!("{}", fs::read_to_string(root.join("status.json"))?);
    } else {
        anyhow::bail!("expected run or collect");
    }
    Ok(())
}
