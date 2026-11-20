//! Supplied Sanya L2 density reference for a fixed empirical receiver gain.
//! L2 composition/temperature/velocity are not imposed on the inversion.
use crate::{campaign, lp480::Manifest};
use anyhow::{ensure, Result};
use chrono::NaiveDateTime;
use hdf5::types::FixedAscii;
use serde::{Deserialize, Serialize};
use std::{collections::BTreeMap, path::Path};

#[derive(Serialize, Deserialize)]
pub struct DensityProfile {
    pub altitude_km: Vec<f64>,
    pub ne_m3: Vec<f64>,
}
pub type Profiles = BTreeMap<String, DensityProfile>;
fn median(v: &mut [f64]) -> f64 {
    v.sort_by(f64::total_cmp);
    v[v.len() / 2]
}
pub fn import(root: &Path, paths: &[String]) -> Result<()> {
    let manifest: Manifest = campaign::load(&root.join("index.h5"))?;
    let mut bins = BTreeMap::<String, BTreeMap<i64, Vec<f64>>>::new();
    for path in paths {
        let f = hdf5::File::open(path)?;
        let shape = f.dataset("Fittedparameter")?.shape();
        ensure!(
            shape.len() == 4 && shape[3] == 4,
            "unsupported L2 parameter layout"
        );
        let (nb, nt, nr) = (shape[0], shape[1], shape[2]);
        ensure!(
            f.dataset("GEO")?.shape() == vec![nb, nt, nr, 3],
            "unsupported L2 GEO layout"
        );
        ensure!(
            f.dataset("Time")?.shape() == vec![nb, nt],
            "unsupported L2 clock layout"
        );
        ensure!(
            f.dataset("Beamdir")?.shape() == vec![nb, 2],
            "unsupported L2 beam layout"
        );
        ensure!(
            f.dataset("Fitstate")?.shape() == vec![nb, nt, nr],
            "unsupported L2 quality layout"
        );
        ensure!(
            f.dataset("Error")?.shape() == shape,
            "unsupported L2 error layout"
        );
        let p = f.dataset("Fittedparameter")?.read_raw::<f64>()?;
        let e = f.dataset("Error")?.read_raw::<f64>()?;
        let state = f.dataset("Fitstate")?.read_raw::<f64>()?;
        let geo = f.dataset("GEO")?.read_raw::<f64>()?;
        let dirs = f.dataset("Beamdir")?.read_raw::<f64>()?;
        let times = f.dataset("Time")?.read_raw::<FixedAscii<20>>()?;
        for b in 0..nb {
            for t in 0..nt {
                let Ok(time) = NaiveDateTime::parse_from_str(
                    times[b * nt + t].as_str().trim(),
                    "%d-%b-%Y %H:%M:%S",
                ) else {
                    continue;
                };
                let start = time.and_utc().timestamp().div_euclid(1800) * 1800;
                let Some(job) = manifest.jobs.iter().find(|j| {
                    j.start == start
                        && ((j.az - dirs[b * 2] + 180.).rem_euclid(360.) - 180.).abs() < 0.05
                        && (j.el - dirs[b * 2 + 1]).abs() < 0.05
                }) else {
                    continue;
                };
                for r in 0..nr {
                    let i = (b * nt + t) * nr + r;
                    let ne = p[i * 4];
                    let err = e[i * 4];
                    let z = geo[i * 3 + 2];
                    if state[i] == 0.
                        && ne.is_finite()
                        && ne > 1e8
                        && err.is_finite()
                        && err >= 0.
                        && err / ne <= 0.5
                        && z.is_finite()
                        && (150. ..=600.).contains(&z)
                    {
                        bins.entry(job.id.clone())
                            .or_default()
                            .entry((z / 5.).round() as i64)
                            .or_default()
                            .push(ne);
                    }
                }
            }
        }
    }
    let mut profiles = Profiles::new();
    for (id, bins) in bins {
        let (mut z, mut ne) = (vec![], vec![]);
        for (height, mut values) in bins {
            z.push(height as f64 * 5.);
            ne.push(median(&mut values));
        }
        if z.len() >= 10 && z[0] <= 210. && *z.last().unwrap() >= 450. {
            profiles.insert(
                id,
                DensityProfile {
                    altitude_km: z,
                    ne_m3: ne,
                },
            );
        }
    }
    ensure!(
        !profiles.is_empty(),
        "no matched, quality-screened L2 density profiles"
    );
    campaign::save(&root.join("density_reference.h5"), &profiles)?;
    campaign::save(
        &root.join("density_reference_metadata.h5"),
        &serde_json::json!({
            "source":"supplied Sanya L2 Ne", "files":paths, "profiles":profiles.len(),
            "selection":"Fitstate=0; positive finite Ne; reported density error <=50%; matched beam within0.05deg and same30min recorded UTC bin",
            "use":"Fixed gain per beam over recording; no fitted temperature/velocity/composition constraint; no absolute calibration guarantee",
            "clock":"L2 text timestamps interpreted as UTC, with no inferred timezone shift"
        }),
    )?;
    println!("L2_DENSITY_PROFILES={}", profiles.len());
    Ok(())
}

#[cfg(test)]
mod tests {
    use super::*;
    #[test]
    fn l2_import_matches_time_beam_and_rejects_bad_density() -> Result<()> {
        let root = std::env::temp_dir().join(format!("lp480-l2-{}", std::process::id()));
        std::fs::create_dir_all(&root)?;
        let id = "1715385600-65171-16384".to_string();
        let manifest = Manifest {
            header_utc_offset_s: 28800,
            clock_repairs: vec![],
            schema: "test".into(),
            jobs: vec![crate::lp480::Job {
                id: id.clone(),
                start: 1715385600,
                az: 358.,
                el: 90.,
                front_us: 767.,
                parts: vec![],
            }],
            pulse_us: 480.,
            ipp_s: 0.016,
            raw_root: "test".into(),
        };
        campaign::save(&root.join("index.h5"), &manifest)?;
        let path = root.join("l2.h5");
        let f = hdf5::File::create(&path)?;
        let (mut p, mut e, mut geo) = (
            vec![0.; 2 * 14 * 4],
            vec![0.; 2 * 14 * 4],
            vec![0.; 2 * 14 * 3],
        );
        for t in 0..2 {
            for r in 0..14 {
                let i = t * 14 + r;
                p[i * 4] = if t == 0 { 1e11 } else { 3e11 };
                e[i * 4] = 1e9;
                geo[i * 3 + 2] = 180. + r as f64 * 25.;
            }
        }
        // Positive but very uncertain density must not become a gain reference.
        p[0] = 1e15;
        e[0] = 2e15;
        for (name, v) in [("Fittedparameter", &p), ("Error", &e)] {
            f.new_dataset::<f64>()
                .shape([1, 2, 14, 4])
                .create(name)?
                .write_raw(v)?;
        }
        f.new_dataset::<f64>()
            .shape([1, 2, 14, 3])
            .create("GEO")?
            .write_raw(&geo)?;
        f.new_dataset::<f64>()
            .shape([1, 2, 14])
            .create("Fitstate")?
            .write_raw(&vec![0.; 28])?;
        f.new_dataset::<f64>()
            .shape([1, 2])
            .create("Beamdir")?
            .write_raw(&[358., 90.])?;
        let times = [
            FixedAscii::<20>::from_ascii("11-May-2024 00:01:00").unwrap(),
            FixedAscii::<20>::from_ascii("11-May-2024 00:06:30").unwrap(),
        ];
        f.new_dataset::<FixedAscii<20>>()
            .shape([1, 2])
            .create("Time")?
            .write_raw(&times)?;
        drop(f);
        import(&root, &[path.display().to_string()])?;
        let profiles: Profiles = campaign::load(&root.join("density_reference.h5"))?;
        assert_eq!(profiles.len(), 1);
        assert_eq!(profiles[&id].ne_m3[0], 3e11);
        assert!(profiles[&id].ne_m3.iter().all(|v| *v < 1e12));
        let f = hdf5::File::open_rw(&path)?;
        f.dataset("Beamdir")?.write_raw(&[0., 60.])?;
        drop(f);
        assert!(import(&root, &[path.display().to_string()]).is_err());
        std::fs::remove_dir_all(root)?;
        Ok(())
    }
}
