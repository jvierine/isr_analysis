//! Append recording days without changing the reference/gain or counting pulses twice.
use crate::{
    campaign,
    lp480::{self, Job, Manifest},
    lp480_fit::WindowFit,
};
use anyhow::{ensure, Result};
use std::{
    collections::{BTreeMap, BTreeSet},
    path::{Path, PathBuf},
};

fn merge_jobs(old: Vec<Job>, extra: Vec<Job>) -> Result<Vec<Job>> {
    let mut jobs: BTreeMap<_, _> = old.into_iter().map(|j| (j.id.clone(), j)).collect();
    let mut paths = BTreeMap::<String, PathBuf>::new();
    for j in extra {
        if let Some(target) = jobs.get_mut(&j.id) {
            ensure!(
                target.start == j.start
                    && target.az == j.az
                    && target.el == j.el
                    && target.front_us == j.front_us,
                "layout changed inside appended beam-window"
            );
            for p in j.parts {
                if !paths.contains_key(&p.file) {
                    paths.insert(p.file.clone(), std::fs::canonicalize(&p.file)?);
                }
                let mut duplicate = false;
                for q in &target.parts {
                    if !paths.contains_key(&q.file) {
                        paths.insert(q.file.clone(), std::fs::canonicalize(&q.file)?);
                    }
                    if paths[&p.file] != paths[&q.file] {
                        continue;
                    }
                    let p_end = p.offset + (p.pulses as u64) * lp480::FRAME as u64;
                    let q_end = q.offset + (q.pulses as u64) * lp480::FRAME as u64;
                    if p.offset < q_end && q.offset < p_end {
                        ensure!(
                            p.offset == q.offset && p_end == q_end && p.block == q.block,
                            "nonidentical overlapping raw pulse segments"
                        );
                        duplicate = true;
                    }
                }
                if !duplicate {
                    target.parts.push(p);
                }
            }
        } else {
            jobs.insert(j.id.clone(), j);
        }
    }
    Ok(jobs.into_values().collect())
}
fn unchanged_times(old: &[Job], new: &[Job]) -> BTreeSet<i64> {
    let mut a = BTreeMap::<i64, Vec<&Job>>::new();
    let mut b = BTreeMap::<i64, Vec<&Job>>::new();
    for j in old {
        a.entry(j.start).or_default().push(j);
    }
    for j in new {
        b.entry(j.start).or_default().push(j);
    }
    a.into_iter()
        .filter_map(|(t, j)| (b.get(&t) == Some(&j)).then_some(t))
        .collect()
}
fn link(source: &Path, destination: &Path) -> Result<()> {
    let source = std::fs::canonicalize(source)?;
    if destination.exists() {
        ensure!(
            std::fs::canonicalize(destination)? == source,
            "refusing to replace an existing extension product"
        );
    } else {
        std::os::unix::fs::symlink(source, destination)?;
    }
    Ok(())
}
pub fn extend(base: &Path, raw: &Path, root: &Path) -> Result<()> {
    let mut m: Manifest = campaign::load(&base.join("decoded/index.h5"))?;
    ensure!(
        m.pulse_us == 480. && m.ipp_s == 0.016,
        "unsupported base pulse layout"
    );
    std::fs::create_dir_all(root.join("decoded/products"))?;
    std::fs::create_dir_all(root.join("fits/initial"))?;
    std::fs::create_dir_all(root.join("input"))?;
    let extra = lp480::index_with_offset(
        raw,
        &root.join("decoded/additional-index.h5"),
        m.header_utc_offset_s,
    )?;
    ensure!(!extra.jobs.is_empty(), "no added raw observations");
    let old = m.jobs.clone();
    m.jobs = merge_jobs(m.jobs, extra.jobs)?;
    m.raw_root = format!("{}; {}", m.raw_root, raw.display());
    let unchanged = unchanged_times(&old, &m.jobs);
    let lookup: BTreeMap<_, _> = old.iter().map(|j| (&j.id, j)).collect();
    let mut products = 0;
    for j in &m.jobs {
        if lookup.get(&j.id).is_some_and(|p| **p == *j) {
            let name = format!("{}.h5", j.id);
            let source = base.join("decoded/products").join(&name);
            if source.exists() {
                link(&source, &root.join("decoded/products").join(name))?;
                products += 1;
            }
        }
    }
    let mut fits = vec![];
    for t in &unchanged {
        let name = format!("{t}.h5");
        let source = base.join("fits/initial").join(&name);
        if !source.exists() {
            continue;
        }
        let w: WindowFit = campaign::load(&source)?;
        if w.status == "converged" || w.status == "data_stable_gradient_pending" {
            ensure!(
                w.metadata["helium_fixed_zero"] == true
                    && w.metadata["topside_temperature_gradient_prior"]
                        .as_str()
                        .is_some_and(|s| s != "off"),
                "base fit uses different composition/temperature assumptions"
            );
            link(&source, &root.join("fits/initial").join(name))?;
            fits.push(*t);
        }
    }
    link(
        &base.join("decoded/background.h5"),
        &root.join("decoded/background.h5"),
    )?;
    for name in ["calibration.h5", "calibration_metadata.h5"] {
        link(
            &base.join("input").join(name),
            &root.join("input").join(name),
        )?;
    }
    campaign::save(
        &root.join("extension.h5"),
        &serde_json::json!({
            "base_run":base,"added_raw":raw,"unchanged_windows":unchanged,
            "reused_initial_windows":fits,"reused_screened_beam_windows":products,
            "method":"Reuse only byte-identical indexed observations and accepted initial fits; boundary windows with additional pulses are screened/refitted. Original quiet reference and fixed beam gains retained. Temporal refinement runs over the combined campaign."
        }),
    )?;
    campaign::save(&root.join("decoded/index.h5"), &m)?;
    println!(
        "EXTENDED jobs={} reused_products={} reused_initial_fits={}",
        m.jobs.len(),
        products,
        fits.len()
    );
    Ok(())
}

#[cfg(test)]
mod tests {
    use super::*;
    #[test]
    fn merge_deduplicates_and_invalidates_changed_boundary_fit() -> Result<()> {
        let root = std::env::temp_dir().join(format!("lp480-extend-{}", std::process::id()));
        std::fs::create_dir_all(&root)?;
        let f = root.join("raw.dat");
        std::fs::write(&f, [])?;
        let job = |offset, pulses| Job {
            id: "1000-0-0".into(),
            start: 1000,
            az: 0.,
            el: 90.,
            front_us: 767.,
            parts: vec![lp480::Part {
                file: f.display().to_string(),
                offset,
                pulses,
                block: 1000,
            }],
        };
        let old = vec![job(0, 10)];
        let same = merge_jobs(old.clone(), old.clone())?;
        assert_eq!(same[0].parts.len(), 1);
        assert_eq!(unchanged_times(&old, &same), BTreeSet::from([1000]));
        let alias = root.join("alias.dat");
        std::os::unix::fs::symlink(&f, &alias)?;
        let mut duplicate = job(0, 10);
        duplicate.parts[0].file = alias.display().to_string();
        assert!(merge_jobs(old.clone(), vec![duplicate])? == old);
        let added = merge_jobs(old.clone(), vec![job(10 * lp480::FRAME as u64, 5)])?;
        assert_eq!(added[0].parts.iter().map(|p| p.pulses).sum::<usize>(), 15);
        assert!(unchanged_times(&old, &added).is_empty());
        assert!(merge_jobs(old, vec![job(5 * lp480::FRAME as u64, 10)]).is_err());
        std::fs::remove_dir_all(root)?;
        Ok(())
    }
}
