//! Append recording days without changing the reference/gain or counting pulses twice.
use crate::{
    campaign,
    lp480::{self, Job, Manifest},
    lp480_fit::WindowFit,
};
use anyhow::{ensure, Result};
use std::io::Read;
use std::{
    collections::{BTreeMap, BTreeSet},
    path::{Path, PathBuf},
};

fn identical_files(a: &Path, b: &Path) -> Result<bool> {
    let n = std::fs::metadata(a)?.len();
    if n != std::fs::metadata(b)?.len() {
        return Ok(false);
    }
    let (mut a, mut b) = (std::fs::File::open(a)?, std::fs::File::open(b)?);
    let (mut x, mut y) = (vec![0u8; 1 << 20], vec![0u8; 1 << 20]);
    let mut remaining = n;
    while remaining > 0 {
        let k = remaining.min(x.len() as u64) as usize;
        a.read_exact(&mut x[..k])?;
        b.read_exact(&mut y[..k])?;
        if x[..k] != y[..k] {
            return Ok(false);
        }
        remaining -= k as u64;
    }
    Ok(true)
}
type Deduplicated = (Vec<Job>, Vec<serde_json::Value>, usize, BTreeSet<i64>);
/// Cache fully validated archive headers; changed archives require a new cache.
pub fn cached_index(raw:&Path,cache:&Path)->Result<()> {
    let mut files=campaign::files(raw,".dat")?;files.sort();
    let signature=files.iter().map(|p|{let m=std::fs::metadata(p)?;Ok(serde_json::json!([std::fs::canonicalize(p)?,m.len(),m.modified()?.duration_since(std::time::UNIX_EPOCH)?.as_nanos().to_string()]))}).collect::<Result<Vec<_>>>()?;
    let stamp=cache.with_extension("signature.h5");
    if cache.exists(){let old:Vec<serde_json::Value>=campaign::load(&stamp)?;ensure!(old==signature,"raw archive changed since cached index: {}",raw.display());let m:Manifest=campaign::load(cache)?;let offset:i64=std::env::var("SYISR_HEADER_UTC_OFFSET_S").unwrap_or_else(|_|"0".into()).parse()?;ensure!(m.header_utc_offset_s==offset,"cached index uses a different clock offset");return Ok(());}
    lp480::index(raw,cache)?;lp480::repair_index(cache)?;campaign::save(&stamp,&signature)?;Ok(())
}
/// UTC half-open daily selection after merging acquisition folders and exact deduplication.
pub fn daily_index(indices:&[PathBuf],start:i64,end:i64,out:&Path)->Result<()> {
    ensure!(end-start==86400&&start.rem_euclid(86400)==0,"expected a UTC calendar day");
    let mut merged:Option<Manifest>=None;
    for p in indices{let mut next:Manifest=campaign::load(p)?;next.jobs.retain(|j|j.start>=start&&j.start<end);
        if let Some(m)=&mut merged{ensure!(m.pulse_us==next.pulse_us&&m.ipp_s==next.ipp_s&&m.header_utc_offset_s==next.header_utc_offset_s,"daily layout/clock mismatch");m.jobs=merge_jobs(std::mem::take(&mut m.jobs),next.jobs)?;m.quarantined_segments.extend(next.quarantined_segments);m.clock_repairs.extend(next.clock_repairs);}else{merged=Some(next);}}
    let mut m=merged.ok_or_else(||anyhow::anyhow!("no acquisition indices"))?;
    let(jobs,excluded,removed,_)=dedup_jobs(m.jobs)?;m.jobs=jobs;m.quarantined_segments.extend(excluded.clone());
    campaign::save(out,&m)?;
    campaign::save(&out.with_extension("coverage.h5"),&serde_json::json!({"start_utc_s":start,"end_utc_s":end,"measured_windows":m.jobs.iter().map(|j|j.start).collect::<BTreeSet<_>>(),"removed_duplicate_pulses":removed,"duplicate_files":excluded,"method":"UTC half-open calendar day; neighboring acquisition folders merged before selection; exact repeated files compared byte-for-byte; no synthetic measurements"}))?;Ok(())
}
fn dedup_jobs(mut jobs: Vec<Job>) -> Result<Deduplicated> {
    let files = jobs
        .iter()
        .flat_map(|j| j.parts.iter().map(|p| p.file.clone()))
        .collect::<BTreeSet<_>>();
    let mut groups = BTreeMap::<String, Vec<String>>::new();
    for p in files {
        let real = std::fs::canonicalize(&p)?;
        groups
            .entry(real.file_name().unwrap().to_string_lossy().into_owned())
            .or_default()
            .push(p);
    }
    let mut aliases = BTreeMap::new();
    let mut excluded = vec![];
    for (group, files) in groups {
        for p in files.iter().skip(1) {
            ensure!(
                identical_files(Path::new(&files[0]), Path::new(p))?,
                "Conflicting same-name acquisition files: {} and {}",
                files[0],
                p
            );
            aliases.insert(p.clone(), files[0].clone());
            excluded.push(serde_json::json!({"kind":"exact_duplicate","file":p,"representative_file":files[0],"acquisition":group,"offset":0,"bytes":std::fs::metadata(p)?.len(),"reason":"Repeated daily-folder acquisition; every byte compared equal; pulses retained only once"}));
        }
    }
    let mut removed = 0;
    let mut changed = BTreeSet::new();
    for j in &mut jobs {
        let old = j.parts.clone();
        let mut keep = Vec::<lp480::Part>::new();
        for mut p in j.parts.drain(..) {
            if let Some(a) = aliases.get(&p.file) {
                p.file = a.clone();
            }
            let mut duplicate = false;
            for q in &keep {
                if p.file != q.file {
                    continue;
                }
                let end = p.offset + p.pulses as u64 * lp480::FRAME as u64;
                let qend = q.offset + q.pulses as u64 * lp480::FRAME as u64;
                if p.offset < qend && q.offset < end {
                    ensure!(
                        p.offset == q.offset && end == qend && p.block == q.block,
                        "Nonidentical overlap after acquisition-file deduplication"
                    );
                    duplicate = true;
                }
            }
            if duplicate {
                removed += p.pulses;
            } else {
                keep.push(p);
            }
        }
        j.parts = keep;
        if j.parts != old {
            changed.insert(j.start);
        }
    }
    Ok((jobs, excluded, removed, changed))
}
/// Correct repeated day-boundary acquisitions; invalidate only affected fits.
pub fn deduplicate(root: &Path, summary: &Path) -> Result<()> {
    let mut m: Manifest = campaign::load(&root.join("decoded/index.h5"))?;
    let (jobs, excluded, removed, changed) = dedup_jobs(m.jobs)?;
    m.jobs = jobs;
    m.quarantined_segments.extend(excluded.iter().cloned());
    if !changed.is_empty() {
        let stamp = std::time::SystemTime::now()
            .duration_since(std::time::UNIX_EPOCH)?
            .as_secs();
        let archive = root.join(format!("superseded-before-dedup-{stamp}"));
        std::fs::create_dir_all(&archive)?;
        let initial = root.join("fits/initial");
        if initial.is_dir() {
            for f in std::fs::read_dir(&initial)? {
                let p = f?.path();
                let name = p.file_name().unwrap().to_string_lossy();
                if changed.iter().any(|t| name.starts_with(&format!("{t}."))) {
                    std::fs::rename(&p, archive.join(p.file_name().unwrap()))?;
                }
            }
        }
        if root.join("fits").is_dir() {
            for f in std::fs::read_dir(root.join("fits"))? {
                let p = f?.path();
                if p.file_name()
                    .unwrap()
                    .to_string_lossy()
                    .starts_with("sweep-")
                {
                    std::fs::rename(&p, archive.join(p.file_name().unwrap()))?;
                }
            }
        }
        for name in ["COMPACT_COMPLETE", "COMPLETE", "latest", "final"] {
            let p = root.join(name);
            if std::fs::symlink_metadata(&p).is_ok() {
                std::fs::rename(p, archive.join(name))?;
            }
        }
        campaign::save(&root.join("decoded/index.h5"), &m)?;
        campaign::save(
            summary,
            &serde_json::json!({"duplicate_files":excluded,"removed_repeated_pulses":removed,"changed_times":changed,"method":"Exact full-file byte comparison; raw unchanged; affected products/checkpoints are recomputed; unrelated initial fits retained"}),
        )?;
    }
    println!(
        "DEDUPLICATED_FILES={} REMOVED_REPEATED_PULSES={} CHANGED_WINDOWS={}",
        excluded.len(),
        removed,
        changed.len()
    );
    Ok(())
}

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
    m.quarantined_segments.extend(extra.quarantined_segments);
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
        let same_grid = w.beams.iter().all(|b| {
            old.iter().find(|j| j.id == b.id).is_some_and(|j| {
                b.spline.knots.last().copied() == Some(lp480::beam_top_km(j) as f64)
            })
        });
        if same_grid && (w.status == "converged" || w.status == "data_stable_gradient_pending") {
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
    fn daily_selection_merges_boundary_parts_and_excludes_next_midnight()->Result<()> {
        let root=std::env::temp_dir().join(format!("lp480-daily-{}",std::process::id()));std::fs::create_dir_all(&root)?;
        let raw=root.join("raw.dat");std::fs::write(&raw,vec![0u8;lp480::FRAME*4])?;
        let job=|t|Job{id:format!("{t}-0-0"),start:t,az:0.,el:90.,front_us:767.,parts:vec![lp480::Part{file:raw.display().to_string(),offset:0,pulses:1,block:t}]};
        let mut m=Manifest{schema:"test".into(),header_utc_offset_s:28800,clock_repairs:vec![],quarantined_segments:vec![],raw_root:root.display().to_string(),jobs:vec![job(-1800),job(0)],pulse_us:480.,ipp_s:0.016};
        campaign::save(&root.join("a.h5"),&m)?;m.jobs=vec![job(0),job(84600),job(86400)];m.jobs[0].parts[0].offset=lp480::FRAME as u64;
        campaign::save(&root.join("b.h5"),&m)?;daily_index(&[root.join("a.h5"),root.join("b.h5")],0,86400,&root.join("out.h5"))?;
        let result:Manifest=campaign::load(&root.join("out.h5"))?;assert_eq!(result.jobs.iter().map(|j|j.start).collect::<Vec<_>>(),vec![0,84600]);assert_eq!(result.jobs[0].parts.len(),2);
        std::fs::remove_dir_all(root)?;Ok(())
    }
    #[test]
    fn repeated_files_must_match_every_byte_and_count_pulses_once() -> Result<()> {
        let root = std::env::temp_dir().join(format!("lp480-file-dedup-{}", std::process::id()));
        for d in ["a", "b"] {
            std::fs::create_dir_all(root.join(d))?;
            std::fs::write(root.join(d).join("raw.dat"), vec![42u8; 2 * lp480::FRAME])?;
        }
        let part = |d: &str| lp480::Part {
            file: root.join(d).join("raw.dat").display().to_string(),
            offset: 0,
            pulses: 2,
            block: 1000,
        };
        let jobs = vec![Job {
            id: "1000-0-0".into(),
            start: 1000,
            az: 0.,
            el: 90.,
            front_us: 767.,
            parts: vec![part("a"), part("b")],
        }];
        let (clean, files, n, t) = dedup_jobs(jobs.clone())?;
        assert_eq!(clean[0].parts.len(), 1);
        assert_eq!(files.len(), 1);
        assert_eq!(n, 2);
        assert!(t.contains(&1000));
        let mut different = vec![42u8; 2 * lp480::FRAME];
        different[lp480::FRAME] = 43;
        std::fs::write(root.join("b/raw.dat"), different)?;
        assert!(dedup_jobs(jobs).is_err());
        std::fs::remove_dir_all(root)?;
        Ok(())
    }
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
