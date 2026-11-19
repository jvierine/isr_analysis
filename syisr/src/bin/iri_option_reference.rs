//! Sample an external IRI composition option on the common comparison geometry.
use anyhow::{ensure, Result};
use chrono::{DateTime, Datelike, Timelike, Utc};
use std::{
    fs,
    io::Write,
    path::Path,
    process::{Command, Stdio},
};
fn main() -> Result<()> {
    let a: Vec<_> = std::env::args().collect();
    ensure!(a.len() == 6, "QUERY_HDF DRIVER MODEL_DIR LABEL OUTPUT_HDF");
    let f = hdf5::File::open(&a[1])?;
    let q = f.dataset("iri_query_points")?.read_raw::<f64>()?;
    let n = q.len() / 6;
    let out = Path::new(&a[5]);
    let input = out.with_extension("query.tmp");
    let response = out.with_extension("response.tmp");
    let mut w = fs::File::create(&input)?;
    for p in q.chunks_exact(6) {
        let t = DateTime::<Utc>::from_timestamp(p[0] as i64, 0).unwrap();
        let ut = t.hour() as f64 + t.minute() as f64 / 60. + t.second() as f64 / 3600.;
        writeln!(
            w,
            "{} {} {} {:.10} {:.10} {:.10} {:.10}",
            t.year(),
            t.month(),
            t.day(),
            ut,
            p[3],
            p[4],
            p[5]
        )?;
    }
    drop(w);
    let status = Command::new(fs::canonicalize(&a[2])?)
        .current_dir(&a[3])
        .stdin(Stdio::from(fs::File::open(&input)?))
        .stdout(Stdio::from(fs::File::create(&response)?))
        .status()?;
    ensure!(status.success(), "IRI option failed");
    let txt = fs::read_to_string(&response)?;
    let lines: Vec<_> = txt.lines().filter(|l| !l.trim().is_empty()).collect();
    ensure!(
        lines.len() == n,
        "IRI option returned {} of {n} queries",
        lines.len()
    );
    let mut fractions = vec![0.; n * 7];
    let mut density = vec![0.; n * 7];
    for (i, line) in lines.iter().enumerate() {
        let row: Vec<f64> = line
            .split_whitespace()
            .map(str::parse)
            .collect::<std::result::Result<_, _>>()?;
        ensure!(
            row.len() == 7
                && (row[0] - q[6 * i + 5]).abs() < 0.02
                && row[1] > 0.
                && row[2..6]
                    .iter()
                    .all(|v| v.is_finite() && (0.0..=1.0).contains(v)),
            "invalid IRI option fractions at {i}"
        );
        ensure!(
            (row[2..6].iter().sum::<f64>() - 1.).abs() < 0.025,
            "IRI species closure"
        );
        for s in 0..4 {
            fractions[i * 7 + s] = row[s + 2];
            density[i * 7 + s] = row[s + 2] * row[1] * 1e-6;
        }
    }
    let h = hdf5::File::create(out)?;
    h.new_dataset::<f64>()
        .shape((n, 7))
        .deflate(4)
        .create("ion_fraction")?
        .write_raw(&fractions)?;
    h.new_dataset::<f64>()
        .shape((n, 7))
        .deflate(4)
        .create("ion_density_cm3")?
        .write_raw(&density)?;
    h.new_dataset::<f64>()
        .shape((n, 6))
        .create("query_points")?
        .write_raw(&q)?;
    let mut meta = serde_json::json!({"model":a[4],"query_count":n,"fraction_normalization":"total ion population; no H/He/O-only renormalization","species_order":["H+","He+","O+","N+","NO+","O2+","N2+"]});
    if let Ok(s) = fs::read_to_string(Path::new(&a[3]).join("provenance.json")) {
        meta["implementation"] = serde_json::from_str(&s)?;
    }
    h.new_dataset_builder()
        .with_data(&serde_json::to_vec(&meta)?)
        .create("metadata_json_utf8")?;
    fs::remove_file(input)?;
    fs::remove_file(response)?;
    println!("{}: {n} matched points", a[4]);
    Ok(())
}
