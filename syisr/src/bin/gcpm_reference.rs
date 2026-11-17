//! Match external GCPMv2.4 to the existing per-gate reference query list.
use anyhow::{ensure, Result};
use chrono::{DateTime, Datelike, Timelike, Utc};
use std::{collections::BTreeMap, fs, io::Write, path::Path};
fn main() -> Result<()> {
    let a: Vec<_> = std::env::args().collect();
    ensure!(
        a.len() == 5,
        "prepare QUERY_HDF APF107_DAT CHUNK_DIR | collect QUERY_HDF CHUNK_DIR OUTPUT_HDF"
    );
    let f = hdf5::File::open(&a[2])?;
    let q = f.dataset("iri_query_points")?.read_raw::<f64>()?;
    let n = q.len() / 6;
    ensure!(q.len() == n * 6, "query shape");
    const CHUNKS: usize = 48;
    if a[1] == "prepare" {
        let mut ap = BTreeMap::new();
        for s in fs::read_to_string(&a[3])?.lines() {
            if s.len() < 44 {
                continue;
            }
            let y: i32 = s[0..3].trim().parse()?;
            let y = if y >= 58 { 1900 + y } else { 2000 + y };
            let m: u32 = s[3..6].trim().parse()?;
            let d: u32 = s[6..9].trim().parse()?;
            let vals: (Vec<f64>, f64) = (
                (9..33)
                    .step_by(3)
                    .map(|i| s[i..i + 3].trim().parse())
                    .collect::<std::result::Result<_, _>>()?,
                s[39..44].trim().parse()?,
            );
            ap.insert((y, m, d), vals);
        }
        let dir = Path::new(&a[4]);
        fs::create_dir_all(dir)?;
        let mut files = (0..CHUNKS)
            .map(|i| fs::File::create(dir.join(format!("chunk-{i:02}.inp"))))
            .collect::<std::io::Result<Vec<_>>>()?;
        let aptable = [
            0., 2., 3., 4., 5., 6., 7., 9., 12., 15., 18., 22., 27., 32., 39., 48., 56., 67., 80.,
            94., 111., 132., 154., 179., 207., 236., 300., 400.,
        ];
        for (i, p) in q.chunks_exact(6).enumerate() {
            let t = DateTime::<Utc>::from_timestamp(p[0] as i64, 0).unwrap();
            let v = &ap[&(t.year(), t.month(), t.day())];
            let aa = v.0[(t.hour() / 3) as usize];
            let k = aptable
                .iter()
                .position(|x| (*x - aa).abs() < 0.1)
                .ok_or_else(|| anyhow::anyhow!("Ap to Kp conversion unsupported: {aa}"))?
                as f64
                / 3.;
            let ut = t.hour() as f64 + t.minute() as f64 / 60. + t.second() as f64 / 3600.;
            writeln!(
                files[i % CHUNKS],
                "{} {} {:.9} {:.9} {:.9} {:.9} {:.9}",
                t.year(),
                t.ordinal(),
                ut,
                p[3],
                p[4],
                p[5],
                k
            )?;
        }
        println!("GCPM prepared {n} points in {CHUNKS} independent processes");
    } else {
        ensure!(a[1] == "collect", "unknown command");
        let dir = Path::new(&a[3]);
        let mut dens = vec![f64::NAN; n * 7];
        let mut fr = vec![f64::NAN; n * 7];
        let mut ne = vec![f64::NAN; n];
        for k in 0..CHUNKS {
            let text = fs::read_to_string(dir.join(format!("chunk-{k:02}.out")))?;
            let rows: Vec<_> = text.lines().filter(|s| !s.trim().is_empty()).collect();
            let indices: Vec<_> = (k..n).step_by(CHUNKS).collect();
            ensure!(rows.len() == indices.len(), "GCPM chunk{k} incomplete");
            for (line, i) in rows.iter().zip(indices) {
                let r: Vec<f64> = line
                    .split_whitespace()
                    .map(str::parse)
                    .collect::<std::result::Result<_, _>>()?;
                ensure!(
                    r.len() == 5
                        && (r[0] - q[i * 6 + 5]).abs() < 0.1
                        && r[1..].iter().all(|v| v.is_finite() && *v >= 0.)
                        && r[1] > 0.,
                    "invalid GCPM output {i}"
                );
                ensure!(
                    ((r[2] + r[3] + r[4]) / r[1] - 1.).abs() < 1e-4,
                    "GCPM quasineutrality"
                );
                ne[i] = r[1];
                for s in 0..7 {
                    dens[7 * i + s] = if s < 3 { r[s + 2] } else { 0. };
                    fr[7 * i + s] = dens[7 * i + s] / ne[i];
                }
            }
        }
        let out = hdf5::File::create(&a[4])?;
        out.new_dataset::<f64>()
            .shape((n, 7))
            .deflate(4)
            .create("ion_density_cm3")?
            .write_raw(&dens)?;
        out.new_dataset::<f64>()
            .shape((n, 7))
            .deflate(4)
            .create("ion_fraction")?
            .write_raw(&fr)?;
        out.new_dataset_builder().with_data(&ne).create("ne_cm3")?;
        out.new_dataset::<f64>()
            .shape((n, 6))
            .create("query_points")?
            .write_raw(&q)?;
        let meta = serde_json::json!({"model":"Legacy GCPMv2.4 with bundled IRI2007; core science sources unmodified; historical solar/geomagnetic index tables updated for2025; bundled legacy IGRF extrapolated beyond original epoch; exploratory reference","species_order":["H+","He+","O+","N+","NO+","O2+","N2+"],"activity":"Three-hour Kp mapped exactly from historical Ap; legacy IRI density/F107 uses refreshed historical tables","units":"cm^-3 and dimensionless total-ion fractions","points":n});
        out.new_dataset_builder()
            .with_data(&serde_json::to_vec(&meta)?)
            .create("metadata_json_utf8")?;
        println!("GCPM valid output {n} points");
    }
    Ok(())
}
