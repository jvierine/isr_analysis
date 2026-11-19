//! Sample external SAMI2 meridian runs at matched gate points, preserving all seven ions.
use anyhow::{ensure, Result};
use std::{fs, path::Path};
fn records(p: &Path) -> Result<Vec<Vec<f64>>> {
    let b = fs::read(p)?;
    let mut i = 0;
    let mut out = vec![];
    while i < b.len() {
        ensure!(i + 8 <= b.len(), "truncated Fortran record");
        let n = u32::from_le_bytes(b[i..i + 4].try_into()?) as usize;
        ensure!(n % 4 == 0 && i + n + 8 <= b.len(), "invalid Fortran record");
        ensure!(
            u32::from_le_bytes(b[i + n + 4..i + n + 8].try_into()?) as usize == n,
            "record trailer"
        );
        out.push(
            b[i + 4..i + 4 + n]
                .chunks_exact(4)
                .map(|v| f32::from_le_bytes(v.try_into().unwrap()) as f64)
                .collect(),
        );
        i += n + 8;
    }
    Ok(out)
}
fn bary(x: f64, y: f64, pts: [[f64; 2]; 3]) -> Option<[f64; 3]> {
    let [[ax, ay], [bx, by], [cx, cy]] = pts;
    let det = (by - cy) * (ax - cx) + (cx - bx) * (ay - cy);
    if det.abs() < 1e-12 {
        return None;
    }
    let u = ((by - cy) * (x - cx) + (cx - bx) * (y - cy)) / det;
    let v = ((cy - ay) * (x - cx) + (ax - cx) * (y - cy)) / det;
    let w = 1. - u - v;
    if [u, v, w].iter().all(|v| *v >= -1e-8 && *v <= 1. + 1e-8) {
        Some([u, v, w])
    } else {
        None
    }
}
struct Plane {
    z: Vec<f64>,
    lat: Vec<f64>,
    lon: Vec<f64>,
    times: Vec<f64>,
    density: Vec<Vec<f64>>,
}
fn main() -> Result<()> {
    let a: Vec<_> = std::env::args().collect();
    ensure!(a.len() == 4, "QUERY_HDF SAMI2_RUN_ROOT OUTPUT_HDF");
    let f = hdf5::File::open(&a[1])?;
    let q = f.dataset("iri_query_points")?.read_raw::<f64>()?;
    let n = q.len() / 6;
    let nz = 101;
    let nf = 98;
    let grid = nz * nf;
    let mut planes = vec![];
    for e in fs::read_dir(&a[2])? {
        let p = e?.path();
        if !p.is_dir() || !p.join("deniu.dat").exists() {
            continue;
        }
        let z = records(&p.join("zaltu.dat"))?.remove(0);
        let lat = records(&p.join("glatu.dat"))?.remove(0);
        let lon = records(&p.join("glonu.dat"))?.remove(0);
        ensure!(
            z.len() == grid && lat.len() == grid && lon.len() == grid,
            "unsupported SAMI2 grid dimensions"
        );
        let d = records(&p.join("deniu.dat"))?;
        let mut times = vec![];
        let mut day = 24.;
        let mut last = 0.;
        for l in fs::read_to_string(p.join("time.dat"))?.lines() {
            let r: Vec<f64> = l
                .split_whitespace()
                .map(str::parse)
                .collect::<std::result::Result<_, _>>()?;
            ensure!(r.len() == 4, "SAMI2 time columns");
            let hr = r[1] + r[2] / 60. + r[3] / 3600.;
            if !times.is_empty() && hr < last - 1. {
                day += 24.;
            }
            times.push(day + hr);
            last = hr;
        }
        ensure!(
            d.len() == times.len()
                && d.len() > 90
                && times.last().is_some_and(|t| *t >= 49.9)
                && d.iter().all(|r| r.len() == grid * 7),
            "incomplete SAMI2 output"
        );
        planes.push(Plane {
            z,
            lat,
            lon,
            times,
            density: d,
        });
    }
    ensure!(
        planes.len() == 6,
        "multiple completed longitude planes required"
    );
    let mut densities = vec![f64::NAN; n * 7];
    let mut fractions = vec![f64::NAN; n * 7];
    let mut missing = 0;
    let mut cache = std::collections::BTreeMap::new();
    for (idx, p) in q.chunks_exact(6).enumerate() {
        let target = 24. + p[0].rem_euclid(86400.) / 3600.;
        let key = ((p[3] * 1e7).round() as i64, (p[5] * 1e5).round() as i64);
        let locations = cache.entry(key).or_insert_with(|| {
            planes
                .iter()
                .enumerate()
                .filter_map(|(pi, pl)| {
                    for j in 0..nf - 1 {
                        for i in 0..nz - 1 {
                            let a = i + nz * j;
                            let b = a + 1;
                            let c = a + nz;
                            let d = c + 1;
                            for ids in [[a, b, d], [a, d, c]] {
                                if let Some(w) = bary(p[3], p[5], ids.map(|k| [pl.lat[k], pl.z[k]]))
                                {
                                    let lon = (0..3).map(|k| w[k] * pl.lon[ids[k]]).sum::<f64>();
                                    return Some((pi, ids, w, lon));
                                }
                            }
                        }
                    }
                    None
                })
                .collect::<Vec<_>>()
        });
        let mut values = vec![];
        for &(pi, ids, w, lon) in locations.iter() {
            let pl = &planes[pi];
            let k = pl
                .times
                .partition_point(|t| *t <= target)
                .saturating_sub(1)
                .min(pl.times.len() - 2);
            if target < pl.times[0] - 0.02 || target > *pl.times.last().unwrap() + 0.02 {
                continue;
            }
            let alpha = ((target - pl.times[k]) / (pl.times[k + 1] - pl.times[k])).clamp(0., 1.);
            let mut v = [0.; 7];
            for s in 0..7 {
                for t in 0..3 {
                    v[s] += w[t]
                        * ((1. - alpha) * pl.density[k][ids[t] + s * grid]
                            + alpha * pl.density[k + 1][ids[t] + s * grid]);
                }
            }
            values.push((lon, v));
        }
        values.sort_by(|a, b| a.0.total_cmp(&b.0));
        let b = values.partition_point(|v| v.0 < p[4]);
        if b == 0 || b == values.len() {
            missing += 1;
            continue;
        }
        let (lon0, v0) = values[b - 1];
        let (lon1, v1) = values[b];
        let alpha = (p[4] - lon0) / (lon1 - lon0);
        let raw: Vec<f64> = (0..7)
            .map(|s| v0[s] * (1. - alpha) + v1[s] * alpha)
            .collect();
        ensure!(
            raw.iter().all(|v| v.is_finite() && *v >= 0.),
            "invalid SAMI2 density"
        );
        let ne: f64 = raw.iter().sum();
        ensure!(ne > 0., "SAMI2 charge density");
        for (s, src) in [0, 4, 1, 6, 2, 3, 5].iter().enumerate() {
            densities[7 * idx + s] = raw[*src];
            fractions[7 * idx + s] = raw[*src] / ne;
        }
    }
    let out = hdf5::File::create(&a[3])?;
    out.new_dataset::<f64>()
        .shape((n, 7))
        .deflate(4)
        .create("ion_density_cm3")?
        .write_raw(&densities)?;
    out.new_dataset::<f64>()
        .shape((n, 7))
        .deflate(4)
        .create("ion_fraction")?
        .write_raw(&fractions)?;
    out.new_dataset::<f64>()
        .shape((n, 6))
        .create("query_points")?
        .write_raw(&q)?;
    let meta = serde_json::json!({"model":"SAMI2-1.00 via sami2py, 6 reference meridians, 24h spinup; seasonal-day October6 2025 reference, constant F107=133,F107a=154.3,Ap=4,HWM14,Fejer-Scherliess ExB","sampling":"Barycentric latitude-altitude interpolation within each curvilinear plane, linear interpolation across bracketing actual plane longitudes and in model time. Densities interpolated before fractions. UT phase after spinup; one fixed seasonal-day reference, not event-driver history. Outside-domain points remain NaN.","species_order":["H+","He+","O+","N+","NO+","O2+","N2+"],"query_points":n,"outside_grid":missing,"complete_meridians":planes.len()});
    out.new_dataset_builder()
        .with_data(&serde_json::to_vec(&meta)?)
        .create("metadata_json_utf8")?;
    println!("SAMI2 sampled {}/{} points", n - missing, n);
    ensure!(
        missing * 20 < n,
        "more than5% of requested points are outside SAMI2 grid"
    );
    Ok(())
}
#[cfg(test)]
mod tests {
    #[test]
    fn barycentric_linear_field() {
        let w = super::bary(0.25, 0.5, [[0., 0.], [1., 0.], [0., 1.]]).unwrap();
        let y = w[0] * 1. + w[1] * 3. + w[2] * 4.;
        assert!((y - 3.).abs() < 1e-12);
        assert!(super::bary(2., 2., [[0., 0.], [1., 0.], [0., 1.]]).is_none());
    }
}
