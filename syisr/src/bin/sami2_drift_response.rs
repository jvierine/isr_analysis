//! Compare measured composition with paired native SAMI2 drift sensitivity cases.
use anyhow::{ensure, Result};
use serde_json::{json, Value};
use std::{collections::BTreeMap, fs, path::Path};
use syisr_ac::{campaign, h_transition::Transition, spline::h_weight};
fn records(p: &Path) -> Result<Vec<Vec<f64>>> {
    let b = fs::read(p)?;
    let mut out = vec![];
    let mut i = 0;
    while i < b.len() {
        ensure!(i + 8 <= b.len(), "truncated record");
        let n = u32::from_le_bytes(b[i..i + 4].try_into()?) as usize;
        ensure!(
            n % 4 == 0
                && i + n + 8 <= b.len()
                && u32::from_le_bytes(b[i + n + 4..i + n + 8].try_into()?) as usize == n,
            "Fortranrecord"
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
fn bary(x: f64, y: f64, p: [[f64; 2]; 3]) -> Option<[f64; 3]> {
    let [[ax, ay], [bx, by], [cx, cy]] = p;
    let d = (by - cy) * (ax - cx) + (cx - bx) * (ay - cy);
    if d.abs() < 1e-12 {
        return None;
    }
    let u = ((by - cy) * (x - cx) + (cx - bx) * (y - cy)) / d;
    let v = ((cy - ay) * (x - cx) + (ax - cx) * (y - cy)) / d;
    let w = 1. - u - v;
    if [u, v, w].iter().all(|v| *v >= -1e-8 && *v <= 1. + 1e-8) {
        Some([u, v, w])
    } else {
        None
    }
}
struct Case {
    times: Vec<f64>,
    d: Vec<Vec<f64>>,
    locations: Vec<Option<([usize; 3], [f64; 3])>>,
    lons: Vec<f64>,
}
fn load(p: &Path, zs: &[f64]) -> Result<Case> {
    let z = records(&p.join("zaltu.dat"))?.remove(0);
    let lat = records(&p.join("glatu.dat"))?.remove(0);
    let lon = records(&p.join("glonu.dat"))?.remove(0);
    ensure!(z.len() == 101 * 98, "unsupportedSAMI2grid");
    let mut times = vec![];
    let mut day = 0.;
    let mut last = 0.;
    for line in fs::read_to_string(p.join("time.dat"))?.lines() {
        let q: Vec<f64> = line
            .split_whitespace()
            .map(str::parse)
            .collect::<std::result::Result<_, _>>()?;
        ensure!(q.len() == 4, "timecolumns");
        let t = q[1] + q[2] / 60. + q[3] / 3600.;
        if !times.is_empty() && t < last - 1. {
            day += 24.;
        }
        times.push(day + t);
        last = t;
    }
    let d = records(&p.join("deniu.dat"))?;
    ensure!(
        d.len() == times.len()
            && times.last().is_some_and(|t| *t >= 49.9)
            && d.iter().all(|r| r.len() == 101 * 98 * 7),
        "incompleteSAMI2case"
    );
    let mut locations = vec![];
    let mut lons = vec![];
    for target in zs {
        let mut found = None;
        for j in 0..97 {
            for i in 0..100 {
                let a = i + 101 * j;
                for ids in [[a, a + 1, a + 102], [a, a + 102, a + 101]] {
                    if let Some(w) = bary(18.3492, *target, ids.map(|k| [lat[k], z[k]])) {
                        found = Some((ids, w));
                        break;
                    }
                }
                if found.is_some() {
                    break;
                }
            }
            if found.is_some() {
                break;
            }
        }
        lons.push(
            found
                .map(|(ids, w)| (0..3).map(|k| w[k] * lon[ids[k]]).sum())
                .unwrap_or(f64::NAN),
        );
        locations.push(found);
    }
    Ok(Case {
        times,
        d,
        locations,
        lons,
    })
}
fn sample(c: &Case, z: usize, t: f64) -> Option<[f64; 7]> {
    let (ids, w) = c.locations[z]?;
    if t < c.times[0] || t > *c.times.last()? {
        return None;
    }
    let k = c
        .times
        .partition_point(|v| *v <= t)
        .saturating_sub(1)
        .min(c.times.len() - 2);
    let a = (t - c.times[k]) / (c.times[k + 1] - c.times[k]);
    let mut n = [0.; 7];
    for s in 0..7 {
        for j in 0..3 {
            n[s] +=
                w[j] * ((1. - a) * c.d[k][ids[j] + s * 9898] + a * c.d[k + 1][ids[j] + s * 9898]);
        }
    }
    Some(n)
}
fn crossing(z: &[f64], h: &[f64]) -> Option<f64> {
    for i in 0..h.len() - 1 {
        if h[i].is_finite()
            && h[i + 1].is_finite()
            && h[i] <= 0.5
            && h[i + 1] >= 0.5
            && h[i + 1] > h[i]
        {
            return Some(z[i] + (z[i + 1] - z[i]) * (0.5 - h[i]) / (h[i + 1] - h[i]));
        }
    }
    None
}
fn main() -> Result<()> {
    let a: Vec<_> = std::env::args().collect();
    ensure!(a.len() == 4, "RUN_ROOT FIT_SNAPSHOT OUTPUT");
    let root = Path::new(&a[1]);
    let mut windows = BTreeMap::new();
    for p in campaign::files(&Path::new(&a[2]).join("shared-windows"), "h5")? {
        let v: Value = campaign::load(&p)?;
        if v["model_inferred_only"] == true {
            continue;
        }
        windows.insert(v["start"].as_i64().unwrap(), v);
    }
    let times: Vec<_> = windows.keys().copied().collect();
    let zs: Vec<_> = (500..=1300).step_by(10).map(|z| z as f64).collect();
    let nt = times.len();
    let nz = zs.len();
    let names = ["fit", "baseline", "measured", "measured_reversed"];
    let mut data = vec![vec![f64::NAN; nt * nz]; 4];
    let mut he = vec![vec![f64::NAN; nt * nz]; 4];
    let mut longitude = vec![];
    for (t, (time, w)) in windows.iter().enumerate() {
        let tr: Transition = serde_json::from_value(w["h_transition"].clone())?;
        let layer: syisr_ac::h_transition::HeliumLayer =
            serde_json::from_value(w["helium_layer"].clone())?;
        for (k, z) in zs.iter().enumerate() {
            let h = tr.shape.evaluate(*z, tr.normalized_parameters).0 * h_weight(*z);
            data[0][k * nt + t] = h;
            he[0][k * nt + t] = (1. - h)
                * syisr_ac::h_transition::helium(*z, layer.normalized_parameters).0
                * h_weight(*z);
        }
        ensure!(*time >= 1759705200, "dataepoch");
    }
    for (m, name) in ["baseline", "measured", "measured-reversed"]
        .iter()
        .enumerate()
    {
        let c = load(&root.join(name), &zs)?;
        if m == 0 {
            longitude = c.lons.clone();
        }
        for (t, time) in times.iter().enumerate() {
            let start = (*time as f64 - 1759622400.) / 3600.;
            for k in 0..nz {
                let mut n = [0.; 7];
                let mut complete = true;
                for (q, weight) in [1., 4., 2., 4., 1.].iter().enumerate() {
                    if let Some(v) = sample(&c, k, start + 0.125 * q as f64) {
                        for s in 0..7 {
                            n[s] += weight * v[s] / 12.;
                        }
                    } else {
                        complete = false;
                    }
                }
                if complete {
                    ensure!(
                        n.iter().all(|v| v.is_finite() && *v >= 0.) && n.iter().sum::<f64>() > 0.,
                        "invalid modelions"
                    );
                    let ne = n.iter().sum::<f64>();
                    data[m + 1][k * nt + t] = n[0] / ne;
                    he[m + 1][k * nt + t] = n[4] / ne;
                }
            }
        }
    }
    let out = hdf5::File::create(&a[3])?;
    out.new_dataset_builder()
        .with_data(&times)
        .create("start_utc_s")?;
    out.new_dataset_builder()
        .with_data(&zs)
        .create("altitude_km")?;
    out.new_dataset_builder()
        .with_data(&longitude)
        .create("model_longitude_deg")?;
    let fit_h50: Vec<_> = (0..nt)
        .map(|t| {
            crossing(
                &zs,
                &(0..nz).map(|k| data[0][k * nt + t]).collect::<Vec<_>>(),
            )
            .unwrap_or(f64::NAN)
        })
        .collect();
    let mut summary = vec![];
    for m in 0..4 {
        out.new_dataset::<f64>()
            .shape((nz, nt))
            .create(format!("{}_h_fraction", names[m]).as_str())?
            .write_raw(&data[m])?;
        out.new_dataset::<f64>()
            .shape((nz, nt))
            .create(format!("{}_he_fraction", names[m]).as_str())?
            .write_raw(&he[m])?;
        let h50: Vec<_> = (0..nt)
            .map(|t| {
                crossing(
                    &zs,
                    &(0..nz).map(|k| data[m][k * nt + t]).collect::<Vec<_>>(),
                )
                .unwrap_or(f64::NAN)
            })
            .collect();
        out.new_dataset_builder()
            .with_data(&h50)
            .create(format!("{}_h50_km", names[m]).as_str())?;
        if m > 0 {
            let mut residual = vec![];
            for k in 0..nz {
                for t in 0..nt {
                    let a = data[m][k * nt + t];
                    let b = data[0][k * nt + t];
                    if a.is_finite() && b.is_finite() {
                        residual.push(a - b);
                    }
                }
            }
            let height_residual: Vec<_> = h50
                .iter()
                .zip(&fit_h50)
                .filter_map(|(a, b)| {
                    if a.is_finite() && b.is_finite() {
                        Some(a - b)
                    } else {
                        None
                    }
                })
                .collect();
            summary.push(json!({"case":names[m],"comparison_points":residual.len(),"fraction_rmse":(residual.iter().map(|v|v*v).sum::<f64>()/residual.len()as f64).sqrt(),"h50_matched_windows":height_residual.len(),"h50_rmse_km":(height_residual.iter().map(|v|v*v).sum::<f64>()/height_residual.len()as f64).sqrt(),"h50_mae_km":height_residual.iter().map(|v|v.abs()).sum::<f64>()/height_residual.len()as f64,"fraction_mae":residual.iter().map(|v|v.abs()).sum::<f64>()/residual.len()as f64}));
        }
    }
    let meta = json!({"cases":names,"qualification":"PairedSAMI2sensitivity; baselineFejer-Scherliess,measuredandoppositesign measuredmeridionalforcing,allHWM14fixed. Absolutehardwarevelocitypolarityunverified; nomodel-basedsignselection.","sampling":"SingleSAMI2planeanchoredatSanya300km; barycentricinterpolationat18.3492Nand500-1300km; actuallongitudeofplane recordedbyaltitude,notassumedconstant.5pointSimpsonaverageofiondensitiesovereach30minwindow,thenfractions. Missingcrossingsoutside500-1300retainedNaN.","seasonal_inputs":"FixedOct6day279,F107133,F107a154.3,Ap4; ~23.5hempiricalspinup, measuredforcingonlyobservedinterval; noevent-specificsolarhistoryreconstruction.","summary":summary});
    out.new_dataset_builder()
        .with_data(&serde_json::to_vec(&meta)?)
        .create("metadata_json_utf8")?;
    println!("{}", serde_json::to_string_pretty(&meta)?);
    Ok(())
}

#[cfg(test)]
mod tests {
    use super::*;
    #[test]
    fn native_grid_interpolation_preserves_linear_density() {
        let p = [[10., 500.], [20., 500.], [20., 1000.]];
        let w = bary(17., 650., p).unwrap();
        assert!((w.iter().sum::<f64>() - 1.).abs() < 1e-12);
        let n = (0..3)
            .map(|k| w[k] * (2. * p[k][0] + 3. * p[k][1]))
            .sum::<f64>();
        assert!((n - (2. * 17. + 3. * 650.)).abs() < 1e-10);
        assert!(bary(25., 650., p).is_none());
    }
    #[test]
    fn transition_height_does_not_extrapolate_outside_observed_domain() {
        assert_eq!(crossing(&[500., 1000., 1300.], &[0.1, 0.4, 0.49]), None);
        assert_eq!(
            crossing(&[500., 1000., 1300.], &[0.1, 0.5, 0.8]),
            Some(1000.)
        );
        assert_eq!(crossing(&[500., 1000., 1300.], &[0.1, f64::NAN, 0.8]), None);
    }
}
