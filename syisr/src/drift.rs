//! Regional velocity inversion: SYISR LOS positive toward radar, ENU vectors east/north/up positive.
use anyhow::{ensure, Result};
use nalgebra::{DMatrix, DVector, Vector3 as V};
use serde::{Deserialize, Serialize};
#[derive(Clone, Serialize, Deserialize)]
pub struct Geometry {
    pub latitude_deg: f64,
    pub longitude_deg: f64,
    pub range_km: f64,
    pub enu_row: [f64; 3],
    pub magnetic_row: [f64; 3],
    pub field_ned_nt: [f64; 3],
}
fn axes(lat: f64, lon: f64) -> [V<f64>; 3] {
    [
        V::new(-lon.sin(), lon.cos(), 0.),
        V::new(-lat.sin() * lon.cos(), -lat.sin() * lon.sin(), lat.cos()),
        V::new(lat.cos() * lon.cos(), lat.cos() * lon.sin(), lat.sin()),
    ]
}
pub fn geometry(z: f64, az: f64, el: f64) -> Result<Geometry> {
    let (lat, lon) = (18.3492_f64.to_radians(), 109.6222_f64.to_radians());
    let [east, north, up] = axes(lat, lon);
    let (a, e) = (az.to_radians(), el.to_radians());
    let ray = up * e.sin() + (north * a.cos() + east * a.sin()) * e.cos();
    let r = crate::campaign::range(z, el);
    let pos = up * 6371.05 + ray * r;
    let plat = (pos.z / pos.norm()).asin();
    let plon = pos.y.atan2(pos.x);
    let enu = axes(plat, plon);
    let b = crate::geomagnetic::igrf14(2025.76, plat.to_degrees(), plon.to_degrees(), z)?;
    let bv = V::new(b[1], b[0], -b[2]).normalize();
    let me = V::new(bv.y, -bv.x, 0.).normalize();
    let mm = me.cross(&bv).normalize();
    // SYISR LOS is positive toward the radar (Zhang et al. 2024, Sec.2.2).
    // ray points away from the radar; physical velocity therefore projects onto -ray.
    let los = -V::new(ray.dot(&enu[0]), ray.dot(&enu[1]), ray.dot(&enu[2]));
    Ok(Geometry {
        latitude_deg: plat.to_degrees(),
        longitude_deg: plon.to_degrees(),
        range_km: r,
        enu_row: los.into(),
        magnetic_row: [los.dot(&me), los.dot(&mm), los.dot(&bv)],
        field_ned_nt: b,
    })
}
fn median(v: &[f64]) -> f64 {
    let mut s = v.to_vec();
    s.sort_by(f64::total_cmp);
    let n = s.len();
    if n % 2 == 0 {
        (s[n / 2 - 1] + s[n / 2]) / 2.
    } else {
        s[n / 2]
    }
}
#[derive(Clone, Serialize, Deserialize)]
pub struct VectorFit {
    pub velocity_m_s: [f64; 3],
    pub condition_number: f64,
    pub residual_m_s: Vec<f64>,
    pub weights: Vec<f64>,
    pub residual_rms_m_s: f64,
    pub robust_scale_m_s: f64,
    pub jackknife_spread_m_s: [f64; 3],
    pub maximum_leave_one_out_error_m_s: f64,
    pub consistency_ok: bool,
}
fn ls(rows: &[[f64; 3]], y: &[f64], w: &[f64]) -> Result<([f64; 3], f64)> {
    let a = DMatrix::from_fn(rows.len(), 3, |i, j| rows[i][j] * w[i].sqrt());
    let b = DVector::from_iterator(y.len(), y.iter().zip(w).map(|(v, w)| v * w.sqrt()));
    let sv = a.svd(true, true);
    let hi = sv.singular_values.max();
    let lo = sv.singular_values.min();
    ensure!(lo > 1e-8, "rank deficient multibeam geometry");
    let x = sv.solve(&b, 1e-8).map_err(|e| anyhow::anyhow!(e))?;
    Ok(([x[0], x[1], x[2]], hi / lo))
}
fn residual(rows: &[[f64; 3]], y: &[f64], x: [f64; 3]) -> Vec<f64> {
    rows.iter()
        .zip(y)
        .map(|(r, v)| v - r.iter().zip(x).map(|(a, b)| a * b).sum::<f64>())
        .collect()
}
pub fn invert(rows: &[[f64; 3]], y: &[f64]) -> Result<VectorFit> {
    ensure!(
        rows.len() == y.len() && y.len() >= 5,
        "at least five usable LOS measurements required"
    );
    ensure!(
        y.iter().all(|v| v.is_finite()) && rows.iter().flatten().all(|v| v.is_finite()),
        "nonfinite LOS input"
    );
    let mut w = vec![1.; y.len()];
    let (mut x, mut cond) = ls(rows, y, &w)?;
    let mut scale = 5.;
    for _ in 0..30 {
        let r = residual(rows, y, x);
        let m = median(&r);
        scale = (1.4826 * median(&r.iter().map(|a| (a - m).abs()).collect::<Vec<_>>())).max(5.);
        for (i, r) in r.iter().enumerate() {
            w[i] = (1.5 * scale / r.abs().max(1e-12)).min(1.);
        }
        let (next, c) = ls(rows, y, &w)?;
        let d = x
            .iter()
            .zip(next)
            .map(|(a, b)| (a - b).powi(2))
            .sum::<f64>()
            .sqrt();
        x = next;
        cond = c;
        if d < 1e-5 {
            break;
        }
    }
    let r = residual(rows, y, x);
    let rms = (r.iter().map(|v| v * v).sum::<f64>() / r.len() as f64).sqrt();
    let mut leave = vec![];
    let mut maxerr = 0f64;
    for skip in 0..y.len() {
        let rr: Vec<_> = rows
            .iter()
            .enumerate()
            .filter(|(i, _)| *i != skip)
            .map(|(_, r)| *r)
            .collect();
        let yy: Vec<_> = y
            .iter()
            .enumerate()
            .filter(|(i, _)| *i != skip)
            .map(|(_, v)| *v)
            .collect();
        let ww: Vec<_> = w
            .iter()
            .enumerate()
            .filter(|(i, _)| *i != skip)
            .map(|(_, v)| *v)
            .collect();
        if let Ok((v, c)) = ls(&rr, &yy, &ww) {
            if c < 30. {
                maxerr = maxerr.max(
                    (y[skip] - rows[skip].iter().zip(v).map(|(a, b)| a * b).sum::<f64>()).abs(),
                );
                leave.push(v);
            }
        }
    }
    let mut spread = [f64::NAN; 3];
    if leave.len() == y.len() {
        for k in 0..3 {
            let mean = leave.iter().map(|v| v[k]).sum::<f64>() / leave.len() as f64;
            spread[k] = ((leave.len() - 1) as f64 / leave.len() as f64
                * leave.iter().map(|v| (v[k] - mean).powi(2)).sum::<f64>())
            .sqrt();
        }
    }
    let ok =
        cond < 15. && rms < 30. && maxerr < 75. && spread.iter().all(|v| v.is_finite() && *v < 50.);
    Ok(VectorFit {
        velocity_m_s: x,
        condition_number: cond,
        residual_m_s: r,
        weights: w,
        residual_rms_m_s: rms,
        robust_scale_m_s: scale,
        jackknife_spread_m_s: spread,
        maximum_leave_one_out_error_m_s: maxerr,
        consistency_ok: ok,
    })
}
#[cfg(test)]
mod tests {
    use super::*;
    fn rows() -> Vec<[f64; 3]> {
        [
            (88., 60.),
            (88., 75.),
            (178., 45.),
            (178., 60.),
            (178., 75.),
            (268., 60.),
            (268., 75.),
            (358., 75.),
            (358., 90.),
        ]
        .iter()
        .map(|(a, e)| geometry(400., *a, *e).unwrap().magnetic_row)
        .collect()
    }
    #[test]
    fn recovers_projected_vector() {
        let r = rows();
        let truth = [37., -22., 45.];
        let y: Vec<_> = r
            .iter()
            .map(|a| a.iter().zip(truth).map(|(a, b)| a * b).sum())
            .collect();
        let f = invert(&r, &y).unwrap();
        assert!(f.consistency_ok);
        for k in 0..3 {
            assert!((f.velocity_m_s[k] - truth[k]).abs() < 1e-9);
        }
        assert!(f.maximum_leave_one_out_error_m_s < 1e-9);
    }
    #[test]
    fn flags_inconsistent_beam() {
        let r = rows();
        let mut y = vec![0.; r.len()];
        y[0] = 300.;
        let f = invert(&r, &y).unwrap();
        assert!(!f.consistency_ok);
        assert!(f.weights[0] < 0.2);
    }
    #[test]
    fn refuses_rank_deficiency() {
        assert!(invert(&vec![[0., 0., 1.]; 6], &vec![1.; 6]).is_err());
    }
    #[test]
    fn zenith_geometry_and_field_basis() {
        let g = geometry(400., 358., 90.).unwrap();
        assert!(g.enu_row[2] < -0.999999);
        assert!(g.enu_row[0].abs() < 1e-10);
        assert!((g.magnetic_row.iter().map(|v| v * v).sum::<f64>() - 1.).abs() < 1e-12);
        assert!((g.range_km - 399.95).abs() < 1.);
    }
}

#[cfg(test)]
mod polarity_test {
    #[test]
    fn physical_upflow_has_negative_toward_positive_zenith_los() {
        let g = super::geometry(400., 358., 90.).unwrap();
        let physical_velocity = [0., 0., 25.];
        let vlos: f64 = g
            .enu_row
            .iter()
            .zip(physical_velocity)
            .map(|(a, b)| a * b)
            .sum();
        assert!((vlos + 25.).abs() < 1e-10);
        let receding_phase = 4. * std::f64::consts::PI * 430.2e6 / 299792458. * 25. * 10e-6;
        let omega = -4. * std::f64::consts::PI * 430.2e6 / 299792458. * 25.;
        let early = num_complex::Complex64::new(1., 0.);
        let late = num_complex::Complex64::from_polar(1., omega * 10e-6);
        let measured = early * late.conj();
        let predicted = num_complex::Complex64::from_polar(1., receding_phase);
        assert!((measured - predicted).norm() < 1e-12);
    }
}
