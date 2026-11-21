//! WGS84 SYISR rays and dated IGRF-14 field-line geometry for LFM morphology.
use anyhow::{ensure, Result};
use nalgebra::Vector3 as V;
pub const LAT: f64 = 18.3492;
pub const LON: f64 = 109.6222;
pub const SITE_KM: f64 = 0.05;
pub const YEAR: f64 = 2026. + 59. / 365.;
pub fn axes(lat: f64, lon: f64) -> [V<f64>; 3] {
    let (a, o) = (lat.to_radians(), lon.to_radians());
    [
        V::new(-o.sin(), o.cos(), 0.),
        V::new(-a.sin() * o.cos(), -a.sin() * o.sin(), a.cos()),
        V::new(a.cos() * o.cos(), a.cos() * o.sin(), a.sin()),
    ]
}
pub fn ecef(lat: f64, lon: f64, h: f64) -> V<f64> {
    let (a, o) = (lat.to_radians(), lon.to_radians());
    let e2 = 6.6943799901413165e-3;
    let n = 6378.137 / (1. - e2 * a.sin().powi(2)).sqrt();
    V::new(
        (n + h) * a.cos() * o.cos(),
        (n + h) * a.cos() * o.sin(),
        (n * (1. - e2) + h) * a.sin(),
    )
}
pub fn geodetic(p: V<f64>) -> [f64; 3] {
    let e2 = 6.6943799901413165e-3;
    let r = p.x.hypot(p.y);
    let mut a = p.z.atan2(r * (1. - e2));
    let mut h = 0.;
    for _ in 0..12 {
        let n = 6378.137 / (1. - e2 * a.sin().powi(2)).sqrt();
        h = r / a.cos() - n;
        let next = p.z.atan2(r * (1. - e2 * n / (n + h)));
        if (next - a).abs() < 1e-13 {
            a = next;
            break;
        }
        a = next;
    }
    [a.to_degrees(), p.y.atan2(p.x).to_degrees(), h]
}
pub fn ray(az: f64, el: f64) -> V<f64> {
    let [e, n, u] = axes(LAT, LON);
    let (a, b) = (az.to_radians(), el.to_radians());
    e * (a.sin() * b.cos()) + n * (a.cos() * b.cos()) + u * b.sin()
}
pub fn position(range: f64, az: f64, el: f64) -> V<f64> {
    ecef(LAT, LON, SITE_KM) + ray(az, el) * range
}
pub fn enu(p: V<f64>) -> V<f64> {
    let d = p - ecef(LAT, LON, SITE_KM);
    let [e, n, u] = axes(LAT, LON);
    V::new(d.dot(&e), d.dot(&n), d.dot(&u))
}
pub fn bunit(p: V<f64>) -> Result<V<f64>> {
    let [a, o, h] = geodetic(p);
    let b = crate::geomagnetic::igrf14(YEAR, a, o, h)?;
    let [e, n, u] = axes(a, o);
    Ok((n * b[0] + e * b[1] - u * b[2]).normalize())
}
pub fn reference() -> Result<(V<f64>, V<f64>, V<f64>, V<f64>)> {
    let p = ecef(LAT, LON, 400.);
    let b = bunit(p)?;
    let [e, _, u] = axes(LAT, LON);
    let e1 = (e - b * e.dot(&b)).normalize();
    let mut e2 = e1.cross(&b).normalize();
    if e2.dot(&u) < 0. {
        e2 = -e2;
    }
    Ok((p, e1, e2, b))
}
pub fn project_plane(p: V<f64>) -> Result<[f64; 3]> {
    let (r, e1, e2, b) = reference()?;
    let d = p - r;
    Ok([d.dot(&e1), d.dot(&e2), d.dot(&b)])
}
pub fn map_field_line(mut p: V<f64>) -> Result<[f64; 3]> {
    let (r, e1, e2, b) = reference()?;
    let start = p;
    let mut length = 0.;
    for _ in 0..800 {
        let distance = (p - r).dot(&b);
        if distance.abs() < 1e-5 {
            return Ok([(p - r).dot(&e1), (p - r).dot(&e2), length]);
        }
        let bp = bunit(p)?;
        let denom = bp.dot(&b);
        ensure!(denom.abs()>0.2,"field line parallel to reference plane");
        let step = (-distance / denom).clamp(-5., 5.);
        let k1 = bp;
        let k2 = bunit(p + k1 * step / 2.)?;
        let k3 = bunit(p + k2 * step / 2.)?;
        let k4 = bunit(p + k3 * step)?;
        p += (k1 + k2 * 2. + k3 * 2. + k4) * step / 6.;
        length += step;
        ensure!((p - start).norm() < 3000., "field-line mapping too long");
    }
    anyhow::bail!("field-line mapping did not converge")
}
#[cfg(test)]
mod tests {
    use super::*;
    #[test]
    fn wgs_roundtrip() {
        for h in [0.05, 200., 400., 900.] {
            let p = ecef(LAT, LON, h);
            let q = geodetic(p);
            assert!((q[0] - LAT).abs() < 1e-9);
            assert!((q[1] - LON).abs() < 1e-9);
            assert!((q[2] - h).abs() < 1e-7);
        }
    }
    #[test]
    fn zenith_height_and_field_plane() {
        let p = position(399.95, 0., 90.);
        assert!((geodetic(p)[2] - 400.).abs() < 1e-7);
        let q = map_field_line(p).unwrap();
        assert!(q[0].abs() < 1e-7 && q[1].abs() < 1e-7);
        let (r, e1, e2, b) = reference().unwrap();
        assert!(e1.dot(&b).abs() < 1e-12 && e2.dot(&b).abs() < 1e-12);
        assert!((b.norm() - 1.).abs() < 1e-12);
        let mapped = map_field_line(r + b * 1.).unwrap();
        assert!(mapped[0].hypot(mapped[1]) < 0.001);
    }
}
