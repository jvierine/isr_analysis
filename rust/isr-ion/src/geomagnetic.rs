//! IGRF-14 field for 2020–2030, evaluated in Rust from NOAA model coefficients.
use anyhow::{ensure, Result};
#[path = "geomagnetic_coefficients.rs"]
mod coefficients;
use coefficients::COEFFICIENTS;
fn legendre(theta: f64) -> [[f64; 14]; 14] {
    let mut p = [[0.; 14]; 14];
    p[0][0] = 1.;
    for m in 0..=13 {
        if m > 0 {
            p[m][m] = (2 * m - 1) as f64 * theta.sin() * p[m - 1][m - 1];
        }
        if m < 13 {
            p[m + 1][m] = (2 * m + 1) as f64 * theta.cos() * p[m][m];
        }
        for n in m + 2..=13 {
            p[n][m] = ((2 * n - 1) as f64 * theta.cos() * p[n - 1][m]
                - (n + m - 1) as f64 * p[n - 2][m])
                / (n - m) as f64;
        }
    }
    for n in 1..=13 {
        for m in 1..=n {
            let mut norm = 2.;
            for k in n - m + 1..=n + m {
                norm /= k as f64;
            }
            p[n][m] *= norm.sqrt();
        }
    }
    p
}
/// North, east, down field in nT at WGS84 geodetic location and ellipsoid height.
/// Field evaluation supports ionospheric altitudes; no WMM 850-km restriction.
pub fn igrf14(
    year: f64,
    latitude_deg: f64,
    longitude_deg: f64,
    height_km: f64,
) -> Result<[f64; 3]> {
    ensure!(
        [year, latitude_deg, longitude_deg, height_km]
            .iter()
            .all(|x| x.is_finite())
            && (2020. ..=2030.).contains(&year)
            && latitude_deg.abs() < 89.9
            && (-1. ..=3000.).contains(&height_km),
        "IGRF input outside implemented domain"
    );
    let phi = latitude_deg.to_radians();
    let lon = longitude_deg.to_radians();
    let ecc = 6.6943799901413165e-3;
    let normal = 6378.137 / (1. - ecc * phi.sin().powi(2)).sqrt();
    let x = (normal + height_km) * phi.cos();
    let z = (normal * (1. - ecc) + height_km) * phi.sin();
    let radius = x.hypot(z);
    let lat = z.atan2(x);
    let theta = std::f64::consts::FRAC_PI_2 - lat;
    let p = legendre(theta);
    let step = 1e-6;
    let plus = legendre(theta + step);
    let minus = legendre(theta - step);
    let mut br = 0.;
    let mut bt = 0.;
    let mut bp = 0.;
    for &(n, m, kind, a, b, sv) in COEFFICIENTS {
        let coeff = if year <= 2025. {
            a + (b - a) * (year - 2020.) / 5.
        } else {
            b + sv * (year - 2025.)
        };
        let angle = m as f64 * lon;
        let (value, dphi) = if kind == 0 {
            (coeff * angle.cos(), -(m as f64) * coeff * angle.sin())
        } else {
            (coeff * angle.sin(), m as f64 * coeff * angle.cos())
        };
        let rad = (6371.2 / radius).powi(n as i32 + 2);
        br += (n + 1) as f64 * rad * value * p[n][m];
        bt -= rad * value * (plus[n][m] - minus[n][m]) / (2. * step);
        bp -= rad * dphi * p[n][m] / theta.sin();
    }
    let north = -bt;
    let down = -br;
    let delta = phi - lat;
    Ok([
        north * delta.cos() + down * delta.sin(),
        bp,
        down * delta.cos() - north * delta.sin(),
    ])
}
/// Central-dipole magnetic longitude in radians, using the same dated IGRF.
pub fn dipole_longitude(year: f64, latitude_deg: f64, longitude_deg: f64) -> Result<f64> {
    ensure!((2020. ..=2030.).contains(&year), "IGRF epoch");
    let coefficient = |m, kind| {
        let &(_, _, _, a, b, sv) = COEFFICIENTS
            .iter()
            .find(|&&(n, j, k, _, _, _)| n == 1 && j == m && k == kind)
            .unwrap();
        if year <= 2025. {
            a + (b - a) * (year - 2020.) / 5.
        } else {
            b + sv * (year - 2025.)
        }
    };
    let g10 = coefficient(0, 0);
    let g11 = coefficient(1, 0);
    let h11 = coefficient(1, 1);
    let norm = (g10 * g10 + g11 * g11 + h11 * h11).sqrt();
    let pole = [-g11 / norm, -h11 / norm, -g10 / norm];
    let xy = pole[0].hypot(pole[1]);
    let east = [-pole[1] / xy, pole[0] / xy, 0.];
    let zero = [
        east[1] * pole[2],
        -east[0] * pole[2],
        east[0] * pole[1] - east[1] * pole[0],
    ];
    let lat = latitude_deg.to_radians();
    let lon = longitude_deg.to_radians();
    let pos = [lat.cos() * lon.cos(), lat.cos() * lon.sin(), lat.sin()];
    let dot = |v: [f64; 3]| pos.iter().zip(v).map(|(a, b)| a * b).sum::<f64>();
    Ok(dot(east).atan2(dot(zero)))
}
