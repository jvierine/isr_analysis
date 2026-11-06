//! Danilov–Yaichnikov (1985) atomic-ion composition, as specified by IRI IONCO1.
//! Validated model domain ends at 1000 km; callers must not silently extrapolate.
use anyhow::{ensure, Result};
/// Fractions in O+, H+, N+, He+ order. Seasonal month is Northern-Hemisphere
/// equivalent decimal month (January 15 = 1.5; shift six months in the south).
pub fn dy1985(
    height_km: f64,
    solar_zenith_deg: f64,
    latitude_deg: f64,
    f107: f64,
    seasonal_month: f64,
) -> Result<[f64; 4]> {
    ensure!(
        [
            height_km,
            solar_zenith_deg,
            latitude_deg,
            f107,
            seasonal_month
        ]
        .iter()
        .all(|x| x.is_finite())
            && (75. ..=1000.).contains(&height_km)
            && f107 > 0.,
        "atomic composition outside model domain"
    );
    const P: [[[f64; 5]; 6]; 4] = [
        [
            [0., 0., 0., 0., 98.5],
            [0., 0., 0., 0., 320.],
            [0., 0., 0., 0., -2.59e-4],
            [2.79e-4, -3.33e-3, -3.52e-3, -5.16e-3, -2.47e-2],
            [0., 0., 0., 0., -2.5e-6],
            [1.04e-3, -1.79e-4, -4.29e-5, 1.01e-5, -1.27e-3],
        ],
        [
            [-4.97e-7, -1.21e-1, -1.31e-1, 0., 98.1],
            [355., -191., -127., 0., 2040.],
            [0., 0., 0., 0., -4.79e-6],
            [-2e-4, 5.67e-4, 2.6e-4, 0., -5.08e-3],
            [0.; 5],
            [0.; 5],
        ],
        [
            [0.76, -5.62, -4.99, 0., 5.79],
            [83., -369., -324., 0., 593.],
            [0., 0., 0., 0., -6.3e-5],
            [-6.74e-3, -7.93e-3, -4.65e-3, 0., -3.26e-3],
            [0., 0., 0., 0., -1.17e-5],
            [4.88e-3, -1.31e-3, -7.03e-4, 0., -2.38e-3],
        ],
        [
            [-0.895, 6.1, 5.39, 0., 8.01],
            [0., 0., 0., 0., 1200.],
            [0., 0., 0., 0., -1.04e-5],
            [1.9e-3, 9.53e-4, 1.06e-3, 0., -3.44e-3],
            [0.; 5],
            [0.; 5],
        ],
    ];
    let basis = [
        solar_zenith_deg.to_radians().cos(),
        latitude_deg.to_radians().cos(),
        (0.013 * (300. - f107)).cos(),
        (0.52 * (seasonal_month - 6.)).cos(),
        1.,
    ];
    let mut fractions = [0.; 4];
    for (i, p) in P.iter().enumerate() {
        let v: Vec<f64> = p
            .iter()
            .map(|r| r.iter().zip(basis).map(|(a, b)| a * b).sum())
            .collect();
        let hx = height_km - v[1];
        let arg = if hx < 0. {
            hx * (hx * v[2] + v[3])
        } else {
            hx * (hx * v[4] + v[5])
        };
        let value = v[0] * arg.exp();
        fractions[i] = if value < 0.005 * v[0] {
            0.
        } else {
            value.min(v[0]).max(0.)
        };
    }
    let sum: f64 = fractions.iter().sum();
    ensure!(sum.is_finite() && sum > 0., "invalid atomic composition");
    for x in &mut fractions {
        *x /= sum;
    }
    Ok(fractions)
}
#[cfg(test)]
mod tests {
    use super::*;
    #[test]
    fn normalized_and_bounded_domain() {
        for h in [300., 500., 800., 1000.] {
            let a = dy1985(h, 30., 18.34, 150., 11.5).unwrap();
            assert!((a.iter().sum::<f64>() - 1.).abs() < 1e-12);
            assert!(a.iter().all(|x| *x >= 0.));
        }
        assert!(dy1985(1500., 30., 18., 150., 11.5).is_err());
    }
}
