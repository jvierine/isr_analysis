//! CALION v2.5 topside composition (Truhlik et al., 2015), 350–2000 km.
//! Magnetic coordinates and the PF10.7 driver must be supplied explicitly.
use anyhow::{ensure, Result};
#[path = "topside_coefficients.rs"]
mod coefficients;
use coefficients::*;
const MIRROR: [f64; 49] = [
    1., -1., 1., -1., 1., -1., 1., 1., -1., 1., -1., 1., -1., 1., -1., 1., -1., 1., -1., 1., -1.,
    1., -1., 1., 1., -1., 1., -1., 1., 1., -1., 1., -1., 1., -1., 1., -1., 1., -1., 1., 1., -1.,
    1., 1., -1., 1., -1., 1., 1.,
];
fn harmonics(invdip: f64, mlt: f64) -> [f64; 49] {
    let theta = (90. - invdip).to_radians();
    let phi = (15. * mlt).to_radians();
    let x = theta.cos();
    let y = theta.sin();
    let mut c = [0.; 49];
    c[0] = 1.;
    c[1] = x;
    let mut k = 2;
    for i in 2..=6 {
        c[k] = ((2 * i - 1) as f64 * x * c[k - 1] - (i - 1) as f64 * c[k - 2]) / i as f64;
        k += 1;
    }
    for m in 1..=6 {
        c[k] = y.powi(m);
        k += 1;
        if m < 6 {
            c[k] = c[k - 1] * x * (2 * m + 1) as f64;
            k += 1;
            for i in m + 2..=6 {
                c[k] = ((2 * i - 1) as f64 * x * c[k - 1] - (i + m - 1) as f64 * c[k - 2])
                    / (i - m) as f64;
                k += 1;
            }
        }
        let n = (7 - m) as usize;
        for _ in 0..n {
            c[k] = c[k - n] * (m as f64 * phi).sin();
            c[k - n] *= (m as f64 * phi).cos();
            k += 1;
        }
    }
    c
}
fn season(day: u32) -> (usize, usize, f64) {
    match day {
        80..=171 => (0, 1, (day - 80) as f64 / 92.),
        172..=265 => (1, 0, (day - 172) as f64 / 94.),
        266..=354 => (0, 2, (day - 266) as f64 / 89.),
        _ => (
            2,
            0,
            ((if day < 80 { day + 365 } else { day }) as f64 - 355.) / 90.,
        ),
    }
}
fn softplus(x: f64) -> f64 {
    if x > 40. {
        x
    } else if x < -40. {
        x.exp()
    } else {
        x.exp().ln_1p()
    }
}
fn log_density(
    height: f64,
    day: u32,
    c: &[f64; 49],
    d: &[[[f64; 49]; 2]; 4],
    high: bool,
    ion: usize,
) -> f64 {
    let (s0, s1, f) = season(day);
    let mut values = [0.; 4];
    for h in 0..4 {
        let value = |s: usize| -> f64 {
            (0..49)
                .map(|j| {
                    c[j] * d[h][if s == 2 { 1 } else { s }][j] * if s == 2 { MIRROR[j] } else { 1. }
                })
                .sum()
        };
        values[h] = value(s0) * (1. - f) + value(s1) * f;
    }
    if ion == 0 || ion == 3 {
        values[3] = values[3].min(values[2]);
    } else {
        values[3] = values[3].max(values[2]);
    }
    let heights = if high {
        [550., 900., 1500., 2250.]
    } else {
        [390., 550., 740., 960.]
    };
    let slopes: Vec<_> = (0..3)
        .map(|i| (values[i + 1] - values[i]) / (heights[i + 1] - heights[i]))
        .collect();
    if height >= heights[3] {
        return values[2] + slopes[2] * (height - heights[2]);
    }
    let mut value = values[0] + slopes[0] * (height - heights[0]);
    for i in 0..2 {
        value += (slopes[i + 1] - slopes[i])
            * 20.
            * (softplus((height - heights[i + 1]) / 20.)
                - softplus((heights[i] - heights[i + 1]) / 20.));
    }
    value
}
/// Relative number fractions in O+, H+, He+, N+ order. `invdip` is IRI's
/// mixed dip/invariant latitude, NOT geographic latitude. `mlt` is central
/// dipole magnetic local time, NOT solar local time. PF10.7 is clamped to
/// 65–260 exactly as in CALION; input values are never modified.
pub fn calion(
    height_km: f64,
    invdip_deg: f64,
    mlt_hours: f64,
    day: u32,
    pf107: f64,
) -> Result<[f64; 4]> {
    ensure!(
        [height_km, invdip_deg, mlt_hours, pf107]
            .iter()
            .all(|x| x.is_finite())
            && (350. ..=2000.).contains(&height_km)
            && (-90. ..=90.).contains(&invdip_deg)
            && (0. ..24.).contains(&mlt_hours)
            && (1..=366).contains(&day)
            && pf107 > 0.,
        "invalid CALION domain or magnetic coordinates"
    );
    let c = harmonics(invdip_deg, mlt_hours);
    let low = [&DOL, &DHL, &DHEL, &DNL];
    let high = [&DOH, &DHH, &DHEH, &DNH];
    let flux = pf107.clamp(65., 260.);
    let alpha = (flux - 85.) / 125.;
    let mut out = [0.; 4];
    for i in 0..4 {
        let a = log_density(height_km, day, &c, low[i], false, i);
        let b = log_density(height_km, day, &c, high[i], true, i);
        out[i] = 10f64.powf(a + (b - a) * alpha);
    }
    if flux < 87.5 {
        let o = [1.872, 1.640, 1.234];
        let h = [0.762, 0.836, 1.033];
        let correction = |v: [f64; 3]| {
            if flux <= 67.5 {
                v[0]
            } else {
                let i = ((flux - 57.5) / 10.).floor() as usize - 1;
                v[i] + (v[i + 1] - v[i]) * (flux - (67.5 + 10. * i as f64)) / 10.
            }
        };
        let oc = correction(o);
        let hc = correction(h);
        out[0] /= oc;
        out[3] /= oc;
        out[1] /= hc;
        out[2] /= hc;
    }
    let sum: f64 = out.iter().sum();
    ensure!(sum.is_finite() && sum > 0., "nonfinite CALION composition");
    for v in &mut out {
        *v /= sum;
    }
    Ok(out)
}
#[cfg(test)]
mod tests {
    use super::*;
    #[test]
    fn normalization_and_domains() {
        for h in [350., 550., 1000., 1500., 2000.] {
            for f in [65., 75., 85., 150., 260.] {
                let x = calion(h, 12.8, 16., 319, f).unwrap();
                assert!((x.iter().sum::<f64>() - 1.).abs() < 1e-12);
                assert!(x.iter().all(|a| a.is_finite() && *a >= 0.));
            }
        }
        assert!(calion(2100., 12.8, 16., 319, 150.).is_err());
    }
}
