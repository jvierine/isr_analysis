//! Supplied voltage FIR, for use inside the range/lag ambiguity integral.
//! Timing origin is deliberately explicit: satellite-calibrated delays already
//! include receiver latency. Never silently add or subtract its centroid again.
use crate::Complex;
use anyhow::{ensure, Result};
use std::path::Path;

/// Load the preserved coefficients as unit-DC discrete convolution weights.
/// `interval_s` explicitly acknowledges the coefficient time grid (the supplied
/// plot suggests 1 us, not the 10 us output sampling period). `origin_s` is the
/// delay to remove from tap times, supplied in the same timing convention as
/// the range calibration. This does not resample or truncate the response.
pub fn load(path: &Path, interval_s: f64, origin_s: f64) -> Result<Vec<(f64, Complex)>> {
    ensure!(interval_s.is_finite() && interval_s > 0. && origin_s.is_finite(), "invalid FIR timing");
    let file = hdf5::File::open(path)?;
    let h = file.dataset("voltage_impulse_response")?.read_raw::<f64>()?;
    ensure!(!h.is_empty() && h.iter().all(|x| x.is_finite()), "invalid FIR coefficients");
    let dc: f64 = h.iter().sum();
    ensure!(dc.is_finite() && dc.abs() > 1e-12, "zero FIR DC gain");
    Ok(h.iter().enumerate().map(|(i, v)| (i as f64 * interval_s - origin_s, Complex::new(v / dc, 0.))).collect())
}

#[cfg(test)]
mod tests {
    use super::*;
    #[test]
    fn supplied_response_preserves_timing_and_gain() {
        let path = Path::new(concat!(env!("CARGO_MANIFEST_DIR"), "/calibration/receiver_100khz.h5"));
        let h = load(path, 1e-6, 131e-6).unwrap();
        assert_eq!(h.len(), 264);
        assert!((h.iter().map(|(_, v)| v.re).sum::<f64>() - 1.).abs() < 1e-12);
        assert!((h[0].0 + 131e-6).abs() < 1e-15);
        assert!(h[131].0.abs() < 1e-15);
        assert!(load(path, 0., 0.).is_err());
        // A uniform stationary signal has |H(f)|^2 voltage-covariance response.
        // Check the exact finite-range integral against that independent result.
        let f = 12000.;
        let transfer: Complex = h.iter().map(|(t,v)| *v * Complex::from_polar(1., std::f64::consts::TAU*f*t)).sum();
        let got = crate::exact_response::rectangular_cell(0.001, 0., 0.01, [-0.02,0.02], &h, &[f]).unwrap()[0];
        let expected: Complex = h.iter().flat_map(|(a,ha)| h.iter().map(move |(b,hb)| *ha * hb.conj() * (1. - (a-b).abs()/0.01) * Complex::from_polar(1.,-std::f64::consts::TAU*f*(a-b)))).sum();
        assert!((got-expected).norm() < 1e-11);
        assert!(transfer.norm_sqr().is_finite());
    }
}
