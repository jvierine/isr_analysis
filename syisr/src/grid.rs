//! Physical, beam-dependent AC gate geometry. One20us baud is2.99792458km
//! of slant range; it is not3km of altitude on an oblique beam. Bottom-side
//! gates contain one native baud cell. Above400km, consecutive native cells
//! are grouped into progressively wider non-overlapping altitude windows.
use crate::campaign::{altitude, range};
use serde::{Deserialize, Serialize};
pub const NATIVE_RANGE_KM: f64 = 2.99792458;
#[derive(Clone, Debug, Serialize, Deserialize)]
pub struct GateBin {
    pub altitude_km: f64,
    pub slant_range_km: f64,
    pub lower_altitude_km: f64,
    pub upper_altitude_km: f64,
    pub native_cells: Vec<usize>,
    pub weights: Vec<f64>,
}
/// An explicitly assumed scale-height-inspired averaging width, not a measured
/// atmospheric scale height. It is continuous at400km and150km at2000km.
pub fn target_width(z: f64, elevation: f64) -> f64 {
    let r = range(400., elevation);
    let native = altitude(r + NATIVE_RANGE_KM / 2., elevation)
        - altitude(r - NATIVE_RANGE_KM / 2., elevation);
    if z <= 400. {
        return native;
    }
    let t = (z - 400.) / 1600.;
    let q2 = 150. / native * (6771f64 / 8371.).powi(2);
    native * (1. + (q2 - 1.) * t) * ((6371. + z) / 6771.).powi(2)
}
pub fn build(elevation: f64, top: f64) -> Vec<GateBin> {
    let ranges: Vec<_> = (0..635)
        .map(|i| 667e-6 * 149896.229 + i as f64 * NATIVE_RANGE_KM)
        .collect();
    let z: Vec<_> = ranges.iter().map(|r| altitude(*r, elevation)).collect();
    let edges: Vec<_> = (0..=635)
        .map(|i| {
            altitude(
                667e-6 * 149896.229 + (i as f64 - 0.5) * NATIVE_RANGE_KM,
                elevation,
            )
        })
        .collect();
    let first = z.partition_point(|h| *h < 100.);
    let end = z.partition_point(|h| *h <= top);
    let mut i = first;
    let mut out = vec![];
    while i < end {
        let mut count = 1usize;
        if z[i] > 400. {
            for _ in 0..5 {
                let center = (z[i] + z[(i + count - 1).min(end - 1)]) / 2.;
                let native = edges[(i + count).min(end)] - edges[i];
                let per = native / count as f64;
                count = ((target_width(center, elevation) / per).round() as usize)
                    .max(1)
                    .min(end - i);
            }
        }
        // Do not merge bottom-side native cells across400km.
        let stop = (i + count).min(end);
        let cells: Vec<_> = (i..stop).collect();
        let width = edges[stop] - edges[i];
        let weights: Vec<_> = cells
            .iter()
            .map(|j| (edges[j + 1] - edges[*j]) / width)
            .collect();
        let height = cells.iter().zip(&weights).map(|(j, w)| z[*j] * w).sum();
        let range = cells
            .iter()
            .zip(&weights)
            .map(|(j, w)| ranges[*j] * w)
            .sum();
        out.push(GateBin {
            altitude_km: height,
            slant_range_km: range,
            lower_altitude_km: edges[i],
            upper_altitude_km: edges[stop],
            native_cells: cells,
            weights,
        });
        i = stop;
    }
    out
}
#[cfg(test)]
mod tests {
    use super::*;
    #[test]
    fn native_bottom_side_every_beam() {
        for el in [30., 45., 60., 75., 90.] {
            let gates = build(el, 1000.);
            assert!(gates.len() > 50);
            for g in gates.iter().filter(|g| g.altitude_km <= 400.) {
                assert_eq!(g.native_cells.len(), 1);
                assert!((g.weights[0] - 1.).abs() < 1e-12);
            }
            for pair in gates.windows(2) {
                assert!((pair[0].upper_altitude_km - pair[1].lower_altitude_km).abs() < 1e-10);
            }
            for g in gates {
                assert!((g.weights.iter().sum::<f64>() - 1.).abs() < 1e-12);
            }
        }
    }
    #[test]
    fn width_endpoint() {
        for el in [30., 60., 90.] {
            assert!((target_width(2000., el) - 150.).abs() < 1e-9);
            assert!((target_width(400.000001, el) - target_width(400., el)).abs() < 1e-5);
        }
    }
    #[test]
    fn zenith_baud_is_three_km() {
        let g = build(90., 1000.);
        assert!((g[5].upper_altitude_km - g[5].lower_altitude_km - NATIVE_RANGE_KM).abs() < 1e-8);
        assert!(g.last().unwrap().upper_altitude_km - g.last().unwrap().lower_altitude_km > 10.);
    }
}

/// Explicit campaign extent; retained in every decoded product through gate edges.
pub fn maximum_altitude_km() -> f64 {
    let z = std::env::var("SYISR_MAX_ALTITUDE_KM").ok().map(|s|s.parse::<f64>().expect("numeric maximum altitude")).unwrap_or(1000.);
    assert!((1000.0..=1300.0).contains(&z), "supported fit extent is 1000 to 1300 km");
    z
}

#[cfg(test)]
mod extent_tests {
    #[test]
    fn extended_grid_keeps_native_bottom_and_reaches_1300() {
        for el in [45.,60.,75.,90.] {
            let old=super::build(el,1000.);let new=super::build(el,1300.);
            assert!(new.last().unwrap().upper_altitude_km>1295.);
            for (a,b) in old.iter().zip(&new).take_while(|(a,_)|a.altitude_km<900.) {
                assert_eq!(a.native_cells,b.native_cells);assert_eq!(a.weights,b.weights);
            }
            assert!(new.iter().flat_map(|g|&g.native_cells).all(|i|*i<635));
        }
    }
}
