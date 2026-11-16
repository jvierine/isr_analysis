//! Zenith range/altitude geometry and IGRF-14 input tables. The field
//! implementation and coefficients come from the earlier Rust ISR branch,
//! commit 29d3ba8abe3207fa3a03675c560efa01631629c2.
use anyhow::{ensure, Result};
use chrono::{Datelike, TimeZone, Utc};
pub fn write(
    path: &std::path::Path,
    ranges: &[f64],
    utc: f64,
    lo: f64,
    hi: f64,
    step: f64,
) -> Result<()> {
    ensure!(step > 0. && hi >= lo, "invalid fitting grid");
    let date = chrono::DateTime::from_timestamp(utc as i64, 0)
        .ok_or_else(|| anyhow::anyhow!("invalid UTC epoch"))?;
    let begin = Utc
        .with_ymd_and_hms(date.year(), 1, 1, 0, 0, 0)
        .single()
        .unwrap();
    let next = Utc
        .with_ymd_and_hms(date.year() + 1, 1, 1, 0, 0, 0)
        .single()
        .unwrap();
    let year = date.year() as f64
        + (utc - begin.timestamp() as f64) / (next.timestamp() - begin.timestamp()) as f64;
    let mut indices = vec![];
    let mut selected = vec![];
    let mut field = vec![];
    let mut angle = vec![];
    for j in 0..=((hi - lo) / step).floor() as usize {
        let target = lo + j as f64 * step;
        let i = (0..ranges.len())
            .min_by(|&a, &b| {
                (ranges[a] - target)
                    .abs()
                    .total_cmp(&(ranges[b] - target).abs())
            })
            .unwrap();
        ensure!(
            target >= ranges[0] && target <= *ranges.last().unwrap(),
            "requested range outside recording"
        );
        let r = ranges[i];
        let b = crate::geomagnetic::igrf14(year, 18.3492, 109.6222, r + 0.05)?;
        let magnitude = b.iter().map(|v| v * v).sum::<f64>().sqrt();
        indices.push(i as i64);
        selected.push(r);
        field.push(magnitude);
        angle.push((b[2].abs() / magnitude).asin().to_degrees());
    }
    let h = hdf5::File::create(path)?;
    h.new_dataset_builder()
        .with_data(&indices)
        .create("range_index")?;
    for (name, v) in [
        ("slant_range_km", selected.clone()),
        ("altitude_km", selected.iter().map(|r| r + 0.05).collect()),
        ("field_nT", field),
        ("angle_from_perpendicular_deg", angle),
    ] {
        h.new_dataset_builder().with_data(&v).create(name)?;
    }
    h.new_attr::<f64>()
        .create("igrf_decimal_year")?
        .write_scalar(&year)?;
    h.new_attr_builder().with_data("Verified zenith only; altitude = slant range + 0.05 km; WGS84 IGRF-14 at Sanya 18.3492 N 109.6222 E; nominal pulse centroid; receiver group delay uncalibrated").create("geometry")?;
    Ok(())
}
#[cfg(test)]
mod tests {
    #[test]
    fn december_2025_field_matches_independent_reference() {
        let b = crate::geomagnetic::igrf14(
            2025. + (351. + 4. / 24.) / 365.,
            18.3492,
            109.6222,
            600.484293374 + 0.05,
        )
        .unwrap();
        let m = b.iter().map(|v| v * v).sum::<f64>().sqrt();
        assert!((m - 32392.33550509953).abs() < 0.05);
        assert!(((b[2].abs() / m).asin().to_degrees() - 25.4972580481).abs() < 1e-4);
    }
}
