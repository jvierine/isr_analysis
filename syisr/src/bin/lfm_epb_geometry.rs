//! WGS84 beam paths and full IGRF field-line labels for March 2026 EPB study.
use anyhow::{ensure, Result};
use hdf5::{File, Group};
use nalgebra::Vector3 as V;
use std::{fs, path::Path};
use syisr_ac::lfm_geometry as g;
fn mat(h: &Group, k: &str, v: &[f64], nr: usize, nc: usize) -> Result<()> {
    h.new_dataset::<f64>()
        .shape((nr, nc))
        .create(k)?
        .write_raw(v)?;
    Ok(())
}
fn vec(h: &Group, k: &str, v: &[f64]) -> Result<()> {
    h.new_dataset_builder().with_data(v).create(k)?;
    Ok(())
}
fn main() -> Result<()> {
    let a: Vec<_> = std::env::args().collect();
    ensure!(a.len() == 3, "MULTIBEAM_ROOT OUTPUT_H5");
    let root = Path::new(&a[1]);
    let out = File::create(&a[2])?;
    let (r, e1, e2, b) = g::reference()?;
    mat(&out, "reference_ecef_km", r.as_slice(), 1, 3)?;
    mat(
        &out,
        "basis_e1_e2_B_ecef",
        &[e1.as_slice(), e2.as_slice(), b.as_slice()].concat(),
        3,
        3,
    )?;
    let enu_b = g::enu(g::ecef(g::LAT, g::LON, g::SITE_KM) + b);
    vec(&out, "reference_B_ENU_unit", enu_b.as_slice())?;
    vec(&out, "site_lat_lon_height", &[g::LAT, g::LON, g::SITE_KM])?;
    vec(&out, "model_epoch", &[g::YEAR])?;
    let mut paths: Vec<_> = fs::read_dir(root.join("beams"))?
        .filter_map(|e| e.ok().map(|e| e.path()))
        .filter(|p| p.is_dir())
        .collect();
    paths.sort();
    for path in paths {
        let h = File::open(path.join("inversion.h5"))?;
        let az = h.attr("beam_azimuth_deg")?.read_scalar::<f64>()?;
        let el = h.attr("beam_elevation_deg")?.read_scalar::<f64>()?;
        let q = out.create_group(path.file_name().unwrap().to_str().unwrap())?;
        vec(&q, "az_el_deg", &[az, el])?;
        let ranges = h.dataset("range_km")?.read_raw::<f64>()?;
        let mut pos = Vec::new();
        let mut enu = Vec::new();
        let mut geo = Vec::new();
        let mut plane = Vec::new();
        let mut mapped = Vec::new();
        let mut quadrature: Vec<f64> = Vec::new();
        let mut edges: Vec<f64> = Vec::new();
        let mut aspects: Vec<f64> = Vec::new();
        for &range in &ranges {
            let p = g::position(range, az, el);
            pos.extend(p.as_slice());
            enu.extend(g::enu(p).as_slice());
            geo.extend(g::geodetic(p));
            plane.extend(g::project_plane(p)?);
            mapped.extend(g::map_field_line(p)?);
            aspects.push(
                g::ray(az, el)
                    .dot(&g::bunit(p)?)
                    .clamp(-1., 1.)
                    .asin()
                    .to_degrees(),
            );
            for (q, _) in g::beam_quadrature(range, az, el) {
                quadrature.extend(g::map_field_line(q)?);
            }
            let width = g::beam_widths(el);
            for j in 0..=16 {
                let phase = 2. * std::f64::consts::PI * j as f64 / 16.;
                let q = g::offset_position(
                    range,
                    az,
                    el,
                    (width[0] / 2.).to_radians() * phase.cos(),
                    (width[1] / 2.).to_radians() * phase.sin(),
                );
                edges.extend(q.as_slice());
            }
        }
        let nr = ranges.len();
        vec(&q, "range_km", &ranges)?;
        mat(&q, "ecef_km", &pos, nr, 3)?;
        mat(&q, "enu_km", &enu, nr, 3)?;
        mat(&q, "geodetic_lat_lon_alt", &geo, nr, 3)?;
        mat(&q, "straight_B_projection_km", &plane, nr, 3)?;
        mat(&q, "field_line_reference_plane_km", &mapped, nr, 3)?;
        q.new_dataset::<f64>()
            .shape((nr, 9, 3))
            .create("beam_quadrature_field_line_km")?
            .write_raw(&quadrature)?;
        q.new_dataset::<f64>()
            .shape((nr, 17, 3))
            .create("oneway_half_power_edges_ecef_km")?
            .write_raw(&edges)?;
        vec(&q, "aspect_deviation_from_perpendicular_deg", &aspects)?;
        vec(&q, "oneway_fwhm_elevation_azimuth_deg", &g::beam_widths(el))?;
        let weights: Vec<f64> = g::beam_quadrature(400., az, el)
            .iter()
            .map(|(_, w)| *w)
            .collect();
        vec(&q, "quadrature_weights", &weights)?;
        println!("GEOMETRY {} points={nr}", path.display());
    }
    // Hypothetical field-aligned tube, 30x70 km cross-section, 1000 km along B.
    // Chosen illustration; not a measured cavity or fitted dimension.
    let hyp = out.create_group("hypothetical")?;
    vec(&hyp, "cross_section_semi_axes_km", &[15., 35.])?;
    vec(&hyp, "along_B_extent_km", &[-500., 500.])?;
    let mut lines: Vec<f64> = Vec::new();
    let mut lengths: Vec<f64> = Vec::new();
    for j in 0..=16 {
        let phase = 2. * std::f64::consts::PI * j as f64 / 16.;
        let seed = r + e1 * (15. * phase.cos()) + e2 * (35. * phase.sin());
        let mut points: Vec<(f64, V<f64>)> = Vec::new();
        for sign in [-1., 1.] {
            let mut p = seed;
            for i in 0..=100 {
                let length = sign * i as f64 * 5.;
                if i > 0 || sign < 0. {
                    points.push((length, p));
                }
                let step = sign * 5.;
                let k1 = g::bunit(p)?;
                let k2 = g::bunit(p + k1 * step / 2.)?;
                let k3 = g::bunit(p + k2 * step / 2.)?;
                let k4 = g::bunit(p + k3 * step)?;
                p += (k1 + k2 * 2. + k3 * 2. + k4) * step / 6.;
            }
        }
        points.sort_by(|a, b| a.0.total_cmp(&b.0));
        for (length, p) in points {
            lines.extend(p.as_slice());
            lengths.push(length);
        }
    }
    hyp.new_dataset::<f64>()
        .shape((17, 201, 3))
        .create("field_aligned_surface_ecef_km")?
        .write_raw(&lines)?;
    out.flush()?;
    Ok(())
}
