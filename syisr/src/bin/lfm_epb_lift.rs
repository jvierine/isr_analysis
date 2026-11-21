//! Lift the fitted perpendicular contrast along dated IGRF field lines.
//! Parallel extent is prescribed: this is the field-aligned model, not a recovered
//! field-parallel length. Unsupported/ineligible cross-field nodes are omitted.
use anyhow::{ensure, Result};
use hdf5::File;
use syisr_ac::lfm_geometry as g;
fn main() -> Result<()> {
    let a: Vec<_> = std::env::args().collect();
    ensure!(a.len() == 3, "FIT_H5 OUTPUT_H5");
    let h = File::open(&a[1])?;
    let x = h.dataset("x_km")?.read_raw::<f64>()?;
    let y = h.dataset("y_km")?.read_raw::<f64>()?;
    let d = h.dataset("contrast")?.read_raw::<f64>()?;
    let support = h.dataset("support_weight")?.read_raw::<f64>()?;
    let maxsupport = support.iter().copied().fold(0., f64::max);
    let (r, e1, e2, _) = g::reference()?;
    let mut pos: Vec<f64> = Vec::new();
    let mut enu: Vec<f64> = Vec::new();
    let mut relative = Vec::new();
    let mut labels = Vec::new();
    let mut lengths = Vec::new();
    for (j, &v) in y.iter().enumerate() {
        for (i, &u) in x.iter().enumerate() {
            let k = j * x.len() + i;
            if support[k] < 0.005 * maxsupport || d[k] > -0.3 {
                continue;
            }
            let seed = r + e1 * u + e2 * v;
            for sign in [-1., 1.] {
                let mut p = seed;
                for s in 0..=80 {
                    let z = g::geodetic(p)[2];
                    if z < 150. || z > 1400. {
                        break;
                    }
                    if s > 0 || sign < 0. {
                        pos.extend(p.as_slice());
                        enu.extend(g::enu(p).as_slice());
                        relative.push(1. + d[k]);
                        labels.extend([u, v]);
                        lengths.push(sign * s as f64 * 5.);
                    }
                    let step = sign * 5.;
                    let k1 = g::bunit(p)?;
                    let k2 = g::bunit(p + k1 * step / 2.)?;
                    let k3 = g::bunit(p + k2 * step / 2.)?;
                    let k4 = g::bunit(p + k3 * step)?;
                    p += (k1 + k2 * 2. + k3 * 2. + k4) * step / 6.;
                }
            }
        }
    }
    let n = relative.len();
    let out = File::create(&a[2])?;
    out.new_dataset::<f64>()
        .shape((n, 3))
        .create("ecef_km")?
        .write_raw(&pos)?;
    out.new_dataset::<f64>()
        .shape((n, 3))
        .create("enu_km")?
        .write_raw(&enu)?;
    out.new_dataset::<f64>()
        .shape((n, 2))
        .create("field_line_labels_km")?
        .write_raw(&labels)?;
    out.new_dataset_builder()
        .with_data(&relative)
        .create("relative_density_proxy")?;
    out.new_dataset_builder()
        .with_data(&lengths)
        .create("distance_along_B_km")?;
    out.new_dataset_builder()
        .with_data(&[-400., 400.])
        .create("prescribed_parallel_extent_km")?;
    out.new_dataset_builder()
        .with_data(&h.dataset("parameters")?.read_raw::<f64>()?)
        .create("fit_parameters")?;
    println!("LIFTED {n} points; ratio<=0.7 and measured node support; prescribed +/-400 km parallel extent");
    Ok(())
}
