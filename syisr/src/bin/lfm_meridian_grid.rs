//! Field-aligned curvilinear display grid; finite parallel extent is prescribed.
use anyhow::Result;
use syisr_ac::lfm_geometry as g;
fn main() -> Result<()> {
    let a: Vec<_> = std::env::args().collect();
    let out = hdf5::File::create(&a[1])?;
    let x: Vec<_> = (-80..=80).map(|i| i as f64 * 10.).collect();
    let y: Vec<_> = (-25..=35).map(|i| i as f64 * 10.).collect();
    let s: Vec<_> = (-30..=30).map(|i| i as f64 * 10.).collect();
    let (r, e1, e2, _) = g::reference()?;
    let mut xyz = vec![0.; x.len() * y.len() * s.len() * 3];
    let mut height = vec![0.; x.len() * y.len() * s.len()];
    for (j, &v) in y.iter().enumerate() {
        for (i, &u) in x.iter().enumerate() {
            let seed = r + e1 * u + e2 * v;
            for sign in [-1i32, 1i32] {
                let mut p = seed;
                for k in 0..=30i32 {
                    let si = (30 + sign * k) as usize;
                    let index = (si * y.len() + j) * x.len() + i;
                    xyz[index * 3..index * 3 + 3].copy_from_slice(g::enu(p).as_slice());
                    height[index] = g::geodetic(p)[2];
                    let step = sign as f64 * 10.;
                    let k1 = g::bunit(p)?;
                    let k2 = g::bunit(p + k1 * step / 2.)?;
                    let k3 = g::bunit(p + k2 * step / 2.)?;
                    let k4 = g::bunit(p + k3 * step)?;
                    p += (k1 + 2. * k2 + 2. * k3 + k4) * step / 6.;
                }
            }
        }
    }
    for (name, data) in [
        ("x_km", x.clone()),
        ("y_km", y.clone()),
        ("along_B_km", s.clone()),
    ] {
        out.new_dataset_builder().with_data(&data).create(name)?;
    }
    out.new_dataset::<f64>()
        .shape((s.len(), y.len(), x.len(), 3))
        .create("enu_km")?
        .write_raw(&xyz)?;
    out.new_dataset::<f64>()
        .shape((s.len(), y.len(), x.len()))
        .create("geodetic_altitude_km")?
        .write_raw(&height)?;
    println!(
        "Curvilinear grid {} field lines; prescribed +/-300 km along B",
        x.len() * y.len()
    );
    Ok(())
}
