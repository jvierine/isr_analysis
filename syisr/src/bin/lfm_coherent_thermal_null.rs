//! Exploratory finite-beam magnetized thermal power reference, not spectral validation.
//! Reuses existing 430.2MHz BGK pilot; actual LFM frequency432MHz is recorded separately.
use anyhow::Result;
use std::{f64::consts::PI, path::Path};
use syisr_ac::{geomagnetic, lfm_geometry as geo, theory::Theory};
fn main() -> Result<()> {
    let args: Vec<_> = std::env::args().collect();
    let out = Path::new(&args[1]);
    // Measured beam9 persistent maximum: mean product60s x~6km at12:48:30UTC.
    let height = 323.6296100373802;
    let aspect = 0.713359464;
    let range = 370.6746372883749;
    let p = geo::position(range, 358., 60.);
    let llh = geo::geodetic(p);
    let field = geomagnetic::igrf14(geo::YEAR, llh[0], llh[1], height)?;
    let b = field.iter().map(|x| x * x).sum::<f64>().sqrt() * 1e-9;
    let mut f: Vec<f64> = (-500..=500)
        .map(|i| i as f64 * 100.)
        .filter(|x| x.abs() > 1000.)
        .collect();
    f.extend((-1000..=1000).map(|i| i as f64));
    f.sort_by(f64::total_cmp);
    let mut rows = Vec::new();
    let mut spectra = Vec::new();
    for nu in [10., 100., 1000.] {
        for width in [0.5, 0.765, 1.] {
            for a in [aspect, 25.4] {
                let mut t = Theory::near_perpendicular_from_beam(
                    range,
                    358.,
                    60.,
                    geo::YEAR,
                    f.clone(),
                    nu,
                    width,
                )?;
                t.b = b;
                t.alpha = a;
                let (s, mass) = t.spectrum_with_mass(1000., 1100., 0., 0., 0., 2.5e12);
                rows.extend([a, width, nu, mass]);
                spectra.extend(s);
                println!("THERMAL aspect={a} width={width} nu={nu} integrated_per_ne={mass}");
            }
        }
    }
    let h = hdf5::File::create(out)?;
    h.new_dataset::<f64>()
        .shape((rows.len() / 4, 4))
        .create("cases_aspect_width_nu_integral")?
        .write_raw(&rows)?;
    h.new_dataset::<f64>()
        .shape((rows.len() / 4, f.len()))
        .create("normalized_spectrum_per_Hz")?
        .write_raw(&spectra)?;
    h.new_dataset_builder()
        .with_data(&f)
        .create("frequency_Hz")?;
    h.new_dataset_builder()
        .with_data(&[
            b,
            height,
            2.5e12,
            1000.,
            1100.,
            430.2e6,
            432e6,
            4. * PI * 430.2e6 / 299792458.,
        ])
        .create("B_height_Ne_Ti_Te_reference_RF_actual_RF_k")?;
    Ok(())
}
