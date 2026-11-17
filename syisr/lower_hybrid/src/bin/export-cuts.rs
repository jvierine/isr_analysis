//! Export exact perpendicular and +900 us two-way-delay PSD cuts to HDF5.
use anyhow::{ensure, Result};
use hdf5::{types::VarLenUnicode, File};
use isr_range_doppler::RangeCut;
use std::str::FromStr;
fn text(g: &hdf5::Group, key: &str, value: &str) -> Result<()> {
    if g.attr_names()?.iter().any(|k| k == key) {
        g.attr(key)?
            .write_scalar(&VarLenUnicode::from_str(value)?)?;
    } else {
        g.new_attr::<VarLenUnicode>()
            .create(key)?
            .write_scalar(&VarLenUnicode::from_str(value)?)?;
    }
    Ok(())
}
fn main() -> Result<()> {
    let args: Vec<_> = std::env::args().collect();
    ensure!(args.len() == 2, "export-cuts SPECTRA.h5");
    let h = File::open_rw(&args[1])?;
    let r = h.dataset("slant_range_km")?.read_raw::<f64>()?;
    let nf = h.dataset("frequency_hz")?.size();
    let nt = h.dataset("window_start_utc_s")?.size();
    let geometry = h.dataset("beam_geometry")?.read_raw::<f64>()?;
    for b in 0..5 {
        if !h.link_exists(&format!("beam_{b}")) {
            continue;
        }
        let g = h.group(&format!("beam_{b}"))?;
        let raw = g
            .dataset("background_subtracted_psd_adu2_per_hz")?
            .read_raw::<f64>()?;
        ensure!(raw.len() == nt * r.len() * nf, "Unexpected spectrum shape");
        let targets = [
            geometry[b * 7 + 3],
            geometry[b * 7 + 3] + 900. * 0.149896229,
        ];
        let cuts = [
            RangeCut::new(&r, targets[0])?,
            RangeCut::new(&r, targets[1])?,
        ];
        let mut indices = Vec::new();
        let mut weights = Vec::new();
        for (cut, name) in cuts.iter().zip([
            "perpendicular_cut_psd_adu2_per_hz",
            "reference_cut_psd_adu2_per_hz",
        ]) {
            let mut values = Vec::with_capacity(nt * nf);
            for plane in raw.chunks(r.len() * nf) {
                values.extend(cut.extract(plane, nf)?);
            }
            if g.link_exists(name) {
                g.unlink(name)?;
            }
            g.new_dataset::<f64>()
                .shape((nt, nf))
                .deflate(3)
                .create(name)?
                .write_raw(&values)?;
            indices.extend(cut.indices.map(|i| i as u64));
            weights.extend(cut.weights);
        }
        for name in [
            "cut_slant_range_km",
            "cut_source_gate_indices",
            "cut_source_gate_weights",
        ] {
            if g.link_exists(name) {
                g.unlink(name)?;
            }
        }
        g.new_dataset_builder()
            .with_data(&targets)
            .create("cut_slant_range_km")?;
        g.new_dataset::<u64>()
            .shape((2, 2))
            .create("cut_source_gate_indices")?
            .write_raw(&indices)?;
        g.new_dataset::<f64>()
            .shape((2, 2))
            .create("cut_source_gate_weights")?
            .write_raw(&weights)?;
        text(&g,"cut_definition","First: nominal perpendicular-to-B slant range. Second: 900 us larger two-way radar delay, +134.9066061 km along the same beam; slant range is not altitude. Linear interpolation of neighbouring gate PSDs; no improvement in physical range resolution.")?;
        println!("beam {b}: {:.3} and {:.3} km", targets[0], targets[1]);
    }
    text(&h, "cut_export_program", "syisr-rd-export-cuts")?;
    text(&h, "cut_export_version", "0.1.0")?;
    text(&h,"cut_log_display","Logarithmic panel shows positive signed-excess estimates only, with gaps at nonpositive bins; no absolute value or positive floor. Linear panel retains all signs.")?;
    Ok(())
}
