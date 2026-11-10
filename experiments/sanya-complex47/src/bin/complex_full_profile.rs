include!("../core.inc.rs");
include!("../cache_complex.inc.rs");
include!("../compute_complex.inc.rs");
fn main() -> Result<()> {
    let a: Vec<_> = std::env::args().collect();
    ensure!(a.len() == 5, "full_profile IRI.txt cache.h5 output step_km");
    let source: Vec<Vec<f64>> = fs::read_to_string(&a[1])?
        .lines()
        .map(|l| l.split_whitespace().map(|x| x.parse().unwrap()).collect())
        .collect();
    let cache = HotCache::load(&a[2])?;
    let dr = a[4].parse::<f64>()?;
    let start = range_for_altitude(100.);
    let stop = source.last().unwrap()[0];
    let rows: Vec<Vec<f64>> = (0..=((stop - start) / dr).floor() as usize)
        .map(|i| {
            let r = start + i as f64 * dr;
            let j = source
                .partition_point(|row| row[0] < r)
                .saturating_sub(1)
                .min(source.len() - 2);
            let u = (r - source[j][0]) / (source[j + 1][0] - source[j][0]);
            source[j]
                .iter()
                .zip(&source[j + 1])
                .map(|(a, b)| a * (1. - u) + b * u)
                .collect()
        })
        .collect();
    let calculated: Vec<_> = rows
        .par_iter()
        .map(|r| compute(r, &cache.at(beam_location(r[0]).0)))
        .collect();
    fs::create_dir_all(&a[3])?;
    let h = hdf5::File::create(format!("{}/profile.h5", a[3]))?;
    for (name, which) in [
        ("thermal_psd_hz_inverse", false),
        ("suprathermal_psd_hz_inverse", true),
    ] {
        let values: Vec<f32> = calculated
            .iter()
            .flat_map(|r| if which { &r.tail } else { &r.thermal })
            .map(|&v| v as f32)
            .collect();
        h.new_dataset::<f32>()
            .shape([rows.len(), NF])
            .chunk([1, NF])
            .shuffle()
            .deflate(4)
            .create(name)?
            .write_raw(&values)?;
    }
    let ranges: Vec<_> = rows.iter().map(|r| r[0]).collect();
    let frequencies: Vec<_> = (0..NF).map(|i| FMIN + (i as f64 + 0.5) * DF).collect();
    h.new_dataset_builder()
        .with_data(&ranges)
        .create("slant_range_km")?;
    h.new_dataset_builder()
        .with_data(&frequencies)
        .create("frequency_hz")?;
    let modes: Vec<_> = calculated
        .iter()
        .flat_map(|r| r.enhanced_modes.clone())
        .collect();
    let metadata = json!({"range_step_km":dr,"frequency_bin_hz":DF,"profiles":rows,"enhanced_modes":modes,"thermal_modes":calculated.iter().flat_map(|r|r.thermal_modes.clone()).collect::<Vec<_>>(),"source":"IRI core plus full complex collisionless SAMI2-PE susceptibility, including causal real part","photoelectron_model":cache.source_meta,"hot_cache":a[2],"hot_real_multiplier":cache.real_multiplier,"angle_from_perpendicular_deg":cache.alpha,"collision_s":calculated.iter().map(|r|r.nu).collect::<Vec<_>>(),"mode_columns":["range_km","frequency_hz","hwhm_hz","area_per_core_electron","stable"],"method":"combined complex thermal and hot dielectric; separate thermal/enhanced mode roots and derivatives; mode-aware quadrature","limitations":["hot perturbation response collisionless; SAMI equilibrium includes its collisional transport","piecewise linear differential number-density reconstruction in energy and mu; outer mu endpoints held constant; gyro-aligned cache validated independently","first archived IRI rows extrapolated from104.2km to100km altitude","not a fitted photoelectron distribution or absolute radar calibration"]});
    let bytes = serde_json::to_vec(&metadata)?;
    h.new_dataset_builder()
        .with_data(&bytes)
        .create("metadata_json_utf8")?;
    fs::write(format!("{}/profile.json", a[3]), bytes)?;
    println!(
        "full SAMI profile: {} rows x {} frequencies",
        rows.len(),
        NF
    );
    Ok(())
}
