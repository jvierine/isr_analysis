include!("../core.inc.rs");
include!("../cache_complex.inc.rs");
include!("../compute_complex.inc.rs");
fn main() -> Result<()> {
    let a: Vec<_> = std::env::args().collect();
    let source: Vec<Vec<f64>> = fs::read_to_string(&a[1])?
        .lines()
        .map(|l| l.split_whitespace().map(|x| x.parse().unwrap()).collect())
        .collect();
    let mut cache = HotCache::load(&a[2])?;
    let heights = [150., 225.5, 300., 500., 850.];
    let rows: Vec<Vec<f64>> = heights
        .iter()
        .map(|&h| {
            let r = range_for_altitude(h);
            let j = source
                .partition_point(|v| v[0] < r)
                .saturating_sub(1)
                .min(source.len() - 2);
            let t = (r - source[j][0]) / (source[j + 1][0] - source[j][0]);
            source[j]
                .iter()
                .zip(&source[j + 1])
                .map(|(a, b)| a * (1. - t) + b * t)
                .collect()
        })
        .collect();
    let mut cases = vec![];
    cache.real_multiplier = 0.;
    let off: Vec<_> = rows
        .par_iter()
        .map(|r| compute(r, &cache.at(beam_location(r[0]).0)))
        .collect();
    cache.real_multiplier = 1.;
    let on: Vec<_> = rows
        .par_iter()
        .map(|r| compute(r, &cache.at(beam_location(r[0]).0)))
        .collect();
    for (j, (x, y)) in off.iter().zip(&on).enumerate() {
        let sum0: f64 = x.tail.iter().sum();
        let sum1: f64 = y.tail.iter().sum();
        let l1: f64 = x
            .tail
            .iter()
            .zip(&y.tail)
            .map(|(a, b)| (a - b).abs())
            .sum::<f64>()
            / sum0;
        let modes:Vec<_>=y.enhanced_modes.iter().filter_map(|m|x.enhanced_modes.iter().min_by(|a,b|(a[1]-m[1]).abs().total_cmp(&(b[1]-m[1]).abs())).map(|n|json!({"old_frequency_hz":n[1],"new_frequency_hz":m[1],"shift_hz":m[1]-n[1],"old_hwhm_hz":n[2],"new_hwhm_hz":m[2],"area_ratio":m[3]/n[3],"new_stable":m[4]}))).collect();
        cases.push(json!({"height_km":heights[j],"density_m3":rows[j][1],"temperature_k":rows[j][2],"B_nT":rows[j][3],"integrated_power_ratio":sum1/sum0,"spectral_l1_relative_to_old":l1,"old_modes":x.enhanced_modes,"new_modes":y.enhanced_modes,"matched_modes":modes,"old_psd":x.tail,"new_psd":y.tail}));
    }
    fs::write(
        &a[3],
        serde_json::to_vec(
            &json!({"frequency_hz":(0..NF).map(|i|FMIN+(i as f64+0.5)*DF).collect::<Vec<_>>(),"cases":cases,"method":"identical source and imaginary response, same complex-cache interpolation and quadrature, real part multiplier 0 versus 1; modes relocated for each dielectric","root_scan_hz":std::env::var("ROOT_SCAN_HZ").unwrap_or("500".into()),"full_quadrature":std::env::var_os("FULL_QUADRATURE").is_some()}),
        )?,
    )?;
    Ok(())
}
