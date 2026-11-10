include!("../core.inc.rs");
include!("../pe.inc.rs");
include!("../dispersion.inc.rs");
fn main() -> Result<()> {
    let a: Vec<_> = std::env::args().collect();
    let pe = PeArchive::load(&a[1])?;
    let source: Vec<Vec<f64>> = fs::read_to_string(&a[2])?
        .lines()
        .map(|l| l.split_whitespace().map(|x| x.parse().unwrap()).collect())
        .collect();
    let g = DispersionGrid::standard();
    let transform = CausalTransform::new(g.n());
    let alpha: f64 = a.get(4).map(|v| v.parse()).transpose()?.unwrap_or(1.);
    let dh: f64 = a.get(5).map(|v| v.parse()).transpose()?.unwrap_or(0.5);
    let heights: Vec<_> = (0..=((900. - 100.) / dh).round() as usize)
        .map(|i| 100. + i as f64 * dh)
        .collect();
    let u0 = 0.05;
    let du = 0.00025;
    let nu = 31801;
    let rows: Vec<Vec<f32>> = heights
        .par_iter()
        .enumerate()
        .map(|(index, &h)| {
            let r = range_for_altitude(h);
            let j = source
                .partition_point(|v| v[0] < r)
                .saturating_sub(1)
                .min(source.len() - 2);
            let t = (r - source[j][0]) / (source[j + 1][0] - source[j][0]);
            let b = (source[j][3] * (1. - t) + source[j + 1][3] * t) * 1e-9;
            let c = Core::new(1., 1800., b, alpha, 442e6);
            let p = pe.local(h);
            let (m, im, re) = signed_hot(&p, &c, g, &transform);
            if index % 100 == 0 {
                println!("complex cache height {index}/{}", heights.len());
            }
            (0..nu)
                .flat_map(|i| {
                    let u = u0 + i as f64 * du;
                    [
                        interpolate(g, &m, u) as f32,
                        interpolate(g, &im, u) as f32,
                        interpolate(g, &re, u) as f32,
                    ]
                })
                .collect()
        })
        .collect();
    let out = hdf5::File::create(&a[3])?;
    out.new_dataset::<f32>()
        .shape([heights.len(), nu, 3])
        .chunk([1, nu, 3])
        .shuffle()
        .deflate(4)
        .create("hot_terms")?
        .write_raw(&rows.iter().flatten().copied().collect::<Vec<_>>())?;
    out.new_dataset_builder()
        .with_data(&heights)
        .create("height_km")?;
    let meta = json!({"u0":u0,"du":du,"nu":nu,"height_step_km":dh,"aspect_from_perpendicular_deg":alpha,"source":a[1],"photoelectron_model":pe.metadata,"fields":["free_half_source_per_volume","imaginary_hot_susceptibility","real_hot_susceptibility"],"real_method":"causal signed-frequency dispersion relation; nonperiodic linear FFT convolution of odd-index discrete PV kernel; no assumed odd symmetry","signed_extent_gyro":g.extent,"signed_base_du":g.du,"dense_edge":g.dense_edge,"middle_edge":g.middle_edge,"middle_stride":g.middle_stride,"outer_stride":g.outer_stride,"physical_normalization":"source divide by actual nc; susceptibility already absolute PE density normalized","limitations":"collisionless hot perturbation; finite signed-frequency extent and quadrature checked by refinement"});
    out.new_dataset_builder()
        .with_data(&serde_json::to_vec(&meta)?)
        .create("metadata_json_utf8")?;
    println!("complex cache complete");
    Ok(())
}
