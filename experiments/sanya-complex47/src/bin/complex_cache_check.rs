include!("../core.inc.rs");
include!("../pe.inc.rs");
include!("../dispersion.inc.rs");
include!("../cache_complex.inc.rs");
fn main() -> Result<()> {
    let a: Vec<_> = std::env::args().collect();
    let old = hdf5::File::open(&a[1])?;
    let cache = HotCache::load(&a[2])?;
    let oldv = old.dataset("hot_terms")?.read_raw::<f32>()?;
    let shape = old.dataset("hot_terms")?.shape();
    ensure!(
        shape == [cache.height.len(), cache.nu, 2],
        "cache dimensions differ"
    );
    let mut unchanged = vec![];
    for k in 0..2 {
        let mut max_abs = 0_f64;
        let mut scale = 0_f64;
        let (mut l1, mut norm) = (0., 0.);
        for j in 0..cache.height.len() * cache.nu {
            let x = oldv[2 * j + k] as f64;
            let y = cache.values[3 * j + k] as f64;
            max_abs = max_abs.max((x - y).abs());
            scale = scale.max(x.abs());
            l1 += (x - y).abs();
            norm += x.abs();
        }
        ensure!(
            max_abs / scale < 1e-5,
            "source/imag cache changed unexpectedly"
        );
        unchanged.push(json!({"field":k,"max_absolute_difference_over_peak":max_abs/scale,"relative_l1_difference":l1/norm}));
    }
    let pe = PeArchive::load(&a[3])?;
    let source: Vec<Vec<f64>> = fs::read_to_string(&a[4])?
        .lines()
        .map(|l| l.split_whitespace().map(|x| x.parse().unwrap()).collect())
        .collect();
    let g = DispersionGrid::standard();
    let transform = CausalTransform::new(g.n());
    let mut cases = vec![];
    for h in [150.23, 225.37, 300.41, 849.83] {
        let r = range_for_altitude(h);
        let j = source
            .partition_point(|v| v[0] < r)
            .saturating_sub(1)
            .min(source.len() - 2);
        let t = (r - source[j][0]) / (source[j + 1][0] - source[j][0]);
        let b = (source[j][3] * (1. - t) + source[j + 1][3] * t) * 1e-9;
        let c = Core::new(1., 1800., b, cache.alpha, 442e6);
        let (_, _, re) = signed_hot(&pe.local(h), &c, g, &transform);
        let mut error = 0_f64;
        let mut scale = 0_f64;
        for n in 1..7 {
            for d in [-0.3, -0.1, -0.03, -0.01, 0., 0.01, 0.03, 0.1, 0.3] {
                let u = n as f64 + d;
                let direct = interpolate(g, &re, u);
                let approx = cache.at(h).response(&c, u * E * b / ME).1.re;
                scale = scale.max(direct.abs());
                error = error.max((approx - direct).abs());
            }
        }
        ensure!(
            error / scale < 0.01,
            "height cache interpolation error exceeds 1 percent"
        );
        cases.push(json!({"height_km":h,"max_real_interpolation_error_over_peak":error/scale}));
    }
    fs::write(
        &a[5],
        serde_json::to_vec_pretty(
            &json!({"unchanged_source_and_imaginary":unchanged,"off_grid_real_interpolation":cases}),
        )?,
    )?;
    Ok(())
}
