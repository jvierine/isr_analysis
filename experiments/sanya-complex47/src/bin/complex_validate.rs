include!("../core.inc.rs");
include!("../pe.inc.rs");
include!("../dispersion.inc.rs");
fn main() -> Result<()> {
    let a: Vec<_> = std::env::args().collect();
    let grids = [DispersionGrid::standard(), DispersionGrid::refined()];
    let mut validation = vec![];
    for g in grids {
        let t = CausalTransform::new(g.n());
        for drift in [0., 0.37] {
            let func = |u: f64| {
                let z = C::new(u - drift, 0.035);
                C::new(0.7, 0.) / (C::new(4.2_f64.powi(2), 0.) - z * z)
            };
            let im: Vec<_> = (0..g.n()).map(|i| func(g.coordinate(i)).im).collect();
            let real = t.real_from_imag(&im);
            let mut err = 0_f64;
            let mut norm = 0_f64;
            for i in 0..=1600 {
                let u = i as f64 / 200.;
                let target = func(u).re;
                err = err.max((interpolate(g, &real, u) - target).abs());
                norm = norm.max(target.abs());
            }
            let rel = err / norm;
            ensure!(rel < 1e-5, "causal transform rational test failed");
            validation.push(json!({"test":"causal shifted oscillator","drift":drift,"du":g.du,"extent":g.extent,"max_error_over_peak":rel}));
        }
        let mut c = Core::new(1e7, 2. * E / KB, 40e-6, 1., 442e6);
        c.nu = 0.;
        let gyro = E * c.b / ME;
        let im: Vec<_> = (0..g.n())
            .into_par_iter()
            .map(|i| c.response(g.coordinate(i) * gyro / (2. * PI)).0.im)
            .collect();
        let real = t.real_from_imag(&im);
        let mut err = 0_f64;
        let mut norm = 0_f64;
        for n in 0..=8 {
            for det in [-0.08, -0.03, -0.01, 0.01, 0.03, 0.08] {
                let u = n as f64 + det;
                let target = c.response(u * gyro / (2. * PI)).0.re - 1.;
                err = err.max((interpolate(g, &real, u) - target).abs());
                norm = norm.max(target.abs());
            }
        }
        ensure!(
            err / norm < 0.001,
            "magnetized Maxwellian causal test failed"
        );
        validation
            .push(json!({"test":"magnetized Maxwellian","du":g.du,"max_error_over_peak":err/norm}));
    }
    if a.len() > 2 {
        let pe = PeArchive::load(&a[1])?;
        for h in [150., 225.5, 300., 850.] {
            let p = pe.local(h);
            let c = Core::new(1., 1800., 40e-6, 1., 442e6);
            let mut tables = vec![];
            for g in grids {
                let t = CausalTransform::new(g.n());
                tables.push(signed_hot(&p, &c, g, &t));
            }
            let mut max_abs = 0_f64;
            let mut peak = 0_f64;
            let mut cases = vec![];
            for n in 1..=6 {
                for det in [-0.3, -0.1, -0.05, -0.01, 0., 0.01, 0.05, 0.1, 0.3] {
                    let u = n as f64 + det;
                    let r0 = interpolate(grids[0], &tables[0].2, u);
                    let r1 = interpolate(grids[1], &tables[1].2, u);
                    peak = peak.max(r1.abs());
                    max_abs = max_abs.max((r0 - r1).abs());
                    cases.push(json!({"u":u,"real_standard":r0,"real_refined":r1}));
                }
            }
            let rel = max_abs / peak;
            ensure!(rel < 0.003, "SAMI real susceptibility needs finer grid");
            validation.push(json!({"test":"SAMI signed PV grid convergence","height_km":h,"max_error_over_peak":rel,"cases":cases}));
        }
    }
    let output = a
        .last()
        .cloned()
        .unwrap_or("complex-validation.json".into());
    fs::write(output, serde_json::to_vec_pretty(&validation)?)?;
    println!("causal real susceptibility checks passed");
    Ok(())
}
