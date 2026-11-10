struct Row {
    thermal: Vec<f64>,
    tail: Vec<f64>,
    thermal_modes: Vec<Vec<f64>>,
    enhanced_modes: Vec<Vec<f64>>,
    nu: f64,
}
fn roots_of(mut fun: impl FnMut(f64) -> f64) -> Vec<f64> {
    let step = std::env::var("ROOT_SCAN_HZ")
        .ok()
        .map(|s| s.parse::<f64>().unwrap())
        .unwrap_or(500.);
    assert!(step > 0.);
    let mut roots = vec![];
    let mut l = FMIN;
    let mut last = fun(l);
    let stop = FMIN + NF as f64 * DF + 0.2e6;
    while l < stop {
        let r = l + step;
        let next = fun(r);
        if last < 0. && next >= 0. {
            let (mut lo, mut hi) = (l, r);
            for _ in 0..45 {
                let mid = (lo + hi) / 2.;
                if fun(mid) < 0. {
                    lo = mid;
                } else {
                    hi = mid;
                }
            }
            roots.push((lo + hi) / 2.);
        }
        l = r;
        last = next;
    }
    roots
}
fn compute(row: &[f64], tail: &HotSlice) -> Row {
    let alpha = tail.cache.alpha;
    let mut c = Core::new(row[1], row[2], row[3] * 1e-9, alpha, 442e6);
    c.nu += neutral_collision(row[0], row[2]).0;
    let total = |f: f64| {
        let (eps, m) = c.response(f);
        let (mh, chi) = tail.response(&c, 2. * PI * f);
        (eps + chi, m + mh)
    };
    let mut thermal_modes = vec![];
    let mut enhanced_modes = vec![];
    for f in roots_of(|f| c.response(f).0.re) {
        let (eps, m) = c.response(f);
        let d = (c.response(f + 10.).0.re - c.response(f - 10.).0.re) / (40. * PI);
        let g = eps.im / d / (2. * PI);
        if g > 0. && g / f < 0.02 {
            thermal_modes.push(vec![row[0], f, g, m / (eps.im * d).abs(), 1.]);
        }
    }
    for f in roots_of(|f| total(f).0.re) {
        let (eps, m) = total(f);
        let d = (total(f + 10.).0.re - total(f - 10.).0.re) / (40. * PI);
        let g = eps.im / d / (2. * PI);
        let stable = g > 0. && m > 0.;
        if g.abs() / f < 0.02 || !stable {
            enhanced_modes.push(vec![
                row[0],
                f,
                g,
                if stable {
                    m / (eps.im * d).abs()
                } else {
                    f64::NAN
                },
                if stable { 1. } else { 0. },
            ]);
        }
    }
    let all: Vec<_> = thermal_modes.iter().chain(&enhanced_modes).collect();
    let nodes = [
        -0.9602898564975363,
        -0.7966664774136267,
        -0.5255324099163290,
        -0.1834346424956498,
        0.1834346424956498,
        0.5255324099163290,
        0.7966664774136267,
        0.9602898564975363,
    ];
    let weights = [
        0.1012285362903763,
        0.2223810344533745,
        0.3137066458778873,
        0.3626837833783620,
        0.3626837833783620,
        0.3137066458778873,
        0.2223810344533745,
        0.1012285362903763,
    ];
    let mut thermal = vec![0.; NF];
    let mut enhanced = vec![0.; NF];
    for i in 0..NF {
        let lo = FMIN + i as f64 * DF;
        let hi = lo + DF;
        let mut breaks = vec![lo, hi];
        for m in &all {
            let g = m[2].abs();
            for factor in [-64., -16., -4., -1., 0., 1., 4., 16., 64.] {
                let f = m[1] + factor * g;
                if f > lo && f < hi {
                    breaks.push(f);
                }
            }
        }
        breaks.sort_by(f64::total_cmp);
        breaks.dedup();
        let (mut t, mut h) = (0., 0.);
        for b in breaks.windows(2) {
            let mid = (b[0] + b[1]) / 2.;
            let half = (b[1] - b[0]) / 2.;
            let near = all
                .iter()
                .any(|m| (mid - m[1]).abs() < (16. * m[2].abs()).max(2. * DF));
            let fnodes = [-0.5773502691896258, 0.5773502691896258];
            let fweights = [1., 1.];
            let (ns, ws): (&[f64], &[f64]) =
                if near || std::env::var_os("FULL_QUADRATURE").is_some() {
                    (&nodes, &weights)
                } else {
                    (&fnodes, &fweights)
                };
            for (&x, &w) in ns.iter().zip(ws) {
                let f = mid + half * x;
                let (eps, m) = c.response(f);
                let (mh, chi) = tail.response(&c, 2. * PI * f);
                t += half * w * 2. * m / eps.norm_sqr();
                h += half * w * 2. * (m + mh) / (eps + chi).norm_sqr();
            }
        }
        thermal[i] = t / DF;
        enhanced[i] = h / DF;
    }
    if enhanced_modes.iter().any(|m| m[4] == 0.) {
        enhanced.fill(f64::NAN);
    }
    Row {
        thermal,
        tail: enhanced,
        thermal_modes,
        enhanced_modes,
        nu: c.nu,
    }
}
