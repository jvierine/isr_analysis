//! Continuous cubic B-spline fields with a proper Matérn-5/2 coefficient prior.
//! The physical ISR response is evaluated at the original quadrature cells.
use super::*;
#[derive(Clone, Serialize, Deserialize)]
#[serde(default, deny_unknown_fields)]
pub struct SmoothSettings {
    pub temporal_neighbours: Vec<TemporalNeighbour>,
    pub knots_per_scale_height: f64,
    /// Standard deviations of ln Ne, ln Ti, ln Te and velocity/(1000 m/s).
    pub field_standard_deviation: [f64; 4],
    /// Optional data-informed prior centre on the physical quadrature grid.
    /// It must be supplied with provenance by the caller; never inferred from display pixels.
    pub centre: Option<Vec<[f64; 4]>>,
}
impl Default for SmoothSettings {
    fn default() -> Self {
        Self {
            temporal_neighbours: vec![],
            knots_per_scale_height: 2.,
            field_standard_deviation: [2., 0.7, 0.7, 0.3],
            centre: None,
        }
    }
}
#[derive(Clone, Serialize, Deserialize)]
pub struct TemporalNeighbour {
    pub centre: Vec<[f64; 4]>,
    pub standard_deviation: [f64; 4],
    pub source: String,
}
#[derive(Clone, Serialize, Deserialize)]
pub struct Representation {
    pub temporal_sources: Vec<String>,
    pub active_bound_coefficients: Vec<usize>,
    pub kind: String,
    pub knots: Vec<f64>,
    pub warped_altitude: Vec<f64>,
    pub coefficients: Vec<[f64; 4]>,
    pub coefficient_covariance: Vec<Vec<f64>>,
    pub prior_standard_deviation: [f64; 4],
    pub prior_centre: Vec<[f64; 4]>,
    pub normalized_gradient_max: f64,
}
fn basis(t: &[f64], u: f64) -> Vec<f64> {
    let n = t.len() - 4;
    if u >= t[t.len() - 1] {
        let mut v = vec![0.; n];
        v[n - 1] = 1.;
        return v;
    }
    let mut b: Vec<_> = t
        .windows(2)
        .map(|w| if u >= w[0] && u < w[1] { 1. } else { 0. })
        .collect();
    for p in 1..=3 {
        b = (0..b.len() - 1)
            .map(|i| {
                let a = t[i + p] - t[i];
                let c = t[i + p + 1] - t[i + 1];
                (if a > 0. { (u - t[i]) / a * b[i] } else { 0. })
                    + if c > 0. {
                        (t[i + p + 1] - u) / c * b[i + 1]
                    } else {
                        0.
                    }
            })
            .collect();
    }
    b
}
fn matern(d: f64) -> f64 {
    let a = 5_f64.sqrt() * d.abs();
    (1. + a + a * a / 3.) * (-a).exp()
}
struct Basis {
    t: Vec<f64>,
    u: Vec<f64>,
    transform: M<f64>,
    q: M<f64>,
    centre: V<f64>,
    initial: V<f64>,
    knots: usize,
}
fn make_basis(input: &Input, s: &SmoothSettings) -> Result<Basis> {
    ensure!(
        input.gates.len() >= 4
            && input
                .gates
                .windows(2)
                .all(|g| g[1].altitude_km > g[0].altitude_km),
        "smooth profiles require >=4 increasing quadrature altitudes"
    );
    ensure!(
        s.knots_per_scale_height.is_finite()
            && s.knots_per_scale_height >= 1.
            && s.knots_per_scale_height <= 8.
            && s.field_standard_deviation
                .iter()
                .all(|v| v.is_finite() && *v > 0.),
        "invalid spline prior"
    );
    let ng = input.gates.len();
    let h = input
        .settings
        .correlation_length_km
        .as_ref()
        .ok_or_else(|| anyhow::anyhow!("MSIS scale heights required for smooth profile"))?;
    ensure!(
        h.len() == ng && h.iter().flatten().all(|x| x.is_finite() && *x > 0.),
        "invalid MSIS scale heights"
    );
    let mut u = vec![0.];
    for i in 1..ng {
        u.push(
            u[i - 1]
                + (input.gates[i].altitude_km - input.gates[i - 1].altitude_km)
                    * 0.5
                    * (1. / h[i - 1][0] + 1. / h[i][0]),
        );
    }
    let end = u[ng - 1];
    let intervals = (end * s.knots_per_scale_height).ceil().max(1.) as usize;
    let mut t = vec![0.; 4];
    for i in 1..intervals {
        t.push(end * i as f64 / intervals as f64);
    }
    t.extend([end; 4]);
    let n = t.len() - 4;
    let rows: Vec<_> = u.iter().map(|&z| basis(&t, z)).collect();
    let b = M::from_fn(ng, n, |i, j| rows[i][j]);
    let transform = M::from_fn(ng * 4, n * 4, |i, j| {
        if i % 4 == j % 4 {
            b[(i / 4, j / 4)]
        } else {
            0.
        }
    });
    let g: Vec<_> = (0..n)
        .map(|i| (t[i + 1] + t[i + 2] + t[i + 3]) / 3.)
        .collect();
    let k = M::from_fn(n, n, |i, j| {
        matern(g[i] - g[j]) + if i == j { 1e-9 } else { 0. }
    });
    let ki = k
        .cholesky()
        .ok_or_else(|| anyhow::anyhow!("Matérn prior not positive definite"))?
        .inverse();
    let mut q = M::from_fn(4 * n, 4 * n, |i, j| {
        if i % 4 == j % 4 {
            ki[(i / 4, j / 4)] / s.field_standard_deviation[i % 4].powi(2)
        } else {
            0.
        }
    });
    let initial: Vec<[f64; 4]> = input
        .gates
        .iter()
        .map(|g| {
            [
                (g.initial.electron_density_m3 / 1e11).ln(),
                (g.initial.ion_temperature_k / 1000.).ln(),
                (g.initial.electron_temperature_k / 1000.).ln(),
                g.initial.velocity_m_s / 1000.,
            ]
        })
        .collect();
    let default_centre = vec![[0., 0., 2_f64.ln(), 0.]; ng];
    let values = s.centre.as_ref().unwrap_or(&default_centre);
    ensure!(
        values.len() == ng && values.iter().flatten().all(|v| v.is_finite()),
        "invalid spline centre"
    );
    // The centre is projected onto the same continuous basis. Small numerical ridge
    // is not a data regularizer; the proper Matérn covariance supplies that.
    let gram = b.transpose() * &b + M::identity(n, n) * 1e-10;
    let chol = gram
        .cholesky()
        .ok_or_else(|| anyhow::anyhow!("spline projection failed"))?;
    let mut centre = V::zeros(n * 4);
    for f in 0..4 {
        let y = V::from_iterator(ng, values.iter().map(|v| v[f]));
        let c = chol.solve(&(b.transpose() * y));
        for j in 0..n {
            centre[4 * j + f] = c[j];
        }
    }
    let mut start = V::zeros(n * 4);
    for f in 0..4 {
        let y = V::from_iterator(ng, initial.iter().map(|v| v[f]));
        let c = chol.solve(&(b.transpose() * y));
        for j in 0..n {
            start[4 * j + f] = c[j];
        }
    }
    // Conditional update of an explicit temporal quadratic penalty. Quadrature
    // weights in warped altitude make the penalty independent of grid subdivision.
    let mut rhs = &q * &centre;
    for neighbour in &s.temporal_neighbours {
        ensure!(
            neighbour.centre.len() == ng
                && neighbour.centre.iter().flatten().all(|x| x.is_finite())
                && neighbour
                    .standard_deviation
                    .iter()
                    .all(|v| v.is_finite() && *v > 0.)
                && !neighbour.source.is_empty(),
            "invalid temporal neighbour"
        );
        for i in 0..ng {
            let du = if i == 0 {
                (u[1] - u[0]) / 2.
            } else if i + 1 == ng {
                (u[i] - u[i - 1]) / 2.
            } else {
                (u[i + 1] - u[i - 1]) / 2.
            };
            for f in 0..4 {
                let w = du / neighbour.standard_deviation[f].powi(2);
                for a in 0..n {
                    rhs[4 * a + f] += w * b[(i, a)] * neighbour.centre[i][f];
                    for z in 0..n {
                        q[(4 * a + f, 4 * z + f)] += w * b[(i, a)] * b[(i, z)];
                    }
                }
            }
        }
    }
    if !s.temporal_neighbours.is_empty() {
        centre = q
            .clone()
            .cholesky()
            .ok_or_else(|| anyhow::anyhow!("temporal precision failed"))?
            .solve(&rhs);
    }
    Ok(Basis {
        t,
        u,
        transform,
        q,
        centre,
        initial: start,
        knots: n,
    })
}
struct Optimized {
    x: V<f64>,
    cov: Option<M<f64>>,
    history: Vec<f64>,
    gradient: f64,
    normalized: f64,
    bound_coefficients: Vec<usize>,
    status: String,
    chi: f64,
}
fn optimize(
    mut x: V<f64>,
    centre: &V<f64>,
    prior: &M<f64>,
    iterations: usize,
    tolerance: f64,
    eval: impl Fn(&V<f64>, bool) -> Result<(V<f64>, Option<M<f64>>)>,
    curvature: impl Fn(&V<f64>, &V<f64>) -> Result<M<f64>>,
) -> Result<Optimized> {
    let bounds = [
        ((1e7_f64 / 1e11).ln(), (1e14_f64 / 1e11).ln()),
        ((100_f64 / 1000.).ln(), (10000_f64 / 1000.).ln()),
        ((100_f64 / 1000.).ln(), (20000_f64 / 1000.).ln()),
        (-3., 3.),
    ];
    let mut history = vec![];
    let mut damping = 1e-2;
    let mut status = "not_converged".to_string();
    for iteration in 0..iterations {
        let (r, j) = eval(&x, true)?;
        let j = j.unwrap();
        let delta = &x - centre;
        let cost = 0.5 * (r.norm_squared() + delta.dot(&(prior * &delta)));
        if history.is_empty() {
            history.push(cost);
        }
        let h = j.transpose() * &j + prior;
        let grad = j.transpose() * &r + prior * &delta;
        let active: Vec<bool> = (0..x.len())
            .map(|i| {
                (x[i] <= bounds[i % 4].0 + 1e-10 && grad[i] > 0.)
                    || (x[i] >= bounds[i % 4].1 - 1e-10 && grad[i] < 0.)
            })
            .collect();
        let mut step_h = h.clone();
        if iteration >= 5 {
            step_h += curvature(&x, &r)?;
        }
        let normalized = (0..x.len())
            .filter(|&i| {
                !((x[i] <= bounds[i % 4].0 + 1e-10 && grad[i] > 0.)
                    || (x[i] >= bounds[i % 4].1 - 1e-10 && grad[i] < 0.))
            })
            .map(|i| grad[i].abs() / h[(i, i)].sqrt().max(1e-12))
            .fold(0., f64::max);
        if iteration % 5 == 0 {
            eprintln!(
                "spline iteration={iteration} objective={cost:.6} normalized_gradient={normalized:.6e} active_bounds={} damping={damping:.3e}",
                active.iter().filter(|v| **v).count()
            );
        }
        if normalized < tolerance {
            status = "optimizer_converged".into();
            break;
        }
        let mut accepted = false;
        for _ in 0..16 {
            let mut hd = step_h.clone();
            let mut rhs = -&grad;
            for i in 0..x.len() {
                hd[(i, i)] += damping * h[(i, i)].max(1.);
            }
            for i in 0..x.len() {
                if active[i] {
                    hd.row_mut(i).fill(0.);
                    hd.column_mut(i).fill(0.);
                    hd[(i, i)] = 1.;
                    rhs[i] = 0.;
                }
            }
            if let Some(c) = hd.cholesky() {
                let step = c.solve(&rhs);
                let mut next = &x + step;
                for i in 0..next.len() {
                    next[i] = next[i].clamp(bounds[i % 4].0, bounds[i % 4].1);
                }
                let (nr, _) = eval(&next, false)?;
                let nd = &next - centre;
                let nc = 0.5 * (nr.norm_squared() + nd.dot(&(prior * &nd)));
                if nc.is_finite() && nc < cost {
                    x = next;
                    history.push(nc);
                    damping = (damping / 3.).max(1e-9);
                    accepted = true;
                    break;
                }
            }
            damping *= 10.;
        }
        if !accepted {
            break;
        }
    }
    let (r, j) = eval(&x, true)?;
    let j = j.unwrap();
    let h = j.transpose() * &j + prior;
    let grad = j.transpose() * &r + prior * (&x - centre);
    let normalized = (0..x.len())
        .filter(|&i| {
            !((x[i] <= bounds[i % 4].0 + 1e-10 && grad[i] > 0.)
                || (x[i] >= bounds[i % 4].1 - 1e-10 && grad[i] < 0.))
        })
        .map(|i| grad[i].abs() / h[(i, i)].sqrt().max(1e-12))
        .fold(0., f64::max);
    if normalized < tolerance {
        status = "optimizer_converged".into();
    }
    let bound_coefficients: Vec<_> = x
        .iter()
        .enumerate()
        .filter_map(|(i, v)| {
            ((v - bounds[i % 4].0).abs() < 1e-6 || (v - bounds[i % 4].1).abs() < 1e-6).then_some(i)
        })
        .collect();
    if !bound_coefficients.is_empty() {
        status = "parameter_boundary".into();
    }
    let cov = h.cholesky().map(|c| c.inverse());
    if cov.is_none() {
        status = "singular".into();
    }
    Ok(Optimized {
        x,
        cov,
        history,
        gradient: grad.amax(),
        normalized,
        bound_coefficients,
        status,
        chi: r.norm_squared(),
    })
}
pub fn solve(input: &Input, s: &SmoothSettings) -> Result<Output> {
    ensure!(
        input
            .gates
            .iter()
            .all(|g| !g.tie_temperatures && g.free_parameters.unwrap_or([true; 4]) == [true; 4]),
        "smooth solver currently requires all four free fields"
    );
    ensure!(!input.measurements.is_empty(), "no measurements");
    for g in &input.gates {
        g.model.validate()?;
    }
    let bas = make_basis(input, s)?;
    let ids: Vec<_> = (0..input.gates.len())
        .flat_map(|g| (0..4).map(move |f| (g, f)))
        .collect();
    let dim = input.measurements.len() * 2;
    ensure!(
        input
            .measurements
            .iter()
            .all(|m| m.sigma.iter().all(|v| v.is_finite() && *v > 0.)
                && m.value.iter().chain(&m.background).all(|v| v.is_finite())
                && m.ranges
                    .iter()
                    .all(|r| r.gate < input.gates.len()
                        && r.kernel.len() == input.frequency_hz.len())),
        "invalid smooth likelihood"
    );
    let w = if let Some(c) = &input.covariance {
        ensure!(
            c.len() == dim && c.iter().all(|r| r.len() == dim),
            "covariance dimensions"
        );
        let m = M::from_fn(dim, dim, |i, j| c[i][j]);
        ensure!(
            (&m - m.transpose()).amax() < 1e-10 * m.amax(),
            "asymmetric covariance"
        );
        Whitening::Dense(
            m.cholesky()
                .ok_or_else(|| anyhow::anyhow!("nonpositive covariance"))?
                .l()
                .try_inverse()
                .ok_or_else(|| anyhow::anyhow!("singular whitening"))?,
        )
    } else {
        Whitening::Diagonal(V::from_iterator(
            dim,
            input
                .measurements
                .iter()
                .flat_map(|m| [1. / m.sigma[0], 1. / m.sigma[1]]),
        ))
    };
    let observed = V::from_iterator(
        dim,
        input
            .measurements
            .iter()
            .flat_map(|m| [m.value[0] - m.background[0], m.value[1] - m.background[1]]),
    );
    let eval = |q: &V<f64>, derivatives: bool| -> Result<(V<f64>, Option<M<f64>>)> {
        let x = &bas.transform * q;
        let r = w.apply(prediction(input, &x, &ids)? - &observed);
        let j = if derivatives {
            let columns = (0..x.len())
                .into_par_iter()
                .map(|k| Ok(w.apply(local_derivative(input, &x, &ids, k)?)))
                .collect::<Result<Vec<_>>>()?;
            {
                let mut reduced = M::zeros(dim, bas.transform.ncols());
                for (i, column) in columns.iter().enumerate() {
                    for j in (i % 4..reduced.ncols()).step_by(4) {
                        let w = bas.transform[(i, j)];
                        if w != 0. {
                            reduced.column_mut(j).axpy(w, column, 1.);
                        }
                    }
                }
                Some(reduced)
            }
        } else {
            None
        };
        Ok((r, j))
    };
    let fit = optimize(
        bas.initial.clone(),
        &bas.centre,
        &bas.q,
        input.settings.max_iterations,
        input.settings.gradient_tolerance,
        eval,
        |q, r| {
            let x = &bas.transform * q;
            let h = residual_curvature(input, &x, &ids, &w, r)?;
            Ok(bas.transform.transpose() * h * &bas.transform)
        },
    )?;
    let x = &bas.transform * &fit.x;
    let plasma: Vec<_> = input
        .gates
        .iter()
        .enumerate()
        .map(|(i, g)| state(g, &x, &ids, i))
        .collect();
    let mut errors = vec![None; ids.len()];
    if let Some(c) = &fit.cov {
        let diag = &bas.transform * c * bas.transform.transpose();
        for (i, &(g, f)) in ids.iter().enumerate() {
            let p = plasma[g];
            let scale = [
                p.electron_density_m3,
                p.ion_temperature_k,
                p.electron_temperature_k,
                1000.,
            ][f];
            errors[i] = Some(diag[(i, i)].max(0.).sqrt() * scale);
        }
    }
    let predicted = prediction(input, &x, &ids)?;
    let centre = &bas.transform * &bas.centre;
    let representation=Representation{active_bound_coefficients:fit.bound_coefficients,temporal_sources:s.temporal_neighbours.iter().map(|n|n.source.clone()).collect(),kind:"cubic B-spline; MSIS-warped coordinate; Matérn-5/2 coefficient covariance; no Chapman function".into(),knots:bas.t,warped_altitude:bas.u,coefficients:(0..bas.knots).map(|i|std::array::from_fn(|f|fit.x[4*i+f])).collect(),coefficient_covariance:fit.cov.as_ref().map(|c|(0..c.nrows()).map(|i|(0..c.ncols()).map(|j|c[(i,j)]).collect()).collect()).unwrap_or_default(),prior_standard_deviation:s.field_standard_deviation,prior_centre:centre.as_slice().chunks_exact(4).map(|v|[v[0],v[1],v[2],v[3]]).collect(),normalized_gradient_max:fit.normalized};
    Ok(Output{schema:"ruisdap-full-profile-map-v1".into(),status:fit.status,plasma,predicted:predicted.as_slice().chunks_exact(2).map(|v|[v[0],v[1]]).collect(),objective_history:fit.history,chi_square:fit.chi,inferred_gradient_variance:s.field_standard_deviation.map(|v|v*v),edge_precision:vec![],gradient_max:fit.gradient,parameter_order:ids,conditional_standard_error:errors,uncertainty:"Conditional spline-coefficient Gauss-Newton covariance propagated to physical fields; is a local Gaussian approximation (not a truncated posterior at active bounds); excludes estimated-noise, prior-scale, forward-model and calibration uncertainty. Temporal neighbours, when supplied, are held fixed in this conditional solve; these errors are not joint time-marginal uncertainties.".into(),scientific_acceptance:"experimental_not_accepted".into(),smooth_profile:Some(representation)})
}
#[cfg(test)]
mod tests {
    use super::*;
    #[test]
    fn residual_curvature_solver_converges_on_curved_valley() {
        let x = V::from_vec(vec![-1.2, 1., 0., 0.]);
        let zero = V::zeros(4);
        let prior = M::identity(4, 4) * 1e-8;
        let fit = optimize(
            x,
            &zero,
            &prior,
            150,
            1e-6,
            |x, der| {
                let r = V::from_vec(vec![10. * (x[1] - x[0] * x[0]), 1. - x[0]]);
                let j = if der {
                    Some(M::from_row_slice(
                        2,
                        4,
                        &[-20. * x[0], 10., 0., 0., -1., 0., 0., 0.],
                    ))
                } else {
                    None
                };
                Ok((r, j))
            },
            |_, r| {
                let mut h = M::zeros(4, 4);
                h[(0, 0)] = -20. * r[0];
                Ok(h)
            },
        )
        .unwrap();
        assert_eq!(fit.status, "optimizer_converged");
        assert!((fit.x[0] - 1.).abs() < 1e-4 && (fit.x[1] - 1.).abs() < 1e-4);
        assert!(fit.history.windows(2).all(|v| v[1] <= v[0]));
    }
    #[test]
    fn cubic_partition_and_slope_continuity() {
        let t = vec![0., 0., 0., 0., 1., 2., 3., 3., 3., 3.];
        for i in 0..301 {
            let b = basis(&t, i as f64 / 100.);
            assert!((b.iter().sum::<f64>() - 1.).abs() < 1e-12);
            assert!(b.iter().all(|x| *x >= 0.));
        }
        let f = |u| {
            basis(&t, u)
                .iter()
                .enumerate()
                .map(|(i, b)| b * (i * i) as f64)
                .sum::<f64>()
        };
        for z in [1., 2.] {
            let e = 1e-5;
            assert!(((f(z) - f(z - e)) / e - (f(z + e) - f(z)) / e).abs() < 1e-3);
        }
    }
    #[test]
    fn heldout_smooth_signal_and_variance_inflation() {
        let n = 8;
        let t: Vec<_> = [vec![0.; 4], vec![1., 2., 3., 4.], vec![5.; 4]].concat();
        let b = M::from_fn(41, n, |i, j| basis(&t, i as f64 / 8.)[j]);
        let truth = V::from_iterator(n, (0..n).map(|i| 0.4 * ((i as f64 - 3.) / 3.).sin()));
        let y = &b * &truth;
        let k = M::from_fn(n, n, |i, j| {
            matern((i as f64 - j as f64) / 2.) + if i == j { 1e-9 } else { 0. }
        });
        let prior = k.cholesky().unwrap().inverse();
        let solve_case = |omit: bool| {
            let take: Vec<_> = (0..41)
                .filter(|&i| !omit || !(16..=24).contains(&i))
                .collect();
            let a = M::from_fn(take.len(), n, |i, j| b[(take[i], j)] / 0.02);
            let obs = V::from_iterator(take.len(), take.iter().map(|&i| y[i] / 0.02));
            let zero = V::zeros(n);
            optimize(
                zero.clone(),
                &zero,
                &prior,
                100,
                1e-6,
                |x, der| Ok((&a * x - &obs, if der { Some(a.clone()) } else { None })),
                |_, _| Ok(M::zeros(n, n)),
            )
            .unwrap()
        };
        let full = solve_case(false);
        let missing = solve_case(true);
        assert_eq!(missing.status, "optimizer_converged");
        let row = b.row(20);
        assert!(((&b * &missing.x)[20] - y[20]).abs() < 0.05);
        let vf = (row * full.cov.unwrap() * row.transpose())[(0, 0)];
        let vm = (row * missing.cov.unwrap() * row.transpose())[(0, 0)];
        assert!(vm > vf);
    }
}
