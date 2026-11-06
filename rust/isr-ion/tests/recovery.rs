use isr_ion::{
    fit::{FitProblem, FitSettings, Measurement},
    spectrum::{IonSpecies, PlasmaState, SpectrumModel},
    Complex as C,
};
use std::f64::consts::PI;
fn problem(p: PlasmaState, collision: bool) -> FitProblem {
    let frequency_hz: Vec<_> = (-800..=800).map(|i| i as f64 * 50.).collect();
    let model = SpectrumModel {
        radar_frequency_hz: 930e6,
        ions: vec![IonSpecies {
            mass_amu: 16.,
            fraction: 1.,
        }],
        electron_collision_s: 0.,
    };
    let measurements = (1..=41)
        .map(|i| {
            let lag = i as f64 * 10e-6;
            Measurement {
                lag_s: lag,
                value: C::new(0., 0.),
                sigma: [2e8; 2],
                kernel: frequency_hz
                    .iter()
                    .map(|f| C::from_polar(50., 2. * PI * f * lag))
                    .collect(),
            }
        })
        .collect();
    let mut problem = FitProblem {
        model,
        frequency_hz,
        measurements,
        settings: FitSettings {
            fit_collision_rate: collision,
            ..Default::default()
        },
    };
    let values = problem.predict(p).unwrap();
    for (m, v) in problem.measurements.iter_mut().zip(values) {
        m.value = v;
    }
    problem
}
#[test]
fn recover_parameters_from_all_41_complex_lags_without_zero_lag() {
    let truth = PlasmaState {
        electron_density_m3: 3e11,
        ion_temperature_k: 900.,
        electron_temperature_k: 1900.,
        velocity_m_s: -230.,
        ion_collision_s: 0.,
    };
    let problem = problem(truth, false);
    let initial = PlasmaState {
        electron_density_m3: 1e11,
        ion_temperature_k: 1300.,
        electron_temperature_k: 1300.,
        velocity_m_s: 0.,
        ion_collision_s: 0.,
    };
    let fit = problem.solve(initial).unwrap();
    assert_eq!(fit.status, "converged", "{fit:?}");
    assert!((fit.plasma.electron_density_m3 / truth.electron_density_m3 - 1.).abs() < 1e-4);
    assert!((fit.plasma.ion_temperature_k / truth.ion_temperature_k - 1.).abs() < 1e-4);
    assert!((fit.plasma.electron_temperature_k / truth.electron_temperature_k - 1.).abs() < 1e-4);
    assert!((fit.plasma.velocity_m_s - truth.velocity_m_s).abs() < 0.01);
    assert_eq!(fit.predicted.len(), 41);
    assert!(fit.standard_error[..4].iter().all(|e| e.is_some()));
    // No hidden fit state: interleaving a different fit must not change this result.
    let other = PlasmaState {
        velocity_m_s: 370.,
        ..truth
    };
    let _ = problem::run_other(other);
    let again = problem.solve(initial).unwrap();
    assert_eq!(
        fit.plasma.electron_density_m3,
        again.plasma.electron_density_m3
    );
}
mod problem {
    use super::*;
    pub fn run_other(p: PlasmaState) -> f64 {
        super::problem(p, false).solve(p).unwrap().chi_square
    }
}
#[test]
fn recover_effective_collision_rate() {
    let truth = PlasmaState {
        electron_density_m3: 2e11,
        ion_temperature_k: 600.,
        electron_temperature_k: 600.,
        velocity_m_s: 70.,
        ion_collision_s: 2500.,
    };
    let mut problem = problem(truth, true);
    problem.settings.tie_electron_ion_temperature = true;
    let fit = problem
        .solve(PlasmaState {
            ion_collision_s: 1000.,
            ion_temperature_k: 800.,
            ..truth
        })
        .unwrap();
    assert_eq!(fit.status, "converged", "{fit:?}");
    assert!((fit.plasma.ion_collision_s / truth.ion_collision_s - 1.).abs() < 1e-3);
    assert!((fit.plasma.ion_temperature_k / truth.ion_temperature_k - 1.).abs() < 1e-3);
}
#[test]
fn invalid_composition_and_measurement_errors_are_rejected() {
    let p = PlasmaState {
        electron_density_m3: 1e11,
        ion_temperature_k: 1000.,
        electron_temperature_k: 1000.,
        velocity_m_s: 0.,
        ion_collision_s: 0.,
    };
    let mut problem = problem(p, false);
    problem.measurements[0].sigma[0] = 0.;
    assert!(problem.solve(p).is_err());
    problem.measurements[0].sigma[0] = 1.;
    problem.model.ions[0].fraction = 0.5;
    assert!(problem.solve(p).is_err());
}

#[test]
fn correlated_replicates_do_not_gain_information_when_lags_are_duplicated() {
    let truth = PlasmaState {
        electron_density_m3: 3e11,
        ion_temperature_k: 900.,
        electron_temperature_k: 1900.,
        velocity_m_s: -230.,
        ion_collision_s: 0.,
    };
    let mut p = problem(truth, false);
    let mut fit = p.solve(truth).unwrap();
    let samples: Vec<_> = (0..12)
        .map(|i| {
            p.predict(PlasmaState {
                electron_density_m3: truth.electron_density_m3
                    * (1. + 0.08 * (i as f64 - 5.5) / 5.5),
                ..truth
            })
            .unwrap()
        })
        .collect();
    p.temporal_uncertainty(&mut fit, &samples).unwrap();
    let error = fit.standard_error[0].unwrap();
    assert!(error > 1e9 && error < 1e10, "density uncertainty {error}");
    let old = p.measurements.clone();
    p.measurements.extend(old);
    let doubled: Vec<_> = samples
        .iter()
        .map(|s| s.iter().chain(s.iter()).copied().collect())
        .collect();
    let mut fit2 = p.solve(truth).unwrap();
    p.temporal_uncertainty(&mut fit2, &doubled).unwrap();
    assert!((fit2.standard_error[0].unwrap() / error - 1.).abs() < 1e-6);
}
#[test]
fn negative_echo_is_preserved_and_does_not_become_a_successful_density_fit() {
    let truth = PlasmaState {
        electron_density_m3: 3e11,
        ion_temperature_k: 900.,
        electron_temperature_k: 1900.,
        velocity_m_s: 0.,
        ion_collision_s: 0.,
    };
    let mut p = problem(truth, false);
    for m in &mut p.measurements {
        m.value = -m.value;
    }
    let stored = p.measurements[0].value;
    let fit = p.solve(truth).unwrap();
    assert_ne!(fit.status, "converged");
    assert_eq!(p.measurements[0].value, stored);
}

#[test]
fn model_band_propagates_correlations_ties_and_shared_response_averages() {
    for tied in [false, true] {
        let truth = PlasmaState {
            electron_density_m3: 3e11,
            ion_temperature_k: 900.,
            electron_temperature_k: if tied { 900. } else { 1900. },
            velocity_m_s: -230.,
            ion_collision_s: 500.,
        };
        let mut p = problem(truth, true);
        p.settings.tie_electron_ion_temperature = tied;
        let mut fit = p.solve(truth).unwrap();
        // Perfectly correlated physical perturbation. The prediction variance
        // must equal the square of an independent directional derivative.
        let direction = [3e9, 20., if tied { 20. } else { 35. }, 5., 50.];
        fit.covariance = fit
            .free_parameter_indices
            .iter()
            .flat_map(|&i| {
                fit.free_parameter_indices
                    .iter()
                    .map(move |&j| direction[i] * direction[j])
            })
            .collect();
        let mut extra = p.measurements.clone();
        for m in &mut extra {
            for k in &mut m.kernel {
                *k *= 2.;
            }
        }
        p.measurements.extend(extra);
        p.measurements.reverse(); // Output must be ordered by lag, not raw address.
        p.model_uncertainty(&mut fit).unwrap();
        let eps = 1e-3;
        let perturb = |sign: f64| PlasmaState {
            electron_density_m3: truth.electron_density_m3 + sign * eps * direction[0],
            ion_temperature_k: truth.ion_temperature_k + sign * eps * direction[1],
            electron_temperature_k: truth.electron_temperature_k + sign * eps * direction[2],
            velocity_m_s: truth.velocity_m_s + sign * eps * direction[3],
            ion_collision_s: truth.ion_collision_s + sign * eps * direction[4],
        };
        let plus = p.predict(perturb(1.)).unwrap();
        let minus = p.predict(perturb(-1.)).unwrap();
        let nominal = p.predict(truth).unwrap();
        assert_eq!(fit.model_curve.len(), 41);
        assert!(fit.model_curve.windows(2).all(|w| w[0].lag_s < w[1].lag_s));
        for point in &fit.model_curve {
            let rows: Vec<_> = p
                .measurements
                .iter()
                .enumerate()
                .filter(|(_, m)| m.lag_s == point.lag_s)
                .map(|(i, _)| i)
                .collect();
            let derivative: C = rows
                .iter()
                .map(|&i| (plus[i] - minus[i]) / (2. * eps))
                .sum::<C>()
                / rows.len() as f64;
            let mean: C = rows.iter().map(|&i| nominal[i]).sum::<C>() / rows.len() as f64;
            let errors = point.standard_error.unwrap();
            for (actual, expected) in errors
                .into_iter()
                .zip([derivative.re.abs(), derivative.im.abs()])
            {
                assert!(
                    (actual - expected).abs() < 1e-4 * expected.max(1.),
                    "{actual} vs {expected}"
                );
            }
            assert!((point.value[0] - mean.re).abs() < 1e-3);
        }
        let previous = fit.model_curve.clone();
        p.measurements.extend(p.measurements.clone());
        p.model_uncertainty(&mut fit).unwrap();
        for (a, b) in previous.iter().zip(&fit.model_curve) {
            for (ea, eb) in a
                .standard_error
                .unwrap()
                .into_iter()
                .zip(b.standard_error.unwrap())
            {
                assert!((ea - eb).abs() < 1e-5 * ea.max(1.));
            }
        }
        fit.covariance.clear();
        p.model_uncertainty(&mut fit).unwrap();
        assert!(fit.model_curve.iter().all(|p| p.standard_error.is_none()));
    }
}

#[test]
fn full_covariance_diagonal_matches_existing_weights_and_rejects_singular() {
    let truth = PlasmaState {
        electron_density_m3: 3e11,
        ion_temperature_k: 900.,
        electron_temperature_k: 1900.,
        velocity_m_s: -230.,
        ion_collision_s: 0.,
    };
    let p = problem(truth, false);
    let n = p.measurements.len() * 2;
    let covariance = nalgebra::DMatrix::from_diagonal(&nalgebra::DVector::from_iterator(
        n,
        p.measurements.iter().flat_map(|m| m.sigma.map(|s| s * s)),
    ));
    let (first, second, stored) = p.solve_two_pass(truth, |_| Ok(covariance.clone())).unwrap();
    assert!((first.chi_square - second.chi_square).abs() < 1e-10);
    assert!((second.plasma.electron_density_m3 / truth.electron_density_m3 - 1.).abs() < 1e-6);
    assert_eq!(stored, covariance);
    assert!(p
        .solve_with_covariance(truth, &nalgebra::DMatrix::zeros(n, n))
        .is_err());
}

#[test]
fn two_ion_fit_recovers_from_cold_and_displaced_initial_guesses() {
    let truth = PlasmaState {
        electron_density_m3: 2e11,
        ion_temperature_k: 750.,
        electron_temperature_k: 1400.,
        velocity_m_s: -180.,
        ion_collision_s: 600.,
    };
    let mut p = problem(truth, true);
    p.model.ions = vec![
        IonSpecies {
            mass_amu: 16.,
            fraction: 0.15,
        },
        IonSpecies {
            mass_amu: 30.5,
            fraction: 0.85,
        },
    ];
    p.settings.max_iterations = 160;
    let measured = p.predict(truth).unwrap();
    for (m, y) in p.measurements.iter_mut().zip(measured) {
        m.value = y;
    }
    let cold = PlasmaState {
        electron_density_m3: 1e11,
        ion_temperature_k: 80.,
        electron_temperature_k: 400.,
        velocity_m_s: 250.,
        ion_collision_s: 3000.,
    };
    let fits = [
        cold,
        PlasmaState {
            ion_temperature_k: 500.,
            electron_temperature_k: 1000.,
            velocity_m_s: 0.,
            ion_collision_s: 300.,
            ..truth
        },
    ]
    .map(|seed| p.solve(seed).unwrap());
    let best = fits
        .iter()
        .min_by(|a, b| a.chi_square.total_cmp(&b.chi_square))
        .unwrap();
    assert_eq!(best.status, "converged", "{best:?}");
    assert!((best.plasma.ion_temperature_k / truth.ion_temperature_k - 1.).abs() < 1e-3);
    assert!((best.plasma.electron_density_m3 / truth.electron_density_m3 - 1.).abs() < 1e-3);
    assert_eq!(best.predicted.len(), 41);
}
