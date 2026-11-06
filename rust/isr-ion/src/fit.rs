//! Bounded Levenberg–Marquardt in physical parameter transforms. No SVD or lag filling.
use crate::{
    spectrum::{PlasmaState, SpectrumModel},
    Complex as C,
};
use anyhow::{ensure, Result};
use nalgebra::{DMatrix, DVector};
use serde::{Deserialize, Serialize};

#[derive(Clone, Debug)]
pub struct Measurement {
    pub lag_s: f64,
    pub value: C,
    pub sigma: [f64; 2],
    /// Frequency response including quadrature weights, in Hz. No assumed ideal lag.
    pub kernel: Vec<C>,
}
#[derive(Clone, Debug, Serialize, Deserialize)]
pub struct FitSettings {
    pub tie_electron_ion_temperature: bool,
    pub fit_collision_rate: bool,
    pub fixed_collision_s: f64,
    pub max_iterations: usize,
    #[serde(default)]
    pub millstone_historical_bounds: bool,
    /// Optional reference density for the spectral shape. The fitted density
    /// remains the calibrated amplitude scale. This reproduces the legacy
    /// ion-line approximation without changing the physically coupled default.
    #[serde(default)]
    pub fixed_shape_density_m3: Option<f64>,
    /// Match legacy ion-line tables that removed the broad electron continuum.
    #[serde(default)]
    pub legacy_ion_only: bool,
}
impl Default for FitSettings {
    fn default() -> Self {
        Self {
            tie_electron_ion_temperature: false,
            fit_collision_rate: false,
            fixed_collision_s: 0.,
            max_iterations: 80,
            millstone_historical_bounds: false,
            fixed_shape_density_m3: None,
            legacy_ion_only: false,
        }
    }
}
/// Display prediction averaged over response groups at the same measured lag.
/// Errors are pointwise standard errors of the fitted mean, not measurement noise.
#[derive(Clone, Debug, Serialize, Deserialize)]
pub struct ModelPoint {
    pub lag_s: f64,
    pub value: [f64; 2],
    pub standard_error: Option<[f64; 2]>,
}
#[derive(Clone, Debug, Serialize, Deserialize)]
pub struct FitResult {
    pub plasma: PlasmaState,
    /// Parameter order: Ne, Ti, Te, signed spectral velocity, collision rate.
    /// None for fixed, unidentified, or singular errors; tied Te copies Ti error.
    pub standard_error: [Option<f64>; 5],
    pub conditional_standard_error: [Option<f64>; 5],
    pub covariance: Vec<f64>,
    pub free_parameter_indices: Vec<usize>,
    pub chi_square: f64,
    pub degrees_of_freedom: usize,
    pub reduced_chi_square: f64,
    pub iterations: usize,
    pub status: String,
    pub predicted: Vec<[f64; 2]>,
    #[serde(default)]
    pub model_curve: Vec<ModelPoint>,
    pub standardized_residual: Vec<[f64; 2]>,
}
#[derive(Clone, Debug)]
pub struct FitProblem {
    pub model: SpectrumModel,
    pub frequency_hz: Vec<f64>,
    pub measurements: Vec<Measurement>,
    pub settings: FitSettings,
}
impl FitProblem {
    pub fn predict(&self, p: PlasmaState) -> Result<Vec<C>> {
        let mut shape = p;
        if let Some(density) = self.settings.fixed_shape_density_m3 {
            ensure!(density.is_finite() && density > 0., "invalid fixed shape density");
            shape.electron_density_m3 = density;
        }
        let s = if self.settings.legacy_ion_only {
            self.model.evaluate_ion_only(shape, &self.frequency_hz)?
        } else {
            self.model.evaluate(shape, &self.frequency_hz)?
        };
        Ok(self
            .measurements
            .iter()
            .map(|m| {
                m.kernel
                    .iter()
                    .zip(&s)
                    .map(|(k, s)| *k * (*s * p.electron_density_m3))
                    .sum()
            })
            .collect())
    }
    fn indices(&self) -> Vec<usize> {
        let mut v = vec![0, 1];
        if !self.settings.tie_electron_ion_temperature {
            v.push(2);
        }
        v.push(3);
        if self.settings.fit_collision_rate {
            v.push(4);
        }
        v
    }
    fn state(&self, x: &DVector<f64>, indices: &[usize]) -> PlasmaState {
        let mut p = [1e11, 1000., 1000., 0., self.settings.fixed_collision_s];
        for (&i, &x) in indices.iter().zip(x.iter()) {
            p[i] = match i {
                0 => 1e11 * x.exp(),
                1 | 2 => 1000. * x.exp(),
                3 => 1000. * x,
                4 => 1000. * x.exp(),
                _ => unreachable!(),
            };
        }
        if self.settings.tie_electron_ion_temperature {
            p[2] = p[1];
        }
        PlasmaState {
            electron_density_m3: p[0],
            ion_temperature_k: p[1],
            electron_temperature_k: p[2],
            velocity_m_s: p[3],
            ion_collision_s: p[4],
        }
    }
    fn residual(
        &self,
        x: &DVector<f64>,
        indices: &[usize],
        whitening: Option<&DMatrix<f64>>,
    ) -> Result<DVector<f64>> {
        let pred = self.predict(self.state(x, indices))?;
        let raw = DVector::from_iterator(
            2 * pred.len(),
            self.measurements
                .iter()
                .zip(pred)
                .flat_map(|(m, p)| [p.re - m.value.re, p.im - m.value.im]),
        );
        if let Some(w) = whitening {
            Ok(w * raw)
        } else {
            Ok(DVector::from_iterator(
                raw.len(),
                raw.iter()
                    .enumerate()
                    .map(|(i, r)| r / self.measurements[i / 2].sigma[i % 2]),
            ))
        }
    }

    fn jacobian(
        &self,
        x: &DVector<f64>,
        ind: &[usize],
        whitening: Option<&DMatrix<f64>>,
    ) -> Result<DMatrix<f64>> {
        let mut j = DMatrix::zeros(self.measurements.len() * 2, ind.len());
        for i in 0..ind.len() {
            let mut a = x.clone();
            let mut b = x.clone();
            a[i] += 1e-4;
            b[i] -= 1e-4;
            j.set_column(
                i,
                &((self.residual(&a, ind, whitening)? - self.residual(&b, ind, whitening)?) / 2e-4),
            );
        }
        Ok(j)
    }
    pub fn solve(&self, initial: PlasmaState) -> Result<FitResult> {
        self.solve_weighted(initial, None)
    }
    /// Refit using a fixed model-based covariance in physical measurement units.
    /// Cholesky whitening preserves every measured lag; no SVD or eigenvalue clipping.
    pub fn solve_with_covariance(
        &self,
        initial: PlasmaState,
        covariance: &DMatrix<f64>,
    ) -> Result<FitResult> {
        let n = 2 * self.measurements.len();
        ensure!(
            covariance.shape() == (n, n) && covariance.iter().all(|v| v.is_finite()),
            "invalid measurement covariance"
        );
        let scale = covariance.amax().max(f64::MIN_POSITIVE);
        ensure!(
            (covariance - covariance.transpose()).amax() <= scale * 1e-10,
            "asymmetric measurement covariance"
        );
        let chol = covariance
            .clone()
            .cholesky()
            .ok_or_else(|| anyhow::anyhow!("measurement covariance is not positive definite"))?;
        let whitening = chol
            .l()
            .try_inverse()
            .ok_or_else(|| anyhow::anyhow!("singular whitening factor"))?;
        self.solve_weighted(initial, Some(&whitening))
    }
    /// Feasible GLS: estimate parameters, build covariance once, then hold it fixed
    /// throughout the second solve. Caller is responsible for validating the first pass.
    pub fn solve_two_pass<F>(
        &self,
        initial: PlasmaState,
        estimate: F,
    ) -> Result<(FitResult, FitResult, DMatrix<f64>)>
    where
        F: FnOnce(&FitResult) -> Result<DMatrix<f64>>,
    {
        let first = self.solve(initial)?;
        let covariance = estimate(&first)?;
        let mut second = self.solve_with_covariance(first.plasma, &covariance)?;
        self.model_uncertainty(&mut second)?;
        Ok((first, second, covariance))
    }
    fn solve_weighted(
        &self,
        initial: PlasmaState,
        whitening: Option<&DMatrix<f64>>,
    ) -> Result<FitResult> {
        self.model.validate()?;
        ensure!(
            self.frequency_hz.len() >= 3 && self.frequency_hz.windows(2).all(|a| a[1] > a[0]),
            "invalid model grid"
        );
        ensure!(
            self.measurements
                .iter()
                .all(|m| m.kernel.len() == self.frequency_hz.len()
                    && m.kernel
                        .iter()
                        .all(|k| k.re.is_finite() && k.im.is_finite())
                    && m.value.re.is_finite()
                    && m.value.im.is_finite()
                    && m.sigma.iter().all(|s| s.is_finite() && *s > 0.)),
            "invalid measurements"
        );
        let ind = self.indices();
        let n = ind.len();
        ensure!(self.measurements.len() * 2 > n, "insufficient measurements");
        let p = [
            initial.electron_density_m3,
            initial.ion_temperature_k,
            initial.electron_temperature_k,
            initial.velocity_m_s,
            initial.ion_collision_s,
        ];
        let mut x = DVector::from_iterator(
            n,
            ind.iter().map(|&i| match i {
                0 => (p[i] / 1e11).ln(),
                1 | 2 => (p[i] / 1000.).ln(),
                3 => p[i] / 1000.,
                4 => (p[i].max(1.) / 1000.).ln(),
                _ => unreachable!(),
            }),
        );
        let mut bounds = [
            ((1e7f64 / 1e11).ln(), (1e14f64 / 1e11).ln()),
            ((80f64 / 1000.).ln(), (10000f64 / 1000.).ln()),
            ((80f64 / 1000.).ln(), (20000f64 / 1000.).ln()),
            (-3., 3.),
            ((0.1f64 / 1000.).ln(), (1e6f64 / 1000.).ln()),
        ];
        if self.settings.millstone_historical_bounds {
            bounds[1]=((150_f64/1000.).ln(),(3000_f64/1000.).ln());
            bounds[2]=((148.5_f64/1000.).ln(),(9000_f64/1000.).ln());bounds[3]=(-1.5,1.5);
        }
        for (j, &i) in ind.iter().enumerate() {
            x[j] = x[j].clamp(bounds[i].0, bounds[i].1);
        }
        if self.settings.millstone_historical_bounds&&!self.settings.tie_electron_ion_temperature {
            let ti=ind.iter().position(|i|*i==1).unwrap();let te=ind.iter().position(|i|*i==2).unwrap();x[te]=x[te].clamp(x[ti]+0.99_f64.ln(),x[ti]+3_f64.ln());
        }
        let mut r = self.residual(&x, &ind, whitening)?;
        let mut cost = r.norm_squared();
        let mut damping = 1e-3;
        let mut converged = false;
        let mut iterations = 0;
        for iteration in 0..self.settings.max_iterations {
            iterations = iteration + 1;
            let j = self.jacobian(&x, &ind, whitening)?;
            let h = j.transpose() * &j;
            let g = j.transpose() * &r;
            if g.amax() < 1e-7 {
                converged = true;
                break;
            }
            let mut damped = h.clone();
            for k in 0..n {
                damped[(k, k)] += damping * h[(k, k)].max(1e-12);
            }
            let Some(chol) = damped.cholesky() else {
                damping *= 10.;
                continue;
            };
            let scaled_gradient = (0..n)
                .map(|k| g[k].abs() / h[(k, k)].max(1e-24).sqrt())
                .fold(0., f64::max)
                / (1. + cost).sqrt();
            let step = chol.solve(&(-g));
            let mut candidate = &x + &step;
            for (k, &i) in ind.iter().enumerate() {
                candidate[k] = candidate[k].clamp(bounds[i].0, bounds[i].1);
            }
            if self.settings.millstone_historical_bounds&&!self.settings.tie_electron_ion_temperature {
                let ti=ind.iter().position(|i|*i==1).unwrap();let te=ind.iter().position(|i|*i==2).unwrap();candidate[te]=candidate[te].clamp(candidate[ti]+0.99_f64.ln(),candidate[ti]+3_f64.ln());
            }
            let next = match self.residual(&candidate, &ind, whitening) {
                Ok(r) if r.iter().all(|x| x.is_finite()) => r,
                _ => {
                    damping *= 8.;
                    if damping > 1e16 {
                        break;
                    }
                    continue;
                }
            };
            let nextcost = next.norm_squared();
            if nextcost < cost {
                let change = cost - nextcost;
                let effective = (&candidate - &x).norm();
                x = candidate;
                r = next;
                cost = nextcost;
                damping = (damping / 3.).max(1e-12);
                if change < 1e-7 * (1. + cost) && effective < 1e-4 && scaled_gradient < 1e-5 {
                    converged = true;
                    break;
                }
            } else {
                damping *= 8.;
                if damping > 1e16 {
                    break;
                }
            }
        }
        let plasma = self.state(&x, &ind);
        let j = self.jacobian(&x, &ind, whitening)?;
        let h = j.transpose() * j;
        let inverse = h.cholesky().map(|c| c.inverse());
        let physical = [
            plasma.electron_density_m3,
            plasma.ion_temperature_k,
            plasma.electron_temperature_k,
            1000.,
            plasma.ion_collision_s,
        ];
        let mut errors = [None; 5];
        let mut covariance = vec![];
        if let Some(inv) = inverse {
            for (a, &i) in ind.iter().enumerate() {
                errors[i] = Some(inv[(a, a)].max(0.).sqrt() * physical[i]);
                for (b, &k) in ind.iter().enumerate() {
                    covariance.push(inv[(a, b)] * physical[i] * physical[k]);
                }
            }
        }
        if self.settings.tie_electron_ion_temperature {
            errors[2] = errors[1];
        }
        let at_boundary = ind
            .iter()
            .enumerate()
            .any(|(k, &i)| (x[k] - bounds[i].0).abs() < 1e-4 || (x[k] - bounds[i].1).abs() < 1e-4);
        let at_boundary=at_boundary||(self.settings.millstone_historical_bounds&&((plasma.electron_temperature_k/plasma.ion_temperature_k-0.99).abs()<1e-4||(plasma.electron_temperature_k/plasma.ion_temperature_k-3.).abs()<1e-4));
        let status = if at_boundary {
            "parameter_boundary"
        } else if covariance.is_empty() {
            "singular"
        } else if !converged {
            "not_converged"
        } else if errors[0].is_none_or(|e| e > plasma.electron_density_m3) {
            "low_information"
        } else {
            "converged"
        };
        let dof = 2 * self.measurements.len() - n;
        let predicted = self.predict(plasma)?.iter().map(|z| [z.re, z.im]).collect();
        let residual = r.as_slice().as_chunks::<2>().0.to_vec();
        Ok(FitResult {
            plasma,
            standard_error: errors,
            conditional_standard_error: errors,
            covariance,
            free_parameter_indices: ind,
            chi_square: cost,
            degrees_of_freedom: dof,
            reduced_chi_square: cost / dof as f64,
            iterations,
            status: status.into(),
            predicted,
            model_curve: vec![],
            standardized_residual: residual,
        })
    }
}

#[cfg(test)]
mod fixed_shape_tests {
    use super::*;
    use crate::spectrum::IonSpecies;

    #[test]
    fn fixed_shape_density_separates_amplitude_from_spectral_shape() {
        let model = SpectrumModel {
            radar_frequency_hz: 440.2e6,
            ions: vec![IonSpecies { mass_amu: 16., fraction: 1. }],
            electron_collision_s: 0.,
        };
        let grid: Vec<_> = (-100..=100).map(|i| i as f64 * 250.).collect();
        let measurements = vec![Measurement {
            lag_s: 1e-5,
            value: C::new(0., 0.),
            sigma: [1., 1.],
            kernel: grid.iter().map(|f| C::from_polar(250., -std::f64::consts::TAU * f * 1e-5)).collect(),
        }];
        let mut settings = FitSettings::default();
        settings.fixed_shape_density_m3 = Some(1e12);
        let problem = FitProblem { model, frequency_hz: grid, measurements, settings };
        let p = PlasmaState { electron_density_m3: 1e11, ion_temperature_k: 900., electron_temperature_k: 1800., velocity_m_s: 20., ion_collision_s: 0. };
        let first = problem.predict(p).unwrap()[0];
        let second = problem.predict(PlasmaState { electron_density_m3: 2e11, ..p }).unwrap()[0];
        assert!((second - first * 2.).norm() < first.norm() * 1e-12);
    }
}

impl FitProblem {
    /// Cluster sandwich covariance: one independent cluster per input integration.
    /// Includes correlation between every lag and range average within a cluster.
    /// Variation of the plasma within the window also increases these errors.
    pub fn temporal_uncertainty(&self, result: &mut FitResult, samples: &[Vec<C>]) -> Result<()> {
        ensure!(
            samples.len() >= 3
                && samples.iter().all(|s| s.len() == self.measurements.len()
                    && s.iter().all(|v| v.re.is_finite() && v.im.is_finite())),
            "invalid temporal replicates"
        );
        let ind = self.indices();
        let p = result.plasma;
        let physical = [
            p.electron_density_m3,
            p.ion_temperature_k,
            p.electron_temperature_k,
            1000.,
            p.ion_collision_s,
        ];
        let values = [
            p.electron_density_m3,
            p.ion_temperature_k,
            p.electron_temperature_k,
            p.velocity_m_s,
            p.ion_collision_s,
        ];
        let x = DVector::from_iterator(
            ind.len(),
            ind.iter().map(|&i| match i {
                0 => (values[i] / 1e11).ln(),
                1 | 2 => (values[i] / 1000.).ln(),
                3 => values[i] / 1000.,
                4 => (values[i] / 1000.).ln(),
                _ => unreachable!(),
            }),
        );
        let j = self.jacobian(&x, &ind, None)?;
        let Some(chol) = (j.transpose() * &j).cholesky() else {
            result.status = "singular".into();
            result.standard_error = [None; 5];
            result.covariance.clear();
            return Ok(());
        };
        let inverse = chol.inverse();
        let mut meat = DMatrix::zeros(ind.len(), ind.len());
        for sample in samples {
            let d = DVector::from_iterator(
                2 * sample.len(),
                sample.iter().zip(&self.measurements).flat_map(|(v, m)| {
                    [
                        (v.re - m.value.re) / m.sigma[0],
                        (v.im - m.value.im) / m.sigma[1],
                    ]
                }),
            );
            let score = j.transpose() * d;
            meat += &score * score.transpose();
        }
        let mut cov = &inverse * meat * &inverse / (samples.len() * (samples.len() - 1)) as f64;
        for a in 0..ind.len() {
            cov[(a, a)] = cov[(a, a)].max(inverse[(a, a)]);
        }
        result.standard_error = [None; 5];
        result.covariance.clear();
        for (a, &i) in ind.iter().enumerate() {
            result.standard_error[i] = Some(cov[(a, a)].max(0.).sqrt() * physical[i]);
            for (b, &k) in ind.iter().enumerate() {
                result
                    .covariance
                    .push(cov[(a, b)] * physical[i] * physical[k]);
            }
        }
        if self.settings.tie_electron_ion_temperature {
            result.standard_error[2] = result.standard_error[1];
        }
        if result.status == "converged"
            && [0, 1, 2]
                .iter()
                .any(|&i| result.standard_error[i].is_none_or(|e| !e.is_finite() || e > values[i]))
        {
            result.status = "low_information".into();
        }
        Ok(())
    }
}

impl FitProblem {
    /// Build a line on measured lags only. Average the response gradients before
    /// propagating covariance: repeated responses share fitted parameters and
    /// must not be treated as independent predictions. Call after covariance updates.
    pub fn model_uncertainty(&self, result: &mut FitResult) -> Result<()> {
        let ind = self.indices();
        ensure!(
            result.free_parameter_indices == ind,
            "fit parameter mismatch"
        );
        ensure!(
            self.measurements.iter().all(|m| m.lag_s.is_finite()),
            "invalid lag"
        );
        let p = result.plasma;
        let values = [
            p.electron_density_m3,
            p.ion_temperature_k,
            p.electron_temperature_k,
            p.velocity_m_s,
            p.ion_collision_s,
        ];
        let scales = [values[0], values[1], values[2], 1000., values[4]];
        let x = DVector::from_iterator(
            ind.len(),
            ind.iter().map(|&i| match i {
                0 => (values[i] / 1e11).ln(),
                1 | 2 | 4 => (values[i] / 1000.).ln(),
                3 => values[i] / 1000.,
                _ => unreachable!(),
            }),
        );
        let pred = self.predict(p)?;
        let cov = if result.covariance.len() == ind.len() * ind.len()
            && result.covariance.iter().all(|v| v.is_finite())
        {
            Some(DMatrix::from_fn(ind.len(), ind.len(), |a, b| {
                result.covariance[a * ind.len() + b] / scales[ind[a]] / scales[ind[b]]
            }))
        } else {
            None
        };
        // Differentiate predictions directly; no subtraction of noisy observations.
        // state() applies tied-temperature constraints to every perturbation.
        let mut derivatives = Vec::new();
        if cov.is_some() {
            for i in 0..ind.len() {
                let mut a = x.clone();
                let mut b = x.clone();
                a[i] += 1e-4;
                b[i] -= 1e-4;
                derivatives.push(
                    self.predict(self.state(&a, &ind))?
                        .iter()
                        .zip(self.predict(self.state(&b, &ind))?)
                        .map(|(a, b)| (*a - b) / 2e-4)
                        .collect::<Vec<_>>(),
                );
            }
        }
        let mut order: Vec<_> = (0..self.measurements.len()).collect();
        order.sort_by(|&a, &b| {
            self.measurements[a]
                .lag_s
                .total_cmp(&self.measurements[b].lag_s)
        });
        result.model_curve.clear();
        for rows in
            order.chunk_by(|&a, &b| self.measurements[a].lag_s == self.measurements[b].lag_s)
        {
            let mean: C = rows.iter().map(|&i| pred[i]).sum::<C>() / rows.len() as f64;
            let error = cov.as_ref().and_then(|cov| {
                let mut errors = [0.; 2];
                for (component, error) in errors.iter_mut().enumerate() {
                    let gradient = DVector::from_iterator(
                        ind.len(),
                        derivatives.iter().map(|d| {
                            let mean: C = rows.iter().map(|&i| d[i]).sum::<C>() / rows.len() as f64;
                            if component == 0 {
                                mean.re
                            } else {
                                mean.im
                            }
                        }),
                    );
                    let variance = gradient.dot(&(cov * &gradient));
                    if !variance.is_finite() || variance < 0. {
                        return None;
                    }
                    *error = variance.sqrt();
                }
                Some(errors)
            });
            result.model_curve.push(ModelPoint {
                lag_s: self.measurements[rows[0]].lag_s,
                value: [mean.re, mean.im],
                standard_error: error,
            });
        }
        Ok(())
    }
}
