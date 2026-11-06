//! Sample-resolved replay of the archived clutter_6 operation.
//! Layout is [cycle, code, received sample]. Adaptive selection is nonlinear:
//! conditional covariance propagation with a selected mask does not include
//! selection uncertainty. No parameter defaults are inferred from a profile.
use crate::{multirate::clutter_operator, Complex as C};
use anyhow::{ensure, Result};
use nalgebra::DMatrix;

#[derive(Clone, Debug)]
pub struct ClutterConfig {
    pub cycles: usize,
    pub codes: usize,
    pub samples: usize,
    pub filtered_samples: usize,
    pub blocks: usize,
    /// Nonnegative: fixed notch; negative: archived adaptive request (e.g. -5).
    /// Disabled filtering is represented by blocks=0, NOT notch=0.
    pub notch_request: i32,
}

#[derive(Clone, Debug)]
pub struct ClutterPlan {
    pub config: ClutterConfig,
    /// -1 means the adaptive selector retained every frequency bin.
    pub block_notches: Vec<i32>,
    sample_filters: Vec<DMatrix<f64>>,
}

impl ClutterConfig {
    /// Parse the eight integer fields in the experiment's *_c.par file.
    /// upar[0] is only used when par[4] is negative, as in clutter_6.
    pub fn from_parameters(par: &[i32], upar0: i32) -> Result<Self> {
        ensure!(
            par.len() == 8
                && par[0] > 0
                && par[1] > 0
                && par[2] > 0
                && par[3] >= 0
                && par[5] >= 0
                && par[1] % par[0] == 0,
            "invalid clutter parameter file"
        );
        let config = Self {
            cycles: (par[1] / par[0]) as usize,
            codes: par[0] as usize,
            samples: par[2] as usize,
            filtered_samples: par[3] as usize,
            blocks: par[5].min(par[3]) as usize,
            notch_request: if par[4] < 0 { upar0 } else { par[4] },
        };
        config.validate()?;
        Ok(config)
    }
    fn validate(&self) -> Result<()> {
        ensure!(
            self.cycles > 1 && self.codes > 0 && self.samples > 0,
            "empty clutter layout"
        );
        ensure!(
            self.filtered_samples <= self.samples,
            "clutter extent exceeds input"
        );
        ensure!(
            self.blocks <= self.filtered_samples,
            "more blocks than filtered samples"
        );
        if self.notch_request >= 0 {
            ensure!(
                2 * self.notch_request as usize + 1 < self.cycles,
                "notch removes all bins"
            );
        } else {
            // The legacy loop reads the NEXT symmetric pair before exiting.
            // Reject configurations that can exhaust bins rather than emulate UB.
            ensure!(
                2 * self.notch_request.unsigned_abs() as usize + 1 < self.cycles,
                "adaptive request can exhaust bins"
            );
        }
        Ok(())
    }

    pub fn fixed_plan(&self, notches: Vec<i32>) -> Result<ClutterPlan> {
        self.validate()?;
        ensure!(
            notches.len() == self.blocks && notches.iter().all(|&v| v >= -1),
            "invalid block masks"
        );
        let mut filters = vec![DMatrix::identity(self.cycles, self.cycles); self.samples];
        for (b, &notch) in notches.iter().enumerate() {
            let f = clutter_operator(
                self.cycles,
                if notch < 0 {
                    None
                } else {
                    Some(notch as usize)
                },
            )?;
            let start = self.filtered_samples * b / self.blocks;
            let end = self.filtered_samples * (b + 1) / self.blocks;
            for item in &mut filters[start..end] {
                *item = f.clone();
            }
        }
        Ok(ClutterPlan {
            config: self.clone(),
            block_notches: notches,
            sample_filters: filters,
        })
    }

    /// Direct DFT implements the FFT energy calculation independently of FFTW.
    pub fn select(&self, voltage: &[C]) -> Result<ClutterPlan> {
        self.validate()?;
        ensure!(
            voltage.len() == self.cycles * self.codes * self.samples,
            "voltage layout mismatch"
        );
        ensure!(
            voltage.iter().all(|z| z.re.is_finite() && z.im.is_finite()),
            "nonfinite voltage"
        );
        let mut masks = Vec::new();
        for block in 0..self.blocks {
            if self.notch_request >= 0 {
                masks.push(self.notch_request);
                continue;
            }
            let mut energy = vec![0.; self.cycles];
            for sample in self.filtered_samples * block / self.blocks
                ..self.filtered_samples * (block + 1) / self.blocks
            {
                for code in 0..self.codes {
                    for (f, e) in energy.iter_mut().enumerate() {
                        let mut z = C::new(0., 0.);
                        for t in 0..self.cycles {
                            let phase =
                                -std::f64::consts::TAU * (f * t) as f64 / self.cycles as f64;
                            z += voltage[(t * self.codes + code) * self.samples + sample]
                                * C::from_polar(1., phase);
                        }
                        *e += z.norm_sqr();
                    }
                }
            }
            let mut notch = 0usize;
            let mut pair = 2. * energy[0];
            let mut remainder: f64 = energy[1..].iter().sum();
            let mut bins = self.cycles - 1;
            while notch < self.notch_request.unsigned_abs() as usize + 1
                && pair / 2. > remainder / bins as f64
            {
                notch += 1;
                pair = energy[notch] + energy[self.cycles - notch];
                remainder -= pair;
                bins -= 2;
            }
            masks.push(notch as i32 - 1);
        }
        self.fixed_plan(masks)
    }
}

impl ClutterPlan {
    pub fn marker(&self) -> f64 {
        if self.block_notches.is_empty() {
            0.
        } else {
            self.block_notches.iter().map(|&v| v as f64).sum::<f64>()
                / self.block_notches.len() as f64
        }
    }
    /// Float replay, optionally including online rint quantization. Overflow is
    /// rejected because conversion to int16 outside its range is not portable.
    pub fn apply(&self, input: &[C], quantize: bool) -> Result<Vec<C>> {
        let c = &self.config;
        ensure!(
            input.len() == c.cycles * c.codes * c.samples,
            "voltage layout mismatch"
        );
        let mut output = vec![C::new(0., 0.); input.len()];
        for sample in 0..c.samples {
            for code in 0..c.codes {
                for t in 0..c.cycles {
                    let mut value = C::new(0., 0.);
                    for u in 0..c.cycles {
                        value += input[(u * c.codes + code) * c.samples + sample]
                            * self.sample_filters[sample][(t, u)];
                    }
                    if quantize && sample < c.filtered_samples && c.blocks > 0 {
                        value = C::new(value.re.round_ties_even(), value.im.round_ties_even());
                        ensure!(
                            value.re >= -32768.
                                && value.re <= 32767.
                                && value.im >= -32768.
                                && value.im <= 32767.,
                            "int16 overflow"
                        );
                    }
                    output[(t * c.codes + code) * c.samples + sample] = value;
                }
            }
        }
        Ok(output)
    }
    /// Cross-cycle, cross-sample covariance for one code. Input cycles are
    /// independent; R_u(i,j) may vary freely with cycle and sample pair.
    /// Quantization and adaptive mask-selection uncertainty are excluded.
    pub fn covariance(
        &self,
        t: usize,
        i: usize,
        s: usize,
        j: usize,
        input: impl Fn(usize, usize, usize) -> C,
    ) -> Result<C> {
        ensure!(
            t < self.config.cycles
                && s < self.config.cycles
                && i < self.config.samples
                && j < self.config.samples,
            "covariance index outside layout"
        );
        Ok((0..self.config.cycles)
            .map(|u| {
                input(u, i, j) * self.sample_filters[i][(t, u)] * self.sample_filters[j][(s, u)]
            })
            .sum())
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    fn config() -> ClutterConfig {
        ClutterConfig {
            cycles: 14,
            codes: 1,
            samples: 3,
            filtered_samples: 2,
            blocks: 2,
            notch_request: -5,
        }
    }
    #[test]
    fn boundary_pair_is_not_uniform_power_mixing() {
        let p = config().fixed_plan(vec![0, 2]).unwrap();
        let unit = |_, _, _| C::new(1., 0.);
        let same = p.covariance(4, 0, 4, 0, unit).unwrap().re;
        let edge = p.covariance(4, 1, 4, 2, unit).unwrap().re;
        assert!((same - 1.).abs() < 1e-12);
        assert!((edge - (9f64 / 14.).sqrt()).abs() < 1e-12);
        assert!((edge - same).abs() > 0.19);
        assert_eq!(p.marker(), 1.);
    }
    #[test]
    fn recovered_beata_parameters_disable_clutter() {
        let c = ClutterConfig::from_parameters(&[64, 896, 432, 0, -1, 1, 1, 1], -5).unwrap();
        assert_eq!(c.blocks, 0);
        let p = c.fixed_plan(vec![]).unwrap();
        assert_eq!(
            p.covariance(8, 0, 8, 1, |_, _, _| C::new(3., 2.)).unwrap(),
            C::new(3., 2.)
        );
        assert_eq!(p.marker(), 0.);
    }
    #[test]
    fn adaptive_blocks_and_unfiltered_tail() {
        let c = config();
        let x = (0..14)
            .flat_map(|t| {
                let ac = C::from_polar(100., std::f64::consts::TAU * 3. * t as f64 / 14.);
                [C::new(1000., 0.) + ac, ac, C::new(t as f64, 7.)]
            })
            .collect::<Vec<_>>();
        let p = c.select(&x).unwrap();
        assert_eq!(p.block_notches, vec![0, -1]);
        let y = p.apply(&x, false).unwrap();
        assert!((0..14).map(|t| y[3 * t]).sum::<C>().norm() < 1e-10);
        for t in 0..14 {
            assert_eq!(y[3 * t + 2], x[3 * t + 2]);
        }
    }
    #[test]
    fn nonstationary_cross_boundary_mean_matches_voltage_replay() {
        let p = config().fixed_plan(vec![0, 2]).unwrap();
        // Deterministic basis summation is an exact second-moment check.
        let mut actual = C::new(0., 0.);
        for u in 0..14 {
            let mut x = vec![C::new(0., 0.); 42];
            x[3 * u + 1] = C::new((1. + u as f64).sqrt(), 0.);
            x[3 * u + 2] = x[3 * u + 1];
            let y = p.apply(&x, false).unwrap();
            actual += y[3 * 4 + 1] * y[3 * 7 + 2].conj();
        }
        let expected = p
            .covariance(4, 1, 7, 2, |u, _, _| C::new(1. + u as f64, 0.))
            .unwrap();
        assert!((actual - expected).norm() < 1e-12);
    }
}
