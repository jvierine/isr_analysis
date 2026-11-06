//! Unmagnetized Maxwellian ISR spectrum with number-conserving BGK collisions.
//! Spectral density is per Hz per electron; integrate both frequency signs.
use crate::Complex as C;
use anyhow::{ensure, Result};
use errorfunctions::ComplexErrorFunctions;
use serde::{Deserialize, Serialize};
use std::f64::consts::PI;
const KB: f64 = 1.380649e-23;
const ME: f64 = 9.1093837139e-31;
const AMU: f64 = 1.66053906892e-27;
const EPS0: f64 = 8.8541878188e-12;
const E: f64 = 1.602176634e-19;

#[derive(Clone, Debug, Serialize, Deserialize)]
pub struct IonSpecies {
    pub mass_amu: f64,
    /// Number fraction. This initial model supports singly charged ions.
    pub fraction: f64,
}
#[derive(Clone, Debug, Serialize, Deserialize)]
pub struct SpectrumModel {
    pub radar_frequency_hz: f64,
    pub ions: Vec<IonSpecies>,
    pub electron_collision_s: f64,
}
#[derive(Clone, Copy, Debug, Serialize, Deserialize)]
pub struct PlasmaState {
    pub electron_density_m3: f64,
    pub ion_temperature_k: f64,
    pub electron_temperature_k: f64,
    /// Positive shifts the model spectrum toward positive baseband frequency.
    /// Physical toward/away sign must be established from the receiver convention.
    pub velocity_m_s: f64,
    /// Effective number-conserving BGK rate in s^-1, not cycles per second.
    pub ion_collision_s: f64,
}
impl SpectrumModel {
    pub fn validate(&self) -> Result<()> {
        ensure!(
            self.radar_frequency_hz.is_finite() && self.radar_frequency_hz > 0.,
            "invalid radar frequency"
        );
        ensure!(
            !self.ions.is_empty()
                && self.ions.iter().all(|i| i.mass_amu.is_finite()
                    && i.mass_amu > 0.
                    && i.fraction.is_finite()
                    && i.fraction >= 0.),
            "invalid composition"
        );
        ensure!(
            (self.ions.iter().map(|i| i.fraction).sum::<f64>() - 1.).abs() < 1e-10,
            "ion fractions must sum to one"
        );
        ensure!(
            self.electron_collision_s.is_finite() && self.electron_collision_s >= 0.,
            "invalid electron collision rate"
        );
        Ok(())
    }
    pub fn wavenumber(&self) -> f64 {
        4. * PI * self.radar_frequency_hz / 299792458.
    }
    pub fn evaluate(&self, state: PlasmaState, frequency_hz: &[f64]) -> Result<Vec<f64>> {
        self.evaluate_components(state, frequency_hz, true)
    }
    /// Ion-only component of the incoherent spectrum. Kept separate from the
    /// physical total spectrum for comparisons with legacy DC-subtracted fits.
    pub fn evaluate_ion_only(&self, state: PlasmaState, frequency_hz: &[f64]) -> Result<Vec<f64>> {
        self.evaluate_components(state, frequency_hz, false)
    }
    fn evaluate_components(&self, state: PlasmaState, frequency_hz: &[f64], include_electron_continuum: bool) -> Result<Vec<f64>> {
        self.validate()?;
        ensure!(
            state.electron_density_m3.is_finite()
                && state.electron_density_m3 > 0.
                && state.ion_temperature_k.is_finite()
                && state.ion_temperature_k > 0.
                && state.electron_temperature_k.is_finite()
                && state.electron_temperature_k > 0.
                && state.velocity_m_s.is_finite()
                && state.ion_collision_s.is_finite()
                && state.ion_collision_s >= 0.,
            "invalid plasma state"
        );
        ensure!(
            frequency_hz.iter().all(|f| f.is_finite()),
            "invalid frequency grid"
        );
        frequency_hz.iter().map(|&f|self.spectrum_point(state,f,include_electron_continuum).map(|v|v.0)).collect()
    }
    /// Integrate spectral power onto linear frequency-node basis functions.
    /// Use only with kernels containing trapezoidal frequency weights. This
    /// resolves ion-acoustic resonances between nodes instead of sampling them.
    pub fn evaluate_integrated(&self,state:PlasmaState,f:&[f64])->Result<Vec<f64>> {
        self.evaluate(state,&[])?;
        ensure!(f.len()>1&&f.windows(2).all(|w|w[0].is_finite()&&w[1].is_finite()&&w[1]>w[0]),"invalid integration grid");
        let points=f.iter().map(|&x|self.spectrum_point(state,x,true)).collect::<Result<Vec<_>>>()?;
        let mut mass=vec![0.;f.len()];
        for i in 0..f.len()-1 {
            let a=f[i];let b=f[i+1];let width=b-a;let (_,ea)=points[i];let (_,eb)=points[i+1];
            let resonance=ea*eb<0.;
            // Apply the same quadrature on every cell. Switching between a
            // linear spectrum and quadrature at a parameter-dependent slope
            // threshold creates discontinuities in finite-difference gradients.
            let mut cuts=vec![a,b];
            if resonance {
                let(mut lo,mut hi,mut e)=(a,b,ea);
                for _ in 0..48 {let mid=(lo+hi)/2.;let em=self.spectrum_point(state,mid,true)?.1;if e*em<=0.{hi=mid;}else{lo=mid;e=em;}}
                let root=(lo+hi)/2.;if root>a&&root<b{cuts.push(root);}
            }
            cuts.sort_by(f64::total_cmp);
            let value=|x:f64|->Result<[f64;2]>{let s=self.spectrum_point(state,x,true)?.0;let r=(x-a)/width;Ok([s*(1.-r),s*r])};
            for pair in cuts.windows(2){let x=pair[0];let y=pair[1];let m=(x+y)/2.;let result=integrate_pair(&value,x,y,value(x)?,value(m)?,value(y)?,28)?;mass[i]+=result[0];mass[i+1]+=result[1];}
        }
        for i in 0..f.len(){let w=if i==0{(f[1]-f[0])/2.}else if i+1==f.len(){(f[i]-f[i-1])/2.}else{(f[i+1]-f[i-1])/2.};mass[i]/=w;}
        Ok(mass)
    }
    fn spectrum_point(&self,state:PlasmaState,f:f64,include_electron_continuum:bool)->Result<(f64,f64)> {
        let k=self.wavenumber();
        let alpha=state.electron_density_m3*E*E/(EPS0*KB*k*k);
        let omega = 2. * PI * f - k * state.velocity_m_s;
        let (ae, fe) = response(
            omega,
            k,
            ME,
            state.electron_temperature_k,
            self.electron_collision_s,
        );
        let ce = ae * (alpha / state.electron_temperature_k);
        let mut ci = C::new(0., 0.);
        let mut fi = 0.;
        for ion in &self.ions {
            let (a, free) = response(
                omega,
                k,
                ion.mass_amu * AMU,
                state.ion_temperature_k,
                state.ion_collision_s,
            );
            ci += a * (alpha * ion.fraction / state.ion_temperature_k);
            fi += ion.fraction * free;
        }
        let eps = C::new(1., 0.) + ce + ci;
        let electron = if include_electron_continuum { (C::new(1., 0.) + ci).norm_sqr() * fe } else { 0. };
        let s = (electron + ce.norm_sqr() * fi) / eps.norm_sqr();
        ensure!(s.is_finite() && s >= 0., "invalid model spectrum");
        Ok((s,eps.re))
    }
}
fn integrate_pair<F:Fn(f64)->Result<[f64;2]>>(fun:&F,a:f64,b:f64,fa:[f64;2],fm:[f64;2],fb:[f64;2],depth:u32)->Result<[f64;2]>{
 let m=(a+b)/2.;let fl=fun((a+m)/2.)?;let fr=fun((m+b)/2.)?;
 let coarse:[f64;2]=std::array::from_fn(|k|(b-a)*(fa[k]+4.*fm[k]+fb[k])/6.);
 let fine:[f64;2]=std::array::from_fn(|k|(b-a)*(fa[k]+4.*fl[k]+2.*fm[k]+4.*fr[k]+fb[k])/12.);
 let ok=(0..2).all(|k|(fine[k]-coarse[k]).abs()<=15.*(1e-19+1e-10*fine[k].abs()));
 if ok{return Ok(std::array::from_fn(|k|fine[k]+(fine[k]-coarse[k])/15.));}
 ensure!(depth>0,"resonance quadrature did not converge");
 let l=integrate_pair(fun,a,m,fa,fl,fm,depth-1)?;let r=integrate_pair(fun,m,b,fm,fr,fb,depth-1)?;Ok([l[0]+r[0],l[1]+r[1]])
}
fn response(omega: f64, k: f64, mass: f64, temperature: f64, collision: f64) -> (C, f64) {
    let a = k * (2. * KB * temperature / mass).sqrt();
    let g0 = C::new(omega / a, collision / a).w() * (PI.sqrt() / a);
    let g = g0 / (C::new(1., 0.) - collision * g0);
    (C::new(1., 0.) + C::new(0., omega) * g, 2. * g.re)
}

#[cfg(test)]
mod tests {
    use super::*;
    #[test]
    fn maxwellian_response_matches_independent_time_quadrature() {
        // Independent Simpson integration of the free-particle correlation.
        for nu in [0., 300., 5000.] {
            let k = 39.;
            let t = 900.;
            let mass = 30.5 * AMU;
            let a = k * (2. * KB * t / mass).sqrt();
            for omega in [0., 2000., -8000., 16000.] {
                let n = 10000;
                let dt = 14. / a / n as f64;
                let mut g = C::new(0., 0.);
                for i in 0..=n {
                    let x = i as f64 * dt;
                    let weight = if i == 0 || i == n {
                        1.
                    } else if i % 2 == 0 {
                        2.
                    } else {
                        4.
                    };
                    g += C::from_polar(weight * (-a * a * x * x / 4. - nu * x).exp(), omega * x);
                }
                g *= dt / 3.;
                g /= C::new(1., 0.) - nu * g;
                let (chi, s) = response(omega, k, mass, t, nu);
                assert!((s - 2. * g.re).abs() < 1e-12);
                assert!((chi - (C::new(1., 0.) + C::new(0., omega) * g)).norm() < 1e-8);
            }
        }
    }
    #[test]
    fn spectrum_symmetry_drift_and_composition() {
        let m = SpectrumModel {
            radar_frequency_hz: 930e6,
            ions: vec![
                IonSpecies {
                    mass_amu: 16.,
                    fraction: 0.7,
                },
                IonSpecies {
                    mass_amu: 30.5,
                    fraction: 0.3,
                },
            ],
            electron_collision_s: 0.,
        };
        let p = PlasmaState {
            electron_density_m3: 3e11,
            ion_temperature_k: 900.,
            electron_temperature_k: 1800.,
            velocity_m_s: 0.,
            ion_collision_s: 100.,
        };
        let f = vec![-2000., 0., 2000.];
        let a = m.evaluate(p, &f).unwrap();
        assert!((a[0] - a[2]).abs() < 1e-15);
        let mut moving = p;
        moving.velocity_m_s = 123.;
        let shift = m.wavenumber() * 123. / (2. * PI);
        let b = m
            .evaluate(moving, &f.iter().map(|x| x + shift).collect::<Vec<_>>())
            .unwrap();
        for (a, b) in a.iter().zip(b) {
            assert!((a - b).abs() < 1e-15);
        }
        let mut reordered = m.clone();
        reordered.ions.reverse();
        assert_eq!(
            m.evaluate(p, &f).unwrap(),
            reordered.evaluate(p, &f).unwrap()
        );
    }
}

#[cfg(test)]
mod normalization_tests {
    use super::*;
    #[test]
    fn integrated_spectrum_temperature_derivative_is_stable(){
        let model=SpectrumModel{radar_frequency_hz:440.2e6,ions:vec![IonSpecies{mass_amu:16.,fraction:1.}],electron_collision_s:0.};
        let frequencies:Vec<_>=(-2000..=2000).map(|i|i as f64*10.).collect();
        for (ti,te) in [(1729.6,1994.8),(80.,10000.)] {
            let state=PlasmaState{electron_density_m3:1.965e11,ion_temperature_k:ti,electron_temperature_k:te,velocity_m_s:50.99,ion_collision_s:0.00415};
            let derivative=|step:f64| {
                let mut a=state;let mut b=state;a.electron_temperature_k*=step.exp();b.electron_temperature_k*=(-step).exp();
                model.evaluate_integrated(a,&frequencies).unwrap().iter().zip(model.evaluate_integrated(b,&frequencies).unwrap()).map(|(x,y)|(x-y)/(2.*step)).collect::<Vec<_>>()
            };
            let a=derivative(1e-4);let b=derivative(5e-5);
            // Compare the observable ACF derivative. Individual hat weights
            // may exchange mass rapidly as a narrow peak crosses a grid node.
            let acf=|s:&[f64],lag:f64|->C {frequencies.iter().zip(s).enumerate().map(|(i,(f,v))|C::from_polar(*v*10.*if i==0||i+1==frequencies.len(){0.5}else{1.},-std::f64::consts::TAU*f*lag)).sum()};
            let scale=acf(&b,0.).norm().max(1e-30);
            for lag in [0.,10e-6,100e-6,450e-6] {
                let error=(acf(&a,lag)-acf(&b,lag)).norm()/scale;
                assert!(error<5e-5,"ACF derivative changes by {error} at lag={lag}, Te={te}");
            }
        }
    }
    #[test]
    fn integrated_narrow_resonance_matches_dense_independent_quadrature(){
        let model=SpectrumModel{radar_frequency_hz:440.2e6,ions:vec![IonSpecies{mass_amu:16.,fraction:1.}],electron_collision_s:0.};
        let state=PlasmaState{electron_density_m3:1e11,ion_temperature_k:80.,electron_temperature_k:10000.,velocity_m_s:123.,ion_collision_s:1.};
        let coarse:Vec<_>=(-1500..=1500).map(|i|i as f64*10.).collect();
        let fine:Vec<_>=(-150000..=150000).map(|i|i as f64*0.1).collect();
        let a=model.evaluate_integrated(state,&coarse).unwrap();let b=model.evaluate(state,&fine).unwrap();
        for lag in [0.,10e-6,450e-6]{
            let acf=|f:&[f64],s:&[f64],step:f64|->C{f.iter().zip(s).enumerate().map(|(i,(hz,s))|C::from_polar(s*step*if i==0||i+1==f.len(){0.5}else{1.},-std::f64::consts::TAU*hz*lag)).sum()};
            let x=acf(&coarse,&a,10.);let y=acf(&fine,&b,0.1);let scale=b.iter().sum::<f64>()*0.1;
            assert!((x-y).norm()/scale<5e-4,"integrated ACF error {}",(x-y).norm()/scale);
        }
    }
    #[test]
    fn ion_only_is_nonnegative_and_below_total_spectrum() {
        let m = SpectrumModel { radar_frequency_hz: 440.2e6,
            ions: vec![IonSpecies { mass_amu: 16., fraction: 1. }], electron_collision_s: 0. };
        let p = PlasmaState { electron_density_m3: 1e11, ion_temperature_k: 1000.,
            electron_temperature_k: 2000., velocity_m_s: 0., ion_collision_s: 0. };
        let f: Vec<_> = (-400..=400).map(|i| i as f64 * 250.).collect();
        let total = m.evaluate(p, &f).unwrap();
        let ion = m.evaluate_ion_only(p, &f).unwrap();
        assert!(ion.iter().zip(total).all(|(i, t)| *i >= 0. && *i <= t));
    }
    #[test]
    fn ion_line_area_matches_equilibrium_debye_limit() {
        let m = SpectrumModel {
            radar_frequency_hz: 930e6,
            ions: vec![IonSpecies {
                mass_amu: 30.5,
                fraction: 1.,
            }],
            electron_collision_s: 0.,
        };
        let p = PlasmaState {
            electron_density_m3: 1e12,
            ion_temperature_k: 1000.,
            electron_temperature_k: 1000.,
            velocity_m_s: 0.,
            ion_collision_s: 0.,
        };
        let f: Vec<_> = (-8000..=8000).map(|i| i as f64 * 10.).collect();
        let s = m.evaluate(p, &f).unwrap();
        let area = s
            .iter()
            .enumerate()
            .map(|(i, value)| value * if i == 0 || i == s.len() - 1 { 5. } else { 10. })
            .sum::<f64>();
        let a = p.electron_density_m3 * E * E
            / (EPS0 * KB * p.electron_temperature_k * m.wavenumber().powi(2));
        let expected = a * a / ((1. + a) * (1. + 2. * a));
        // Finite electron mass and the included electron continuum make the
        // finite ion-band integral differ slightly from the zero-mass limit.
        assert!(
            (area / expected - 1.).abs() < 0.015,
            "area {area}, equilibrium {expected}"
        );
    }
}
