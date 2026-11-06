//! An explicit transport-scale guide for a profile prior, not a covariance law.
use anyhow::{ensure,Result};
use serde::{Serialize,Deserialize};
#[derive(Clone,Serialize,Deserialize)]
#[serde(deny_unknown_fields)]
pub struct MeanFreePathPrior {
 pub multiplier:[f64;4],
 pub minimum_km:f64,
 pub maximum_km:f64,
 /// Frozen reference state; never optimized to weaken its own prior.
 pub reference_density_m3:f64,
 pub reference_ion_temperature_k:f64,
 pub coulomb_log:f64,
}
#[derive(Serialize)]
pub struct Scale {
 pub thermal_speed_m_s:f64,
 pub ion_neutral_collision_s:f64,
 pub effective_ion_ion_collision_s:f64,
 pub mean_free_path_km:f64,
 pub correlation_length_km:[f64;4],
}
impl MeanFreePathPrior {
 /// NRL formulary ion collision rate, single effective singly charged species.
 /// n is converted m^-3 -> cm^-3; Ti K -> eV. Mixture transport is approximate.
 pub fn evaluate(&self,mass_amu:f64,neutral_collision_s:f64)->Result<Scale>{
  ensure!(mass_amu.is_finite()&&mass_amu>0.&&neutral_collision_s.is_finite()&&neutral_collision_s>=0.,"invalid collision inputs");
  ensure!([self.minimum_km,self.maximum_km,self.reference_density_m3,self.reference_ion_temperature_k,self.coulomb_log].iter().all(|v|v.is_finite()&&*v>0.)&&self.maximum_km>=self.minimum_km&&self.multiplier.iter().all(|v|v.is_finite()&&*v>0.),"invalid mean-free-path prior");
  let v=(1.380649e-23*self.reference_ion_temperature_k/(mass_amu*1.66053906660e-27)).sqrt();
  let ti_ev=self.reference_ion_temperature_k*1.380649e-23/1.602176634e-19;
  let mass_proton=mass_amu*1.66053906660e-27/1.67262192369e-27;
  let nu=4.80e-8/mass_proton.sqrt()*(self.reference_density_m3*1e-6)*self.coulomb_log/ti_ev.powf(1.5);
  let lambda=v/(neutral_collision_s+nu)/1000.;
  Ok(Scale{thermal_speed_m_s:v,ion_neutral_collision_s:neutral_collision_s,effective_ion_ion_collision_s:nu,mean_free_path_km:lambda,correlation_length_km:self.multiplier.map(|a|(a*lambda).clamp(self.minimum_km,self.maximum_km))})
 }
}
#[cfg(test)] mod tests{
 use super::*;
 fn config()->MeanFreePathPrior{MeanFreePathPrior{multiplier:[1.;4],minimum_km:0.001,maximum_km:1000.,reference_density_m3:1e11,reference_ion_temperature_k:1000.,coulomb_log:15.}}
 #[test]fn density_and_temperature_transport_scaling(){let a=config();let x=a.evaluate(16.,0.).unwrap();let mut b=config();b.reference_density_m3*=2.;let y=b.evaluate(16.,0.).unwrap();assert!((y.mean_free_path_km/x.mean_free_path_km-0.5).abs()<1e-12);b=config();b.reference_ion_temperature_k*=2.;assert!((b.evaluate(16.,0.).unwrap().mean_free_path_km/x.mean_free_path_km-4.).abs()<1e-12);assert!(a.evaluate(16.,100.).unwrap().mean_free_path_km<x.mean_free_path_km);}
 #[test]fn explicit_bounds_and_invalid_inputs(){let mut a=config();a.minimum_km=10.;a.maximum_km=100.;assert_eq!(a.evaluate(16.,1e8).unwrap().correlation_length_km,[10.;4]);assert!(a.evaluate(16.,f64::NAN).is_err());a.reference_density_m3=0.;assert!(a.evaluate(16.,0.).is_err());}
}
