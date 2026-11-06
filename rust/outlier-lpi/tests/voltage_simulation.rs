//! Acceptance components, not a declaration that an Arecibo decoder is validated.
use outlier_lpi::{C,solve};
use std::f64::consts::TAU;
#[test]
fn range_grid_300m_recovers_coded_profile_with_missing_equations(){
 let sample_interval_s=1e-6;let bin=2;let grid=299_792_458.*sample_interval_s*bin as f64/2.;
 assert!((grid-299.792458).abs()<1e-8);
 let n=128;let code:Vec<_>=(0..440).map(|i|if ((i/20)*17+(i/20)*(i/20)*3)%11<5{1.}else{-1.}).collect();
 for lag in [0,10,80,160]{
  let truth:Vec<_>=(0..n).map(|r|{let height=r as f64*grid;let p=1.+2.*(-((height-18000.)/4500.).powi(2)).exp();C::from_polar(p*(-(lag as f64)/140.).exp(),-0.017*lag as f64)}).collect();
  let mut rows=vec![];let mut measured=vec![];
  for receive_bin in 0..n+code.len()/bin{
   // Missing equations stand for a supplied RFI/saturation mask. This test does
   // not assess a classifier's ability to discover contamination from raw IQ.
   if receive_bin%7==0{continue;}
   let row:Vec<_>=(0..n).filter_map(|r|{
    let j=(receive_bin as isize-r as isize)*bin as isize;let mut a=0.;
    for offset in 0..bin{let k=j+offset as isize;if k>=0&&(k as usize)+lag<code.len(){a+=code[k as usize]*code[k as usize+lag]/bin as f64;}}
    (a!=0.).then_some((r,C::new(a,0.)))
   }).collect();
   if !row.is_empty(){measured.push(row.iter().map(|(j,a)|truth[*j]*a).sum());rows.push(row);}
  }
  let (x,variance,condition)=solve(&rows,&measured,&vec![1.;rows.len()],n).unwrap();
  let error=x.iter().zip(truth).map(|(a,b)|(*a-b).norm()).fold(0.,f64::max);
  assert!(error<1e-8,"lag {lag}: {error}, conditioning proxy {condition}");
  assert!(variance.iter().all(|v|v.is_finite()&&*v>0.));
 }
}
#[test]
fn same_code_pair_difference_removes_coherent_clutter_without_factor_two_bias(){
 let lag=13.;let frequency=0.11;let ion_amplitude=2.7;let clutter_amplitude=40.;
 let mut raw=C::default();let mut differenced=C::default();
 for a in 0..3{for b in 0..3{
  let ion=|t:f64,p:usize|C::from_polar(ion_amplitude,frequency*t+TAU*p as f64/3.);
  let clutter=|t:f64|C::from_polar(clutter_amplitude,0.003*t+0.7);
  let first=ion(0.,a)+clutter(0.);let first_lag=ion(lag,a)+clutter(lag);
  let second=ion(0.,b)+clutter(0.);let second_lag=ion(lag,b)+clutter(lag);
  raw+=first*first_lag.conj()/9.;
  differenced+=(first-second)*(first_lag-second_lag).conj()/18.;
 }}
 let truth=C::from_polar(ion_amplitude*ion_amplitude,-frequency*lag);
 assert!((differenced-truth).norm()<1e-11);
 assert!((raw-truth).norm()>1000.);
}
#[test]
fn filtering_a_coded_voltage_is_not_a_separable_single_lag_response(){
 let code=[1.,-1.,1.,1.,-1.,-1.,1.,-1.];let omega=0.8;let h=[0.25,0.5,0.25];let lag=1;
 let voltage:Vec<_>=code.iter().enumerate().map(|(i,a)|C::from_polar(*a,omega*i as f64)).collect();
 let filter=|x:&[C],t:usize|h.iter().enumerate().map(|(k,a)|x[t+k-1]*a).sum::<C>();
 let complex_code:Vec<_>=code.iter().map(|a|C::new(*a,0.)).collect();
 let t=3;let actual=filter(&voltage,t)*filter(&voltage,t+lag).conj();
 let assumed=filter(&complex_code,t)*filter(&complex_code,t+lag).conj()*C::from_polar(1.,-omega*lag as f64);
 assert!((actual-assumed).norm()>0.02,"the counterexample must exercise mixed native lags");
}
